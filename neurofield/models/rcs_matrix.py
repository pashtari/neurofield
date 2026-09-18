"""Row-contiguous sparse matrices with differentiable dense multiplication.

Each row stores a contiguous segment of ``L`` values and its starting column.
:func:`rcs_product` computes ``(A_1 @ B_1) * ... * (A_D @ B_D)``, the
elementwise product of several sparse-by-dense products (FUTON's CP combiner),
and ``A @ B`` is its one-matrix case.

On CUDA, Triton kernels compute the product without materializing its ``D``
factors: each output tile gathers, per matrix, the ``L`` rows of ``B_i`` that
its rows touch. The backward pass recomputes the factors instead of storing
them. It sorts the rows of all matrices by start column once, then reduces the
gradient of each row of ``B_i`` over the rows starting near it, in a fixed
order and without atomics, so first-order gradients are deterministic.
``create_graph=True`` switches to PyTorch operations for double backward.
Accumulation uses float32, or float64 for double inputs, and CUDA autocast
follows matmul's dtype rules. Kernels specialize on the segment length and on
the number of matrices. Elsewhere, products use CSR during inference and a
gather-and-contract path when gradients are needed.
"""

import copy
from collections.abc import Callable, Sequence
from functools import reduce
from itertools import accumulate
from operator import mul
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None

__all__ = ["RCSMatrix", "rcs_product"]

_MATMUL_FUNCS = {
    torch.matmul,
    torch.mm,
    Tensor.matmul,
    Tensor.mm,
    Tensor.__matmul__,
}

# Rows that one program of the backward reduces, and the most programs per column.
_ROWS_PER_PROGRAM = 512
_MAX_CHUNKS = 16


def _stack_rows(matrices: Sequence[Tensor]) -> Tensor:
    """Stack matrices by rows, padded to a multiple of 16 values for Triton.

    Triton vectorizes row loads only when it can prove their alignment.
    """
    stacked = torch.cat(matrices) if len(matrices) > 1 else matrices[0]
    padding = -stacked.shape[1] % 16
    return (F.pad(stacked, (0, padding)) if padding else stacked).contiguous()


def _product_torch(values: Tensor, cols: Tensor, factors: Tensor) -> Tensor:
    """Differentiable PyTorch form of the stacked product; see :func:`rcs_product`."""
    windows = cols.unsqueeze(-1) + torch.arange(values.shape[-1], device=cols.device)
    return (values.unsqueeze(-2) @ factors[windows]).squeeze(-2).prod(0)


# Kernel notation: V = values (D, M, L) and C = start columns (D, M), both
# contiguous; each matrix's columns are offset to its own rows of F, the stacked
# dense operands (N, R) with rows padded to ``stride_f`` values; G = the output
# gradient (M, R). Row m of factor d is sum_l V[d, m, l] F[C[d, m] + l].
if triton is not None:

    @triton.jit
    def _factor(
        v_ptr, c_ptr, f_ptr, d, rows, cols, mask_rows, mask, M, stride_f,
        L: tl.constexpr, ACC: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_R: tl.constexpr,
    ):  # fmt: skip
        """A (BLOCK_M, BLOCK_R) tile of factor ``d``."""
        start = tl.load(c_ptr + d * M + rows, mask=mask_rows, other=0)
        out = tl.zeros((BLOCK_M, BLOCK_R), dtype=ACC)
        for l in tl.static_range(L):
            value = tl.load(v_ptr + (d * M + rows) * L + l, mask=mask_rows, other=0.0)
            row = tl.load(
                f_ptr + (start + l)[:, None] * stride_f + cols[None, :],
                mask=mask,
                other=0.0,
            )
            out += value.to(ACC)[:, None] * row.to(ACC)
        return out

    @triton.jit
    def _cofactor(
        g_ptr, v_ptr, c_ptr, f_ptr, a, rows, cols, mask_rows, mask, M, R, stride_f,
        D: tl.constexpr, L: tl.constexpr, ACC: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_R: tl.constexpr,
    ):  # fmt: skip
        """G times every factor but ``a``: the gradient of factor ``a``'s tile."""
        out = tl.load(g_ptr + rows[:, None] * R + cols[None, :], mask=mask, other=0.0)
        out = out.to(ACC)
        for d in tl.static_range(D):
            if d != a:
                out *= _factor(
                    v_ptr, c_ptr, f_ptr, d, rows, cols, mask_rows, mask, M, stride_f,
                    L, ACC, BLOCK_M, BLOCK_R,
                )  # fmt: skip
        return out

    # Small row tiles keep the gathered rows in registers; the best column
    # width depends on L, R and the GPU. Tuning cannot change the results:
    # every output is summed in the same order, by one thread.
    _PRODUCT_CONFIGS = [
        triton.Config({"BLOCK_M": 16, "BLOCK_R": 32}, num_warps=4),
        triton.Config({"BLOCK_M": 16, "BLOCK_R": 32}, num_warps=8),
        triton.Config({"BLOCK_M": 16, "BLOCK_R": 64}, num_warps=8),
        triton.Config({"BLOCK_M": 16, "BLOCK_R": 128}, num_warps=8),
        triton.Config({"BLOCK_M": 16, "BLOCK_R": 256}, num_warps=8),
        triton.Config({"BLOCK_M": 32, "BLOCK_R": 64}, num_warps=4),
    ]

    @triton.autotune(configs=_PRODUCT_CONFIGS, key=["D", "L", "R"])
    @triton.jit
    def _product_kernel(
        v_ptr, c_ptr, f_ptr, out_ptr, M, R, stride_f,
        D: tl.constexpr, L: tl.constexpr, ACC: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_R: tl.constexpr,
    ):  # fmt: skip
        """out = prod_d (A_d @ F) over a (BLOCK_M, BLOCK_R) tile."""
        rows = (tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
        cols = tl.program_id(1) * BLOCK_R + tl.arange(0, BLOCK_R)
        mask_rows = rows < M
        mask = mask_rows[:, None] & (cols < R)[None, :]
        out = tl.full((BLOCK_M, BLOCK_R), 1.0, dtype=ACC)
        for d in tl.static_range(D):
            out *= _factor(
                v_ptr, c_ptr, f_ptr, d, rows, cols, mask_rows, mask, M, stride_f,
                L, ACC, BLOCK_M, BLOCK_R,
            )  # fmt: skip
        tl.store(
            out_ptr + rows[:, None] * R + cols[None, :],
            out.to(out_ptr.dtype.element_ty),
            mask=mask,
        )

    @triton.jit
    def _segment_kernel(
        order_ptr, bounds_ptr, v_ptr, c_ptr, f_ptr, g_ptr, part_ptr,
        M, R, stride_f, chunks, D: tl.constexpr, L: tl.constexpr,
        L_P2: tl.constexpr, ACC: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_R: tl.constexpr,
    ):  # fmt: skip
        """Reduce one chunk of the rows that start at column ``j`` of F.

        Rows of matrix ``a`` starting at ``j`` contribute ``V[a, m, l] * h[m]``
        to row ``j + l`` of grad F, where ``h = G * prod_{d != a} (A_d @ F)``;
        ``part[j, chunk, l]`` holds their sum over the chunk.
        """
        j = tl.program_id(0)
        chunk = tl.program_id(1)
        cols = tl.program_id(2) * BLOCK_R + tl.arange(0, BLOCK_R)
        mask_cols = cols < R
        lo = tl.load(bounds_ptr + j)
        hi = tl.load(bounds_ptr + j + 1)
        size = tl.cdiv(hi - lo, chunks)
        start = lo + chunk * size
        end = tl.minimum(start + size, hi)
        # Matrices address disjoint rows of F, so one matrix owns column j.
        a = tl.load(order_ptr + lo, mask=lo < hi, other=0) // M
        taps = tl.arange(0, L_P2)
        acc = tl.zeros((L_P2, BLOCK_R), dtype=ACC)
        for p in range(start, end, BLOCK_M):
            ids = p + tl.arange(0, BLOCK_M)
            mask_rows = ids < end
            rows = tl.load(order_ptr + ids, mask=mask_rows, other=0) % M
            mask = mask_rows[:, None] & mask_cols[None, :]
            h = _cofactor(
                g_ptr, v_ptr, c_ptr, f_ptr, a, rows, cols, mask_rows, mask, M, R,
                stride_f, D, L, ACC, BLOCK_M, BLOCK_R,
            )  # fmt: skip
            for l in tl.static_range(L):
                value = tl.load(
                    v_ptr + (a * M + rows) * L + l, mask=mask_rows, other=0.0
                )
                total = tl.sum(value.to(ACC)[:, None] * h, axis=0)
                acc = tl.where((taps == l)[:, None], acc + total[None, :], acc)
        tl.store(
            part_ptr
            + ((j * chunks + chunk) * L + taps).to(tl.int64)[:, None] * stride_f
            + cols[None, :],
            acc,
            mask=(taps < L)[:, None] & mask_cols[None, :],
        )

    @triton.jit
    def _tap_kernel(
        part_ptr, out_ptr, R, stride_f, chunks,
        L: tl.constexpr, ACC: tl.constexpr, BLOCK_R: tl.constexpr,
    ):  # fmt: skip
        """grad F[n] = sum over taps l and chunks of part[n - l, chunk, l]."""
        n = tl.program_id(0)
        cols = tl.program_id(1) * BLOCK_R + tl.arange(0, BLOCK_R)
        mask = cols < R
        acc = tl.zeros((BLOCK_R,), dtype=ACC)
        for l in tl.static_range(L):
            if n >= l:
                for chunk in range(chunks):
                    index = ((n - l) * chunks + chunk) * L + l
                    acc += tl.load(
                        part_ptr + index.to(tl.int64) * stride_f + cols,
                        mask=mask,
                        other=0.0,
                    )
        tl.store(
            out_ptr + n.to(tl.int64) * R + cols,
            acc.to(out_ptr.dtype.element_ty),
            mask=mask,
        )

    @triton.jit
    def _values_kernel(
        v_ptr, c_ptr, f_ptr, g_ptr, gv_ptr, M, R, stride_f,
        D: tl.constexpr, L: tl.constexpr, L_P2: tl.constexpr, ACC: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_R: tl.constexpr,
    ):  # fmt: skip
        """grad V[a, m, l] = dot(h[m], F[C[a, m] + l]), h = G * prod_{d != a} (A_d @ F).

        Writes a contiguous (D, M, L) gradient; grid = (M-blocks, D).
        """
        rows = (tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
        a = tl.program_id(1).to(tl.int64)
        mask_rows = rows < M
        start = tl.load(c_ptr + a * M + rows, mask=mask_rows, other=0)
        taps = tl.arange(0, L_P2)
        acc = tl.zeros((BLOCK_M, L_P2), dtype=ACC)
        for r in range(0, R, BLOCK_R):
            cols = r + tl.arange(0, BLOCK_R)
            mask = mask_rows[:, None] & (cols < R)[None, :]
            h = _cofactor(
                g_ptr, v_ptr, c_ptr, f_ptr, a, rows, cols, mask_rows, mask, M, R,
                stride_f, D, L, ACC, BLOCK_M, BLOCK_R,
            )  # fmt: skip
            for l in tl.static_range(L):
                row = tl.load(
                    f_ptr + (start + l)[:, None] * stride_f + cols[None, :],
                    mask=mask,
                    other=0.0,
                )
                total = tl.sum(h * row.to(ACC), axis=1)
                acc = tl.where((taps == l)[None, :], acc + total[:, None], acc)
        tl.store(
            gv_ptr + (a * M + rows)[:, None] * L + taps[None, :],
            acc.to(gv_ptr.dtype.element_ty),
            mask=mask_rows[:, None] & (taps < L)[None, :],
        )

    def _accumulator(dtype: torch.dtype) -> tl.dtype:
        return tl.float64 if dtype == torch.float64 else tl.float32

    def _launch_product(
        values: Tensor, cols: Tensor, factors: Tensor, width: int
    ) -> Tensor:
        """Multiply on CUDA; ``factors`` holds the stacked rows, padded beyond ``width``."""
        num_factors, num_rows, segment_length = values.shape
        out = torch.empty(num_rows, width, dtype=values.dtype, device=values.device)
        grid = lambda meta: (
            triton.cdiv(num_rows, meta["BLOCK_M"]),
            triton.cdiv(width, meta["BLOCK_R"]),
        )
        # Triton launches kernels on the current CUDA device.
        with torch.cuda.device(values.device):
            _product_kernel[grid](
                values, cols, factors, out, num_rows, width, factors.stride(0),
                D=num_factors, L=segment_length, ACC=_accumulator(values.dtype),
            )  # fmt: skip
        return out

    def _launch_grad_factors(
        values: Tensor, cols: Tensor, factors: Tensor, grad: Tensor
    ) -> Tensor:
        """Reduce the ``(N, R)`` gradient of the stacked dense operands."""
        num_factors, num_rows, segment_length = values.shape
        num_cols, width = factors.shape[0], grad.shape[1]
        acc_dtype = torch.float64 if values.dtype == torch.float64 else torch.float32
        # Group the rows of all matrices by start column, in a stable order.
        keys, order = torch.sort(cols.reshape(-1).to(torch.int32), stable=True)
        bounds = torch.searchsorted(
            keys, torch.arange(num_cols + 1, device=keys.device, dtype=keys.dtype)
        )
        rows_per_column = num_factors * num_rows // num_cols
        chunks = min(max(rows_per_column // _ROWS_PER_PROGRAM, 1), _MAX_CHUNKS)
        part = factors.new_empty(
            num_cols, chunks, segment_length, factors.shape[1], dtype=acc_dtype
        )
        grad_factors = grad.new_empty(num_cols, width)
        with torch.cuda.device(values.device):
            # A fixed tile fixes the summation order, so gradients reproduce
            # across runs, unlike an autotuned one.
            _segment_kernel[(num_cols, chunks, triton.cdiv(width, 32))](
                order, bounds, values, cols, factors, grad, part,
                num_rows, width, factors.stride(0), chunks,
                D=num_factors, L=segment_length,
                L_P2=triton.next_power_of_2(segment_length),
                ACC=_accumulator(values.dtype), BLOCK_M=64, BLOCK_R=32, num_warps=2,
            )  # fmt: skip
            block_r = min(triton.next_power_of_2(width), 128)
            _tap_kernel[(num_cols, triton.cdiv(width, block_r))](
                part, grad_factors, width, factors.stride(0), chunks,
                L=segment_length, ACC=_accumulator(values.dtype), BLOCK_R=block_r,
            )  # fmt: skip
        return grad_factors

    def _launch_grad_values(
        values: Tensor, cols: Tensor, factors: Tensor, grad: Tensor
    ) -> Tensor:
        """Compute the ``(D, M, L)`` gradient of the segment values."""
        num_factors, num_rows, segment_length = values.shape
        width = grad.shape[1]
        grad_values = torch.empty_like(values)
        with torch.cuda.device(values.device):
            _values_kernel[(triton.cdiv(num_rows, 16), num_factors)](
                values, cols, factors, grad, grad_values,
                num_rows, width, factors.stride(0),
                D=num_factors, L=segment_length,
                L_P2=triton.next_power_of_2(segment_length),
                ACC=_accumulator(values.dtype),
                BLOCK_M=16, BLOCK_R=min(triton.next_power_of_2(width), 64), num_warps=4,
            )  # fmt: skip
        return grad_values

    class _RCSProductFn(torch.autograd.Function):
        """Triton product with a PyTorch path for higher-order gradients.

        Takes contiguous stacked values ``(D, M, L)`` and offset start columns
        ``(D, M)``, and the ``D`` dense operands, all of one floating dtype, and
        computes only the gradients that ``needs_input_grad`` requests.
        """

        @staticmethod
        @torch.amp.custom_fwd(device_type="cuda")
        def forward(ctx: Any, values: Tensor, cols: Tensor, *others: Tensor) -> Tensor:
            factors = _stack_rows(others)
            out = _launch_product(values, cols, factors, others[0].shape[1])
            # Save the original operands for double backward to differentiate.
            ctx.save_for_backward(values, cols, factors, *others)
            return out

        @staticmethod
        @torch.amp.custom_bwd(device_type="cuda")
        def backward(ctx: Any, grad: Tensor) -> tuple[Tensor | None, ...]:
            values, cols, factors, *others = ctx.saved_tensors
            need_values, _, *need_others = ctx.needs_input_grad
            if torch.is_grad_enabled():
                # PyTorch operations keep this backward differentiable.
                inputs = [values, *others]
                needed = [
                    t for t, need in zip(inputs, (need_values, *need_others)) if need
                ]
                out = _product_torch(values, cols, torch.cat(others))
                grads = iter(torch.autograd.grad(out, needed, grad, create_graph=True))
                return (
                    next(grads) if need_values else None,
                    None,
                    *(next(grads) if need else None for need in need_others),
                )
            grad = grad.contiguous()
            grad_values = grad_others = None
            if need_values:
                grad_values = _launch_grad_values(values, cols, factors, grad)
            if any(need_others):
                grad_factors = _launch_grad_factors(values, cols, factors, grad)
                grad_others = grad_factors.split([other.shape[0] for other in others])
            return grad_values, None, *(grad_others or [None] * len(others))

else:
    _RCSProductFn = None


def rcs_product(matrices: Sequence["RCSMatrix"], others: Sequence[Tensor]) -> Tensor:
    """Return ``(matrices[0] @ others[0]) * ... * (matrices[-1] @ others[-1])``.

    All matrices need the same number of rows ``M`` and all dense operands the
    same number of columns ``R``, and ``others[i]`` has ``matrices[i].shape[1]``
    rows. On CUDA with Triton, matching segment lengths and one dtype, fused
    kernels compute the ``(M, R)`` product without materializing its factors,
    for any number of matrices; otherwise each product is computed in turn.
    Gradients flow to the matrices' ``values`` and to ``others``.

    Example::

        a = RCSMatrix(torch.rand(5, 2), torch.tensor([0, 1, 2, 3, 0]), 4)
        b = RCSMatrix(torch.rand(5, 2), torch.tensor([2, 0, 1, 1, 0]), 3)
        out = rcs_product([a, b], [torch.randn(4, 8), torch.randn(3, 8)])  # (5, 8)
    """
    if not matrices or len(matrices) != len(others):
        raise ValueError(
            f"expected as many dense operands as matrices, got {len(others)} "
            f"for {len(matrices)}"
        )
    num_rows = matrices[0].shape[0]
    width = others[0].shape[-1]
    for matrix, other in zip(matrices, others):
        if not isinstance(matrix, RCSMatrix) or isinstance(other, RCSMatrix):
            raise TypeError("expected RCS matrices and dense operands")
        if matrix.shape[0] != num_rows:
            raise ValueError(f"expected {num_rows} rows, got {matrix.shape[0]}")
        if other.ndim != 2 or other.shape != (matrix.shape[1], width):
            raise ValueError(
                f"expected an operand of shape ({matrix.shape[1]}, {width}), "
                f"got {tuple(other.shape)}"
            )

    values = [matrix.values for matrix in matrices]
    segment_length = values[0].shape[1]
    dtype, device = values[0].dtype, values[0].device
    if (
        _RCSProductFn is not None
        and device.type == "cuda"
        and num_rows > 0
        and width > 0
        and segment_length > 0
        and all(v.shape[1] == segment_length for v in values)
        and all(t.dtype == dtype and t.device == device for t in (*values, *others))
    ):
        # Match matmul autocast, leaving float64 and low-precision inputs alone.
        if dtype == torch.float32 and torch.is_autocast_enabled("cuda"):
            dtype = torch.get_autocast_dtype("cuda")
        offsets = accumulate((m.shape[1] for m in matrices), initial=0)
        cols = torch.stack([m._columns() + o for m, o in zip(matrices, offsets)])
        return _RCSProductFn.apply(
            torch.stack(values).to(dtype), cols, *(other.to(dtype) for other in others)
        )
    return reduce(mul, (m._matmul_torch(o) for m, o in zip(matrices, others)))


class RCSMatrix(Tensor):
    """Sparse matrix whose rows each store one contiguous segment.

    ``values[i]`` fills ``L`` columns starting at ``start_cols[i]`` in an
    ``(M, N)`` matrix; all other entries are zero. Shorter segments can be
    zero-padded. Multiplication by a dense ``(N, K)`` matrix or ``(N,)`` vector
    uses the compact representation and supports gradients for both operands,
    including second derivatives; :func:`rcs_product` fuses the elementwise
    product of several such multiplications. See the module notes for kernel
    selection.

    Supported operations are ``A @ B``, ``A.matmul(B)``, ``A.mm(B)``,
    ``torch.matmul(A, B)``, ``torch.mm(A, B)``, and :meth:`to_dense`.
    ``matmul`` also accepts ``input=``/``other=`` keywords and ``out=None``.
    Metadata access, pickling, and deep copying preserve the compact form;
    other tensor operations raise ``NotImplementedError``.

    Args:
        values: Segment values of shape ``(M, L)``, retained without copying.
        start_cols: Integer start columns of shape ``(M,)`` satisfying
            ``0 <= start_cols[i] <= num_cols - L``. Copied as int64 to the
            device of ``values``; later changes to the argument are ignored.
        num_cols: Number of columns ``N`` in the represented matrix.
        check_invariants: Check the bounds of ``start_cols``, which
            synchronizes with the device once. Callers that guarantee them,
            such as the FUTON bases, skip the check.

    The ``values`` attribute shadows ``Tensor.values``. If its storage moves
    to another device in place (for example through ``module.cuda()``),
    computation follows it, while wrapper metadata retains the original device.

    Example::

        values = torch.randn(4, 2, requires_grad=True)
        matrix = RCSMatrix(values, torch.tensor([0, 1, 2, 3]), num_cols=5)
        product = matrix @ torch.randn(5, 3)  # (4, 3)
    """

    values: Tensor
    start_cols: Tensor

    @staticmethod
    def __new__(
        cls,
        values: Tensor,
        start_cols: Tensor,
        num_cols: int,
        check_invariants: bool = True,
    ) -> "RCSMatrix":
        if values.ndim != 2:
            raise ValueError(
                f"values must have shape (M, L), got {tuple(values.shape)}"
            )
        if num_cols < 0:
            raise ValueError(f"num_cols must be non-negative, got {num_cols}")
        return torch.Tensor._make_wrapper_subclass(
            cls,
            (values.shape[0], num_cols),
            dtype=values.dtype,
            device=values.device,
            requires_grad=values.requires_grad,
        )

    def __init__(
        self,
        values: Tensor,
        start_cols: Tensor,
        num_cols: int,
        check_invariants: bool = True,
    ) -> None:
        num_rows, segment_length = values.shape
        if start_cols.shape != (num_rows,):
            raise ValueError(
                f"start_cols must have shape ({num_rows},), got {tuple(start_cols.shape)}"
            )
        if (
            start_cols.dtype.is_floating_point
            or start_cols.dtype.is_complex
            or start_cols.dtype == torch.bool
        ):
            raise ValueError(
                f"start_cols must be an integer tensor, got {start_cols.dtype}"
            )
        start_cols = start_cols.to(device=values.device, dtype=torch.long, copy=True)
        # Combine the bounds checks to synchronize with CUDA only once.
        if (
            check_invariants
            and num_rows > 0
            and bool(
                ((start_cols < 0) | (start_cols + segment_length > num_cols)).any()
            )
        ):
            raise ValueError(
                f"start_cols must lie in [0, {num_cols - segment_length}] "
                f"for L={segment_length}, N={num_cols}"
            )

        self.values = values
        self.start_cols = start_cols
        self._col_idx: Tensor | None = None

    def _columns(self) -> Tensor:
        """Return the start columns, following any device change of ``values``."""
        if self.start_cols.device != self.values.device:
            self.start_cols = self.start_cols.to(self.values.device)
            self._col_idx = None
        return self.start_cols

    def _column_indices(self) -> Tensor:
        """Return the ``(M, L)`` column of every stored value, cached."""
        start_cols = self._columns()
        if self._col_idx is None:
            self._col_idx = start_cols[:, None] + torch.arange(
                self.values.shape[1], device=start_cols.device
            )
        return self._col_idx

    def to_dense(self) -> Tensor:
        """Return a dense matrix on the dtype and device of ``values``, retaining gradients."""
        return self.values.new_zeros(self.shape).scatter_(
            1, self._column_indices(), self.values
        )

    def _matmul_dense(self, other: Tensor) -> Tensor:
        """Compute ``self @ other`` using the compact representation."""
        if other.ndim == 1:
            return self._matmul_dense(other.unsqueeze(1)).squeeze(1)
        return rcs_product([self], [other])

    def _matmul_torch(self, other: Tensor) -> Tensor:
        """Compute ``self @ other`` with PyTorch operations."""
        num_rows, num_cols = self.shape
        segment_length = self.values.shape[1]
        grad_needed = torch.is_grad_enabled() and (
            self.values.requires_grad or other.requires_grad
        )
        if segment_length > 0 and not grad_needed:
            # CSR avoids materializing (M, L, K) windows, but does not support
            # the higher-order derivatives of the gather path below.
            columns = self._column_indices()
            row_offsets = torch.arange(
                0, num_rows * segment_length + 1, segment_length, device=columns.device
            )
            csr = torch.sparse_csr_tensor(
                row_offsets,
                columns.reshape(-1),
                self.values.reshape(-1),
                size=(num_rows, num_cols),
            )
            return csr @ other
        # Gather only rows touched by each segment, keeping this path differentiable.
        windows = other[self._column_indices()]
        return torch.bmm(self.values.unsqueeze(1), windows).squeeze(1)

    @classmethod
    def __torch_function__(
        cls,
        func: Callable[..., Any],
        types: tuple[type, ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        if kwargs is None:
            kwargs = {}
        if func in _MATMUL_FUNCS:
            remaining = dict(kwargs)
            out = remaining.pop("out", None)
            operands = list(args)
            if not operands and "input" in remaining:
                operands.append(remaining.pop("input"))
            if len(operands) == 1 and "other" in remaining:
                operands.append(remaining.pop("other"))
            if len(operands) == 2 and not remaining and out is None:
                left, right = operands
                if (
                    isinstance(left, RCSMatrix)
                    and isinstance(right, Tensor)
                    and not isinstance(right, RCSMatrix)
                ):
                    return left._matmul_dense(right)
        # Metadata accessors (shape, dtype, device, ...) work on the
        # storage-less wrapper; data ops fall through to __torch_dispatch__.
        with torch._C.DisableTorchFunctionSubclass():
            return func(*args, **kwargs)

    @classmethod
    def __torch_dispatch__(
        cls,
        func: Callable[..., Any],
        types: tuple[type, ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        raise NotImplementedError(
            f"{func} is not supported on the compact RCS form; "
            "convert with .to_dense() first"
        )

    def __deepcopy__(self, memo: dict[int, Any]) -> "RCSMatrix":
        result = RCSMatrix(
            copy.deepcopy(self.values, memo),
            copy.deepcopy(self.start_cols, memo),
            self.shape[1],
            check_invariants=False,
        )
        memo[id(self)] = result
        return result

    def __repr__(self) -> str:
        num_rows, num_cols = self.shape
        return (
            f"RCSMatrix(shape=({num_rows}, {num_cols}), L={self.values.shape[1]}, "
            f"dtype={self.dtype}, device={self.device},\n"
            f"values={self.values},\nstart_cols={self.start_cols})"
        )
