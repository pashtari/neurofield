"""Row-contiguous sparse matrices with differentiable dense multiplication.

Each row stores a contiguous segment of ``L`` values and its starting column.
Matching CUDA operands use Triton kernels when available. Other products use
CSR during inference or a gather-and-contract path when gradients are needed.

Triton backward uses a cached, stably sorted transpose structure to reduce
contributions in a fixed order without atomics. First-order gradients are
deterministic; ``create_graph=True`` switches to PyTorch operations for double
backward, whose CUDA ``index_add_`` need not be deterministic. Accumulation
uses float32, or float64 for double inputs. CUDA autocast follows matmul's
dtype rules. Kernels specialize on segment length; long segments increase
compilation cost because their loops are unrolled.
"""

import copy
from collections.abc import Callable
from typing import Any, TypeAlias

import torch
from torch import Tensor

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None

__all__ = ["RCSMatrix"]

_MATMUL_FUNCS = {
    torch.matmul,
    torch.mm,
    Tensor.matmul,
    Tensor.mm,
    Tensor.__matmul__,
}

# Column boundaries, source rows, and source offsets within each row segment.
_TransposeCache: TypeAlias = tuple[Tensor, Tensor, Tensor]


def _build_transpose_cache(
    start_cols: Tensor, segment_length: int, num_cols: int
) -> _TransposeCache:
    """Group column contributors in a stable order for deterministic reduction.

    Returns int64 column boundaries of shape ``(num_cols + 1,)`` and int32
    source rows and segment offsets of shape ``(M * L,)``. Requires ``L > 0``.
    """
    device = start_cols.device
    starts = start_cols.to(torch.long).reshape(-1)
    offsets = torch.arange(segment_length, device=device, dtype=torch.long)
    # int32 keys halve the radix-sort passes (up to ~10% faster training steps).
    columns = (starts[:, None] + offsets).reshape(-1).to(torch.int32)
    order = torch.argsort(columns, stable=True)
    column_boundaries = torch.searchsorted(
        columns[order], torch.arange(num_cols + 1, device=device, dtype=torch.int32)
    )
    return (
        column_boundaries.contiguous(),
        (order // segment_length).to(torch.int32).contiguous(),
        (order % segment_length).to(torch.int32).contiguous(),
    )


# Kernel notation: G = values (M, L), J = start_cols (M,), B = dense right
# operand (N, K); J_i is the start column of row i.
if triton is not None:

    # Tune tile shape with few candidates to limit first-call overhead.
    # Replaying forward is safe: each program overwrites its own output tile.
    _FWD_CONFIGS = [
        triton.Config({"BLOCK_M": 32, "BLOCK_K": 128}, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_K": 64}, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_K": 32}, num_warps=4),
    ]

    @triton.autotune(configs=_FWD_CONFIGS, key=["K", "L"])
    @triton.jit
    def _rcs_fwd_kernel(
        g_ptr,
        j_ptr,
        b_ptr,
        out_ptr,
        M,
        K,
        stride_gm,
        stride_gl,
        stride_bn,
        stride_bk,
        stride_om,
        stride_ok,
        L: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
        ACC_DTYPE: tl.constexpr,
    ):
        """out[i, k] = sum_l G[i, l] * B[J_i + l, k] over a (BLOCK_M, BLOCK_K) tile."""
        pid_m = tl.program_id(0)
        pid_k = tl.program_id(1)
        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rk = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_m = rm < M
        mask_k = rk < K
        mask = mask_m[:, None] & mask_k[None, :]
        rm = rm.to(tl.int64)  # 64-bit addressing so large M * strides never overflow
        j = tl.load(j_ptr + rm, mask=mask_m, other=0)

        acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=ACC_DTYPE)
        for l in tl.static_range(L):
            g = tl.load(g_ptr + rm * stride_gm + l * stride_gl, mask=mask_m, other=0.0)
            b = tl.load(
                b_ptr + (j + l)[:, None] * stride_bn + rk[None, :] * stride_bk,
                mask=mask,
                other=0.0,
            )
            acc += g.to(ACC_DTYPE)[:, None] * b.to(ACC_DTYPE)

        tl.store(
            out_ptr + rm[:, None] * stride_om + rk[None, :] * stride_ok,
            acc.to(out_ptr.dtype.element_ty),
            mask=mask,
        )

    @triton.jit
    def _rcs_bwd_g_kernel(
        go_ptr,
        j_ptr,
        b_ptr,
        gg_ptr,
        M,
        K,
        stride_gom,
        stride_gok,
        stride_bn,
        stride_bk,
        stride_ggm,
        stride_ggl,
        L: tl.constexpr,
        L_P2: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
        ACC_DTYPE: tl.constexpr,
    ):
        """grad_G[i, l] = dot(grad_out[i, :], B[J_i + l, :]); grid = (M-blocks,).

        Each grad_out tile is loaded once and reused for all L accumulators
        (a (BLOCK_M, L_P2) register accumulator updated via a constexpr one-hot
        select, which compiles to a predicated register move).
        """
        pid_m = tl.program_id(0)
        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = rm < M
        rm = rm.to(tl.int64)
        j = tl.load(j_ptr + rm, mask=mask_m, other=0)

        l_idx = tl.arange(0, L_P2)
        acc = tl.zeros((BLOCK_M, L_P2), dtype=ACC_DTYPE)
        for k0 in range(0, K, BLOCK_K):
            rk = k0 + tl.arange(0, BLOCK_K)
            mask_k = rk < K
            mask = mask_m[:, None] & mask_k[None, :]
            go = tl.load(
                go_ptr + rm[:, None] * stride_gom + rk[None, :] * stride_gok,
                mask=mask,
                other=0.0,
            ).to(ACC_DTYPE)
            for li in tl.static_range(L):
                b = tl.load(
                    b_ptr + (j + li)[:, None] * stride_bn + rk[None, :] * stride_bk,
                    mask=mask,
                    other=0.0,
                )
                partial = tl.sum(go * b.to(ACC_DTYPE), axis=1)
                acc = tl.where((l_idx == li)[None, :], acc + partial[:, None], acc)

        tl.store(
            gg_ptr + rm[:, None] * stride_ggm + l_idx[None, :] * stride_ggl,
            acc.to(gg_ptr.dtype.element_ty),
            mask=mask_m[:, None] & (l_idx < L)[None, :],
        )

    @triton.jit
    def _rcs_bwd_b_kernel(
        rowptr_ptr,
        srci_ptr,
        srcl_ptr,
        g_ptr,
        go_ptr,
        gb_ptr,
        K,
        stride_gm,
        stride_gl,
        stride_gom,
        stride_gok,
        stride_gbn,
        stride_gbk,
        BLOCK_E: tl.constexpr,
        BLOCK_K: tl.constexpr,
        ACC_DTYPE: tl.constexpr,
    ):
        """Accumulate one (row j, K-tile) block of grad_B over its transpose segment.

        grad_B[j, k_tile] = sum over p in rowptr[j]..rowptr[j+1] of
        G[src_i[p], src_l[p]] * grad_out[src_i[p], k_tile].
        One program per (j, K-tile); fixed traversal order => deterministic.
        """
        j = tl.program_id(0)
        pid_k = tl.program_id(1)
        rk = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_k = rk < K
        rk = rk.to(tl.int64)

        start = tl.load(rowptr_ptr + j)
        end = tl.load(rowptr_ptr + j + 1)

        acc = tl.zeros((BLOCK_K,), dtype=ACC_DTYPE)
        for p0 in range(start, end, BLOCK_E):
            re = p0 + tl.arange(0, BLOCK_E)
            mask_e = re < end
            i = tl.load(srci_ptr + re, mask=mask_e, other=0).to(tl.int64)
            l = tl.load(srcl_ptr + re, mask=mask_e, other=0).to(tl.int64)
            g = tl.load(
                g_ptr + i * stride_gm + l * stride_gl, mask=mask_e, other=0.0
            ).to(ACC_DTYPE)
            go = tl.load(
                go_ptr + i[:, None] * stride_gom + rk[None, :] * stride_gok,
                mask=mask_e[:, None] & mask_k[None, :],
                other=0.0,
            ).to(ACC_DTYPE)
            acc += tl.sum(g[:, None] * go, axis=0)

        tl.store(
            gb_ptr + j.to(tl.int64) * stride_gbn + rk * stride_gbk,
            acc.to(gb_ptr.dtype.element_ty),
            mask=mask_k,
        )

    def _accumulator_dtype(dtype: torch.dtype) -> tl.dtype:
        return tl.float64 if dtype == torch.float64 else tl.float32

    def _launch_forward(values: Tensor, start_cols: Tensor, other: Tensor) -> Tensor:
        """Multiply segment values ``(M, L)`` by a dense matrix ``(N, K)`` on CUDA."""
        num_rows, segment_length = values.shape
        out_features = other.shape[1]
        out = torch.empty(
            num_rows, out_features, dtype=values.dtype, device=values.device
        )
        if num_rows == 0 or out_features == 0:
            return out
        if segment_length == 0:
            return out.zero_()
        grid = lambda meta: (
            triton.cdiv(num_rows, meta["BLOCK_M"]),
            triton.cdiv(out_features, meta["BLOCK_K"]),
        )
        _rcs_fwd_kernel[grid](
            values,
            start_cols,
            other,
            out,
            num_rows,
            out_features,
            values.stride(0),
            values.stride(1),
            other.stride(0),
            other.stride(1),
            out.stride(0),
            out.stride(1),
            L=segment_length,
            ACC_DTYPE=_accumulator_dtype(values.dtype),
        )
        return out

    def _launch_grad_values(
        grad_out: Tensor,
        start_cols: Tensor,
        other: Tensor,
        num_rows: int,
        segment_length: int,
    ) -> Tensor:
        """Compute the ``(M, L)`` values gradient in ``grad_out.dtype``."""
        out_features = grad_out.shape[1]
        grad_values = torch.empty(
            num_rows, segment_length, dtype=grad_out.dtype, device=grad_out.device
        )
        if num_rows == 0 or segment_length == 0:
            return grad_values
        if out_features == 0:
            return grad_values.zero_()
        block_k = min(128, triton.next_power_of_2(out_features))
        grid = (triton.cdiv(num_rows, 16),)
        _rcs_bwd_g_kernel[grid](
            grad_out,
            start_cols,
            other,
            grad_values,
            num_rows,
            out_features,
            grad_out.stride(0),
            grad_out.stride(1),
            other.stride(0),
            other.stride(1),
            grad_values.stride(0),
            grad_values.stride(1),
            L=segment_length,
            L_P2=triton.next_power_of_2(segment_length),
            BLOCK_M=16,
            BLOCK_K=block_k,
            ACC_DTYPE=_accumulator_dtype(grad_out.dtype),
            num_warps=4,
        )
        return grad_values

    def _launch_grad_other(
        values: Tensor, cache: _TransposeCache | None, grad_out: Tensor, num_cols: int
    ) -> Tensor:
        """Reduce the ``(N, K)`` right-operand gradient in float32 or float64.

        ``cache`` may be ``None`` only for an empty product; the caller casts the
        result back to the operand dtype.
        """
        num_rows, segment_length = values.shape
        out_features = grad_out.shape[1]
        acc_dtype = torch.float64 if values.dtype == torch.float64 else torch.float32
        if num_rows == 0 or out_features == 0 or segment_length == 0 or num_cols == 0:
            return torch.zeros(
                num_cols, out_features, dtype=acc_dtype, device=values.device
            )
        column_boundaries, source_rows, source_offsets = cache
        grad_other = torch.empty(
            num_cols, out_features, dtype=acc_dtype, device=values.device
        )
        block_k = min(128, triton.next_power_of_2(out_features))
        grid = (num_cols, triton.cdiv(out_features, block_k))
        _rcs_bwd_b_kernel[grid](
            column_boundaries,
            source_rows,
            source_offsets,
            values,
            grad_out,
            grad_other,
            out_features,
            values.stride(0),
            values.stride(1),
            grad_out.stride(0),
            grad_out.stride(1),
            grad_other.stride(0),
            grad_other.stride(1),
            BLOCK_E=32,
            BLOCK_K=block_k,
            ACC_DTYPE=_accumulator_dtype(acc_dtype),
            num_warps=4,
        )
        return grad_other

    class _RCSMatmulFn(torch.autograd.Function):
        """Triton multiplication with a PyTorch path for higher-order gradients.

        The optional transpose cache is built lazily when the right operand needs
        a gradient. Only gradients requested by ``needs_input_grad`` are computed.
        """

        @staticmethod
        @torch.amp.custom_fwd(device_type="cuda")
        def forward(
            ctx: Any,
            values: Tensor,
            start_cols: Tensor,
            other: Tensor,
            cache: _TransposeCache | None = None,
        ) -> Tensor:
            if not (values.is_cuda and other.is_cuda and start_cols.is_cuda):
                raise RuntimeError("triton RCS matmul requires CUDA tensors")

            compute_dtype = torch.promote_types(values.dtype, other.dtype)
            # Match matmul autocast, leaving float64 and low-precision inputs alone.
            if compute_dtype == torch.float32 and torch.is_autocast_enabled("cuda"):
                compute_dtype = torch.get_autocast_dtype("cuda")

            start_cols = start_cols.to(torch.long).contiguous()
            out = _launch_forward(
                values.to(compute_dtype), start_cols, other.to(compute_dtype)
            )

            # Save original operands so backward can reapply casts under grad
            # mode and retain the graph for higher-order derivatives.
            ctx.save_for_backward(values, start_cols, other)
            ctx.compute_dtype = compute_dtype
            ctx.cache = cache
            return out

        @staticmethod
        @torch.amp.custom_bwd(device_type="cuda")
        def backward(
            ctx: Any, grad_out: Tensor
        ) -> tuple[Tensor | None, None, Tensor | None, None]:
            values, start_cols, other = ctx.saved_tensors
            need_values, _, need_other = ctx.needs_input_grad[:3]
            num_rows, segment_length = values.shape
            num_cols = other.shape[0]
            grad_values = grad_other = None

            if torch.is_grad_enabled():
                # PyTorch operations keep this backward differentiable.
                columns = start_cols[:, None] + torch.arange(
                    segment_length, device=start_cols.device
                )
                compute_values = values.to(grad_out.dtype)
                compute_other = other.to(grad_out.dtype)
                if need_values:
                    grad_values = (
                        (grad_out.unsqueeze(1) * compute_other[columns])
                        .sum(-1)
                        .to(values.dtype)
                    )
                if need_other:
                    contributions = (
                        compute_values.unsqueeze(-1) * grad_out.unsqueeze(1)
                    ).reshape(num_rows * segment_length, -1)
                    grad_other = (
                        torch.zeros_like(compute_other)
                        .index_add_(0, columns.reshape(-1), contributions)
                        .to(other.dtype)
                    )
                return grad_values, None, grad_other, None

            compute_dtype = ctx.compute_dtype
            compute_grad = grad_out.to(compute_dtype)
            if need_values:
                grad_values = _launch_grad_values(
                    compute_grad,
                    start_cols,
                    other.to(compute_dtype),
                    num_rows,
                    segment_length,
                )
                grad_values = grad_values.to(values.dtype)
            if need_other:
                cache = ctx.cache
                if cache is None and segment_length > 0:
                    cache = _build_transpose_cache(start_cols, segment_length, num_cols)
                grad_other = _launch_grad_other(
                    values.to(compute_dtype), cache, compute_grad, num_cols
                )
                grad_other = grad_other.to(other.dtype)
            return grad_values, None, grad_other, None

else:
    _RCSMatmulFn = None


class RCSMatrix(Tensor):
    """Sparse matrix whose rows each store one contiguous segment.

    ``values[i]`` fills ``L`` columns starting at ``start_cols[i]`` in an
    ``(M, N)`` matrix; all other entries are zero. Shorter segments can be
    zero-padded. Multiplication by a dense ``(N, K)`` matrix or ``(N,)`` vector
    uses the compact representation and supports gradients for both operands,
    including second derivatives. See the module notes for kernel selection.

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
    def __new__(cls, values: Tensor, start_cols: Tensor, num_cols: int) -> "RCSMatrix":
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

    def __init__(self, values: Tensor, start_cols: Tensor, num_cols: int) -> None:
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
        if num_rows > 0 and bool(
            ((start_cols < 0) | (start_cols + segment_length > num_cols)).any()
        ):
            raise ValueError(
                f"start_cols must lie in [0, {num_cols - segment_length}] "
                f"for L={segment_length}, N={num_cols}"
            )

        self.values = values
        self.start_cols = start_cols
        self._col_idx = start_cols[:, None] + torch.arange(
            segment_length, device=values.device
        )
        # Build the transpose only when a right-operand gradient is needed.
        self._transpose_cache: _TransposeCache | None = None

    def _column_indices(self) -> Tensor:
        """Return cached column indices, following any device change of ``values``."""
        if self._col_idx.device != self.values.device:
            self._col_idx = self._col_idx.to(self.values.device)
            self.start_cols = self.start_cols.to(self.values.device)
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
        num_rows, num_cols = self.shape
        if other.ndim != 2 or other.shape[0] != num_cols:
            raise ValueError(
                f"expected other of shape ({num_cols}, K) or ({num_cols},), "
                f"got {tuple(other.shape)}"
            )
        segment_length = self.values.shape[1]
        if (
            _RCSMatmulFn is not None
            and segment_length > 0
            and self.values.is_cuda
            and other.is_cuda
            and self.values.dtype == other.dtype
        ):
            self._column_indices()  # Keep start_cols on the compute device.
            if torch.is_grad_enabled() and other.requires_grad:
                if (
                    self._transpose_cache is None
                    or self._transpose_cache[0].device != self.values.device
                ):
                    self._transpose_cache = _build_transpose_cache(
                        self.start_cols, segment_length, num_cols
                    )
            return _RCSMatmulFn.apply(
                self.values, self.start_cols, other, self._transpose_cache
            )
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
