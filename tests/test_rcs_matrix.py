"""Tests for the RCSMatrix compact row-contiguous sparse matrix."""

import copy
import io
import pickle
from functools import reduce
from operator import mul

import pytest
import torch
from torch import nn

from neurofield.models import RCSMatrix, rcs_product

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


def make_rcs(M, N, L, device="cpu", requires_grad=False, zero_pad_rows=False, seed=0):
    """Build a random RCSMatrix and return it with its raw (values, start_cols)."""
    g = torch.Generator().manual_seed(seed)
    values = torch.randn(M, L, generator=g)
    if zero_pad_rows and M > 0:
        values[::2, L // 2 :] = 0.0  # rows with fewer than L non-zeros
    values = values.to(device).requires_grad_(requires_grad)
    start_cols = torch.randint(0, max(N - L + 1, 1), (M,), generator=g).to(device)
    return RCSMatrix(values, start_cols, N), values, start_cols


class TestMatmul:
    @pytest.mark.parametrize(
        "M, N, L, K",
        [
            (7, 11, 3, 5),
            (1, 1, 1, 1),
            (4, 6, 6, 3),
            (32, 100, 1, 8),
            (0, 10, 4, 2),
            (5, 8, 3, 1),
        ],
    )
    def test_matches_dense(self, M, N, L, K):
        A, _, _ = make_rcs(M, N, L, zero_pad_rows=True)
        B = torch.randn(N, K)
        out = A @ B
        assert out.shape == (M, K)
        assert torch.allclose(out, A.to_dense() @ B, atol=1e-5)

    def test_all_call_forms(self):
        A, _, _ = make_rcs(5, 9, 3)
        B = torch.randn(9, 4)
        ref = A.to_dense() @ B
        for out in (
            A @ B,
            A.matmul(B),
            torch.matmul(A, B),
            torch.mm(A, B),
            torch.matmul(input=A, other=B),
            torch.matmul(A, other=B),
            torch.matmul(A, B, out=None),
        ):
            assert torch.allclose(out, ref, atol=1e-5)

    @pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=CUDA)])
    def test_inference_path_matches_autograd_path(self, device):
        A, values, _ = make_rcs(32, 100, 4, device=device)
        B = torch.randn(100, 8, device=device)
        with torch.no_grad():
            out_csr = A @ B  # CSR kernel
        values.requires_grad_(True)
        out_gather = A @ B  # differentiable gather path
        assert torch.allclose(out_csr, out_gather.detach(), atol=1e-5)
        assert torch.allclose(out_csr, A.to_dense().detach() @ B, atol=1e-5)

    def test_result_is_plain_tensor(self):
        A, _, _ = make_rcs(5, 9, 3)
        assert type(A @ torch.randn(9, 4)) is torch.Tensor

    def test_matvec(self):
        A, _, _ = make_rcs(5, 9, 3)
        v = torch.randn(9)
        out = A @ v
        assert out.shape == (5,)
        assert torch.allclose(out, A.to_dense() @ v, atol=1e-5)

    def test_parameter_rhs(self):
        A, _, _ = make_rcs(5, 9, 3)
        P = nn.Parameter(torch.randn(9, 4))
        out = A @ P
        assert torch.allclose(out, A.to_dense() @ P, atol=1e-5)
        out.sum().backward()
        P2 = P.detach().clone().requires_grad_(True)
        (A.to_dense() @ P2).sum().backward()
        assert torch.allclose(P.grad, P2.grad, atol=1e-5)

    def test_wrong_rhs_rows_raises(self):
        A, _, _ = make_rcs(3, 5, 2)
        with pytest.raises(ValueError):
            A @ torch.randn(4, 2)

    def test_real_out_kwarg_raises(self):
        A, _, _ = make_rcs(5, 9, 3)
        with pytest.raises(NotImplementedError):
            torch.matmul(A, torch.randn(9, 4), out=torch.empty(5, 4))

    def test_dense_at_rcs_raises(self):
        A, _, _ = make_rcs(3, 5, 2)
        with pytest.raises((NotImplementedError, RuntimeError)):
            torch.randn(4, 3) @ A


def make_product(Ks, M, L, R, device="cpu", dtype=torch.float32, seed=0):
    """Random RCS matrices of widths ``Ks`` and dense operands, all requiring grad."""
    g = torch.Generator().manual_seed(seed)
    matrices, others = [], []
    for K in Ks:
        values = torch.randn(M, L, generator=g, dtype=dtype).to(device).requires_grad_()
        start_cols = torch.randint(0, K - L + 1, (M,), generator=g).to(device)
        matrices.append(RCSMatrix(values, start_cols, K))
        others.append(
            torch.randn(K, R, generator=g, dtype=dtype).to(device).requires_grad_()
        )
    return matrices, others


class TestProduct:
    @pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=CUDA)])
    @pytest.mark.parametrize(
        "Ks, M, L, R",
        [
            ([9], 257, 3, 5),
            ([7, 12], 257, 2, 18),
            ([16, 16, 16], 257, 6, 218),  # rows not a multiple of 16 values
            ([10, 20, 30], 100, 4, 1),
            ([8, 8, 8], 20000, 2, 40),  # several programs per column
        ],
    )
    def test_matches_dense(self, device, Ks, M, L, R):
        matrices, others = make_product(Ks, M, L, R, device)
        inputs = [m.values for m in matrices] + others
        out = rcs_product(matrices, others)
        grad = torch.randn_like(out)
        grads = torch.autograd.grad(out, inputs, grad)
        ref = reduce(mul, (m.to_dense() @ o for m, o in zip(matrices, others)))
        ref_grads = torch.autograd.grad(ref, inputs, grad)
        assert torch.allclose(out, ref, rtol=1e-4, atol=1e-4)
        for got, expected in zip(grads, ref_grads):
            assert torch.allclose(got, expected, rtol=1e-4, atol=1e-3)

    @pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=CUDA)])
    def test_gradcheck(self, device):
        matrices, others = make_product([6, 5, 7], 9, 3, 4, device, torch.float64)
        starts = [m.start_cols for m in matrices]

        def product(*tensors):
            rcs = [
                RCSMatrix(v, s, o.shape[0])
                for v, s, o in zip(tensors, starts, tensors[3:])
            ]
            return rcs_product(rcs, tensors[3:])

        inputs = (*(m.values for m in matrices), *others)
        assert torch.autograd.gradcheck(product, inputs)
        assert torch.autograd.gradgradcheck(product, inputs)

    @CUDA
    def test_bit_deterministic_gradients(self):
        matrices, others = make_product([64, 64, 64], 4096, 4, 48, "cuda")
        inputs = [m.values for m in matrices] + others
        grads = [
            torch.autograd.grad(rcs_product(matrices, others).square().sum(), inputs)
            for _ in range(3)
        ]
        for repeat in grads[1:]:
            assert all(torch.equal(a, b) for a, b in zip(repeat, grads[0]))

    def test_rejects_mismatched_operands(self):
        matrices, others = make_product([6, 5], 4, 2, 3)
        with pytest.raises(ValueError):
            rcs_product(matrices, others[:1])
        with pytest.raises(ValueError):
            rcs_product(matrices, [others[0], torch.randn(5, 4)])
        with pytest.raises(ValueError):
            rcs_product([matrices[0], make_product([5], 3, 2, 3)[0][0]], others)
        with pytest.raises(TypeError):
            rcs_product(matrices, [others[0], matrices[1]])


class TestToDense:
    def test_layout(self):
        A, values, start_cols = make_rcs(6, 12, 4)
        dense = A.to_dense()
        for i in range(6):
            row = torch.zeros(12)
            row[start_cols[i] : start_cols[i] + 4] = values[i]
            assert torch.equal(dense[i], row)


class TestAutograd:
    def test_grads_match_dense(self):
        A, values, start_cols = make_rcs(5, 9, 3, requires_grad=True)
        B = torch.randn(9, 4, requires_grad=True)
        (A @ B).square().sum().backward()
        values2 = values.detach().clone().requires_grad_(True)
        B2 = B.detach().clone().requires_grad_(True)
        (RCSMatrix(values2, start_cols, 9).to_dense() @ B2).square().sum().backward()
        assert torch.allclose(values.grad, values2.grad, atol=1e-5)
        assert torch.allclose(B.grad, B2.grad, atol=1e-5)

    def test_gradcheck(self):
        values = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)
        start_cols = torch.randint(0, 6, (4,))
        B = torch.randn(8, 2, dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(
            lambda g, b: RCSMatrix(g, start_cols, 8) @ b, (values, B)
        )

    def test_gradgradcheck(self):
        values = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)
        start_cols = torch.randint(0, 6, (4,))
        B = torch.randn(8, 2, dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradgradcheck(
            lambda g, b: RCSMatrix(g, start_cols, 8) @ b, (values, B)
        )

    def test_to_dense_gradcheck(self):
        values = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)
        start_cols = torch.randint(0, 6, (4,))
        assert torch.autograd.gradcheck(
            lambda g: RCSMatrix(g, start_cols, 8).to_dense(), (values,)
        )


class TestMetadata:
    def test_shape_dtype_device_repr(self):
        A, _, _ = make_rcs(6, 12, 4)
        assert A.shape == (6, 12)
        assert A.dtype == torch.float32
        assert A.device.type == "cpu"
        assert A.ndim == 2
        assert repr(A).startswith("RCSMatrix(shape=(6, 12)")

    def test_unsupported_op_raises(self):
        A, _, _ = make_rcs(3, 5, 2)
        with pytest.raises(NotImplementedError):
            A + 1


class TestValidation:
    @pytest.mark.parametrize(
        "values, start_cols, num_cols",
        [
            (torch.randn(3, 2), torch.rand(3), 5),  # float start_cols
            (torch.randn(2, 2), torch.tensor([True, False]), 5),  # bool start_cols
            (torch.randn(3, 2), torch.tensor([0, 1, 4]), 5),  # segment out of range
            (torch.randn(3, 2), torch.zeros(4, dtype=torch.long), 5),  # wrong J shape
            (torch.randn(3), torch.zeros(3, dtype=torch.long), 5),  # 1D values
            (
                torch.randn(0, 3),
                torch.zeros(0, dtype=torch.long),
                -5,
            ),  # negative N, M=0
            (
                torch.randn(3, 2),
                torch.zeros(3, dtype=torch.long),
                -7,
            ),  # negative N, M>0
        ],
    )
    def test_rejected(self, values, start_cols, num_cols):
        with pytest.raises(ValueError):
            RCSMatrix(values, start_cols, num_cols)

    def test_start_cols_copied(self):
        start_cols = torch.arange(5, dtype=torch.long)
        A = RCSMatrix(torch.ones(5, 2), start_cols, 8)
        before = A.to_dense().clone()
        start_cols += 1
        assert A.start_cols is not start_cols
        assert torch.equal(A.to_dense(), before)


class _Holder(nn.Module):
    def __init__(self):
        super().__init__()
        self.G = nn.Parameter(torch.randn(5, 3))
        self.A = RCSMatrix(self.G, torch.randint(0, 7, (5,)), 9)

    def forward(self, x):
        return self.A @ x


class TestCopy:
    def test_deepcopy_matrix(self):
        A, _, _ = make_rcs(5, 9, 3)
        A2 = copy.deepcopy(A)
        assert torch.equal(A2.to_dense(), A.to_dense())
        assert A2.values.data_ptr() != A.values.data_ptr()

    def test_deepcopy_module(self):
        m = _Holder()
        m2 = copy.deepcopy(m)
        assert torch.equal(m2.A.to_dense().detach(), m.A.to_dense().detach())

    def test_pickle_round_trip(self):
        A, _, _ = make_rcs(5, 9, 3)
        buf = io.BytesIO()
        pickle.dump(A, buf)
        buf.seek(0)
        A2 = pickle.load(buf)
        assert torch.equal(A2.to_dense(), A.to_dense())


@CUDA
class TestCuda:
    def test_matmul(self):
        A, _, _ = make_rcs(64, 256, 8, device="cuda")
        B = torch.randn(256, 16, device="cuda")
        assert torch.allclose(A @ B, A.to_dense() @ B, atol=1e-4)

    def test_module_cuda_moves_compute(self):
        m = _Holder().cuda()
        x = torch.randn(9, 2, device="cuda")
        assert m(x).device.type == "cuda"
        assert m.A.to_dense().device.type == "cuda"
        ref = RCSMatrix(m.G.detach(), m.A.start_cols, 9).to_dense()
        assert torch.allclose(m.A.to_dense().detach(), ref)

    def test_cuda_gradcheck(self):
        values = torch.randn(
            4, 3, dtype=torch.float64, device="cuda", requires_grad=True
        )
        start_cols = torch.randint(0, 6, (4,), device="cuda")
        B = torch.randn(8, 2, dtype=torch.float64, device="cuda", requires_grad=True)
        # nondet_tol: the triton grad_B kernel accumulates with float atomics
        assert torch.autograd.gradcheck(
            lambda g, b: RCSMatrix(g, start_cols, 8) @ b, (values, B), nondet_tol=1e-9
        )

    def test_cuda_gradgradcheck(self):
        values = torch.randn(
            4, 3, dtype=torch.float64, device="cuda", requires_grad=True
        )
        start_cols = torch.randint(0, 6, (4,), device="cuda")
        B = torch.randn(8, 2, dtype=torch.float64, device="cuda", requires_grad=True)
        assert torch.autograd.gradgradcheck(
            lambda g, b: RCSMatrix(g, start_cols, 8) @ b, (values, B), nondet_tol=1e-9
        )

    @pytest.mark.parametrize("amp_dtype", [torch.float16, torch.bfloat16])
    def test_autocast(self, amp_dtype):
        values = torch.randn(64, 8, device="cuda", requires_grad=True)
        start_cols = torch.randint(0, 249, (64,), device="cuda")
        A = RCSMatrix(values, start_cols, 256)
        B = torch.randn(256, 16, device="cuda", requires_grad=True)
        ref = A @ B
        with torch.autocast("cuda", dtype=amp_dtype):
            out = A @ B
        assert out.dtype == amp_dtype
        out.float().square().sum().backward()
        assert values.grad.dtype == torch.float32
        assert torch.isfinite(values.grad).all() and torch.isfinite(B.grad).all()
        tol = 1e-2 if amp_dtype == torch.float16 else 5e-2
        assert torch.allclose(out.float(), ref, atol=tol, rtol=tol)

    def test_no_sync_in_hot_path(self):
        values = torch.randn(64, 8, device="cuda", requires_grad=True)
        start_cols = torch.randint(0, 249, (64,), device="cuda")
        A = RCSMatrix(values, start_cols, 256)
        B = torch.randn(256, 16, device="cuda", requires_grad=True)
        # Warm up lazy device moves and triton JIT (forward + backward kernels)
        (A @ B).square().sum().backward()
        A.to_dense()
        torch.cuda.synchronize()
        torch.cuda.set_sync_debug_mode("error")
        try:
            (A @ B).square().sum().backward()
            A.to_dense()
        finally:
            torch.cuda.set_sync_debug_mode("default")

    def test_no_host_copies_in_steady_state(self):
        from torch.profiler import ProfilerActivity, profile

        values = torch.randn(64, 8, device="cuda", requires_grad=True)
        start_cols = torch.randint(0, 249, (64,), device="cuda")
        A = RCSMatrix(values, start_cols, 256)
        B = torch.randn(256, 16, device="cuda", requires_grad=True)
        for _ in range(3):
            (A @ B).square().sum().backward()
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            (A @ B).square().sum().backward()
            torch.cuda.synchronize()
        # DtoD copies (from the gather's backward) are device-local and fine;
        # host transfers in the hot path are not.
        host_copies = [
            e.key for e in prof.key_averages() if "HtoD" in e.key or "DtoH" in e.key
        ]
        assert not host_copies, host_copies

    def test_bit_deterministic_gradients(self):
        values = torch.randn(2048, 4, device="cuda", requires_grad=True)
        start_cols = torch.randint(0, 253, (2048,), device="cuda")
        A = RCSMatrix(values, start_cols, 256)
        B = torch.randn(256, 64, device="cuda", requires_grad=True)
        grads = []
        for _ in range(3):
            values.grad = B.grad = None
            (A @ B).square().sum().backward()
            grads.append((values.grad.clone(), B.grad.clone()))
        for gv, gb in grads[1:]:
            assert torch.equal(gv, grads[0][0])
            assert torch.equal(gb, grads[0][1])

    def test_training_step_decreases_loss(self):
        G = nn.Parameter(torch.randn(64, 8, device="cuda"))
        start_cols = torch.randint(0, 249, (64,), device="cuda")
        opt = torch.optim.Adam([G], lr=1e-2)
        X = torch.randn(256, 16, device="cuda")
        Y = torch.randn(64, 16, device="cuda")
        losses = []
        for _ in range(50):
            opt.zero_grad()
            loss = (RCSMatrix(G, start_cols, 256) @ X - Y).square().mean()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        assert losses[-1] < 0.5 * losses[0]

    def test_constructor_single_sync(self):
        import warnings

        torch.cuda.set_sync_debug_mode("warn")
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                RCSMatrix(
                    torch.randn(64, 4, device="cuda"),
                    torch.randint(0, 90, (64,), device="cuda"),
                    100,
                )
        finally:
            torch.cuda.set_sync_debug_mode("default")
        syncs = sum("synchronizing" in str(w.message) for w in caught)
        assert syncs == 1
