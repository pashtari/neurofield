"""Benchmark RCSMatrix @ B against naive dense matmul.

Covers CPU and GPU, inference and forward+backward. Two methodology details
matter for trustworthy numbers:

* Cache flushing. When the ``(M, K)`` output fits in last-level cache (32 MB
  L2 on an RTX 4060 Ti), a warm benchmark loop reuses the same buffer and the
  stores never reach DRAM, inflating the memory-bound RCS side by ~2x. Every
  timed call is preceded by a cache flush, excluded from the measurement.
  Pass ``--warm`` to skip the flush and see the cache-resident numbers.
* Gradients via ``autograd.grad``. A ``.square().sum().backward()`` epilogue
  adds full passes over the output that both sides pay, diluting the measured
  ratio (2.6x where the matmul alone gives 5.4x). Gradients are taken with
  preallocated ``grad_outputs`` so only the matmul is timed.

The ``ceiling`` column is the roofline limit ``N * BW / (2 * F)`` for a
memory-bound RCS product against a compute-bound dense GEMM, from the
bandwidth and FLOP rate calibrated at startup. Being at the ceiling means the
kernel cannot be improved; only moving fewer bytes (fusing away the output)
or raising ``N`` helps.
"""

import argparse
import time

import torch

from neurofield.models import RCSMatrix

SHAPES = [
    # (M, N, L, K)
    (300_000, 512, 8, 512),
    (200_000, 256, 8, 256),
]

_FLUSH = {}


def _sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def _flush_cache(device):
    """Evict the timed data from last-level cache."""
    buf = _FLUSH.get(device)
    if buf is None:
        mb = 64 if device == "cuda" else 128
        buf = torch.empty(mb * 1024 * 1024 // 4, device=device)
        _FLUSH[device] = buf
    buf.fill_(1.0)


def bench(fn, device, warm=False, target_s=0.3, max_iters=100):
    """Median-of-3 timing. Unless ``warm``, flushes cache before each call."""
    fn()  # warmup / lazy compiles
    _sync(device)
    t0 = time.perf_counter()
    fn()
    _sync(device)
    once = time.perf_counter() - t0
    iters = max(3, min(max_iters, int(target_s / max(once, 1e-9))))

    runs = []
    for _ in range(3):
        if warm:
            for _ in range(2):
                fn()
            _sync(device)
            t0 = time.perf_counter()
            for _ in range(iters):
                fn()
            _sync(device)
            runs.append((time.perf_counter() - t0) / iters)
        elif device == "cuda":
            # CUDA events time the call alone, excluding the flush.
            total = 0.0
            for _ in range(iters):
                _flush_cache(device)
                start, end = torch.cuda.Event(True), torch.cuda.Event(True)
                start.record()
                fn()
                end.record()
                torch.cuda.synchronize()
                total += start.elapsed_time(end) / 1e3
            runs.append(total / iters)
        else:
            total = 0.0
            for _ in range(iters):
                _flush_cache(device)
                t0 = time.perf_counter()
                fn()
                total += time.perf_counter() - t0
            runs.append(total / iters)
    return sorted(runs)[1] * 1e3  # ms


def calibrate(device):
    """Measure copy bandwidth (GB/s) and dense GEMM throughput (TFLOP/s).

    The copy buffer must exceed last-level cache (32 MB L2 on the GPU, up to
    64+ MB L3 on the CPU) or this measures cache bandwidth, not DRAM.
    """
    n = (64 if device == "cuda" else 512) * 1024 * 1024 // 4
    src = torch.randn(n, device=device)
    dst = torch.empty_like(src)
    t = bench(lambda: dst.copy_(src), device, warm=True, target_s=0.1)
    bw = 2 * n * 4 / (t / 1e3) / 1e9

    size = 8192 if device == "cuda" else 2048
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)
    t = bench(lambda: a @ b, device, warm=True, target_s=0.1)
    flops = 2 * size**3 / (t / 1e3) / 1e12
    del src, dst, a, b
    if device == "cuda":
        torch.cuda.empty_cache()
    return bw, flops


def run(device, warm=False):
    name = f" ({torch.cuda.get_device_name()})" if device == "cuda" else ""
    print(f"\n{'=' * 92}\ndevice: {device}{name}\n{'=' * 92}")
    bw, flops = calibrate(device)
    tf32 = torch.backends.cuda.matmul.allow_tf32 if device == "cuda" else False
    print(
        f"calibration: {bw:.0f} GB/s copy bandwidth, {flops:.1f} TFLOP/s dense fp32"
        + (f" (TF32 {'on' if tf32 else 'off'})" if device == "cuda" else "")
    )
    print(
        f"timing: {'WARM (cache-resident, optimistic)' if warm else 'cold cache'}"
        f", gradients via autograd.grad"
    )

    header = (
        f"{'M':>7} {'N':>5} {'L':>2} {'K':>4} {'out MB':>7} | "
        f"{'inf RCS':>9} {'inf dense':>10} {'speedup':>8} | "
        f"{'f+b RCS':>9} {'f+b dense':>10} {'speedup':>8} | {'ceiling':>7}"
    )
    print(header)
    print("-" * len(header))

    for M, N, L, K in SHAPES:
        torch.manual_seed(0)
        try:
            values = torch.randn(M, L, device=device)
            start_cols = torch.randint(0, N - L + 1, (M,), device=device)
            A = RCSMatrix(values, start_cols, N)
            B = torch.randn(N, K, device=device)
            D = A.to_dense()

            with torch.no_grad():
                err = (A @ B - D @ B).abs().max().item()
                assert err < 1e-3, f"mismatch: {err}"
                t_inf_rcs = bench(lambda: A @ B, device, warm)
                t_inf_dense = bench(lambda: D @ B, device, warm)

            values.requires_grad_(True)
            B.requires_grad_(True)
            Dg = D.detach().clone().requires_grad_(True)
            grad_out = torch.ones(M, K, device=device)

            t_fb_rcs = bench(
                lambda: torch.autograd.grad(A @ B, [values, B], grad_out), device, warm
            )
            t_fb_dense = bench(
                lambda: torch.autograd.grad(Dg @ B, [Dg, B], grad_out), device, warm
            )

            ceiling = N * (bw * 1e9) / (2 * flops * 1e12)
            print(
                f"{M:>7} {N:>5} {L:>2} {K:>4} {M * K * 4 / 1e6:>7.0f} | "
                f"{t_inf_rcs:>7.3f}ms {t_inf_dense:>8.3f}ms {t_inf_dense / t_inf_rcs:>7.1f}x | "
                f"{t_fb_rcs:>7.3f}ms {t_fb_dense:>8.3f}ms {t_fb_dense / t_fb_rcs:>7.1f}x | "
                f"{ceiling:>6.1f}x"
            )
            del values, start_cols, A, B, D, Dg, grad_out
        except (RuntimeError, torch.OutOfMemoryError) as e:
            print(f"{M:>7} {N:>5} {L:>2} {K:>4} {'':>7} | skipped: {str(e)[:50]}")
        if device == "cuda":
            torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--warm",
        action="store_true",
        help="skip cache flushing (optimistic, cache-resident numbers)",
    )
    parser.add_argument("--device", choices=["cpu", "cuda", "both"], default="both")
    args = parser.parse_args()

    if args.device in ("cpu", "both"):
        run("cpu", args.warm)
    if args.device in ("cuda", "both") and torch.cuda.is_available():
        run("cuda", args.warm)


if __name__ == "__main__":
    main()
