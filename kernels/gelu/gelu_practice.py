import os
import time
from typing import Optional

# 这些默认值仅作用于当前练习进程，不修改系统 CUDA、GCC 或 Conda 配置。
os.environ.setdefault("CC", "/usr/bin/gcc-9")
os.environ.setdefault("CXX", "/usr/bin/g++-9")
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.6")

import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

torch.set_grad_enabled(False)

lib = load(
    name="gelu_practice_lib",
    sources=["gelu_practice.cu"],
    extra_cuda_cflags=[
        "-O3",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
    ],
    extra_cflags=["-std=c++17"],
)


def torch_gelu_tanh(x: torch.Tensor) -> torch.Tensor:
    return F.gelu(x, approximate="tanh")


def assert_gelu(perf_func: callable, x: torch.Tensor, tag: str) -> torch.Tensor:
    out = torch.full_like(x, torch.nan)
    perf_func(x, out)
    torch.cuda.synchronize()
    expected = torch_gelu_tanh(x)
    try:
        if x.dtype == torch.float16:
            # 练习沿用纯 FP16 的 exp 形式近似 tanh；PyTorch 通常使用
            # 不同的内部精度路径，因此允许约两个 FP16 ULP 的误差。
            torch.testing.assert_close(out, expected, rtol=3e-3, atol=2e-3)
        else:
            torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-6)
    except AssertionError as error:
        raise AssertionError(
            f"{tag} correctness check failed; complete or inspect its TODO in "
            "gelu_practice.cu"
        ) from error
    return out


def make_test_input(n: int, dtype: torch.dtype) -> torch.Tensor:
    # 纯 FP16 通过 exp 重写 tanh；使用 [-3, 3] 覆盖 GELU 的主要非线性区间，
    # 同时避免把 exp 溢出误认为向量化或尾部错误。
    return torch.linspace(-3.0, 3.0, n, device="cuda", dtype=dtype)


def run_correctness_tests() -> None:
    implementations = [
        (1, lib.gelu_f32, "f32", torch.float32),
        (2, lib.gelu_f32x4, "f32x4", torch.float32),
        (3, lib.gelu_f16, "f16", torch.float16),
        (4, lib.gelu_f16x2, "f16x2", torch.float16),
        (5, lib.gelu_f16x8, "f16x8", torch.float16),
        (6, lib.gelu_f16x8_pack, "f16x8pack", torch.float16),
    ]
    sizes = [1, 2, 3, 4, 5, 7, 8, 9, 17, 257]
    sample_values = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]

    # 按 TODO 优先、尺寸次序验证；每项全部通过后立即打印 PASS。
    for todo, func, tag, dtype in implementations:
        for n in sizes:
            x = make_test_input(n, dtype)
            assert_gelu(func, x, f"TODO {todo} ({tag}), N={n}")

        sample = torch.tensor(sample_values, device="cuda", dtype=dtype)
        sample_out = assert_gelu(func, sample, f"TODO {todo} ({tag}), sample")
        print(
            f"PASS TODO {todo}: {tag:<9} "
            f"sizes={sizes}, sample_out={sample_out.cpu().tolist()}",
            flush=True,
        )

    print("All correctness tests: PASS", flush=True)


def run_benchmark(
    perf_func: callable,
    x: torch.Tensor,
    tag: str,
    out: Optional[torch.Tensor] = None,
    warmup: int = 10,
    iters: int = 1000,
    show_all: bool = False,
):
    uses_output_buffer = out is not None
    if uses_output_buffer:
        out.fill_(0)

    for _ in range(warmup):
        if uses_output_buffer:
            perf_func(x, out)
        else:
            out = perf_func(x)

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(iters):
        if uses_output_buffer:
            perf_func(x, out)
        else:
            out = perf_func(x)
    torch.cuda.synchronize()
    mean_time = (time.time() - start) * 1000 / iters

    expected = torch_gelu_tanh(x)
    if x.dtype == torch.float16:
        torch.testing.assert_close(out, expected, rtol=3e-3, atol=2e-3)
    else:
        torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-6)

    out_val = [round(v, 8) for v in out.flatten().detach().cpu().tolist()[:2]]
    print(f"{'out_' + tag:>18}: {out_val}, time:{mean_time:.8f}ms")
    if show_all:
        print(out)
    return out, mean_time


def run_benchmarks() -> None:
    for S, K in [(S, K) for S in [1024, 2048, 4096] for K in [1024, 2048, 4096]]:
        print("-" * 85)
        print(" " * 40 + f"S={S}, K={K}")
        # 限制 benchmark 输入范围，以匹配纯 FP16 exp-tanh 近似的有效区间。
        x = torch.randn((S, K), device="cuda", dtype=torch.float32).clamp_(-3, 3)
        x = x.contiguous()
        y = torch.zeros_like(x)
        run_benchmark(lib.gelu_f32, x, "f32", y)
        run_benchmark(lib.gelu_f32x4, x, "f32x4", y)
        run_benchmark(torch_gelu_tanh, x, "f32_th")

        print("-" * 85)
        x_f16 = x.half().contiguous()
        y_f16 = torch.zeros_like(x_f16)
        run_benchmark(lib.gelu_f16, x_f16, "f16", y_f16)
        run_benchmark(lib.gelu_f16x2, x_f16, "f16x2", y_f16)
        run_benchmark(lib.gelu_f16x8, x_f16, "f16x8", y_f16)
        run_benchmark(lib.gelu_f16x8_pack, x_f16, "f16x8pack", y_f16)
        run_benchmark(torch_gelu_tanh, x_f16, "f16_th")
        print("-" * 85)


if __name__ == "__main__":
    run_correctness_tests()
    run_benchmarks()
