import os
import time
from typing import Optional

# 这些设置仅作用于当前练习进程，不修改系统 CUDA、GCC 或 Conda 配置。
os.environ.setdefault("CC", "/usr/bin/gcc-9")
os.environ.setdefault("CXX", "/usr/bin/g++-9")
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.6")

import torch
from torch.utils.cpp_extension import load

torch.set_grad_enabled(False)

lib = load(
    name="relu_practice_lib",
    sources=["relu_practice.cu"],
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


def assert_relu(perf_func: callable, x: torch.Tensor, tag: str) -> torch.Tensor:
    out = torch.full_like(x, torch.nan)
    perf_func(x, out)
    torch.cuda.synchronize()
    expected = torch.relu(x)
    try:
        torch.testing.assert_close(out, expected)
    except AssertionError as error:
        raise AssertionError(
            f"{tag} correctness check failed; complete or inspect its TODO in "
            "relu_practice.cu"
        ) from error
    return out


def run_correctness_tests() -> None:
    # 按 TODO 顺序验证。一个实现通过全部边界尺寸后立即打印 PASS，
    # 再进入下一个 TODO；因此后续未完成的 TODO 不会掩盖前面的完成情况。
    implementations = [
        (1, lib.relu_f32, "f32", torch.float32),
        (2, lib.relu_f32x4, "f32x4", torch.float32),
        (3, lib.relu_f16, "f16", torch.float16),
        (4, lib.relu_f16x2, "f16x2", torch.float16),
        (5, lib.relu_f16x8, "f16x8", torch.float16),
        (6, lib.relu_f16x8_pack, "f16x8pack", torch.float16),
    ]
    sizes = [1, 2, 3, 4, 5, 7, 8, 9, 17, 257]
    sample_values = [-2.0, -0.5, 0.0, 0.5, 2.0]

    # 同时覆盖完整向量组和 2/4/8 元素向量化的尾部路径。
    for todo, func, tag, dtype in implementations:
        for n in sizes:
            x = torch.randn(n, device="cuda", dtype=dtype).contiguous()
            assert_relu(func, x, f"TODO {todo} ({tag}), N={n}")

        sample = torch.tensor(sample_values, device="cuda", dtype=dtype)
        sample_out = assert_relu(func, sample, f"TODO {todo} ({tag}), sample")
        print(
            f"PASS TODO {todo}: {tag:<9} "
            f"sizes={sizes}, sample_out={sample_out.cpu().tolist()}",
            flush=True,
        )

    print("All correctness tests: PASS")


def run_benchmark(
    perf_func: callable,
    x: torch.Tensor,
    tag: str,
    out: Optional[torch.Tensor] = None,
    warmup: int = 10,
    iters: int = 1000,
    show_all: bool = False,
):
    # 自定义扩展写入预分配的 out；PyTorch 参考实现则返回新 Tensor。
    # 单独保存调用方式，避免第一次得到返回值后误把 out 当作位置参数传入。
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

    expected = torch.relu(x)
    try:
        torch.testing.assert_close(out, expected)
    except AssertionError as error:
        raise AssertionError(f"{tag} benchmark correctness check failed") from error

    out_val = [round(v, 8) for v in out.flatten().detach().cpu().tolist()[:2]]
    print(f"{'out_' + tag:>18}: {out_val}, time:{mean_time:.8f}ms")
    if show_all:
        print(out)
    return out, mean_time


def run_benchmarks() -> None:
    for S, K in [(S, K) for S in [1024, 2048, 4096] for K in [1024, 2048, 4096]]:
        print("-" * 85)
        print(" " * 40 + f"S={S}, K={K}")
        x = torch.randn((S, K), device="cuda", dtype=torch.float32).contiguous()
        y = torch.zeros_like(x)
        run_benchmark(lib.relu_f32, x, "f32", y)
        run_benchmark(lib.relu_f32x4, x, "f32x4", y)
        run_benchmark(torch.relu, x, "f32_th")

        print("-" * 85)
        x_f16 = x.half().contiguous()
        y_f16 = torch.zeros_like(x_f16)
        run_benchmark(lib.relu_f16, x_f16, "f16", y_f16)
        run_benchmark(lib.relu_f16x2, x_f16, "f16x2", y_f16)
        run_benchmark(lib.relu_f16x8, x_f16, "f16x8", y_f16)
        run_benchmark(lib.relu_f16x8_pack, x_f16, "f16x8pack", y_f16)
        run_benchmark(torch.relu, x_f16, "f16_th")
        print("-" * 85)


if __name__ == "__main__":
    run_correctness_tests()
    run_benchmarks()
