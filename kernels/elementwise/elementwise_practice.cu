#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <vector>

#define WARP_SIZE 32
#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])

// FP32
// ElementWise Add grid(N/256),
// block(256) a: Nx1, b: Nx1, c: Nx1, c = elementwise_add(a, b)
__global__ void elementwise_add_f32_kernel(float *a, float *b, float *c,
                                           int N) {
  // TODO 1（入门）: 实现一个线程处理一个 FP32 元素的逐元素加法。
  // 要求：计算全局线程索引，并进行 idx < N 的边界检查。

  // blockIdx.x   // 当前是第几个 block
  // blockDim.x   // 每个 block 有多少个线程
  // threadIdx.x  // 当前线程在 block 内的编号
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < N)
  {
    c[idx] = a[idx] + b[idx];
  }
}

// ElementWise Add + Vec4
// grid(N/256), block(256/4)
// a: Nx1, b: Nx1, c: Nx1, c = elementwise_add(a, b)
__global__ void elementwise_add_f32x4_kernel(float *a, float *b, float *c,
                                             int N) {
  // TODO 2（向量化）: 使用 float4 让一个线程处理四个 FP32 元素。
  // 要求：实现向量化主路径，并正确处理不足四个元素的尾部。
  int idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
  if(idx + 3 < N)
  {
    float4 reg_a = FLOAT4(a[idx]);
    float4 reg_b = FLOAT4(b[idx]);

    float4 reg_c;

    reg_c.x = reg_a.x + reg_b.x;
    reg_c.y = reg_a.y + reg_b.y;
    reg_c.z = reg_a.z + reg_b.z;
    reg_c.w = reg_a.w + reg_b.w;

    FLOAT4(c[idx]) = reg_c;

  }
  // 处理idx>=N-3 && idx<N的情况
  else if(idx < N)
  {
    for(int i=0; i + idx < N; i++)
    {
      c[idx + i] = a[idx + i] + b[idx + i];
    }
  }

}

// FP16
// ElementWise Add grid(N/256),
// block(256) a: Nx1, b: Nx1, c: Nx1, c = elementwise_add(a, b)
__global__ void elementwise_add_f16_kernel(half *a, half *b, half *c, int N) {
  // TODO 3（FP16）: 实现一个线程处理一个 FP16 元素的逐元素加法。
  // 要求：进行边界检查，并使用 CUDA FP16 加法 intrinsic。
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < N)
  {
    c[idx] = __hadd(a[idx], b[idx]);
  }
}

// a: Nx1, b: Nx1, c: Nx1, c = elementwise_add(a, b)
__global__ void elementwise_add_f16x2_kernel(half *a, half *b, half *c, int N) {
  // TODO 4（half2）: 使用 half2 让一个线程处理两个 FP16 元素。
  // 要求：实现双元素主路径，并正确处理最后一个落单元素。
  int idx = 2 * (blockIdx.x * blockDim.x + threadIdx.x);

  if(idx + 1 < N)
  {
    half2 reg_a = HALF2(a[idx]);
    half2 reg_b = HALF2(b[idx]);
    half2 reg_c;

    // reg_c.x = __hadd(reg_a.x, reg_b.x);
    // reg_c.y = __hadd(reg_a.y, reg_b.y);
    reg_c = __hadd2(reg_a, reg_b);

    HALF2(c[idx]) = reg_c;
  }
  else if(idx < N)
  {
    for(int i=0;idx + i < N;i++)
    {
      c[i+idx] = __hadd(a[i+idx], b[i+idx]);
    }
  }

}

__global__ void elementwise_add_f16x8_kernel(half *a, half *b, half *c, int N) {
  // TODO 5（8 个 FP16）: 让一个线程处理八个 FP16 元素。
  // 要求：使用四组 half2 完成主路径，并正确处理尾部元素。

  int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
  if(idx + 7 < N)
  {
    half2 reg_a;
    half2 reg_b;
    half2 reg_c;
    for(int i = 0;i < 8;i += 2)
    {
      reg_a = HALF2(a[i+idx]);
      reg_b = HALF2(b[i+idx]);
      reg_c = __hadd2(reg_a, reg_b);
      HALF2(c[i+idx]) = reg_c;
    }
  }
  else if(idx < N)
  {
    for(int i=0;i+idx < N;i++)
    {
      c[i+idx] = a[i+idx] + b[i+idx];
    }
  }
}

__global__ void elementwise_add_f16x8_pack_kernel(half *a, half *b, half *c,
                                                  int N) {
  // TODO 6（128-bit pack）: 一次加载/存储八个 FP16 元素。
  // 要求：使用 LDST128BITS 和 __hadd2 实现主路径，并处理尾部元素。
  int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);

  if(idx + 7 < N)
  {
    // float4 16字节，用来存8个FP16元素
    float4 reg_a = LDST128BITS(a[idx]);
    float4 reg_b = LDST128BITS(b[idx]);

    float4 reg_c;

    HALF2(reg_c.x) = __hadd2(HALF2(reg_a.x), HALF2(reg_b.x));
    HALF2(reg_c.y) = __hadd2(HALF2(reg_a.y), HALF2(reg_b.y));
    HALF2(reg_c.z) = __hadd2(HALF2(reg_a.z), HALF2(reg_b.z));
    HALF2(reg_c.w) = __hadd2(HALF2(reg_a.w), HALF2(reg_b.w));

    LDST128BITS(c[idx]) = reg_c;
  }
  else if(idx < N)
  {
    for(int i=0;i+idx < N;i++)
    {
      c[i+idx] = __hadd(a[i+idx], b[i+idx]);
    }
  }

}

#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func)                                   \
  m.def(STRINGFY(func), &func, STRINGFY(func));

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if (((T).options().dtype() != (th_type))) {                                  \
    std::cout << "Tensor Info:" << (T).options() << std::endl;                 \
    throw std::runtime_error("values must be " #th_type);                      \
  }

#define TORCH_BINDING_ELEM_ADD(packed_type, th_type, element_type, n_elements) \
  void elementwise_add_##packed_type(torch::Tensor a, torch::Tensor b,         \
                                     torch::Tensor c) {                        \
    CHECK_TORCH_TENSOR_DTYPE(a, (th_type))                                     \
    CHECK_TORCH_TENSOR_DTYPE(b, (th_type))                                     \
    CHECK_TORCH_TENSOR_DTYPE(c, (th_type))                                     \
    const int ndim = a.dim();                                                  \
    if (ndim != 2) {                                                           \
      int N = 1;                                                               \
      for (int i = 0; i < ndim; ++i) {                                         \
        N *= a.size(i);                                                        \
      }                                                                        \
      dim3 block(256 / (n_elements));                                          \
      dim3 grid((N + 256 - 1) / 256);                                          \
      elementwise_add_##packed_type##_kernel<<<grid, block>>>(                 \
          reinterpret_cast<element_type *>(a.data_ptr()),                      \
          reinterpret_cast<element_type *>(b.data_ptr()),                      \
          reinterpret_cast<element_type *>(c.data_ptr()), N);                  \
    } else {                                                                   \
      const int S = a.size(0);                                                 \
      const int K = a.size(1);                                                 \
      const int N = S * K;                                                     \
      if ((K / (n_elements)) <= 1024) {                                        \
        dim3 block(K / (n_elements));                                          \
        dim3 grid(S);                                                          \
        elementwise_add_##packed_type##_kernel<<<grid, block>>>(               \
            reinterpret_cast<element_type *>(a.data_ptr()),                    \
            reinterpret_cast<element_type *>(b.data_ptr()),                    \
            reinterpret_cast<element_type *>(c.data_ptr()), N);                \
      } else {                                                                 \
        int N = 1;                                                             \
        for (int i = 0; i < ndim; ++i) {                                       \
          N *= a.size(i);                                                      \
        }                                                                      \
        dim3 block(256 / (n_elements));                                        \
        dim3 grid((N + 256 - 1) / 256);                                        \
        elementwise_add_##packed_type##_kernel<<<grid, block>>>(               \
            reinterpret_cast<element_type *>(a.data_ptr()),                    \
            reinterpret_cast<element_type *>(b.data_ptr()),                    \
            reinterpret_cast<element_type *>(c.data_ptr()), N);                \
      }                                                                        \
    }                                                                          \
  }

TORCH_BINDING_ELEM_ADD(f32, torch::kFloat32, float, 1)
TORCH_BINDING_ELEM_ADD(f32x4, torch::kFloat32, float, 4)
TORCH_BINDING_ELEM_ADD(f16, torch::kHalf, half, 1)
TORCH_BINDING_ELEM_ADD(f16x2, torch::kHalf, half, 2)
TORCH_BINDING_ELEM_ADD(f16x8, torch::kHalf, half, 8)
TORCH_BINDING_ELEM_ADD(f16x8_pack, torch::kHalf, half, 8)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  TORCH_BINDING_COMMON_EXTENSION(elementwise_add_f32)
  TORCH_BINDING_COMMON_EXTENSION(elementwise_add_f32x4)
  TORCH_BINDING_COMMON_EXTENSION(elementwise_add_f16)
  TORCH_BINDING_COMMON_EXTENSION(elementwise_add_f16x2)
  TORCH_BINDING_COMMON_EXTENSION(elementwise_add_f16x8)
  TORCH_BINDING_COMMON_EXTENSION(elementwise_add_f16x8_pack)
}
