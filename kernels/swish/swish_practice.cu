#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <limits>
#include <torch/extension.h>
#include <torch/types.h>

#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])


__device__ __forceinline__ float swish(float x)
{
  return x / (1.0f + expf(-x));
}
// TODO 1（FP32 标量）: 一个线程处理一个 FP32 元素。
// 要求：计算全局索引、检查边界，并融合完成 sigmoid 与输入乘法。
// FP32
// Swish x: N, y: N, y[i] = x[i] * sigmoid(x[i])
//                         = x[i] / (1 + exp(-x[i]))
// grid(ceil(N/256)), block(256)
__global__ void swish_f32_kernel(float *x, float *y, int N) {
  // TODO: 由你实现。
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < N)
  {
    y[idx] = swish(x[idx]);
  }
}

// TODO 2（FP32x4）: 使用 float4 让一个线程处理四个 FP32 元素。
// 要求：实现向量化主路径，并安全处理不足四个元素的尾部。
// FP32x4
// Swish x: N, y: N, y[i] = x[i] * sigmoid(x[i])
//                         = x[i] / (1 + exp(-x[i]))
// grid(ceil(N/256)), block(256/4)
__global__ void swish_f32x4_kernel(float *x, float *y, int N) {
  // TODO: 由你实现。
  int idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
  if(idx + 3 < N)
  {
    float4 reg_x = FLOAT4(x[idx]);
    float4 reg_y;

    reg_y.x = swish(reg_x.x);
    reg_y.y = swish(reg_x.y);
    reg_y.z = swish(reg_x.z);
    reg_y.w = swish(reg_x.w);

    FLOAT4(y[idx]) = reg_y;
  }
  else if(idx < N)
  {
    for(int i=0;i+idx < N;i++)
    {
      y[i+idx] = swish(x[i+idx]);
    }
  }
}

__device__ __forceinline__ half swish_half(half x)
{
  half one = __float2half(1.0f);
  return x / __hadd(one, hexp(-x));
}

// TODO 3（FP16 标量）: 一个线程处理一个 FP16 元素。
// 要求：检查边界，并使用 CUDA FP16 指数、乘除和加法运算。
// FP16
// Swish x: N, y: N, y[i] = x[i] * sigmoid(x[i])
//                         = x[i] / (1 + exp(-x[i]))
// grid(ceil(N/256)), block(256)
__global__ void swish_f16_kernel(half *x, half *y, int N) {
  // TODO: 由你实现。
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < N)
  {
    // half reg_x = x[idx];
    // half one = __float2half(1.0f);
    // y[idx] = reg_x / __hadd(one, hexp(-reg_x));
    y[idx] = swish_half(x[idx]);
  }
}

// TODO 4（FP16x2）: 使用 half2 让一个线程处理两个 FP16 元素。
// 要求：实现完整双元素主路径，并安全处理最后一个落单元素。
// FP16x2
// Swish x: N, y: N, y[i] = x[i] * sigmoid(x[i])
//                         = x[i] / (1 + exp(-x[i]))
// grid(ceil(N/256)), block(256/2)
__global__ void swish_f16x2_kernel(half *x, half *y, int N) {
  // TODO: 由你实现。
  int idx = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
  if(idx + 1 < N)
  {
    half2 reg_x = HALF2(x[idx]);
    half2 reg_y;

    reg_y.x = swish_half(reg_x.x);
    reg_y.y = swish_half(reg_x.y);

    HALF2(y[idx]) = reg_y;
  }
  else if(idx < N)
  {
    for(int i=0; i+idx < N;i++)
    {
      y[i+idx] = swish_half(x[i+idx]);
    }
  }
}

// TODO 5（FP16x8）: 使用四组 half2 让一个线程处理八个 FP16 元素。
// 要求：实现完整八元素主路径，并安全处理不足八个元素的尾部。
// FP16x8
// Swish x: N, y: N, y[i] = x[i] * sigmoid(x[i])
//                         = x[i] / (1 + exp(-x[i]))
// grid(ceil(N/256)), block(256/8)
__global__ void swish_f16x8_kernel(half *x, half *y, int N) {
  // TODO: 由你实现。
  int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
  if(idx + 7 < N)
  {
    half2 reg_x;
    half2 reg_y;
    for(int i=0;i<4;i++)
    {
      reg_x = HALF2(x[2*i+idx]);

      reg_y.x = swish_half(reg_x.x);
      reg_y.y = swish_half(reg_x.y);
      HALF2(y[2*i+idx]) = reg_y;
    }
  }
  else if(idx < N)
  {
    for(int i=0;i+idx < N; i++)
    {
      y[i+idx] = swish_half(x[i+idx]);
    }
  }
}

// TODO 6（128-bit pack）: 一次加载/存储八个 FP16 元素。
// 要求：使用 LDST128BITS 实现主路径，并安全处理尾部元素。
// FP16x8 pack
// Swish x: N, y: N, y[i] = x[i] * sigmoid(x[i])
//                         = x[i] / (1 + exp(-x[i]))
// grid(ceil(N/256)), block(256/8)
__global__ void swish_f16x8_pack_kernel(half *x, half *y, int N) {
  // TODO: 由你实现。
  int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
  if(idx + 7 < N)
  {
    half2 ldst_x[4];
    half2 ldst_y[4];
    LDST128BITS(ldst_x[0]) = LDST128BITS(x[idx]);

    half2 reg_x;
    half2 reg_y;
    for(int i=0;i<4;i++)
    {
      reg_x = HALF2(ldst_x[i]);
      // reg_y.x = swish_half(reg_x.x);
      // reg_y.y = swish_half(reg_x.y);
      const half2 one = __float2half2_rn(1.0f);
      reg_y = reg_x / __hadd2(one, h2exp(-reg_x));
      ldst_y[i] = reg_y;
    }
    LDST128BITS(y[idx]) = LDST128BITS(ldst_y[0]);
  }
  else if(idx < N)
  {
    for(int i=0; i+idx < N; i++)
    {
      y[i+idx] = swish_half(x[i+idx]);
    }
  }
}

#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func)                                   \
  m.def(STRINGFY(func), &func, STRINGFY(func));

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  TORCH_CHECK((T).scalar_type() == (th_type), #T " has an unexpected dtype")

// 将连续 Tensor 展平成一维，并让每个 block 覆盖 256 个标量元素。
// 对向量宽度 W，block 使用 256/W 个线程，尾部路径因此可以实际到达。
#define TORCH_BINDING_SWISH(packed_type, th_type, element_type, n_elements)    \
  void swish_##packed_type(torch::Tensor x, torch::Tensor y) {                 \
    CHECK_TORCH_TENSOR_DTYPE(x, (th_type));                                    \
    CHECK_TORCH_TENSOR_DTYPE(y, (th_type));                                    \
    TORCH_CHECK(x.is_cuda() && y.is_cuda(), "x and y must be CUDA tensors");   \
    TORCH_CHECK(x.is_contiguous() && y.is_contiguous(),                         \
                "x and y must be contiguous");                                \
    TORCH_CHECK(x.numel() == y.numel(), "x and y must have equal numel");      \
    TORCH_CHECK(x.numel() <= std::numeric_limits<int>::max(),                   \
                "tensor is too large for this exercise");                     \
    const int N = static_cast<int>(x.numel());                                  \
    if (N == 0)                                                                \
      return;                                                                  \
    dim3 block(256 / (n_elements));                                             \
    dim3 grid((N + 256 - 1) / 256);                                             \
    swish_##packed_type##_kernel<<<grid, block>>>(                              \
        reinterpret_cast<element_type *>(x.data_ptr()),                         \
        reinterpret_cast<element_type *>(y.data_ptr()), N);                    \
  }

TORCH_BINDING_SWISH(f32, torch::kFloat32, float, 1)
TORCH_BINDING_SWISH(f32x4, torch::kFloat32, float, 4)
TORCH_BINDING_SWISH(f16, torch::kHalf, half, 1)
TORCH_BINDING_SWISH(f16x2, torch::kHalf, half, 2)
TORCH_BINDING_SWISH(f16x8, torch::kHalf, half, 8)
TORCH_BINDING_SWISH(f16x8_pack, torch::kHalf, half, 8)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  TORCH_BINDING_COMMON_EXTENSION(swish_f32)
  TORCH_BINDING_COMMON_EXTENSION(swish_f32x4)
  TORCH_BINDING_COMMON_EXTENSION(swish_f16)
  TORCH_BINDING_COMMON_EXTENSION(swish_f16x2)
  TORCH_BINDING_COMMON_EXTENSION(swish_f16x8)
  TORCH_BINDING_COMMON_EXTENSION(swish_f16x8_pack)
}
