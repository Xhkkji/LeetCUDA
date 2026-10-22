#include <algorithm>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <limits>
#include <torch/extension.h>
#include <torch/types.h>

#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])

// TODO 1（FP32 标量）: 一个线程处理一个 FP32 元素。
// 要求：计算全局索引、检查边界，并实现 y=max(0,x)。
__global__ void relu_f32_kernel(float *x, float *y, int N) {
  // TODO: 由你实现。
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < N)
  {
    y[idx] = fmaxf(0.0f, x[idx]);
  }
}

// TODO 2（FP32x4）: 使用 float4 让一个线程处理四个 FP32 元素。
// 要求：实现向量化主路径，并安全处理不足四个元素的尾部。
__global__ void relu_f32x4_kernel(float *x, float *y, int N) {
  // TODO: 由你实现。
  int idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
  if(idx + 3 < N)
  {
    float4 reg_x = FLOAT4(x[idx]);
    float4 reg_y;

    reg_y.x = fmaxf(0.0f, reg_x.x);
    reg_y.y = fmaxf(0.0f, reg_x.y);
    reg_y.z = fmaxf(0.0f, reg_x.z);
    reg_y.w = fmaxf(0.0f, reg_x.w);

    FLOAT4(y[idx]) = reg_y;
  }
  else if(idx < N)
  {
    for(int i=0;i+idx < N;i++)
    {
      y[i+idx] = fmaxf(0.0f, x[i+idx]);
    }
  }
}

// TODO 3（FP16 标量）: 一个线程处理一个 FP16 元素。
// 要求：检查边界，并使用 CUDA FP16 max intrinsic。
__global__ void relu_f16_kernel(half *x, half *y, int N) {
  // TODO: 由你实现。
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < N)
  {
    y[idx] = __hmax(half(0.0f), x[idx]);
  }
}

// TODO 4（FP16x2）: 使用 half2 让一个线程处理两个 FP16 元素。
// 要求：实现完整双元素主路径，并安全处理最后一个落单元素。
__global__ void relu_f16x2_kernel(half *x, half *y, int N) {
  // TODO: 由你实现。
  int idx = 2 * (blockIdx.x * blockDim.x + threadIdx.x);

  if(idx + 1 < N)
  {
    half2 reg_x = HALF2(x[idx]);
    half2 reg_y;

    const half2 zero = __float2half2_rn(0.0f);
    reg_y = __hmax2(zero, reg_x);
    HALF2(y[idx]) = reg_y;
  }
  else if(idx < N)
  {
    for(int i=0; i+idx < N;i++)
    {
      y[i+idx] = __hmax(half(0.0f), x[i+idx]);
    }
  }
}

// TODO 5（FP16x8）: 使用四组 half2 让一个线程处理八个 FP16 元素。
// 要求：实现完整八元素主路径，并安全处理不足八个元素的尾部。
__global__ void relu_f16x8_kernel(half *x, half *y, int N) {
  // TODO: 由你实现。
  int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
  if(idx + 7 < N)
  {
    for(int i=0;i<7;i+=2)
    {
      half2 reg_x = HALF2(x[i+idx]);
      half2 reg_y;

      const half2 zero = __float2half2_rn(0.0f);
      reg_y = __hmax2(zero, reg_x);
      HALF2(y[i+idx]) = reg_y;
    }
  }
  else if(idx < N)
  {
    for(int i=0;i+idx < N;i++)
    {
      y[i+idx] = __hmax(half(0.0f), x[i+idx]);
    }
  }
}

// TODO 6（128-bit pack）: 一次加载/存储八个 FP16 元素。
// 要求：使用 LDST128BITS 和 FP16x2 max intrinsic 实现主路径，并处理尾部。
__global__ void relu_f16x8_pack_kernel(half *x, half *y, int N) {
  // TODO: 由你实现。
  int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
  if(idx + 7 < N)
  {
    float4 reg_x = FLOAT4(x[idx]);
    float4 reg_y;

    const half2 zero = __float2half2_rn(0.0f);
    HALF2(reg_y.x) = __hmax2(zero, HALF2(reg_x.x));
    HALF2(reg_y.y) = __hmax2(zero, HALF2(reg_x.y));
    HALF2(reg_y.z) = __hmax2(zero, HALF2(reg_x.z));
    HALF2(reg_y.w) = __hmax2(zero, HALF2(reg_x.w));

    FLOAT4(y[idx]) = reg_y;
  }
  else if(idx < N)
  {
    for(int i=0;i+idx < N;i++)
    {
      y[i+idx] = __hmax(0.0f, x[i+idx]);
    }
  }
}

#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func)                                   \
  m.def(STRINGFY(func), &func, STRINGFY(func));

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  TORCH_CHECK((T).scalar_type() == (th_type), #T " has an unexpected dtype")

// 练习脚手架统一把 Tensor 展平成一维，并按每个 block 覆盖 256 个标量元素
// 启动。这样一维/二维和不能被 2、4、8 整除的长度都会实际进入尾部路径。
#define TORCH_BINDING_RELU(packed_type, th_type, element_type, n_elements)     \
  void relu_##packed_type(torch::Tensor x, torch::Tensor y) {                  \
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
    relu_##packed_type##_kernel<<<grid, block>>>(                               \
        reinterpret_cast<element_type *>(x.data_ptr()),                         \
        reinterpret_cast<element_type *>(y.data_ptr()), N);                    \
  }

TORCH_BINDING_RELU(f32, torch::kFloat32, float, 1)
TORCH_BINDING_RELU(f32x4, torch::kFloat32, float, 4)
TORCH_BINDING_RELU(f16, torch::kHalf, half, 1)
TORCH_BINDING_RELU(f16x2, torch::kHalf, half, 2)
TORCH_BINDING_RELU(f16x8, torch::kHalf, half, 8)
TORCH_BINDING_RELU(f16x8_pack, torch::kHalf, half, 8)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  TORCH_BINDING_COMMON_EXTENSION(relu_f32)
  TORCH_BINDING_COMMON_EXTENSION(relu_f32x4)
  TORCH_BINDING_COMMON_EXTENSION(relu_f16)
  TORCH_BINDING_COMMON_EXTENSION(relu_f16x2)
  TORCH_BINDING_COMMON_EXTENSION(relu_f16x8)
  TORCH_BINDING_COMMON_EXTENSION(relu_f16x8_pack)
}
