#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <limits>
#include <torch/extension.h>
#include <torch/types.h>

#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])

#define MAX_EXP_F32 88.3762626647949f
#define MIN_EXP_F32 -88.3762626647949f
#define MAX_EXP_F16 __float2half(11.089866488461016f)
#define MIN_EXP_F16 __float2half(-9.704060527839234f)

// TODO 1（FP32 标量）: 一个线程处理一个 FP32 元素。
// 要求：计算全局索引、检查边界，并实现数值安全的 sigmoid。
// FP32
// Sigmoid x: N, y: N, y[i] = 1 / (1 + exp(-x[i]))
// grid(ceil(N/256)), block(256)
__global__ void sigmoid_f32_kernel(float *x, float *y, int N) {
  // TODO: 由你实现。
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < N)
  {
    float v = x[idx];
    v = fmaxf(fminf(v, MAX_EXP_F32), MIN_EXP_F32);
    y[idx] = 1 / (1 + expf(-v));
  }
}

// TODO 2（FP32x4）: 使用 float4 让一个线程处理四个 FP32 元素。
// 要求：实现向量化主路径，并安全处理不足四个元素的尾部。
// FP32x4
// Sigmoid x: N, y: N, y[i] = 1 / (1 + exp(-x[i]))
// grid(ceil(N/256)), block(256/4)
__global__ void sigmoid_f32x4_kernel(float *x, float *y, int N) {
  // TODO: 由你实现。
  int idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
  if(idx + 3 < N)
  {
    float4 reg_x = FLOAT4(x[idx]);
    float4 reg_y;

    float v0 = fmaxf(fminf(reg_x.x, MAX_EXP_F32), MIN_EXP_F32);
    float v1 = fmaxf(fminf(reg_x.y, MAX_EXP_F32), MIN_EXP_F32);
    float v2 = fmaxf(fminf(reg_x.z, MAX_EXP_F32), MIN_EXP_F32);
    float v3 = fmaxf(fminf(reg_x.w, MAX_EXP_F32), MIN_EXP_F32);

    reg_y.x = 1 / (1 + expf(-v0));
    reg_y.y = 1 / (1 + expf(-v1));
    reg_y.z = 1 / (1 + expf(-v2));
    reg_y.w = 1 / (1 + expf(-v3));

    FLOAT4(y[idx]) = reg_y;
  }
  else if(idx < N)
  {
    for(int i=0;i+idx<N;i++)
    {
      float v = fmaxf(fminf(x[i+idx], MAX_EXP_F32), MIN_EXP_F32);
      y[i+idx] = 1 / (1 + expf(-v));
    }
  }
}

// TODO 3（FP16 标量）: 一个线程处理一个 FP16 元素。
// 要求：检查边界，限制指数输入范围，并使用 CUDA FP16 数学运算。
// FP16
// Sigmoid x: N, y: N, y[i] = 1 / (1 + exp(-x[i]))
// grid(ceil(N/256)), block(256)
__global__ void sigmoid_f16_kernel(half *x, half *y, int N) {
  // TODO: 由你实现。
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < N)
  {
    half reg_x = __hmax(__hmin(x[idx], MAX_EXP_F16), MIN_EXP_F16);
    const half one = __hmax(__hmin(half(1.0f), MAX_EXP_F16), MIN_EXP_F16);
    y[idx] = one / __hadd(one, hexp(-reg_x));
  }
}

// TODO 4（FP16x2）: 使用 half2 让一个线程处理两个 FP16 元素。
// 要求：实现完整双元素主路径，并安全处理最后一个落单元素。
// FP16x2
// Sigmoid x: N, y: N, y[i] = 1 / (1 + exp(-x[i]))
// grid(ceil(N/256)), block(256/2)
__global__ void sigmoid_f16x2_kernel(half *x, half *y, int N) {
  // TODO: 由你实现。
  int idx = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
  if(idx + 1 < N)
  {
    half2 reg_x = HALF2(x[idx]);
    half2 reg_y;

    // __halves2half2 的功能是：把两个独立的 half 标量组合成一个 half2 向量。
    const half2 min_v = __halves2half2(MIN_EXP_F16, MIN_EXP_F16);
    const half2 max_v = __halves2half2(MAX_EXP_F16, MAX_EXP_F16);

    half2 v = __hmin2(__hmax2(reg_x, min_v), max_v);
    const half2 one = __float2half2_rn(1.0f);
    reg_y = one / __hadd2(one, h2exp(-v));
    HALF2(y[idx]) = reg_y;
  }
  else if(idx < N)
  {
    for(int i=0;i+idx < N;i++)
    {
      half v = __hmin(__hmax(x[i+idx], MIN_EXP_F16), MAX_EXP_F16);
      const half one = __float2half(1.0f);
      y[i+idx] = one / __hadd(one, hexp(-v));
    }
  }
}

// TODO 5（FP16x8）: 使用四组 half2 让一个线程处理八个 FP16 元素。
// 要求：实现完整八元素主路径，并安全处理不足八个元素的尾部。
// FP16x8
// Sigmoid x: N, y: N, y[i] = 1 / (1 + exp(-x[i]))
// grid(ceil(N/256)), block(256/8)
__global__ void sigmoid_f16x8_kernel(half *x, half *y, int N) {
  // TODO: 由你实现。
  int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
  if(idx + 7 < N)
  {
    half2 reg_x;
    half2 reg_y;
    for(int i=0;i<8;i+=2)
    {
      reg_x = HALF2(x[i+idx]);

      const half2 min_v = __halves2half2(MIN_EXP_F16, MIN_EXP_F16);
      const half2 max_v = __halves2half2(MAX_EXP_F16, MAX_EXP_F16);
      half2 v = __hmin2(__hmax2(reg_x, min_v), max_v);

      const half2 one = __float2half2_rn(1.0f);
      reg_y = one / __hadd2(one, h2exp(-v));
      HALF2(y[i+idx]) = reg_y;
    }
  }
  else if(idx < N)
  {
    for(int i=0;i+idx < N;i++)
    {
      half v = x[i+idx];
      v = __hmin(__hmax(v, MIN_EXP_F16), MAX_EXP_F16);
      const half one = __float2half(1.0f);
      y[i+idx] = one / __hadd(one, hexp(-v));
    }
  }
}

// TODO 6（128-bit pack）: 一次加载/存储八个 FP16 元素。
// 要求：使用 LDST128BITS 实现主路径，并安全处理尾部元素。
// FP16x8 pack
// Sigmoid x: N, y: N, y[i] = 1 / (1 + exp(-x[i]))
// grid(ceil(N/256)), block(256/8)
__global__ void sigmoid_f16x8_pack_kernel(half *x, half *y, int N) {
  // TODO: 由你实现。
  int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
  if(idx + 7 < N)
  {
    half2 ldst_x[4];
    half2 ldst_y[4];
    LDST128BITS(ldst_x[0]) = LDST128BITS(x[idx]);

    for(int i=0;i<4;i++)
    {
      const half2 min_v = __halves2half2(MIN_EXP_F16, MIN_EXP_F16);
      const half2 max_v = __halves2half2(MAX_EXP_F16, MAX_EXP_F16);
      half2 v = __hmin2(__hmax2(ldst_x[i], min_v), max_v);

      const half2 one = __float2half2_rn(1.0f);
      ldst_y[i] = one / __hadd2(one, h2exp(-v));
      // HALF2(y[2*i+idx]) = ldst_y[i];  // 没有LDST128bit写回
    }
    // LDST128BITS(ldst_y[0])与LDST128BITS(ldst_y)有区别，最好带上[0];
    LDST128BITS(y[idx]) = LDST128BITS(ldst_y[0]);
  }
  else if(idx < N)
  {
    for(int i=0;i+idx < N;i++)
    {
      half v = x[i+idx];
      v = __hmin(__hmax(v, MIN_EXP_F16), MAX_EXP_F16);
      const half one = __float2half(1.0f);
      y[i+idx] = one / __hadd(one, hexp(-v));
    }
  }
}

#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func)                                   \
  m.def(STRINGFY(func), &func, STRINGFY(func));

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  TORCH_CHECK((T).scalar_type() == (th_type), #T " has an unexpected dtype")

// 练习脚手架把 Tensor 展平成一维，并按每个 block 覆盖 256 个标量元素
// 启动。这样不能被 2、4、8 整除的长度也会实际进入 learner-owned 尾部路径。
#define TORCH_BINDING_SIGMOID(packed_type, th_type, element_type, n_elements)  \
  void sigmoid_##packed_type(torch::Tensor x, torch::Tensor y) {               \
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
    sigmoid_##packed_type##_kernel<<<grid, block>>>(                            \
        reinterpret_cast<element_type *>(x.data_ptr()),                         \
        reinterpret_cast<element_type *>(y.data_ptr()), N);                    \
  }

TORCH_BINDING_SIGMOID(f32, torch::kFloat32, float, 1)
TORCH_BINDING_SIGMOID(f32x4, torch::kFloat32, float, 4)
TORCH_BINDING_SIGMOID(f16, torch::kHalf, half, 1)
TORCH_BINDING_SIGMOID(f16x2, torch::kHalf, half, 2)
TORCH_BINDING_SIGMOID(f16x8, torch::kHalf, half, 8)
TORCH_BINDING_SIGMOID(f16x8_pack, torch::kHalf, half, 8)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  TORCH_BINDING_COMMON_EXTENSION(sigmoid_f32)
  TORCH_BINDING_COMMON_EXTENSION(sigmoid_f32x4)
  TORCH_BINDING_COMMON_EXTENSION(sigmoid_f16)
  TORCH_BINDING_COMMON_EXTENSION(sigmoid_f16x2)
  TORCH_BINDING_COMMON_EXTENSION(sigmoid_f16x8)
  TORCH_BINDING_COMMON_EXTENSION(sigmoid_f16x8_pack)
}
