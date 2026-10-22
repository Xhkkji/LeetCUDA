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
#define GELU_SCALE_F32 0.7978845608028654f
#define GELU_CUBIC_F32 0.044715f
#define GELU_SCALE_F16 __float2half(0.7978845608028654f)
#define GELU_CUBIC_F16 __float2half(0.044715f)

// 利用已有的数学常量宏，组合得到：√(2/π) == GELU_SCALE_F32
// M_SQRT2    = √2   M_2_SQRTPI = 2/√π
#define SQRT_2_PI M_SQRT2 * M_2_SQRTPI * 0.5f


__inline__ __device__ float gelu_float32(float x)
{
  return 0.5f * x * (1.0f + tanhf(SQRT_2_PI * (x + GELU_CUBIC_F32 * x * x * x)));
}
// TODO 1（FP32 标量）: 一个线程处理一个 FP32 元素。
// 要求：计算全局索引、检查边界，并实现 GELU 的 tanh 近似形式。
// FP32
// GELU x: N, y: N
// y[i] = 0.5*x[i]*(1+tanh(sqrt(2/pi)*(x[i]+0.044715*x[i]^3)))
// grid(ceil(N/256)), block(256)
__global__ void gelu_f32_kernel(float *x, float *y, int N) {
  // TODO: 由你实现。
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < N)
  {
    float reg_x = x[idx];
    reg_x = fminf(fmaxf(reg_x, MIN_EXP_F32), MAX_EXP_F32);
    y[idx] = gelu_float32(reg_x);
  }
}

// TODO 2（FP32x4）: 使用 float4 让一个线程处理四个 FP32 元素。
// 要求：实现向量化主路径，并安全处理不足四个元素的尾部。
// FP32x4，使用与 TODO 1 相同的 tanh 近似公式。
// GELU x: N, y: N
// grid(ceil(N/256)), block(256/4)
__global__ void gelu_f32x4_kernel(float *x, float *y, int N) {
  // TODO: 由你实现。
  int idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
  if(idx + 3 < N)
  {
    float4 reg_x = FLOAT4(x[idx]);
    float4 reg_y;

    float v0 = fminf(fmax(reg_x.x, MIN_EXP_F32), MAX_EXP_F32);
    float v1 = fminf(fmax(reg_x.y, MIN_EXP_F32), MAX_EXP_F32);
    float v2 = fminf(fmax(reg_x.z, MIN_EXP_F32), MAX_EXP_F32);
    float v3 = fminf(fmax(reg_x.w, MIN_EXP_F32), MAX_EXP_F32);
    reg_y.x = gelu_float32(v0);
    reg_y.y = gelu_float32(v1);
    reg_y.z = gelu_float32(v2);
    reg_y.w = gelu_float32(v3);

    FLOAT4(y[idx]) = reg_y;
  }
  else if(idx < N)
  {
    for(int i=0; i+idx < N; i++)
    {
      float v = fminf(fmax(x[i+idx], MIN_EXP_F32), MAX_EXP_F32);
      y[i+idx] = gelu_float32(v);
    }
  }
}

#define HALF_DIV2 __float2half(0.5f)
#define HALF_ONE __float2half(1.0f)
#define HALF_TWO __float2half(2.0f)

__inline__ __device__ half gelu_float16(half x)
{
  half inner = (GELU_SCALE_F16 * (x + GELU_CUBIC_F16 * x * x * x));
  half exp_2inner = hexp(HALF_TWO * inner);
  return HALF_DIV2 * x * __hadd(HALF_ONE ,(__hadd(exp_2inner, -HALF_ONE) / __hadd(exp_2inner, HALF_ONE)));
}
// TODO 3（FP16 标量）: 一个线程处理一个 FP16 元素。
// 要求：检查边界，并使用 FP16 运算实现 tanh 近似 GELU。
// FP16；CUDA half 没有直接 tanh，可利用
// tanh(z) = (exp(2*z)-1)/(exp(2*z)+1)。
// GELU x: N, y: N
// grid(ceil(N/256)), block(256)
__global__ void gelu_f16_kernel(half *x, half *y, int N) {
  // TODO: 由你实现。
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < N)
  {
    half reg_x = x[idx];
    half v = __hmin(__hmax(reg_x, MIN_EXP_F16), MAX_EXP_F16);
    y[idx] = gelu_float16(v);
  }
}

// TODO 4（FP16x2）: 使用 half2 让一个线程处理两个 FP16 元素。
// 要求：实现完整双元素主路径，并安全处理最后一个落单元素。
// FP16x2，使用与 TODO 3 相同的 tanh 近似公式。
// GELU x: N, y: N
// grid(ceil(N/256)), block(256/2)
__global__ void gelu_f16x2_kernel(half *x, half *y, int N) {
  // TODO: 由你实现。
  int idx = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
  if(idx + 1 < N)
  {
    half2 reg_x = HALF2(x[idx]);
    half2 reg_y;

    half v0 = __hmin(__hmax(reg_x.x, MIN_EXP_F16), MAX_EXP_F16);
    half v1 = __hmin(__hmax(reg_x.y, MIN_EXP_F16), MAX_EXP_F16);

    reg_y.x = gelu_float16(v0);
    reg_y.y = gelu_float16(v1);

    HALF2(y[idx]) = reg_y;
  }
  else if(idx < N)
  {
    for(int i=0; i+idx < N; i++)
    {
      half v = __hmin(__hmax(x[i+idx], MIN_EXP_F16), MAX_EXP_F16);
      y[i+idx] = gelu_float16(v);
    }
  }
}

// TODO 5（FP16x8）: 使用四组 half2 让一个线程处理八个 FP16 元素。
// 要求：实现完整八元素主路径，并安全处理不足八个元素的尾部。
// FP16x8，使用与 TODO 3 相同的 tanh 近似公式。
// GELU x: N, y: N
// grid(ceil(N/256)), block(256/8)
__global__ void gelu_f16x8_kernel(half *x, half *y, int N) {
  // TODO: 由你实现。
  int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
  if(idx + 7 < N)
  {
    half2 reg_x;
    half2 reg_y;
    for(int i=0; i<4; i++)
    {
      reg_x = HALF2(x[2*i+idx]);

      half v0 = __hmin(__hmax(reg_x.x, MIN_EXP_F16), MAX_EXP_F16);
      half v1 = __hmin(__hmax(reg_x.y, MIN_EXP_F16), MAX_EXP_F16);

      reg_y.x = gelu_float16(v0);
      reg_y.y = gelu_float16(v1);
      HALF2(y[2*i+idx]) = reg_y;
    }
  }
  else if(idx < N)
  {
    for(int i=0; i+idx < N; i++)
    {
      half v = __hmin(__hmax(x[i+idx], MIN_EXP_F16), MAX_EXP_F16);
      y[i+idx] = gelu_float16(v);
    }
  }
}

// TODO 6（128-bit pack）: 一次加载/存储八个 FP16 元素。
// 要求：使用 LDST128BITS 实现主路径，并安全处理尾部元素。
// FP16x8 pack，使用与 TODO 3 相同的 tanh 近似公式。
// GELU x: N, y: N
// grid(ceil(N/256)), block(256/8)
__global__ void gelu_f16x8_pack_kernel(half *x, half *y, int N) {
  // TODO: 由你实现。
  int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
  if(idx + 7 < N)
  {
    half pack_x[8];
    half pack_y[8];
    LDST128BITS(pack_x[0]) = LDST128BITS(x[idx]);
    for(int i=0 ;i<8; i++)
    {
      half v = __hmin(__hmax(pack_x[i], MIN_EXP_F16), MAX_EXP_F16);
      pack_y[i] = gelu_float16(v);
    }
    LDST128BITS(y[idx]) = LDST128BITS(pack_y[0]);
  }
  else if(idx < N)
  {
    for(int i=0; i+idx < N; i++)
    {
      half v = __hmin(__hmax(x[i+idx], MIN_EXP_F16), MAX_EXP_F16);
      y[i+idx] = gelu_float16(v);
    }
  }
}

#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func)                                   \
  m.def(STRINGFY(func), &func, STRINGFY(func));

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  TORCH_CHECK((T).scalar_type() == (th_type), #T " has an unexpected dtype")

// 将连续 Tensor 展平成一维。每个 block 覆盖 256 个标量元素；对向量
// 宽度 W，block 使用 256/W 个线程，使完整主路径和标量尾部都可到达。
#define TORCH_BINDING_GELU(packed_type, th_type, element_type, n_elements)     \
  void gelu_##packed_type(torch::Tensor x, torch::Tensor y) {                  \
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
    gelu_##packed_type##_kernel<<<grid, block>>>(                               \
        reinterpret_cast<element_type *>(x.data_ptr()),                         \
        reinterpret_cast<element_type *>(y.data_ptr()), N);                    \
  }

TORCH_BINDING_GELU(f32, torch::kFloat32, float, 1)
TORCH_BINDING_GELU(f32x4, torch::kFloat32, float, 4)
TORCH_BINDING_GELU(f16, torch::kHalf, half, 1)
TORCH_BINDING_GELU(f16x2, torch::kHalf, half, 2)
TORCH_BINDING_GELU(f16x8, torch::kHalf, half, 8)
TORCH_BINDING_GELU(f16x8_pack, torch::kHalf, half, 8)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  TORCH_BINDING_COMMON_EXTENSION(gelu_f32)
  TORCH_BINDING_COMMON_EXTENSION(gelu_f32x4)
  TORCH_BINDING_COMMON_EXTENSION(gelu_f16)
  TORCH_BINDING_COMMON_EXTENSION(gelu_f16x2)
  TORCH_BINDING_COMMON_EXTENSION(gelu_f16x8)
  TORCH_BINDING_COMMON_EXTENSION(gelu_f16x8_pack)
}
