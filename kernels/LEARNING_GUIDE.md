# CUDA 算子学习与练习指南

这份指南面向第一次接触 CUDA 算子的学习者。目标不是一次读懂本目录里的全部优化代码，而是从最简单的 FP32 逐元素加法开始，建立下面这条完整链路：

```text
理解数学定义
  → 编写 CUDA kernel
  → 配置 grid 和 block
  → 从 Python 调用 kernel
  → 验证正确性与边界
  → 比较性能
  → 尝试向量化和低精度优化
```

## 当前学习进度

| 算子 | 状态 | 已完成内容 |
|---|---|---|
| Elementwise Add | 已完成 | FP32、FP32x4、FP16、FP16x2、FP16x8、FP16x8 pack |
| ReLU | 已完成 | 六个 TODO、边界正确性测试和性能测试 |
| Sigmoid | 已完成 | 六个 TODO、指数范围处理、`half2` 运算、128-bit pack 和全部边界测试 |
| Swish | 进行中 | 已创建练习文件，下一步从 TODO 1 开始 |

最近一次进度更新：2026-10-19。Sigmoid 的六个实现已经实际运行并全部输出 `PASS`，当前练习目标切换为 Swish。

## 1. 这个目录是什么

本目录实现的是逐元素加法：

```text
c[i] = a[i] + b[i]
```

其中 `a`、`b` 和 `c` 的元素数量相同。每个输出元素彼此独立，因此它很适合作为第一个 CUDA 算子。

目录中的主要文件：

```text
elementwise.cu    CUDA kernel、启动代码和 PyTorch C++ 绑定
elementwise.py    编译扩展、生成数据和运行性能测试
run.sh            当前 A40 机器使用的局部启动配置
README.md         项目原始说明和参考输出
```

需要特别注意：`elementwise.cu` 已经是完整的参考实现，并没有等待填写的 TODO。练习时应当把它视为“答案集”，先自己思考和实现，再回来对照。

## 2. 当前文件中有哪些实现

`elementwise.cu` 从简单到复杂提供了以下版本：

| 实现 | 数据类型 | 每个线程处理的元素数 | 当前阶段是否需要掌握 |
|---|---|---:|---|
| `elementwise_add_f32_kernel` | FP32 | 1 | 是，第一道练习 |
| `elementwise_add_f32x4_kernel` | FP32 | 4 | 完成基础版后学习 |
| `elementwise_add_f16_kernel` | FP16 | 1 | FP32 阶段完成后学习 |
| `elementwise_add_f16x2_kernel` | FP16 | 2 | 后续学习 |
| `elementwise_add_f16x8_kernel` | FP16 | 8 | 后续学习 |
| `elementwise_add_f16x8_pack_kernel` | FP16 | 8，128-bit 打包访存 | 最后学习 |

第一次阅读时，只关注 `elementwise_add_f32_kernel`，暂时跳过 FP16、PyBind 宏和打包访存。

## 3. 第一个 kernel 在做什么

基础实现的核心逻辑如下：

```cpp
__global__ void elementwise_add_f32_kernel(float *a, float *b, float *c,
                                           int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N)
    c[idx] = a[idx] + b[idx];
}
```

### 3.1 `__global__`

`__global__` 表示这是一个在 GPU 上执行、由 CPU 发起调用的 CUDA kernel。

### 3.2 全局线程编号

```cpp
int idx = blockIdx.x * blockDim.x + threadIdx.x;
```

- `threadIdx.x`：线程在线程块中的编号。
- `blockIdx.x`：当前线程块的编号。
- `blockDim.x`：每个线程块包含的线程数。
- `idx`：这个线程在整个一维任务中的全局编号。

假设每个 block 有 4 个线程：

```text
block 0：idx = 0, 1, 2, 3
block 1：idx = 4, 5, 6, 7
block 2：idx = 8, 9, 10, 11
```

如果 `N=10`，最后两个线程的 `idx` 分别是 10 和 11，它们不应该访问数组。

### 3.3 边界判断

```cpp
if (idx < N)
```

CUDA 通常会启动足够多的完整 block 来覆盖输入，因此线程总数可能略大于 `N`。这个判断防止最后一个 block 中多出的线程越界访问显存。

### 3.4 逐元素计算

```cpp
c[idx] = a[idx] + b[idx];
```

每个线程执行三次全局内存操作：读取 `a[idx]`、读取 `b[idx]`、写入 `c[idx]`，以及一次浮点加法。这个算子通常受显存带宽限制，而不是受浮点计算能力限制。

## 4. Grid 和 Block 如何启动

CUDA kernel 使用下面的语法启动：

```cpp
kernel<<<grid, block>>>(arguments...);
```

对于一个线程处理一个元素的基础版本，可以写成：

```cpp
int block_size = 256;
int grid_size = (N + block_size - 1) / block_size;

elementwise_add_f32_kernel<<<grid_size, block_size>>>(a, b, c, N);
```

向上取整公式：

```text
(N + block_size - 1) / block_size
```

例如 `N=257`、`block_size=256` 时需要两个 block，共启动 512 个线程。其中只有前 257 个线程真正参与计算，其余线程由边界判断拦截。

## 5. Python 如何调用 CUDA

完整调用过程如下：

```text
elementwise.py
  ↓ torch.utils.cpp_extension.load 编译 elementwise.cu
PyTorch C++/PyBind 绑定
  ↓ 准备 grid、block 和数据指针
elementwise_add_f32_kernel<<<grid, block>>>
  ↓ GPU 并行计算
c = a + b
```

`elementwise.py` 中的：

```python
lib = load(
    name="elementwise_lib",
    sources=["elementwise.cu"],
    ...
)
```

会把 CUDA 文件编译成 Python 扩展。文件末尾的 `PYBIND11_MODULE` 负责把 C++ 函数暴露给 Python，因此可以调用：

```python
lib.elementwise_add_f32(a, b, c)
```

## 6. 本机运行方法

当前机器使用 NVIDIA A40，计算能力是 `sm_86`。仓库内的 `run.sh` 只为本次进程指定已有的编译器和 GPU 架构，不会修改系统 CUDA、驱动、GCC 默认版本或 Conda 环境。

```bash
conda activate pytorch
cd /home/xhk/LeetCUDA/kernels/elementwise
./run.sh
```

第一次执行会编译 CUDA 扩展，因此耗时明显长于后续运行。后续运行一般会复用 PyTorch 扩展缓存。

运行前也可以检查环境：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name())"
```

## 7. 推荐练习方式

不要直接删除参考实现。建议先创建个人练习分支：

```bash
cd /home/xhk/LeetCUDA
git switch -c practice
```

然后复制练习文件：

```bash
cd kernels/elementwise
cp elementwise.cu elementwise_practice.cu
cp elementwise.py elementwise_practice.py
```

在 `elementwise_practice.py` 中修改扩展名称和源码路径，避免使用原实现或旧编译缓存：

```python
lib = load(
    name="elementwise_practice_lib",
    sources=["elementwise_practice.cu"],
    ...
)
```

如果希望继续通过 `run.sh` 使用本机局部配置，可以临时将脚本最后一行改成练习入口，或者直接执行：

```bash
CC=/usr/bin/gcc-9 \
CXX=/usr/bin/g++-9 \
TORCH_CUDA_ARCH_LIST=8.6 \
python elementwise_practice.py
```

## 8. 分阶段练习任务

### 任务 0：能解释线程编号

先不要改代码。手算以下配置中的全局线程编号：

```text
N = 10
blockDim.x = 4
gridDim.x = 3
```

验收要求：

- 能写出每个 block 覆盖的 `idx`。
- 能指出哪些线程会被 `idx < N` 拦截。
- 能解释删除边界判断可能导致什么问题。

### 任务 1：独立实现 FP32 基础版

在练习文件中清空函数体，不看参考答案重新实现：

```cpp
__global__ void elementwise_add_f32_kernel(float *a, float *b, float *c,
                                           int N) {
  // 由你实现
}
```

验收要求：

- 正确计算全局线程编号。
- 正确进行边界判断。
- 执行 `c[i] = a[i] + b[i]`。
- `N=1、5、256、257` 时均正确。

### 任务 2：加入严格的正确性验证

原测试主要展示结果和耗时。练习时应加入明确断言：

```python
expected = a + b
lib.elementwise_add_f32(a, b, c)
torch.cuda.synchronize()
torch.testing.assert_close(c, expected)
print("PASS: elementwise_add_f32")
```

先用小输入验证：

```python
a = torch.tensor([1., 2., 3., 4., 5.], device="cuda")
b = torch.tensor([10., 20., 30., 40., 50.], device="cuda")
c = torch.zeros_like(a)
```

预期结果：

```text
[11, 22, 33, 44, 55]
```

验收要求：

- 必须使用 `torch.testing.assert_close`，不能只看前两个打印值。
- 必须覆盖不能被 block size 整除的 `N`。
- 修改 kernel 后重新编译，确保没有误用旧扩展缓存。

### 任务 3：比较不同 Block Size

依次测试：

```text
32、64、128、256、512
```

记录以下信息：

| Block Size | 正确性 | 平均耗时 | 备注 |
|---:|---|---:|---|
| 32 | | | |
| 64 | | | |
| 128 | | | |
| 256 | | | |
| 512 | | | |

验收要求：

- 所有合法配置的计算结果一致。
- 知道 block size 通常影响性能，但不应影响正确性。
- 计时前进行预热，并在计时后同步 GPU。

### 任务 4：实现 FP32x4 向量化版本

该版本让每个线程处理连续 4 个 FP32 元素：

```text
线程 0：元素 0～3
线程 1：元素 4～7
线程 2：元素 8～11
```

主要问题：

1. 全局索引需要乘以 4。
2. 主路径一次加载和存储 4 个元素。
3. `N` 不能被 4 整除时必须处理尾部元素。
4. 向量化访问需要考虑指针对齐。

建议测试：

```text
N = 1、3、4、5、7、8、9、1025
```

验收要求：

- 所有尺寸都与 `torch.add` 一致。
- 能解释为什么不能只处理 `N % 4 == 0` 的输入。
- 能解释 `float4` 版本减少了什么类型的指令。

### 任务 5：学习 FP16 和 Half2

按以下顺序学习：

```text
f16 → f16x2 → f16x8 → f16x8_pack
```

需要理解：

- `float` 占 32 bit。
- `half` 占 16 bit。
- `half2` 可以同时表示和运算两个 FP16。
- 8 个 FP16 共 128 bit。
- FP16 与 FP32 的结果可能存在精度差异。

验收要求：

- 使用适合 FP16 的误差容忍度比较结果。
- 能区分输入/输出数据类型和累加数据类型。
- 能解释 `__hadd` 与 `__hadd2` 的区别。

## 9. 正确的性能测试顺序

优化时始终遵循：

```text
先正确，再快速
```

推荐顺序：

1. 小输入手工验证。
2. 多种边界尺寸验证。
3. 与 PyTorch 参考结果比较。
4. 预热 GPU。
5. 多次运行取平均耗时。
6. 比较基础版、向量化版和 PyTorch 版本。

CUDA kernel 是异步启动的，因此直接使用 CPU 计时之前和之后都需要注意同步：

```python
torch.cuda.synchronize()
start = time.time()

for _ in range(iters):
    lib.elementwise_add_f32(a, b, c)

torch.cuda.synchronize()
elapsed = time.time() - start
```

更严谨的 GPU 计时可以使用 `torch.cuda.Event` 或 CUDA Event。

## 10. 常见问题

### 10.1 为什么结果没有变化

可能原因：

- 修改的是练习文件，但 Python 仍在编译原始 `elementwise.cu`。
- 扩展名称没有改变，复用了旧缓存。
- 忘记保存文件。
- kernel 没有真正被 Python 调用。

可以同时修改扩展名称和 `sources`，强制区分练习版本。

### 10.2 为什么必须测试 `N=257`

如果 block size 是 256，`N=256` 恰好没有多余线程，错误的边界逻辑可能不会暴露；`N=257` 会启动第二个 block，更容易验证向上取整和边界判断。

### 10.3 为什么向量化版本不一定更快

可能受到以下因素影响：

- 输入规模太小，启动开销占主导。
- 地址没有满足对齐条件。
- 内存带宽已经接近上限。
- 每个线程工作增加后，并行线程数量下降。
- 测量方法或预热方式不正确。

向量化是一种工具，不是对所有尺寸都必然加速的保证。

### 10.4 为什么出现系统头文件编译错误

本机曾出现：

```text
identifier "__builtin_dynamic_object_size" is undefined
```

这属于 NVCC、主机编译器与系统头文件自动探测时的工具链问题，不是 kernel 算法错误。本目录的 `run.sh` 已通过局部指定编译器和 `sm_86` 解决；不要为此修改 GPU 驱动或系统全局编译器。

## 11. 暂时可以跳过什么

在完成 FP32 基础版和 FP32x4 之前，可以先跳过：

- PyBind 宏的具体展开过程。
- BF16 和 FP8 头文件。
- FP16x8 pack 的全部细节。
- Tensor Core、MMA、CuTe 和 FlashAttention。
- 是否立即超过 PyTorch 性能。

当前更重要的是形成一个可重复的练习闭环，而不是追求复杂度。

## 12. 第一阶段完成标准

以下第一阶段项目已经完成：

- [x] 能解释 `blockIdx.x`、`blockDim.x`、`threadIdx.x`。
- [x] 能独立写出 FP32 elementwise add kernel。
- [x] 能正确计算 grid size。
- [x] 能解释并验证边界判断。
- [x] 能从 Python 调用自己的 CUDA kernel。
- [x] 能使用 `torch.testing.assert_close` 验证结果。
- [x] 能测试 `N=1、5、256、257` 等边界尺寸。
- [x] 能解释 FP32x4 向量化的基本思路。
- [x] 能正确预热、同步并测量 GPU kernel。

第一阶段已经结束，当前进入逐元素融合算子 Swish 的练习。

## 13. 后续算子学习路线

Elementwise Add、ReLU 和 Sigmoid 已完成。当前题目是 `kernels/swish/`，暂时不要直接进入 FlashAttention。

### 13.1 已完成：Sigmoid

Sigmoid 与 ReLU 的练习结构基本一致，也包含以下六种实现：

```text
sigmoid_f32
sigmoid_f32x4
sigmoid_f16
sigmoid_f16x2
sigmoid_f16x8
sigmoid_f16x8_pack
```

因此可以继续巩固：

- 全局线程索引与边界判断；
- `float4` 向量化；
- `half` 和 `half2`；
- 一个线程处理 2、4 或 8 个元素；
- 128-bit 加载与存储；
- 输入长度不能被向量宽度整除时的尾部处理。

Sigmoid 的数学定义为：

```text
sigmoid(x) = 1 / (1 + exp(-x))
```

它会在 ReLU 的基础上增加以下知识点：

- FP32 的 `expf` 与 FP16 的 `hexp`；
- FP32、FP16 和 `half2` 常量的正确构造；
- 使用 `fminf`、`fmaxf`、`__hmin` 和 `__hmax` 限制指数输入；
- 指数溢出、下溢和数值稳定性；
- `--use_fast_math` 对计算精度与性能的影响；
- ReLU 这种访存密集型算子与 Sigmoid 这种计算量更高的逐元素算子的区别。

练习时仍然按照下面的顺序逐项完成和验证：

```text
FP32 标量
  → FP32x4
  → FP16 标量
  → FP16x2
  → FP16x8
  → FP16x8 pack
```

每完成一项，都应先运行该项的全部边界尺寸并打印 `PASS TODO N`，然后再进入下一项。建议至少覆盖：

```text
N = 1、2、3、4、5、7、8、9、17、257
```

需要注意，参考实现中的向量化代码不一定具备完整的尾部保护。创建练习文件时不能机械复制参考实现的边界行为，应采用：

```text
完整向量范围 → 向量化主路径
不足向量宽度 → 安全的标量尾部路径
```

同时增加极端输入测试，例如较大的正数、较大的负数、零以及 FP16 边界附近的数值，并与 `torch.sigmoid` 比较。

### 13.2 当前任务：Swish

Swish 的数学定义为：

```text
swish(x) = x * sigmoid(x) = x / (1 + exp(-x))
```

它在 Sigmoid 的基础上增加了一次输入乘法，是学习简单算子融合的合适下一步。练习仍按以下顺序完成：

```text
FP32 标量
  → FP32x4
  → FP16 标量
  → FP16x2
  → FP16x8
  → FP16x8 pack
```

重点关注：

- 不要只计算 `sigmoid(x)`，最终输出还必须乘以原输入 `x`；
- 区分 FP32 的 `expf` 与 FP16/FP16x2 的指数和算术操作；
- 向量化主路径只能在完整的 2、4 或 8 元素范围内读写；
- 尾部路径要覆盖不能被向量宽度整除的输入；
- 128-bit pack 版本应先整体加载，在局部完成计算后再整体写回；
- 每完成一个 TODO，都先通过全部边界尺寸，再开始下一项。

### 13.3 推荐的完整进阶顺序

```text
Elementwise Add（已完成）
  → ReLU（已完成）
  → Sigmoid（已完成）
  → Swish（进行中）
  → GELU
  → Reduce
  → Softmax
  → LayerNorm / RMSNorm
  → 矩阵乘法
  → FlashAttention
```

各阶段的主要学习目标如下：

| 阶段 | 主要新增知识 |
|---|---|
| Sigmoid | 指数计算、数值范围限制、FP16 数学函数 |
| Swish | `x * sigmoid(x)`，简单算子融合 |
| GELU | 更复杂的激活函数与近似计算 |
| Reduce | warp shuffle、共享内存、block reduction |
| Softmax | max-reduce、指数、sum-reduce、归一化和数值稳定性 |
| LayerNorm / RMSNorm | 多阶段归约、均值/方差或平方均值、融合计算 |
| 矩阵乘法 | 分块、共享内存、访存复用和 Tensor Core 基础 |
| FlashAttention | 矩阵乘法、在线 Softmax、共享内存与 warp 协作的综合应用 |

### 13.4 为什么暂时不直接写 FlashAttention

FlashAttention 不是一个独立的小技巧，而是多个基础能力的组合：

- QK 矩阵乘法；
- shared memory 分块与数据复用；
- warp 间协作；
- reduce；
- 数值稳定的在线 Softmax；
- PV 矩阵乘法；
- 寄存器、共享内存和占用率之间的权衡。

如果还没有独立完成 Reduce、Softmax 和基础矩阵乘法，直接阅读 FlashAttention 很容易停留在逐行模仿。按照上述路线逐步练习后，再进入 `kernels/flash-attn/`，会更容易理解每个设计选择解决的具体问题。
