# 一、方案定位

* Qwen dense 模型推理
* C++/CUDA runtime
* continuous batching
* paged KV cache
* prefill / decode 分离
* 长上下文支持
* 在线 serving

你真正要解决的问题不是“让模型能跑”，而是：

* 多请求怎么高效调度
* KV cache 怎么省显存、少碎片
* decode 怎么快
* 长短请求混跑怎么不崩
* 在线服务怎么做到稳定吞吐

---

# 二、总体架构

建议按 6 个模块来做。

## 1. API / Serving 层

负责：

* OpenAI 兼容接口
* 请求接入
* 流式输出
* sampling 参数解析
* 会话状态管理

这层先别做太重，能跑通即可。

---

## 2. Scheduler 层

这是核心之一。

负责：

* 请求排队
* continuous batching
* prefill / decode 分离
* 动态组 batch
* token budget 控制
* 长短请求混合调度

这里的目标不是“按请求调度”，而是尽量做到**按 token 调度**。

---

## 3. Cache Manager 层

这是第二核心。

负责：

* paged KV cache
* block 分配与回收
* block table 管理
* prefix cache 预留接口
* cache 生命周期管理

因为 Qwen 是标准 Transformer/GQA 路线，所以 KV cache 是整个系统的命门。

---

## 4. Model Runtime 层

负责：

* 模型权重加载
* Qwen layer 执行
* prefill forward
* decode forward
* logits 输出
* CUDA graph 接入点

建议一开始就做成 **Qwen 专用 executor**，不要上来做泛型大抽象。

---

## 5. CUDA Kernel 层

负责：

* RMSNorm
* RoPE
* QKV projection 相关路径
* GQA attention
* paged decode attention
* MLP / SwiGLU
* sampling 前后处理
* cache gather / scatter

这一层决定性能上限。

---

## 6. Metrics / Benchmark 层

负责：

* TTFT
* TPOT
* tokens/s
* GPU memory usage
* cache hit / miss
* block fragmentation
* scheduler occupancy

没有这层，后面优化会非常痛苦。

---

# 三、技术路线

建议分成四个阶段。

---

## Phase 1：做一个能稳定跑的 Qwen baseline

目标：

* 支持一个 Qwen dense 模型
* 单机单卡
* BF16
* C++/CUDA 推理
* 基础 prefill + decode
* 基础 API 服务

### 这一阶段必须完成的事

### 模型支持

* 选定一个具体模型作为 baseline
  建议先从：

  * Qwen3-0.6B

先不要一上来搞最大模型。

### 权重加载

* safetensors 加载
* 参数映射到内部结构
* 权重 layout 预处理
* embedding / lm_head / layer 参数组织

### 基础执行链路

* tokenizer 接入
* input ids -> embedding
* RMSNorm
* RoPE
* attention
* MLP
* lm_head
* sampling

### CUDA 基础算子

* RMSNorm kernel
* RoPE apply kernel
* matmul 路径先可调用 cuBLAS / CUTLASS
* 简单 attention baseline
* 简单 sampling

### 服务闭环

* 单请求推理
* 流式输出
* stop condition
* max_tokens / temperature / top_p 支持

### 验收标准

* 能正确生成文本
* 与参考实现 logits 基本对齐
* 单请求链路稳定
* 长度和采样逻辑正确

---

## Phase 2：做 Qwen 的高吞吐系统基本盘

目标：

* continuous batching
* paged KV cache
* prefill / decode 分离
* 多请求并发
* 在线服务可用

这是最关键阶段。

### 这一阶段必须完成的事

### Scheduler

* waiting / running / finished 请求状态机
* prefill queue
* decode queue
* decode step 调度
* 动态 batch 组装
* token budget 控制

### KV cache

* block-based KV cache
* free list / bitmap 管理
* sequence -> block table 映射
* append token
* release cache
* block reuse

### Decode 路径优化

* 单 token decode batch
* paged KV gather
* GQA-aware decode attention
* 长短请求混合 batch

### API 层增强

* 多请求并发
* streaming 稳定输出
* cancel / timeout 处理
* request id / tracing

### 验收标准

* 多请求同时推理稳定
* 显存不会快速碎片化
* decode 吞吐显著提升
* 请求完成后 cache 正确释放
* 长短请求混跑不雪崩

---

## Phase 3：做 Qwen 长上下文和服务优化

目标：

* chunked prefill
* prefix cache
* CUDA graph
* 长上下文优化
* 高并发服务稳定性增强

### 这一阶段必须完成的事

### Chunked Prefill

* 长 prompt 分块 prefill
* prefill 与 decode 抢占策略
* 避免长 prompt 阻塞全局

### Prefix Cache

* prompt 前缀 hash
* prefix 复用
* block refcount
* shared block 生命周期管理

### CUDA Graph

* decode shape bucket
* graph capture
* graph replay
* 常见 batch shape 命中优化

### Long Context 路径

* position 处理优化
* RoPE 长上下文路径
* 大上下文下 cache 访问优化
* block table 查找开销控制

### 服务增强

* admission control
* queue pressure control
* latency / throughput 模式切换
* 核心 metrics 暴露

### 验收标准

* 长 prompt 不再严重阻塞短请求
* 重复前缀场景明显加速
* decode launch 开销下降
* 长上下文性能可接受
* 系统在高并发下更稳

---

## Phase 4：做工程化和性能打磨

目标：

* FP8 / INT8
* kernel fusion
* benchmark 体系
* 多卡扩展预留
* 生产级鲁棒性

### 这一阶段必须完成的事

### 精度和量化

* BF16 baseline 固定
* FP8 尝试
* INT8 可选
* 精度回归测试

### Kernel 优化

* residual + rmsnorm fuse
* RoPE + Q/K 处理局部融合
* paged gather 优化
* decode attention 深度优化

### Benchmark

* prefill tokens/s
* decode tokens/s
* TTFT
* TPOT
* batch size sweep
* seq length sweep
* memory usage
* fragmentation rate

### 工程能力

* 配置系统
* 日志系统
* profiling 工具接入
* 故障恢复
* 模型热加载预留接口

### 验收标准

* 性能有系统性 benchmark 结果
* 优化前后收益可量化
* 可以稳定跑压测
* 可以作为后续扩展基础

---

# 四、核心 Plan

下面给你压成一个更适合执行的 plan。

## Plan A：最小可用版本

先做：

* Qwen2.5-7B
* 单卡
* BF16
* 单模型
* 单 tokenizer
* 单种 sampling 路径
* 不做 prefix cache
* 不做量化
* 不做多卡

目标：

* 2 到 4 周内做出可运行最小版本

---

## Plan B：性能可用版本

再做：

* paged KV cache
* continuous batching
* chunked prefill
* decode 优化
* benchmark

目标：

* 能明显体现“类 vLLM”的核心能力

---

## Plan C：服务可用版本

最后做：

* prefix cache
* CUDA graph
* admission control
* 稳定 metrics
* profiling + regression

目标：

* 进入真实服务级别实验

---

# 五、Todo 清单

下面是你可以直接拿去排 Jira/Notion 的 todo。

---

## A. 基础设施 Todo

* [ ] 确定第一版目标模型：Qwen2.5-7B 或 Qwen2.5-14B
* [ ] 确定权重格式：safetensors
* [ ] 建立 CMake / Bazel 构建系统
* [ ] 接入 CUDA / cuBLAS / CUTLASS
* [ ] 定义项目目录结构
* [ ] 搭建基础日志和配置系统
* [ ] 接入 tokenizer

---

## B. 模型执行 Todo

* [ ] 定义 QwenConfig
* [ ] 定义 Tensor / DeviceBuffer 抽象
* [ ] 实现模型权重加载器
* [ ] 实现 embedding
* [ ] 实现 RMSNorm
* [ ] 实现 RoPE
* [ ] 实现 attention baseline
* [ ] 实现 MLP / SwiGLU
* [ ] 实现 lm_head
* [ ] 实现 sampling

---

## C. Runtime Todo

* [ ] 定义 Sequence 对象
* [ ] 定义 BatchView
* [ ] 定义 ModelExecutor 接口
* [ ] 实现 QwenExecutor::prefill
* [ ] 实现 QwenExecutor::decode
* [ ] 实现 logits 输出接口
* [ ] 实现 token append 流程
* [ ] 实现 stop / eos / max_tokens 逻辑

---

## D. KV Cache Todo

* [ ] 设计 block/page 数据结构
* [ ] 实现 BlockPool
* [ ] 实现 block alloc/free
* [ ] 实现 sequence block table
* [ ] 实现 KV append
* [ ] 实现 cache release
* [ ] 实现 page gather 接口
* [ ] 验证长序列 cache correctness

---

## E. Scheduler Todo

* [ ] 定义 request 生命周期状态机
* [ ] 实现 waiting queue
* [ ] 实现 prefill queue
* [ ] 实现 decode queue
* [ ] 实现 continuous batching
* [ ] 实现 decode step 调度
* [ ] 实现 token budget 控制
* [ ] 实现长短请求混合调度
* [ ] 实现超时 / cancel

---

## F. CUDA Kernel Todo

* [ ] RMSNorm kernel 优化
* [ ] RoPE kernel 优化
* [ ] GQA attention kernel
* [ ] paged decode attention kernel
* [ ] KV gather/scatter kernel
* [ ] sampling 辅助 kernel
* [ ] kernel profiling
* [ ] kernel correctness test

---

## G. 服务层 Todo

* [ ] HTTP / gRPC / OpenAI-compatible API
* [ ] 流式输出
* [ ] request tracing
* [ ] metrics endpoint
* [ ] 模型加载和初始化流程
* [ ] 错误处理与恢复

---

## H. 优化 Todo

* [ ] chunked prefill
* [ ] prefix cache
* [ ] CUDA graph
* [ ] BF16 baseline benchmark
* [ ] FP8 feasibility study
* [ ] memory fragmentation 分析
* [ ] latency / throughput 模式切换
* [ ] admission control

---

## I. 测试 Todo

* [ ] 单层数值对齐测试
* [ ] 全模型 logits 对齐测试
* [ ] 单请求生成测试
* [ ] 多请求并发测试
* [ ] 长上下文测试
* [ ] cache correctness 测试
* [ ] 调度稳定性测试
* [ ] benchmark 自动化
* [ ] 回归测试

---

# 六、推荐的执行顺序

最推荐你按下面这个顺序做，不要乱跳。

## 第 1 步

先让 Qwen 单请求跑通：

* 权重加载
* forward
* sampling
* 正确输出

## 第 2 步

做最小 scheduler：

* prefill
* decode
* 多请求 batch

## 第 3 步

做 paged KV cache：

* block pool
* block table
* append/release

## 第 4 步

做 decode attention 优化：

* GQA
* paged gather
* 高并发 decode

## 第 5 步

做 chunked prefill 和长上下文：

* 防止长请求拖死系统

## 第 6 步

做 prefix cache 和 CUDA graph：

* 进一步榨性能

## 第 7 步

做 benchmark 和量化：

* 用数据指导后续优化

---

# 七、最小里程碑

你可以把项目拆成这 4 个 milestone。

## M1：模型能跑

输出正确文本，单请求稳定。

## M2：系统能并发

多请求 + scheduler + KV cache 可用。

## M3：系统能高吞吐

continuous batching + decode 优化 + chunked prefill。

## M4：系统能优化

prefix cache + CUDA graph + benchmark + profiling。

