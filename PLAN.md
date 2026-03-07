# LocalLLM
## Logging

The project now routes runtime logs through `glog`.

- Logs default to `stderr` and do not write log files automatically.
- `stdout` remains reserved for generated model text so existing CLI piping behavior is unchanged.
- If you need more verbose logging later, extend the startup configuration with additional `glog` flags.

## brpc

- Bundled `brpc` support is available through the CMake option `LOCALLLM_ENABLE_BRPC`.
- The option defaults to `OFF` so existing builds are unchanged until RPC code is added.
- When enabled, the project exposes a `localllm::brpc` target that wraps the vendored `3rdparty/brpc` build.
- `brpc` still requires its system dependencies to be installed and discoverable by CMake, including `gflags`, `protobuf`, `libprotoc`, `leveldb`, and `OpenSSL`.

LocalLLM 是一个使用 C++ 实现的 Qwen 专用推理引擎原型，目标架构参考 vLLM 思路，当前已完成 Phase 1 baseline：

- 支持本地 `Qwen3-0.6B` Hugging Face 格式模型
- 支持 `config.json` 与 `model.safetensors` 加载
- 支持原生 BPE tokenizer
- 支持单请求 `prefill + decode`
- 支持 CLI 文本生成
- 支持 gtest 单元测试和 smoke test

详细架构规划见 [`architecture.md`](./architecture.md)。

## 当前状态

当前仓库已经验证通过的能力：

- CMake 工程可生成
- `localllm` 可执行文件可构建
- `localllm_tests` 测试可构建并通过
- CLI 可实际加载 `models/Qwen3-0.6B` 并完成单步生成

当前限制：

- 这次验证环境中的 `torch` 是 CPU-only，因此运行时会自动回退到 CPU
- `USE_CUDA=ON` 的完整 CUDA 路径还没有在当前机器上打通
- 尚未接入 Hugging Face 参考实现的 logits 对齐测试

## 依赖

仓库当前使用到的核心依赖：

1. PyTorch / libtorch
2. googletest
3. json11
4. CUDA Toolkit（可选，当前构建可关闭）

## 开发约束

1. 严格遵守 Google C++ Style
2. 严格遵守 TDD：新增 C++/CUDA 代码时，优先补测试
3. 模型测试目录默认使用 `models/Qwen3-0.6B`

## Windows + WSL 使用方式

这个仓库适合在“WSL 里编辑、Windows 侧构建/运行”的模式下使用。

原因：

- 代码和模型目录当前位于 Windows 文件系统
- 当前已经验证通过的是 Visual Studio 2022 + Windows CMake + Windows Python/PyTorch 路径
- 如果你直接在 WSL 里构建，需要自己准备 Linux 版 libtorch / CUDA / gtest 环境；这不是当前 README 的默认路径

推荐工作流：

1. 在 WSL 中编辑代码
2. 在 Windows PowerShell 或 Developer PowerShell 中执行构建、运行和测试
3. 如果你从 WSL 进入仓库，对应路径通常是 `/mnt/c/Software/Codes/cuda/LocalLLM`

## 环境准备

### 1. Windows 侧工具

需要准备：

- Visual Studio 2022（包含 C++ 工具链）
- CMake
- Python 3
- PyTorch

当前仓库是通过下面的 Windows 工具链验证的：

- Visual Studio 2022
- CMake（VS 自带路径可用）
- Python 3.14
- `torch 2.10.0+cpu`

### 2. Python 包

在 Windows PowerShell 中安装：

```powershell
python -m pip install torch transformers safetensors sentencepiece
```

如果你后续要切换到 CUDA 运行，还需要安装带 CUDA 的 PyTorch，并保证它的 CMake 包可被找到。

### 3. 模型目录

默认模型路径：

```text
models/Qwen3-0.6B
```

这个目录下至少需要：

- `config.json`
- `model.safetensors`
- `vocab.json`
- `merges.txt`
- `tokenizer_config.json`

## 如何构建

### 方式一：Windows PowerShell

在仓库根目录执行：

```powershell
& "C:\Program Files\Microsoft Visual Studio\2022\Professional\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe" `
  -S . `
  -B build `
  -G "Visual Studio 17 2022" `
  -A x64 `
  -DUSE_CUDA=OFF
```

然后构建主程序：

```powershell
& "C:\Program Files\Microsoft Visual Studio\2022\Professional\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe" `
  --build build `
  --config Release `
  --target localllm
```

构建测试：

```powershell
& "C:\Program Files\Microsoft Visual Studio\2022\Professional\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe" `
  --build build `
  --config Release `
  --target localllm_tests
```

### 方式二：从 WSL 调用 Windows 工具

如果你在 WSL 中工作，可以直接调用 Windows 侧 `cmake.exe`：

```bash
/mnt/c/Program\ Files/Microsoft\ Visual\ Studio/2022/Professional/Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe \
  -S /mnt/c/Software/Codes/cuda/LocalLLM \
  -B /mnt/c/Software/Codes/cuda/LocalLLM/build \
  -G "Visual Studio 17 2022" \
  -A x64 \
  -DUSE_CUDA=OFF
```

然后构建：

```bash
/mnt/c/Program\ Files/Microsoft\ Visual\ Studio/2022/Professional/Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe \
  --build /mnt/c/Software/Codes/cuda/LocalLLM/build \
  --config Release \
  --target localllm
```

## 如何执行

### 1. 最小执行示例

```powershell
build\Release\localllm.exe --prompt Hello --max-new-tokens 1 --no-stream
```

### 2. 流式输出

```powershell
build\Release\localllm.exe --prompt "Give me a short introduction to large language models." --max-new-tokens 64
```

### 3. Chat 模式

```powershell
build\Release\localllm.exe --chat --prompt "你好，介绍一下你自己。" --max-new-tokens 64
```

### 4. 指定采样参数

```powershell
build\Release\localllm.exe `
  --prompt "Write a one sentence summary of CUDA." `
  --max-new-tokens 32 `
  --temperature 0.7 `
  --top-p 0.9 `
  --top-k 20
```

### 5. 从标准输入读取 prompt

```powershell
Get-Content prompt.txt | build\Release\localllm.exe --max-new-tokens 64
```

## 如何测试

### 1. 构建测试目标

```powershell
& "C:\Program Files\Microsoft Visual Studio\2022\Professional\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe" `
  --build build `
  --config Release `
  --target localllm_tests
```

### 2. 运行全部测试

```powershell
& "C:\Program Files\Microsoft Visual Studio\2022\Professional\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\ctest.exe" `
  --test-dir build `
  -C Release `
  --output-on-failure
```

### 3. 运行单个测试

例如只跑 tokenizer 测试：

```powershell
build\Release\localllm_tests.exe --gtest_filter=BpeTokenizerTest.*
```

例如只跑执行器 smoke test：

```powershell
build\Release\localllm_tests.exe --gtest_filter=QwenExecutorSmokeTest.*
```

## 当前测试覆盖

当前已经包含：

- `QwenConfig` 解析测试
- `safetensors` 元数据读取与张量加载测试
- BPE tokenizer 编码/解码测试
- sampling 行为测试
- `QwenExecutor` 单步生成 smoke test

## CUDA 说明

如果你后续准备打通 CUDA 路径，需要满足两个前提：

1. 安装带 CUDA 的 PyTorch / libtorch
2. Visual Studio 的 CUDA toolset 可被 CMake 正常发现

满足后可以尝试：

```powershell
& "C:\Program Files\Microsoft Visual Studio\2022\Professional\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe" `
  -S . `
  -B build `
  -G "Visual Studio 17 2022" `
  -A x64 `
  -DUSE_CUDA=ON
```

如果 CMake 无法发现 CUDA toolset，当前工程会建议回退到 `-DUSE_CUDA=OFF`。

## 常见问题

### 1. `cmake` 不在 PATH

直接使用 Visual Studio 自带的 `cmake.exe` 全路径即可，见上文命令。

### 2. 测试运行时报缺少 DLL

当前 `CMakeLists.txt` 已在构建后复制 Torch 运行时 DLL 到目标目录。如果仍然缺少 DLL，优先检查本机 PyTorch 安装是否完整。

### 3. 程序提示回退到 CPU

这通常说明当前 PyTorch 是 CPU-only，或者 CUDA 运行时不可用。当前仓库在这种情况下仍然可以完成 baseline 验证。

## 后续建议

下一步更合理的方向：

1. 增加 Hugging Face 参考实现 logits 对齐测试
2. 在 CUDA libtorch 环境中打通 `USE_CUDA=ON`
3. 继续进入 Phase 2：scheduler、paged KV cache、continuous batching
