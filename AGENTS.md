LocalLLM是使用C++和CUDA实现的高性能Qwen模型专用推理引擎，VLLM架构

## third party dependency
1. flashinfer
2. gguf
3. pytorch
4. googletest, glog, gflag
5. meta folly

## Rule
1. **严格遵守Google C++ Style**
2. **严格遵守TDD规范**：每个C++和CUDA都需要在test下使用gtest进行严格测试，通过后才能进行下一步任务

## 项目架构

阅读当前文件夹下"architecture.md"文件

## Others
1. qwen model file in models/Qwen3-0.6B , huggingface format, for testing
2. use git for progress managment