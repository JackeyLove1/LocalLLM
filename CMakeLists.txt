cmake_minimum_required(VERSION 3.24)

project(LocalLLM LANGUAGES CXX)

option(USE_CUDA "Build optional CUDA helper kernels" ON)
option(LOCALLLM_BUILD_TESTS "Build unit and integration tests" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(Python3_EXECUTABLE "/usr/bin/python3")
find_package(Python3 REQUIRED COMPONENTS Interpreter)

execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(TORCH_CMAKE_PREFIX_PATH)
    list(PREPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PREFIX_PATH}")
endif()

find_package(Torch REQUIRED)

get_filename_component(TORCH_INSTALL_PREFIX "${TORCH_CMAKE_PREFIX_PATH}/../.." ABSOLUTE)
file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")

if(USE_CUDA)
    find_program(NVCC_EXECUTABLE nvcc)
    if(NVCC_EXECUTABLE)
        find_package(CUDAToolkit QUIET)
    endif()
    if(CUDAToolkit_FOUND)
        enable_language(CUDA)
        add_compile_definitions(LOCALLLM_USE_CUDA=1)
    else()
        message(WARNING "CUDA toolkit or VS CUDA toolset not found, building CPU-only baseline.")
        set(USE_CUDA OFF)
    endif()
endif()

add_library(
    localllm_lib
    src/common/file_util.cpp
    src/common/logging.cpp
    src/model/qwen_config.cpp
    src/model/safetensors_loader.cpp
    src/model/model_weights.cpp
    src/tokenizer/bpe_tokenizer.cpp
    src/runtime/sampling.cpp
    src/executor/qwen_executor.cpp
    src/app/cli_options.cpp
    src/app/application.cpp
    3rdparty/json11/json11.cpp
)

target_include_directories(
    localllm_lib
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/json11
)

target_link_libraries(localllm_lib PUBLIC ${TORCH_LIBRARIES})
target_compile_definitions(localllm_lib PUBLIC NOMINMAX)

if(USE_CUDA AND CUDAToolkit_FOUND)
    target_sources(localllm_lib PRIVATE src/runtime/cuda_kernels.cu)
    set_target_properties(
        localllm_lib
        PROPERTIES
            CUDA_STANDARD 17
            CUDA_STANDARD_REQUIRED ON
    )
    target_link_libraries(localllm_lib PRIVATE CUDA::cudart)
endif()

add_executable(localllm main.cpp)
target_link_libraries(localllm PRIVATE localllm_lib)
if(TORCH_DLLS)
    add_custom_command(
        TARGET localllm POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${TORCH_DLLS} $<TARGET_FILE_DIR:localllm>
        COMMAND_EXPAND_LISTS
    )
endif()

if(MSVC)
    target_compile_options(localllm_lib PRIVATE /W4 /EHsc)
    target_compile_options(localllm PRIVATE /W4 /EHsc)
else()
    target_compile_options(localllm_lib PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(localllm PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(LOCALLLM_BUILD_TESTS)
    enable_testing()
    add_subdirectory(3rdparty/googletest EXCLUDE_FROM_ALL)

    add_executable(
        localllm_tests
        test/qwen_config_test.cpp
        test/safetensors_loader_test.cpp
        test/bpe_tokenizer_test.cpp
        test/sampling_test.cpp
        test/qwen_executor_smoke_test.cpp
    )
    target_link_libraries(localllm_tests PRIVATE localllm_lib gtest_main)
    if(TORCH_DLLS)
        add_custom_command(
            TARGET localllm_tests POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different ${TORCH_DLLS} $<TARGET_FILE_DIR:localllm_tests>
            COMMAND_EXPAND_LISTS
        )
    endif()
    include(GoogleTest)
    gtest_discover_tests(
        localllm_tests
        DISCOVERY_MODE PRE_TEST
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    )
endif()
