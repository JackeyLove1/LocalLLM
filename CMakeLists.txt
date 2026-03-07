cmake_minimum_required(VERSION 3.24)

project(LocalLLM LANGUAGES CXX)

option(USE_CUDA "Build optional CUDA helper kernels" ON)
option(LOCALLLM_BUILD_TESTS "Build unit and integration tests" ON)
option(LOCALLLM_ENABLE_BRPC "Build bundled brpc support from 3rdparty/brpc" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build shared libraries" FORCE)
set(BUILD_EXAMPLES OFF CACHE BOOL "Build examples" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "Enable testing" FORCE)
set(WITH_GFLAGS OFF CACHE BOOL "Use gflags" FORCE)
set(WITH_GTEST OFF CACHE BOOL "Use Google Test" FORCE)

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

add_subdirectory(3rdparty/glog EXCLUDE_FROM_ALL)
file(WRITE "${CMAKE_BINARY_DIR}/3rdparty/glog/CTestTestfile.cmake" "")
set(BUILD_TESTING ON CACHE BOOL "Enable testing" FORCE)

if(LOCALLLM_ENABLE_BRPC)
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/brpc/CMakeLists.txt")
        message(FATAL_ERROR "LOCALLLM_ENABLE_BRPC is ON, but 3rdparty/brpc is missing.")
    endif()

    list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/brpc/cmake")

    find_package(GFLAGS QUIET)
    find_package(Protobuf QUIET)
    find_package(OpenSSL QUIET)
    find_path(LEVELDB_INCLUDE_PATH NAMES leveldb/db.h)
    find_library(LEVELDB_LIB NAMES leveldb)
    find_library(PROTOC_LIB NAMES protoc)

    set(_localllm_brpc_missing_deps "")
    if(NOT GFLAGS_FOUND)
        list(APPEND _localllm_brpc_missing_deps "gflags")
    endif()
    if(NOT Protobuf_FOUND)
        list(APPEND _localllm_brpc_missing_deps "protobuf")
    endif()
    if(NOT OpenSSL_FOUND)
        list(APPEND _localllm_brpc_missing_deps "OpenSSL")
    endif()
    if(NOT LEVELDB_INCLUDE_PATH OR NOT LEVELDB_LIB)
        list(APPEND _localllm_brpc_missing_deps "leveldb")
    endif()
    if(NOT PROTOC_LIB)
        list(APPEND _localllm_brpc_missing_deps "libprotoc")
    endif()
    if(_localllm_brpc_missing_deps)
        list(JOIN _localllm_brpc_missing_deps ", " _localllm_brpc_missing_deps_text)
        message(
            FATAL_ERROR
            "LOCALLLM_ENABLE_BRPC requires system dependencies that were not found: "
            "${_localllm_brpc_missing_deps_text}"
        )
    endif()

    set(WITH_GLOG OFF CACHE BOOL "With glog" FORCE)
    set(BUILD_UNIT_TESTS OFF CACHE BOOL "Whether to build unit tests" FORCE)
    set(BUILD_FUZZ_TESTS OFF CACHE BOOL "Whether to build fuzz tests" FORCE)
    set(BUILD_BRPC_TOOLS OFF CACHE BOOL "Whether to build brpc tools" FORCE)
    set(DOWNLOAD_GTEST OFF CACHE BOOL "Download and build a fresh copy of googletest." FORCE)

    add_subdirectory(3rdparty/brpc EXCLUDE_FROM_ALL)
    file(WRITE "${CMAKE_BINARY_DIR}/3rdparty/brpc/CTestTestfile.cmake" "")

    if(NOT TARGET brpc-static)
        message(FATAL_ERROR "Bundled brpc build did not produce the expected brpc-static target.")
    endif()

    add_library(localllm_brpc INTERFACE)
    add_library(localllm::brpc ALIAS localllm_brpc)
    target_link_libraries(localllm_brpc INTERFACE brpc-static)
    target_include_directories(
        localllm_brpc
        INTERFACE
            ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/brpc/src
            ${CMAKE_CURRENT_BINARY_DIR}/3rdparty/brpc
    )
endif()

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

target_link_libraries(localllm_lib PUBLIC ${TORCH_LIBRARIES} glog::glog)
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
target_link_libraries(localllm PRIVATE localllm_lib glog::glog)
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
        test/test_main.cpp
        test/logging_test.cpp
        test/qwen_config_test.cpp
        test/safetensors_loader_test.cpp
        test/bpe_tokenizer_test.cpp
        test/sampling_test.cpp
        test/qwen_executor_smoke_test.cpp
    )
    target_link_libraries(localllm_tests PRIVATE localllm_lib gtest)
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
