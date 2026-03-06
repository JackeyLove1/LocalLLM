cmake_minimum_required(VERSION 3.10)

project(LocalLLM CXX C CUDA)
option(USE_CUDA "Use CUDA" ON)
option(DEBUG "Debug Mode" ON)


message(STATUS "CUDA: ${USE_CUDA}")
message(STATUS "CUDA Architecture: ${CUDA_ARCH}")
message(STATUS "Debug Mode: ${DEBUG}")

if(NOT DEFINED CUDA_ARCH)
    set(CUDA_ARCH "native")
endif()

if (USE_CUDA)
    enable_language(CUDA)
    add_compile_definitions(USE_CUDA)
    set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH})
endif()

if(DEBUG)
    add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-G;-g;-lineinfo>)
endif()


set(CMAKE_PREFIX_PATH "/opt/libtorch")
find_package(Torch REQUIRED)
find_package(GTest REQUIRED)
find_package(OpenMP REQUIRED)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()


include_directories(src)
include_directories(include)
include_directories(3rdparty/flashinfer)
include_directories(3rdparty/gguf)
include_directories(3rdparty/json11)
include_directories(3rdparty/pybind11)

add_executable(
    localllm main.cpp
)


target_include_directories(localllm
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)