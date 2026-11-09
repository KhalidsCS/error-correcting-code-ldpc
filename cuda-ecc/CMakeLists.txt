cmake_minimum_required(VERSION 3.24)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

# Must come before project(), which otherwise defaults single-config MSVC builds to Debug.
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type")
endif()

project(cuda_ecc LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_library(ldpc_dataset STATIC ldpc_dataset.cpp)
target_include_directories(ldpc_dataset PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(make_dataset make_dataset.cu)
target_link_libraries(make_dataset PRIVATE ldpc_dataset)

add_executable(ldpc_cuda
    main.cu
    ldpc.cu
)
target_link_libraries(ldpc_cuda PRIVATE ldpc_dataset)

add_executable(ldpc_cpu ldpc_cpu.cpp)
target_link_libraries(ldpc_cpu PRIVATE ldpc_dataset)
