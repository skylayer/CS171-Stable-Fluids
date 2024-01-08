cmake_minimum_required(VERSION 3.27)
project(RtStableFluids LANGUAGES CXX CUDA)

# Set the C++ and CUDA standard versions
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Output toolchain file information
message(STATUS "Using toolchain file: ${CMAKE_TOOLCHAIN_FILE}")

# Find required packages
find_package(fmt CONFIG REQUIRED)
find_package(VTK CONFIG REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)

# Add the executable
add_executable(RtStableFluids
        solver.cpp
        main.cpp
        fluid.cpp
        kernels/solver.cu
        fluid.cu
        fluid.cuh
        kernels/render.cu
        kernels/utils.cu
        kernels/utils.cuh
)

include_directories(${PROJECT_SOURCE_DIR})

# Link libraries
target_link_libraries(RtStableFluids PRIVATE
        fmt::fmt
        VTK::CommonCore VTK::CommonDataModel VTK::IOXML
        Eigen3::Eigen
)

# Set CUDA separate compilation to support device and host code in the same file
set_target_properties(RtStableFluids PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Set CUDA PIE
target_compile_options(RtStableFluids PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler -fPIC>)
