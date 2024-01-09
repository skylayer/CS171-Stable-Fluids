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
find_package(glad CONFIG REQUIRED)
find_package(glfw3 CONFIG REQUIRED)

add_library(kernels
        kernels/solver.cu
        fluid.cu
        fluid.cuh
        kernels/render.cu
        kernels/utils.cu
        kernels/utils.cuh
)

# Set CUDA PIE
target_compile_options(kernels PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler -fPIE>)

# Set CUDA separate compilation to support device and host code in the same file
set_target_properties(kernels PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(kernels PRIVATE fmt::fmt)

# Add the executable
add_executable(RtStableFluids
        solver.cpp
        main.cpp
        fluid.cpp
)

include_directories(${PROJECT_SOURCE_DIR})

# Link libraries
target_link_libraries(RtStableFluids PRIVATE
        fmt::fmt
        VTK::CommonCore VTK::CommonDataModel VTK::IOXML
        Eigen3::Eigen
        kernels
)

add_executable(opengl
        gl.cu
)

target_link_libraries(opengl PRIVATE
        fmt::fmt
        glad::glad
        glfw
        Eigen3::Eigen
        kernels
)
