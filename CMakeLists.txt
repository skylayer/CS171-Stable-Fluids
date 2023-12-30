cmake_minimum_required(VERSION 3.27)
MESSAGE(STATUS "Using toolchain file: ${CMAKE_TOOLCHAIN_FILE}")

project(RtStableFluids CUDA)
set(CMAKE_CUDA_STANDARD 17)

find_package(Eigen3 CONFIG REQUIRED)
find_package(fmt CONFIG REQUIRED)

add_executable(RtStableFluids main.cu)
target_link_libraries(RtStableFluids PRIVATE Eigen3::Eigen)
target_link_libraries(RtStableFluids PRIVATE fmt::fmt)


set_target_properties(RtStableFluids PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
