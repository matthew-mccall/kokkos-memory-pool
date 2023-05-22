cmake_minimum_required(VERSION 3.25)
project(kokkos_memory_pool)

add_subdirectory(libs/kokkos)
include_directories(${Kokkos_INCLUDE_DIRS_RET})


set(CMAKE_CXX_STANDARD 17)

add_executable(kokkos_memory_pool main.cpp)
target_link_libraries(kokkos_memory_pool Kokkos::kokkos)
