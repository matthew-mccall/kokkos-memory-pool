cmake_minimum_required(VERSION 3.25)
project(kokkos_memory_pool)

add_subdirectory(libs/kokkos)
include_directories(${Kokkos_INCLUDE_DIRS_RET})

add_subdirectory(libs/Catch2)

set(CMAKE_CXX_STANDARD 17)

add_executable(kokkos_memory_pool src/MemoryPool/MemoryPool.cpp src/MemoryPool/MemoryPool.hpp test/test.cpp)
target_include_directories(kokkos_memory_pool PRIVATE ${Kokkos_INCLUDE_DIRS_RET} src)
target_link_libraries(kokkos_memory_pool PRIVATE Kokkos::kokkos Catch2::Catch2)

include(CTest)
include(Catch)

catch_discover_tests(kokkos_memory_pool)