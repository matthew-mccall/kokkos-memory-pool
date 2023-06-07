# kokkos-memory-pool [![CMake](https://github.com/matthew-mccall/kokkos-memory-pool/actions/workflows/cmake.yml/badge.svg)](https://github.com/matthew-mccall/kokkos-memory-pool/actions/workflows/cmake.yml)
A implementation of a memory pool in Kokkos.

### Build
1. Clone the repository.
2. Make sure to update the submodules with `git submodule init --update --recursive`.
3. Generate build files. For example, `cmake -B build .` (Note: If you are using the submoduled version of Kokkos, include [configuration options for Kokkos](https://kokkos.github.io/kokkos-core-wiki/ProgrammingGuide/Compiling.html#) here)
4. Build `cmake --build build`
5. Run the tests `cd build && ctest`
6. To run the benchmarks `./kokkos_memory_pool "[\!benchmark]"`
7. For CSV output when running the benchmarks `./kokkos_memory_pool "[\!benchmark]" --success --reporter csv`
