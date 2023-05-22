#include <iostream>

#include "MemoryPool/MemoryPool.hpp"

int main(int argc, char** argv) {
    Kokkos::ScopeGuard scopeGuard(argc, argv);

    MemoryPool pool(10);

    auto a = pool.allocate<int>(10);
    a(0) = 69;
    a(sizeof(std::optional<int>)  / sizeof(int)) = 0xdead;
    a((sizeof(std::optional<int>)  / sizeof(int))+ 1) = 0xcafe;
    a((sizeof(std::optional<int>)  / sizeof(int))+ 2) = 0xbeef;

    auto b = pool.allocate<int>(10);

    std::cout << "Hello, World!" << std::endl;
    return 0;
}
