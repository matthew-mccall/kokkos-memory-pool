//
// Created by Matthew McCall on 5/22/23.
//

#include "MemoryPool.hpp"

MemoryPool::MemoryPool(size_t numChunks) : pool("pool", numChunks) {
    freeList = 0;

    Kokkos::parallel_for("MemoryPool::MemoryPool", numChunks - 1, KOKKOS_LAMBDA(size_t i) {
        pool(i).next = i + 1;
    });

    pool(numChunks - 1).next = std::nullopt;
    Kokkos::fence();
}

