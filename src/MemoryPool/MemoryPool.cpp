//
// Created by Matthew McCall on 5/22/23.
//
#include <vector>

#include "MemoryPool.hpp"

MemoryPool::MemoryPool(size_t numChunks) : pool("pool", numChunks) {
    freeList = 0;

    Kokkos::parallel_for("MemoryPool::MemoryPool", numChunks - 1, KOKKOS_LAMBDA(int32_t i) {
        pool(i).next = i + 1;
    });

    pool(numChunks - 1).next = -1;
    Kokkos::fence();
}

void MemoryPool::print() {
    std::vector<bool> used(pool.size(), false);

    for (const auto& [ptr, indices]: allocations) {
        for (size_t i = indices.first; i < indices.second; i++) {
            used[i] = true;
        }
    }

    for (auto i : used) {
        std::cout << (i ? "X" : "-");
    }

    std::cout << std::endl;
}

std::ostream &operator<<(std::ostream &os, const MemoryPool &pool) {
    std::vector<bool> used(pool.pool.size(), false);

    for (const auto& [ptr, indices]: pool.allocations) {
        for (size_t i = indices.first; i < indices.second; i++) {
            used[i] = true;
        }
    }

    for (auto i : used) {
        os << (i ? "X" : "-");
    }

    os << std::endl;
    return os;
}

