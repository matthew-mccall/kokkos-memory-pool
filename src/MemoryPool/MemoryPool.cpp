//
// Created by Matthew McCall on 5/22/23.
//
#include <vector>

#include "MemoryPool.hpp"

MemoryPool::MemoryPool(size_t numChunks) : pool("pool", numChunks * DEFAULT_CHUNK_SIZE) {
    freeList.emplace_back(0, numChunks);
}

std::ostream &operator<<(std::ostream &os, const MemoryPool &pool) {
    std::vector<bool> used(pool.pool.size(), false);

    for (const auto& [ptr, indices]: pool.allocations) {
        for (int i = indices.first; i < indices.second; i++) {
            used[i] = true;
        }
    }

    for (auto i : used) {
        os << (i ? "X" : "-");
    }

    os << std::endl;
    return os;
}

std::pair<uint32_t, uint32_t>
MemoryPool::chunkIndicesToBytes(std::pair<uint32_t, uint32_t> chunkIndices) {
    return std::make_pair(chunkIndices.first * DEFAULT_CHUNK_SIZE, chunkIndices.second * DEFAULT_CHUNK_SIZE);
}

