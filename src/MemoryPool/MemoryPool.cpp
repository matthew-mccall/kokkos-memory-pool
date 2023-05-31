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

unsigned MemoryPool::getNumAllocations() const {
    return allocations.size();
}

unsigned MemoryPool::getNumFreeChunks() const {
    unsigned numFreeChunks = 0;

    for (const auto& [beginIndex, endIndex] : freeList) {
        numFreeChunks += endIndex - beginIndex;
    }

    return numFreeChunks;
}

unsigned MemoryPool::getNumAllocatedChunks() const {
    unsigned numAllocatedChunks = 0;

    for (const auto& [ptr, indices] : allocations) {
        numAllocatedChunks += indices.second - indices.first;
    }

    return numAllocatedChunks;
}

unsigned MemoryPool::getNumChunks() const {
    return pool.size() / DEFAULT_CHUNK_SIZE;
}

std::pair<uint32_t, uint32_t>
MemoryPool::chunkIndicesToBytes(std::pair<uint32_t, uint32_t> chunkIndices) {
    return std::make_pair(chunkIndices.first * DEFAULT_CHUNK_SIZE, chunkIndices.second * DEFAULT_CHUNK_SIZE);
}

