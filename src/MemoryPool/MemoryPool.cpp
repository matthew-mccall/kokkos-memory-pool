//
// Created by Matthew McCall on 5/22/23.
//
#include <vector>

#include "MemoryPool.hpp"

MemoryPool::MemoryPool(size_t numChunks) : pool("Memory Pool", numChunks * DEFAULT_CHUNK_SIZE) {
    freeList.emplace_back(0, numChunks);
}

uint8_t *MemoryPool::allocate(size_t n) {
    if (freeList.empty()) {
        return {};
    }

    // Find the smallest sequence of chunks that can hold numElements
    size_t requestedChunks = (n / DEFAULT_CHUNK_SIZE);
    if (n % DEFAULT_CHUNK_SIZE) {
        requestedChunks++;
    }

    auto current = freeList.begin();

    while (current != freeList.end()) {
        auto [beginIndex, endIndex] = *current;

        if (beginIndex + requestedChunks <= endIndex) {
            auto allocatedChunkIndices = std::make_pair(beginIndex, beginIndex + requestedChunks);
            uint8_t* beginChunk = Kokkos::subview(pool, Kokkos::pair(beginIndex * DEFAULT_CHUNK_SIZE, (beginIndex + requestedChunks) * DEFAULT_CHUNK_SIZE)).data();
            allocations[beginChunk] = allocatedChunkIndices;

            if (endIndex == beginIndex + requestedChunks) {
                freeList.erase(current);
            } else {
                current->first = allocatedChunkIndices.second;
            }

            return beginChunk;
        }

        current++;
    }

    return nullptr;
}

void MemoryPool::deallocate(uint8_t *data) {
    auto itr = allocations.find(data);
    assert(itr != allocations.end());
    auto [ptr, chunkIndices] = *itr; // [begin, end)

    allocations.erase(itr);

    auto current = freeList.begin();
    while (current != freeList.end() && current->first < chunkIndices.second) {
        current++;
    }

    freeList.insert(current, chunkIndices);

    // Merge adjacent free chunks
    current = freeList.begin();
    while (current != freeList.end()) {
        auto next = current;
        next++;

        if (next != freeList.end() && current->second == next->first) {
            current->second = next->second;
            freeList.erase(next);
        }

        current++;
    }
}

std::ostream &operator<<(std::ostream &os, const MemoryPool &pool) {
    std::vector<bool> used(pool.getNumChunks(), false);

    for (const auto& [ptr, indices]: pool.allocations) {
        for (size_t i = indices.first; i < indices.second; i++) {
            used[i] = true;
        }
    }

    for (auto i : used) {
        os << (i ? "X" : "-");
    }

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

size_t MultiPool::getChunkSize() const {
    return DEFAULT_CHUNK_SIZE;
}

MultiPool::MultiPool(size_t initialChunks) {
    pools.emplace_back(initialChunks);
}

uint8_t *MultiPool::allocate(size_t n) {
    auto current = pools.begin();
    unsigned mostAmountOfChunks = 0;

    while (current != pools.end()) {
        uint8_t* ptr = current->allocate(n);
        if (ptr) {
            allocations[ptr] = current;
            return ptr;
        }

        if (current->getNumChunks() > mostAmountOfChunks) {
            mostAmountOfChunks = current->getNumChunks();
        }

        current++;
    }

    pools.emplace_back((mostAmountOfChunks * 2) + (n / DEFAULT_CHUNK_SIZE) + 1);
    uint8_t* ptr = pools.back().allocate(n);
    allocations[ptr] = --pools.end();

    return ptr;
}

void MultiPool::deallocate(uint8_t *data) {
    auto itr = allocations.find(data);
    assert(itr != allocations.end());
    itr->second->deallocate(data);
    allocations.erase(itr);
}

std::ostream &operator<<(std::ostream &os, const MultiPool &multiPool) {
    for (const auto& pool : multiPool.pools) {
        os << pool << ' ';
    }

    return os;
}

unsigned MultiPool::getNumAllocations() const {
    unsigned numAllocations = 0;

    for (const auto& pool : pools) {
        numAllocations += pool.getNumAllocations();
    }

    return numAllocations;
}

unsigned MultiPool::getNumFreeChunks() const {
    unsigned numFreeChunks = 0;

    for (const auto& pool : pools) {
        numFreeChunks += pool.getNumFreeChunks();
    }

    return numFreeChunks;
}

unsigned MultiPool::getNumAllocatedChunks() const {
    unsigned numAllocatedChunks = 0;

    for (const auto& pool : pools) {
        numAllocatedChunks += pool.getNumAllocatedChunks();
    }

    return numAllocatedChunks;
}

unsigned MultiPool::getNumChunks() const {
    unsigned numChunks = 0;

    for (const auto& pool : pools) {
        numChunks += pool.getNumChunks();
    }

    return numChunks;
}
