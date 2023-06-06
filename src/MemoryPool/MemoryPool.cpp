//
// Created by Matthew McCall on 5/22/23.
//
#include <vector>

#include "MemoryPool.hpp"

bool CompareFreeIndices::operator()(FreeListT::iterator lhs, FreeListT::iterator rhs) const {
    // Sort by size then by type
    auto [lhsStart, lhsEnd] = *lhs;
    auto [rhsStart, rhsEnd] = *rhs;
    size_t lhsSize = lhsEnd - lhsStart;
    size_t rhsSize = rhsEnd - rhsStart;

    if (lhsSize == rhsSize) {
        return *lhs < *rhs;
    }

    return lhsSize < rhsSize;
}

bool CompareFreeIndices::operator()(FreeListT::iterator lhs, size_t rhs) const {
    auto [lhsStart, lhsEnd] = *lhs;
    return (lhsEnd - lhsStart) < rhs;
}

bool CompareFreeIndices::operator()(size_t lhs, FreeListT::iterator rhs) const {
    auto [rhsStart, rhsEnd] = *rhs;
    return lhs < (rhsEnd - rhsStart);
}

bool CompareFreeIndices::operator()(FreeListT::iterator lhs, IndexPair rhs) const {
    auto [lhsStart, lhsEnd] = *lhs;
    auto [rhsStart, rhsEnd] = rhs;
    size_t lhsSize = lhsEnd - lhsStart;
    size_t rhsSize = rhsEnd - rhsStart;

    if (lhsSize == rhsSize) {
        return *lhs < rhs;
    }

    return lhsSize < rhsSize;}

bool CompareFreeIndices::operator()(IndexPair lhs, FreeListT::iterator rhs) const {
    auto [lhsStart, lhsEnd] = lhs;
    auto [rhsStart, rhsEnd] = *rhs;
    size_t lhsSize = lhsEnd - lhsStart;
    size_t rhsSize = rhsEnd - rhsStart;

    if (lhsSize == rhsSize) {
        return lhs < *rhs;
    }

    return lhsSize < rhsSize;
}

MemoryPool::MemoryPool(size_t numChunks) : pool("Memory Pool", numChunks * DEFAULT_CHUNK_SIZE) {
    auto initialChunkIndices = std::make_pair(0, numChunks);

    freeList.emplace_back(initialChunkIndices);
    freeSetBySize.insert(freeList.begin());
}

uint8_t *MemoryPool::allocate(size_t n) {
    if (freeList.empty()) {
        return {};
    }

    // Find the smallest sequence of chunks that can hold numElements
    size_t requestedChunks = getRequiredChunks(n);

    auto freeSetItr = freeSetBySize.lower_bound(requestedChunks);
    if (freeSetItr == freeSetBySize.end()) {
        return nullptr;
    }

    auto freeListItr = *freeSetItr;
    auto [beginIndex, endIndex] = *freeListItr;

    freeSetBySize.erase(freeSetItr);

    if (endIndex - beginIndex == requestedChunks) {
        freeList.erase(freeListItr);
    } else {
        freeListItr->first += requestedChunks;
        freeSetBySize.insert(freeListItr);
    }

    uint8_t* ptr = pool.data() + (beginIndex * DEFAULT_CHUNK_SIZE);
    allocations[ptr] = std::make_pair(beginIndex, beginIndex + requestedChunks);

    return ptr;
}

void MemoryPool::deallocate(uint8_t *data) {
    auto allocationsItr = allocations.find(data);
    assert(allocationsItr != allocations.end());
    auto [ptr, chunkIndices] = *allocationsItr; // [begin, end)

    allocations.erase(allocationsItr);

    auto current = freeList.begin();
    while (current != freeList.end() && current->first < chunkIndices.second) {
        current++;
    }

    auto freeListItr = freeList.insert(current, chunkIndices);
    freeSetBySize.insert(freeListItr);

    // Merge adjacent free chunks
    current = freeList.begin();
    while (current != freeList.end()) {
        auto next = current;
        next++;

        if (next != freeList.end() && current->second == next->first) {
            auto freeSetCurrentItr = freeSetBySize.find(*current);
            auto freeSetNextItr = freeSetBySize.find(*next);

            assert(freeSetCurrentItr != freeSetBySize.end());
            assert(freeSetNextItr != freeSetBySize.end());

            current->second = next->second;

            freeSetBySize.erase(freeSetCurrentItr);
            freeSetBySize.erase(freeSetNextItr);

            freeSetBySize.insert(current);

            current = freeList.erase(next);
            continue;
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

    os << "\nFree List: ";

    for (const auto& [beginIndex, endIndex] : pool.freeList) {
        os << "[" << beginIndex << ", " << endIndex << ") ";
    }

    os << "\nFree Set:  ";

    for (const auto itr : pool.freeSetBySize) {
        auto [beginIndex, endIndex] = *itr;
        os << "[" << beginIndex << ", " << endIndex << ") ";
    }

    os << "\n";

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

size_t MemoryPool::getRequiredChunks(size_t n) {
    return (n / DEFAULT_CHUNK_SIZE) + (n % DEFAULT_CHUNK_SIZE ? 1 : 0);
}

size_t MultiPool::getChunkSize() const {
    return MemoryPool::DEFAULT_CHUNK_SIZE;
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

    pools.emplace_back((mostAmountOfChunks * 2) + MemoryPool::getRequiredChunks(n));
    uint8_t* ptr = pools.back().allocate(n);
    allocations[ptr] = --pools.end();

    return ptr;
}

void MultiPool::deallocate(uint8_t *data) {
    allocations.at(data)->deallocate(data);
    auto itr = allocations.find(data);
    allocations.erase(itr);
}

std::ostream &operator<<(std::ostream &os, const MultiPool &multiPool) {
    for (const auto& pool : multiPool.pools) {
        os << pool << ' ';
    }

    return os;
}

unsigned MultiPool::getNumAllocations() const {
    return allocations.size();
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
