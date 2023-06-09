//
// Created by Matthew McCall on 5/22/23.
//
#include <cassert>
#include <iterator>
#include <vector>

#include "MemoryPool.hpp"

bool CompareFreeIndices::operator()(IndexPair lhs, IndexPair rhs) const {
    auto [lhsStart, lhsEnd] = lhs;
    auto [rhsStart, rhsEnd] = rhs;
    size_t lhsSize = lhsEnd - lhsStart;
    size_t rhsSize = rhsEnd - rhsStart;

    if (lhsSize == rhsSize) {
        return lhs < rhs;
    }

    return lhsSize < rhsSize;
}

bool CompareFreeIndices::operator()(IndexPair lhs, size_t rhs) const {
    auto [lhsStart, lhsEnd] = lhs;
    return (lhsEnd - lhsStart) < rhs;
}

bool CompareFreeIndices::operator()(size_t lhs, IndexPair rhs) const {
    auto [rhsStart, rhsEnd] = rhs;
    return lhs < (rhsEnd - rhsStart);
}

MemoryPool::MemoryPool(size_t numChunks) : pool("Memory Pool", numChunks * DEFAULT_CHUNK_SIZE) {
    insertIntoSets({0, numChunks});
}

std::pair<MultiSetBySizeT::iterator, SetByIndexT::iterator> MemoryPool::insertIntoSets(IndexPair indices) {
    auto setBySizeItr = freeSetBySize.insert(indices);
    auto [setByIndexItr, inserted] = freeSetByIndex.insert(indices);
    
    assert(inserted);

    return {setBySizeItr, setByIndexItr};
}

void MemoryPool::removeFromSets(IndexPair indices) {
    freeSetBySize.erase(indices);
    freeSetByIndex.erase(indices);
}

uint8_t *MemoryPool::allocate(size_t n) {
    if (freeSetBySize.empty()) {
        return {};
    }

    // Find the smallest sequence of chunks that can hold numElements
    size_t requestedChunks = getRequiredChunks(n);

    auto freeSetItr = freeSetBySize.lower_bound(requestedChunks);
    if (freeSetItr == freeSetBySize.end()) {
        return nullptr;
    }

    auto [beginIndex, endIndex] = *freeSetItr;

    removeFromSets(*freeSetItr);

    if (endIndex - beginIndex != requestedChunks) {
        insertIntoSets({beginIndex + requestedChunks, endIndex});
    }

    uint8_t* ptr = pool.data() + (beginIndex * DEFAULT_CHUNK_SIZE);
    allocations[ptr] = std::make_pair(beginIndex, beginIndex + requestedChunks);

    return ptr;
}

void MemoryPool::deallocate(uint8_t *data) {
    auto allocationsItr = allocations.find(data);
    assert(allocationsItr != allocations.end());
    auto [ptr, chunkIndices] = *allocationsItr; // [begin, end)

    auto [freeSetBySizeItr, freeSetByIndexItr] = insertIntoSets(chunkIndices);
    allocations.erase(allocationsItr);

    // Merge adjacent free chunks
    if (freeSetByIndexItr != freeSetByIndex.begin()) {
        auto prevItr = std::prev(freeSetByIndexItr);
        auto [prevBeginIndex, prevEndIndex] = *prevItr;

        if (prevEndIndex == chunkIndices.first) {
            removeFromSets(*prevItr);
            removeFromSets(chunkIndices);
            freeSetByIndexItr = insertIntoSets({prevBeginIndex, chunkIndices.second}).second;
        }
    }

    if (std::next(freeSetByIndexItr) != freeSetByIndex.end()) {
        auto nextItr = std::next(freeSetByIndexItr);
        auto [nextBeginIndex, nextEndIndex] = *nextItr;
        auto [beginIndex, endIndex] = *freeSetByIndexItr;

        if (chunkIndices.second == nextBeginIndex) {
            removeFromSets(*freeSetByIndexItr);
            removeFromSets(*nextItr);
            insertIntoSets({beginIndex, nextEndIndex});
        }
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

    os << "\nFree Set:  ";

    for (const auto [beginIndex, endIndex] : pool.freeSetBySize) {
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

    for (const auto& [beginIndex, endIndex] : freeSetBySize) {
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

unsigned MemoryPool::getNumFreeFragments() const {
    return freeSetBySize.size();
}

size_t MemoryPool::getRequiredChunks(size_t n) {
    return (n / DEFAULT_CHUNK_SIZE) + (n % DEFAULT_CHUNK_SIZE ? 1 : 0);
}

size_t MultiPool::getChunkSize() const {
    return MemoryPool::DEFAULT_CHUNK_SIZE;
}

unsigned MultiPool::getNumFreeFragments() const {
    unsigned numFreeFragments = 0;

    for (const auto& pool : pools) {
        numFreeFragments += pool.getNumFreeFragments();
    }

    return numFreeFragments;
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
