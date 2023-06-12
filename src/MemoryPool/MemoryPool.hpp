//
// Created by Matthew McCall on 5/22/23.
//

#ifndef KOKKOS_MEMORY_POOL_MEMORYPOOL_HPP
#define KOKKOS_MEMORY_POOL_MEMORYPOOL_HPP

#include <cstddef>
#include <list>
#include <map>
#include <set>
#include <utility>
#include <ostream>

#include "Kokkos_Core.hpp"

using IndexPair = std::pair<size_t, size_t>;
using FreeListT = std::list<IndexPair>;

class CompareFreeIndices {
public:
    using is_transparent = void; // https://www.fluentcpp.com/2017/06/09/search-set-another-type-key/

    bool operator()(IndexPair lhs, IndexPair rhs) const;
    bool operator()(IndexPair lhs, size_t rhs) const;
    bool operator()(size_t lhs, IndexPair rhs) const;
};


using MultiSetBySizeT = std::multiset<IndexPair, CompareFreeIndices>;
using SetByIndexT = std::set<IndexPair>;

class MemoryPool {
public:
    explicit MemoryPool(size_t numChunks);

    uint8_t* allocate(size_t n);
    void deallocate(uint8_t* data);

    friend std::ostream &operator<<(std::ostream &os, const MemoryPool &pool);

    unsigned getNumAllocations() const;
    unsigned getNumFreeChunks() const;
    unsigned getNumAllocatedChunks() const;
    unsigned getNumChunks() const;
    unsigned getNumFreeFragments() const;

    static constexpr size_t DEFAULT_CHUNK_SIZE = 128;
    static size_t getRequiredChunks(size_t n);

private:
    std::pair<MultiSetBySizeT::iterator, SetByIndexT::iterator> insertIntoSets(IndexPair indices);
    void removeFromSets(IndexPair indices);

    Kokkos::View<uint8_t*> pool;
    MultiSetBySizeT freeSetBySize; // For finding free chunks logarithmically
    SetByIndexT freeSetByIndex; // For merging adjacent free chunks
    std::map<uint8_t*, IndexPair> allocations;
};

class MultiPool {
public:
    explicit MultiPool(size_t initialChunks);

    uint8_t* allocate(size_t n);
    void deallocate(uint8_t* data);

    template<typename DataType>
    Kokkos::View<DataType*> allocateView(size_t n) {
        return Kokkos::View<DataType*>(reinterpret_cast<DataType*>(allocate(n * sizeof(DataType))), n);
    }

    template<typename DataType>
    void deallocateView(Kokkos::View<DataType*> view) {
        deallocate(reinterpret_cast<uint8_t*>(view.data()));
    }

    friend std::ostream &operator<<(std::ostream &os, const MultiPool &pool);

    unsigned getNumAllocations() const;
    unsigned getNumFreeChunks() const;
    unsigned getNumAllocatedChunks() const;
    unsigned getNumChunks() const;
    unsigned getNumFreeFragments() const;
    size_t getChunkSize() const;

private:
    using PoolListT = std::list<MemoryPool>;

    PoolListT pools;
    std::map<uint8_t*, PoolListT::iterator> allocations;
};

#endif //KOKKOS_MEMORY_POOL_MEMORYPOOL_HPP
