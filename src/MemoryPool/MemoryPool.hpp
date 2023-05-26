//
// Created by Matthew McCall on 5/22/23.
//

#ifndef KOKKOS_MEMORY_POOL_MEMORYPOOL_HPP
#define KOKKOS_MEMORY_POOL_MEMORYPOOL_HPP

#include <cstddef>
#include <list>
#include <map>
#include <utility>
#include <ostream>

#include "Kokkos_Core.hpp"

constexpr size_t DEFAULT_CHUNK_SIZE = 128;

class MemoryPool {

    struct Chunk {
        uint8_t data[DEFAULT_CHUNK_SIZE];
    };

public:
    explicit MemoryPool(size_t numChunks);

    /**
     * Allocates a view of DataType of size numElements
     * @tparam DataType The type of the view to allocate
     * @param numElements The number of elements of DataType to allocate
     * @return A view of DataType of size numElements or an empty view if there is not enough space
     */
    template<typename DataType>
    Kokkos::View<DataType*> allocate(size_t numElements) {
        if (freeList.empty()) {
            return {};
        }

        // Find the smallest sequence of chunks that can hold numElements
        size_t requestedSize = numElements * sizeof(DataType);

        auto current = freeList.begin();
        auto beginSequence = freeList.begin(); // The first chunk in the current sequence
        size_t currentSize = 0;

        while (current != freeList.end()) {
            currentSize += DEFAULT_CHUNK_SIZE;

            auto next = current;
            next++;

            if (currentSize >= requestedSize) {
                auto allocatedBlocksIndices = std::make_pair(*beginSequence, *current + 1); // [begin, end)
                auto subview = Kokkos::subview(pool, allocatedBlocksIndices);

                Chunk* beginChunk = subview.data();
                auto& allocation = allocations[beginChunk];

                freeList.splice(allocation.end(), freeList, beginSequence, next); // Remove the allocated blocks from the free list
                return Kokkos::View<DataType*>(reinterpret_cast<DataType *>(beginChunk), numElements);
            }

            // If the next chunk is not the next chunk in the pool, then the current sequence is broken
            if ((next != freeList.end()) && *next != *current + 1) {
                currentSize = 0;
                beginSequence = next;
            }

            current = next;
        }

        return {};
    }

    template<typename DataType>
    void deallocate(Kokkos::View<DataType*> view) {
        auto* beginChunk = reinterpret_cast<Chunk*>(view.data());

        auto itr = allocations.find(beginChunk);
        assert(itr != allocations.end());
        auto [ptr, chunks] = *itr; // [begin, end)

        allocations.erase(itr);

        auto current = freeList.begin();

        while (current != freeList.end() && *current < chunks.front()) {
            current++;
        }

        freeList.splice(current, chunks);
    }

    friend std::ostream &operator<<(std::ostream &os, const MemoryPool &pool);

private:
    Kokkos::View<MemoryPool::Chunk*> pool;
    std::list<int32_t> freeList;
    std::map<Chunk*, std::list<int32_t>> allocations;
};

#endif //KOKKOS_MEMORY_POOL_MEMORYPOOL_HPP
