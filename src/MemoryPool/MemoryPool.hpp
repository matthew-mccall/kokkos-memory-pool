//
// Created by Matthew McCall on 5/22/23.
//

#ifndef KOKKOS_MEMORY_POOL_MEMORYPOOL_HPP
#define KOKKOS_MEMORY_POOL_MEMORYPOOL_HPP

#include <cstddef>
#include <map>
#include <optional>
#include <utility>

#include "Kokkos_Core.hpp"

constexpr size_t DEFAULT_CHUNK_SIZE = 128;

class MemoryPool {

    struct Chunk {
        std::optional<size_t> next;
        std::byte data[DEFAULT_CHUNK_SIZE - sizeof(Chunk*)];
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
        if (!freeList) {
            return {};
        }

        // Find the smallest sequence of chunks that can hold numElements
        size_t requestedSize = numElements * sizeof(DataType);

        std::optional<size_t> current = freeList;
        size_t beforeBeginSequence = *freeList; // The chunk before the current sequence
        size_t beginSequence = *freeList; // The first chunk in the current sequence
        size_t currentSize = 0;

        while (current) {
            currentSize += DEFAULT_CHUNK_SIZE;
            auto next = pool(*current).next;
            if (currentSize >= requestedSize) {
                auto allocatedBlocksIndices = std::make_pair(beginSequence, *current + 1); // [begin, end)
                auto subview = Kokkos::subview(pool, allocatedBlocksIndices);

                if (beginSequence == *freeList) {
                    freeList = next;
                } else {
                    pool(beforeBeginSequence).next = next;
                }

                Chunk* beginChunk = subview.data();
                assert(beginChunk == &pool(beginSequence));

                allocations[beginChunk] = allocatedBlocksIndices;
                return Kokkos::View<DataType*>(reinterpret_cast<DataType *>(beginChunk), numElements);
            }

            // If the next chunk is not the next chunk in the pool, then the current sequence is broken
            if (next && *next != *current + 1) {
                currentSize = 0;
                beforeBeginSequence = *current;
                beginSequence = *next;
            }

            current = next;
        }

        return {};
    }

    template<typename DataType>
    void deallocate(Kokkos::View<DataType*> view) {
        auto* beginChunk = reinterpret_cast<Chunk*>(view.data());

        assert(allocations.find(beginChunk) != allocations.end());
        auto [beginIndex, endIndex] = allocations[beginChunk]; // [begin, end)

        Kokkos::parallel_for("MemoryPool::deallocate", endIndex - beginIndex, [beginIndex = beginIndex, this](size_t i) { // Apple Clang has issues with capturing structured bindings
            pool(beginIndex + i) = Chunk();
            pool(beginIndex + i).next = beginIndex + i + 1; // Rebuild chunks and list structure
        });

        if (!freeList) { // If the free list is empty, then the beginIndex is the new free list
            freeList = beginIndex;
            pool(endIndex - 1).next = std::nullopt;
            return;
        }

        if (beginIndex < *freeList) { // If the beginIndex is less than the free list, then the beginIndex is the new free list
            pool(endIndex - 1).next = freeList;
            freeList = beginIndex;
            return;
        }

        std::optional<size_t> current = freeList;
        size_t last = *freeList;

        while (current && *current < beginIndex) { // Find the chunk before the beginIndex
            last = *current;
            current = pool(*current).next;
        }

        pool(last).next = beginIndex;
        pool(endIndex - 1).next = current;
    }

private:
    Kokkos::View<MemoryPool::Chunk*> pool;
    std::optional<size_t> freeList;
    std::map<Chunk*, std::pair<size_t, size_t>> allocations;
};

#endif //KOKKOS_MEMORY_POOL_MEMORYPOOL_HPP
