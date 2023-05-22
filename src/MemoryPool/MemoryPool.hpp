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

constexpr size_t DEFAULT_CHUNK_SIZE = 1024;

class MemoryPool {

    struct Chunk {
        std::optional<size_t> next;
        std::byte data[DEFAULT_CHUNK_SIZE - sizeof(Chunk*)];
    };

public:
    explicit MemoryPool(size_t numChunks);

    template<typename DataType>
    Kokkos::View<DataType*> allocate(size_t numElements) {
        // Find the smallest sequence of chunks that can hold numElements
        size_t requestedSize = numElements * sizeof(DataType);

        // TODO: What if the pool is full?
        std::optional<size_t> current = freeList;
        size_t last = freeList.value_or(0); // The chunk before (that pointed to) the current chunk
        size_t beforeBeginSequence = freeList.value_or(0); // The chunk before the current sequence
        size_t beginSequence = freeList.value_or(0); // The first chunk in the current sequence
        size_t currentSize = 0;

        while (current) {
            currentSize += DEFAULT_CHUNK_SIZE;
            if (currentSize >= requestedSize) {
                auto allocatedBlocksIndices = std::make_pair(beginSequence, *current + 1);
                auto subview = Kokkos::subview(pool, allocatedBlocksIndices);

                if (beginSequence == 0) {
                    freeList = *pool(*current).next;
                } else {
                    pool(beforeBeginSequence).next = *pool(*current).next;
                }

                Chunk* beginChunk = subview.data();
                assert(beginChunk == &pool(beginSequence));

                allocations[beginChunk] = allocatedBlocksIndices;
                return Kokkos::View<DataType*>(reinterpret_cast<DataType *>(beginChunk), numElements);
            }

            // If the next chunk is not the next chunk in the pool, then the current sequence is broken
            if (*pool(*current).next != *current + 1) {
                currentSize = 0;
                beforeBeginSequence = *current;
                beginSequence = *pool(*current).next;
            }

            last = *current;
            current = pool(*current).next;
        }

        // If we get here, then we need to allocate more chunks
        size_t numChunks = (requestedSize / DEFAULT_CHUNK_SIZE) + 1;
        Kokkos::resize(pool, pool.extent(0) + numChunks);

        Kokkos::parallel_for("MemoryPool::allocate", numChunks - 1, KOKKOS_LAMBDA(size_t i) {
            pool(last + i).next = i + 1;
        });

        pool(last + numChunks - 1).next = std::nullopt;

        auto allocatedBlocksIndices = std::make_pair(last, last + numChunks);
        auto subview = Kokkos::subview(pool, Kokkos::make_pair(last, last + numChunks));

        if (last == 0) {
            freeList = *pool(last).next;
        }

        Chunk* beginChunk = subview.data();
        assert(beginChunk == &pool(beginSequence));

        allocations[beginChunk] = allocatedBlocksIndices;
        return Kokkos::View<DataType*>(reinterpret_cast<DataType *>(beginChunk), numElements);
    }

    template<typename DataType>
    void deallocate(Kokkos::View<DataType*> view) {
        auto* beginChunk = reinterpret_cast<Chunk*>(view.data());

        assert(allocations.find(beginChunk) != allocations.end());
        auto allocatedBlocksIndices = allocations[beginChunk];

        Kokkos::parallel_for("MemoryPool::deallocate", allocatedBlocksIndices.second - allocatedBlocksIndices.first, KOKKOS_LAMBDA(size_t i) {
            pool(allocatedBlocksIndices.first + i) = Chunk();
            pool(allocatedBlocksIndices.first + i).next = allocatedBlocksIndices.first + i + 1;
        });

        if (!freeList) {
            freeList = allocatedBlocksIndices.first;
            pool(allocatedBlocksIndices.second - 1).next = std::nullopt;
            return;
        }

        if (allocatedBlocksIndices.first < *freeList) {
            pool(allocatedBlocksIndices.second - 1).next = freeList;
            freeList = allocatedBlocksIndices.first;
            return;
        }

        std::optional<size_t> current = freeList;
        size_t last = *freeList;
        while (*current < allocatedBlocksIndices.first && current) {
            last = *current;
            current = *pool(current).next;
        }
    }

private:
    Kokkos::View<MemoryPool::Chunk*> pool;
    std::optional<size_t> freeList;
    std::map<Chunk*, std::pair<size_t, size_t>> allocations;
};

#endif //KOKKOS_MEMORY_POOL_MEMORYPOOL_HPP
