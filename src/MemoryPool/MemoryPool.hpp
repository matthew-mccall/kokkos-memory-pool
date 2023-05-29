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
        uint32_t requestedChunks = (requestedSize / DEFAULT_CHUNK_SIZE);

        if (requestedSize % DEFAULT_CHUNK_SIZE) {
            requestedChunks++;
        }

        auto current = freeList.begin();

        while (current != freeList.end()) {
            auto [beginIndex, endIndex] = *current;

            if (beginIndex + requestedChunks <= endIndex) {
                auto allocatedChunkIndices = std::make_pair(beginIndex, beginIndex + requestedChunks);
                auto subview = Kokkos::subview(pool, allocatedChunkIndices);
                Chunk* beginChunk = subview.data();
                allocations[beginChunk] = allocatedChunkIndices;

                if (endIndex == beginIndex + requestedChunks) {
                    freeList.erase(current);
                } else {
                    current->first = beginIndex + requestedChunks;
                }

                return Kokkos::View<DataType*>(reinterpret_cast<DataType*>(beginChunk), numElements);
            }

            current++;
        }

        return {};
    }

    template<typename DataType>
    void deallocate(Kokkos::View<DataType*> view) {
        auto* beginChunk = reinterpret_cast<Chunk*>(view.data());

        auto itr = allocations.find(beginChunk);
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

    friend std::ostream &operator<<(std::ostream &os, const MemoryPool &pool);

private:
    Kokkos::View<MemoryPool::Chunk*> pool;
    std::list<std::pair<uint32_t, uint32_t>> freeList;
    std::map<Chunk*, std::pair<int32_t, int32_t>> allocations;
};

#endif //KOKKOS_MEMORY_POOL_MEMORYPOOL_HPP
