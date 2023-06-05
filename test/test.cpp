//
// Created by Matthew McCall on 5/23/23.
//

#include "catch2/catch_session.hpp"
#include "catch2/catch_test_macros.hpp"
#include "catch2/benchmark/catch_benchmark.hpp"
#include "catch2/generators/catch_generators.hpp"

#include "fmt/core.h"

#include "MemoryPool/MemoryPool.hpp"

constexpr size_t TEST_POOL_SIZE = 4;

#define EXPECTED_CHUNKS(DataType) (MemoryPool::getRequiredChunks(sizeof(DataType)))

#define EXPECT_CHUNKS_AND_ALLOCS_IN_POOL(pool, chunks, allocs) \
    REQUIRE(pool.getNumAllocatedChunks() == chunks); \
    REQUIRE(pool.getNumAllocations() == allocs); \
    REQUIRE(pool.getNumFreeChunks() == pool.getNumChunks() - chunks)

struct VeryLargeStruct {
    uint8_t data[DEFAULT_CHUNK_SIZE * TEST_POOL_SIZE];
};

struct LargeStruct {
    uint8_t data[DEFAULT_CHUNK_SIZE * TEST_POOL_SIZE / 2];
};

int main(int argc, char* argv[]) {
    Kokkos::ScopeGuard guard(argc, argv);
    int result = Catch::Session().run(argc, argv);
    return result;
}

TEST_CASE("Memory Pool allocates primitives successfully", "[MemoryPool][allocation][primitives]") {
    MultiPool pool(TEST_POOL_SIZE); // 512 bytes

    auto view = pool.allocateView<int>(1);
    CAPTURE(pool);
    REQUIRE(view.size() == 1);
    EXPECT_CHUNKS_AND_ALLOCS_IN_POOL(pool, 1, 1);

    SECTION("Allocating from a pool with one chunk used") {
        auto view2 = pool.allocateView<int>(1);
        CAPTURE(pool);
        REQUIRE(view2.size() == 1);
        EXPECT_CHUNKS_AND_ALLOCS_IN_POOL(pool, 2, 2);
    }
}

TEST_CASE("Memory Pool allocates custom types successfully", "[MemoryPool][MemoryPool][allocation][structs]") {
    MultiPool pool(TEST_POOL_SIZE); // 512 bytes

    auto view = pool.allocateView<VeryLargeStruct>(1);
    CAPTURE(pool);
    REQUIRE(view.size() == 1);
    EXPECT_CHUNKS_AND_ALLOCS_IN_POOL(pool, TEST_POOL_SIZE, 1);

    SECTION("Allocating from a full pool causes a resize") {
        auto view2 = pool.allocateView<VeryLargeStruct>(1);
        CAPTURE(pool);
        REQUIRE(view2.size() == 1);
        EXPECT_CHUNKS_AND_ALLOCS_IN_POOL(pool, (EXPECTED_CHUNKS(VeryLargeStruct) * 2), 2);
    }
}

TEST_CASE("Memory Pool allocates and deallocates primitives successfully", "[MemoryPool][MemoryPool][allocation][deallocation][primitives]") {
    MultiPool pool(4); // 512 bytes

    auto view = pool.allocateView<int>(1);
    CAPTURE(pool);
    REQUIRE(view.size() == 1);
    EXPECT_CHUNKS_AND_ALLOCS_IN_POOL(pool, 1, 1);

    SECTION("Allocating and deallocating from a new pool") {
        pool.deallocateView(view);
        CAPTURE(pool);
        EXPECT_CHUNKS_AND_ALLOCS_IN_POOL(pool, 0, 0);
    }

    SECTION("Allocating 2 chunks and deallocating the first chunk first") {
        auto view2 = pool.allocateView<int>(1);
        CAPTURE(pool);
        REQUIRE(view2.size() == 1);
        EXPECT_CHUNKS_AND_ALLOCS_IN_POOL(pool, 2, 2);

        pool.deallocateView(view);
        CAPTURE(pool); // -X--
        EXPECT_CHUNKS_AND_ALLOCS_IN_POOL(pool, 1, 1);

        pool.deallocateView(view2);
        CAPTURE(pool); // ----
        EXPECT_CHUNKS_AND_ALLOCS_IN_POOL(pool, 0, 0);
    }

    SECTION("Allocating a large chunk from a pool with one chunk used") {
        auto view2 = pool.allocateView<int>(1);
        CAPTURE(pool);
        REQUIRE(view2.size() == 1);
        EXPECT_CHUNKS_AND_ALLOCS_IN_POOL(pool, 2, 2);

        pool.deallocateView(view);
        CAPTURE(pool); // -X--
        EXPECT_CHUNKS_AND_ALLOCS_IN_POOL(pool, 1, 1);

        auto view3 = pool.allocateView<LargeStruct>(1);
        CAPTURE(pool); // -XXX
        REQUIRE(view3.size() == 1);
        EXPECT_CHUNKS_AND_ALLOCS_IN_POOL(pool, 3, 2);
    }

    SECTION("Allocating 2 chunks and deallocating the second chunk first") {
        auto view2 = pool.allocateView<int>(1);
        CAPTURE(pool);
        REQUIRE(view2.size() == 1);
        EXPECT_CHUNKS_AND_ALLOCS_IN_POOL(pool, 2, 2);

        pool.deallocateView(view2);
        CAPTURE(pool); // X---
        EXPECT_CHUNKS_AND_ALLOCS_IN_POOL(pool, 1, 1);

        pool.deallocateView(view);
        CAPTURE(pool); // ----
        EXPECT_CHUNKS_AND_ALLOCS_IN_POOL(pool, 0, 0);
    }
}

TEST_CASE("Memory Pool allocates and deallocates custom types successfully", "[MemoryPool][allocation][deallocation][structs]") {
    MultiPool pool(4); // 512 bytes

    auto view = pool.allocateView<VeryLargeStruct>(1);
    CAPTURE(pool);
    REQUIRE(view.size() == 1);
    EXPECT_CHUNKS_AND_ALLOCS_IN_POOL(pool, (EXPECTED_CHUNKS(VeryLargeStruct)), 1);

    SECTION("Allocating and deallocating custom type") {
        pool.deallocateView(view);
        CAPTURE(pool);
        EXPECT_CHUNKS_AND_ALLOCS_IN_POOL(pool, 0, 0);
    }

    SECTION("Allocating and deallocating from a full pool causes a resize") {
        auto view2 = pool.allocateView<VeryLargeStruct>(1);
        CAPTURE(pool);
        REQUIRE(view2.size() == 1);
        EXPECT_CHUNKS_AND_ALLOCS_IN_POOL(pool, (EXPECTED_CHUNKS(VeryLargeStruct) * 2), 2);

        pool.deallocateView(view);
        CAPTURE(pool);
        EXPECT_CHUNKS_AND_ALLOCS_IN_POOL(pool, EXPECTED_CHUNKS(VeryLargeStruct), 1);

        SECTION("Allocating from original pool free returns a non-empty view") {
            auto view3 = pool.allocateView<VeryLargeStruct>(1);
            CAPTURE(pool);
            REQUIRE(view3.size() == 1);
            EXPECT_CHUNKS_AND_ALLOCS_IN_POOL(pool, (EXPECTED_CHUNKS(VeryLargeStruct) * 2), 2);
        }
    }

    SECTION("Allocating contiguous chunks from a fragmented pool finds largest contiguous region") {
        auto view2 = pool.allocateView<VeryLargeStruct>(2);
        CAPTURE(pool);
        REQUIRE(view2.size() == 2);
        EXPECT_CHUNKS_AND_ALLOCS_IN_POOL(pool, (EXPECTED_CHUNKS(VeryLargeStruct) * 3), 2);

        pool.deallocateView(view);
        CAPTURE(pool);
        EXPECT_CHUNKS_AND_ALLOCS_IN_POOL(pool, (EXPECTED_CHUNKS(VeryLargeStruct) * 2), 1);

        auto view3 = pool.allocateView<VeryLargeStruct>(2);
        CAPTURE(pool);
        REQUIRE(view3.size() == 2);
        EXPECT_CHUNKS_AND_ALLOCS_IN_POOL(pool, (EXPECTED_CHUNKS(VeryLargeStruct) * 4), 2);
    }
}

TEST_CASE("Pool works under fragmentation", "[MemoryPool][allocation][deallocation][primitives][fragmentation]") {
    MultiPool pool(25); // 512 bytes

    std::vector<Kokkos::View<int[DEFAULT_CHUNK_SIZE / sizeof(int)]>> views(25);

    for (auto& view : views) {
        view = pool.allocateView<int>(DEFAULT_CHUNK_SIZE / sizeof(int));
        REQUIRE(view.size() == DEFAULT_CHUNK_SIZE / sizeof(int));
    }

    CAPTURE(pool);
    EXPECT_CHUNKS_AND_ALLOCS_IN_POOL(pool, 25, 25);

    unsigned step = GENERATE(2, 3, 4, 5);

    for (unsigned i = 0; i < views.size(); i++) {
        if (i % step != 0) {
            CAPTURE(i);
            CAPTURE(step);
            pool.deallocateView<int>(views[i]);
            CAPTURE(pool);
            unsigned expectedChunks = ((i / step) * (step - 1)) + (i % step);
            REQUIRE(pool.getNumFreeChunks() == expectedChunks);
        }
    }
}

TEST_CASE("Benchmarks", "[!benchmark]") {
    constexpr size_t NUMBER_OF_VIEWS = 1'000'000;
    constexpr size_t SIZE_OF_VIEWS = 1024;

    BENCHMARK_ADVANCED(fmt::format("Kokkos Allocating {L} Views of {} ints", NUMBER_OF_VIEWS, SIZE_OF_VIEWS))(Catch::Benchmark::Chronometer meter) {
        std::vector<Kokkos::View<int[SIZE_OF_VIEWS]>> views(NUMBER_OF_VIEWS);

        meter.measure([&] {
            for (auto& view : views) {
                view = Kokkos::View<int[SIZE_OF_VIEWS]>("view", SIZE_OF_VIEWS);
                REQUIRE(view.size() == SIZE_OF_VIEWS);
            }

            return views.size();
        });
    };

    BENCHMARK_ADVANCED(fmt::format("MultiPool Allocation {L} Views of {} ints", NUMBER_OF_VIEWS, SIZE_OF_VIEWS))(Catch::Benchmark::Chronometer meter) {
        const size_t TOTAL_CHUNK_SIZE = MemoryPool::getRequiredChunks(sizeof(int) * SIZE_OF_VIEWS * NUMBER_OF_VIEWS);

        MultiPool pool(TOTAL_CHUNK_SIZE);
        std::vector<Kokkos::View<int[SIZE_OF_VIEWS]>> views(NUMBER_OF_VIEWS);

        meter.measure([&] {
            for (auto& view : views) {
                view = pool.allocateView<int>(SIZE_OF_VIEWS);
                REQUIRE(view.size() == SIZE_OF_VIEWS);
            }

//            EXPECT_CHUNKS_AND_ALLOCS_IN_POOL(pool, TOTAL_CHUNK_SIZE, NUMBER_OF_VIEWS);

            return views.size();
        });
    };

    BENCHMARK_ADVANCED("Allocating under heavy fragmentation")(Catch::Benchmark::Chronometer meter) {
        const size_t BEFORE_FRAGMENTATION_TOTAL_CHUNK_SIZE = MemoryPool::getRequiredChunks(sizeof(int) * SIZE_OF_VIEWS * NUMBER_OF_VIEWS);

        MultiPool pool(BEFORE_FRAGMENTATION_TOTAL_CHUNK_SIZE);
        std::vector<Kokkos::View<int*>> views(NUMBER_OF_VIEWS);

        for (auto& view : views) {
            view = pool.allocateView<int>(SIZE_OF_VIEWS);
            REQUIRE(view.size() == SIZE_OF_VIEWS);
        }

//        EXPECT_CHUNKS_AND_ALLOCS_IN_POOL(pool, BEFORE_FRAGMENTATION_TOTAL_CHUNK_SIZE, NUMBER_OF_VIEWS);

        for (unsigned i = 0; i < NUMBER_OF_VIEWS; i+= 2) {
            pool.deallocateView<int>(views[i]);
        }

//        EXPECT_CHUNKS_AND_ALLOCS_IN_POOL(pool, BEFORE_FRAGMENTATION_TOTAL_CHUNK_SIZE / 2, NUMBER_OF_VIEWS / 2);

        meter.measure([&] {
            for (unsigned i = 0; i < NUMBER_OF_VIEWS; i+= 2) {
                views[i] = pool.allocateView<int>(SIZE_OF_VIEWS);
                REQUIRE(views[i].size() == SIZE_OF_VIEWS);
            }
//            EXPECT_CHUNKS_AND_ALLOCS_IN_POOL(pool, BEFORE_FRAGMENTATION_TOTAL_CHUNK_SIZE * 1.5, NUMBER_OF_VIEWS * 1);

            return views.size();
        });
    };
}