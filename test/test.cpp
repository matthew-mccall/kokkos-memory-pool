//
// Created by Matthew McCall on 5/23/23.
//

#include "catch2/catch_session.hpp"
#include "catch2/catch_test_macros.hpp"

#include "MemoryPool/MemoryPool.hpp"

constexpr size_t TEST_POOL_SIZE = 4;

#define EXPECT_CHUNKS_AND_ALLOCS_IN_POOL(pool, chunks, allocs) \
    REQUIRE(pool.getNumAllocations() == allocs); \
    REQUIRE(pool.getNumAllocatedChunks() == chunks); \
    REQUIRE(pool.getNumFreeChunks() == TEST_POOL_SIZE - chunks)

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

TEST_CASE("Memory Pool allocates successfully", "[MemoryPool]") {
    MemoryPool pool(TEST_POOL_SIZE); // 512 bytes

    SECTION("Allocating from a new pool") {
        auto view = pool.allocate<int>(1);
        CAPTURE(pool);
        REQUIRE(view.size() == 1);
        EXPECT_CHUNKS_AND_ALLOCS_IN_POOL(pool, 1, 1);
    }

    SECTION("Allocating from a pool with one chunk used") {
        auto view = pool.allocate<int>(1);
        CAPTURE(pool);
        REQUIRE(view.size() == 1);
        EXPECT_CHUNKS_AND_ALLOCS_IN_POOL(pool, 1, 1);

        auto view2 = pool.allocate<int>(1);
        CAPTURE(pool);
        REQUIRE(view2.size() == 1);
        EXPECT_CHUNKS_AND_ALLOCS_IN_POOL(pool, 2, 2);
    }

    SECTION("Allocating custom type") {
        auto view = pool.allocate<VeryLargeStruct>(1);
        CAPTURE(pool);
        REQUIRE(view.size() == 1);
        EXPECT_CHUNKS_AND_ALLOCS_IN_POOL(pool, TEST_POOL_SIZE, 1);
    }

    SECTION("Allocating from a full pool returns an empty view") {
        auto view = pool.allocate<VeryLargeStruct>(1);
        CAPTURE(pool);
        REQUIRE(view.size() == 1);
        EXPECT_CHUNKS_AND_ALLOCS_IN_POOL(pool, TEST_POOL_SIZE, 1);

        auto view2 = pool.allocate<VeryLargeStruct>(1);
        CAPTURE(pool);
        REQUIRE(view2.size() == 0);
        EXPECT_CHUNKS_AND_ALLOCS_IN_POOL(pool, TEST_POOL_SIZE, 1);
    }
}

TEST_CASE("Memory Pool allocates and deallocates successfully", "[MemoryPool]") {
    MemoryPool pool(4); // 512 bytes

    SECTION("Allocating and deallocating from a new pool") {
        auto view = pool.allocate<int>(1);
        CAPTURE(pool);
        REQUIRE(view.size() == 1);
        EXPECT_CHUNKS_AND_ALLOCS_IN_POOL(pool, 1, 1);

        pool.deallocate(view);
        CAPTURE(pool);
        EXPECT_CHUNKS_AND_ALLOCS_IN_POOL(pool, 0, 0);
    }

    SECTION("Allocating 2 chunks and deallocating the first chunk first") {
        auto view = pool.allocate<int>(1);
        CAPTURE(pool);
        REQUIRE(view.size() == 1);
        EXPECT_CHUNKS_AND_ALLOCS_IN_POOL(pool, 1, 1);

        auto view2 = pool.allocate<int>(1);
        CAPTURE(pool);
        REQUIRE(view2.size() == 1);
        EXPECT_CHUNKS_AND_ALLOCS_IN_POOL(pool, 2, 2);

        pool.deallocate(view);
        CAPTURE(pool); // -X--
        EXPECT_CHUNKS_AND_ALLOCS_IN_POOL(pool, 1, 1);

        pool.deallocate(view2);
        CAPTURE(pool); // ----
        EXPECT_CHUNKS_AND_ALLOCS_IN_POOL(pool, 0, 0);
    }

    SECTION("Allocating a large chunk from a pool with one chunk used") {
        auto view = pool.allocate<int>(1);
        CAPTURE(pool);
        REQUIRE(view.size() == 1);
        EXPECT_CHUNKS_AND_ALLOCS_IN_POOL(pool, 1, 1);

        auto view2 = pool.allocate<int>(1);
        CAPTURE(pool);
        REQUIRE(view2.size() == 1);
        EXPECT_CHUNKS_AND_ALLOCS_IN_POOL(pool, 2, 2);

        pool.deallocate(view);
        CAPTURE(pool); // -X--
        EXPECT_CHUNKS_AND_ALLOCS_IN_POOL(pool, 1, 1);

        auto view3 = pool.allocate<LargeStruct>(1);
        CAPTURE(pool); // -XXX
        REQUIRE(view3.size() == 1);
        EXPECT_CHUNKS_AND_ALLOCS_IN_POOL(pool, 3, 2);
    }

    SECTION("Allocating 2 chunks and deallocating the second chunk first") {
        auto view = pool.allocate<int>(1);
        CAPTURE(pool);
        REQUIRE(view.size() == 1);
        EXPECT_CHUNKS_AND_ALLOCS_IN_POOL(pool, 1, 1);

        auto view2 = pool.allocate<int>(1);
        CAPTURE(pool);
        REQUIRE(view2.size() == 1);
        EXPECT_CHUNKS_AND_ALLOCS_IN_POOL(pool, 2, 2);

        pool.deallocate(view2);
        CAPTURE(pool); // X---
        EXPECT_CHUNKS_AND_ALLOCS_IN_POOL(pool, 1, 1);

        pool.deallocate(view);
        CAPTURE(pool); // ----
        EXPECT_CHUNKS_AND_ALLOCS_IN_POOL(pool, 0, 0);
    }

    SECTION("Allocating and deallocating custom type") {
        auto view = pool.allocate<VeryLargeStruct>(1);
        CAPTURE(pool);
        REQUIRE(view.size() == 1);
        EXPECT_CHUNKS_AND_ALLOCS_IN_POOL(pool, TEST_POOL_SIZE, 1);

        pool.deallocate(view);
        CAPTURE(pool);
        EXPECT_CHUNKS_AND_ALLOCS_IN_POOL(pool, 0, 0);
    }

    SECTION("Allocating and deallocating from a full pool returns an empty view") {
        auto view = pool.allocate<VeryLargeStruct>(1);
        CAPTURE(pool);
        REQUIRE(view.size() == 1);
        EXPECT_CHUNKS_AND_ALLOCS_IN_POOL(pool, TEST_POOL_SIZE, 1);

        auto view2 = pool.allocate<VeryLargeStruct>(1);
        CAPTURE(pool);
        REQUIRE(view2.size() == 0);
        EXPECT_CHUNKS_AND_ALLOCS_IN_POOL(pool, TEST_POOL_SIZE, 1);

        pool.deallocate(view);
        CAPTURE(pool);
        EXPECT_CHUNKS_AND_ALLOCS_IN_POOL(pool, 0, 0);
    }

    SECTION("Deallocating from a full pool used returns a non-empty view") {
        auto view = pool.allocate<VeryLargeStruct>(1);
        CAPTURE(pool);
        REQUIRE(view.size() == 1);
        EXPECT_CHUNKS_AND_ALLOCS_IN_POOL(pool, TEST_POOL_SIZE, 1);

        auto view2 = pool.allocate<VeryLargeStruct>(1);
        REQUIRE(view2.size() == 0);
        EXPECT_CHUNKS_AND_ALLOCS_IN_POOL(pool, TEST_POOL_SIZE, 1);

        pool.deallocate(view);
        CAPTURE(pool);
        EXPECT_CHUNKS_AND_ALLOCS_IN_POOL(pool, 0, 0);

        auto view3 = pool.allocate<VeryLargeStruct>(1);
        CAPTURE(pool);
        REQUIRE(view3.size() == 1);
        EXPECT_CHUNKS_AND_ALLOCS_IN_POOL(pool, TEST_POOL_SIZE, 1);
    }
}