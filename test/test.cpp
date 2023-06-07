//
// Created by Matthew McCall on 5/23/23.
//

#include <chrono>
#include <locale>
#include <map>
#include <set>

#include "catch2/catch_session.hpp"
#include "catch2/catch_test_macros.hpp"
#include "catch2/benchmark/catch_benchmark.hpp"
#include "catch2/generators/catch_generators.hpp"
#include "catch2/generators/catch_generators_range.hpp"
#include "catch2/reporters/catch_reporter_event_listener.hpp"
#include "catch2/reporters/catch_reporter_streaming_base.hpp"
#include "catch2/reporters/catch_reporter_registrars.hpp"

#include "fmt/format.h"
#include "fmt/chrono.h"

#include "MemoryPool/MemoryPool.hpp"

constexpr size_t TEST_POOL_SIZE = 4;

#define EXPECTED_CHUNKS(DataType) (MemoryPool::getRequiredChunks(sizeof(DataType)))

#define EXPECT_CHUNKS_AND_ALLOCS_IN_POOL(pool, chunks, allocs) \
    REQUIRE(pool.getNumAllocatedChunks() == chunks); \
    REQUIRE(pool.getNumAllocations() == allocs); \
    REQUIRE(pool.getNumFreeChunks() == pool.getNumChunks() - chunks)

struct VeryLargeStruct {
    uint8_t data[MemoryPool::DEFAULT_CHUNK_SIZE * TEST_POOL_SIZE];
};

static_assert(sizeof(VeryLargeStruct) == MemoryPool::DEFAULT_CHUNK_SIZE * TEST_POOL_SIZE);

struct LargeStruct {
    uint8_t data[MemoryPool::DEFAULT_CHUNK_SIZE * TEST_POOL_SIZE / 2];
};

static_assert(sizeof(LargeStruct) == MemoryPool::DEFAULT_CHUNK_SIZE * TEST_POOL_SIZE / 2);

class TestControl : public Catch::EventListenerBase {
public:
    using EventListenerBase::EventListenerBase;

    void testRunStarting(const Catch::TestRunInfo &testRunInfo) override {
        Kokkos::initialize();
    }

    void testRunEnded(const Catch::TestRunStats &testRunStats) override {
        Kokkos::finalize();
    }

};

CATCH_REGISTER_LISTENER(TestControl)

class CSVReporter : public Catch::StreamingReporterBase {
public:
    using StreamingReporterBase::StreamingReporterBase;

    static std::string getDescription() {
        return "Reports logs from INFO macros";
    }

    void testRunStarting(const Catch::TestRunInfo &_testRunInfo) override {
        StreamingReporterBase::testRunStarting(_testRunInfo);
        fmt::print("Implementation,NumberOfViews,SizeOfViews,ChunksBetweenAllocations,ChunksRequestedOnSecond,Mean\n");
    }

    void sectionStarting(const Catch::SectionInfo &_sectionInfo) override {
        StreamingReporterBase::sectionStarting(_sectionInfo);
        currentSectionName = _sectionInfo.name;
    }

    void assertionEnded(const Catch::AssertionStats &stats) override {
        StreamingReporterBase::assertionEnded(stats);

        for (const auto &log: stats.infoMessages) {
            if (log.message.find("csv") != std::string::npos) {
                std::string message = log.message.substr(3);

                auto& sectionLogs = logs[currentSectionName];
                sectionLogs.insert(message);
            }
        }
    }

    void benchmarkEnded(const Catch::BenchmarkStats<> &stats) override {
        StreamingReporterBase::benchmarkEnded(stats);

        for (const auto &log: logs[currentSectionName]) {
            fmt::print("{},{:.0}\n", log, std::chrono::duration_cast<std::chrono::milliseconds>(stats.mean.point));
        }
    }

private:
    std::map<std::string, std::set<std::string>> logs;
    std::string currentSectionName;
};

CATCH_REGISTER_REPORTER("csv", CSVReporter)

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
    constexpr unsigned FRAGMENT_TEST_POOL_SIZE = 25;
    constexpr size_t NUMBER_OF_INTS_PER_CHUNK = MemoryPool::DEFAULT_CHUNK_SIZE / sizeof(int);

    MultiPool pool(FRAGMENT_TEST_POOL_SIZE);
    std::vector<Kokkos::View<int *>> views(FRAGMENT_TEST_POOL_SIZE);

    for (auto &view: views) {
        view = pool.allocateView<int>(NUMBER_OF_INTS_PER_CHUNK);
        REQUIRE(view.size() == NUMBER_OF_INTS_PER_CHUNK);
    }

    EXPECT_CHUNKS_AND_ALLOCS_IN_POOL(pool, FRAGMENT_TEST_POOL_SIZE, FRAGMENT_TEST_POOL_SIZE);

    int deallocStep = GENERATE(range(2, 5));
    int reallocFill = GENERATE_COPY(range(1, deallocStep));

    for (unsigned i = 0; i < views.size(); i++) {
        if (i % deallocStep != 0) {
            CAPTURE(i);
            CAPTURE(deallocStep);
            pool.deallocateView<int>(views[i]);
            CAPTURE(pool);
            unsigned expectedChunks = ((i / deallocStep) * (deallocStep - 1)) + (i % deallocStep);
            REQUIRE(pool.getNumFreeChunks() == expectedChunks);
        }
    }

    CAPTURE(pool);
    REQUIRE(pool.getNumFreeChunks() == (int) (FRAGMENT_TEST_POOL_SIZE * ((float) (deallocStep - 1) / deallocStep)));

    for (unsigned i = 0; i < views.size(); i++) {
        if (i % deallocStep != 0) {
            CAPTURE(i);
            CAPTURE(deallocStep);
            CAPTURE(reallocFill);
            views[i] = pool.allocateView<int>(NUMBER_OF_INTS_PER_CHUNK * reallocFill);
            CAPTURE(pool);
            REQUIRE(views[i].size() == NUMBER_OF_INTS_PER_CHUNK * reallocFill);
        }
    }

    CAPTURE(pool);
}

TEST_CASE("Benchmarks", "[!benchmark]") {
    constexpr size_t NUMBER_OF_VIEWS = 100'000;
    constexpr size_t SIZE_OF_VIEWS = 1024;

    const size_t TOTAL_CHUNK_SIZE = MemoryPool::getRequiredChunks(sizeof(int) * SIZE_OF_VIEWS) * NUMBER_OF_VIEWS;

    std::locale loc("en_US.UTF-8"); // For thousands separator

    BENCHMARK(fmt::format(loc, "Kokkos Allocating {:L} Views of {:L} ints", NUMBER_OF_VIEWS, SIZE_OF_VIEWS)) {
        std::vector<Kokkos::View<int[SIZE_OF_VIEWS]>> views(NUMBER_OF_VIEWS);

        for (auto &view: views) {
            view = Kokkos::View<int[SIZE_OF_VIEWS]>("view", SIZE_OF_VIEWS);
            REQUIRE(view.size() == SIZE_OF_VIEWS);
        }

        return views.size();
    };

    BENCHMARK(fmt::format(loc, "MultiPool Allocation {:L} Views of {:L} ints", NUMBER_OF_VIEWS, SIZE_OF_VIEWS)) {
        MultiPool pool(TOTAL_CHUNK_SIZE);
        std::vector<Kokkos::View<int[SIZE_OF_VIEWS]>> views(NUMBER_OF_VIEWS);

        for (auto &view: views) {
            view = pool.allocateView<int>(SIZE_OF_VIEWS);
            REQUIRE(view.size() == SIZE_OF_VIEWS);
        }

        EXPECT_CHUNKS_AND_ALLOCS_IN_POOL(pool, TOTAL_CHUNK_SIZE, NUMBER_OF_VIEWS);
        return views.size();
    };
}

TEST_CASE("Fragmentation Benchmarks", "[!benchmark][fragmentation]") {
    constexpr size_t NUMBER_OF_VIEWS = 10'000;
    constexpr size_t SIZE_OF_VIEWS = 1024;
    const size_t INITIAL_CHUNKS_PER_VIEW = MemoryPool::getRequiredChunks(sizeof(int) * SIZE_OF_VIEWS);

    const size_t TOTAL_CHUNK_SIZE = INITIAL_CHUNKS_PER_VIEW * NUMBER_OF_VIEWS;

    std::locale loc("en_US.UTF-8"); // For thousands separator

    int deallocStep = GENERATE(range(2, 5));
    int reallocFill = GENERATE_COPY(range(1, deallocStep));

    std::string kokkosBenchmarkName = fmt::format(loc, "Kokkos Allocation of {:L} Views of {:L} ints with {:L} free chunks between allocations and {:L} chunks requested in following allocations", NUMBER_OF_VIEWS, SIZE_OF_VIEWS, (deallocStep - 1) * INITIAL_CHUNKS_PER_VIEW, reallocFill * INITIAL_CHUNKS_PER_VIEW);

    SECTION(kokkosBenchmarkName) {
        // CSV output
        INFO(fmt::format("csvKokkos,{},{},{},{}", NUMBER_OF_VIEWS, SIZE_OF_VIEWS, (deallocStep - 1) * INITIAL_CHUNKS_PER_VIEW, reallocFill * INITIAL_CHUNKS_PER_VIEW));

        BENCHMARK(std::move(kokkosBenchmarkName)) {
            std::vector<Kokkos::View<int *>> views(NUMBER_OF_VIEWS);

            for (auto &view: views) {
                view = Kokkos::View<int *>("view", SIZE_OF_VIEWS);
                REQUIRE(view.size() == SIZE_OF_VIEWS);
            }

            for (unsigned i = 0; i < views.size(); i++) {
                if (i % deallocStep != 0) {
                    CAPTURE(i);
                    CAPTURE(deallocStep);
                    CAPTURE(reallocFill);
                    views[i] = Kokkos::View<int *>("view", SIZE_OF_VIEWS * reallocFill);
                    REQUIRE(views[i].size() == SIZE_OF_VIEWS * reallocFill);
                }
            }

            return views.size();
        };
    }

    std::string multiPoolBenchmarkName = fmt::format(loc, "Fragmented MultiPool Allocation of {:L} Views of {:L} ints with {:L} free chunks between allocations and {:L} chunks requested in following allocations", NUMBER_OF_VIEWS, SIZE_OF_VIEWS, (deallocStep - 1) * INITIAL_CHUNKS_PER_VIEW, reallocFill * INITIAL_CHUNKS_PER_VIEW);

    SECTION(multiPoolBenchmarkName) {
        // CSV output
        INFO(fmt::format("csvMultiPool,{},{},{},{}", NUMBER_OF_VIEWS, SIZE_OF_VIEWS, (deallocStep - 1) * INITIAL_CHUNKS_PER_VIEW, reallocFill * INITIAL_CHUNKS_PER_VIEW));

        BENCHMARK(std::move(multiPoolBenchmarkName)) {
            MultiPool pool(TOTAL_CHUNK_SIZE);
            std::vector<Kokkos::View<int *>> views(NUMBER_OF_VIEWS);

            for (auto &view: views) {
                view = pool.allocateView<int>(SIZE_OF_VIEWS);
                REQUIRE(view.size() == SIZE_OF_VIEWS);
            }

            EXPECT_CHUNKS_AND_ALLOCS_IN_POOL(pool, TOTAL_CHUNK_SIZE, NUMBER_OF_VIEWS);

            for (unsigned i = 0; i < views.size(); i++) {
                if (i % deallocStep != 0) {
                    CAPTURE(i);
                    CAPTURE(deallocStep);
                    pool.deallocateView<int>(views[i]);
                    unsigned expectedChunks =
                            (((i / deallocStep) * (deallocStep - 1)) + (i % deallocStep)) * INITIAL_CHUNKS_PER_VIEW;
                    REQUIRE(pool.getNumFreeChunks() == expectedChunks);
                }
            }

            REQUIRE(pool.getNumFreeChunks() ==
                    static_cast<int>(NUMBER_OF_VIEWS * (static_cast<float>(deallocStep - 1) / deallocStep)) *
                    INITIAL_CHUNKS_PER_VIEW);

            for (unsigned i = 0; i < views.size(); i++) {
                if (i % deallocStep != 0) {
                    CAPTURE(i);
                    CAPTURE(deallocStep);
                    CAPTURE(reallocFill);
                    views[i] = pool.allocateView<int>(SIZE_OF_VIEWS * reallocFill);
                    REQUIRE(views[i].size() == SIZE_OF_VIEWS * reallocFill);
                }
            }

            return views.size();
        };
    }
}