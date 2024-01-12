// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#include "test_atomic_functions.h"

#include <moonray/common/mcrt_util/Atomic.h>

#include <atomic>
#include <iterator>
#include <random>
#include <thread>
#include <vector>

CPPUNIT_TEST_SUITE_REGISTRATION(TestAtomicFunctions);

namespace {
template <typename RandomIterator, typename F, typename... Args>
std::vector<std::thread> distributeThreads(F f,
                                           RandomIterator first,
                                           RandomIterator last,
                                           std::size_t nthreads,
                                           Args&&... args)
{
    std::size_t nelements = std::distance(first, last);

    std::vector<std::thread> threads;
    threads.reserve(nthreads);

    while (nelements > 0) {
        const auto nt = nelements/nthreads;
        threads.emplace_back(f, first, first + nt, std::forward<Args>(args)...);
        first += nt;
        --nthreads;
        nelements -= nt;
    }

    MNRY_ASSERT(nthreads == 0);
    MNRY_ASSERT(first == last);

    return threads;
}

template <typename Iterator>
void findSum(std::atomic<bool>& ready,
             typename std::iterator_traits<Iterator>::value_type& value,
             Iterator first,
             Iterator last)
{
    // Sync threads with a busy wait
    while (!ready.load(std::memory_order_relaxed)) ;

    for ( ; first != last; ++first) {
        const auto f = *first;
        moonray::util::atomicAdd(std::addressof(value), f);
    }
}

template <typename Iterator>
void findMin(std::atomic<bool>& ready,
             typename std::iterator_traits<Iterator>::value_type& value,
             Iterator first,
             Iterator last)
{
    // Sync threads with a busy wait
    while (!ready.load(std::memory_order_relaxed)) ;

    for ( ; first != last; ++first) {
        const auto f = *first;
        moonray::util::atomicMin(std::addressof(value), f);
    }
}

template <typename Iterator>
void findMax(std::atomic<bool>& ready,
             typename std::iterator_traits<Iterator>::value_type& value,
             Iterator first,
             Iterator last)
{
    // Sync threads with a busy wait
    while (!ready.load(std::memory_order_relaxed)) ;

    for ( ; first != last; ++first) {
        const auto f = *first;
        moonray::util::atomicMax(std::addressof(value), f);
    }
}

template <typename Iterator>
void findClosest(std::atomic<bool>& ready,
                 typename std::iterator_traits<Iterator>::value_type& value,
                 Iterator first,
                 Iterator last)
{
    // Sync threads with a busy wait
    while (!ready.load(std::memory_order_relaxed)) ;

    for ( ; first != last; ++first) {
        const auto f = *first;
        moonray::util::atomicAssignIfClosest(value.values, f.values);
    }
}

template <typename T>
void runTestAdd()
{
    using Container = std::vector<T>;

    constexpr std::size_t nvalues = 2'999;
    Container v(nvalues);
    std::iota(v.begin(), v.end(), T(1));

    std::mt19937 rng(42);
    std::shuffle(v.begin(), v.end(), rng);

    // Triangle number summation
    const T expectedValue = (T(nvalues) * (T(nvalues) + T(1))) / T(2);

    T sumValue = 0;

    const int nthreads = std::max(8u, std::thread::hardware_concurrency());
    std::atomic<bool> ready(false);
    auto f = [&ready, &sumValue](auto first, auto last) { findSum(ready, sumValue, first, last); };
    auto threads = distributeThreads(f, v.cbegin(), v.cend(), nthreads);
    ready.store(true, std::memory_order_relaxed);
    for (auto& t: threads) {
        t.join();
    }

    CPPUNIT_ASSERT(expectedValue == sumValue);
}

template <typename T>
void runTestMin()
{
    using Container = std::vector<T>;

    constexpr std::size_t nvalues = 10'007;
    Container v(nvalues);
    std::iota(v.begin(), v.end(), T(1));

    std::mt19937 rng(42);
    std::shuffle(v.begin(), v.end(), rng);

    T minValue = 50;

    const int nthreads = std::max(8u, std::thread::hardware_concurrency());
    std::atomic<bool> ready(false);
    auto f = [&ready, &minValue](auto first, auto last) { findMin(ready, minValue, first, last); };
    auto threads = distributeThreads(f, v.cbegin(), v.cend(), nthreads);
    ready.store(true, std::memory_order_relaxed);
    for (auto& t: threads) {
        t.join();
    }

    CPPUNIT_ASSERT(minValue == T(1));
}

template <typename T>
void runTestMax()
{
    using Container = std::vector<T>;

    constexpr std::size_t nvalues = 10'007;
    Container v(nvalues);
    std::iota(v.begin(), v.end(), T(1));

    std::mt19937 rng(42);
    std::shuffle(v.begin(), v.end(), rng);

    T maxValue = 50;

    const int nthreads = std::max(8u, std::thread::hardware_concurrency());
    std::atomic<bool> ready(false);
    auto f = [&ready, &maxValue](auto first, auto last) { findMax(ready, maxValue, first, last); };
    auto threads = distributeThreads(f, v.cbegin(), v.cend(), nthreads);
    ready.store(true, std::memory_order_relaxed);
    for (auto& t: threads) {
        t.join();
    }

    CPPUNIT_ASSERT(maxValue == static_cast<T>(nvalues));
}

void runTestClosest()
{
    struct alignas(moonray::util::kDoubleQuadWordAtomicAlignment) Array
    {
        explicit Array(float distance) noexcept
        : values{distance * 11.0f, distance * 9.0f, distance * 5.0f, distance}
        {
        }

        float values[4];
    };

    using Container = std::vector<Array>;

    constexpr std::size_t nvalues = 10'007;
    Container v;
    v.reserve(nvalues);
    for (std::size_t i = 0; i < nvalues; ++i) {
        // Let's not start at 0: it makes all of our values zero the way we have defined our constructor, which is no
        // fun.
        v.push_back(Array(i + 1));
    }

    Array result(50);

    // Make sure we start with a value who has a depth value that is somewhere in the middle.
    CPPUNIT_ASSERT(result.values[3] > v.front().values[3]);
    CPPUNIT_ASSERT(result.values[3] < v.back().values[3]);

    std::mt19937 rng(42);
    std::shuffle(v.begin(), v.end(), rng);

    const int nthreads = std::max(8u, std::thread::hardware_concurrency());
    std::atomic<bool> ready(false);
    auto f = [&ready, &result](auto first, auto last) { findClosest(ready, result, first, last); };
    auto threads = distributeThreads(f, v.cbegin(), v.cend(), nthreads);
    ready.store(true, std::memory_order_relaxed);
    for (auto& t: threads) {
        t.join();
    }

    for (std::size_t i = 0; i < 4; ++i) {
        CPPUNIT_ASSERT_EQUAL(result.values[0], 11.0f);
        CPPUNIT_ASSERT_EQUAL(result.values[1],  9.0f);
        CPPUNIT_ASSERT_EQUAL(result.values[2],  5.0f);
        CPPUNIT_ASSERT_EQUAL(result.values[3],  1.0f);
    }
}

} // anonymous namespace

void TestAtomicFunctions::testAdd()
{
    for (int i = 0; i < 1024; ++i) {
        runTestAdd<int32_t>();
        runTestAdd<int64_t>();
        runTestAdd<__int128>();
        runTestAdd<float>();
        runTestAdd<double>();
    }
}

void TestAtomicFunctions::testMin()
{
    for (int i = 0; i < 1024; ++i) {
        runTestMin<int32_t>();
        runTestMin<int64_t>();
        runTestMin<__int128>();
        runTestMin<float>();
        runTestMin<double>();
    }
}

void TestAtomicFunctions::testMax()
{
    for (int i = 0; i < 1024; ++i) {
        runTestMax<int32_t>();
        runTestMax<int64_t>();
        runTestMax<__int128>();
        runTestMax<float>();
        runTestMax<double>();
    }
}

void TestAtomicFunctions::testClosest()
{
    for (int i = 0; i < 1024; ++i) {
        runTestClosest();
    }
}

