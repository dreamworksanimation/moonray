// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#include "test_wait.h"

#include <moonray/common/mcrt_util/Wait.h>

#include <scene_rdl2/render/util/AtomicFloat.h>

#include <atomic>
#include <chrono>
#include <thread>
#include <utility>
#include <vector>

using namespace std::chrono_literals;

CPPUNIT_TEST_SUITE_REGISTRATION(TestWait);

namespace {

struct NonScalar
{
    constexpr NonScalar()
    : m_a(0)
    , m_b(0)
    {
    }

    constexpr NonScalar(float n)
    : m_a(n)
    , m_b(n)
    {
    }

    constexpr NonScalar(float n, float m)
    : m_a(n)
    , m_b(m)
    {
    }

    float m_a;
    float m_b;
};

bool operator==(const NonScalar& a, const NonScalar& b) noexcept
{
    return a.m_a == b.m_a && a.m_b == b.m_b;
}

NonScalar operator+(const NonScalar& a, const NonScalar& b) noexcept
{
    return { a.m_a + b.m_a, a.m_b + b.m_b };
}

std::ostream& operator<<(std::ostream& outs, const NonScalar& a) noexcept
{
    return outs << a.m_a << ' ' << a.m_b;
}

enum Enum1
{
    ALPHA,   //  0
    BETA,    //  1
    GAMMA,   //  2
    DELTA,   //  3
    EPSILON, //  4
    ZETA,    //  5
    ETA,     //  6
    THETA,   //  7
    IOTA,    //  8
    KAPPA,   //  9
    LAMBDA,  // 10
    MU,      // 11
    NU,      // 12
    XI,      // 13
    OMICRON, // 14
    PI,      // 15
    NUM_VALUES
};

enum class Enum2 : std::uint8_t
{
    ALPHA,   //  0
    BETA,    //  1
    GAMMA,   //  2
    DELTA,   //  3
    EPSILON, //  4
    ZETA,    //  5
    ETA,     //  6
    THETA,   //  7
    IOTA,    //  8
    KAPPA,   //  9
    LAMBDA,  // 10
    MU,      // 11
    NU,      // 12
    XI,      // 13
    OMICRON, // 14
    PI,      // 15
    NUM_VALUES
};

enum class Enum3 : std::uint32_t
{
    ALPHA,   //  0
    BETA,    //  1
    GAMMA,   //  2
    DELTA,   //  3
    EPSILON, //  4
    ZETA,    //  5
    ETA,     //  6
    THETA,   //  7
    IOTA,    //  8
    KAPPA,   //  9
    LAMBDA,  // 10
    MU,      // 11
    NU,      // 12
    XI,      // 13
    OMICRON, // 14
    PI,      // 15
    NUM_VALUES
};

enum class Enum4 : std::uint64_t
{
    ALPHA,   //  0
    BETA,    //  1
    GAMMA,   //  2
    DELTA,   //  3
    EPSILON, //  4
    ZETA,    //  5
    ETA,     //  6
    THETA,   //  7
    IOTA,    //  8
    KAPPA,   //  9
    LAMBDA,  // 10
    MU,      // 11
    NU,      // 12
    XI,      // 13
    OMICRON, // 14
    PI,      // 15
    NUM_VALUES
};

std::ostream& operator<<(std::ostream& outs, Enum1 e)
{
    switch (e) {
    case Enum1::ALPHA:
        outs << "alpha";
        break;
    case Enum1::BETA:
        outs << "beta";
        break;
    case Enum1::GAMMA:
        outs << "gamma";
        break;
    case Enum1::DELTA:
        outs << "delta";
        break;
    case Enum1::EPSILON:
        outs << "epsilon";
        break;
    case Enum1::ZETA:
        outs << "zeta";
        break;
    case Enum1::ETA:
        outs << "eta";
        break;
    case Enum1::THETA:
        outs << "theta";
        break;
    case Enum1::IOTA:
        outs << "iota";
        break;
    case Enum1::KAPPA:
        outs << "kappa";
        break;
    case Enum1::LAMBDA:
        outs << "lambda";
        break;
    case Enum1::MU:
        outs << "mu";
        break;
    case Enum1::NU:
        outs << "nu";
        break;
    case Enum1::XI:
        outs << "xi";
        break;
    case Enum1::OMICRON:
        outs << "omicron";
        break;
    case Enum1::PI:
        outs << "pi";
        break;
    default:
        outs << "unknown";
        break;
    }
    return outs;
}

std::ostream& operator<<(std::ostream& outs, Enum2 e)
{
    switch (e) {
    case Enum2::ALPHA:
        outs << "alpha";
        break;
    case Enum2::BETA:
        outs << "beta";
        break;
    case Enum2::GAMMA:
        outs << "gamma";
        break;
    case Enum2::DELTA:
        outs << "delta";
        break;
    case Enum2::EPSILON:
        outs << "epsilon";
        break;
    case Enum2::ZETA:
        outs << "zeta";
        break;
    case Enum2::ETA:
        outs << "eta";
        break;
    case Enum2::THETA:
        outs << "theta";
        break;
    case Enum2::IOTA:
        outs << "iota";
        break;
    case Enum2::KAPPA:
        outs << "kappa";
        break;
    case Enum2::LAMBDA:
        outs << "lambda";
        break;
    case Enum2::MU:
        outs << "mu";
        break;
    case Enum2::NU:
        outs << "nu";
        break;
    case Enum2::XI:
        outs << "xi";
        break;
    case Enum2::OMICRON:
        outs << "omicron";
        break;
    case Enum2::PI:
        outs << "pi";
        break;
    default:
        outs << "unknown";
        break;
    }
    return outs;
}

std::ostream& operator<<(std::ostream& outs, Enum3 e)
{
    switch (e) {
    case Enum3::ALPHA:
        outs << "alpha";
        break;
    case Enum3::BETA:
        outs << "beta";
        break;
    case Enum3::GAMMA:
        outs << "gamma";
        break;
    case Enum3::DELTA:
        outs << "delta";
        break;
    case Enum3::EPSILON:
        outs << "epsilon";
        break;
    case Enum3::ZETA:
        outs << "zeta";
        break;
    case Enum3::ETA:
        outs << "eta";
        break;
    case Enum3::THETA:
        outs << "theta";
        break;
    case Enum3::IOTA:
        outs << "iota";
        break;
    case Enum3::KAPPA:
        outs << "kappa";
        break;
    case Enum3::LAMBDA:
        outs << "lambda";
        break;
    case Enum3::MU:
        outs << "mu";
        break;
    case Enum3::NU:
        outs << "nu";
        break;
    case Enum3::XI:
        outs << "xi";
        break;
    case Enum3::OMICRON:
        outs << "omicron";
        break;
    case Enum3::PI:
        outs << "pi";
        break;
    default:
        outs << "unknown";
        break;
    }
    return outs;
}

std::ostream& operator<<(std::ostream& outs, Enum4 e)
{
    switch (e) {
    case Enum4::ALPHA:
        outs << "alpha";
        break;
    case Enum4::BETA:
        outs << "beta";
        break;
    case Enum4::GAMMA:
        outs << "gamma";
        break;
    case Enum4::DELTA:
        outs << "delta";
        break;
    case Enum4::EPSILON:
        outs << "epsilon";
        break;
    case Enum4::ZETA:
        outs << "zeta";
        break;
    case Enum4::ETA:
        outs << "eta";
        break;
    case Enum4::THETA:
        outs << "theta";
        break;
    case Enum4::IOTA:
        outs << "iota";
        break;
    case Enum4::KAPPA:
        outs << "kappa";
        break;
    case Enum4::LAMBDA:
        outs << "lambda";
        break;
    case Enum4::MU:
        outs << "mu";
        break;
    case Enum4::NU:
        outs << "nu";
        break;
    case Enum4::XI:
        outs << "xi";
        break;
    case Enum4::OMICRON:
        outs << "omicron";
        break;
    case Enum4::PI:
        outs << "pi";
        break;
    default:
        outs << "unknown";
        break;
    }
    return outs;
}

#if defined(__cpp_lib_to_underlying)
using std::underlying_type;
#else
template <typename Enum>
constexpr std::underlying_type_t<Enum> to_underlying(Enum e) noexcept
{
    using type = typename std::underlying_type<Enum>::type;
    return static_cast<type>(e);
}
#endif

template <typename T>
constexpr T zero(static_cast<T>(0));

template <typename T>
constexpr T one(static_cast<T>(1));

template <typename T>
constexpr T two(static_cast<T>(2));

template <>
constexpr NonScalar zero<NonScalar>{ 0.0f, 0.0f };

template <>
constexpr NonScalar one<NonScalar>{ 1.0f, 1.0f };

template <>
constexpr NonScalar two<NonScalar>{ 2.0f, 2.0f };

template <typename T, bool isEnum>
struct MaxThreadsHelper;

template <typename T>
struct MaxThreadsHelper<T, false /*isEnum*/>
{
    static constexpr size_t value = std::numeric_limits<T>::max();
};

// We don't want more threads than our enum type can handle, otherwise we overflow.
template <typename T>
struct MaxThreadsHelper<T, true /*isEnum*/>
{
    static constexpr size_t value = to_underlying(T::NUM_VALUES);
};

template <typename T>
struct MaxThreads
{
    static constexpr size_t value = MaxThreadsHelper<T, std::is_enum<T>::value>::value;
};

template <>
struct MaxThreads<float>
{
    static constexpr size_t value = 1u << 24u; // The largest value before the interval is beyond 1.0
};

template <>
struct MaxThreads<NonScalar>
{
    static constexpr size_t value = 1024u;
};


template <bool isEnum>
struct IncrementHelper;

template <>
struct IncrementHelper<false /*isEnum*/>
{
    template <typename T>
    static T increment(T a) noexcept
    {
        return a + one<T>;
    }
};

template <>
struct IncrementHelper<true /*isEnum*/>
{
    template <typename T>
    static T increment(T a) noexcept
    {
        using type     = typename std::underlying_type<T>::type;
        const auto val = to_underlying(a);
        return T(static_cast<type>(val + type(1)));
    }
};

template <typename T>
T increment(T a) noexcept
{
    return IncrementHelper<std::is_enum<T>::value>::increment(a);
}

template <typename T>
void atomic_increment(std::atomic<T>& a) noexcept
{
    T val = a.load();
    T next;
    do {
        next = increment(val);
    } while (!a.compare_exchange_weak(val, next));
}

template <typename T>
void notifyOneThread(std::atomic<T>& a, std::atomic<bool>& ready)
{
    ready = true;
    wait(a, zero<T>);
    atomic_increment(a);
}

template <typename T>
void doTestNotifyOne()
{
    std::atomic<bool> ready(false);

    std::atomic<T> a(zero<T>);
    std::thread    t(notifyOneThread<T>, std::ref(a), std::ref(ready));

    while (!ready) {
        CPPUNIT_ASSERT_EQUAL(zero<T>, a.load());
    }

    std::this_thread::sleep_for(1s);
    CPPUNIT_ASSERT_EQUAL(zero<T>, a.load());

    a.store(one<T>);
    notify_one(a);

    t.join();
    CPPUNIT_ASSERT_EQUAL(two<T>, a.load());
}

template <typename T>
void notifyAllThreads(std::atomic<T>& a, std::atomic<int>& launchCount)
{
    CPPUNIT_ASSERT_EQUAL(zero<T>, a.load());
    ++launchCount;
    wait(a, zero<T>);
    atomic_increment(a);
}

template <typename T>
size_t getNumThreads()
{
    const size_t maxThreads = MaxThreads<T>::value;
    const size_t minThreads = 8u;
    const size_t hw         = std::thread::hardware_concurrency();
    return std::max(minThreads, std::min(maxThreads, hw));
}

template <typename T>
void doTestNotifyAll()
{
    const int numThreads = getNumThreads<T>() - 1u;

    std::atomic<int> launchCount(0);
    std::atomic<T>   a(zero<T>);

    std::vector<std::thread> threads;
    threads.reserve(numThreads);
    for (int i = 0; i < numThreads; ++i) {
        threads.emplace_back(notifyAllThreads<T>, std::ref(a), std::ref(launchCount));
    }

    while (launchCount < numThreads) {
        CPPUNIT_ASSERT_EQUAL(zero<T>, a.load());
    }

    std::this_thread::sleep_for(1s);
    CPPUNIT_ASSERT_EQUAL(zero<T>, a.load());

    a.store(one<T>);
    notify_all(a);

    for (auto& t : threads) {
        t.join();
    }

    CPPUNIT_ASSERT_EQUAL(T(numThreads + 1), a.load()); // The main thread sets it to 1 to start off with
}

} // anonymous namespace

void TestWait::testNotifyOne()
{
    doTestNotifyOne<float>();
    doTestNotifyOne<NonScalar>();
    doTestNotifyOne<std::int32_t>();
    doTestNotifyOne<Enum1>();
    doTestNotifyOne<Enum2>();
    doTestNotifyOne<Enum3>();
    doTestNotifyOne<Enum4>();
}

void TestWait::testNotifyAll()
{
    doTestNotifyAll<float>();
    doTestNotifyAll<NonScalar>();
    doTestNotifyAll<std::int32_t>();
    doTestNotifyAll<Enum1>();
    doTestNotifyAll<Enum2>();
    doTestNotifyAll<Enum3>();
    doTestNotifyAll<Enum4>();
}

