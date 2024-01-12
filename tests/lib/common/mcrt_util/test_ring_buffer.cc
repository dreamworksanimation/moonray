// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#include "test_ring_buffer.h"

#include <moonray/common/mcrt_util/RingBuffer.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <forward_list>
#include <iostream>
#include <iterator>
#include <random>
#include <thread>
#include <vector>

CPPUNIT_TEST_SUITE_REGISTRATION(TestRingBuffer);

using namespace std::chrono_literals;

namespace {
class StopWatch
{
public:
    using representation_t = long double;
    using clock_t          = std::chrono::steady_clock;
    using time_point_t     = std::chrono::time_point<clock_t>;
    using duration_t       = std::chrono::duration<representation_t>;
    using period           = clock_t::period;

    StopWatch();
    void             start();
    void             stop();
    duration_t       duration() const;
    representation_t count() const;
    void             restart();

private:
    time_point_t mStart;
    duration_t   mDuration;
};

inline StopWatch::StopWatch()
: mDuration(duration_t::zero())
{
}

inline void StopWatch::start()
{
    mStart = clock_t::now();
}

inline void StopWatch::stop()
{
    time_point_t tp = clock_t::now();
    mDuration += tp - mStart;
}

inline StopWatch::duration_t StopWatch::duration() const
{
    return mDuration;
}

inline StopWatch::representation_t StopWatch::count() const
{
    return mDuration.count();
}

inline void StopWatch::restart()
{
    mDuration = duration_t::zero();
}

constexpr std::size_t k_log_queue_size_test    = 10;
constexpr std::size_t k_log_queue_size_profile = 10;
constexpr bool        k_verbose                = true;
constexpr int         k_nvalues                = 1 << 18;
constexpr double      p_throw_construct        = 0.0002;
constexpr double      p_throw_copy_constructor = 0.0000; // Leave at zero: we're not exception safe on return
constexpr double      p_throw_copy_assignment  = 0.0002;
constexpr double      p_throw_move_constructor = 0.0000;
constexpr double      p_throw_move_assignment  = 0.0002;
constexpr auto        k_overall_test_time      = std::chrono::microseconds(500);

template <typename T, typename Traits>
using buffer_test_t = RingBufferImpl<T, k_log_queue_size_test, Traits>;

class TestException : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

namespace timing {
using representation_t = long double;
using clock_t          = std::chrono::steady_clock;
using time_point_t     = std::chrono::time_point<clock_t>;
using duration_t       = std::chrono::duration<representation_t>;
} // namespace timing

template <typename Container>
class GenericInputIterator
{
    using base_iterator = typename Container::const_iterator;

public:
    using value_type        = typename std::iterator_traits<base_iterator>::value_type;
    using difference_type   = typename std::iterator_traits<base_iterator>::difference_type;
    using pointer           = typename std::iterator_traits<base_iterator>::pointer;
    using reference         = typename std::iterator_traits<base_iterator>::reference;
    using iterator_category = std::input_iterator_tag;

    static GenericInputIterator begin(const Container& c) { return GenericInputIterator(c.cbegin()); }

    static GenericInputIterator end(const Container& c) { return GenericInputIterator(c.cend()); }

    const value_type& operator*() const { return m_iterator.operator*(); };

    const value_type* operator->() const { return m_iterator.operator->(); };

    GenericInputIterator& operator++()
    {
        ++m_iterator;
        return *this;
    }

    GenericInputIterator operator++(int)
    {
        GenericInputIterator temp(*this);
        this->               operator++();
        return temp;
    }

    friend bool operator==(const GenericInputIterator& a, const GenericInputIterator& b)
    {
        return a.m_iterator == b.m_iterator;
    }

private:
    explicit GenericInputIterator(base_iterator it)
    : m_iterator(it)
    {
    }

    base_iterator m_iterator;
};

template <typename Container>
bool operator!=(const GenericInputIterator<Container>& a, const GenericInputIterator<Container>& b)
{
    return !(a == b);
}

std::ostream& print_lock_traits(std::ostream& outs, LockTraits traits)
{
    switch (traits) {
    case LockTraits::AUTO_DETECT:
        outs << "auto-detect\n";
        break;
    case LockTraits::SPIN_LOCK:
        outs << "spin-lock\n";
        break;
    case LockTraits::LOCK_FREE:
        outs << "lock-free\n";
        break;
    }
    return outs;
}

template <typename Traits>
std::ostream& print_traits_info(std::ostream& outs)
{
    switch (Traits::producer_traits) {
    case ProducerTraits::MULTIPLE_PRODUCERS:
        outs << "multiple producer support";
        break;
    case ProducerTraits::SINGLE_PRODUCER:
        outs << "single producer support";
        break;
    }
    outs << " and ";
    switch (Traits::consumer_traits) {
    case ConsumerTraits::MULTIPLE_CONSUMERS:
        outs << "multiple consumer support";
        break;
    case ConsumerTraits::SINGLE_CONSUMER:
        outs << "single consumer support";
        break;
    }
    outs << " with ";
    print_lock_traits(outs, Traits::lock_traits);
    outs << std::endl;
    return outs;
}

template <typename Traits>
void verify_preconditions(int nproducers, int nconsumers)
{
    switch (Traits::producer_traits) {
    case ProducerTraits::MULTIPLE_PRODUCERS:
        break;
    case ProducerTraits::SINGLE_PRODUCER:
        CPPUNIT_ASSERT_EQUAL(1, nproducers);
        break;
    }
    switch (Traits::consumer_traits) {
    case ConsumerTraits::MULTIPLE_CONSUMERS:
        break;
    case ConsumerTraits::SINGLE_CONSUMER:
        CPPUNIT_ASSERT_EQUAL(1, nconsumers);
        break;
    }
}

struct TestDataNoExceptions
{
    // We pass this in the constructor because we want to know the latency of enqueueing, not the latency of
    // construction/destruction..
    TestDataNoExceptions(int v, timing::duration_t construction_time, timing::duration_t copy_time) noexcept
    : m_start(now())
    , m_copy_time(copy_time)
    , m_payload(v)
    {
        ++s_constructed;
        std::this_thread::sleep_for(construction_time);
    }

    explicit TestDataNoExceptions(int v) noexcept
    : m_start(now())
    , m_copy_time(timing::duration_t::zero())
    , m_payload(v)
    {
        ++s_constructed;
    }

    TestDataNoExceptions(TestDataNoExceptions&& other) noexcept
    : m_start(other.m_start)
    , m_copy_time(other.m_copy_time)
    , m_payload(other.m_payload)
    {
        ++s_constructed;
    }

    TestDataNoExceptions(const TestDataNoExceptions& other) noexcept
    : m_start(other.m_start)
    , m_copy_time(other.m_copy_time)
    , m_payload(other.m_payload)
    {
        ++s_constructed;
        std::this_thread::sleep_for(m_copy_time);
    }

    ~TestDataNoExceptions() noexcept { --s_constructed; }

    TestDataNoExceptions& operator=(TestDataNoExceptions&& other) noexcept
    {
        // We implement this (instead of default) so that we can break.
        m_start     = other.m_start;
        m_copy_time = other.m_copy_time;
        m_payload   = other.m_payload;
        return *this;
    }

    TestDataNoExceptions& operator=(const TestDataNoExceptions& other) noexcept
    {
        m_start     = other.m_start;
        m_copy_time = other.m_copy_time;
        m_payload   = other.m_payload;
        std::this_thread::sleep_for(m_copy_time);
        return *this;
    }

    static timing::time_point_t now() { return timing::clock_t::now(); }

    operator int() const noexcept { return m_payload; }

    timing::time_point_t    m_start;
    timing::duration_t      m_copy_time;
    int                     m_payload;
    static std::atomic<int> s_constructed;
};

std::atomic<int> TestDataNoExceptions::s_constructed(0);

struct TestDataExceptions : private TestDataNoExceptions
{
    using TestDataNoExceptions::now;
    using TestDataNoExceptions::operator int;
    using TestDataNoExceptions::m_copy_time;
    using TestDataNoExceptions::m_payload;
    using TestDataNoExceptions::m_start;
    using TestDataNoExceptions::s_constructed;

    TestDataExceptions(int                v,
                       timing::duration_t construction_time,
                       timing::duration_t copy_time,
                       bool               throw_in_construction,
                       bool               throw_in_copy_constructor,
                       bool               throw_in_copy_assignment,
                       bool               throw_in_move_constructor,
                       bool               throw_in_move_assignment)
    : TestDataNoExceptions(v, construction_time, copy_time)
    , m_throw_in_copy_constructor(throw_in_copy_constructor)
    , m_throw_in_copy_assignment(throw_in_copy_assignment)
    , m_throw_in_move_constructor(throw_in_move_constructor)
    , m_throw_in_move_assignment(throw_in_move_assignment)
    {
        if (throw_in_construction) {
            throw TestException("Throw in constructor");
        }
    }

    explicit TestDataExceptions(int v) noexcept
    : TestDataNoExceptions(v)
    , m_throw_in_copy_constructor(false)
    , m_throw_in_copy_assignment(false)
    , m_throw_in_move_constructor(false)
    , m_throw_in_move_assignment(false)
    {
    }

    TestDataExceptions(TestDataExceptions&& other)
    : TestDataNoExceptions(std::move(other))
    , m_throw_in_copy_constructor(other.m_throw_in_copy_constructor)
    , m_throw_in_copy_assignment(other.m_throw_in_copy_assignment)
    , m_throw_in_move_constructor(other.m_throw_in_move_constructor)
    , m_throw_in_move_assignment(other.m_throw_in_move_assignment)
    {
        if (m_throw_in_move_constructor) {
            throw TestException("Throw in move constructor");
        }
    }

    TestDataExceptions(const TestDataExceptions& other)
    : TestDataNoExceptions(other)
    , m_throw_in_copy_constructor(other.m_throw_in_copy_constructor)
    , m_throw_in_copy_assignment(other.m_throw_in_copy_assignment)
    , m_throw_in_move_constructor(other.m_throw_in_move_constructor)
    , m_throw_in_move_assignment(other.m_throw_in_move_assignment)
    {
        if (m_throw_in_copy_constructor) {
            throw TestException("Throw in copy constructor");
        }
    }

    TestDataExceptions& operator=(TestDataExceptions&& other)
    {
        TestDataNoExceptions::operator=(std::move(other));
        m_throw_in_copy_constructor = other.m_throw_in_copy_constructor;
        m_throw_in_copy_assignment  = other.m_throw_in_copy_assignment;
        m_throw_in_move_constructor = other.m_throw_in_move_constructor;
        m_throw_in_move_assignment  = other.m_throw_in_move_assignment;
        if (m_throw_in_move_assignment) {
            throw TestException("Throw in move assignment");
        }
        return *this;
    }

    TestDataExceptions& operator=(const TestDataExceptions& other)
    {
        TestDataNoExceptions::operator=(other);
        m_throw_in_copy_constructor = other.m_throw_in_copy_constructor;
        m_throw_in_copy_assignment  = other.m_throw_in_copy_assignment;
        m_throw_in_move_constructor = other.m_throw_in_move_constructor;
        m_throw_in_move_assignment  = other.m_throw_in_move_assignment;
        if (m_throw_in_copy_assignment) {
            throw TestException("Throw in copy assignment");
        }
        return *this;
    }

    bool m_throw_in_copy_constructor;
    bool m_throw_in_copy_assignment;
    bool m_throw_in_move_constructor;
    bool m_throw_in_move_assignment;
};

struct TestDataCopyExceptions : private TestDataNoExceptions
{
    using TestDataNoExceptions::now;
    using TestDataNoExceptions::operator int;
    using TestDataNoExceptions::m_copy_time;
    using TestDataNoExceptions::m_payload;
    using TestDataNoExceptions::m_start;
    using TestDataNoExceptions::s_constructed;

    TestDataCopyExceptions(int                v,
                           timing::duration_t construction_time,
                           timing::duration_t copy_time,
                           bool               throw_in_construction,
                           bool               throw_in_copy_constructor,
                           bool               throw_in_copy_assignment)
    : TestDataNoExceptions(v, construction_time, copy_time)
    , m_throw_in_copy_constructor(throw_in_copy_constructor)
    , m_throw_in_copy_assignment(throw_in_copy_assignment)
    {
        if (throw_in_construction) {
            throw TestException("Throw in constructor");
        }
    }

    explicit TestDataCopyExceptions(int v) noexcept
    : TestDataNoExceptions(v)
    , m_throw_in_copy_constructor(false)
    , m_throw_in_copy_assignment(false)
    {
    }

    TestDataCopyExceptions(TestDataCopyExceptions&& other) noexcept = default;

    TestDataCopyExceptions(const TestDataCopyExceptions& other)
    : TestDataNoExceptions(other)
    , m_throw_in_copy_constructor(other.m_throw_in_copy_constructor)
    , m_throw_in_copy_assignment(other.m_throw_in_copy_assignment)
    {
        if (m_throw_in_copy_constructor) {
            throw TestException("Throw in copy constructor");
        }
    }

    TestDataCopyExceptions& operator=(TestDataCopyExceptions&& other) noexcept = default;

    TestDataCopyExceptions& operator=(const TestDataCopyExceptions& other)
    {
        TestDataNoExceptions::operator=(other);
        m_throw_in_copy_constructor = other.m_throw_in_copy_constructor;
        m_throw_in_copy_assignment  = other.m_throw_in_copy_assignment;
        if (m_throw_in_copy_assignment) {
            throw TestException("Throw in copy assignment");
        }
        return *this;
    }

    bool m_throw_in_copy_constructor;
    bool m_throw_in_copy_assignment;
};

struct TestDataMoveExceptions : private TestDataNoExceptions
{
    using TestDataNoExceptions::now;
    using TestDataNoExceptions::operator int;
    using TestDataNoExceptions::m_copy_time;
    using TestDataNoExceptions::m_payload;
    using TestDataNoExceptions::m_start;
    using TestDataNoExceptions::s_constructed;

    TestDataMoveExceptions(int                v,
                           timing::duration_t construction_time,
                           timing::duration_t copy_time,
                           bool               throw_in_construction,
                           bool               throw_in_move_constructor,
                           bool               throw_in_move_assignment)
    : TestDataNoExceptions(v, construction_time, copy_time)
    , m_throw_in_move_constructor(throw_in_move_constructor)
    , m_throw_in_move_assignment(throw_in_move_assignment)
    {
        if (throw_in_construction) {
            throw TestException("Throw in constructor");
        }
    }

    explicit TestDataMoveExceptions(int v) noexcept
    : TestDataNoExceptions(v)
    , m_throw_in_move_constructor(false)
    , m_throw_in_move_assignment(false)
    {
    }

    TestDataMoveExceptions(TestDataMoveExceptions&& other)
    : TestDataNoExceptions(std::move(other))
    , m_throw_in_move_constructor(other.m_throw_in_move_constructor)
    , m_throw_in_move_assignment(other.m_throw_in_move_assignment)
    {
        if (m_throw_in_move_constructor) {
            throw TestException("Throw in move constructor");
        }
    }

    TestDataMoveExceptions(const TestDataMoveExceptions& other) noexcept = default;

    TestDataMoveExceptions& operator=(TestDataMoveExceptions&& other)
    {
        TestDataNoExceptions::operator=(std::move(other));
        m_throw_in_move_constructor = other.m_throw_in_move_constructor;
        m_throw_in_move_assignment  = other.m_throw_in_move_assignment;
        if (m_throw_in_move_assignment) {
            throw TestException("Throw in move assignment");
        }
        return *this;
    }

    TestDataMoveExceptions& operator=(const TestDataMoveExceptions& other) noexcept = default;

    bool m_throw_in_move_constructor;
    bool m_throw_in_move_assignment;
};

template <typename Duration>
class RandomTime
{
public:
    using rep                 = typename Duration::rep;
    using ExtremeDistribution = std::bernoulli_distribution;
    using MeanDistribution    = std::poisson_distribution<rep>;

    RandomTime(Duration mean, double probability_of_extreme)
    : m_extreme(probability_of_extreme)
    , m_mean(mean.count())
    {
    }

    RandomTime()
    : RandomTime(Duration::zero(), 0.0)
    {
    }

    void set_mean(Duration mean) { m_mean.param(typename MeanDistribution::param_type(mean.count())); }

    void set_probability_of_extreme(double p) { m_extreme.param(typename ExtremeDistribution::param_type(p)); }

    template <typename RNG>
    Duration operator()(RNG& rng)
    {
        auto r = m_mean(rng);
        if (m_extreme(rng)) {
            r *= 16;
        }

        return Duration(r);
    }

private:
    ExtremeDistribution m_extreme;
    MeanDistribution    m_mean;
};

template <typename IntType>
class WorkDistributor
{
public:
    struct Segment
    {
        Segment() noexcept
        : first(0)
        , last(0)
        , valid(false)
        {
        }

        Segment(IntType a, IntType b) noexcept
        : first(a)
        , last(b)
        , valid(a <= b)
        {
        }

        explicit operator bool() const noexcept { return valid; }

        IntType first;
        IntType last;
        bool    valid;
    };

    WorkDistributor(IntType nvalues, IntType nthreads) noexcept
    : m_last_start(0)
    , m_nvalues(nvalues)
    , m_nthreads(nthreads)
    {
    }

    // This is a mutable function. This class acts as a generator.
    Segment operator()() noexcept
    {
        if (m_nthreads > 0) {
            const IntType v = m_nvalues / m_nthreads;
            m_nvalues -= v;
            --m_nthreads;
            const IntType start = m_last_start;
            const IntType end   = start + v;
            m_last_start        = end;
            return Segment{ start, end };
        }

        return Segment{};
    }

private:
    IntType m_last_start;
    IntType m_nvalues;
    IntType m_nthreads;
};

template <typename IntType>
WorkDistributor<IntType> make_work_distributor(IntType nvalues, IntType nthreads) noexcept
{
    return WorkDistributor<IntType>{ nvalues, nthreads };
}

template <typename DurationType>
DurationType get_task_duration(int num_producers, int num_consumers)
{
#if 1
    // We have sleeps in our test types to mimic real-world construction/copy costs.
    // This function attempts to scale the sleep time so that all tests complete in k_overall_test_time.
    // This is far from accurate: it doesn't account for overhead, context switching, number of copies, etc.
    // It is good, however, at speeding up the tests.
    const auto test_time      = std::chrono::duration_cast<DurationType>(k_overall_test_time);
    const int  fewest_threads = std::min(num_producers, num_consumers);
    // Assume perfect concurrency...
    const auto time_per_task  = (test_time * fewest_threads) / k_nvalues;
    return time_per_task;
#else
    return DurationType::zero();
#endif
}

template <typename DataType>
class Emplacer
{
public:
    template <typename BufferType, typename RNG>
    void do_emplace(BufferType& buffer, int value, RNG& rng);
};

template <>
class Emplacer<TestDataNoExceptions>
{
    using DurationType = std::chrono::nanoseconds;
    RandomTime<std::chrono::nanoseconds> m_random_delay;

public:
    Emplacer(int num_producers, int num_consumers)
    : m_random_delay()
    {
        m_random_delay.set_mean(get_task_duration<DurationType>(num_producers, num_consumers) / 2);
        m_random_delay.set_probability_of_extreme(0.01);
    }

    template <typename BufferType, typename RNG>
    void do_emplace(BufferType& buffer, int value, RNG& rng)
    {
        buffer.emplace(value, m_random_delay(rng), m_random_delay(rng));
    }
};

template <>
class Emplacer<TestDataExceptions>
{
    using DurationType = std::chrono::nanoseconds;
    RandomTime<std::chrono::nanoseconds> m_random_delay;

public:
    Emplacer(int num_producers, int num_consumers)
    : m_random_delay()
    {
        m_random_delay.set_mean(get_task_duration<DurationType>(num_producers, num_consumers) / 2);
        m_random_delay.set_probability_of_extreme(0.01);
    }

    template <typename BufferType, typename RNG>
    void do_emplace(BufferType& buffer, int value, RNG& rng)
    {
        m_bernoulli_distribution.param(std::bernoulli_distribution::param_type(p_throw_construct));
        const bool throw_construct = m_bernoulli_distribution(rng);
        m_bernoulli_distribution.param(std::bernoulli_distribution::param_type(p_throw_copy_constructor));
        const bool throw_copy_constructor = m_bernoulli_distribution(rng);
        m_bernoulli_distribution.param(std::bernoulli_distribution::param_type(p_throw_copy_assignment));
        const bool throw_copy_assignment = m_bernoulli_distribution(rng);
        m_bernoulli_distribution.param(std::bernoulli_distribution::param_type(p_throw_move_constructor));
        const bool throw_move_constructor = m_bernoulli_distribution(rng);
        m_bernoulli_distribution.param(std::bernoulli_distribution::param_type(p_throw_move_assignment));
        const bool throw_move_assignment = m_bernoulli_distribution(rng);

        buffer.emplace(value,
                       m_random_delay(rng),
                       m_random_delay(rng),
                       throw_construct,
                       throw_copy_constructor,
                       throw_copy_assignment,
                       throw_move_constructor,
                       throw_move_assignment);
    }

private:
    std::bernoulli_distribution m_bernoulli_distribution;
};

template <>
class Emplacer<TestDataCopyExceptions>
{
    using DurationType = std::chrono::nanoseconds;
    RandomTime<std::chrono::nanoseconds> m_random_delay;

public:
    Emplacer(int num_producers, int num_consumers)
    : m_random_delay()
    {
        m_random_delay.set_mean(get_task_duration<DurationType>(num_producers, num_consumers) / 2);
        m_random_delay.set_probability_of_extreme(0.01);
    }

    template <typename BufferType, typename RNG>
    void do_emplace(BufferType& buffer, int value, RNG& rng)
    {
        m_bernoulli_distribution.param(std::bernoulli_distribution::param_type(p_throw_construct));
        const bool throw_construct = m_bernoulli_distribution(rng);
        m_bernoulli_distribution.param(std::bernoulli_distribution::param_type(p_throw_copy_constructor));
        const bool throw_copy_constructor = m_bernoulli_distribution(rng);
        m_bernoulli_distribution.param(std::bernoulli_distribution::param_type(p_throw_copy_assignment));
        const bool throw_copy_assignment = m_bernoulli_distribution(rng);

        buffer.emplace(value,
                       m_random_delay(rng),
                       m_random_delay(rng),
                       throw_construct,
                       throw_copy_constructor,
                       throw_copy_assignment);
    }

private:
    std::bernoulli_distribution m_bernoulli_distribution;
};

template <>
class Emplacer<TestDataMoveExceptions>
{
    using DurationType = std::chrono::nanoseconds;
    RandomTime<std::chrono::nanoseconds> m_random_delay;

public:
    Emplacer(int num_producers, int num_consumers)
    : m_random_delay()
    {
        m_random_delay.set_mean(get_task_duration<DurationType>(num_producers, num_consumers) / 2);
        m_random_delay.set_probability_of_extreme(0.01);
    }

    template <typename BufferType, typename RNG>
    void do_emplace(BufferType& buffer, int value, RNG& rng)
    {
        m_bernoulli_distribution.param(std::bernoulli_distribution::param_type(p_throw_construct));
        const bool throw_construct = m_bernoulli_distribution(rng);
        m_bernoulli_distribution.param(std::bernoulli_distribution::param_type(p_throw_move_constructor));
        const bool throw_move_constructor = m_bernoulli_distribution(rng);
        m_bernoulli_distribution.param(std::bernoulli_distribution::param_type(p_throw_move_assignment));
        const bool throw_move_assignment = m_bernoulli_distribution(rng);

        buffer.emplace(value,
                       m_random_delay(rng),
                       m_random_delay(rng),
                       throw_construct,
                       throw_move_constructor,
                       throw_move_assignment);
    }

private:
    std::bernoulli_distribution m_bernoulli_distribution;
};

template <typename T, typename Traits>
class ProducerTest
{
    using RNG = std::mt19937;

public:
    ProducerTest(int first, int last, buffer_test_t<T, Traits>& buffer) noexcept
    : m_first(first)
    , m_last(last)
    , m_rng()
    , m_buffer(std::addressof(buffer))
    {
        std::seed_seq seed{ first, last };
        m_rng.seed(seed);
    }

    void operator()(int num_producers, int num_consumers)
    {
        Emplacer<T> emplacer(num_producers, num_consumers);

        for (int i = m_first; i < m_last;) {
            try {
                emplacer.do_emplace(*m_buffer, i, m_rng);
                ++i;
            } catch (TestException& e) {
                std::cerr << "Testing exception: " << e.what() << '\n';
            } catch (std::exception& e) {
                std::cerr << "Unexpected exception: " << e.what() << '\n';
                throw;
            } catch (...) {
                std::cerr << "Unknown exception\n";
                throw;
            }
        }
    }

private:
    int                       m_first;
    int                       m_last;
    RNG                       m_rng;
    buffer_test_t<T, Traits>* m_buffer;
};

template <typename T, typename Traits>
class ConsumerTest
{
public:
    ConsumerTest(int nv, buffer_test_t<T, Traits>& buffer)
    : m_values{}
    , m_nvalues(nv)
    , m_buffer(std::addressof(buffer))
    {
    }

    void operator()()
    {
        for (int i = 0; i < m_nvalues; ++i) {
            try {
                const auto v      = m_buffer->pop();
                const int  as_int = v;
                m_values.push_back(as_int);
            } catch (TestException& e) {
                std::cerr << "Testing Exception: " << e.what() << '\n';
            } catch (std::exception& e) {
                std::cerr << "Unknown exception: " << e.what() << '\n';
                throw;
            } catch (...) {
                std::cerr << "Unknown exception\n";
                throw;
            }
        }
    }

    std::vector<int>          m_values;
    int                       m_nvalues;
    buffer_test_t<T, Traits>* m_buffer;
};

template <typename T, typename Traits>
void do_test(int nproducers, int nconsumers)
{
    StopWatch watch;
    watch.start();
    verify_preconditions<Traits>(nproducers, nconsumers);
    if (T::s_constructed != 0) {
        T::s_constructed = 0;
    }

    {
        buffer_test_t<T, Traits> buffer;

        std::cout << "------------------------\n";
        std::cout << "Testing with ";
        print_traits_info<Traits>(std::cout);
        std::cout << "Using resolved lock_policy: ";
        print_lock_traits(std::cout, buffer_test_t<T, Traits>::get_lock_traits());
        std::cout << "With " << nproducers << " producer(s) and " << nconsumers << " consumer(s)" << std::endl;

        std::vector<std::thread> producer_threads;
        producer_threads.reserve(nproducers);

        std::vector<std::thread> consumer_threads;
        consumer_threads.reserve(nconsumers);

        std::vector<ConsumerTest<T, Traits>> consumers;
        auto                                 consumer_distributor = make_work_distributor(k_nvalues, nconsumers);
        while (auto segment = consumer_distributor()) {
            consumers.emplace_back(segment.last - segment.first, buffer);
        }

        auto producer_distributor = make_work_distributor(k_nvalues, nproducers);
        while (auto segment = producer_distributor()) {
            producer_threads.emplace_back(ProducerTest<T, Traits>{ segment.first, segment.last, buffer }, nproducers, nconsumers);
        }

        for (int i = 0; i < nconsumers; ++i) {
            consumer_threads.emplace_back(std::ref(consumers[i]));
        }
        for (int i = 0; i < nproducers; ++i) {
            producer_threads[i].join();
        }
        for (int i = 0; i < nconsumers; ++i) {
            consumer_threads[i].join();
        }

        std::vector<int> results;
        for (int i = 0; i < nconsumers; ++i) {
            results.insert(results.end(), consumers[i].m_values.cbegin(), consumers[i].m_values.cend());
        }
        std::sort(results.begin(), results.end());
        for (int i = 0; i < k_nvalues; ++i) {
            // Make sure we can find i.
            CPPUNIT_ASSERT_EQUAL(i, results[i]);
        }
    }
    CPPUNIT_ASSERT_EQUAL(0, T::s_constructed.load());
    watch.stop();
    std::cout << watch.count() << '\n';
}

struct BatchInsertInputIterator
{
    template <typename BufferType>
    static void do_batch_insert(BufferType& buffer, int /*preferred_batch_size*/)
    {
        using Container     = std::vector<TestDataNoExceptions>;
        using InputIterator = GenericInputIterator<Container>;

        static_assert(
            std::is_same<std::iterator_traits<InputIterator>::iterator_category, std::input_iterator_tag>::value,
            "Trying to test an input iterator for batch insertion");

        std::vector<TestDataNoExceptions> values;
        values.reserve(k_nvalues);
        for (int i = 0; i < k_nvalues; ++i) {
            values.emplace_back(i);
        }

        buffer.push_batch(InputIterator::begin(values), InputIterator::end(values));
    }
};

struct BatchInsertForwardIterator
{
    template <typename BufferType>
    static void do_batch_insert(BufferType& buffer, int /*preferred_batch_size*/)
    {
        using Container = std::forward_list<TestDataNoExceptions>;
        Container data;
        static_assert(std::is_same<std::iterator_traits<typename Container::const_iterator>::iterator_category,
                                   std::forward_iterator_tag>::value,
                      "Trying to test a forward iterator for batch insertion");
        for (int i = k_nvalues; i > 0; --i) {
            data.emplace_front(i - 1);
        }

        buffer.push_batch(data.cbegin(), data.cend());
    }
};

struct BatchInsertRandomAccessIterator
{
    template <typename BufferType>
    static void do_batch_insert(BufferType& buffer, int preferred_batch_size)
    {
        using Container = std::vector<TestDataNoExceptions>;
        Container values;
        static_assert(std::is_same<std::iterator_traits<typename Container::const_iterator>::iterator_category,
                                   std::random_access_iterator_tag>::value,
                      "Trying to test a random access iterator for batch insertion");
        values.reserve(k_nvalues);
        for (int i = 0; i < k_nvalues; ++i) {
            values.emplace_back(i);
        }

        const int batch_size = std::min(k_nvalues, std::min(static_cast<int>(buffer.capacity()), preferred_batch_size));
        auto      first      = values.cbegin();
        auto      next       = first + batch_size;
        while (first != values.cend()) {
            buffer.push_batch(first, next);
            const int nleft = std::distance(next, values.cend());
            const int d     = std::min(nleft, batch_size);
            first           = next;
            std::advance(next, d);
        }
    }
};

struct BatchInsertCompleteRandomAccessIterator
{
    template <typename BufferType>
    static void do_batch_insert(BufferType& buffer, int preferred_batch_size)
    {
        using Container = std::vector<TestDataNoExceptions>;
        Container values;
        static_assert(std::is_same<std::iterator_traits<typename Container::const_iterator>::iterator_category,
                                   std::random_access_iterator_tag>::value,
                      "Trying to test a random access iterator for complete batch insertion");
        values.reserve(k_nvalues);
        for (int i = 0; i < k_nvalues; ++i) {
            values.emplace_back(i);
        }

        buffer.push_batch(values.cbegin(), values.cend());
    }
};

template <typename Traits, typename InsertionTraits>
void test_batch(int nconsumers, int batch_size)
{
    verify_preconditions<Traits>(1, nconsumers);
    std::cout << "Testing batch with ";
    print_traits_info<Traits>(std::cout);
    std::cout << "With " << nconsumers << " consumer(s)" << std::endl;

    static_assert(Traits::producer_traits == ProducerTraits::SINGLE_PRODUCER);

    using BufferType = buffer_test_t<TestDataNoExceptions, Traits>;
    if (TestDataNoExceptions::s_constructed != 0) {
        TestDataNoExceptions::s_constructed = 0;
    }

    try {
        BufferType buffer;

        std::vector<std::thread> consumer_threads;
        consumer_threads.reserve(nconsumers);

        std::vector<ConsumerTest<TestDataNoExceptions, Traits>> consumers;
        auto consumer_distributor = make_work_distributor(k_nvalues, nconsumers);
        while (auto segment = consumer_distributor()) {
            consumers.emplace_back(segment.last - segment.first, buffer);
        }

        for (int i = 0; i < nconsumers; ++i) {
            consumer_threads.emplace_back(std::ref(consumers[i]));
        }

        /////
        InsertionTraits::do_batch_insert(buffer, batch_size);
        /////

        for (int i = 0; i < nconsumers; ++i) {
            consumer_threads[i].join();
        }

        std::vector<int> results;
        for (int i = 0; i < nconsumers; ++i) {
            results.insert(results.end(), consumers[i].m_values.cbegin(), consumers[i].m_values.cend());
        }
        std::sort(results.begin(), results.end());
        for (int i = 0; i < k_nvalues; ++i) {
            // Make sure we can find i.
            CPPUNIT_ASSERT_EQUAL(i, results[i]);
        }
    } catch (const std::bad_alloc& e) {
        CPPUNIT_FAIL("Memory allocation failed");
    } catch (...) {
        CPPUNIT_FAIL("Unknown error");
    }
    CPPUNIT_ASSERT_EQUAL(0, TestDataNoExceptions::s_constructed.load());
    std::cout << "Done with test" << std::endl;
}

} // anonymous namespace

static_assert(std::is_nothrow_move_constructible<TestDataNoExceptions>::value);
static_assert(std::is_nothrow_move_assignable<TestDataNoExceptions>::value);

static_assert(!std::is_nothrow_move_constructible<TestDataExceptions>::value);
static_assert(!std::is_nothrow_move_assignable<TestDataExceptions>::value);

static_assert(std::is_nothrow_move_constructible<TestDataCopyExceptions>::value);
static_assert(std::is_nothrow_move_assignable<TestDataCopyExceptions>::value);

static_assert(!std::is_nothrow_move_constructible<TestDataMoveExceptions>::value);
static_assert(!std::is_nothrow_move_assignable<TestDataMoveExceptions>::value);

static_assert(RingBuffer<TestDataNoExceptions, 4>::get_lock_traits() == LockTraits::SPIN_LOCK);
static_assert(RingBuffer<TestDataExceptions, 4>::get_lock_traits() == LockTraits::LOCK_FREE);
static_assert(RingBuffer<TestDataCopyExceptions, 4>::get_lock_traits() == LockTraits::SPIN_LOCK);
static_assert(RingBuffer<TestDataMoveExceptions, 4>::get_lock_traits() == LockTraits::LOCK_FREE);

static_assert(RingBufferImpl<TestDataNoExceptions, 4, LockFreeTraits>::get_lock_traits() == LockTraits::LOCK_FREE);
static_assert(RingBufferImpl<TestDataExceptions, 4, LockFreeTraits>::get_lock_traits() == LockTraits::LOCK_FREE);
static_assert(RingBufferImpl<TestDataCopyExceptions, 4, LockFreeTraits>::get_lock_traits() == LockTraits::LOCK_FREE);
static_assert(RingBufferImpl<TestDataMoveExceptions, 4, LockFreeTraits>::get_lock_traits() == LockTraits::LOCK_FREE);

static_assert(RingBufferSingleProducer<TestDataNoExceptions, 4>::get_lock_traits() == LockTraits::SPIN_LOCK);
static_assert(RingBufferSingleProducer<TestDataExceptions, 4>::get_lock_traits() == LockTraits::LOCK_FREE);
static_assert(RingBufferSingleProducer<TestDataCopyExceptions, 4>::get_lock_traits() == LockTraits::SPIN_LOCK);
static_assert(RingBufferSingleProducer<TestDataMoveExceptions, 4>::get_lock_traits() == LockTraits::LOCK_FREE);

static_assert(RingBufferSingleConsumer<TestDataNoExceptions, 4>::get_lock_traits() == LockTraits::SPIN_LOCK);
static_assert(RingBufferSingleConsumer<TestDataExceptions, 4>::get_lock_traits() == LockTraits::LOCK_FREE);
static_assert(RingBufferSingleConsumer<TestDataCopyExceptions, 4>::get_lock_traits() == LockTraits::SPIN_LOCK);
static_assert(RingBufferSingleConsumer<TestDataMoveExceptions, 4>::get_lock_traits() == LockTraits::LOCK_FREE);

static_assert(RingBufferSingleProducerSingleConsumer<TestDataNoExceptions, 4>::get_lock_traits() ==
              LockTraits::SPIN_LOCK);
static_assert(RingBufferSingleProducerSingleConsumer<TestDataExceptions, 4>::get_lock_traits() ==
              LockTraits::LOCK_FREE);
static_assert(RingBufferSingleProducerSingleConsumer<TestDataCopyExceptions, 4>::get_lock_traits() ==
              LockTraits::SPIN_LOCK);
static_assert(RingBufferSingleProducerSingleConsumer<TestDataMoveExceptions, 4>::get_lock_traits() ==
              LockTraits::LOCK_FREE);

void TestRingBuffer::testSingleItemProducer()
{
    const int total_threads = std::thread::hardware_concurrency();
    const int nproducers    = total_threads / 2;
    const int nconsumers    = total_threads - nproducers;

    do_test<TestDataNoExceptions, DefaultRingBufferTraits>(nproducers, nconsumers);
    do_test<TestDataNoExceptions, SingleProducerSingleConsumerRingBufferTraits>(1, 1);
    do_test<TestDataNoExceptions, SingleProducerRingBufferTraits>(1, total_threads);
    do_test<TestDataNoExceptions, SingleConsumerRingBufferTraits>(total_threads, 1);

    do_test<TestDataExceptions, DefaultRingBufferTraits>(nproducers, nconsumers);
    do_test<TestDataExceptions, SingleProducerSingleConsumerRingBufferTraits>(1, 1);
    do_test<TestDataExceptions, SingleProducerRingBufferTraits>(1, total_threads);
    do_test<TestDataExceptions, SingleConsumerRingBufferTraits>(total_threads, 1);

    do_test<TestDataCopyExceptions, DefaultRingBufferTraits>(nproducers, nconsumers);
    do_test<TestDataCopyExceptions, SingleProducerSingleConsumerRingBufferTraits>(1, 1);
    do_test<TestDataCopyExceptions, SingleProducerRingBufferTraits>(1, total_threads);
    do_test<TestDataCopyExceptions, SingleConsumerRingBufferTraits>(total_threads, 1);

    do_test<TestDataMoveExceptions, DefaultRingBufferTraits>(nproducers, nconsumers);
    do_test<TestDataMoveExceptions, SingleProducerSingleConsumerRingBufferTraits>(1, 1);
    do_test<TestDataMoveExceptions, SingleProducerRingBufferTraits>(1, total_threads);
    do_test<TestDataMoveExceptions, SingleConsumerRingBufferTraits>(total_threads, 1);
}

void TestRingBuffer::testBatchItemProducer()
{
    constexpr int batch_size = 31; // Prime

    const int total_threads = std::thread::hardware_concurrency();
    const int nconsumers    = std::max(1, total_threads - 1);
    test_batch<SingleProducerRingBufferTraits, BatchInsertInputIterator>(nconsumers, batch_size);
    test_batch<SingleProducerRingBufferTraits, BatchInsertForwardIterator>(nconsumers, batch_size);
    test_batch<SingleProducerRingBufferTraits, BatchInsertRandomAccessIterator>(nconsumers, batch_size);
    test_batch<SingleProducerRingBufferTraits, BatchInsertCompleteRandomAccessIterator>(nconsumers, batch_size);
    test_batch<SingleProducerSingleConsumerRingBufferTraits, BatchInsertInputIterator>(1, batch_size);
    test_batch<SingleProducerSingleConsumerRingBufferTraits, BatchInsertForwardIterator>(1, batch_size);
    test_batch<SingleProducerSingleConsumerRingBufferTraits, BatchInsertRandomAccessIterator>(1, batch_size);
    test_batch<SingleProducerSingleConsumerRingBufferTraits, BatchInsertCompleteRandomAccessIterator>(1, batch_size);
}

