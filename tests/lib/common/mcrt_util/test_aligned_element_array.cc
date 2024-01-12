// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#include "test_aligned_element_array.h"

#include <moonray/common/mcrt_util/AlignedElementArray.h>

#include <numeric>

CPPUNIT_TEST_SUITE_REGISTRATION(TestAlignedElementArray);

struct Constructable
{
    Constructable() noexcept
    : mVal(42)
    {
        ++mCount;
    }

    explicit Constructable(int n) noexcept
    : mVal(n)
    {
        ++mCount;
    }

    Constructable(const Constructable& other) noexcept
    : mVal(other.mVal)
    {
        ++mCount;
    }

    Constructable(Constructable&&) noexcept = default;
    Constructable& operator=(const Constructable&) = default;
    Constructable& operator=(Constructable&&) = default;

    ~Constructable() { --mCount; }

    static int mCount;
    int mVal;
};

struct NonDefaultConstructable
{
    NonDefaultConstructable() = delete;
    explicit NonDefaultConstructable(int n) noexcept
    : mVal(n)
    {
        ++mCount;
    }

    NonDefaultConstructable(const NonDefaultConstructable& other) noexcept
    : mVal(other.mVal)
    {
        ++mCount;
    }

    NonDefaultConstructable(NonDefaultConstructable&&) noexcept = default;
    NonDefaultConstructable& operator=(const NonDefaultConstructable&) = default;
    NonDefaultConstructable& operator=(NonDefaultConstructable&&) = default;

    ~NonDefaultConstructable() { --mCount; }

    static int mCount;
    int mVal;
};

struct NonCopyable
{
    NonCopyable() noexcept
    : mVal(42)
    {
        ++mCount;
    }

    explicit NonCopyable(int n) noexcept
    : mVal(n)
    {
        ++mCount;
    }

    NonCopyable(const NonCopyable& other) = delete;

    NonCopyable(NonCopyable&&) noexcept = default;
    NonCopyable& operator=(const NonCopyable&) = delete;
    NonCopyable& operator=(NonCopyable&&) = default;

    ~NonCopyable() { --mCount; }

    static int mCount;
    int mVal;
};

int Constructable::mCount = 0;
int NonCopyable::mCount = 0;
int NonDefaultConstructable::mCount = 0;

template <typename T>
void testGenericHelper()
{
    CPPUNIT_ASSERT_EQUAL(T::mCount, 0);
    {
        constexpr int length = 5;
        moonray::util::AlignedElementArray<T> a0(length, 99);

        for (int i = 0; i < length; ++i) {
            CPPUNIT_ASSERT_EQUAL(a0[i].mVal, 99);
            CPPUNIT_ASSERT_EQUAL(a0.at(i).mVal, 99);
        }

        moonray::util::AlignedElementArray<T> a1(std::move(a0));
        for (int i = 0; i < length; ++i) {
            CPPUNIT_ASSERT_EQUAL(a1[i].mVal, 99);
            CPPUNIT_ASSERT_EQUAL(a1.at(i).mVal, 99);
        }

        for (const auto& x: a1) {
            CPPUNIT_ASSERT_EQUAL(x.mVal, 99);
        }
        for (auto it = a1.begin(); it != a1.end(); ++it) {
            CPPUNIT_ASSERT_EQUAL(it->mVal, 99);
        }
        for (auto it = a1.cbegin(); it != a1.cend(); ++it) {
            CPPUNIT_ASSERT_EQUAL(it->mVal, 99);
        }
    }
    CPPUNIT_ASSERT_EQUAL(T::mCount, 0);
}

void TestAlignedElementArray::testGeneric()
{
    testGenericHelper<Constructable>();
    testGenericHelper<NonCopyable>();
    testGenericHelper<NonDefaultConstructable>();
}

void TestAlignedElementArray::testConstructors()
{
    CPPUNIT_ASSERT_EQUAL(Constructable::mCount, 0);
    {
        constexpr int length = 5;
        moonray::util::AlignedElementArray<Constructable> a0(length);
        moonray::util::AlignedElementArray<Constructable> a1(length, 99);

        std::vector<int> v0(length);
        std::iota(v0.begin(), v0.end(), 0);

        moonray::util::AlignedElementArray<Constructable> a2(v0.cbegin(), v0.cend());
        moonray::util::AlignedElementArray<Constructable> a3(a2);

        for (int i = 0; i < length; ++i) {
            CPPUNIT_ASSERT_EQUAL(a0[i].mVal, 42);
            CPPUNIT_ASSERT_EQUAL(a0.at(i).mVal, 42);
            CPPUNIT_ASSERT_EQUAL(a1[i].mVal, 99);
            CPPUNIT_ASSERT_EQUAL(a1.at(i).mVal, 99);
            CPPUNIT_ASSERT_EQUAL(a2[i].mVal, i);
            CPPUNIT_ASSERT_EQUAL(a2.at(i).mVal, i);
            CPPUNIT_ASSERT_EQUAL(a3[i].mVal, i);
            CPPUNIT_ASSERT_EQUAL(a3.at(i).mVal, i);
        }

        moonray::util::AlignedElementArray<Constructable> a4(std::move(a2));
        for (int i = 0; i < length; ++i) {
            CPPUNIT_ASSERT_EQUAL(a4[i].mVal, i);
            CPPUNIT_ASSERT_EQUAL(a4.at(i).mVal, i);
        }
    }
    CPPUNIT_ASSERT_EQUAL(Constructable::mCount, 0);
}

void TestAlignedElementArray::testIterators()
{
    CPPUNIT_ASSERT_EQUAL(Constructable::mCount, 0);
    {
        constexpr int length = 5;
        std::vector<int> v0(length);
        std::iota(v0.begin(), v0.end(), 0);

        moonray::util::AlignedElementArray<Constructable> a0(v0.cbegin(), v0.cend());
        const moonray::util::AlignedElementArray<Constructable> a1(v0.cbegin(), v0.cend());

        int count = 0;
        for (const auto& x: a0) {
            CPPUNIT_ASSERT_EQUAL(x.mVal, count++);
        }
        CPPUNIT_ASSERT_EQUAL(count, length);

        for (int i = 0; i < length; ++i) {
            CPPUNIT_ASSERT_EQUAL(a0.begin()[i].mVal, i);
            CPPUNIT_ASSERT_EQUAL(a1.begin()[i].mVal, i);
            CPPUNIT_ASSERT_EQUAL((a0.begin() + i)->mVal, i);
            CPPUNIT_ASSERT_EQUAL((a1.begin() + i)->mVal, i);
        }

        for (auto& x: a0) {
            x.mVal = 99;
        }
        for (const auto& x: a0) {
            CPPUNIT_ASSERT_EQUAL(x.mVal, 99);
        }

        for (auto it = a0.begin(); it != a0.end(); ++it) {
            CPPUNIT_ASSERT_EQUAL(it->mVal, 99);
        }

        for (auto it = a0.cbegin(); it != a0.cend(); ++it) {
            CPPUNIT_ASSERT_EQUAL(it->mVal, 99);
        }

        for (auto it = a0.begin(); it < a0.end(); ++it) {
            CPPUNIT_ASSERT_EQUAL(it->mVal, 99);
        }

        for (auto it = a0.cbegin(); it < a0.cend(); ++it) {
            CPPUNIT_ASSERT_EQUAL(it->mVal, 99);
        }

        count = 0;
        for (auto it = a1.begin(); it != a1.end(); ++it) {
            CPPUNIT_ASSERT_EQUAL(it->mVal, count++);
        }

        count = 0;
        for (auto it = a1.cbegin(); it != a1.cend(); ++it) {
            CPPUNIT_ASSERT_EQUAL(it->mVal, count++);
        }

        count = 0;
        for (auto it = a1.begin(); it < a1.end(); ++it) {
            CPPUNIT_ASSERT_EQUAL(it->mVal, count++);
        }

        count = 0;
        for (auto it = a1.cbegin(); it < a1.cend(); ++it) {
            CPPUNIT_ASSERT_EQUAL(it->mVal, count++);
        }

        CPPUNIT_ASSERT(  a0.begin() == a0.begin());
        CPPUNIT_ASSERT(!(a0.begin() != a0.begin()));
        CPPUNIT_ASSERT(!(a0.begin()  < a0.begin()));
        CPPUNIT_ASSERT(!(a0.begin()  > a0.begin()));
        CPPUNIT_ASSERT(  a0.begin() <= a0.begin());
        CPPUNIT_ASSERT(  a0.begin() >= a0.begin());

        CPPUNIT_ASSERT(!(a0.begin() == std::next(a0.begin())));
        CPPUNIT_ASSERT(  a0.begin() != std::next(a0.begin()));
        CPPUNIT_ASSERT(  a0.begin()  < std::next(a0.begin()));
        CPPUNIT_ASSERT(!(a0.begin()  > std::next(a0.begin())));
        CPPUNIT_ASSERT(  a0.begin() <= std::next(a0.begin()));
        CPPUNIT_ASSERT(!(a0.begin() >= std::next(a0.begin())));

        CPPUNIT_ASSERT(  a1.begin() == a1.begin());
        CPPUNIT_ASSERT(!(a1.begin() != a1.begin()));
        CPPUNIT_ASSERT(!(a1.begin()  < a1.begin()));
        CPPUNIT_ASSERT(!(a1.begin()  > a1.begin()));
        CPPUNIT_ASSERT(  a1.begin() <= a1.begin());
        CPPUNIT_ASSERT(  a1.begin() >= a1.begin());

        CPPUNIT_ASSERT(!(a1.begin() == std::next(a1.begin())));
        CPPUNIT_ASSERT(  a1.begin() != std::next(a1.begin()));
        CPPUNIT_ASSERT(  a1.begin()  < std::next(a1.begin()));
        CPPUNIT_ASSERT(!(a1.begin()  > std::next(a1.begin())));
        CPPUNIT_ASSERT(  a1.begin() <= std::next(a1.begin()));
        CPPUNIT_ASSERT(!(a1.begin() >= std::next(a1.begin())));
    }
    CPPUNIT_ASSERT_EQUAL(Constructable::mCount, 0);
}

void TestAlignedElementArray::testAlignment()
{
    constexpr size_t desiredAlignment = 32;

    auto isAligned = [desiredAlignment](auto& value) {
        return reinterpret_cast<std::uintptr_t>(std::addressof(value)) % desiredAlignment == 0;
    };

    constexpr int numValues = 101;
    std::vector<int> initValues(numValues);
    std::iota(initValues.begin(), initValues.end(), 0);

    CPPUNIT_ASSERT_EQUAL(NonDefaultConstructable::mCount, 0);
    {
        moonray::util::AlignedElementArray<NonDefaultConstructable,desiredAlignment> array(initValues.cbegin(),
                                                                                           initValues.cend());

        for (int i = 0; i < numValues; ++i) {
            CPPUNIT_ASSERT_EQUAL(i, array[i].mVal);
            CPPUNIT_ASSERT(isAligned(array[i].mVal));
        }
    }
    CPPUNIT_ASSERT_EQUAL(NonDefaultConstructable::mCount, 0);
}

