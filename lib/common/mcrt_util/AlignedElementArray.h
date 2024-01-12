// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cassert>
#include <cstddef>
#include <iterator>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include <stdlib.h> // posix_memalign


namespace moonray {
namespace util {

namespace AEAImpl {
#if defined(__cpp_lib_void_t)
template <class... A>
using void_t = std::void_t<A...>;
#else
template <class...>
using void_t = void;
#endif
} // namespace AEAImpl

template <typename T, typename = void>
struct IsIterator : std::false_type { };

template <typename T>
struct IsIterator<T, AEAImpl::void_t<typename std::iterator_traits<T>::iterator_category>> : std::true_type { };

template <typename T>
constexpr bool isInputIterator(std::false_type)
{
    return false;
}

template <typename T>
constexpr bool isInputIterator(std::true_type)
{
    return std::is_base_of<std::input_iterator_tag, typename std::iterator_traits<T>::iterator_category>::value;
}

template <typename T>
constexpr bool isInputIterator()
{
    return isInputIterator<T>(IsIterator<T>{});
}

// TODO: move this to scene_rdl and deal with 32/64-bit overloading
template <typename Integer>
constexpr Integer roundUpPower2(Integer v) noexcept
{
    static_assert(std::is_integral<Integer>::value, "Only works on integers");
    static_assert(sizeof(Integer) == 8, "Only works on 64-bit integers");

    --v;
    v |= v >>  1;
    v |= v >>  2;
    v |= v >>  4;
    v |= v >>  8;
    v |= v >> 16;
    v |= v >> 32;
    ++v;
    return v;
}

template <typename T, std::size_t desired_alignment = alignof(T)>
class AlignedElementArray
{
    static_assert(alignof(T) <= desired_alignment, "The alignment does not meet the minimum requirement");
    static constexpr std::size_t kAlignment = std::max(sizeof(void*), roundUpPower2(desired_alignment));

public:
    using value_type      = T;
    using size_type       = std::size_t;
    using difference_type = std::ptrdiff_t;
    using reference       = value_type&;
    using const_reference = const value_type&;
    using pointer         = value_type*;
    using const_pointer   = const value_type*;

    template <typename ContainerType>
    class iterator_base
    {
    public:
        using value_type        = T;
        using difference_type   = std::ptrdiff_t;
        using iterator_category = std::random_access_iterator_tag;
        using reference         = value_type&;
        using pointer           = value_type*;
        using const_reference   = const value_type&;
        using const_pointer     = const value_type*;

        iterator_base() noexcept
        : mArray(nullptr)
        {
            // mIdx not initialized, because, frankly, we don't care in the
            // default initializer case. mArray will always compare null.
        }

        iterator_base(ContainerType& array, difference_type idx) noexcept
        : mArray(std::addressof(array))
        , mIdx(idx)
        {
        }

    protected:
        ~iterator_base() = default;

    public:

        reference access() noexcept
        {
            return (*mArray)[mIdx];
        }

        const_reference access() const noexcept
        {
            return (*mArray)[mIdx];
        }

        reference access(difference_type n) noexcept
        {
            return (*mArray)[mIdx + n];
        }

        const_reference access(difference_type n) const noexcept
        {
            return (*mArray)[mIdx + n];
        }

        iterator_base& operator++() noexcept
        {
            ++mIdx;
            return *this;
        }

        iterator_base operator++(int) noexcept
        {
            iterator_base cpy(*this);
            this->operator++();
            return cpy;
        }

        iterator_base& operator--() noexcept
        {
            --mIdx;
            return *this;
        }

        iterator_base operator--(int) noexcept
        {
            iterator_base cpy(*this);
            this->operator--();
            return cpy;
        }

        iterator_base& operator+=(difference_type n) noexcept
        {
            mIdx += n;
            return *this;
        }

        iterator_base& operator-=(difference_type n) noexcept
        {
            mIdx -= n;
            return *this;
        }

        friend bool comp_equal(const iterator_base& a, const iterator_base& b) noexcept
        {
            return a.mArray == b.mArray && a.mIdx == b.mIdx;
        }

        friend bool comp_less_equal(const iterator_base& a, const iterator_base& b) noexcept
        {
            assert(a.mArray == b.mArray);
            return a.mIdx <= b.mIdx;
        }

        friend difference_type difference(const iterator_base& a, const iterator_base& b) noexcept
        {
            assert(a.mArray == b.mArray);
            return a.mIdx - b.mIdx;
        }

    private:
        ContainerType* mArray;
        difference_type mIdx;
    };

    class iterator : private iterator_base<AlignedElementArray>
    {
        using Base = iterator_base<AlignedElementArray>;
    public:
        using typename Base::value_type;
        using typename Base::difference_type;
        using typename Base::iterator_category;
        using reference = typename Base::reference;
        using pointer   = typename Base::pointer;

        using Base::Base;
        using Base::operator++;
        using Base::operator--;
        using Base::operator+=;
        using Base::operator-=;

        reference operator*() noexcept
        {
            return Base::access();
        }

        pointer operator->() noexcept
        {
            return std::addressof(Base::access());
        }

        reference operator[](difference_type n) noexcept
        {
            return Base::access(n);
        }

        friend iterator operator+(const iterator& a, difference_type n) noexcept
        {
            iterator result(a);
            result += n;
            return result;
        }

        friend iterator operator+(difference_type n, iterator& a) noexcept
        {
            return a + n;
        }

        friend iterator operator-(const iterator& a, difference_type n) noexcept
        {
            iterator result(a);
            result -= n;
            return result;
        }

        friend difference_type operator-(const iterator& a, const iterator& b) noexcept
        {
            return difference(a, b);
        }

        friend bool operator==(const iterator& a, const iterator& b) noexcept
        {
            return comp_equal(a, b);
        }

        friend bool operator!=(const iterator& a, const iterator& b) noexcept
        {
            return !(a == b);
        }

        friend bool operator<=(const iterator& a, const iterator& b) noexcept
        {
            return comp_less_equal(a, b);
        }

        friend bool operator> (const iterator& a, const iterator& b) noexcept
        {
            return !(a <= b);
        }

        friend bool operator< (const iterator& a, const iterator& b) noexcept
        {
            return (b > a);
        }

        friend bool operator>=(const iterator& a, const iterator& b) noexcept
        {
            return !(a < b);
        }
    };

    class const_iterator : private iterator_base<const AlignedElementArray>
    {
        using Base = iterator_base<const AlignedElementArray>;
    public:
        using typename Base::value_type;
        using typename Base::difference_type;
        using typename Base::iterator_category;
        using reference = typename Base::const_reference;
        using pointer   = typename Base::const_pointer;

        using Base::Base;
        using Base::operator++;
        using Base::operator--;
        using Base::operator+=;
        using Base::operator-=;

        reference operator*() const noexcept
        {
            return Base::access();
        }

        pointer operator->() const noexcept
        {
            return std::addressof(Base::access());
        }

        reference operator[](difference_type n) const noexcept
        {
            return Base::access(n);
        }

        friend const_iterator operator+(const const_iterator& a, difference_type n) noexcept
        {
            const_iterator result(a);
            result += n;
            return result;
        }

        friend const_iterator operator+(difference_type n, const_iterator& a) noexcept
        {
            return a + n;
        }

        friend const_iterator operator-(const const_iterator& a, difference_type n) noexcept
        {
            const_iterator result(a);
            result -= n;
            return result;
        }

        friend difference_type operator-(const const_iterator& a, const const_iterator& b) noexcept
        {
            return difference(a, b);
        }

        friend bool operator==(const const_iterator& a, const const_iterator& b) noexcept
        {
            return comp_equal(a, b);
        }

        friend bool operator!=(const const_iterator& a, const const_iterator& b) noexcept
        {
            return !(a == b);
        }

        friend bool operator<=(const const_iterator& a, const const_iterator& b) noexcept
        {
            return comp_less_equal(a, b);
        }

        friend bool operator> (const const_iterator& a, const const_iterator& b) noexcept
        {
            return !(a <= b);
        }

        friend bool operator< (const const_iterator& a, const const_iterator& b) noexcept
        {
            return (b > a);
        }

        friend bool operator>=(const const_iterator& a, const const_iterator& b) noexcept
        {
            return !(a < b);
        }
    };

    explicit AlignedElementArray(std::size_t n)
    : mSize(0)
    , mData(static_cast<ElementType*>(doAllocate(n * sizeof(ElementType))))
    {
        try {
            for ( ; mSize < n; ++mSize) {
                new (getPointer(mSize)) T();
            }
        } catch (...) {
            destroy(mSize);
            throw;
        }
    }

    template <typename U>
    explicit AlignedElementArray(std::size_t n, const U& init)
    : mSize(0)
    , mData(static_cast<ElementType*>(doAllocate(n * sizeof(ElementType))))
    {
        try {
            for ( ; mSize < n; ++mSize) {
                new (getPointer(mSize)) T(init);
            }
        } catch (...) {
            destroy(mSize);
            throw;
        }
    }

    template <typename Iterator, std::enable_if_t<isInputIterator<Iterator>(), bool> = true>
    explicit AlignedElementArray(Iterator first, Iterator last)
    : mSize(0)
    // This is only efficient for random access iterators, and won't work at all for input iterators
    , mData(static_cast<ElementType*>(doAllocate(std::distance(first, last) * sizeof(ElementType))))
    {
        try {
            for ( ; first != last; ++first) {
                new (getPointer(mSize++)) T(*first);
            }
        } catch (...) {
            destroy(mSize);
            throw;
        }
    }

    AlignedElementArray(AlignedElementArray&& other) noexcept
    : mSize(std::exchange(other.mSize, 0))
    , mData(std::exchange(other.mData, nullptr))
    {
    }

    AlignedElementArray(const AlignedElementArray& other)
    : mSize(0)
    , mData(static_cast<ElementType*>(doAllocate(other.mSize * sizeof(ElementType))))
    {
        try {
            for ( ; mSize < other.mSize; ++mSize) {
                new (getPointer(mSize)) T(other[mSize]);
            }
        } catch (...) {
            destroy(mSize);
            throw;
        }
    }

    ~AlignedElementArray()
    {
        destroy(mSize);
    }

    constexpr std::size_t alignment() const noexcept
    {
        return kAlignment;
    }

    AlignedElementArray& operator=(AlignedElementArray&& other) noexcept
    {
        this->swap(other);
        return *this;
    }

    AlignedElementArray& operator=(const AlignedElementArray& other)
    {
        AlignedElementArray tmp(other);
        this->swap(tmp);
        return *this;
    }

    reference at(std::size_t i)
    {
        if (i >= mSize) {
            throw std::out_of_range{"Bad range for at"};
        }
        return *getPointer(i);
    }

    const_reference at(std::size_t i) const
    {
        if (i >= mSize) {
            throw std::out_of_range{"Bad range for at"};
        }
        return *getPointer(i);
    }

    reference operator[](std::size_t i) noexcept
    {
        return *getPointer(i);
    }

    const_reference operator[](std::size_t i) const noexcept
    {
        return *getPointer(i);
    }

    std::size_t size() const noexcept
    {
        return mSize;
    }

    bool empty() const noexcept
    {
        return mSize == 0;
    }

    void swap(AlignedElementArray& other) noexcept
    {
        using std::swap; // Allow ADL
        swap(mSize, other.mSize);
        swap(mData, other.mData);
    }

    iterator begin() noexcept
    {
        return iterator(*this, 0);
    }

    iterator end() noexcept
    {
        return iterator(*this, mSize);
    }

    const_iterator begin() const noexcept
    {
        return const_iterator(*this, 0);
    }

    const_iterator end() const noexcept
    {
        return const_iterator(*this, mSize);
    }

    const_iterator cbegin() const noexcept
    {
        return const_iterator(*this, 0);
    }

    const_iterator cend() const noexcept
    {
        return const_iterator(*this, mSize);
    }

private:
    [[gnu::alloc_size(1)]] [[gnu::malloc]] static void* doAllocate(std::size_t bytes)
    {
        void* mem;
        if (int error = posix_memalign(std::addressof(mem), kAlignment, bytes)) {
            if (error == EINVAL) {
                throw std::invalid_argument{"Bad alignment"};
            } else {
                throw std::bad_alloc{};
            }
        }
        return mem;
    };

    static void doFree(void* ptr) noexcept
    {
        free(ptr);
    }

    static T* checkAlignment(T* p) noexcept
    {
        assert(reinterpret_cast<std::uintptr_t>(p) % kAlignment == 0);
        return p;
    }

    static const T* checkAlignment(const T* p) noexcept
    {
        assert(reinterpret_cast<std::uintptr_t>(p) % kAlignment == 0);
        return p;
    }

    T* getPointer(std::size_t i) noexcept
    {
#if defined(__cpp_lib_launder)
        return checkAlignment(std::launder(reinterpret_cast<T*>(mData + i)));
#else
        return checkAlignment(reinterpret_cast<T*>(mData + i));
#endif
    }

    const T* getPointer(std::size_t i) const noexcept
    {
#if defined(__cpp_lib_launder)
        return checkAlignment(std::launder(reinterpret_cast<const T*>(mData + i)));
#else
        return checkAlignment(reinterpret_cast<const T*>(mData + i));
#endif
    }

    void destroy(std::size_t n) noexcept
    {
        for (std::size_t i = 0; i < n; ++i) {
            getPointer(i)->~T();
        }
        doFree(mData);
    }

    using ElementType = typename std::aligned_storage<sizeof(T), kAlignment>::type;

    std::size_t mSize;
    ElementType* mData;
};

} // namespace util
} // namespace moonray

