// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file StaticVector.h
/// $Id$
///

#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace moonray {
namespace util {

template <typename T, std::size_t N>
class StaticVector
{
public:
    typedef T                                     value_type;
    typedef std::size_t                           size_type;
    typedef std::ptrdiff_t                        difference_type;
    typedef value_type&                           reference;
    typedef const value_type&                     const_reference;
    typedef value_type*                           pointer;
    typedef const value_type*                     const_pointer;
    typedef pointer                               iterator;
    typedef const_pointer                         const_iterator;
    typedef std::reverse_iterator<iterator>       reverse_iterator;
    typedef std::reverse_iterator<const_iterator> const_reverse_iterator;

    constexpr StaticVector() noexcept : mData{}, mSize(0)
    {
    }

    explicit StaticVector(size_type n) : mData{}, mSize(0)
    {
        assert(n <= N);
        for ( ; mSize < n; ++mSize) {
            pointer const addr = get_address(mSize);
            new(addr) T();
        }
    }

    StaticVector(size_type n, const T& value) : mData{}, mSize(0)
    {
        initialize(n, value, std::true_type());
    }

    template <typename InputIt>
    StaticVector(InputIt first, InputIt last) : mData{}, mSize(0)
    // There can be an ambiguity here if somebody creates
    // StaticVector<int>(10, 3), so we have to dispatch at compile time to get
    // the correct behavior for that case and the iterator case.
    {
        initialize(first, last, std::is_integral<InputIt>());
    }

    StaticVector(const StaticVector& other) : mData{}, mSize(0)
    {
        std::copy(other.begin(), other.end(), std::back_inserter(*this));
    }

    StaticVector(StaticVector&& other) noexcept(std::is_nothrow_move_constructible<T>::value) :
        mData{}, mSize(0)
    {
        std::move(other.begin(), other.end(), std::back_inserter(*this));
    }

    StaticVector(std::initializer_list<T> init) : mData{}, mSize(0)
    {
        for (const auto& t : init) {
            // Copying an initializer_list is shallow (n3290 18.9.2).
            // Therefore, we shouldn't move from the list. Perhaps an r-value
            // reference constructor would make sense, but std::vector doesn't
            // have one.
            push_back(t);
        }
    }

    ~StaticVector()
    {
        for (size_type i = 0; i < mSize; ++i) {
            operator[](i).~T();
        }
    }

    StaticVector& operator=(const StaticVector& other)
    {
        clear();
        std::copy(other.cbegin(), other.cend(), std::back_inserter(*this));
        return *this;
    }

    StaticVector& operator=(StaticVector&& other) noexcept(std::is_nothrow_move_assignable<T>::value)
    {
        clear();
        std::move(other.begin(), other.end(), std::back_inserter(*this));
        return *this;
    }

    void assign(size_type n, const T& value)
    {
        assert(n <= N);
        clear();
        for (size_type i = 0; i < n; ++i) {
            push_back(value);
        }
    }

    template <typename InputIt>
    void assign(InputIt first, InputIt last)
    {
        clear();
        initialize(first, last, std::is_integral<InputIt>());
    }
    
    ///////////////////////////////////////////////////////////////////////////
    // Element Access
    ///////////////////////////////////////////////////////////////////////////

    reference at(size_type n)
    {
        if (n >= size()) {
            throw std::out_of_range("Element access out of range.");
        }
        return operator[](n);
    }

    const_reference at(size_type n) const
    {
        if (n >= size()) {
            throw std::out_of_range("Element access out of range.");
        }
        return operator[](n);
    }

    reference operator[](size_type n)
    {
        assert(n < size());
        return reinterpret_cast<reference>(mData[n]);
    }

    const_reference operator[](size_type n) const
    {
        assert(n < size());
        return reinterpret_cast<const_reference>(mData[n]);
    }

    reference front()
    {
        assert(!empty());
        return operator[](0);
    }

    const_reference front() const
    {
        assert(!empty());
        return operator[](0);
    }

    reference back()
    {
        assert(!empty());
        return operator[](size()-1);
    }

    const_reference back() const
    {
        assert(!empty());
        return operator[](size()-1);
    }

    pointer data() noexcept
    {
        return get_address(0);
    }

    const_pointer data() const noexcept
    {
        return get_address(0);
    }

    ///////////////////////////////////////////////////////////////////////////
    // Iterators
    ///////////////////////////////////////////////////////////////////////////

    iterator begin() noexcept
    {
        return get_address(0);
    }

    iterator end() noexcept
    {
        return get_address(mSize);
    }

    const_iterator begin() const noexcept
    {
        return get_address(0);
    }

    const_iterator end() const noexcept
    {
        return get_address(mSize);
    }

    const_iterator cbegin() const noexcept
    {
        return get_address(0);
    }

    const_iterator cend() const noexcept
    {
        return get_address(mSize);
    }

    reverse_iterator rbegin() noexcept
    {
        return std::reverse_iterator<iterator>(end());
    }

    reverse_iterator rend() noexcept
    {
        return std::reverse_iterator<iterator>(begin());
    }

    const_reverse_iterator rbegin() const noexcept
    {
        return std::reverse_iterator<const_iterator>(end());
    }

    const_reverse_iterator rend() const noexcept
    {
        return std::reverse_iterator<const_iterator>(begin());
    }

    const_reverse_iterator crbegin() const noexcept
    {
        return std::reverse_iterator<const_iterator>(end());
    }

    const_reverse_iterator crend() const noexcept
    {
        return std::reverse_iterator<const_iterator>(begin());
    }

    ///////////////////////////////////////////////////////////////////////////
    // Capacity
    ///////////////////////////////////////////////////////////////////////////

    bool empty() const noexcept
    {
        return mSize == 0;
    }

    size_type size() const noexcept
    {
        return mSize;
    }

    constexpr size_type max_size() const noexcept
    {
        return N;
    }

    constexpr size_type capacity() const noexcept
    {
        return N;
    }

    ///////////////////////////////////////////////////////////////////////////
    // Modifiers
    ///////////////////////////////////////////////////////////////////////////

    void clear() noexcept
    {
        std::for_each(begin(), end(), [](T& t) { t.~T(); });
        mSize = 0;
    }

    iterator insert(const_iterator pos, const T& value)
    {
        return insert(pos, 1, value, std::true_type());
    }

    iterator insert(const_iterator pos, size_type count, const T& value)
    {
        assert(mSize + count < N);

        iterator pc(pos - cbegin() + begin());
        std::move_backward(pc, end(), end()+count);

        for (iterator first = pc, last = pc + count; first != last; ++first) {
            new(std::addressof(*first)) T(value);
            ++mSize;
        }
        return pc;
    }

    iterator insert(const_iterator pos, T&& value)
    {
        assert(mSize + 1 < N);

        iterator pc(pos - cbegin() + begin());
        std::move_backward(pc, end(), end()+1);
        new(std::addressof(*pc)) T(std::move(value));
        ++mSize;
        return pc;
    }

    template <typename InputIt>
    iterator insert(const_iterator pos, InputIt first, InputIt last)
    {
        return insert(pos, first, last, std::is_integral<InputIt>());
    }

    //iterator insert(const_iterator pos, std::initializer_list<T> ilist);

    template <typename... Args>
    iterator emplace(const_iterator pos, Args&&... args)
    {
        assert(mSize + 1 < N);

        iterator pc(pos - cbegin() + begin());
        std::move_backward(pc, end(), end()+1);
        new(std::addressof(*pc)) T(std::forward<Args>(args)...);
        ++mSize;
        return pc;
    }

    iterator erase(const_iterator pos)
    {
        iterator pc(pos - cbegin() + begin());
        auto it = std::move(pc+1, end(), pc);
        it->~T();
        --mSize;
        return pc;
    }

    iterator erase(const_iterator first, const_iterator last)
    {
        iterator pfirst(first - cbegin() + begin());
        if (first >= last) {
            return pfirst;
        }
        iterator plast(last - cbegin() + begin());
        auto it = std::move(plast, end(), pfirst);
        for ( ; it != end(); ++it) {
            it->~T();
        }
        mSize -= std::distance(first, last);
        return pfirst;
    }

    void push_back(const T& value)
    {
        assert(mSize < N);
        pointer const addr = get_address(mSize);
        new(addr) T(value);
        ++mSize;
    }

    void push_back(T&& value)
    {
        assert(mSize < N);
        pointer const addr = get_address(mSize);
        new(addr) T(std::move(value));
        ++mSize;
    }

    template <typename... Args>
    void emplace_back(Args&&... args)
    {
        assert(mSize < N);
        pointer const addr = get_address(mSize);
        new(addr) T(std::forward<Args>(args)...);
        ++mSize;
    }

    void pop_back()
    {
        assert(mSize > 0);
        operator[](mSize-1u).~T();
        --mSize;
    }

    void resize(size_type count)
    {
        assert(count <= N);
        if (count < mSize) {
            std::for_each(begin() + count, end(), [](T& t) { t.~T(); });
        } else if (count > mSize) {
            for ( ; mSize < count; ++mSize) {
                new(get_address(mSize)) T;
            }
        }
        mSize = count;
    }

    void resize(size_type count, const value_type& value)
    {
        assert(count <= N);
        if (count < mSize) {
            std::for_each(begin() + count, end(), [](T& t) { t.~T(); });
        } else if (count > mSize) {
            for ( ; mSize < count; ++mSize) {
                new(get_address(mSize)) T(value);
            }
        }
        mSize = count;
    }

    void swap(StaticVector& other) noexcept(noexcept(StaticVector(std::move(other))))
    {
        StaticVector temp(std::move_if_noexcept(other));
        other = std::move_if_noexcept(*this);
        *this = std::move_if_noexcept(temp);
    }

private:
    template <typename InputIt>
    static std::size_t iterator_distance(InputIt first, InputIt last)
    {
        const auto dist = std::distance(first, last);
        assert(dist >= 0);
        return static_cast<std::size_t>(dist);
    }

    pointer get_address(size_type n)
    {
        return reinterpret_cast<pointer>(mData + n);
    }

    const_pointer get_address(size_type n) const
    {
        return reinterpret_cast<const_pointer>(mData + n);
    }

    void initialize(size_type n, const T& value, const std::true_type& /*is_non_iterator*/)
    {
        assert(empty()); // Assuming we're only using this on empty vectors.
        assert(n <= N);
        for ( ; mSize < n; ++mSize) {
            pointer const addr = get_address(mSize);
            new(addr) T(value);
        }
    }

    template <typename InputIt>
    void initialize(InputIt first, InputIt last, const std::false_type& /*is_non_iterator*/)
    {
        assert(empty()); // Assuming we're only using this on empty vectors.
        assert(iterator_distance(first, last) <= N);
        for ( ; first != last; ++first, ++mSize) {
            pointer const addr = get_address(mSize);
            new(addr) T(*first);
        }
    }

    iterator insert(const_iterator pos, size_type count, const T& value, const std::true_type& /*is_non_iterator*/)
    {
        assert(mSize + count < N);

        iterator pc(pos - cbegin() + begin());
        std::move_backward(pc, end(), end()+count);

        for (iterator first = pc, last = pc + count; first != last; ++first) {
            new(std::addressof(*first)) T(value);
            ++mSize;
        }
        return pc;
    }

    template <typename InputIt>
    iterator insert(const_iterator pos, InputIt first, InputIt last, const std::false_type& /*is_non_iterator*/)
    {
        const auto count = iterator_distance(first, last);
        assert(mSize + count < N);

        iterator pc(pos - cbegin() + begin());
        std::move_backward(pc, end(), end()+count);

        for (iterator loc = pc; first != last; ++first, ++loc) {
            new(std::addressof(*loc)) T(*first);
            ++mSize;
        }
        return pc;
    }

    using AlignedStorage = typename std::aligned_storage<sizeof(T), std::alignment_of<T>::value>::type;
    union
    {
        AlignedStorage mData[N];

        // Used for debugging inspection. We should never initialize the union
        // to the array version, because we get constructors/destructors we
        // don't expect.
        T mPtr[N];
    };

    static_assert(sizeof(mData) == sizeof(mPtr), "These values must overlap");
    static_assert(std::alignment_of<AlignedStorage>::value == std::alignment_of<T[N]>::value,
                  "These values must overlap");

    size_type mSize;
};

template <typename T, std::size_t N, std::size_t M>
inline bool operator==(const StaticVector<T, N>& x, const StaticVector<T, M>& y)
{
    return x.size() == y.size() && std::equal(x.begin(), x.end(), y.begin());
}

template <typename T, std::size_t N, std::size_t M>
inline bool operator<(const StaticVector<T, N>& x, const StaticVector<T, M>& y)
{
    return !(x == y);
}

template <typename T, std::size_t N, std::size_t M>
inline bool operator!=(const StaticVector<T, N>& x, const StaticVector<T, M>& y)
{
    return std::lexicographical_compare(x.begin(), x.end(), y.begin(), y.end());
}

template <typename T, std::size_t N, std::size_t M>
inline bool operator>(const StaticVector<T, N>& x, const StaticVector<T, M>& y)
{
    return y < x;
}

template <typename T, std::size_t N, std::size_t M>
inline bool operator>=(const StaticVector<T, N>& x, const StaticVector<T, M>& y)
{
    return !(x < y);
}

template <typename T, std::size_t N, std::size_t M>
inline bool operator<=(const StaticVector<T, N>& x, const StaticVector<T, M>& y)
{
    return !(x > y);
}

} // namespace util
} // namespace moonray

namespace std {
template <typename T, std::size_t N>
void swap(moonray::util::StaticVector<T, N>& a, moonray::util::StaticVector<T, N>& b) noexcept(noexcept(a.swap(b)))
{
    a.swap(b);
}
} // namespace std

