// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "StaticVector.h"

#include <array>
#include <cassert>
#include <type_traits>
#include <vector>

struct CanonicalRange
{
    static constexpr float min = 0.0f;
    static constexpr float max = 1.0f;
};

struct DiskRange
{
    static constexpr float min = -1.0f;
    static constexpr float max = +1.0f;
};

struct RangeScale
{
    template <typename U, typename FromRange, typename ToRange>
    static U scale(U x) noexcept
    {
        const U p = (x - FromRange::min)/(FromRange::max - FromRange::min);
        return p * (ToRange::max - ToRange::min) + ToRange::min;
    }

    template <typename U, typename FromRange>
    static U scale(U x, U toMin, U toMax) noexcept
    {
        const U p = (x - FromRange::min)/(FromRange::max - FromRange::min);
        return p * (toMax - toMin) + toMin;
    }
};

enum class SearchResult
{
    TERMINATED_EARLY,
    SEARCH_COMPLETE
};

template <typename T>
struct is_search_result : public std::false_type
{
};

template <>
struct is_search_result<SearchResult> : public std::true_type
{
};

template <typename T, unsigned Dimensions, typename IncomingRange = CanonicalRange>
class StaticHyperGrid
{
public:
    friend StaticHyperGrid<T, Dimensions+1, IncomingRange>;

    typedef StaticHyperGrid<T, Dimensions-1u, IncomingRange> value_type;
    typedef unsigned size_type;

    // We allow two return types for function objects passed into the visit
    // functions: those that return void, and those that terminate early by
    // returning a SearchResult.
    template <typename F>
    using ResultReturnType = typename std::enable_if_t<is_search_result<decltype(std::declval<F>()(std::declval<T&>()))>::value, SearchResult>;

    template <typename F>
    using VoidReturnType = typename std::enable_if_t<!is_search_result<decltype(std::declval<F>()(std::declval<T&>()))>::value, void>;

    StaticHyperGrid()
    : mOrder(1)
    , mData(1)
    {
    }

    explicit StaticHyperGrid(size_type order)
    : mOrder(order)
    , mData()
    {
        // Initialize sub-trees with the same order.
        mData.reserve(order);
        for (size_type i = 0; i < order; ++i) {
            mData.emplace_back(order);
        }
    }

    size_type left(size_type x) const noexcept
    {
        return (x == 0) ? mOrder - 1u : x - 1u;
    }

    size_type center(size_type x) const noexcept
    {
        return x;
    }

    size_type right(size_type x) const noexcept
    {
        return (x == mOrder - 1u) ? 0u : x + 1u;
    }

    template <typename F>
    ResultReturnType<F> visitNeighbors(F&& f, const T& p) const
    {
        if (mOrder <= 3) {
            return visitAll(std::forward<F>(f));
        }

        // Find cell
        const size_type cell = offset(p);
        const std::array<size_type, 3> vals = { { left(cell), center(cell), right(cell) } };

        for (const auto current : vals) {
            if (mData[current].visitNeighbors(std::forward<F>(f), p) == SearchResult::TERMINATED_EARLY) {
                return SearchResult::TERMINATED_EARLY;
            }
        }
        return SearchResult::SEARCH_COMPLETE;
    }

    template <typename F>
    VoidReturnType<F> visitNeighbors(F&& f, const T& p) const
    {
        if (mOrder <= 3) {
            return visitAll(std::forward<F>(f));
        }

        // Find cell
        const size_type cell = offset(p);
        const std::array<size_type, 3> vals = { { left(cell), center(cell), right(cell) } };

        for (const auto current : vals) {
            mData[current].visitNeighbors(std::forward<F>(f), p);
        }
    }

    template <typename F>
    ResultReturnType<F> visitProjectedNeighbors(F&& f, const T& p) const
    {
        if (mOrder <= 3) {
            return visitAll(std::forward<F>(f));
        }

        // Find cell
        const size_type cell = offset(p);
        const auto l = left(cell);
        const auto r = right(cell);
        const auto c = center(cell);

        // We need to visit everybody along our dimension. We skip ourselves,
        // because those will be taken care of in a lower visit call.
        for (size_type i = 0; i < mOrder; ++i) {
            if (i != c && i != l && i != r) {
                if (mData[i].visitNeighbors(std::forward<F>(f), p) == SearchResult::TERMINATED_EARLY) {
                    return SearchResult::TERMINATED_EARLY;
                }
            }
        }

        if (mData[l].visitProjectedNeighbors(std::forward<F>(f), p) == SearchResult::TERMINATED_EARLY) {
            return SearchResult::TERMINATED_EARLY;
        }
        return mData[r].visitProjectedNeighbors(std::forward<F>(f), p);
    }

    template <typename F>
    VoidReturnType<F> visitProjectedNeighbors(F&& f, const T& p) const
    {
        if (mOrder <= 3) {
            return visitAll(std::forward<F>(f));
        }

        // Find cell
        const size_type cell = offset(p);
        const auto l = left(cell);
        const auto r = right(cell);
        const auto c = center(cell);

        // We need to visit everybody along our dimension. We skip ourselves,
        // because those will be taken care of in a lower visit call.
        for (size_type i = 0; i < mOrder; ++i) {
            if (i != c && i != l && i != r) {
                mData[i].visitNeighbors(std::forward<F>(f), p);
            }
        }
        mData[l].visitProjectedNeighbors(std::forward<F>(f), p);
        mData[r].visitProjectedNeighbors(std::forward<F>(f), p);
    }

    template <typename F>
    ResultReturnType<F> visitAll(F&& f) const
    {
        for (const auto& n : mData) {
            if (n.visitAll(std::forward<F>(f)) == SearchResult::TERMINATED_EARLY) {
                return SearchResult::TERMINATED_EARLY;
            }
        }
        return SearchResult::SEARCH_COMPLETE;
    }

    template <typename F>
    VoidReturnType<F> visitAll(F&& f) const
    {
        for (const auto& n : mData) {
            n.visitAll(std::forward<F>(f));
        }
    }

    void add(T&& t)
    {
        // Find cell
        const auto idx = offset(t);
        assert(idx < mData.size());
        return mData[idx].add(std::move(t));
    }

    void add(T t)
    {
        // Find cell
        const auto idx = offset(t);
        assert(idx < mData.size());
        return mData[idx].add(std::move(t));
    }

    // O(n)
    void remove(const T& t)
    {
        // Find cell
        const auto idx = offset(t);
        assert(idx < mData.size());
        return mData[idx].remove(t);
    }

    size_type order() const noexcept
    {
        return mOrder;
    }

    void swap(StaticHyperGrid& other) noexcept
    {
        using std::swap; // Allow adl
        swap(mOrder, other.mOrder);
        swap(mData, other.mData);
    }

private:
    size_type offset(const T& p) const noexcept
    {
        assert(p[Dimensions-1u] >= IncomingRange::min);
        assert(p[Dimensions-1u] <  IncomingRange::max);
        return static_cast<size_type>(RangeScale::scale<float, IncomingRange>(p[Dimensions-1u], 0, mOrder));
    }

    typedef std::vector<value_type> ContainerType;

    size_type mOrder;
    ContainerType mData;
};

template <typename T, typename IncomingRange>
class StaticHyperGrid<T, 1, IncomingRange>
{
public:
    friend StaticHyperGrid<T, 2, IncomingRange>;

    typedef T value_type;
    typedef unsigned size_type;

    // We allow two return types for function objects passed into the visit
    // functions: those that return void, and those that terminate early by
    // returning a SearchResult.
    template <typename F>
    using ResultReturnType = typename std::enable_if_t<is_search_result<decltype(std::declval<F>()(std::declval<T&>()))>::value, SearchResult>;

    template <typename F>
    using VoidReturnType = typename std::enable_if_t<!is_search_result<decltype(std::declval<F>()(std::declval<T&>()))>::value, void>;

    StaticHyperGrid()
    : mOrder(1)
    , mData(1)
    {
    }

    explicit StaticHyperGrid(size_type size)
    : mOrder(size)
    , mData(size)
    {
    }

    size_type left(size_type x) const noexcept
    {
        return (x == 0) ? mOrder - 1u : x - 1u;
    }

    size_type center(size_type x) const noexcept
    {
        return x;
    }

    size_type right(size_type x) const noexcept
    {
        return (x == mOrder - 1u) ? 0u : x + 1u;
    }

    template <typename F>
    ResultReturnType<F> visitNeighbors(F&& f, const T& p) const
    {
        if (mOrder <= 3) {
            return visitAll(std::forward<F>(f));
        }

        // Find cell
        const size_type cell = offset(p);
        const std::array<size_type, 3> vals = { { left(cell), center(cell), right(cell) } };

        for (const auto current : vals) {
            for (const auto& p : mData[current].mData) {
                if (std::forward<F>(f)(p) == SearchResult::TERMINATED_EARLY) {
                    return SearchResult::TERMINATED_EARLY;
                }
            }
        }
        return SearchResult::SEARCH_COMPLETE;
    }

    template <typename F>
    VoidReturnType<F> visitNeighbors(F&& f, const T& p) const
    {
        if (mOrder <= 3) {
            return visitAll(std::forward<F>(f));
        }

        // Find cell
        const size_type cell = offset(p);
        const std::array<size_type, 3> vals = { { left(cell), center(cell), right(cell) } };

        for (const auto current : vals) {
            for (const auto& p : mData[current]) {
                std::forward<F>(f)(p);
            }
        }
    }

    template <typename F>
    auto visitProjectedNeighbors(F&& f, const T& p) const
    {
        // We're just one dimension.
        return visitAll(std::forward<F>(f));
    }

    template <typename F>
    ResultReturnType<F> visitAll(F&& f) const
    {
        for (const auto& n : mData) {
            for (const auto& p : n.mData) {
                if (std::forward<F>(f)(p) == SearchResult::TERMINATED_EARLY) {
                    return SearchResult::TERMINATED_EARLY;
                }
            }
        }
        return SearchResult::SEARCH_COMPLETE;
    }

    template <typename F>
    VoidReturnType<F> visitAll(F&& f) const
    {
        for (const auto& n : mData) {
            for (const auto& p : n) {
                std::forward<F>(f)(p);
            }
        }
    }

    void add(T t)
    {
        // Find cell
        const auto idx = offset(t);
        assert(idx < mData.size());
        mData[idx].push_back(std::move(t));
    }

    // O(n)
    void remove(const T& t)
    {
        // Find cell
        const auto idx = offset(t);
        assert(idx < mData.size());
        auto& container = mData[idx];
        auto it = std::remove(container.begin(), container.end(), t);
        if (it == container.end()) {
            std::cerr << "Not found\n";
        }
        container.erase(it, container.end());
        //container.erase(std::remove(container.begin(), container.end(), t), container.end());
    }

    size_type order() const noexcept
    {
        return mOrder;
    }

    void swap(StaticHyperGrid& other) noexcept
    {
        using std::swap; // Allow adl
        swap(mOrder, other.mOrder);
        swap(mData, other.mData);
    }

private:
    size_type offset(const T& p) const noexcept
    {
        assert(p[0] >= IncomingRange::min);
        assert(p[0] <  IncomingRange::max);
        return static_cast<size_type>(RangeScale::scale<float, IncomingRange>(p[0], 0, mOrder));
    }

    typedef std::vector<value_type> CellType;
    typedef std::vector<CellType> ContainerType;

    size_type mOrder;
    ContainerType mData;
};

template <typename T, unsigned Dimensions, typename IncomingRange>
class StaticHyperGrid<T*, Dimensions, IncomingRange>
{
public:
    friend StaticHyperGrid<T*, Dimensions+1, IncomingRange>;

    typedef StaticHyperGrid<T*, Dimensions-1u, IncomingRange> value_type;
    typedef unsigned size_type;

    // We allow two return types for function objects passed into the visit
    // functions: those that return void, and those that terminate early by
    // returning a SearchResult.
    template <typename F>
    using ResultReturnType = typename std::enable_if_t<is_search_result<decltype(std::declval<F>()(std::declval<T*>()))>::value, SearchResult>;

    template <typename F>
    using VoidReturnType = typename std::enable_if_t<!is_search_result<decltype(std::declval<F>()(std::declval<T*>()))>::value, void>;

    StaticHyperGrid()
    : mOrder(1)
    , mData(1)
    {
    }

    explicit StaticHyperGrid(size_type order)
    : mOrder(order)
    , mData()
    {
        // Initialize sub-trees with the same order.
        mData.reserve(order);
        for (size_type i = 0; i < order; ++i) {
            mData.emplace_back(order);
        }
    }

    size_type left(size_type x) const noexcept
    {
        return (x == 0) ? mOrder - 1u : x - 1u;
    }

    size_type center(size_type x) const noexcept
    {
        return x;
    }

    size_type right(size_type x) const noexcept
    {
        return (x == mOrder - 1u) ? 0u : x + 1u;
    }

    template <typename F>
    ResultReturnType<F> visitNeighbors(F&& f, const T* p) const
    {
        assert(p);
        if (mOrder <= 3) {
            return visitAll(std::forward<F>(f));
        }

        // Find cell
        const size_type cell = offset(*p);
        const std::array<size_type, 3> vals = { { left(cell), center(cell), right(cell) } };

        for (const auto current : vals) {
            if (mData[current].visitNeighbors(std::forward<F>(f), p) == SearchResult::TERMINATED_EARLY) {
                return SearchResult::TERMINATED_EARLY;
            }
        }
        return SearchResult::SEARCH_COMPLETE;
    }

    template <typename F>
    VoidReturnType<F> visitNeighbors(F&& f, const T* p) const
    {
        assert(p);
        if (mOrder <= 3) {
            return visitAll(std::forward<F>(f));
        }

        // Find cell
        const size_type cell = offset(*p);
        const std::array<size_type, 3> vals = { { left(cell), center(cell), right(cell) } };

        for (const auto current : vals) {
            mData[current].visitNeighbors(std::forward<F>(f), p);
        }
    }

    template <typename F>
    ResultReturnType<F> visitAll(F&& f) const
    {
        for (const auto& n : mData) {
            if (n.visitAll(std::forward<F>(f)) == SearchResult::TERMINATED_EARLY) {
                return SearchResult::TERMINATED_EARLY;
            }
        }
        return SearchResult::SEARCH_COMPLETE;
    }

    template <typename F>
    VoidReturnType<F> visitAll(F&& f) const
    {
        for (const auto& n : mData) {
            n.visitAll(std::forward<F>(f));
        }
    }

    void add(const T* const t)
    {
        assert(t);
        // Find cell
        const auto idx = offset(*t);
        assert(idx < mData.size());
        return mData[idx].add(t);
    }

    void add(T* t)
    {
        assert(t);
        // Find cell
        const auto idx = offset(*t);
        assert(idx < mData.size());
        return mData[idx].add(t);
    }

    void remove(const T* t)
    {
        assert(t);
        // Find cell
        const auto idx = offset(*t);
        assert(idx < mData.size());
        return mData[idx].remove(t);
    }

    size_type order() const noexcept
    {
        return mOrder;
    }

    void swap(StaticHyperGrid& other) noexcept
    {
        using std::swap; // Allow adl
        swap(mOrder, other.mOrder);
        swap(mData, other.mData);
    }

private:
    size_type offset(const T& p) const noexcept
    {
        assert(p[Dimensions-1u] >= IncomingRange::min);
        assert(p[Dimensions-1u] <  IncomingRange::max);
        return static_cast<size_type>(RangeScale::scale<float, IncomingRange>(p[Dimensions-1u], 0, mOrder));
    }

    typedef std::vector<value_type> ContainerType;

    size_type mOrder;
    ContainerType mData;
};

template <typename T, typename IncomingRange>
class StaticHyperGrid<T*, 1, IncomingRange>
{
public:
    friend StaticHyperGrid<T*, 2, IncomingRange>;

    typedef T* value_type;
    typedef unsigned size_type;

    // We allow two return types for function objects passed into the visit
    // functions: those that return void, and those that terminate early by
    // returning a SearchResult.
    template <typename F>
    using ResultReturnType = typename std::enable_if_t<is_search_result<decltype(std::declval<F>()(std::declval<T*>()))>::value, SearchResult>;

    template <typename F>
    using VoidReturnType = typename std::enable_if_t<!is_search_result<decltype(std::declval<F>()(std::declval<T*>()))>::value, void>;

    StaticHyperGrid()
    : mOrder(1)
    , mData(1)
    {
    }

    explicit StaticHyperGrid(size_type order)
    : mOrder(order)
    , mData(order)
    {
    }

    size_type left(size_type x) const noexcept
    {
        return (x == 0) ? mOrder - 1u : x - 1u;
    }

    size_type center(size_type x) const noexcept
    {
        return x;
    }

    size_type right(size_type x) const noexcept
    {
        return (x == mOrder - 1u) ? 0u : x + 1u;
    }

    template <typename F>
    ResultReturnType<F> visitNeighbors(F&& f, const T* p) const
    {
        assert(p);
        if (mOrder <= 3) {
            return visitAll(std::forward<F>(f));
        }

        // Find cell
        const size_type cell = offset(*p);
        const std::array<size_type, 3> vals = { { left(cell), center(cell), right(cell) } };

        for (const auto current : vals) {
            for (const auto& p : mData[current].mData) {
                if (std::forward<F>(f)(p) == SearchResult::TERMINATED_EARLY) {
                    return SearchResult::TERMINATED_EARLY;
                }
            }
        }
        return SearchResult::SEARCH_COMPLETE;
    }

    template <typename F>
    VoidReturnType<F> visitNeighbors(F&& f, const T* p) const
    {
        assert(p);
        if (mOrder <= 3) {
            return visitAll(std::forward<F>(f));
        }

        // Find cell
        const size_type cell = offset(*p);
        const std::array<size_type, 3> vals = { { left(cell), center(cell), right(cell) } };

        for (const auto current : vals) {
            for (const auto& p : mData[current]) {
                std::forward<F>(f)(p);
            }
        }
    }

    template <typename F>
    ResultReturnType<F> visitAll(F&& f) const
    {
        for (const auto& n : mData) {
            for (const auto& p : n.mData) {
                if (std::forward<F>(f)(p) == SearchResult::TERMINATED_EARLY) {
                    return SearchResult::TERMINATED_EARLY;
                }
            }
        }
        return SearchResult::SEARCH_COMPLETE;
    }

    template <typename F>
    VoidReturnType<F> visitAll(F&& f) const
    {
        for (const auto& n : mData) {
            for (const auto& p : n) {
                std::forward<F>(f)(p);
            }
        }
    }

    void add(const T* const t)
    {
        assert(t);
        // Find cell
        const auto idx = offset(*t);
        assert(idx < mData.size());
        mData[idx].push_back(t);
    }

    void add(T* t)
    {
        assert(t);
        // Find cell
        const auto idx = offset(*t);
        assert(idx < mData.size());
        mData[idx].push_back(t);
    }

    void remove(const T* t)
    {
        assert(t);
        // Find cell
        const auto idx = offset(*t);
        assert(idx < mData.size());
        auto& container = mData[idx];
        auto it = std::remove(container.begin(), container.end(), t);
        if (it == container.end()) {
            std::cerr << "Not found\n";
        }
        container.erase(it, container.end());
        //container.erase(std::remove(container.begin(), container.end(), t), container.end());
    }

    size_type order() const noexcept
    {
        return mOrder;
    }

    void swap(StaticHyperGrid& other) noexcept
    {
        using std::swap; // Allow adl
        swap(mOrder, other.mOrder);
        swap(mData, other.mData);
    }

private:
    size_type offset(const T& p) const noexcept
    {
        assert(p[0] >= IncomingRange::min);
        assert(p[0] <  IncomingRange::max);
        return static_cast<size_type>(RangeScale::scale<float, IncomingRange>(p[0], 0, mOrder));
    }

    typedef std::vector<value_type> CellType;
    typedef std::vector<CellType> ContainerType;

    size_type mOrder;
    ContainerType mData;
};
