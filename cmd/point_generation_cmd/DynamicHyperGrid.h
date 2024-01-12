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
    static constexpr double min = 0.0;
    static constexpr double max = 1.0;
};

struct DiskRange
{
    static constexpr double min = -1.0;
    static constexpr double max = +1.0;
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
class DynamicHyperGrid
{
public:
    friend DynamicHyperGrid<T, Dimensions+1, IncomingRange>;

    typedef DynamicHyperGrid<T, Dimensions-1u, IncomingRange> value_type;
    typedef unsigned size_type;

    // We allow two return types for function objects passed into the visit
    // functions: those that return void, and those that terminate early by
    // returning a SearchResult.
    template <typename F>
    using ResultReturnType = typename std::enable_if_t<is_search_result<decltype(std::declval<F>()(std::declval<T>()))>::value, SearchResult>;

    template <typename F>
    using VoidReturnType = typename std::enable_if_t<!is_search_result<decltype(std::declval<F>()(std::declval<T>()))>::value, void>;

    DynamicHyperGrid() :
        mOrder(1),
        mData(1)
    {
    }

    explicit DynamicHyperGrid(size_type order) :
        mOrder(order),
        mData()
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

    static constexpr size_type maxNeighboringItems() noexcept
    {
        return 3u * DynamicHyperGrid<T, Dimensions-1u, IncomingRange>::maxNeighboringItems();
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
            if (mData[current].visitNeighbors(f, p) == SearchResult::TERMINATED_EARLY) {
                return SearchResult::TERMINATED_EARLY;
            }
        }
        return SearchResult::SEARCH_COMPLETE;
    }

    template <typename F>
    VoidReturnType<F> visitNeighbors(F f, const T& p) const
    {
        if (mOrder <= 3) {
            return visitAll(f);
        }

        // Find cell
        const size_type cell = offset(p);
        const std::array<size_type, 3> vals = { { left(cell), center(cell), right(cell) } };

        for (const auto current : vals) {
            mData[current].visitNeighbors(f, p);
        }
    }

    template <typename F>
    ResultReturnType<F> visitProjectedNeighbors(F&& f, const T& p) const
    {
        if (mOrder <= 3) {
            return visitAll(f);
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

        if (mData[l].visitProjectedNeighbors(f, p) == SearchResult::TERMINATED_EARLY) {
            return SearchResult::TERMINATED_EARLY;
        }
        if (mData[c].visitProjectedNeighbors(f, p) == SearchResult::TERMINATED_EARLY) {
            return SearchResult::TERMINATED_EARLY;
        }
        return mData[r].visitProjectedNeighbors(f, p);
    }

    template <typename F>
    VoidReturnType<F> visitProjectedNeighbors(F f, const T& p) const
    {
        if (mOrder <= 3) {
            return visitAll(f);
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
                mData[i].visitNeighbors(f, p);
            }
        }
        mData[l].visitProjectedNeighbors(f, p);
        mData[c].visitProjectedNeighbors(f, p);
        mData[r].visitProjectedNeighbors(f, p);
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
    VoidReturnType<F> visitAll(F f) const
    {
        for (const auto& n : mData) {
            n.visitAll(f);
        }
    }

    void add(const T& t)
    {
        if (!privateAdd(t)) {
            repartition();
            add(t);
        }
    }

    size_type order() const noexcept
    {
        return mOrder;
    }

    void swap(DynamicHyperGrid& other) noexcept
    {
        using std::swap; // Allow adl
        swap(mOrder, other.mOrder);
        swap(mData, other.mData);
    }

private:
    // An add function that does not check if the cell is full.
    void unsafeAdd(const T& t)
    {
        // Find cell
        const auto idx = offset(t);
        assert(idx < mData.size());
        return mData[idx].unsafeAdd(t);
    }

    bool privateAdd(const T& t)
    {
        // Find cell
        const auto idx = offset(t);
        assert(idx < mData.size());
        return mData[idx].privateAdd(t);
    }

    size_type offset(const T& p) const noexcept
    {
        assert(p[Dimensions-1u] >= IncomingRange::min);
        assert(p[Dimensions-1u] <  IncomingRange::max);
        return static_cast<size_type>(RangeScale::scale<double, IncomingRange>(p[Dimensions-1u], 0, mOrder));
    }

    void repartition()
    {
        // Although it's been shown that 1.5 is a better multiplier than 2 when
        // rescaling containers, we do 2. By doing 2, we can split the strata
        // in half, guaranteeing that we have, at most, as many points in any
        // stratum as we had before, making it so that we can do an unsafe add.
        // If we had another multiplier, our strata extents change, and we no
        // longer have that guarantee.
        DynamicHyperGrid other(mOrder * 2u);

        visitAll([&other](const T& p) {
            other.unsafeAdd(p);
        });

        // Do non-throwing work at the end
        swap(other);
    }

    typedef std::vector<value_type> ContainerType;

    size_type mOrder;
    ContainerType mData;
};

template <typename T, typename IncomingRange>
class DynamicHyperGrid<T, 1, IncomingRange>
{
public:
    friend DynamicHyperGrid<T, 2, IncomingRange>;

    typedef T value_type;
    typedef unsigned size_type;

    // We allow two return types for function objects passed into the visit
    // functions: those that return void, and those that terminate early by
    // returning a SearchResult.
    template <typename F>
    using ResultReturnType = typename std::enable_if_t<is_search_result<decltype(std::declval<F>()(std::declval<T>()))>::value, SearchResult>;

    template <typename F>
    using VoidReturnType = typename std::enable_if_t<!is_search_result<decltype(std::declval<F>()(std::declval<T>()))>::value, void>;

    DynamicHyperGrid() :
        mOrder(1),
        mData(1)
    {
    }

    explicit DynamicHyperGrid(size_type size) :
        mOrder(size),
        mData(size)
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

    static constexpr size_type maxNeighboringItems() noexcept
    {
        return 3u * NodeType::sMaxElements;
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
                if (f(p) == SearchResult::TERMINATED_EARLY) {
                    return SearchResult::TERMINATED_EARLY;
                }
            }
        }
        return SearchResult::SEARCH_COMPLETE;
    }

    template <typename F>
    VoidReturnType<F> visitNeighbors(F f, const T& p) const
    {
        if (mOrder <= 3) {
            return visitAll(f);
        }

        // Find cell
        const size_type cell = offset(p);
        const std::array<size_type, 3> vals = { { left(cell), center(cell), right(cell) } };

        for (const auto current : vals) {
            for (const auto& p : mData[current].mData) {
                f(p);
            }
        }
    }

    template <typename F>
    auto visitProjectedNeighbors(F f, const T& /*p*/) const
    {
        // We're just one dimension.
        return visitAll(f);
    }

    template <typename F>
    ResultReturnType<F> visitAll(F&& f) const
    {
        for (const auto& n : mData) {
            for (const auto& p : n.mData) {
                if (f(p) == SearchResult::TERMINATED_EARLY) {
                    return SearchResult::TERMINATED_EARLY;
                }
            }
        }
        return SearchResult::SEARCH_COMPLETE;
    }

    template <typename F>
    VoidReturnType<F> visitAll(F f) const
    {
        for (const auto& n : mData) {
            for (const auto& p : n.mData) {
                f(p);
            }
        }
    }

    void add(const T& t)
    {
        if (!privateAdd(t)) {
            repartition();
            add(t);
        }
    }

    size_type order() const noexcept
    {
        return mOrder;
    }

    void swap(DynamicHyperGrid& other) noexcept
    {
        using std::swap; // Allow adl
        swap(mOrder, other.mOrder);
        swap(mData, other.mData);
    }

private:
    // An add function that does not check if the cell is full.
    void unsafeAdd(const T& t)
    {
        // Find cell
        const auto idx = offset(t);
        assert(idx < mData.size());
        assert(!mData[idx].full());
        mData[idx].mData.push_back(t);
    }

    bool privateAdd(const T& t)
    {
        // Find cell
        const auto idx = offset(t);
        assert(idx < mData.size());

        if (!mData[idx].full()) {
            // If there is room in the cell, add the point
            mData[idx].mData.push_back(t);
            return true;
        } else {
            return false;
        }
    }

    size_type offset(const T& p) const noexcept
    {
        assert(p[0] >= IncomingRange::min);
        assert(p[0] <  IncomingRange::max);
        return static_cast<size_type>(RangeScale::scale<double, IncomingRange>(p[0], 0, mOrder));
    }

    void repartition()
    {
        // Although it's been shown that 1.5 is a better multiplier than 2 when
        // rescaling containers, we do 2. By doing 2, we can split the strata
        // in half, guaranteeing that we have, at most, as many points in any
        // stratum as we had before, making it so that we can do an unsafe add.
        // If we had another multiplier, our strata extents change, and we no
        // longer have that guarantee.
        DynamicHyperGrid other(mOrder * 2u);

        // Partition points...
        visitAll([&other](const T& p) {
            other.unsafeAdd(p);
        });

        // Do non-throwing work at the end
        swap(other);
    }

    template <typename S>
    struct Node
    {
        bool full() const noexcept
        {
            assert(mData.size() <= sMaxElements);
            return mData.size() == sMaxElements;
        }

        static const std::size_t sMaxElements = 8;
        StaticVector<S, sMaxElements> mData;
    };

    typedef Node<value_type> NodeType;
    typedef std::vector<NodeType> ContainerType;

    size_type mOrder;
    ContainerType mData;
};

