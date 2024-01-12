// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

//
// Created by kjeffery on 6/20/16.
//

#pragma once

#include "Formatters.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <memory>
#include <utility>

namespace moonray_stats {

template <std::size_t N>
using StringList = std::array<std::string, N>;

template <typename T>
std::unique_ptr<Dimensionless> createTypePointer(T&& t)
{
    return moonray_stats::make_unique<Dimensionless>(std::forward<T>(t));
}

template <typename T>
static std::unique_ptr<T> createTypePointer(std::unique_ptr<T> t)
{
    return t;
}

template <std::size_t columns>
class StatsTableRow
{
    using Row = std::array<std::unique_ptr<Type>, columns>;

public:
    template <typename... T>
    explicit StatsTableRow(T&&... t) :
        mRow{createTypePointer(std::forward<T>(t))...}
    {
        static_assert(sizeof...(T) == columns,
                      "The number of arguments must match the number of columns");
    }

    StatsTableRow() = delete;
    StatsTableRow(const StatsTableRow&) = delete;
    StatsTableRow& operator=(const StatsTableRow&) = delete;

    StatsTableRow(StatsTableRow&& other) noexcept :
        mRow{}
    {
        std::move(std::begin(other.mRow), std::end(other.mRow), std::begin(mRow));
    }

    StatsTableRow& operator=(StatsTableRow&& other) noexcept
    {
        std::move(std::begin(other.mRow), std::end(other.mRow), std::begin(mRow));
        return *this;
    }

    Type& operator[](std::size_t col)
    {
        return *mRow[col];
    }

    const Type& operator[](std::size_t col) const
    {
        return *mRow[col];
    }

    Type& at(std::size_t col)
    {
        return *mRow.at(col);
    }

    const Type& at(std::size_t col) const
    {
        return *mRow.at(col);
    }

private:
    Row mRow;
};

template <std::size_t columns>
class StatsTable
{
public:
    using HeaderRow = StringList<columns>;

    template <typename... T>
    explicit StatsTable(std::string title, T&&... headers) :
        mTitle(title),
        mHeaders{ std::forward<T>(headers)... },
        mRows()
    {
        static_assert(sizeof...(headers) == 0 || sizeof...(headers) == columns,
                      "We either expect no headers or all headers specified");
    }

    StatsTable(const StatsTable&) = delete;
    StatsTable& operator=(const StatsTable&) = delete;

    StatsTable(StatsTable&& other) :
        mTitle(std::move(other.mTitle)),
        mHeaders(),
        mRows(std::move(other.mRows)),
        mSepIndices(std::move(other.mSepIndices))
    {
        std::move(other.mHeaders.begin(), other.mHeaders.end(), mHeaders.begin());
    }

    StatsTable& operator=(StatsTable&& other)
    {
        if (this != std::addressof(other)) {
            mTitle = std::move(other.mTitle);
            mRows = std::move(other.mRows);
            mSepIndices = std::move(other.mSepIndices);
            std::move(other.mHeaders.begin(), other.mHeaders.end(),
                      mHeaders.begin());
        }
        return *this;
    }

    template <typename... T>
    void emplace_back(T&&... t)
    {
        static_assert(sizeof...(T) == columns,
                      "The number of arguments must match the number of columns");

        mRows.emplace_back(std::forward<T>(t)...);
    }

    void addSeparator()
    {
        mSepIndices.push_back(mRows.size());
    }

    Type& operator()(std::size_t row, std::size_t col)
    {
        return mRows.at(row).at(col);
    }

    const Type& operator()(std::size_t row, std::size_t col) const
    {
        return mRows.at(row).at(col);
    }

    bool empty() const
    {
        return mRows.empty();
    }

    static constexpr std::size_t getNumColumns()
    {
        return columns;
    }

    std::size_t getNumRows() const
    {
        return mRows.size();
    }

    const HeaderRow& getHeaders() const
    {
        return mHeaders;
    }

    HeaderRow& getHeaders()
    {
        return mHeaders;
    }

    const std::string& getTitle() const
    {
        return mTitle;
    }

    bool isSeparator(std::size_t i) const
    {
        assert(std::is_sorted(mSepIndices.cbegin(), mSepIndices.cend()));
        return std::binary_search(mSepIndices.cbegin(), mSepIndices.cend(), i);
    }

    bool hasHeaders() const
    {
        return std::any_of(mHeaders.begin(), mHeaders.end(),
                           [](const std::string& s) { return !s.empty(); });
    }

private:
    using OutputSeparatorIndices = std::vector<std::size_t>;

    std::string mTitle;
    HeaderRow mHeaders;
    std::vector<StatsTableRow<columns>> mRows;
    OutputSeparatorIndices mSepIndices;
};

} // namespace moonray_stats


