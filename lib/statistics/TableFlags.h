// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

//
// Created by kjeffery on 7/1/16.
//

#pragma once

#include "IOSFlags.h"
#include "Util.h"

namespace moonray_stats {

class TableFlags
{
public:
    virtual ~TableFlags() = default;
    const IOSFlags& get(std::size_t row, std::size_t col) const
    {
        return getImpl(row, col);
    }

    IOSFlags& get(std::size_t row, std::size_t col)
    {
        return getImpl(row, col);
    }

    std::unique_ptr<TableFlags> clone() const
    {
        return cloneImpl();
    }

private:
    virtual const IOSFlags& getImpl(std::size_t row, std::size_t col) const = 0;
    virtual IOSFlags& getImpl(std::size_t row, std::size_t col) = 0;
    virtual std::unique_ptr<TableFlags> cloneImpl() const = 0;
};

class ConstantFlags : public TableFlags
{
public:
    explicit ConstantFlags(const std::ostream& outs) : mFlags(outs) {}

    IOSFlags& set()
    {
        return mFlags;
    }

private:
    const IOSFlags& getImpl(std::size_t /*row*/, std::size_t /*col*/) const override
    {
        return mFlags;
    }

    IOSFlags& getImpl(std::size_t /*row*/, std::size_t /*col*/) override
    {
        return mFlags;
    }

    virtual std::unique_ptr<TableFlags> cloneImpl() const override
    {
        return make_unique<ConstantFlags>(*this);
    }

    IOSFlags mFlags;
};

template <std::size_t N>
class ColumnFlags : public TableFlags
{
public:
    explicit ColumnFlags(const std::ostream& outs) :
        mFlags()
    {
        std::fill(mFlags.begin(), mFlags.end(), IOSFlags(outs));
    }

    IOSFlags& set(std::size_t col)
    {
        return mFlags[col];
    }

private:
    const IOSFlags& getImpl(std::size_t /*row*/, std::size_t col) const override
    {
        return mFlags[col];
    }

    IOSFlags& getImpl(std::size_t /*row*/, std::size_t col) override
    {
        return mFlags[col];
    }

    virtual std::unique_ptr<TableFlags> cloneImpl() const override
    {
        return make_unique<ColumnFlags>(*this);
    }

    std::array<IOSFlags, N> mFlags;
};

template <std::size_t N>
class FullFlags : public TableFlags
{
public:
    explicit FullFlags(const std::ostream& outs, std::size_t nrows) :
        mFlags(nrows, ColumnFlags<N>(outs))
    {
    }

    explicit FullFlags(const ColumnFlags<N>& in, std::size_t nrows) :
        mFlags(nrows, in)
    {
    }

    IOSFlags& set(std::size_t row, std::size_t col)
    {
        return mFlags.at(row).set(col);
    }

private:
    const IOSFlags& getImpl(std::size_t row, std::size_t col) const override
    {
        return mFlags.at(row).get(0, col);
    }

    IOSFlags& getImpl(std::size_t row, std::size_t col) override
    {
        return mFlags.at(row).get(0, col);
    }

    virtual std::unique_ptr<TableFlags> cloneImpl() const override
    {
        return make_unique<FullFlags>(*this);
    }

    std::vector<ColumnFlags<N>> mFlags;
};

} // namespace moonray_stats


