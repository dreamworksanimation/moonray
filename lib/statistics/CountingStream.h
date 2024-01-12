// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <numeric>
#include <ostream>
#include <streambuf>
#include <vector>

// This class does not output anything, it simply counts characters. If there
// are multiple lines (as separated by '\n'), it keeps track of the maximum
// number of characters. Each line can be broken up into multiple columns, and
// the maximum length for each column is recorded. In order to meet the
// interface of std::streambuf, nextColumn does not have to be called. Instead,
// the sentinel sRecordSeparator can be streamed in.
class CountingStreamBuf : public std::streambuf
{
public:
    static const char sRecordSeparator = 30;

    inline CountingStreamBuf();
    inline CountingStreamBuf(const CountingStreamBuf&) = delete;
    inline CountingStreamBuf& operator=(const CountingStreamBuf&) = delete;
    inline ~CountingStreamBuf();

    inline std::size_t getWidth() const;
    inline std::size_t getColumnWidth(std::size_t column) const;
    inline void nextColumn();

private:
    inline int_type overflow(int_type c) override;

    std::size_t mCurrentSize;
    std::size_t mCurrentColumn;
    std::vector<std::size_t> mMaxSizes;
};

CountingStreamBuf::CountingStreamBuf() :
    mCurrentSize(0),
    mCurrentColumn(0),
    mMaxSizes(1, 0)
{
}

CountingStreamBuf::~CountingStreamBuf()
{
    pubsync();
}

std::size_t CountingStreamBuf::getWidth() const
{
    return std::accumulate(mMaxSizes.cbegin(), mMaxSizes.cend(), 0);
}

std::size_t CountingStreamBuf::getColumnWidth(std::size_t column) const
{
    return mMaxSizes.at(column);
}

void CountingStreamBuf::nextColumn()
{
    ++mCurrentColumn;
    mCurrentSize = 0;
    if (mCurrentColumn >= mMaxSizes.size()) {
        mMaxSizes.resize(mCurrentColumn + 1, 0);
    }
}

CountingStreamBuf::int_type CountingStreamBuf::overflow(int_type ch)
{
    if (!traits_type::eq_int_type(ch, traits_type::eof())) {
        if (traits_type::eq_int_type(ch, traits_type::to_int_type('\n'))) {
            mCurrentColumn = 0;
            mCurrentSize = 0;
        } else if (ch == sRecordSeparator) {
            nextColumn();
        } else {
            ++mCurrentSize;
            mMaxSizes[mCurrentColumn] = std::max(mMaxSizes[mCurrentColumn], mCurrentSize);
        }
        return traits_type::not_eof(ch);
    }

    return traits_type::eof();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

class CountingStream : public std::ostream
{
public:
    static const char sRecordSeparator = CountingStreamBuf::sRecordSeparator;

    CountingStream() :
        std::ostream(),
        mBuf()
    {
        this->init(&mBuf);
    }

    std::size_t getWidth() const
    {
        return mBuf.getWidth();
    }

    std::size_t getColumnWidth(std::size_t column) const
    {
        return mBuf.getColumnWidth(column);
    }

    void nextColumn()
    {
        mBuf.nextColumn();
    }

private:
    CountingStreamBuf mBuf;
};

