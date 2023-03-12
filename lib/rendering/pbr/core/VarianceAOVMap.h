// Copyright 2023 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#pragma once

#include <iterator>
#include <unordered_map>

/// @class VarianceAOVMap
/// @brief Links AovSchema members by index to/from corresponding variance aovs
///
/// When calculating the variance of an aov, we create another aov that
/// references the original aov (the source). This class allows us to look up
/// the corresponding aovs. A aov may have more than one variance aovs using it
/// as the source, but a variance aov can only reference a single source.
class VarianceAOVMap
{
    using SourceToVarianceMap = std::unordered_multimap<int, int>;
    using VarianceToSourceMap = std::unordered_map<int, int>;

public:

    // It would be easiest to return an iterator from the map directly, but the
    // map iterators have a first and a second. In the cases in which we're
    // interested, we only care about the value, so we wrap the iterator to only
    // return the value to be simpler for the user and allow for further
    // abstraction.
    class SourceToVarianceIterator
    {
        using BaseIterator = std::unordered_multimap<int, int>::const_iterator;

    public:
        using value_type        = int;
        using difference_type   = std::ptrdiff_t;
        using reference         = const int&;
        using pointer           = const int*;
        using iterator_category = std::forward_iterator_tag;

        SourceToVarianceIterator() noexcept(noexcept(BaseIterator())) = default;

        explicit SourceToVarianceIterator(BaseIterator it) noexcept(noexcept(BaseIterator{it}))
        : mIter(it)
        {
        }

        explicit SourceToVarianceIterator(BaseIterator&& it) noexcept(noexcept(BaseIterator{std::move(it)}))
        : mIter(std::move(it))
        {
        }

        SourceToVarianceIterator(const SourceToVarianceIterator& other) noexcept(noexcept(BaseIterator(std::declval<BaseIterator&>()))) = default;
        SourceToVarianceIterator(SourceToVarianceIterator&& other) noexcept(noexcept(BaseIterator(std::declval<BaseIterator&&>()))) = default;

        SourceToVarianceIterator& operator=(const SourceToVarianceIterator& other) = default;
        SourceToVarianceIterator& operator=(SourceToVarianceIterator&& other) = default;

        value_type operator*() const
        {
            return mIter->second;
        }

        SourceToVarianceIterator& operator++()
        {
            ++mIter;
            return *this;
        }

        const SourceToVarianceIterator operator++(int)
        {
            SourceToVarianceIterator it(*this);
            ++it;
            return it;
        }

        friend bool operator==(const SourceToVarianceIterator& a, const SourceToVarianceIterator& b);

    private:
        BaseIterator mIter;
    };

    // We don't use a std::pair so that we can unambiguously overload begin and end.
    struct SourceToVarianceRange
    {
        SourceToVarianceIterator first;
        SourceToVarianceIterator last;
    };

    /// @brief Get the indices of aovs that compute the the variance of an aov
    /// \param sourceAovIdx the aov index of a potential variance source
    /// \return an iterator range of indices of aovs that compute the variance of this aov
    SourceToVarianceRange getSourceToVarianceAOVs(int sourceAovIdx) const
    {
        const auto range = mSourceToVariance.equal_range(sourceAovIdx);
        return { SourceToVarianceIterator{range.first}, SourceToVarianceIterator{range.second} };
    }

    /// @brief Get the indices of the aov that is the source of the variance being computed by an aov
    /// \param varianceAovIdx the variance aov
    /// \return the index of the source aov from which \p varianceAovIdx is computing the variance. -1 if this isn't a variance aov.
    int getVarianceToSourceAOV(int varianceAovIdx) const
    {
        const auto itr = mVarianceToSource.find(varianceAovIdx);
        if (itr == mVarianceToSource.end()) {
            return -1;
        } else {
            return itr->second;
        }
    }

    // Does this particular aov have another aov that is using it as the source of its variance calculation?
    bool hasVarianceEntry(int sourceAovIdx) const
    {
        return mSourceToVariance.find(sourceAovIdx) != mSourceToVariance.cend();
    }

    // Is this aov computing the variance of another aov?
    bool isVarianceEntry(int varianceAovIdx) const
    {
        return mVarianceToSource.find(varianceAovIdx) != mVarianceToSource.cend();
    }

    void linkVarianceOutput(int source, int variance)
    {
        mSourceToVariance.emplace(source, variance);
        const auto r __attribute__ ((unused)) = mVarianceToSource.emplace(variance, source);
        MNRY_ASSERT(r.second);
    }

private:
    SourceToVarianceMap mSourceToVariance;
    VarianceToSourceMap mVarianceToSource;
};

inline bool operator==(const VarianceAOVMap::SourceToVarianceIterator& a, const VarianceAOVMap::SourceToVarianceIterator& b)
{
    return a.mIter == b.mIter;
}

inline bool operator!=(const VarianceAOVMap::SourceToVarianceIterator& a, const VarianceAOVMap::SourceToVarianceIterator& b)
{
    return !(a == b);
}

inline VarianceAOVMap::SourceToVarianceIterator begin(const VarianceAOVMap::SourceToVarianceRange& range)
{
    return range.first;
}

inline VarianceAOVMap::SourceToVarianceIterator end(const VarianceAOVMap::SourceToVarianceRange& range)
{
    return range.last;
}


