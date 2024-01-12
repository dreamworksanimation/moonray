// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

//

#pragma once


#include <scene_rdl2/common/platform/Platform.h>
#include <scene_rdl2/common/math/Math.h>
#include <limits>


namespace moonray {
namespace util {


//---------------------------------------------------------------------------

/* Average is a template class that tracks the sum of values as well
 * as how many times it was incremented (via += operator).
 * This class is useful for tracking timings for functions calls or blocks of
 * code that get call more then once.
 * It allows you to add the time in your desired units and then increments
 * the number of times it has been tracked. It allows you to get the total
 * sum (sum of all add() calls) as well as the Average.
 */
template <typename T>
class Average
{
public:
    __forceinline Average()
    {
        reset();
    }

    __forceinline Average(unsigned int count, T sum)
    {
        mCount = count;
        mSum = sum;
    }

    __forceinline ~Average()
    {
    }

    __forceinline void reset()
    {
        mCount = 0;
        mSum = 0;
    }

    __forceinline unsigned long getCount() const  { return mCount; }
    __forceinline T getSum() const  { return mSum; }
    double getAverage() const
    {
        if (mCount == 0) {
            return 0.f;
        } else {
            return mSum / mCount;
        }
    }

    __forceinline void operator+=(T value)
    {
        mSum += value;
        mCount++;
    }

    __forceinline void operator+=(const Average<T> &other)
    {
        mSum += other.mSum;
        mCount += other.mCount;
    }


protected:
    unsigned int mCount;
    T mSum;
};

typedef Average<int64>  AverageInt64;
typedef Average<double> AverageDouble;


/**
 * An extension of Average that can be used to maintain both inclusive and exclusive tick
 * counts (exclusive tick counts don't include ticks spent on sub-tasks). 
 */
template <typename T>
class InclusiveExclusiveAverage : public Average<T>
{
 public:
    __forceinline InclusiveExclusiveAverage() 
    {
        reset();
    }

    __forceinline void reset()
    {
        Average<T>::reset();
        mInclusiveSum = 0;
    }

    __forceinline void operator+=(T value)
    {
        Average<T>::operator+=(value);
        mInclusiveSum += value;
    }

    __forceinline void operator+=(const InclusiveExclusiveAverage<T> &other)
    {
        Average<T>::operator+=(other);
        mInclusiveSum += other.mInclusiveSum;
    }

    __forceinline void decrementExclusive(T value) 
    {
        Average<T>::mSum -= value;
    }

    __forceinline T getInclusiveSum() const
    {
        return mInclusiveSum;
    }
 private:
    T mInclusiveSum;
};


/* MixMaxAverage much like the Average class it a template class that
 * in addition adds the ability to track the max and min values that have
 * been observed. A little more heavy weight due to the extra storage.
 */
template <typename T>
class MixMaxAverage
{
public:
    __forceinline MixMaxAverage()
    {
        reset();
    }

    ~MixMaxAverage() {};

    __forceinline void reset() {
        mCount = 0;
        mSum = 0;
        mMin = std::numeric_limits<T>::max();
        mMax = std::numeric_limits<T>::min();
    }

    __forceinline unsigned long getCount() const    { return mCount; }
    __forceinline T getSum() const                  { return mSum; }
    __forceinline T getMin() const                  { return mMin; }
    __forceinline T getMax() const                  { return mMax; }
    __forceinline double getAverage() const
    {
        if (mCount == 0) {
            return 0.f;
        } else {
            return double(mSum) / mCount;
        }
    }

    __forceinline void operator+=(T value)
    {
        mCount++;
        mSum += value;
        value = scene_rdl2::math::max(value, mMax);
        value = scene_rdl2::math::min(value, mMin);
    }

protected:
    unsigned long mCount;
    T mSum;
    T mMin;
    T mMax;
};


//---------------------------------------------------------------------------

} // namespace util
} // namespace moonray


