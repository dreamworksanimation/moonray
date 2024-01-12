// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file Statistics.h
/// $Id$
///

#pragma once
#include <scene_rdl2/common/platform/Platform.h>

namespace moonray {
namespace pbr {

// We are deliberately defining the StatCounters enum inside of the moonray::pbr
// namespace here.
#include "Statistics.hh"

//----------------------------------------------------------------------------

// Expose for HUD validation.
class Statistics;
typedef pbr::Statistics PbrStatistics;

///
/// @class Statistics Statistics.h <pbr/Statistics.h>
/// @brief This class is a simple POD type which collects all the stats
///        we need during rendering.
///
class Statistics
{
public:
    /// Constructor
    Statistics() {
        reset();
    }

    /// Reset the stats
    void reset() {
        for (unsigned i = 0; i < NUM_STATS_COUNTERS; ++i) {
            mCounters[i] = 0;
        }
        mMcrtTime = 0.0;
        mMcrtUtilization = 0.0;
    }

    Statistics &operator += (Statistics const &rhs) {
        for (unsigned i = 0; i < NUM_STATS_COUNTERS; ++i) {
            mCounters[i] += rhs.mCounters[i];
        }
        return *this;
    }

    uint64_t getCounter(unsigned counter) const
    {
        MNRY_ASSERT(counter < NUM_STATS_COUNTERS);
        return mCounters[counter];
    }

    void incCounter(unsigned counter)
    {
        MNRY_ASSERT(counter < NUM_STATS_COUNTERS);
        ++mCounters[counter];
    }

    void addToCounter(unsigned counter, unsigned count)
    {
        MNRY_ASSERT(counter < NUM_STATS_COUNTERS);
        mCounters[counter] += uint64_t(count);
    }

    // HUD validation.
    static uint32_t hudValidation(bool verbose) { PBR_STATISTICS_VALIDATION; }

//private:
    PBR_STATISTICS_MEMBERS;
};

//----------------------------------------------------------------------------

} // namespace pbr
} // namespace moonray


