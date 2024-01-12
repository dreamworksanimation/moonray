// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file Statistics.h
///
#pragma once

#include <scene_rdl2/common/platform/Platform.h>

#include <cstdint>

namespace moonray {
namespace geom {
namespace internal {

enum StatCounters
{
    // Volume Grid Sampling
    STATS_VELOCITY_GRID_SAMPLES = 0,
    STATS_DENSITY_GRID_SAMPLES,
    STATS_EMISSION_GRID_SAMPLES,
    STATS_COLOR_GRID_SAMPLES,
    STATS_BAKED_DENSITY_GRID_SAMPLES,
    NUM_STATS_COUNTERS
};

class Statistics;
typedef geom::internal::Statistics GeomStatistics;

/* Statistics collects stats during rendering.
 * It has an array of counters, where each counter
 * is a different statistic. It is not thread safe, so
 * this should be used only in a per thread object.
 */
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

private:
    uint64_t mCounters[NUM_STATS_COUNTERS];
};

} // namespace internal
} // namespace geom
} // namespace moonray

