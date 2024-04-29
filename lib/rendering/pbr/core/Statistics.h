// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file Statistics.h
/// $Id$
///

#pragma once
#include <moonray/common/mcrt_util/Average.h>
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
        mAdaptiveLightSamplingOverhead.reset();
        mLightSamplingTime.clear();
        mLightSamples.clear();
        mUsefulLightSamples.clear();
    }

    void initLightStats(size_t numLights) 
    {
        mLightSamplingTime.resize(numLights, moonray::util::AverageDouble());
        mLightSamples.resize(numLights, 0);
        mUsefulLightSamples.resize(numLights, 0);
    }

    Statistics &operator += (Statistics const &rhs) {
        for (unsigned i = 0; i < NUM_STATS_COUNTERS; ++i) {
            mCounters[i] += rhs.mCounters[i];
        }

        MNRY_ASSERT(mLightSamples.size() == rhs.mLightSamples.size());
        for (size_t i = 0; i < mLightSamples.size(); i++) {
            mLightSamplingTime[i] += rhs.mLightSamplingTime[i];
            mLightSamples[i] += rhs.mLightSamples[i];
            mUsefulLightSamples[i] += rhs.mUsefulLightSamples[i];
        }

        mAdaptiveLightSamplingOverhead += rhs.mAdaptiveLightSamplingOverhead;
    
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

    void incLightSamples(int lightIdx)
    {
        if (lightIdx == -1) return;
        mLightSamples[lightIdx]++;
    }

    void incUsefulLightSamples(int lightIdx)
    {
        if (lightIdx == -1) return;
        mUsefulLightSamples[lightIdx]++;
    }

    // HUD validation.
    static uint32_t hudValidation(bool verbose) { PBR_STATISTICS_VALIDATION; }

    PBR_STATISTICS_MEMBERS;
    std::vector<moonray::util::AverageDouble> mLightSamplingTime;
    std::vector<uint32_t> mLightSamples;
    std::vector<uint32_t> mUsefulLightSamples;
    moonray::util::AverageDouble mAdaptiveLightSamplingOverhead;
};

//----------------------------------------------------------------------------

} // namespace pbr
} // namespace moonray


