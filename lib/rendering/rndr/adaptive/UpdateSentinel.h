// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

//

#pragma once

#include <scene_rdl2/common/fb_util/FbTypes.h>

#include <algorithm>
#include <vector>

namespace scene_rdl2 {

namespace fb_util {
    class Tiler;
}
}

namespace moonray {
    
namespace rndr {

class UpdateSentinel
{
public:
    UpdateSentinel() :
        mAdjustUpdateTimingCondition(false)
    {}

    inline void disableAdjustUpdateTiming();
    inline void enableAdjustUpdateTiming(const std::vector<unsigned> &adaptiveIterationPixSampleIdTbl);

    bool shouldUpdate(const unsigned endSampleId) const;

    inline unsigned getUpdateCounter(const unsigned adaptiveRegionTreeId) const;

private:
    bool mAdjustUpdateTimingCondition;

    // pixel variant evaluation sample count endSampleId sequence
    std::vector<unsigned> mAdaptiveIterationPixSampleIdTable;
};

inline void
UpdateSentinel::disableAdjustUpdateTiming()
{
    mAdjustUpdateTimingCondition = false;
    mAdaptiveIterationPixSampleIdTable.clear();
}

inline void
UpdateSentinel::enableAdjustUpdateTiming(const std::vector<unsigned> &adaptiveIterationPixSampleIdTbl)
{
    mAdjustUpdateTimingCondition = true;
    mAdaptiveIterationPixSampleIdTable = adaptiveIterationPixSampleIdTbl;
}

inline bool    
UpdateSentinel::shouldUpdate(const unsigned endSampleIdx) const
{
    if (!mAdjustUpdateTimingCondition) {
        return true; // If adjust update timing logic condition is disable, we always evaluate adaptiveTree
    }

    auto itr = std::lower_bound(mAdaptiveIterationPixSampleIdTable.begin(),
                                mAdaptiveIterationPixSampleIdTable.end(),
                                endSampleIdx);
    if (*itr != endSampleIdx) {
        return false; // endSampleId is not a boundary of adaptiveTree update sample number interval
    }
    return true;
}

} // namespace rndr
} // namespace moonray


