// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

//
//
#pragma once

#include <scene_rdl2/common/rec_time/RecTimeLap.h>

namespace moonray {
namespace mcrt_rt_dispatch_computation {

class McrtRtDispatchComputationStatistics
{
public:
    McrtRtDispatchComputationStatistics() {
        mLap.setName("==>> Dispatch onIdle() <<==");
        mLap.setMessageInterval(MSG_INTERVAL_SEC);
        mId_whole = mLap.sectionRegistration("    whole");
        mId_geo   = mLap.sectionRegistration("      geo");
        mId_rdl   = mLap.sectionRegistration("      rdl");
    }

    static const float MSG_INTERVAL_SEC = 10.0f;

    rec_time::RecTimeLap mLap;
    size_t mId_whole;
    size_t mId_geo;
    size_t mId_rdl;
};
    
} // namespace mcrt_rt_dispatch_computation
} // namespace moonray

