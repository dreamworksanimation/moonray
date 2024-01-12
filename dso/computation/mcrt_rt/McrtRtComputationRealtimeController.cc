// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

//
//
#include "McrtRtComputationRealtimeController.h"

#include <scene_rdl2/common/platform/Platform.h>

namespace moonray {
namespace mcrt_rt_computation {

McrtRtComputationRealtimeController::McrtRtComputationRealtimeController(const double defaultFps) :
    mConstantFps(false),               // dynamic fps mode
    mDefaultFps(defaultFps),
    mReceivedGeometryDataTime(0.0),
    mFps(defaultFps),
    mStartGapIntervalCondition(false), // outside gap interval region 
    mGapStartTime(0.0),
    mLastTime(0.0),
    mFrameInterval(0.0),
    mPureGapInterval(0.0),
    mOverrun(0.0),
    mAdjustDuration(0.0),
    mPredictFrameInterval(0.0),
    mGapInterval(0.0)
{
    initIntervalArray();
}

void
McrtRtComputationRealtimeController::markReceivedGeometryData()
{
    double oldTime = mReceivedGeometryDataTime;
    mReceivedGeometryDataTime = util::getSeconds();

    if (!mConstantFps) {
        //
        // dynamically update fps based on inteval of received geometry data when dynamicFps mode
        //
        double currInterval = (oldTime == 0.0)? 1.0 / mFps: mReceivedGeometryDataTime - oldTime;
        mFps = computeFps(currInterval);
    }
}

#define _F6_2 std::setw(6) << std::fixed << std::setprecision(2)

bool
McrtRtComputationRealtimeController::isFrameComplete()
{
    double now = util::getSeconds(); // get current time

    if (!mStartGapIntervalCondition) {
        mGapStartTime = now;    // start gap interval
        mStartGapIntervalCondition = true;
    }

    {
        //    
        //                 |<------- total time budge ------->|
        //                 |      (mPredictFrameInterval)     |
        //  prev frame ....+......................+ . . . . . +.... next frame
        //                 ^<---- render time --->^<-- gap -->^
        //                 |                      |  interval |
        //                 +- frame start         |           |
        //                    (mLastTime)         |           |
        //                                        |           |
        //                           render end --+           |
        //                           (mGapStartTime)          |
        //                                                    |
        //                                        frame end --+
        //
        //  1 frame = <render time> + <gap interval>
        //
        //  <render time> is from frame start ~ render end.
        //  <gap interval> is total time budget - render time.
        //  Sampling volume is estimated and start render.
        //  Sometimes it's finished before frame end (gap interval > 0) or
        //  overrun (gap interval is negative) depending on how accurate the estimation is.
        //  Close to zero gap interval is always good and zero is perfect.
        //

        //
        // Try to find just frame complete moment
        //
        double frameInterval = (mLastTime == 0.0)? 1.0 / mFps : now - mLastTime;
        double currOverrun = (mPredictFrameInterval == 0.0)? 0.0: frameInterval - mPredictFrameInterval;
        if (currOverrun < 0.0) {
            return false; // still inside gap interval. need to wait bit more
        }

        //
        // Entire frame is completed. We are update some timing value here.
        //
        mLastTime = now;                        // frame complete time
        mFrameInterval = frameInterval;         // last frame's time length
        mPureGapInterval = now - mGapStartTime; // compute gap interval
        mOverrun = currOverrun;                 // overrun time length

        mStartGapIntervalCondition = false;     // reset gap interval condition

    }

    if (mReceivedGeometryDataTime > 0.0) {
        //
        // Try to compute adjustment time offset to move geometry data message receive timing as
        // middle of render start and end.
        //
        double durationGeoEnd = now - mReceivedGeometryDataTime;
        double currentAdjustDuration = (1.0 / mFps * 0.5) - durationGeoEnd; // use current fps
        mAdjustDuration = computeAdjustDuration(currentAdjustDuration);
    }

    {
        //
        // Compute new gap interval value
        //
        // We've already averaged adjustDuration but try to apply one more scale here.
        // This is very experimental but works and easy to adjust overall behavior.
        //
        double adjustScale = (mConstantFps)? 0.125: 0.5; // experimental scale value
        double adjustDuration = - mOverrun + mAdjustDuration * adjustScale;
        mPredictFrameInterval = 1.0 / mFps + adjustDuration;
        mGapInterval = mPureGapInterval + adjustDuration;
    }

    return true;
}

} // namespace mcrt_rt_computation
} // namespace moonray

