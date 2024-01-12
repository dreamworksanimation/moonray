// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

//
//
#pragma once

#include <vector>
#include <stddef.h>

namespace moonray {
namespace mcrt_rt_computation {

class McrtRtComputationRealtimeController
{
public:
    McrtRtComputationRealtimeController(const double defaultFps);

    // We have 2 modes and select one of them. ConstantFps and DynamicFps.
    void setConstantFps(const double fps) { mConstantFps = true; mFps = fps; }
    void setDynamicFps() { mConstantFps = false; mFps = mDefaultFps; }

    void markReceivedGeometryData(); // Called when received geometryData message to compute dynamic fps rate
    double getFps() const { return mFps; }

    // Main function to adjust timing. Internally update parameters and try to estimate next frame time budget
    bool isFrameComplete();     // Run through entire time budget (=true) or should render more (=false) ?

    // Getter result of estimation for next frame.
    double getFrameInterval() const { return mFrameInterval; } // return last frame's itnerval sec for logging purpose
    double getPureGapInterval() const { return mPureGapInterval; } // return pure gap interval sec for logging purpose
    double getOverrun() const { return mOverrun; } // return overrun sec for logging purpose
    double getAdjustDuration() const { return mAdjustDuration; } // return adjust duration for loggin purpose
    double getGapInterval() const { return mGapInterval; } // return updated gap interval for next frame control

private:
    bool mConstantFps;   // true  : use constant fps value which user defined.
                         // false : dynamically update fps based on interval of received geometry message (default)
    double mDefaultFps;  // used for dynamic rate of fps

    double mReceivedGeometryDataTime;         // sec

    std::vector<double> mIntervalArray;
    std::vector<double> mAdjustArray; // value buffer for adjust duration which related to geometry message receiving timing

    double mFps;                              // current fps

    //
    //                    last frame             +--- now (mLastTime)
    //        |<------- total time budget ------>|
    //        |         (mFrameInterval)         V
    //    ....+......................+ . . . . . +.... next frame
    //        ^<---- render time --->^<-- gap -->^
    //        |                      |  interval |
    //        +- frame start         |     ^     |
    //           (mLastTime)         |     |     |
    //                               |     +-----|--- mPureGapInterval
    //                  render end --+           |
    //                  (mGapStartTime)          |
    //                                           |
    //                               frame end --+
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
    bool mStartGapIntervalCondition; // condition of GapInterval is started or not. Init condition = false;
    double mGapStartTime;            // sec : GapInterval was started at this timing.

    double mLastTime;             // sec : last frame's rendering stop timing (= last gap interval end timing)
    double mFrameInterval;        // sec : last frame's interval sec (= last render time + last gap interval)
    double mPureGapInterval;      // sec : last frame's gap interval (= mLastTiem - mGapStartTime)
    double mOverrun;              // sec : last frame's overrun sec (delta from predict frame end timing)
    double mAdjustDuration;       // sec : adjust duration which related to geometry data message receiving timing
    double mPredictFrameInterval; // sec : time budget of next frame.
    double mGapInterval;          // sec : updated (might be extended) gap interval for next frame. (see isFrameComplete())

    void initIntervalArray() {
        // Several different test, 5 frame average works reasonable so far.
        // This size is defines sensitivity of adjustDuration and dynamicFPS value change.
        static const int size = 5;
        mIntervalArray.resize(size, 1.0 / mDefaultFps); // take 5 frames average
        mAdjustArray.resize(size, 0.0);
    }

    double computeFps(const double cInterval) {
        return 1.0 / shiftPushAndAverageArray(mIntervalArray, cInterval);
    }

    double computeAdjustDuration(const double currDuration) {
        //
        // Returns recent averaged adjust duration values. This logic works well to remove unpredicted spike
        // of adjust duration change. In other word, adjust duration should not change sensitibly.
        // Sensitibity is controlled by number of array (see initIntervalArray()).
        // Big buffer -> insensitive and flat, Small buffer -> peaky.
        //
        return shiftPushAndAverageArray(mAdjustArray, currDuration);
    }

    double shiftPushAndAverageArray(std::vector<double> &array, double v) {
        //
        // We only keep some number of recent values and averaged them.
        // number is defined by initIntervalArray()
        // 
        double total = 0.0;
        for (size_t i = array.size() - 1; i > 0; --i) {
            array[i] = array[i - 1];
            total += array[i];
        }
        array[0] = v;
        total += array[0];
        return total / (double)array.size();
    }
};

} // namespace mcrt_rt_computation
} // namespace moonray

