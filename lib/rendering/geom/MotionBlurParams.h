// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file MotionBlurParams.h
/// $Id$
///
#pragma once

#include <scene_rdl2/common/math/Math.h>

namespace moonray {
namespace geom {

class MotionBlurParams {
public:
    MotionBlurParams() {}

    MotionBlurParams(const std::vector<float>& motionSteps,
            float shutterOpen, float shutterClose,
            bool isMotionBlurOn, float fps):
        mMotionSteps(motionSteps),
        mShutterOpen(shutterOpen),
        mShutterClose(shutterClose),
        mIsMotionBlurOn(isMotionBlurOn),
        mInvFps(1.f / fps)
    {
        // calculate the delta fraction of shutter open/close time in
        // motionSteps duration. For example: if first motion step is -1
        // and the last motion step is 1, and shutter open time is -0.75
        // shutter close time is 0.3
        // -1                                                     1
        // --------------------------------------------------------
        //      -0.75                        0.3
        // then shutterOpenDelta  = (-0.75 - (-1)) / (1 - (-1)) = 0.125
        //      shutterCloseDelta = ( 0.3  - (-1)) / (1 - (-1)) = 0.65
        //
        // this info is used to interpolate vertex data and primitive atttribute
        // from frame time coordinate to shutter time coordinate
        // (where shutter open time = 0 and shutter close time = 1)

        if (mMotionSteps.size() <= 1) {
            // static case
            mShutterOpenDelta  = 0.0f;
            mShutterCloseDelta = 0.0f;
            mDt = 0.0f;
        } else if (mMotionSteps.size() == 2) {
            // We have an animation, interpolate/extrapolate to shutter open/close
            float m0 = mMotionSteps[0];
            float m1 = mMotionSteps[1];
            if (!scene_rdl2::math::isEqual(m0, m1)) {
                mShutterOpenDelta  = (mShutterOpen  - m0) / (m1 - m0);
                mShutterCloseDelta = (mShutterClose - m0) / (m1 - m0);
            } else {
                // it makes no sense to have two same motion steps,
                // but we should avoid generating nan if it happens
                mShutterOpenDelta  = 0.0f;
                mShutterCloseDelta = 0.0f;
            }
            mT0 = m0 * mInvFps;
            mT1 = m1 * mInvFps;
            mDt = mT1 - mT0;
        } else {
            MNRY_ASSERT_REQUIRE(false, "only support two time samples for motionblur at this moment");
        }
    }

    const std::vector<float>& getMotionSteps() const { return mMotionSteps; }

    float getShutterOpen() const { return mShutterOpen; }

    float getShutterClose() const { return mShutterClose; }

    void getMotionBlurDelta(float& shutterOpenDelta, float& shutterCloseDelta) const
    {
        shutterOpenDelta = mShutterOpenDelta;
        shutterCloseDelta = mShutterCloseDelta;
    }

    void getMotionStepTimes(float& t0, float& t1) const
    {
        t0 = mT0;
        t1 = mT1;
    }

    float getDt() const { return mDt; }

    float getInvFps() const { return mInvFps; }

    bool isMotionBlurOn() const { return mIsMotionBlurOn; }

private:
    std::vector<float> mMotionSteps;
    float mShutterOpen;
    float mShutterClose;
    // delta fraction of shutter open/close time in motionSteps duration
    float mShutterOpenDelta;
    float mShutterCloseDelta;
    bool  mIsMotionBlurOn;
    float mT0, mT1, mDt;
    float mInvFps;
};

} // namespace geom
} // namespace moonray


