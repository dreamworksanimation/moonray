// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file MotionTransform.h
/// $Id$
///
#pragma once

#include <moonray/rendering/bvh/shading/Xform.h>
#include <moonray/rendering/geom/Types.h>
#include <scene_rdl2/common/math/Math.h>

namespace moonray {
namespace geom {
namespace internal {

class MotionTransform {
public:

    MotionTransform(): mUninitializedData({Mat43(scene_rdl2::math::one)}),
        mState(State::UNINITIALIZED) {}

    MotionTransform(const shading::XformSamples& xformSamples,
            float t0, float t1) :
        mUninitializedData(xformSamples), mState(State::UNINITIALIZED),
        mSampleStart(t0), mSampleDelta(t1 - t0) {}

    // The use of union seems causing the default copy constructor and
    // assignment operator to be marked as delete...
    // need to manual implement them to avoid compile error
    MotionTransform(const MotionTransform& rhs)
    {
        memcpy(this, &rhs, sizeof(MotionTransform));
    }

    MotionTransform& operator=(const MotionTransform& rhs)
    {
        memcpy(this, &rhs, sizeof(MotionTransform));
        return *this;
    }

    const Mat43& getUninitializedSample0() const
    {
        MNRY_ASSERT(!isInitialized());
        return mUninitializedData.mSample0;
    }

    const Mat43& getUninitializedSample1() const
    {
        MNRY_ASSERT(!isInitialized());
        return mUninitializedData.mSample1;
    }

    const Mat43& getStaticXform() const
    {
        MNRY_ASSERT(isStatic());
        return mStaticData.mXform;
    }

    const Mat43& getStaticInverse() const
    {
        MNRY_ASSERT(isStatic());
        return mStaticData.mInverseXform;
    }

    Mat43 eval(float t) const
    {
        MNRY_ASSERT(isMotion());
        return scene_rdl2::math::slerp(mMotionData.mComponent0, mMotionData.mComponent1,
            mSampleStart + t * mSampleDelta).combined();
    }

    scene_rdl2::math::Xform3fv eval(const simdf& t) const
    {
        MNRY_ASSERT(isMotion());
        simdf time = mSampleStart + t * mSampleDelta;
        return scene_rdl2::math::slerp(mMotionData.mComponent0, mMotionData.mComponent1,
            *(reinterpret_cast<scene_rdl2::math::Floatv*>(&time)));
    }

    void appendXform(const shading::XformSamples& xformSamples)
    {
        MNRY_ASSERT(!isInitialized());
        mUninitializedData.appendXform(xformSamples);
    }

    void setSampleInterval(float t0, float t1)
    {
        MNRY_ASSERT(!isInitialized());
        mSampleStart = t0;
        mSampleDelta = t1 - t0;
    }

    void initialize()
    {
        // MOONRAY-4193 - https://jira.dreamworks.net/browse/MOONRAY-4193
        // We should not enter this function twice but it is happening
        // in the AssignmentAndXformSetter in GeometryManager.cc
        // MNRY_ASSERT(!isInitialized());
        if (isInitialized()) {
            return;
        }
        if ((mUninitializedData.mSample0 == mUninitializedData.mSample1) ||
            (mSampleStart == 0.0f && mSampleDelta == 0.0f)) {
            mState = State::INITIALIZED_STATIC;
            mStaticData = StaticData(mUninitializedData.mSample0);
        } else {
            scene_rdl2::math::XformComponent3f sample0;
            decompose(mUninitializedData.mSample0, sample0);
            scene_rdl2::math::XformComponent3f sample1;
            decompose(mUninitializedData.mSample1, sample1);
            correctSlerpAngle(sample0, sample1);
            float t0 = mSampleStart;
            float t1 = t0 + mSampleDelta;
            if (scene_rdl2::math::isEqual(t0, t1)) {
                mState = State::INITIALIZED_STATIC;
                mStaticData = StaticData(
                    scene_rdl2::math::slerp(sample0, sample1, t0).combined());
            } else {
                mState = State::INITIALIZED_MOTION;
                mMotionData = MotionData(sample0, sample1);
            }
        }
    }

    bool isInitialized() const { return mState != State::UNINITIALIZED; }

    bool isStatic() const { return mState == State::INITIALIZED_STATIC; }

    bool isMotion() const { return mState == State::INITIALIZED_MOTION; }

private:
    void correctSlerpAngle(scene_rdl2::math::XformComponent3f& c0,
            scene_rdl2::math::XformComponent3f& c1) {
        // avoid flipping rotation by checking the angle between two transforms
        // (one rotation matrix can be represented by two opposite quaternions)
        if (scene_rdl2::math::dot(c0.r, c1.r) < 0.0f) {
            c1.r = -c1.r;
        }
    }

private:
    enum class State {
        UNINITIALIZED,
        INITIALIZED_STATIC,
        INITIALIZED_MOTION
    };

    class UninitializedData {
    public:
        UninitializedData(const shading::XformSamples& xformSamples) {
            size_t xformSamplesCount = xformSamples.size();
            if (xformSamplesCount == 1) {
                mSample0 = xformSamples[0];
                mSample1 = xformSamples[0];
            } else if (xformSamplesCount == 2) {
                mSample0 = xformSamples[0];
                mSample1 = xformSamples[1];
            } else {
                MNRY_ASSERT(false,
                    "we only support at most two xform samples now");
            }
        }

        void appendXform(const shading::XformSamples& xformSamples) {
            size_t xformSamplesCount = xformSamples.size();
            if (xformSamplesCount == 1) {
                mSample0 = mSample0 * xformSamples[0];
                mSample1 = mSample1 * xformSamples[0];
            } else if (xformSamplesCount == 2) {
                mSample0 = mSample0 * xformSamples[0];
                mSample1 = mSample1 * xformSamples[1];
            } else {
                MNRY_ASSERT(false,
                    "we only support at most two xform samples now");
            }
        }

    public:
        Mat43 mSample0;
        Mat43 mSample1;
    };

    class StaticData {
    public:
        StaticData(const Mat43& xform):
            mXform(xform), mInverseXform(xform.inverse()) {}
    public:
        Mat43 mXform;
        Mat43 mInverseXform;
    };

    class MotionData {
    public:
        MotionData(const scene_rdl2::math::XformComponent3f& c0,
                const scene_rdl2::math::XformComponent3f& c1) :
            mComponent0(c0), mComponent1(c1) {}
    public:
        scene_rdl2::math::XformComponent3f mComponent0;
        scene_rdl2::math::XformComponent3f mComponent1;
    };

    union {
        UninitializedData mUninitializedData;
        StaticData mStaticData;
        MotionData mMotionData;
    };

    State mState;
    float mSampleStart, mSampleDelta;
};

} // namespace internal
} // namespace geom
} // namespace moonray


