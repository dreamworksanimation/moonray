// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <moonray/rendering/shading/Util.h>
#include <scene_rdl2/common/math/Vec3.h>

#include <moonray/rendering/shading/ispc/BsdfComponent_ispc_stubs.h>

namespace moonray {
namespace shading {

class Iridescence
{
public:
    Iridescence(const scene_rdl2::math::Vec3f& N,
                const float strength,
                const scene_rdl2::math::Color &primary,
                const scene_rdl2::math::Color &secondary,
                const bool flipHue,
                const float thickness,
                const float exponent,
                const float iridescenceAt0,
                const float iridescenceAt90) :
        mN(N),
        mStrength(strength),
        mColorControl(ispc::SHADING_IRIDESCENCE_COLOR_USE_HUE_INTERPOLATION),
        mPrimary(primary),
        mSecondary(secondary),
        mFlipHue(flipHue),
        mRampInterpolationMode(ispc::COLOR_RAMP_CONTROL_SPACE_RGB),
        mRampNumPoints(0),
        mRampPositions(nullptr),
        mRampInterpolators(nullptr),
        mRampColors(nullptr),
        mThickness(thickness),
        mExponent(exponent),
        mIridescenceAt0(iridescenceAt0),
        mIridescenceAt90(iridescenceAt90)
    {}

    Iridescence(const scene_rdl2::math::Vec3f& N,
                const float strength,
                const ispc::ColorRampControlSpace rampInterpolationMode,
                const int numRampPoints,
                const float* rampPositions,
                const ispc::RampInterpolatorMode* rampInterpolators,
                const scene_rdl2::math::Color* rampColors,
                const float thickness,
                const float exponent,
                const float iridescenceAt0,
                const float iridescenceAt90) :
        mN(N),
        mStrength(strength),
        mColorControl(ispc::SHADING_IRIDESCENCE_COLOR_USE_RAMP),
        mFlipHue(false),
        mRampInterpolationMode(rampInterpolationMode),
        mRampNumPoints(numRampPoints),
        mRampPositions(rampPositions),
        mRampInterpolators(rampInterpolators),
        mRampColors(rampColors),
        mThickness(thickness),
        mExponent(exponent),
        mIridescenceAt0(iridescenceAt0),
        mIridescenceAt90(iridescenceAt90)
    {}

    ~Iridescence() {}

    finline const scene_rdl2::math::Vec3f& getN()                    const { return mN; }
    finline float getStrength()                                      const { return mStrength; }
    finline ispc::SHADING_IridescenceColorMode  getColorControl()    const { return mColorControl; }
    finline const scene_rdl2::math::Color& getPrimary()              const { return mPrimary; }
    finline const scene_rdl2::math::Color& getSecondary()            const { return mSecondary; }
    finline bool  getFlipHue()                                       const { return mFlipHue; }
    finline ispc::ColorRampControlSpace getRampInterpolationMode()   const { return mRampInterpolationMode; }
    finline int   getRampNumPoints()                                 const { return mRampNumPoints; }
    finline const float* getRampPositions()                          const { return mRampPositions; }
    finline const ispc::RampInterpolatorMode* getRampInterpolators() const { return mRampInterpolators; }
    finline const scene_rdl2::math::Color* getRampColors()           const { return mRampColors; }
    finline float getThickness()                                     const { return mThickness; }
    finline float getExponent()                                      const { return mExponent; }
    finline float getIridescenceAt0()                                const { return mIridescenceAt0; }
    finline float getIridescenceAt90()                               const { return mIridescenceAt90; }

private:
    const scene_rdl2::math::Vec3f mN;
    const float mStrength;
    const ispc::SHADING_IridescenceColorMode mColorControl;
    const scene_rdl2::math::Color mPrimary;
    const scene_rdl2::math::Color mSecondary;
    const bool  mFlipHue;
    const ispc::ColorRampControlSpace mRampInterpolationMode;
    const int mRampNumPoints;
    const float* mRampPositions;
    const ispc::RampInterpolatorMode* mRampInterpolators;
    const scene_rdl2::math::Color* mRampColors;
    const float mThickness;
    const float mExponent;
    const float mIridescenceAt0;
    const float mIridescenceAt90;
};

} // end namespace shading
} // end namespace moonray

