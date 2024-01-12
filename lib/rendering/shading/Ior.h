// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#pragma once

#include <moonray/rendering/bvh/shading/State.h>
#include <scene_rdl2/common/math/Color.h>

namespace moonray {
namespace shading {

/**
 * This structure stores incident and transmitted Index Of Refraction (Ior)
 * as well as their ratio
 */
class ShaderIor
{
public:

    /// @brief Initialize a ShaderIor object given the current shading state and
    /// material index of refraction. It automatically handles computing the correct
    /// incident and transmitted indices of refraction, as well as their ratio.
    ShaderIor(const shading::State &state, const float materialIor, bool isThinGeometry = false)
    {
        // If the geometry is infinitely "thin" (non-manifold, not water-tight) then we should always treat
        // the state as 'entering' regardless of what state.isEntering() reports. There can never be
        // a case where the ray origin is "inside" the geometry because it is infinitely thin, so we treat the
        // observer as always "outside" the thin geometry.
        mIncident = (state.isEntering() || isThinGeometry) ? state.getMediumIor() : materialIor;
        mTransmitted = (state.isEntering() || isThinGeometry) ? materialIor : state.getMediumIor();

        mRatio = mIncident * scene_rdl2::math::rcp(mTransmitted);
    }

    /// @brief This special constructor is only for cases where we need to explicitly specify
    /// the mediumIor using a value that comes from the shader rather than the state.
    /// This usage case should be extremely rare and therefore this constructor should
    /// very rarely be used.
    ShaderIor(const shading::State &state, const float mediumIor, const float materialIor, bool isThinGeometry = false)
    {
        mIncident = state.isEntering() ? mediumIor : isThinGeometry ? mediumIor : materialIor;
        mTransmitted = state.isEntering() ? materialIor : isThinGeometry ? materialIor : mediumIor;
        mRatio = mIncident * scene_rdl2::math::rcp(mTransmitted);
    }

    /// Accessors
    float getIncident() const       {  return mIncident;  }
    float getTransmitted() const    {  return mTransmitted;  }
    float getRatio() const          {  return mRatio;  }

private:
    float mIncident;
    float mTransmitted;
    float mRatio;            // mIncident / mTransmitted
};

/**
 * This class stores complex index of refraction that is generally used to represent
 * metallic surface interactions.
 */
class ShaderComplexIor
{
public:

    /// @brief Initialize a ShaderComplexIor object given the material complex index of refraction.
    ShaderComplexIor(const scene_rdl2::math::Color& eta,
                     const scene_rdl2::math::Color& absorption) :
                         mEta(eta),
                         mAbsorption(absorption)
    { }

    static ShaderComplexIor
    createComplexIorFromColor(const scene_rdl2::math::Color& color,
                              const scene_rdl2::math::Color& edgeColor)
    {
        // Make sure no color channel is at 1.0
        const scene_rdl2::math::Color maxRefl = scene_rdl2::math::Color(0.999f);
        const scene_rdl2::math::Color clampedColor = clamp(color, scene_rdl2::math::sBlack, maxRefl);
        const scene_rdl2::math::Color clampedEdgeColor = clamp(edgeColor, scene_rdl2::math::sBlack, maxRefl);
        // Convert the colors to complex IOR
        const scene_rdl2::math::Color eta = computeEta(clampedColor,
                                                       clampedEdgeColor);
        const scene_rdl2::math::Color k   = computeK(clampedColor,
                                                     eta);

        return ShaderComplexIor(eta, k);
    }

    scene_rdl2::math::Color getEta()        const     { return mEta; }
    scene_rdl2::math::Color getAbsorption() const     { return mAbsorption; }

private:
    // functions for computing complex IOR values for conductor Fresnel
    // from 'reflectivity' and 'edge tint' colors.
    // See paper: "Artist Friendly Metallic Fresnel", by Ole Gulbrandsen
    // from Framestore, published at JCGT in 2014 (http://jcgt.org)
    finline static scene_rdl2::math::Color
    nMin(const scene_rdl2::math::Color &r)
    {
        return (scene_rdl2::math::sWhite - r) / (scene_rdl2::math::sWhite + r);
    }

    finline static scene_rdl2::math::Color
    nMax(const scene_rdl2::math::Color &r)
    {
        scene_rdl2::math::Color rSqrt = sqrt(r);
        return (scene_rdl2::math::sWhite + rSqrt) / (scene_rdl2::math::sWhite - rSqrt);
    }

    finline static scene_rdl2::math::Color
    computeEta(const scene_rdl2::math::Color &r, const scene_rdl2::math::Color &g)
    {
        return g * nMin(r) + (scene_rdl2::math::sWhite - g) * nMax(r);
    }

    finline static scene_rdl2::math::Color
    computeK(const scene_rdl2::math::Color &r, const scene_rdl2::math::Color &n)
    {
        const scene_rdl2::math::Color a = n + scene_rdl2::math::sWhite;
        const scene_rdl2::math::Color b = n - scene_rdl2::math::sWhite;
        // Take a max() here to get rid of any numerical -0 etc
        const scene_rdl2::math::Color nr = max(scene_rdl2::math::sBlack, r * a * a - b * b);
        return sqrt(nr / (scene_rdl2::math::sWhite - r));
    }

private:
    scene_rdl2::math::Color mEta;
    scene_rdl2::math::Color mAbsorption;
};


} // namespace shading
} // namespace moonray

