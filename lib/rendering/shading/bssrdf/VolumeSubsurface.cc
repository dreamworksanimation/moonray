// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file VolumeSubsurface.cc

#include "VolumeSubsurface.h"

#include <moonray/rendering/shading/PbrValidity.h>

#include <scene_rdl2/common/math/ispc/Typesv.h>
#include <scene_rdl2/render/util/Arena.h>

// TODO for some reasons the IOR matching mapping below produce a much closer
// albedo matching in semi-infinite slab case than the revised mapping with
// IOR 1.4 as boundary condition.
//#define USE_IOR14_ALBEDO_INVERSION

namespace moonray {
namespace shading {

// Large Default Zero Scatter SigmaT
static constexpr float sLargeZeroScatterSigmaT = 1e5;

VolumeSubsurface::VolumeSubsurface(const scene_rdl2::math::Color& trlColor,
                                   const scene_rdl2::math::Color& trlRadius,
                                   const scene_rdl2::rdl2::Material* material,
                                   scene_rdl2::rdl2::EvalNormalFunc evalNormalFn,
                                   const scene_rdl2::math::Vec3f& N,
                                   bool resolveSelfIntersections):
            mScatteringColor( scene_rdl2::math::max(0.001f, trlColor[0]),
                              scene_rdl2::math::max(0.001f, trlColor[1]),
                              scene_rdl2::math::max(0.001f, trlColor[2])),
            mScatteringRadius(scene_rdl2::math::max(0.001f, trlRadius[0] * 0.3333f),
                              scene_rdl2::math::max(0.001f, trlRadius[1] * 0.3333f),
                              scene_rdl2::math::max(0.001f, trlRadius[2] * 0.3333f)),
            mScale(1.0f),
            mFresnel(nullptr),
            mTraceSet(nullptr),
            mMaterial(material),
            mEvalNormalFunc(evalNormalFn),
            mLabel(0),
            mPropertyFlags(PROPERTY_COLOR | PROPERTY_RADIUS | PROPERTY_PBR_VALIDITY),
            mN(N),
            mResolveSelfIntersections(resolveSelfIntersections)
{
    // Clamping mScatteringColor to a min, low value similar to what we do in diffusion based BSSRDFs.
    // Scale the scattering radius by 3
    // For Random-Walk SSS, we use the Disney equations from
    // "Practical and Controllable Subsurface Scattering for Production Path Tracing"
    // to convert the scattering-radius to extinction coefficient (sigmaT).
    // This differs from our diffusion-based bssrdf in that the scattering-radius
    // gets used to calculate volume mean free path and not diffuse mean free path.
    // This makes the same scattering-radius value behave wildly differently between the
    // different SSS models. In addition, the sigmaT equations are based on
    // Disney simulations to calculate a scaling factor based on the volume mean free path
    // set to 1 ("Approximate Reflectance Profiles for Subsurface Scattering").
    // Now, modeling tells us that we still model in shrekles and it appears Disney models in cm.
    // 1 shrekle = 3cm
    // Modeling confirms there are no plans to change this until PipeX.
    // Via testing, it appears, dividing the scattering-radius by 3 to bring it in cm
    // gives:
    // (a) comparable behavior with the diffusion-based SSS and
    // (b) higher range of values to be used with SSS
    // (c) an artist-workflow where we set the scattering-radius based on the scale observed via usdview/mm_view etc.
    // Similar scaling seems to exist in PBRTv3:
    // https://github.com/mmp/pbrt-v3/blob/9f717d847a807793fa966cf0eaa366852efef167/src/materials/disney.cpp
    // Line 371
    // Here the input scattering-radius get scaled by 0.2 with a weird note alluding to
    // private conversations with Brent Burley and Matt Chiang. Equally Suspect, but only to have some
    // precedence for scaling the scattering radius.

    // albedo inversion that transform user specified multiple scattering color
    // and scattering radius to single scattering volume coefficient
    // sigmaT and albedo
    // the mapping is a third order exponential-of polynomial curve fitting
#ifdef USE_IOR14_ALBEDO_INVERSION
    // with the assumption that transmission IOR is 1.4 from
    // "The Design and Evolution of Disney's Hyperion Renderer" - Burley et al
    // 4.4.2 - Path-Traced Skin
    const scene_rdl2::math::Color& A = trlColor;
    const scene_rdl2::math::Color& d = trlRadius;
    mAlbedo = scene_rdl2::math::Color(1.0f) - exp(-11.43f * A + 15.38f * A * A - 13.91f * A * A * A);
    scene_rdl2::math::Color s = scene_rdl2::math::Color(
        4.012f) - 15.21f * A + 32.34f * A * A - 34.68f * A * A * A + 13.91f * A * A * A * A;
    mSigmaT = scene_rdl2::math::Color(1.0f / (d[0] * s[0]), 1.0f / (d[1] * s[1]), 1.0f / (d[2] * s[2]));
#else
    // this is the mapping assuming IOR matching in boundary from original talk
    // (a) "Practical and Controllable Subsurface Scattering for Production Path Tracing"
    // Matt Jen-Yuan Chiang et al
    const scene_rdl2::math::Color A = mScatteringColor;
    const scene_rdl2::math::Color d = mScatteringRadius;
    // Disney Equation (1) in (a):
    // mAlbedo = Color(1.0f) - exp(-5.09406f * A + 2.61188f * A * A - 4.31805f * A * A * A);
    // Our custom equation - see MOONSHINE-1027
    mAlbedo = scene_rdl2::math::Color(1.0f) - exp(-3.25172*A*A*A + 1.2166*A*A - 4.75914*A);

    // Equation (2) in (a)
    scene_rdl2::math::Color s = scene_rdl2::math::Color(1.9f) - A + 3.5f * (A - scene_rdl2::math::Color(0.8f)) * (A - scene_rdl2::math::Color(0.8f));
    // Equation (3) in (a)
    mSigmaT = scene_rdl2::math::Color(1.0f / (d[0] * s[0]),
                                      1.0f / (d[1] * s[1]),
                                      1.0f / (d[2] * s[2]));

    // Dwivedi Sampling Parameters
    // References:
    // (1) https://cgg.mff.cuni.cz/~jaroslav/papers/2014-zerovar/2014-zerovar-abstract.pdf
    // (2) https://jo.dreggn.org/home/2016_dwivedi.pdf
    // (3) https://jo.dreggn.org/home/2016_dwivedi_additional.pdf
    // Look at the supplemental notes for the formula to
    // derive V0
    const scene_rdl2::math::Color omA = scene_rdl2::math::sWhite - mAlbedo;
    const scene_rdl2::math::Color kappa0 = scene_rdl2::math::sqrt(3.0f*omA) *
            (scene_rdl2::math::sWhite -
             omA*(2.0f/5.0f + omA*(12.0f/175.0f + omA*(2.f/125.f + omA*166.f/67375.f))));
    mDwivediV0 = scene_rdl2::math::rcp(kappa0);
    mDwivediNormPDF = scene_rdl2::math::log((mDwivediV0 + scene_rdl2::math::sWhite)/
                                (mDwivediV0 - scene_rdl2::math::sWhite));
    mDwivediNormPDF = scene_rdl2::math::rcp(mDwivediNormPDF);

    // Zero Scatter - when a path exits the surface without scattering inside
    // Pure Beer's Law Absorption Case:
    mZeroScatterSigmaT[0] = scene_rdl2::math::isZero(mScatteringRadius[0]) ? sLargeZeroScatterSigmaT :
            -scene_rdl2::math::log(mScatteringColor[0] - scene_rdl2::math::sEpsilon) / mScatteringRadius[0];
    mZeroScatterSigmaT[1] = scene_rdl2::math::isZero(mScatteringRadius[1]) ? sLargeZeroScatterSigmaT :
            -scene_rdl2::math::log(mScatteringColor[1] - scene_rdl2::math::sEpsilon) / mScatteringRadius[1];
    mZeroScatterSigmaT[2] = scene_rdl2::math::isZero(mScatteringRadius[2]) ? sLargeZeroScatterSigmaT :
            -scene_rdl2::math::log(mScatteringColor[2] - scene_rdl2::math::sEpsilon) / mScatteringRadius[2];

#endif
}

inline scene_rdl2::math::Color
colorvToColor(const scene_rdl2::math::Colorv &col, int lane)
{
    return scene_rdl2::math::Color(col.r[lane],
                       col.g[lane],
                       col.b[lane]);
}

VolumeSubsurface::VolumeSubsurface(scene_rdl2::alloc::Arena *arena,
                                   const VolumeSubsurfacev &volumeSubsurfacev,
                                   int lane) :
                                           mFresnel(nullptr)
{
    if (volumeSubsurfacev.mFresnel && ((1 << lane) & volumeSubsurfacev.mFresnelMask)) {
        setTransmissionFresnel(createFresnel(arena, volumeSubsurfacev.mFresnel, lane));
    }

    mScatteringColor = colorvToColor(volumeSubsurfacev.mScatteringColor, lane);
    mScatteringRadius = colorvToColor(volumeSubsurfacev.mScatteringRadius, lane);
    mSigmaT = colorvToColor(volumeSubsurfacev.mSigmaT, lane);
    mZeroScatterSigmaT = colorvToColor(volumeSubsurfacev.mZeroScatterSigmaT, lane);
    mAlbedo = colorvToColor(volumeSubsurfacev.mAlbedo, lane);
    mDwivediV0 = colorvToColor(volumeSubsurfacev.mDwivediV0, lane);
    mDwivediNormPDF = colorvToColor(volumeSubsurfacev.mDwivediNormPDF, lane);

    mScale = colorvToColor(volumeSubsurfacev.mScale, lane);

    mLabel = volumeSubsurfacev.mLabel;
    mPropertyFlags = volumeSubsurfacev.mPropertyFlags;
    mTraceSet = (const scene_rdl2::rdl2::TraceSet *)volumeSubsurfacev.mTraceSet;
    mMaterial = (const scene_rdl2::rdl2::Material*)volumeSubsurfacev.mMaterial;
    mEvalNormalFunc = (scene_rdl2::rdl2::EvalNormalFunc)volumeSubsurfacev.mEvalNormalFn;

    mN = scene_rdl2::math::Vec3f(volumeSubsurfacev.mN.x[lane],
                                 volumeSubsurfacev.mN.y[lane],
                                 volumeSubsurfacev.mN.z[lane]);

    mResolveSelfIntersections = volumeSubsurfacev.mResolveSelfIntersections;
}

bool
VolumeSubsurface::getProperty(Property property, float *dest) const
{
    bool handled = true;
    switch (property)
    {
    case PROPERTY_COLOR:
        *dest       = mScatteringColor[0];
        *(dest + 1) = mScatteringColor[1];
        *(dest + 2) = mScatteringColor[2];
        break;
    case PROPERTY_RADIUS:
        *dest       = mScatteringRadius[0];
        *(dest + 1) = mScatteringRadius[1];
        *(dest + 2) = mScatteringRadius[2];
        break;
    case PROPERTY_PBR_VALIDITY:
        {
        scene_rdl2::math::Color result = computeAlbedoPbrValidity(mScatteringColor);
        *dest       = result.r;
        *(dest + 1) = result.g;
        *(dest + 2) = result.b;
        }
    default:
        handled = false;
        break;
    }
    return handled;
}

VolumeSubsurface*
createVolumeSubsurface(scene_rdl2::alloc::Arena* arena,
                       const scene_rdl2::math::Color& trlColor,
                       const scene_rdl2::math::Color& trlRadius,
                       const scene_rdl2::rdl2::Material* material,
                       scene_rdl2::rdl2::EvalNormalFunc evalNormalFn,
                       const scene_rdl2::math::Vec3f& N,
                       bool resolveSelfIntersections)
{
    return arena->allocWithArgs<VolumeSubsurface>(trlColor,
                                                  trlRadius,
                                                  material,
                                                  evalNormalFn,
                                                  N,
                                                  resolveSelfIntersections);
}

VolumeSubsurface*
createVolumeSubsurface(scene_rdl2::alloc::Arena* arena,
                       const VolumeSubsurfacev* volumeSubsurfacev,
                       int lane)
{
    VolumeSubsurface *volumeSubsurface = nullptr;
    if (volumeSubsurfacev && ((1 << lane) & volumeSubsurfacev->mMask)) {
        volumeSubsurface =
            arena->allocWithArgs<VolumeSubsurface>(arena,
                                                   *((const VolumeSubsurfacev *)volumeSubsurfacev),
                                                   lane);
    }
    return volumeSubsurface;
}

} // namespace shading
} // namespace moonray

