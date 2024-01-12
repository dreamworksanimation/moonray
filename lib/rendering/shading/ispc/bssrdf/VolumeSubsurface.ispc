// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file VolumeSubsurface.ispc

#include "VolumeSubsurface.isph"

#include <moonray/rendering/shading/ispc/PbrValidity.isph>

#include <scene_rdl2/common/math/ispc/asA.isph>
#include <scene_rdl2/common/platform/IspcUtil.isph>

// Large Default Zero Scatter SigmaT
static const float sLargeZeroScatterSigmaT = 1e5;

Color
exp(Color A) {
    return Color_ctor(exp(A.r),exp(A.g),exp(A.b));
}

void
VolumeSubsurface_init(varying VolumeSubsurface * uniform bssrdf,
                      const varying Vec3f& N,
                      const varying Color& trlColor,
                      const varying Color& trlRadius,
                      uniform bool resolveSelfIntersections,
                      const uniform Material *uniform  material,
                      const uniform intptr_t evalNormalFn,
                      const varying Color &scale,
                      varying Fresnel * uniform fresnel,
                      const uniform TraceSet * uniform traceSet)
{
    bssrdf->mScatteringColor = Color_ctor(max(0.001f, trlColor.r),
                                          max(0.001f, trlColor.g),
                                          max(0.001f, trlColor.b));

    // Clamping it to a min, low value similar to what we do in diffusion based BSSRDFs.
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
    bssrdf->mScatteringRadius = Color_ctor(max(0.001f, trlRadius.r * 0.3333f),
                                           max(0.001f, trlRadius.g * 0.3333f),
                                           max(0.001f, trlRadius.b * 0.3333f));

    bssrdf->mN = N;
    bssrdf->mResolveSelfIntersections = resolveSelfIntersections;

    // this is the mapping assuming IOR matching in boundary from original talk
    // (a) "Practical and Controllable Subsurface Scattering for Production Path Tracing"
    // Matt Jen-Yuan Chiang et al
    const Color A = bssrdf->mScatteringColor;
    const Color d = bssrdf->mScatteringRadius;
    // Disney Equation (1) in (a):
    // mAlbedo = Color(1.0f) - exp(-5.09406f * A + 2.61188f * A * A - 4.31805f * A * A * A);
    // Our custom equation
    bssrdf->mAlbedo = sWhite - exp(-3.25172*A*A*A + 1.2166*A*A - 4.75914*A);

    // Equation (2) in (a)
    Color s = Color_ctor(1.9f) - A + 3.5f * (A - Color_ctor(0.8f)) * (A - Color_ctor(0.8f));
    // Equation (3) in (a)
    bssrdf->mSigmaT = Color_ctor(1.0f / (d.r * s.r),
                                 1.0f / (d.g * s.g),
                                 1.0f / (d.b * s.b));

    // Dwivedi Sampling Parameters
    // References:
    // (1) https://cgg.mff.cuni.cz/~jaroslav/papers/2014-zerovar/2014-zerovar-abstract.pdf
    // (2) https://jo.dreggn.org/home/2016_dwivedi.pdf
    // (3) https://jo.dreggn.org/home/2016_dwivedi_additional.pdf
    // Look at the supplemental notes for the formula to
    // derive V0
    const Color omA = sWhite - bssrdf->mAlbedo;
    const Color kappa0 = sqrt(3.0f*omA) *
            (sWhite -
             omA*(2.0f/5.0f + omA*(12.0f/175.0f + omA*(2.f/125.f + omA*(166.f/67375.f)))));
    bssrdf->mDwivediV0 = rcp(kappa0);
    bssrdf->mDwivediNormPDF = log((bssrdf->mDwivediV0 + sWhite)/
                                (bssrdf->mDwivediV0 - sWhite));
    bssrdf->mDwivediNormPDF = rcp(bssrdf->mDwivediNormPDF);

    // Zero Scatter - when a path exits the surface without scattering inside
    // Pure Beer's Law Absorption Case:
    bssrdf->mZeroScatterSigmaT.r = isZero(bssrdf->mScatteringRadius.r) ? sLargeZeroScatterSigmaT :
            -log(bssrdf->mScatteringColor.r - sEpsilon) / bssrdf->mScatteringRadius.r;
    bssrdf->mZeroScatterSigmaT.g = isZero(bssrdf->mScatteringRadius.g) ? sLargeZeroScatterSigmaT :
            -log(bssrdf->mScatteringColor.g - sEpsilon) / bssrdf->mScatteringRadius.g;
    bssrdf->mZeroScatterSigmaT.b = isZero(bssrdf->mScatteringRadius.b) ? sLargeZeroScatterSigmaT :
            -log(bssrdf->mScatteringColor.b - sEpsilon) / bssrdf->mScatteringRadius.b;

    bssrdf->mType = SUBSURFACE_RANDOM_WALK;
    bssrdf->mFresnelMask = lanemask();
    bssrdf->mMask = lanemask();
    bssrdf->mLabel = 0;
    bssrdf->mPropertyFlags =
            SUBSURFACE_PROPERTY_COLOR | SUBSURFACE_PROPERTY_RADIUS | SUBSURFACE_PROPERTY_PBR_VALIDITY;

    bssrdf->mFresnel = fresnel;
    bssrdf->mScale = scale;

    bssrdf->mTraceSet = traceSet;
    bssrdf->mMaterial = material;
    bssrdf->mEvalNormalFn = evalNormalFn;
}

void
VolumeSubsurface_init(varying VolumeSubsurface * uniform vS,
                      const varying Vec3f &N,
                      const varying Color &albedo,
                      const varying Color &radius,
                      uniform bool resolveSelfIntersections,
                      const uniform Material *uniform  material,
                      const uniform intptr_t evalNormalFn)
{
    VolumeSubsurface_init(vS,
                          N,
                          albedo,
                          radius,
                          resolveSelfIntersections,
                          material,
                          evalNormalFn,
                          (varying Color) sWhite,   // scale
                          nullptr,                  // Fresnel
                          nullptr                   // traceSet
                          );
}

uniform bool
VolumeSubsurface_getProperty(const varying VolumeSubsurface * uniform bssrdf,
                             uniform SubsurfaceProperty property,
                             varying float * uniform dest)
{
    uniform bool handled = true;

    switch (property)
    {
    case SUBSURFACE_PROPERTY_COLOR:
        *dest       = bssrdf->mScatteringColor.r;
        *(dest + 1) = bssrdf->mScatteringColor.g;
        *(dest + 2) = bssrdf->mScatteringColor.b;
        break;
    case SUBSURFACE_PROPERTY_RADIUS:
        *dest       = bssrdf->mScatteringRadius.r;
        *(dest + 1) = bssrdf->mScatteringRadius.g;
        *(dest + 2) = bssrdf->mScatteringRadius.b;
        break;
    case SUBSURFACE_PROPERTY_PBR_VALIDITY:
        {
            Color res = computeAlbedoPbrValidity(bssrdf->mScatteringColor);
            *dest       = res.r;
            *(dest + 1) = res.g;
            *(dest + 2) = res.b;
        }
        break;
    default:
        handled = false;
    }

    return handled;

}

//----------------------------------------------------------------------------

#pragma ignore warning(all)
ISPC_UTIL_EXPORT_STRUCT_TO_HEADER(VolumeSubsurface);

