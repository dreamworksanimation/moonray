// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file Bssrdf.cc

#include "Bssrdf.h"
#include "Dipole.h"
#include "NormalizedDiffusion.h"
#include <moonray/rendering/shading/bsdf/Fresnel.h>
#include <moonray/rendering/shading/Util.h>
#include <scene_rdl2/render/util/Arena.h>


using namespace scene_rdl2;
using namespace scene_rdl2::math;

namespace moonray {
namespace shading {

//----------------------------------------------------------------------------

Bssrdf::Bssrdf(const Vec3f &N, int32_t propertyFlags,
               const scene_rdl2::rdl2::Material* material,
               scene_rdl2::rdl2::EvalNormalFunc evalNormalFn) :
    mFresnel(nullptr),
    mFrame(N),
    mScale(scene_rdl2::math::sWhite),
    mMaxRadius(0.f),
    mLabel(0),
    mPropertyFlags(propertyFlags),
    mBssrdfTraceSet(nullptr),
    mMaterial(material),
    mEvalNormalFunc(evalNormalFn)
{
}

Bssrdf::Bssrdf(scene_rdl2::alloc::Arena *arena, const Bssrdfv &bssrdfv, int lane)
  : mFresnel(nullptr)
{
    if (bssrdfv.mFresnel && ((1 << lane) & bssrdfv.mFresnelMask)) {
        setTransmissionFresnel(createFresnel(arena, bssrdfv.mFresnel, lane));
    }

    mFrame.mX = Vec3f(bssrdfv.mFrame.mX.x[lane], bssrdfv.mFrame.mX.y[lane], bssrdfv.mFrame.mX.z[lane]);
    mFrame.mY = Vec3f(bssrdfv.mFrame.mY.x[lane], bssrdfv.mFrame.mY.y[lane], bssrdfv.mFrame.mY.z[lane]);
    mFrame.mZ = Vec3f(bssrdfv.mFrame.mZ.x[lane], bssrdfv.mFrame.mZ.y[lane], bssrdfv.mFrame.mZ.z[lane]);
    mScale = Color(bssrdfv.mScale.r[lane], bssrdfv.mScale.g[lane], bssrdfv.mScale.b[lane]);
    mMaxRadius = bssrdfv.mMaxRadius[lane];
    mLabel = bssrdfv.mLabel;
    mPropertyFlags = bssrdfv.mPropertyFlags;
    mBssrdfTraceSet = (const scene_rdl2::rdl2::TraceSet *)bssrdfv.mBssrdfTraceSet;
    mMaterial = (const scene_rdl2::rdl2::Material*)bssrdfv.mMaterial;
    mEvalNormalFunc = (scene_rdl2::rdl2::EvalNormalFunc)bssrdfv.mEvalNormalFn;
}

//----------------------------------------------------------------------------

float
Bssrdf::sampleGlobal(float r1, float r2, float cosThetaMax, Vec3f &ws) const
{
    // Sample direction within a spherical cap below the surface
    ws = sampleLocalSphericalCapUniform(r1, r2, cosThetaMax);

    float pdf = 1.0f / (sTwoPi * (1.0f - cosThetaMax));

    return pdf;
}

bool
Bssrdf::getProperty(Property property, float *dest) const
{
    MNRY_ASSERT(0 && "unknown property");
    return false; // no properties handled at the base class level
}

//----------------------------------------------------------------------------

Bssrdf *
createBSSRDF(ispc::SubsurfaceType type,
             scene_rdl2::alloc::Arena* arena,
             const Vec3f &N,
             const Color& trlColor,
             const Color& trlRadius,
             const scene_rdl2::rdl2::Material* material,
             const scene_rdl2::rdl2::EvalNormalFunc evalNormalFn)
{
    Bssrdf *bssrdf = nullptr;

    switch(type) {
    case ispc::SubsurfaceType::SUBSURFACE_DIPOLE_DIFFUSION:
        bssrdf = arena->allocWithArgs<DipoleBssrdf>(N, /* eta = */ 1.0f, trlColor, trlRadius, material, evalNormalFn);
        break;

    case ispc::SubsurfaceType::SUBSURFACE_NORMALIZED_DIFFUSION:
    default :
        bssrdf = arena->allocWithArgs<NormalizedDiffusionBssrdf>(N, trlColor, trlRadius, material, evalNormalFn);
        break;
    }
    return bssrdf;
}

Bssrdf *
createBSSRDF(scene_rdl2::alloc::Arena* arena,
             const Bssrdfv *bssrdfv,
             int lane)
{
    Bssrdf *bssrdf = nullptr;
    if (bssrdfv && ((1 << lane) & bssrdfv->mMask)) {

        switch (bssrdfv->mType) {
        case ispc::SUBSURFACE_DIPOLE_DIFFUSION:
            bssrdf = arena->allocWithArgs<DipoleBssrdf>(arena, *((const DipoleBssrdfv *)bssrdfv), lane);
            break;

        case ispc::SUBSURFACE_NORMALIZED_DIFFUSION:
            bssrdf = arena->allocWithArgs<NormalizedDiffusionBssrdf>(arena, *((const NormalizedDiffusionBssrdfv *)bssrdfv), lane);
            break;

        default:
            MNRY_ASSERT(0);
        }
    }
    return bssrdf;
}

} // namespace shading
} // namespace moonray

