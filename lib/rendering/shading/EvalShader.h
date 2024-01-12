// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file EvalShader.h
/// $Id$
///

#pragma once

#include <moonray/rendering/shading/BsdfUtil.hh>

#include <moonray/rendering/bvh/shading/State.h>
#include <scene_rdl2/scene/rdl2/rdl2.h>
#include <scene_rdl2/scene/rdl2/Material.h>
#include <scene_rdl2/common/math/Math.h>
#include <scene_rdl2/common/math/Color.h>

namespace moonray {
namespace shading {

class Bsdf;
class BsdfBuilder;
class TLState;

///
/// This header contains low-level API calls to run shaders (material, map,
/// displacement, etc.). These API calls are used by rendering applications
/// to call shaders directly (integrator, displacement mapping) or by high-level
/// functions that allow shaders to evaluate their attributes bound to other
/// map shaders (See: EvalAttribute.h)
///

/// Sample a child map from a parent shader
//  TODO: MOONRAY-3174
//  The timers in this function have been commented out to allow
//  this function to remain inlined for performance reasons, but
//  the TLState and time::RAIIInclusiveExclusiveTicker are not
//  public types.  Work must be done to allow timing by making
//  the necessary types public, or by other means.
finline void
sample(const scene_rdl2::rdl2::Shader *parent, const scene_rdl2::rdl2::Map *map, shading::TLState *tls,
       const State &state, scene_rdl2::math::Color* result)
{
    MNRY_ASSERT(map);

//  TODO: MOONRAY-3174
//  this timing functionality has been temporarily removed until a solution
//  can be found that maintains the current performance of this function due
//  to forcing inlining.
/* #define SHADING_BRACKET_TIMING_ENABLED */
/* #ifdef SHADING_BRACKET_TIMING_ENABLED */
    /* auto threadIndex = tls->mThreadIdx; */
    /* time::RAIIInclusiveExclusiveTicker<int64> ticker( */
            /* MNRY_VERIFY(map->getThreadLocalObjectState())[threadIndex].mShaderCallStat, */
            /* MNRY_VERIFY(parent->getThreadLocalObjectState())[threadIndex].mShaderCallStat); */
/* #endif */

    map->sample(tls, state, result);
}

/// Sample a child normal map from a parent shader
finline void
sampleNormal(const scene_rdl2::rdl2::Shader *parent,
             const scene_rdl2::rdl2::NormalMap *normalMap,
             shading::TLState *tls,
             const State &state, scene_rdl2::math::Vec3f* result)
{
    MNRY_ASSERT(normalMap);
    normalMap->sampleNormal(tls, state, result);
}

/// Transform the shader local aov labels to global aov label ids
void xformLobeLabels(const scene_rdl2::rdl2::Material &material, Bsdf *bsdf, int parentLobeCount);

/// Hack in bsdf label ids used for aovs
void setBsdfLabels(const scene_rdl2::rdl2::Material &material,
                   const State &state, Bsdf *bsdf,
                   int parentLobeCount);

/// Evaluate a root material from the rendering application
void shade(const scene_rdl2::rdl2::Material *material, shading::TLState *tls,
           const State &state, BsdfBuilder &bsdfBuilder);

void shade(const scene_rdl2::rdl2::Material *parent, const scene_rdl2::rdl2::Material *material,
           shading::TLState *tls, const State &state, BsdfBuilder &bsdfBuilder);

/// Evaluate a displacement shader
void displace(const scene_rdl2::rdl2::Displacement *displacement, shading::TLState *tls,
              const State &state, scene_rdl2::math::Vec3f *result);

/// Evaluate presence
float presence(const scene_rdl2::rdl2::Material *material, shading::TLState *tls,
               const State &state);

float presence(const scene_rdl2::rdl2::Material *parent,
               const scene_rdl2::rdl2::Material *material,
               shading::TLState *tls,
               const shading::State &state);

/// Evaluate ior
finline float ior(const scene_rdl2::rdl2::Material *material,
                  shading::TLState *tls,
                  const State &state)
{
    MNRY_ASSERT(material);
    float ior = material->ior(tls, state);

    return ior;
}


/// Evaluate a child material ior from a parent material
finline float ior(const scene_rdl2::rdl2::Material *parent,
                  const scene_rdl2::rdl2::Material *material,
                  shading::TLState *tls,
                  const shading::State &state)
{
    MNRY_ASSERT(parent);
    MNRY_ASSERT(material);

    return material->ior(tls, state);
}

/// Evaluate whether material is preventing light culling
/// Which is necessary when the material is using an input normal that is
/// no longer in the same hemisphere as the geometric normal as part of 
/// certain non-photoreal techniques
finline bool preventLightCulling(const scene_rdl2::rdl2::Material *material,
                                 const State &state)
{
    MNRY_ASSERT(material);
    return material->preventLightCulling(state);
}


/// Evaluate whether a child material preventing light culling from a parent material
/// Which is necessary when the material is using an input normal that is
/// no longer in the same hemisphere as the geometric normal as part of 
/// certain non-photoreal techniques
finline bool preventLightCulling(const scene_rdl2::rdl2::Material *parent,
                                 const scene_rdl2::rdl2::Material *material,
                                 const shading::State &state)
{
    MNRY_ASSERT(parent);
    MNRY_ASSERT(material);

    return material->preventLightCulling(state);
}

//---------------------------------------------------------------------------

} // namespace shading
} // namespace moonray


