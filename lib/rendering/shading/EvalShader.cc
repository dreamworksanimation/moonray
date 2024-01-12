// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file EvalShader.cc
/// $Id$
///

#include "EvalShader.h"

#include <moonray/rendering/shading/AovLabels.h>
#include <moonray/rendering/shading/bsdf/Bsdf.h>
#include <moonray/rendering/shading/bssrdf/Bssrdf.h>
#include <moonray/rendering/shading/bssrdf/VolumeSubsurface.h>
#include <moonray/rendering/shading/BsdfBuilder.h>
#include <moonray/rendering/shading/Geometry.h>
#include <moonray/rendering/shading/Material.h>

#include <moonray/common/time/Ticker.h>
#include <moonray/rendering/bvh/shading/AttributeKey.h>
#include <moonray/rendering/bvh/shading/Intersection.h>
#include <moonray/rendering/bvh/shading/ShadingTLState.h>
#include <moonray/rendering/bvh/shading/ThreadLocalObjectState.h>

// Enable / disable shader call & timing statistics tracking
#define SHADING_BRACKET_TIMING_ENABLED

namespace moonray {
namespace shading {

/// Transform the shader local aov labels to global aov label ids
void
xformLobeLabels(const scene_rdl2::rdl2::Material &material, Bsdf *bsdf, int parentLobeCount)
{
    MNRY_ASSERT(material.hasExtension());
    if (material.hasExtension()) {
        const auto &ext = material.get<shading::Material>();
        const int  materialLabelId    = ext.getMaterialLabelId();    // material aovs
        const auto &lobeLabelIds      = ext.getLobeLabelIds();       // material aovs
        const int  lpeMaterialLabelId = ext.getLpeMaterialLabelId(); // lpe aovs
        const auto &lpeLobeLabelIds   = ext.getLpeLobeLabelIds();    // lpe aovs

        // Transform the shader local aov labels to global aov label ids.
        // We only transform the lobe labels of the current material.
        // Lobes are added in a top down manner, with parent lobes
        // added before child lobes. Lobe labels, on the other hand,
        // Are transformed in a bottom up manner, with child labels
        // transformed before parent labels.
        // Therefore we must skip the lobes of the parent material here.
        // If this material has a child material, its lobe labels are
        // already transformed. Labels that are already transformed
        // are ignored in aovEncodeLabels.
        for (int i = parentLobeCount; i < bsdf->getLobeCount(); ++i) {
            BsdfLobe *lobe = bsdf->getLobe(i);
            lobe->setLabel(aovEncodeLabels(lobe->getLabel(),
                                           materialLabelId, lpeMaterialLabelId,
                                           lobeLabelIds, lpeLobeLabelIds));
        }

        Bssrdf *bssrdf = bsdf->getBssrdf();
        if (bssrdf) {
            bssrdf->setLabel(aovEncodeLabels(bssrdf->getLabel(),
                                             materialLabelId, lpeMaterialLabelId,
                                             lobeLabelIds, lpeLobeLabelIds));
        }

        VolumeSubsurface *vs = bsdf->getVolumeSubsurface();
        if (vs) {
            vs->setLabel(aovEncodeLabels(vs->getLabel(),
                                         materialLabelId, lpeMaterialLabelId,
                                         lobeLabelIds, lpeLobeLabelIds));
        }
    }
}

/// Hack in bsdf label ids used for aovs
void
setBsdfLabels(const scene_rdl2::rdl2::Material &material, const State &state,
              Bsdf *bsdf, int parentLobeCount)
{
    MNRY_ASSERT(material.hasExtension());
    if (material.hasExtension()) {
        const auto &ext = material.get<shading::Material>();
        const scene_rdl2::rdl2::Geometry *geometry = state.getGeometryObject();
        int geomLabelId = -1;
        MNRY_ASSERT(geometry);
        MNRY_ASSERT(geometry->hasExtension());
        if (geometry && geometry->hasExtension()) {
            geomLabelId = geometry->get<shading::Geometry>().getGeomLabelId();
        }
        bsdf->setLabelIds(ext.getMaterialLabelId(), ext.getLpeMaterialLabelId(),
                          geomLabelId);
    }
    xformLobeLabels(material, bsdf, parentLobeCount);
}

/// Evaluate a root material from the rendering application
//  TODO: MOONRAY-3174
//  Ideally, this function would be called by the PathIntegrator any more because it
//  could not be inlined and caused a shading performance regression.
//
//  This function is currently only called from lib/rendering/pvr/integrator/Picking.cc.
//
//  All non-public  types used here need to be made public or replaced with similar
//  functionality provided by a public API so this function can once again be inlined.
//
//
//  problematic types:
//  -----------------------------------------------------------------------------------------------------
//  * EXCL_ACCUMULATOR_PROFILE, EXCL_ACCUM_SHADING    (rendering/mcrt_common/ProfileAccumulatorHandles.h)
//  * time::RAIITicker<>                              (common/time/Ticker.h)
//  * util::InclusiveExclusiveAverage<>               (common/mcrt_util/Average.h)
void
shade(const scene_rdl2::rdl2::Material *material, shading::TLState *tls,
      const State &state, BsdfBuilder &bsdfBuilder)
{
    MNRY_ASSERT(material);

    EXCL_ACCUMULATOR_PROFILE(tls, EXCL_ACCUM_SHADING);

#ifdef SHADING_BRACKET_TIMING_ENABLED
    auto threadIndex = tls->mThreadIdx;
    time::RAIITicker<util::InclusiveExclusiveAverage<int64> > ticker(
            MNRY_VERIFY(material->getThreadLocalObjectState())[threadIndex].mShaderCallStat);
#endif

    auto bsdf = const_cast<Bsdf*>(bsdfBuilder.getBsdf());
    int parentLobeCount = bsdf->getLobeCount();
    material->shade(tls, state, bsdfBuilder);
    setBsdfLabels(*material, state, bsdf, parentLobeCount);
#ifdef SHADING_PRINT_DEBUG_BSDF_INFO_ENABLED
    bsdf->show(material->getSceneClass().getName(), material->getName(), std::cout);
#endif
}


/// Evaluate a child material from a parent material using BsdfBuilder
//  TODO: MOONRAY-3174
//  Profiling shows that this function would benefit from inlining, but
//  TLState and time::RAIIInclusiveExclusiveTicker are not currentlu
//  public.  There are essentially three places where shade() is called:
//  1) by the integrator  (actually, this does not currently call shade() see below)
//  2) by lib/rendering/pbr/integrator/Picker.cc
//  3) by certain "parent" materials for shading "child" materials
//
//  The current state of affairs is as follows:
//  1) integrator: for performance reasons, shade() is not actually called
//     from the integrator currently (ideally it would be).  Until the functions
//     can all be inlined, a copy of the relevant functionality is built directly
//     into PathIntegrator.cc.
//  2) lib/rendering/pbr/integrator/Picker.cc:   The function above is called,
//     since performance doesn't seem critical
//  3) "parent" materials:  They call this function, although it is likely
//     not as performant as it would be were it inlined.  Efforts should be
//     made to improve the performance of this function by inlining or
//     other means.
void
shade(const scene_rdl2::rdl2::Material *parent, const scene_rdl2::rdl2::Material *material,
      shading::TLState *tls, const State &state, BsdfBuilder& bsdfBuilder)
{
    MNRY_ASSERT(material);

    EXCL_ACCUMULATOR_PROFILE(tls, EXCL_ACCUM_SHADING);

#ifdef SHADING_BRACKET_TIMING_ENABLED
    auto threadIndex = tls->mThreadIdx;
    time::RAIIInclusiveExclusiveTicker<int64> ticker(
            MNRY_VERIFY(material->getThreadLocalObjectState())[threadIndex].mShaderCallStat,
            MNRY_VERIFY(parent->getThreadLocalObjectState())[threadIndex].mShaderCallStat);
#endif

    int parentLobeCount = bsdfBuilder.getBsdf()->getLobeCount();
    material->shade(tls, state, bsdfBuilder);
    setBsdfLabels(*material, state, const_cast<Bsdf*>(bsdfBuilder.getBsdf()), parentLobeCount);
}



/// Evaluate a displacement shader
void
displace(const scene_rdl2::rdl2::Displacement *displacement, shading::TLState *tls,
         const State &state, scene_rdl2::math::Vec3f *result)
{
    MNRY_ASSERT(displacement);

#ifdef SHADING_BRACKET_TIMING_ENABLED
    auto threadIndex = tls->mThreadIdx;
    time::RAIITicker<util::InclusiveExclusiveAverage<int64> > ticker(
            MNRY_VERIFY(displacement->getThreadLocalObjectState())[threadIndex].mShaderCallStat);
#endif

    tls->getAttributeOffsetsFromRootShader(*displacement);
    displacement->displace(tls, state, result);
    tls->clearAttributeOffsets();
}

/// Evaluate presence
float
presence(const scene_rdl2::rdl2::Material *material, shading::TLState *tls,
         const State &state)
{
    MNRY_ASSERT(material);
    return material->presence(tls, state);
}

/// Evaluate a child presence material from a parent material
float
presence(const scene_rdl2::rdl2::Material *parent,
         const scene_rdl2::rdl2::Material *material,
         shading::TLState *tls,
         const shading::State &state)
{
    MNRY_ASSERT(parent);
    MNRY_ASSERT(material);

    return material->presence(tls, state);
}

} // namespace shading
} // namespace moonray

