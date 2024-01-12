// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file EvalShader.h

#pragma once

#include <moonray/rendering/shading/ispc/Shadingv.h>

#include <moonray/rendering/shading/AovLabels.h>
#include <moonray/rendering/shading/bsdf/Bsdf.h>
#include <moonray/rendering/shading/bssrdf/Bssrdf.h>
#include <moonray/rendering/shading/bssrdf/VolumeSubsurface.h>
#include <moonray/rendering/shading/Geometry.h>
#include <moonray/rendering/shading/Material.h>

namespace moonray {
namespace shading {
namespace internal {

finline void
setBsdfLabels(const scene_rdl2::rdl2::Material *material,
              shading::TLState *tls,
              int numStatev,
              const shading::Statev *statev,
              scene_rdl2::rdl2::Bsdfv *bsdfv,
              const int parentLobeCount)
{
    // The Bsdf object holds several different bits of label data used by Aovs.
    // This data must be set or transformed after shading, prior to integration.
    //
    // --
    // BsdfLobe::mLabel
    // --
    // The shader can place labels on the lobes it outputs.  These labels,
    // stored in mLabel, are simple integers that are only unique to the
    // shader itself.  For example, two different shaders will use
    // mLabel  = "1", "2", "3" etc... to mean completely different labels.
    // Within the shader's extension object is a mapping from the local label
    // ids of the shader to the global label ids used by the aov system.
    // There is a separate mapping for material aovs (lobeLabelIds)
    // and light path expression aovs (lpeLobeLabelIds).  This code transforms
    // BsdfLobe::mLabel into a value that encodes the global lobeLabelId, and
    // lpeLobeLabelId for the lobe.
    //
    // It is safe to call this function multiple times on the same Bsdf.  Once
    // lobes are transformed, the transform operation is a no-op.
    //
    // The BsdfLobe::mLabel is a uniform member, so no per-lane processing
    // is needed.
    //
    // --
    // Bsdf::mGeomLabelId
    // --
    // Geometry object's have a user visible "label" attribute.  This label
    // can appear in material aov expressions.  The geometry extension object
    // contains the integer mapping this label to the global material aov system.
    //
    // Since the geometry object can vary (but probably often doesn't) per lane,
    // the geometry id must be explicitly set for each lane.
    //
    // It is safe to set this data multiple times from the same Bsdf as long
    // as the Intersection remains invariant (i.e. the geometry id per lane does not
    // change).


    if (material->hasExtension()) {
        const shading::Material &ext = material->get<shading::Material>();
        Bsdfv *shadingBsdfv = reinterpret_cast<Bsdfv *>(bsdfv);
        for (int i = 0; i < numStatev; ++i) {
            const int  materialLabelId    = ext.getMaterialLabelId();    // material aovs
            const auto &lobeLabelIds      = ext.getLobeLabelIds();       // material aovs
            const int  lpeMaterialLabelId = ext.getLpeMaterialLabelId(); // lpe aovs
            const auto &lpeLobeLabelIds   = ext.getLpeLobeLabelIds();    // lpe aovs

            // uniform labels - we have only a single material per bsdf
            shadingBsdfv[i].mMaterialLabelId = materialLabelId;
            shadingBsdfv[i].mLpeMaterialLabelId = lpeMaterialLabelId;

            // potentially have a different geometry per-lane
            // multiple geometry can potentially share the same material
            const mcrt_common::ConstAddress64v &geometryObject =
                reinterpret_cast<const mcrt_common::ConstAddress64v &>(statev[i].mGeometryObject);
            for (unsigned int lane = 0; lane < VLEN; ++lane) {
                const scene_rdl2::rdl2::Geometry *geometry =
                    static_cast<const scene_rdl2::rdl2::Geometry *>(geometryObject.get(lane));
                if (geometry && geometry->hasExtension()) {
                    shadingBsdfv[i].mGeomLabelId[lane] =
                        geometry->get<shading::Geometry>().getGeomLabelId();
                }
            }

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
            for (int li = parentLobeCount; li < shadingBsdfv[i].mNumLobes; ++li) {
                BsdfLobev *lobev = shadingBsdfv[i].mLobes[li];
                lobev->mLabel = aovEncodeLabels(lobev->mLabel,
                                                materialLabelId, lpeMaterialLabelId,
                                                lobeLabelIds, lpeLobeLabelIds);
            }
            Bssrdfv *bssrdfv = shadingBsdfv[i].mBssrdf;
            if (bssrdfv) {
                bssrdfv->mLabel = aovEncodeLabels(bssrdfv->mLabel,
                                                  materialLabelId, lpeMaterialLabelId,
                                                  lobeLabelIds, lpeLobeLabelIds);
            }
            VolumeSubsurfacev *vsv = shadingBsdfv[i].mVolumeSubsurface;
            if (vsv) {
                vsv->mLabel = aovEncodeLabels(vsv->mLabel,
                                                   materialLabelId, lpeMaterialLabelId,
                                                   lobeLabelIds, lpeLobeLabelIds);
            }
        }
    }
}

} // namespace internal
} // namespace shading
} // namespace moonray


