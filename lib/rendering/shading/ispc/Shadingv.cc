// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file Shadingv.cc

#include "Shadingv.h"

#include <moonray/rendering/shading/ispc/BsdfLabels.h>

#include <moonray/rendering/shading/bsdf/Bsdfv.h>
#include <moonray/rendering/shading/BsdfBuilder.h>
#include <moonray/rendering/bvh/shading/ShadingTLState.h>

#include <scene_rdl2/common/platform/Platform.h>
#include <scene_rdl2/render/util/Arena.h>
#include <scene_rdl2/scene/rdl2/Displacement.h>
#include <scene_rdl2/scene/rdl2/Geometry.h>
#include <scene_rdl2/scene/rdl2/Map.h>
#include <scene_rdl2/scene/rdl2/NormalMap.h>
#include <scene_rdl2/scene/rdl2/Material.h>

using namespace scene_rdl2;

namespace scene_rdl2 {
    namespace rdl2 {
        class BsdfBuilderv;
    }
}

void
moonray::shading::displacev(const rdl2::Displacement *displacement,
                          moonray::shading::TLState *tls,
                          int numStatev,
                          const moonray::shading::Statev *statev,
                          moonray::shading::Vec3fv *result)
{
    // hook up attribute offset table to evaluate primitive attributes
    // in ispc.
    tls->getAttributeOffsetsFromRootShader(*displacement);

    // safely cast the opaque rdl2 types
    displacement->displacev(tls,
                            numStatev,
                            reinterpret_cast<const rdl2::Statev *>(statev),
                            reinterpret_cast<rdl2::Vec3fv *>(result));

    tls->clearAttributeOffsets();
}

void
moonray::shading::samplev(const rdl2::Map *map,
                        moonray::shading::TLState *tls,
                        const moonray::shading::Statev *statev,
                        moonray::shading::Colorv *result)
{
    EXCL_ACCUMULATOR_PROFILE(tls, EXCL_ACCUM_SHADING);

    // safely cast the opaque rdl2 types
    map->samplev(tls,
                 reinterpret_cast<const rdl2::Statev *>(statev),
                 reinterpret_cast<rdl2::Colorv *>(result));
}

void
moonray::shading::sampleNormalv(const rdl2::NormalMap *normalMap,
                              moonray::shading::TLState *tls,
                              const moonray::shading::Statev *statev,
                              moonray::shading::Colorv *result)
{
    EXCL_ACCUMULATOR_PROFILE(tls, EXCL_ACCUM_SHADING);

    // safely cast the opaque rdl2 types
    normalMap->sampleNormalv(tls,
                             reinterpret_cast<const rdl2::Statev *>(statev),
                             reinterpret_cast<rdl2::Vec3fv *>(result));
}

// helper to allow calling ispc function via func ptr below
typedef void (__cdecl * BsdfBuilderArrayInitFuncv)
    (moonray::shading::TLState *tls,
     unsigned numStatev,
     const moonray::shading::Statev* state,
     rdl2::Bsdfv* bsdfv,
     moonray::shading::BsdfBuilderv* bsdfBuilderv,
     SIMD_MASK_TYPE implicitMask);

void
moonray::shading::shadev(const rdl2::Material *material,
                       moonray::shading::TLState *tls,
                       int numStatev,
                       const moonray::shading::Statev *statev,
                       rdl2::Bsdfv *bsdfv)
{

    // hook up attribute offset table to evaluate primitive attributes
    // in ispc.
    tls->getAttributeOffsetsFromRootShader(*material);

    // safely cast the opaque rdl2 types
    {
        EXCL_ACCUMULATOR_PROFILE(tls, EXCL_ACCUM_SHADING);
        tls->startIspcAccumulator();

        // Our Bsdv array is already allocated and initialized, but for
        // shading each on e needs to be "wrapped" inside a BsdfBuilder.

        // Alloc an array of BsdfBuilderv
        BsdfBuilderv *builderv = tls->mArena->allocArray<BsdfBuilderv>(numStatev, CACHE_LINE_SIZE);

        // Initialize the BsdfBuilderv's (and the Bsdfv's) by calling ispc
        // function via func ptr.  This approach allows casting from c++ to
        // ispc types:
        // * shading::TLState -> ispc::ShadingTLState
        // * shading::State   -> ispc::State
        BsdfBuilderArrayInitFuncv initBsdfBuilderArray = (BsdfBuilderArrayInitFuncv)ispc::getBsdfBuilderInitFunc();
        initBsdfBuilderArray(tls, numStatev, statev, bsdfv, builderv, scene_rdl2::util::sAllOnMask);
        material->shadev(tls,
                         numStatev,
                         reinterpret_cast<const rdl2::Statev *>(statev),
                         reinterpret_cast< rdl2::BsdfBuilderv *>(builderv));
        tls->stopIspcAccumulator();
    }

    // Evaluate and store the post scatter extra aovs on the bsdf object.
    // They will be accumulated after ray scattering.
    {
        EXCL_ACCUMULATOR_PROFILE(tls, EXCL_ACCUM_AOVS);
        const moonray::shading::Material &ext = material->get<moonray::shading::Material>();
        const std::vector<Material::ExtraAov> &extraAovs = ext.getPostScatterExtraAovs();
        if (!extraAovs.empty()) {
            // colorv layout:
            // N = numStatev, M = num extra Aovs
            // bsdf0/aov0,1,2,3...,M-1,bsdf1/aov0,1,2,3..., M-1,...,bsdfN-1,aov0,1,2,3,...,M-1
            math::Colorv *colorv = tls->mArena->allocArray<Colorv>(numStatev * extraAovs.size(), CACHE_LINE_SIZE);
            // labelId layout
            // label ids are the same for each bsdf
            // label0, label1, ... labelM-1
            int *labelIds = tls->mArena->allocArray<int>(extraAovs.size(), CACHE_LINE_SIZE);
            for (size_t i = 0; i < extraAovs.size(); ++i) {
                const Material::ExtraAov &ea = extraAovs[i];
                labelIds[i] = ea.mLabelId;
                for (int j = 0; j < numStatev; ++j) {
                    const size_t coff = j * extraAovs.size() + i;
                    samplev(ea.mMap, tls, statev + j, colorv + coff);
                }
            }
            for (int j = 0; j < numStatev; ++j) {
                Bsdfv_setPostScatterExtraAovs(((moonray::shading::Bsdfv *) bsdfv) + j,
                                              extraAovs.size(),
                                              labelIds,
                                              colorv + j * extraAovs.size());
            }
        }
    }

    tls->clearAttributeOffsets();

    // hack in the bsdf label ids
    {
        EXCL_ACCUMULATOR_PROFILE(tls, EXCL_ACCUM_AOVS);
        internal::setBsdfLabels(material,
                                tls,
                                numStatev,
                                statev,
                                bsdfv,
                                0); // parentLobeCount is always 0 since this is
                                    // called directly from ShadeBundleHandler and
                                    // this is therefore always the top level material
    }
}

