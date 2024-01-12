// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file TestBase.cc

#include "TestBase.h"
#include <moonray/rendering/shading/bsdf/Bsdf.h>

#include <moonray/rendering/shading/ispc/Shadingv.h>
#include <moonray/rendering/shading/BsdfBuilder.h>

#include <moonray/rendering/bvh/shading/ShadingTLState.h>

#include <scene_rdl2/render/util/Memory.h>
#include <scene_rdl2/scene/rdl2/rdl2.h>


using namespace moonray;
using moonray::shading::unittest::TestBase;

void
TestBase::shade()
{
    mcrt_common::ThreadLocalState *topLevelTls = mcrt_common::getFrameUpdateTLS();
    shading::TLState *tls = MNRY_VERIFY(topLevelTls->mShadingTls.get());
    scene_rdl2::alloc::Arena *arena = MNRY_VERIFY(tls)->mArena;
    SCOPED_MEM(arena);

    // setup a simple rdl scene consisting of a TestBase material
    scene_rdl2::rdl2::SceneContext ctx;
    ctx.setDsoPath(ctx.getDsoPath() + ":" + RDL2DSO_PATH + ":dso/material");
    scene_rdl2::rdl2::Material *material =
        ctx.createSceneObject("BaseMaterial", "/base")->asA<scene_rdl2::rdl2::Material>();
    material->applyUpdates();

    // all defaults, nothing varying
    shading::Statev *statev = arena->alloc<shading::Statev>(CACHE_LINE_SIZE);
    memset(statev, 0, sizeof(shading::Statev));
    for (unsigned int i = 0; i < VLEN; ++i) {
        statev->mN.x[i] = 1.f; // must have a unit normal
        statev->mNg.x[i] = 1.f;
        statev->mWo.x[i] = 1.f;
        statev->mdPds.y[1] = 1.f;
    }
    Bsdfv *bsdfv = arena->alloc<Bsdfv>();

    // Alloc an array of BsdfBuilderv
    BsdfBuilderv *builderv = tls->mArena->allocArray<BsdfBuilderv>(1, CACHE_LINE_SIZE);

    // helper to allow calling ispc function via func ptr below
    typedef void (__cdecl * BsdfBuilderArrayInitFuncv)
        (shading::TLState *tls,
         unsigned numStatev,
         const shading::Statev* state,
         scene_rdl2::rdl2::Bsdfv* bsdfv,
         shading::BsdfBuilderv* bsdfBuilderv,
         SIMD_MASK_TYPE implicitMask);

    // Initialize the BsdfBuilderv's (and the Bsdfv's) by calling ispc
    // function via func ptr.  This approach allows casting from c++ to
    // ispc types:
    // * shading::TLState -> ispc::ShadingTLState
    // * shading::State   -> ispc::State
    BsdfBuilderArrayInitFuncv initBsdfBuilderArray = (BsdfBuilderArrayInitFuncv)ispc::getBsdfBuilderInitFunc();
    initBsdfBuilderArray(tls, 1, statev, reinterpret_cast<scene_rdl2::rdl2::Bsdfv*>(bsdfv), builderv, scene_rdl2::util::sAllOnMask);

    material->shadev(tls, 1, reinterpret_cast<const scene_rdl2::rdl2::Statev *>(statev), reinterpret_cast<scene_rdl2::rdl2::BsdfBuilderv *>(builderv));
    CPPUNIT_ASSERT(bsdfv->mNumLobes == 2);
    // specular
    CPPUNIT_ASSERT(bsdfv->mLobes[0]->mName == BsdfLobeName::BSDF_LOBE_COOK_TORRANCE);
    int allmask = 0;
    for (unsigned int i = 0; i < VLEN; ++i) {
        allmask |= (1 << i);
    }
    CPPUNIT_ASSERT(bsdfv->mLobes[0]->mMask == allmask);
    CPPUNIT_ASSERT(bsdfv->mLobes[0]->mFresnel != NULL);
    CPPUNIT_ASSERT(bsdfv->mLobes[0]->mFresnel->mType == FresnelType::FRESNEL_TYPE_SCHLICK_FRESNEL);
    const SchlickFresnelv *specFresnel = reinterpret_cast<const SchlickFresnelv *>(bsdfv->mLobes[0]->mFresnel);
    CPPUNIT_ASSERT(specFresnel->mMask == allmask);
    for (unsigned int i = 0; i < VLEN; ++i) {
        CPPUNIT_ASSERT(specFresnel->mSpec.r[i] == 0.1f &&
                       specFresnel->mSpec.g[i] == 0.1f &&
                       specFresnel->mSpec.b[i] == 0.1f);
        CPPUNIT_ASSERT(specFresnel->mFactor[i] == 1.0f);
    }
    // directional diffuse is black
    // translucency is black
    // transmission is black
    // diffuse
    CPPUNIT_ASSERT(bsdfv->mLobes[1]->mName == BsdfLobeName::BSDF_LOBE_LAMBERT);
    CPPUNIT_ASSERT(bsdfv->mLobes[1]->mMask == allmask);
    CPPUNIT_ASSERT(bsdfv->mLobes[1]->mFresnel != NULL);
    CPPUNIT_ASSERT(bsdfv->mLobes[1]->mFresnel->mType ==
                   FresnelType::FRESNEL_TYPE_ONE_MINUS_ROUGH_SCHLICK_FRESNEL);
    const OneMinusRoughSchlickFresnelv *omrsf =
        reinterpret_cast<const OneMinusRoughSchlickFresnelv *>(bsdfv->mLobes[1]->mFresnel);
    CPPUNIT_ASSERT(omrsf->mMask == allmask);
    for (unsigned int i = 0; i < VLEN; ++i) {
        CPPUNIT_ASSERT(omrsf->mSpecRoughness[i] == 0.09f);
    }
    CPPUNIT_ASSERT(omrsf->mTopFresnel == specFresnel);
    // self-emission
    for (unsigned int i = 0; i < VLEN; ++i) {
        CPPUNIT_ASSERT(bsdfv->mSelfEmission.r[i] == 0.0f &&
                       bsdfv->mSelfEmission.g[i] == 0.0f &&
                       bsdfv->mSelfEmission.b[i] == 0.0f);
    }
}

