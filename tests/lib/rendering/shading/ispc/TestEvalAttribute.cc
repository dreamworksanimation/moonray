// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file TestEvalAttribute.cc

#include "TestEvalAttribute.h"

#include <moonray/rendering/shading/ispc/Shadingv.h>

#include <moonray/rendering/bvh/shading/ShadingTLState.h>
#include <moonray/rendering/shading/BsdfBuilder.h>
#include <moonray/rendering/shading/bsdf/Bsdf.h>
#include <moonray/rendering/shading/bsdf/BsdfLambert.h>
#include <moonray/rendering/shading/bsdf/cook_torrance/BsdfCookTorrance.h>
#include <scene_rdl2/render/util/Memory.h>
#include <scene_rdl2/scene/rdl2/rdl2.h>

using namespace scene_rdl2;
using namespace moonray::shading;
using moonray::shading::unittest::TestEvalAttribute;

namespace {

class SceneSetup
{
public:
    SceneSetup();
    Bsdfv *shade();

private:
    scene_rdl2::rdl2::SceneContext mCtx;
    scene_rdl2::rdl2::Material *mMaterial;
    scene_rdl2::rdl2::Map *mMap;
};

SceneSetup::SceneSetup()
{
    mCtx.setDsoPath(mCtx.getDsoPath() + ":dso/map:dso/material");
    mMap =
        mCtx.createSceneObject("TestMap", "/map")->asA<scene_rdl2::rdl2::Map>();
    {
        scene_rdl2::rdl2::SceneObject::UpdateGuard update(mMap);
        mMap->set("mult", 1.0f);
    }
    mMaterial =
        mCtx.createSceneObject("TestMaterial", "/mat")->asA<scene_rdl2::rdl2::Material>();
    {
        scene_rdl2::rdl2::SceneObject::UpdateGuard update(mMaterial);
        mMaterial->set("color non-comp", scene_rdl2::rdl2::Rgb(0, 1, 0));
        mMaterial->setBinding("color non-comp", mMap);

        mMaterial->set("color", true);
        mMaterial->set("color color", scene_rdl2::rdl2::Rgb(1, 0, 0));
        mMaterial->setBinding("color color", mMap);
        mMaterial->set("color factor", .5f);
    }
    mMaterial->applyUpdates();
}

moonray::shading::TLState *
getShadingTLS()
{
    moonray::mcrt_common::ThreadLocalState *tls = moonray::mcrt_common::getFrameUpdateTLS();
    moonray::shading::TLState *shadingTls = MNRY_VERIFY(tls->mShadingTls.get());
    return shadingTls;
}

Bsdfv *
SceneSetup::shade()
{
    moonray::shading::TLState *tls = getShadingTLS();
    alloc::Arena *arena = MNRY_VERIFY(tls)->mArena;

    // mostly 0s with a few notable exceptions in lane0 and lane1
    moonray::shading::Statev *statev = arena->alloc<moonray::shading::Statev>(CACHE_LINE_SIZE);
    memset(statev, 0, sizeof(moonray::shading::Statev));
    statev->mNg.x[0] = 1.0;
    statev->mNg.y[1] = 1.0;
    statev->mN.x[0]  = 1.0;
    statev->mN.y[1]  = 1.0;
    statev->mSt.x[0] = .5;
    statev->mSt.x[1] = .25;
    statev->mWo.x[0] = 1.0;
    statev->mWo.x[1] = 1.0;
    statev->mdPds.y[0] = 1.0; // orthogonal to Ng
    statev->mdPds.x[1] = 1.0; // orthogonal to Ng
    // now assume we have a valid lane0 and lane1, fill out the
    // remaining lanes equal to lane1.  it is an error to
    // pass unitialized lanes into shading.
    for (unsigned i = 2; i < VLEN; ++i) {
        statev->mNg.x[i] = statev->mNg.x[1];
        statev->mN.y[i]  = statev->mN.y[1];
        statev->mSt.x[i] = statev->mSt.x[1];
        statev->mWo.x[i] = statev->mWo.x[1];
        statev->mdPds.y[i] = statev->mdPds.y[1];
    }

    // pass in 1 AOSOA (statevs), return 1 AOSOA (bsdfvs)
    Bsdfv *bsdfv = arena->alloc<Bsdfv>(CACHE_LINE_SIZE);

    // Alloc an array of BsdfBuilderv
    BsdfBuilderv *builderv = tls->mArena->allocArray<BsdfBuilderv>(1, CACHE_LINE_SIZE);

    // helper to allow calling ispc function via func ptr below
    typedef void (__cdecl * BsdfBuilderArrayInitFuncv)
        (moonray::shading::TLState *tls,
         unsigned numStatev,
         const moonray::shading::Statev* state,
         scene_rdl2::rdl2::Bsdfv* bsdfv,
         moonray::shading::BsdfBuilderv* bsdfBuilderv,
         SIMD_MASK_TYPE implicitMask);

    // Initialize the BsdfBuilderv's (and the Bsdfv's) by calling ispc
    // function via func ptr.  This approach allows casting from c++ to
    // ispc types:
    // * moonray::shading::TLState -> ispc::ShadingTLState
    // * moonray::shading::State   -> ispc::State
    BsdfBuilderArrayInitFuncv initBsdfBuilderArray = (BsdfBuilderArrayInitFuncv)ispc::getBsdfBuilderInitFunc();
    initBsdfBuilderArray(tls, 1, statev, reinterpret_cast<scene_rdl2::rdl2::Bsdfv*>(bsdfv), builderv, scene_rdl2::util::sAllOnMask);

    mMaterial->shadev(tls, 1, reinterpret_cast<const scene_rdl2::rdl2::Statev *>(statev), reinterpret_cast<scene_rdl2::rdl2::BsdfBuilderv *>(builderv));
    CPPUNIT_ASSERT(bsdfv->mNumLobes == 2); // evalColor and
                                               // evalColorComponent
    return bsdfv;
}

} // namespace

void
TestEvalAttribute::testEvalColor()
{
    SceneSetup scene;
    Bsdfv &bsdf = *(scene.shade());

    // evalColor test results are in lobe0
    CPPUNIT_ASSERT(bsdf.mLobes[0]->mName == BsdfLobeName::BSDF_LOBE_LAMBERT);
    LambertBsdfLobev &lobe = *(LambertBsdfLobev *) bsdf.mLobes[0];

    // lane0
    CPPUNIT_ASSERT(lobe.mScale.r[0] == 0.0 &&
                   lobe.mScale.g[0] == 0.5 &&
                   lobe.mScale.b[0] == 0.0);
    CPPUNIT_ASSERT(lobe.mFrame.mZ.x[0] == 1.0 &&
                   lobe.mFrame.mZ.y[0] == 0.0 &&
                   lobe.mFrame.mZ.z[0] == 0.0);

    // lane1
    CPPUNIT_ASSERT(lobe.mScale.r[1] == 0.0  &&
                   lobe.mScale.g[1] == 0.25 &&
                   lobe.mScale.b[1] == 0.0);
    CPPUNIT_ASSERT(lobe.mFrame.mZ.x[1] == 0.0 &&
                   lobe.mFrame.mZ.y[1] == 1.0 &&
                   lobe.mFrame.mZ.z[1] == 0.0);
}

void
TestEvalAttribute::testEvalColorComponent()
{
    SceneSetup scene;
    Bsdfv &bsdf = *(scene.shade());

    // evalColorComponent results are in lobe1
    CPPUNIT_ASSERT(bsdf.mLobes[1]->mName == BsdfLobeName::BSDF_LOBE_COOK_TORRANCE);
    CookTorranceBsdfLobev &lobe = *(CookTorranceBsdfLobev *) bsdf.mLobes[1];

    // When testing mFrame we use isEqual() due to minor error
    // introduced in adaptNormal() during lobe creation

    const float tol = 0.006f;// things have gotten sloppier when moving to ispc-1.14.1

    // lane0
    CPPUNIT_ASSERT(lobe.mScale.r[0] == 0.25 &&
                   lobe.mScale.g[0] == 0.0  &&
                   lobe.mScale.b[0] == 0.0);
    CPPUNIT_ASSERT(scene_rdl2::math::isEqual(lobe.mFrame.mZ.x[0], 1.f, tol) &&
                   scene_rdl2::math::isEqual(lobe.mFrame.mZ.y[0], 0.f, tol) &&
                   scene_rdl2::math::isEqual(lobe.mFrame.mZ.z[0], 0.f, tol));
    CPPUNIT_ASSERT(lobe.mRoughness[0] == 1.0);

    // lane1
    CPPUNIT_ASSERT(lobe.mScale.r[1] == 0.125 &&
                   lobe.mScale.g[1] == 0.0   &&
                   lobe.mScale.b[1] == 0.0);
    CPPUNIT_ASSERT(scene_rdl2::math::isEqual(lobe.mFrame.mZ.x[1], 0.f, tol) &&
                   scene_rdl2::math::isEqual(lobe.mFrame.mZ.y[1], 1.f, tol) &&
                   scene_rdl2::math::isEqual(lobe.mFrame.mZ.z[1], 0.f, tol));
    CPPUNIT_ASSERT(lobe.mRoughness[1] == 1.0);
}

void
TestEvalAttribute::testGetBool()
{
    moonray::shading::TLState *tls = getShadingTLS();

    scene_rdl2::rdl2::SceneContext ctx;
    ctx.setDsoPath(ctx.getDsoPath() + ":dso/map");
    scene_rdl2::rdl2::Map *map = ctx.createSceneObject("TestMap", "/map")->asA<scene_rdl2::rdl2::Map>();
    {
        scene_rdl2::rdl2::SceneObject::UpdateGuard update(map);
        map->set("mode", 2);
        map->set("bool1", true);
    }
    map->applyUpdates();
    {
        moonray::shading::Statev statev;
        moonray::shading::Colorv result;
        moonray::shading::samplev(map, tls, &statev, &result);
        for (int i = 0; i < (int) VLEN; ++i) {
            CPPUNIT_ASSERT(result.r[i] == 1.0);
            CPPUNIT_ASSERT(result.g[i] == 1.0);
            CPPUNIT_ASSERT(result.b[i] == 1.0);
        }
    }
    ctx.resetUpdates(nullptr);
    {
        scene_rdl2::rdl2::SceneObject::UpdateGuard update(map);
        map->set("bool1", false);
    }
    map->applyUpdates();
    {
        moonray::shading::Statev statev;
        moonray::shading::Colorv result;
        moonray::shading::samplev(map, tls, &statev, &result);
        for (int i = 0; i < (int) VLEN; ++i) {
            CPPUNIT_ASSERT(result.r[i] == 0.0);
            CPPUNIT_ASSERT(result.g[i] == 0.0);
            CPPUNIT_ASSERT(result.b[i] == 0.0);
        }
    }
}

void
TestEvalAttribute::testGetColor()
{
    moonray::shading::TLState *tls = getShadingTLS();

    scene_rdl2::rdl2::SceneContext ctx;
    ctx.setDsoPath(ctx.getDsoPath() + ":dso/map");
    scene_rdl2::rdl2::Map *map = ctx.createSceneObject("TestMap", "/map")->asA<scene_rdl2::rdl2::Map>();
    {
        scene_rdl2::rdl2::SceneObject::UpdateGuard update(map);
        map->set("mode", 1);
        map->set("color1", scene_rdl2::rdl2::Rgb(0.25, 0.5, 0.75));
    }
    map->applyUpdates();
    {
        moonray::shading::Statev statev;
        moonray::shading::Colorv result;
        moonray::shading::samplev(map, tls, &statev, &result);
        for (int i = 0; i < (int) VLEN; ++i) {
            CPPUNIT_ASSERT(result.r[i] == 0.25);
            CPPUNIT_ASSERT(result.g[i] == 0.5);
            CPPUNIT_ASSERT(result.b[i] == 0.75);
        }
    }
}

void
TestEvalAttribute::testGetInt()
{
    moonray::shading::TLState *tls = getShadingTLS();

    scene_rdl2::rdl2::SceneContext ctx;
    ctx.setDsoPath(ctx.getDsoPath() + ":dso/map");
    scene_rdl2::rdl2::Map *map = ctx.createSceneObject("TestMap", "/map")->asA<scene_rdl2::rdl2::Map>();
    {
        scene_rdl2::rdl2::SceneObject::UpdateGuard update(map);
        map->set("mode", 3);
        map->set("int1", 3);
    }
    map->applyUpdates();
    {
        moonray::shading::Statev statev;
        moonray::shading::Colorv result;
        moonray::shading::samplev(map, tls, &statev, &result);
        for (int i = 0; i < (int) VLEN; ++i) {
            CPPUNIT_ASSERT(result.r[i] == 3.0);
            CPPUNIT_ASSERT(result.g[i] == 3.0);
            CPPUNIT_ASSERT(result.b[i] == 3.0);
        }
    }
}

void
TestEvalAttribute::testGetFloat()
{
    moonray::shading::TLState *tls = getShadingTLS();

    scene_rdl2::rdl2::SceneContext ctx;
    ctx.setDsoPath(ctx.getDsoPath() + ":dso/map");
    scene_rdl2::rdl2::Map *map = ctx.createSceneObject("TestMap", "/map")->asA<scene_rdl2::rdl2::Map>();
    {
        scene_rdl2::rdl2::SceneObject::UpdateGuard update(map);
        map->set("mode", 6);
        map->set("float1", 2.f);
    }
    map->applyUpdates();
    {
        moonray::shading::Statev statev;
        moonray::shading::Colorv result;
        moonray::shading::samplev(map, tls, &statev, &result);
        for (int i = 0; i < (int) VLEN; ++i) {
            CPPUNIT_ASSERT(result.r[i] == 2.0);
            CPPUNIT_ASSERT(result.g[i] == 2.0);
            CPPUNIT_ASSERT(result.b[i] == 2.0);
        }
    }
}

void
TestEvalAttribute::testGetVec2f()
{
    moonray::shading::TLState *tls = getShadingTLS();

    scene_rdl2::rdl2::SceneContext ctx;
    ctx.setDsoPath(ctx.getDsoPath() + ":dso/map");
    scene_rdl2::rdl2::Map *map = ctx.createSceneObject("TestMap", "/map")->asA<scene_rdl2::rdl2::Map>();
    {
        scene_rdl2::rdl2::SceneObject::UpdateGuard update(map);
        map->set("mode", 4);
        map->set("vec21", scene_rdl2::rdl2::Vec2f(1.0, .5));
    }
    map->applyUpdates();
    {
        moonray::shading::Statev statev;
        moonray::shading::Colorv result;
        moonray::shading::samplev(map, tls, &statev, &result);
        for (int i = 0; i < (int) VLEN; ++i) {
            CPPUNIT_ASSERT(result.r[i] == 1.0);
            CPPUNIT_ASSERT(result.g[i] == 0.5);
            CPPUNIT_ASSERT(result.b[i] == 0.0);
        }
    }
}

void
TestEvalAttribute::testGetVec3f()
{
    moonray::shading::TLState *tls = getShadingTLS();

    scene_rdl2::rdl2::SceneContext ctx;
    ctx.setDsoPath(ctx.getDsoPath() + ":dso/map");
    scene_rdl2::rdl2::Map *map = ctx.createSceneObject("TestMap", "/map")->asA<scene_rdl2::rdl2::Map>();
    {
        scene_rdl2::rdl2::SceneObject::UpdateGuard update(map);
        map->set("mode", 5);
        map->set("vec31", scene_rdl2::rdl2::Vec3f(1.0, .5, .5));
    }
    map->applyUpdates();
    {
        moonray::shading::Statev statev;
        moonray::shading::Colorv result;
        moonray::shading::samplev(map, tls, &statev, &result);
        for (int i = 0; i < (int) VLEN; ++i) {
            CPPUNIT_ASSERT(result.r[i] == 1.0);
            CPPUNIT_ASSERT(result.g[i] == 0.5);
            CPPUNIT_ASSERT(result.b[i] == 0.5);
        }
    }
}

void
TestEvalAttribute::testEvalAttrFloat()
{
    moonray::shading::TLState *tls = getShadingTLS();

    scene_rdl2::rdl2::SceneContext ctx;
    ctx.setDsoPath(ctx.getDsoPath() + ":dso/map");
    scene_rdl2::rdl2::Map *map = ctx.createSceneObject("TestMap", "/map")->asA<scene_rdl2::rdl2::Map>();
    {
        scene_rdl2::rdl2::SceneObject::UpdateGuard update(map);
        map->set("mode", 8);
        map->set("float2", .5f);
    }
    scene_rdl2::rdl2::Map *map2 = ctx.createSceneObject("TestMap", "/map2")->asA<scene_rdl2::rdl2::Map>();
    {
        scene_rdl2::rdl2::SceneObject::UpdateGuard update(map2);
        map2->set("mode", 8);
        map2->set("float2", .5f);
        map2->setBinding("float2", map);
    }
    map2->applyUpdates();
    {
        moonray::shading::Statev statev;
        moonray::shading::Colorv result;
        moonray::shading::samplev(map2, tls, &statev, &result);
        for (int i = 0; i < (int) VLEN; ++i) {
            CPPUNIT_ASSERT(result.r[i] == 0.25);
            CPPUNIT_ASSERT(result.g[i] == 0.25);
            CPPUNIT_ASSERT(result.b[i] == 0.25);
        }
    }
}

void
TestEvalAttribute::testEvalAttrColor()
{
    moonray::shading::TLState *tls = getShadingTLS();

    scene_rdl2::rdl2::SceneContext ctx;
    ctx.setDsoPath(ctx.getDsoPath() + ":dso/map");
    scene_rdl2::rdl2::Map *map = ctx.createSceneObject("TestMap", "/map")->asA<scene_rdl2::rdl2::Map>();
    {
        scene_rdl2::rdl2::SceneObject::UpdateGuard update(map);
        map->set("mode", 7);
        map->set("color2", scene_rdl2::rdl2::Rgb(1.f, .5f, .25f));
    }
    scene_rdl2::rdl2::Map *map2 = ctx.createSceneObject("TestMap", "/map2")->asA<scene_rdl2::rdl2::Map>();
    {
        scene_rdl2::rdl2::SceneObject::UpdateGuard update(map2);
        map2->set("mode", 7);
        map2->set("color2", scene_rdl2::rdl2::Rgb(.5f, .5f, .5f));
        map2->setBinding("color2", map);
    }
    map2->applyUpdates();
    {
        moonray::shading::Statev statev;
        moonray::shading::Colorv result;
        moonray::shading::samplev(map2, tls, &statev, &result);
        for (int i = 0; i < (int) VLEN; ++i) {
            CPPUNIT_ASSERT(result.r[i] == 0.5f);
            CPPUNIT_ASSERT(result.g[i] == 0.25f);
            CPPUNIT_ASSERT(result.b[i] == 0.125f);
        }
    }
}

void
TestEvalAttribute::testEvalNormal()
{
    moonray::shading::TLState *tls = getShadingTLS();

    scene_rdl2::rdl2::SceneContext ctx;
    ctx.setDsoPath(ctx.getDsoPath() + ":dso/map");
    scene_rdl2::rdl2::Map *map = ctx.createSceneObject("TestMap", "/map")->asA<scene_rdl2::rdl2::Map>();
    {
        scene_rdl2::rdl2::SceneObject::UpdateGuard update(map);
        map->set("mode", 10);
        map->set("vec3f10", scene_rdl2::rdl2::Vec3f(1.0f, 2.0f, 2.0f));
    }
    scene_rdl2::rdl2::Map *map2 = ctx.createSceneObject("TestMap", "/map2")->asA<scene_rdl2::rdl2::Map>();
    {
        scene_rdl2::rdl2::SceneObject::UpdateGuard update(map2);
        map2->set("mode", 11);
        map2->set("float11", 0.5f);
        map2->setBinding("vec3f11", map);
    }
    map2->applyUpdates();
    {
        moonray::shading::Statev statev;
        const scene_rdl2::math::Vec3f wo = scene_rdl2::math::normalize(scene_rdl2::math::Vec3f(0.f, 1.f, 1.f));
        for (int i = 0; i < (int) VLEN; ++i) {
            statev.mN.x[i] = 0.0f;
            statev.mN.y[i] = 1.0f;
            statev.mN.z[i] = 0.0f;
            statev.mdPds.x[i] = 0.0f;
            statev.mdPds.y[i] = 0.0f;
            statev.mdPds.z[i] = -1.0f;

            // normal mapping now requires a view direction
            statev.mWo.x[i] = wo.x;
            statev.mWo.y[i] = wo.y;
            statev.mWo.z[i] = wo.z;

            // and a geometric normal
            statev.mNg.x[i] = statev.mN.x[i];
            statev.mNg.y[i] = statev.mN.y[i];
            statev.mNg.z[i] = statev.mN.z[i];
        }
        moonray::shading::Colorv result;
        moonray::shading::samplev(map2, tls, &statev, &result);
        for (int i = 0; i < (int) VLEN; ++i) {
            CPPUNIT_ASSERT(scene_rdl2::math::isEqual(result.r[i], -0.374551f));
            CPPUNIT_ASSERT(scene_rdl2::math::isEqual(result.g[i], 0.918762f));
            CPPUNIT_ASSERT(scene_rdl2::math::isEqual(result.b[i], -0.12485f));
        }
    }
}

void
TestEvalAttribute::testEvalAttrVec3f()
{
    moonray::shading::TLState *tls = getShadingTLS();

    scene_rdl2::rdl2::SceneContext ctx;
    ctx.setDsoPath(ctx.getDsoPath() + ":dso/map");
    scene_rdl2::rdl2::Map *map = ctx.createSceneObject("TestMap", "/map")->asA<scene_rdl2::rdl2::Map>();
    {
        scene_rdl2::rdl2::SceneObject::UpdateGuard update(map);
        map->set("mode", 9);
        map->set("vec32", scene_rdl2::rdl2::Vec3f(1.f, .5f, .25f));
    }
    scene_rdl2::rdl2::Map *map2 = ctx.createSceneObject("TestMap", "/map2")->asA<scene_rdl2::rdl2::Map>();
    {
        scene_rdl2::rdl2::SceneObject::UpdateGuard update(map2);
        map2->set("mode", 9);
        map2->set("vec32", scene_rdl2::rdl2::Vec3f(.5f, .5f, .5f));
        map2->setBinding("vec32", map);
    }
    map2->applyUpdates();
    {
        moonray::shading::Statev statev;
        moonray::shading::Colorv result;
        moonray::shading::samplev(map2, tls, &statev, &result);
        for (int i = 0; i < (int) VLEN; ++i) {
            CPPUNIT_ASSERT(result.r[i] == 1.0f);
            CPPUNIT_ASSERT(result.g[i] == 0.5f);
            CPPUNIT_ASSERT(result.b[i] == 0.25f);
        }
    }
}

void
TestEvalAttribute::testEvalAttrVec2f()
{
    moonray::shading::TLState *tls = getShadingTLS();

    scene_rdl2::rdl2::SceneContext ctx;
    ctx.setDsoPath(ctx.getDsoPath() + ":dso/map");
    scene_rdl2::rdl2::Map *map = ctx.createSceneObject("TestMap", "/map")->asA<scene_rdl2::rdl2::Map>();
    {
        scene_rdl2::rdl2::SceneObject::UpdateGuard update(map);
        map->set("mode", 12);
        map->set("vec22", scene_rdl2::rdl2::Vec2f(.5f, .25f));
    }
    scene_rdl2::rdl2::Map *map2 = ctx.createSceneObject("TestMap", "/map2")->asA<scene_rdl2::rdl2::Map>();
    {
        scene_rdl2::rdl2::SceneObject::UpdateGuard update(map2);
        map2->set("mode", 12);
        map2->set("vec22", scene_rdl2::rdl2::Vec2f(.5f, .5f));
        map2->setBinding("vec22", map);
    }
    map2->applyUpdates();
    {
        moonray::shading::Statev statev;
        moonray::shading::Colorv result;
        moonray::shading::samplev(map2, tls, &statev, &result);
        for (int i = 0; i < (int) VLEN; ++i) {
            CPPUNIT_ASSERT(result.r[i] == 0.25f);
            CPPUNIT_ASSERT(result.g[i] == 0.125f);
        }
    }
}

