// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file TestDisplace.cc

#include "TestDisplace.h"

#include <scene_rdl2/render/util/Alloc.h>

#include <moonray/rendering/bvh/shading/ShadingTLState.h>
#include <scene_rdl2/scene/rdl2/rdl2.h>

#include <moonray/rendering/shading/ispc/Shadingv.h>

using namespace scene_rdl2;
using namespace moonray;
using moonray::shading::unittest::TestDisplace;

void
TestDisplace::displace()
{
    mcrt_common::ThreadLocalState *topLevelTls = mcrt_common::getFrameUpdateTLS();
    shading::TLState *tls = MNRY_VERIFY(topLevelTls->mShadingTls.get());
    alloc::Arena *arena = MNRY_VERIFY(tls)->mArena;
    SCOPED_MEM(arena);

    // setup a simple rdl scene consisting of a displacment object with
    // a map binding.
    rdl2::SceneContext ctx;
    ctx.setDsoPath(ctx.getDsoPath() + ":dso/map:dso/displacement");
    rdl2::Map *map =
        ctx.createSceneObject("TestMap", "/map")->asA<rdl2::Map>();
    {
        rdl2::SceneObject::UpdateGuard update(map);
        map->set("mult", .5f);
    }
    rdl2::Displacement *displacement =
        ctx.createSceneObject("TestDisplacement", "/displacement")->
        asA<rdl2::Displacement>();
    {
        rdl2::SceneObject::UpdateGuard update(displacement);
        displacement->set("height", .5f);
        displacement->setBinding("height", map);
    }
    displacement->applyUpdates();

    // process four vector bundles (16 points in SSE).
    const int numBundles = 4;

    // initialize state input.  since we are displacing along the
    // normal, we'll set each normal to (1, 0, 0), (0, 1, 0), (0, 0, 1) etc..
    shading::Statev *statev = arena->allocArray<shading::Statev>(numBundles, SIMD_MEMORY_ALIGNMENT);
    memset(statev, 0, sizeof(shading::Statev) * numBundles);

    for (int i = 0; i < numBundles; ++i) {
        for (int lane = 0; lane < (int)VLEN; ++lane) {

            statev[i].mSt.x[lane] = 1.0;

            int point = i * VLEN + lane;
            statev[i].mN.x[lane] = (point - 0) % 3 == 0? 1.0 : 0.0;
            statev[i].mN.y[lane] = (point - 1) % 3 == 0? 1.0 : 0.0;
            statev[i].mN.z[lane] = (point - 2) % 3 == 0? 1.0 : 0.0;
        }
    }

    // setup storage for displacement output
    Vec3fv *outv = arena->allocArray<Vec3fv>(numBundles, SIMD_MEMORY_ALIGNMENT);

    // displace
    displacev(displacement, tls, numBundles, statev, outv);

    // now verify the results
    // should be (.25, 0, 0), (0, .25, 0), (0, 0, .25) ...
    for (int i = 0; i < numBundles; ++i) {
        for (int lane = 0; lane < (int)VLEN; ++lane) {
            int point = i * VLEN + lane;
            const float x = (point - 0) % 3 == 0? .25 : 0;
            const float y = (point - 1) % 3 == 0? .25 : 0;
            const float z = (point - 2) % 3 == 0? .25 : 0;
            CPPUNIT_ASSERT(outv[i].x[lane] == x);
            CPPUNIT_ASSERT(outv[i].y[lane] == y);
            CPPUNIT_ASSERT(outv[i].z[lane] == z);
        }
    }
}

