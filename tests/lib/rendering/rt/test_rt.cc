// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#include "test_rt.h"
#include <moonray/rendering/rt/EmbreeAccelerator.h>
#include <moonray/rendering/rt/GeomContext.h>
#include <moonray/rendering/geom/PrimitiveGroup.h>
#include <moonray/rendering/geom/PrimitiveVisitor.h>
#include <moonray/rendering/geom/Procedural.h>
#include <moonray/rendering/geom/ProceduralLeaf.h>
#include <moonray/rendering/geom/prim/Instance.h>
#include <moonray/rendering/geom/prim/PrimitivePrivateAccess.h>
#include <moonray/rendering/shading/Shading.h>

#include <scene_rdl2/scene/rdl2/rdl2.h>

#include <tbb/tick_count.h>

CPPUNIT_TEST_SUITE_REGISTRATION(TestRenderingRT);

using namespace moonray;
using namespace moonray::rt;

void TestRenderingRT::testRay()
{
    scene_rdl2::math::Vec3f org(1, 2, 3);
    scene_rdl2::math::Vec3f dir(4, 5, 6);
    scene_rdl2::math::Vec3f  ng(7, 8, 9);

    mcrt_common::Ray ray;
    ray.org = org;
    ray.dir = dir;
    ray.setNg(ng);
    mcrt_common::Ray ray2(org, dir, 777, 888, 0.234, 0xDEADBEEF);

    CPPUNIT_ASSERT(ray.getOrigin() == org);
    CPPUNIT_ASSERT(ray.getDirection() == dir);
    CPPUNIT_ASSERT(ray.getNg() == ng);
    CPPUNIT_ASSERT(ray.getOrigin() == ray2.getOrigin());
    CPPUNIT_ASSERT(ray.getDirection() == ray2.getDirection());
}


void TestRenderingRT::testIntersectPolygon()
{
    // setup a simple rdl scene consisting of a geometry object to
    // generate primitives for intersection test
    scene_rdl2::rdl2::SceneContext ctx;
    ctx.setDsoPath(ctx.getDsoPath() +
        ":dso/geometry/TestRtGeometry:dso/material/TestRtMaterial");
    scene_rdl2::rdl2::Geometry* geom =
        ctx.createSceneObject("TestRtGeometry", "geom")->asA<scene_rdl2::rdl2::Geometry>();
    scene_rdl2::rdl2::Layer* layer =
        ctx.createSceneObject("Layer", "/seq/shot/layer")->asA<scene_rdl2::rdl2::Layer>();
    scene_rdl2::rdl2::Material* mtl =
        ctx.createSceneObject("TestRtMaterial", "mtl")->asA<scene_rdl2::rdl2::Material>();
    scene_rdl2::rdl2::LightSet* lgt =
        ctx.createSceneObject("LightSet", "lgt")->asA<scene_rdl2::rdl2::LightSet>();
    layer->beginUpdate();
    layer->assign(geom, "", mtl, lgt);
    layer->endUpdate();
    layer->applyUpdates();

    geom->beginUpdate();
    // polygon test case
    geom->set("test mode", 0);
    geom->endUpdate();
    geom->applyUpdates();

    geom->loadProcedural();
    GeomGenerateContext generateContext(nullptr, geom, moonray::shading::AttributeKeySet(),
        0, 1, moonray::geom::MotionBlurParams({0.f}, 0.f, 0.f, false, 24.f));
    scene_rdl2::math::Xform3f p2r(scene_rdl2::math::one);
    geom->getProcedural()->generate(generateContext, {p2r});

    scene_rdl2::rdl2::GeometrySet* geomSet =
        ctx.createSceneObject("GeometrySet", "geomSet")->asA<scene_rdl2::rdl2::GeometrySet>();
    geomSet->beginUpdate();
    geomSet->add(geom);
    geomSet->endUpdate();
    geomSet->applyUpdates();

    // setup the BVH
    AcceleratorOptions options;
    EmbreeAccelerator bvh(options);
    bvh.build(OptimizationTarget::FAST_BVH_BUILD, ChangeFlag::ALL, layer, {geomSet});

    // intersect test agains a quad (for detail see TestRtGeometry)
    // (-1, 1, -1) ------------ (1, 1, -1)
    //            |            |
    //            |            |
    //            |            |
    //            |            |
    //            |            |
    // (-1,-1, -1) ------------ (1,-1, -1)
    //
    mcrt_common::Ray ray1;
    ray1.org = scene_rdl2::math::Vec3f(0.5f, 0.5f,  0.0f);
    ray1.dir = scene_rdl2::math::Vec3f(0.0f, 0.0f, -1.0f);
    bool correctRay1 = bvh.occluded(ray1);

    mcrt_common::Ray ray2;
    ray2.org = scene_rdl2::math::Vec3f(1.5f, 1.5f,  0.0f);
    ray2.dir = scene_rdl2::math::Vec3f(0.0f, 0.0f, -1.0f);
    bool correctRay2 = !bvh.occluded(ray2);

    mcrt_common::Ray ray3;
    ray3.org = scene_rdl2::math::Vec3f(-0.5f, -0.5f,  0.0f);
    ray3.dir = scene_rdl2::math::Vec3f( 0.0f,  0.0f, -1.0f);
    bvh.intersect(ray3);
    bool correctRay3 = ray3.geomID != RTC_INVALID_GEOMETRY_ID &&
        scene_rdl2::math::isEqual(ray3.tfar, 1.0f);

    mcrt_common::Ray ray4;
    ray4.org = scene_rdl2::math::Vec3f(-1.5f, -1.5f,  0.0f);
    ray4.dir = scene_rdl2::math::Vec3f( 0.0f,  0.0f, -1.0f);
    bvh.intersect(ray4);
    bool correctRay4 = ray4.geomID == RTC_INVALID_GEOMETRY_ID &&
        scene_rdl2::math::isEqual(ray4.tfar, FLT_MAX);

    // update the scene and BVH

    GeomUpdateContext updateContext(nullptr, geom,
        0, 1, moonray::geom::MotionBlurParams({0.f}, 0.f, 0.f, false, 24.f));
    geom->getProcedural()->update(updateContext, {p2r});
    bvh.build(OptimizationTarget::FAST_BVH_BUILD, ChangeFlag::UPDATE,
        layer, {geomSet});
    // quad is moved during update call (for detail see TestRtGeometry)
    // its current position should be
    // ( 2,-2, -1) ------------ (4,-2, -1)
    //            |            |
    //            |            |
    //            |            |
    //            |            |
    //            |            |
    // ( 2,-4, -1) ------------ (4,-4, -1)
    //
    mcrt_common::Ray ray5;
    ray5.org = scene_rdl2::math::Vec3f(2.5f,-2.5f,  0.0f);
    ray5.dir = scene_rdl2::math::Vec3f(0.0f, 0.0f, -1.0f);
    bool correctRay5 = bvh.occluded(ray5);

    mcrt_common::Ray ray6;
    ray6.org = scene_rdl2::math::Vec3f(0.5f, 0.5f,  0.0f);
    ray6.dir = scene_rdl2::math::Vec3f(0.0f, 0.0f, -1.0f);
    bool correctRay6 = !bvh.occluded(ray6);

    mcrt_common::Ray ray7;
    ray7.org = scene_rdl2::math::Vec3f(3.5f, -3.5f,  0.0f);
    ray7.dir = scene_rdl2::math::Vec3f(0.0f,  0.0f, -1.0f);
    bvh.intersect(ray7);
    bool correctRay7 = ray7.geomID != RTC_INVALID_GEOMETRY_ID &&
        scene_rdl2::math::isEqual(ray7.tfar, 1.0f);

    mcrt_common::Ray ray8;
    ray8.org = scene_rdl2::math::Vec3f(-0.5f, -0.5f,  0.0f);
    ray8.dir = scene_rdl2::math::Vec3f( 0.0f,  0.0f, -1.0f);
    bvh.intersect(ray8);
    bool correctRay8 = ray8.geomID == RTC_INVALID_GEOMETRY_ID &&
        scene_rdl2::math::isEqual(ray8.tfar, FLT_MAX);

    geom->getProcedural()->clear();

    CPPUNIT_ASSERT(correctRay1);
    CPPUNIT_ASSERT(correctRay2);
    CPPUNIT_ASSERT(correctRay3);
    CPPUNIT_ASSERT(correctRay4);
    CPPUNIT_ASSERT(correctRay5);
    CPPUNIT_ASSERT(correctRay6);
    CPPUNIT_ASSERT(correctRay7);
    CPPUNIT_ASSERT(correctRay8);
}

class XformInitializer : public moonray::geom::PrimitiveVisitor
{
public:
    XformInitializer() = default;

    virtual void visitPrimitiveGroup(moonray::geom::PrimitiveGroup& pg) override
    {
        pg.forEachPrimitive(*this);
    }

    virtual void visitTransformedPrimitive(moonray::geom::TransformedPrimitive& t) override
    {
        t.getPrimitive()->accept(*this);
    }

    virtual void visitInstance(moonray::geom::Instance& i) override
    {
        moonray::geom::internal::Instance* pInstance =
            static_cast<moonray::geom::internal::Instance*>(
                moonray::geom::internal::PrimitivePrivateAccess::getPrimitiveImpl(&i));
        pInstance->initializeXform();
        const std::shared_ptr<moonray::geom::SharedPrimitive>& ref = i.getReference();
        // visit the referenced Primitive if it's not visited yet
        if (mSharedPrimitives.insert(ref).second) {
            XformInitializer xi;
            ref->getPrimitive()->accept(xi);
        }
    }

private:
    moonray::geom::SharedPrimitiveSet mSharedPrimitives;
};

void TestRenderingRT::testIntersectInstances()
{
    // setup a simple rdl scene consisting of a geometry object to
    // generate primitives for intersection test
    scene_rdl2::rdl2::SceneContext ctx;
    ctx.setDsoPath(ctx.getDsoPath() +
        ":dso/geometry/TestRtGeometry:dso/material/TestRtMaterial");
    scene_rdl2::rdl2::Geometry *geom =
        ctx.createSceneObject("TestRtGeometry", "geom")->asA<scene_rdl2::rdl2::Geometry>();
    scene_rdl2::rdl2::Layer* layer =
        ctx.createSceneObject("Layer", "/seq/shot/layer")->asA<scene_rdl2::rdl2::Layer>();
    scene_rdl2::rdl2::Material* mtl =
        ctx.createSceneObject("TestRtMaterial", "mtl")->asA<scene_rdl2::rdl2::Material>();
    scene_rdl2::rdl2::LightSet* lgt =
        ctx.createSceneObject("LightSet", "lgt")->asA<scene_rdl2::rdl2::LightSet>();
    layer->beginUpdate();
    layer->assign(geom, "", mtl, lgt);
    layer->endUpdate();
    layer->applyUpdates();

    geom->beginUpdate();
    // instance test case
    geom->set("test mode", 1);
    geom->endUpdate();
    geom->applyUpdates();

    geom->loadProcedural();
    GeomGenerateContext generateContext(nullptr, geom, moonray::shading::AttributeKeySet(),
        0, 1, moonray::geom::MotionBlurParams({0.f}, 0.f, 0.f, false, 24.f));
    scene_rdl2::math::Xform3f p2r(scene_rdl2::math::one);
    geom->getProcedural()->generate(generateContext, {p2r});

    scene_rdl2::rdl2::GeometrySet* geomSet =
        ctx.createSceneObject("GeometrySet", "geomSet")->asA<scene_rdl2::rdl2::GeometrySet>();
    geomSet->beginUpdate();
    geomSet->add(geom);
    geomSet->endUpdate();
    geomSet->applyUpdates();

    // initialize the xforms for the Instance primitives
    XformInitializer xi;
    geom->getProcedural()->forEachPrimitive(xi);

    // setup the BVH
    AcceleratorOptions options;
    EmbreeAccelerator bvh(options);
    bvh.build(OptimizationTarget::HIGH_QUALITY_BVH_BUILD, ChangeFlag::ALL, layer, {geomSet});

    // intersect test against two instances with above shared quad
    //(for detail see TestRtGeometry)
    //
    // (-1, 3, -1) ------------ (1, 3, -1)
    //            |            |
    //            |            |
    //            |            |
    //            |            |
    //            |            |
    // (-1, 1, -1) ------------ (1, 1, -1)
    //
    //
    //
    //
    // (-1,-1, -1) ------------ (1,-1, -1)
    //            |            |
    //            |            |
    //            |            |
    //            |            |
    //            |            |
    // (-1,-3, -1) ------------ (1,-3, -1)


    mcrt_common::Ray ray1;
    ray1.org = scene_rdl2::math::Vec3f(0.5f, 0.5f,  0.0f);
    ray1.dir = scene_rdl2::math::Vec3f(0.0f, 0.0f, -1.0f);
    bool correctRay1 = !bvh.occluded(ray1);

    mcrt_common::Ray ray2;
    ray2.org = scene_rdl2::math::Vec3f(0.5f, 2.5f,  0.0f);
    ray2.dir = scene_rdl2::math::Vec3f(0.0f, 0.0f, -1.0f);
    bool correctRay2 = bvh.occluded(ray2);

    mcrt_common::Ray ray3;
    ray3.org = scene_rdl2::math::Vec3f(-0.5f, -2.0f,  0.0f);
    ray3.dir = scene_rdl2::math::Vec3f( 0.0f,  0.0f, -1.0f);
    bvh.intersect(ray3);
    bool correctRay3 = ray3.geomID != RTC_INVALID_GEOMETRY_ID &&
        scene_rdl2::math::isEqual(ray3.tfar, 1.0f);

    mcrt_common::Ray ray4;
    ray4.org = scene_rdl2::math::Vec3f(0.0f, 0.0f,  0.0f);
    ray4.dir = scene_rdl2::math::Vec3f(0.0f, 0.0f, -1.0f);
    bvh.intersect(ray4);
    bool correctRay4 = ray4.geomID == RTC_INVALID_GEOMETRY_ID &&
        scene_rdl2::math::isEqual(ray4.tfar, FLT_MAX);

    geom->getProcedural()->clear();

    CPPUNIT_ASSERT(correctRay1);
    CPPUNIT_ASSERT(correctRay2);
    CPPUNIT_ASSERT(correctRay3);
    CPPUNIT_ASSERT(correctRay4);
}

// this is part of GeometryManager's duty to concatenate transform hierarchy
// for SharedPrimitive. In unit test we need to manually concatenate for
// nested instancing test case
class TransformConcatenator : public moonray::geom::PrimitiveVisitor
{
public:
    explicit TransformConcatenator(moonray::geom::SharedPrimitiveSet& sharedPrimitives):
        mSharedPrimitives(sharedPrimitives) {}

    virtual void visitPrimitiveGroup(moonray::geom::PrimitiveGroup& pg) override {
        pg.forEachPrimitive(*this);
        moonray::geom::internal::PrimitivePrivateAccess::transformPrimitive(&pg,
            moonray::geom::MotionBlurParams({0.f}, 0.f, 0.f, false, 24.f),
            moonray::shading::XformSamples(pg.getMotionSamplesCount(),
            scene_rdl2::math::Xform3f(scene_rdl2::math::one)));
    }

    virtual void visitInstance(moonray::geom::Instance& i) override {
        const auto& ref = i.getReference();
        // visit the referenced Primitive if it's not visited yet
        if (mSharedPrimitives.insert(ref).second) {
            ref->getPrimitive()->accept(*this);
        }
    }
private:
    moonray::geom::SharedPrimitiveSet& mSharedPrimitives;
};



void TestRenderingRT::testIntersectNestedInstances()
{
    // setup a simple rdl scene consisting of a geometry object to
    // generate primitives for intersection test
    scene_rdl2::rdl2::SceneContext ctx;
    ctx.setDsoPath(ctx.getDsoPath() +
        ":dso/geometry/TestRtGeometry:dso/material/TestRtMaterial");
    scene_rdl2::rdl2::Geometry* geom =
        ctx.createSceneObject("TestRtGeometry", "geom")->asA<scene_rdl2::rdl2::Geometry>();
    scene_rdl2::rdl2::Layer* layer =
        ctx.createSceneObject("Layer", "/seq/shot/layer")->asA<scene_rdl2::rdl2::Layer>();
    scene_rdl2::rdl2::Material* mtl =
        ctx.createSceneObject("TestRtMaterial", "mtl")->asA<scene_rdl2::rdl2::Material>();
    scene_rdl2::rdl2::LightSet* lgt =
        ctx.createSceneObject("LightSet", "lgt")->asA<scene_rdl2::rdl2::LightSet>();
    layer->beginUpdate();
    layer->assign(geom, "", mtl, lgt);
    layer->endUpdate();
    layer->applyUpdates();

    geom->beginUpdate();
    // nested instance test case
    geom->set("test mode", 2);
    geom->endUpdate();
    geom->applyUpdates();

    geom->loadProcedural();
    GeomGenerateContext generateContext(nullptr, geom, moonray::shading::AttributeKeySet(),
        0, 1, moonray::geom::MotionBlurParams({0.f}, 0.f, 0.f, false, 24.f));
    scene_rdl2::math::Xform3f p2r(scene_rdl2::math::one);
    geom->getProcedural()->generate(generateContext, {p2r});

    {
    moonray::geom::SharedPrimitiveSet sharedPrimitives;
    TransformConcatenator concatenator(sharedPrimitives);
    geom->getProcedural()->forEachPrimitive(concatenator);
    }

    scene_rdl2::rdl2::GeometrySet* geomSet =
        ctx.createSceneObject("GeometrySet", "geomSet")->asA<scene_rdl2::rdl2::GeometrySet>();
    geomSet->beginUpdate();
    geomSet->add(geom);
    geomSet->endUpdate();
    geomSet->applyUpdates();

    // initialize the xforms for the Instance primitives
    XformInitializer xi;
    geom->getProcedural()->forEachPrimitive(xi);

    // setup the BVH
    AcceleratorOptions options;
    EmbreeAccelerator bvh(options);
    bvh.build(OptimizationTarget::HIGH_QUALITY_BVH_BUILD, ChangeFlag::ALL, layer, {geomSet});

    // intersect test against a nested instancing scenario
    //(for detail see TestRtGeometry)
    //                  PG                                 PG
    //
    // (-3, 6, -1) ------------ (-1, 6,-1) (1, 6, -1) ------------ (3, 6, -1)
    //            |     P2     |                     |     P2     |
    //            |            |                     |            |
    // (-3, 5, -1) ------------ (-1, 5,-1) (1, 5, -1) ------------ (3, 5, -1)
    //
    //
    //
    // (-3, 3, -1) ------------ (-1, 3,-1) (1, 3, -1) ------------ (3, 3, -1)
    //            |            |                     |            |
    //            |            |                     |            |
    //            |     P1     |                     |     P1     |
    //            |            |                     |            |
    //            |            |                     |            |
    // (-3, 1, -1) ------------ (-1, 1,-1) (1, 1, -1) ------------ (3, 1,-1)
    //
    //
    //
    //
    // (-3,-1, -1) ------------ (-1,-1,-1) (1,-1, -1) ------------ (3,-1,-1)
    //            |            |                     |            |
    //            |            |                     |            |
    //            |     P1     |                     |     P1     |
    //            |            |                     |            |
    //            |            |                     |            |
    // (-3,-3, -1) ------------ (-1,-3,-1) (1,-3, -1) ------------ (3,-3,-1)

    mcrt_common::Ray ray1;
    ray1.org = scene_rdl2::math::Vec3f( 0.0f, 0.0f,  0.0f);
    ray1.dir = scene_rdl2::math::Vec3f( 0.0f, 0.0f, -1.0f);
    bool correctRay1 = !bvh.occluded(ray1);

    mcrt_common::Ray ray2;
    ray2.org = scene_rdl2::math::Vec3f(-2.5f, 5.5f,  0.0f);
    ray2.dir = scene_rdl2::math::Vec3f( 0.0f, 0.0f, -1.0f);
    bool correctRay2 = bvh.occluded(ray2);

    mcrt_common::Ray ray3;
    ray3.org = scene_rdl2::math::Vec3f(-2.5f, 2.5f,  0.0f);
    ray3.dir = scene_rdl2::math::Vec3f( 0.0f, 0.0f, -1.0f);
    bool correctRay3 = bvh.occluded(ray3);

    mcrt_common::Ray ray4;
    ray4.org = scene_rdl2::math::Vec3f(-2.5f,-2.5f,  0.0f);
    ray4.dir = scene_rdl2::math::Vec3f( 0.0f, 0.0f, -1.0f);
    bool correctRay4 = bvh.occluded(ray4);

    mcrt_common::Ray ray5;
    ray5.org = scene_rdl2::math::Vec3f( 2.5f, 5.5f,  0.0f);
    ray5.dir = scene_rdl2::math::Vec3f( 0.0f, 0.0f, -1.0f);
    bool correctRay5 = bvh.occluded(ray5);

    mcrt_common::Ray ray6;
    ray6.org = scene_rdl2::math::Vec3f( 2.5f, 2.5f,  0.0f);
    ray6.dir = scene_rdl2::math::Vec3f( 0.0f, 0.0f, -1.0f);
    bool correctRay6 = bvh.occluded(ray6);

    mcrt_common::Ray ray7;
    ray7.org = scene_rdl2::math::Vec3f( 2.5f,-2.5f,  0.0f);
    ray7.dir = scene_rdl2::math::Vec3f( 0.0f, 0.0f, -1.0f);
    bool correctRay7 = bvh.occluded(ray7);

    mcrt_common::Ray ray8;
    ray8.org = scene_rdl2::math::Vec3f(0.0f, 1.0f,  0.0f);
    ray8.dir = scene_rdl2::math::Vec3f(0.0f, 0.0f, -1.0f);
    bvh.intersect(ray8);
    bool correctRay8 = ray8.geomID == RTC_INVALID_GEOMETRY_ID &&
        scene_rdl2::math::isEqual(ray8.tfar, FLT_MAX);

    mcrt_common::Ray ray9;
    ray9.org = scene_rdl2::math::Vec3f(-1.5f,  5.5f,  0.0f);
    ray9.dir = scene_rdl2::math::Vec3f( 0.0f,  0.0f, -1.0f);
    bvh.intersect(ray9);
    bool correctRay9 = ray9.geomID != RTC_INVALID_GEOMETRY_ID &&
        scene_rdl2::math::isEqual(ray9.tfar, 1.0f);

    mcrt_common::Ray ray10;
    ray10.org = scene_rdl2::math::Vec3f(-1.5f,  2.0f,  0.0f);
    ray10.dir = scene_rdl2::math::Vec3f( 0.0f,  0.0f, -1.0f);
    bvh.intersect(ray10);
    bool correctRay10 = ray10.geomID != RTC_INVALID_GEOMETRY_ID &&
        scene_rdl2::math::isEqual(ray10.tfar, 1.0f);

    mcrt_common::Ray ray11;
    ray11.org = scene_rdl2::math::Vec3f(-1.5f, -2.0f,  0.0f);
    ray11.dir = scene_rdl2::math::Vec3f( 0.0f,  0.0f, -1.0f);
    bvh.intersect(ray11);
    bool correctRay11 = ray11.geomID != RTC_INVALID_GEOMETRY_ID &&
        scene_rdl2::math::isEqual(ray11.tfar, 1.0f);

    mcrt_common::Ray ray12;
    ray12.org = scene_rdl2::math::Vec3f( 1.5f,  5.5f,  0.0f);
    ray12.dir = scene_rdl2::math::Vec3f( 0.0f,  0.0f, -1.0f);
    bvh.intersect(ray12);
    bool correctRay12 = ray12.geomID != RTC_INVALID_GEOMETRY_ID &&
        scene_rdl2::math::isEqual(ray12.tfar, 1.0f);

    mcrt_common::Ray ray13;
    ray13.org = scene_rdl2::math::Vec3f( 1.5f,  2.0f,  0.0f);
    ray13.dir = scene_rdl2::math::Vec3f( 0.0f,  0.0f, -1.0f);
    bvh.intersect(ray13);
    bool correctRay13 = ray13.geomID != RTC_INVALID_GEOMETRY_ID &&
        scene_rdl2::math::isEqual(ray13.tfar, 1.0f);

    mcrt_common::Ray ray14;
    ray14.org = scene_rdl2::math::Vec3f( 1.5f, -2.0f,  0.0f);
    ray14.dir = scene_rdl2::math::Vec3f( 0.0f,  0.0f, -1.0f);
    bvh.intersect(ray14);
    bool correctRay14 = ray14.geomID != RTC_INVALID_GEOMETRY_ID &&
        scene_rdl2::math::isEqual(ray14.tfar, 1.0f);

    geom->getProcedural()->clear();

    CPPUNIT_ASSERT(correctRay1);
    CPPUNIT_ASSERT(correctRay2);
    CPPUNIT_ASSERT(correctRay3);
    CPPUNIT_ASSERT(correctRay4);
    CPPUNIT_ASSERT(correctRay5);
    CPPUNIT_ASSERT(correctRay6);
    CPPUNIT_ASSERT(correctRay7);
    CPPUNIT_ASSERT(correctRay8);
    CPPUNIT_ASSERT(correctRay9);
    CPPUNIT_ASSERT(correctRay10);
    CPPUNIT_ASSERT(correctRay11);
    CPPUNIT_ASSERT(correctRay12);
    CPPUNIT_ASSERT(correctRay13);
    CPPUNIT_ASSERT(correctRay14);
}

