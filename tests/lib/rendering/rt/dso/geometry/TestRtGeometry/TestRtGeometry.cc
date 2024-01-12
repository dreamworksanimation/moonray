// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file TestRtGeometry.cc
/// $Id$
///

#include "attributes.cc"
#include <moonray/rendering/geom/Api.h>
#include <moonray/rendering/geom/PrimitiveVisitor.h>
#include <moonray/rendering/geom/ProceduralLeaf.h>
#include <moonray/rendering/geom/Types.h>
#include <moonray/rendering/shading/Shading.h>
#include <scene_rdl2/scene/rdl2/rdl2.h>

RDL2_DSO_CLASS_BEGIN(TestRtGeometry, scene_rdl2::rdl2::Geometry)

public:
    RDL2_DSO_DEFAULT_CTOR(TestRtGeometry)
    moonray::geom::Procedural* createProcedural() const;
    void destroyProcedural() const;    
    bool deformed() const;
    void resetDeformed();

RDL2_DSO_CLASS_END(TestRtGeometry)

//------------------------------------------------------------------------------

namespace moonray {
namespace geom {

class PolygonTestCaseUpdate : public PrimitiveVisitor
{
public:
    PolygonTestCaseUpdate(const moonray::shading::XformSamples& prim2render) : mP2R(prim2render)
    {}

    void visitPolygonMesh(PolygonMesh& p) override {
        // update test quad face from original position
        // (-1, 1, -1) ------------ (1, 1, -1)
        //            |            |
        //            |            |
        //            |            |
        //            |            |
        //            |            |
        // (-1,-1, -1) ------------ (1,-1, -1)
        //
        // to
        //
        //                          ( 2,-2, -1) ------------ (4,-2, -1)
        //                                     |            |
        //                                     |            |
        //                                     |            |
        //                                     |            |
        //                                     |            |
        //                          ( 2,-4, -1) ------------ (4,-4, -1)
        //
        std::vector<float> newVertices({
            2, -2, -1,
            2, -4, -1,
            4, -4, -1,
            4, -2, -1});
        p.updateVertexData(newVertices, mP2R);
    }

private:
    const moonray::shading::XformSamples& mP2R;
};

class TemplateProcedural : public ProceduralLeaf
{
public:
    // constructor can be freely extended but should always pass in State to
    // construct base Procedural class
    TemplateProcedural(const State& state) :
        ProceduralLeaf(state) {}

    void generate(const GenerateContext& generateContext,
			const moonray::shading::XformSamples& parent2render)
    {     
        int testMode = generateContext.getRdlGeometry()->get(attrTestMode);
        if (testMode == 0) {
            createPolygonTestCase(generateContext, parent2render);
        } else if (testMode == 1) {
            createInstanceTestCase(generateContext, parent2render);
        } else if (testMode == 2) {
            createNestedInstanceTestCase(generateContext, parent2render);
        }
    }

    void update(const UpdateContext& updateContext,
			const moonray::shading::XformSamples& parent2render)
    {
        // Implement this method to update primitives created from
        // generate call
        PolygonTestCaseUpdate updater(parent2render);
        forEachPrimitive(updater);
        mDeformed = true;
    }

private:

    void createPolygonTestCase(const GenerateContext& generateContext,
            const moonray::shading::XformSamples& parent2render) {
        // create quad face
        // (-1, 1, -1) ------------ (1, 1, -1)
        //            |            |
        //            |            |
        //            |            |
        //            |            |
        //            |            |
        // (-1,-1, -1) ------------ (1,-1, -1)
        PolygonMesh::FaceVertexCount faceVertexCount({4});
        PolygonMesh::IndexBuffer indices({0, 1, 2, 3});
        PolygonMesh::VertexBuffer vertices;
        vertices.push_back(Vec3fa(-1,  1, -1, 0));
        vertices.push_back(Vec3fa(-1, -1, -1, 0));
        vertices.push_back(Vec3fa( 1, -1, -1, 0));
        vertices.push_back(Vec3fa( 1,  1, -1, 0));
        std::unique_ptr<PolygonMesh> mesh = createPolygonMesh(
            std::move(faceVertexCount), std::move(indices),
            std::move(vertices), LayerAssignmentId(0));
        addPrimitive(std::move(mesh), generateContext.getMotionBlurParams(),
            parent2render);
    }

    void createInstanceTestCase(const GenerateContext& generateContext,
            const moonray::shading::XformSamples& parent2render) {
        // form a shared  quad face to instance around, in its local space:
        // (-1, 1, -1) ------------ (1, 1, -1)
        //            |            |
        //            |            |
        //            |            |
        //            |            |
        //            |            |
        // (-1,-1, -1) ------------ (1,-1, -1)

        PolygonMesh::FaceVertexCount faceVertexCount({4});
        PolygonMesh::IndexBuffer indices({0, 1, 2, 3});
        PolygonMesh::VertexBuffer vertices;
        vertices.push_back(Vec3fa(-1,  1, -1, 0));
        vertices.push_back(Vec3fa(-1, -1, -1, 0));
        vertices.push_back(Vec3fa( 1, -1, -1, 0));
        vertices.push_back(Vec3fa( 1,  1, -1, 0));
        std::unique_ptr<PolygonMesh> mesh = createPolygonMesh(
            std::move(faceVertexCount), std::move(indices),
            std::move(vertices), LayerAssignmentId(0));

        std::shared_ptr<SharedPrimitive> sharedP = createSharedPrimitive(
             std::move(mesh));
        // create two instances with above shared quad, in render space:
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

        addPrimitive(createInstance(Mat43::translate(Vec3f(0, 2, 0)), sharedP),
            generateContext.getMotionBlurParams(), parent2render);

        addPrimitive(createInstance(Mat43::translate(Vec3f(0, -2, 0)), sharedP),
            generateContext.getMotionBlurParams(), parent2render);
    }

    void createNestedInstanceTestCase(const GenerateContext& generateContext,
            const moonray::shading::XformSamples& parent2render) {
        // form a shared  quad face to instance around, in its local space:
        // (-1, 1, -1) ------------ (1, 1, -1)
        //            |            |
        //            |            |
        //            |    P1      |
        //            |            |
        //            |            |
        // (-1,-1, -1) ------------ (1,-1, -1)

        PolygonMesh::FaceVertexCount faceVertexCount({4});
        PolygonMesh::IndexBuffer indices({0, 1, 2, 3});
        PolygonMesh::VertexBuffer vertices;
        vertices.push_back(Vec3fa(-1,  1, -1, 0));
        vertices.push_back(Vec3fa(-1, -1, -1, 0));
        vertices.push_back(Vec3fa( 1, -1, -1, 0));
        vertices.push_back(Vec3fa( 1,  1, -1, 0));
        std::unique_ptr<PolygonMesh> P1 = createPolygonMesh(
            std::move(faceVertexCount), std::move(indices),
            std::move(vertices), LayerAssignmentId(0));

        std::shared_ptr<SharedPrimitive> sharedP1 = createSharedPrimitive(
             std::move(P1));
        // create primitive group with two instances of above shared quad P1,
        // and add in one regular primitive P2:
        //                  PG
        //
        // (-1, 6, -1) ------------ (1, 6, -1)
        //            |     P2     |
        //            |            |
        // (-1, 5, -1) ------------ (1, 5, -1)
        //
        //
        //
        // (-1, 3, -1) ------------ (1, 3, -1)
        //            |            |
        //            |            |
        //            |     P1     |
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
        //            |     P1     |
        //            |            |
        //            |            |
        // (-1,-3, -1) ------------ (1,-3, -1)

        std::unique_ptr<PrimitiveGroup> PG = createPrimitiveGroup();

        PolygonMesh::FaceVertexCount faceVertexCount2({4});
        PolygonMesh::IndexBuffer indices2({0, 1, 2, 3});
        PolygonMesh::VertexBuffer vertices2;
        vertices2.push_back(Vec3fa(-1,  6, -1, 0));
        vertices2.push_back(Vec3fa(-1,  5, -1, 0));
        vertices2.push_back(Vec3fa( 1,  5, -1, 0));
        vertices2.push_back(Vec3fa( 1,  6, -1, 0));
        std::unique_ptr<PolygonMesh> P2 = createPolygonMesh(
            std::move(faceVertexCount2), std::move(indices2),
            std::move(vertices2), LayerAssignmentId(0));
        PG->addPrimitive(std::move(P2));
        PG->addPrimitive(createInstance(Mat43::translate(Vec3f(0, 2, 0)),
            sharedP1));
        PG->addPrimitive(createInstance(Mat43::translate(Vec3f(0,-2, 0)),
            sharedP1));


        // now create two instances with above primitive group PG
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

        std::shared_ptr<SharedPrimitive> sharedPG =
            createSharedPrimitive(std::move(PG));
        addPrimitive(
            createInstance(Mat43::translate(Vec3f(-2, 0, 0)), sharedPG),
            generateContext.getMotionBlurParams(), parent2render);
        addPrimitive(
            createInstance(Mat43::translate(Vec3f(2, 0, 0)), sharedPG),
            generateContext.getMotionBlurParams(), parent2render);
    }

};

} // namespace geom
} // namespace moonray

//------------------------------------------------------------------------------

moonray::geom::Procedural* TestRtGeometry::createProcedural() const
{
    moonray::geom::State state;
    return new moonray::geom::TemplateProcedural(state);
}

void TestRtGeometry::destroyProcedural() const
{
    delete mProcedural;
}

bool TestRtGeometry::deformed() const
{
    return mProcedural->deformed();
}

void TestRtGeometry::resetDeformed()
{
    mProcedural->resetDeformed();
}

