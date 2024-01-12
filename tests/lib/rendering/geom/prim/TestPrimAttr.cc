// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#include "TestPrimAttr.h"
#include "TestPrimUtils.h"

#include <moonray/rendering/geom/prim/OpenSubdivMesh.h>
#include <moonray/rendering/bvh/shading/AttributeKey.h>
#include <moonray/rendering/bvh/shading/Attributes.h>
#include <moonray/rendering/bvh/shading/InstanceAttributes.h>
#include <scene_rdl2/common/math/Xform.h>
#include <scene_rdl2/scene/rdl2/Geometry.h>
#include <scene_rdl2/scene/rdl2/SceneContext.h>
#include <scene_rdl2/scene/rdl2/Layer.h>
#include <scene_rdl2/render/util/stdmemory.h>

#include <memory>
#include <vector>

namespace moonray {
namespace geom {
namespace unittest {

using namespace shading;

void TestRenderingPrimAttr::setUp() {
    TestAttributes::init();

    mC0 = scene_rdl2::math::Color(0.371f,0.952f,0.231f);
    mC1 = scene_rdl2::math::Color(0.825f,0.111f,0.713f);
    mC2 = scene_rdl2::math::Color(0.871f,0.282f,0.141f);
    mC3 = scene_rdl2::math::Color(0.925f,0.333f,0.615f);
    mS0 = std::string("testing string0");
    mS1 = std::string("testing string1");
    mF0 = 423.117f;
    mF1 = 3.27f;
    mF2 = 100.7f;
    mF3 = -42.3f;
}

void TestRenderingPrimAttr::tearDown() {
}


template<typename T>
static bool verifyConstant(const std::unique_ptr<Attributes>& attr,
    TypedAttributeKey<T> key, const T& canonical)
{
    T result;
    char* ptr = (char*)(&result);
    attr->getConstant(key, ptr);
    return result == canonical;
}

template<>
bool verifyConstant(const std::unique_ptr<Attributes>& attr,
    TypedAttributeKey<std::string> key, const std::string& canonical)
{
    std::string* result;
    char* ptr = (char*)(&result);
    attr->getConstant(key, ptr);
    return *result == canonical;
}

template<typename T>
static bool verifyPart(const std::unique_ptr<Attributes>& attr,
    TypedAttributeKey<T> key, int part, const T& canonical)
{
    T result;
    char* ptr = (char*)(&result);
    attr->getPart(key, part, ptr);
    return scene_rdl2::math::isEqual(result, canonical);
}

template<>
bool verifyPart(const std::unique_ptr<Attributes>& attr,
    TypedAttributeKey<std::string> key, int part, const std::string& canonical)
{
    std::string* result;
    char* ptr = (char*)(&result);
    attr->getPart(key, part, ptr);
    return *result == canonical;
}

template<typename T>
static bool verifyUniform(const std::unique_ptr<Attributes>& attr,
    TypedAttributeKey<T> key, int face, const T& canonical)
{
    T result;
    char* ptr = (char*)(&result);
    attr->getUniform(key, face, ptr);
    return scene_rdl2::math::isEqual(result, canonical);
}

template<>
bool verifyUniform(const std::unique_ptr<Attributes>& attr,
    TypedAttributeKey<std::string> key, int face, const std::string& canonical)
{
    std::string* result;
    char* ptr = (char*)(&result);
    attr->getUniform(key, face, ptr);
    return *result == canonical;
}



void TestRenderingPrimAttr::testConstant()
{
    PrimitiveAttributeTable table;
    table.addAttribute(TestAttributes::sTestFloat0, RATE_CONSTANT, std::vector<float>{(mF0)});
    table.addAttribute(TestAttributes::sTestColor0, RATE_CONSTANT, {mC0});
    table.addAttribute(TestAttributes::sTestColor1, RATE_CONSTANT, {mC1});
    table.addAttribute(TestAttributes::sTestString0, RATE_CONSTANT, {mS0});
    std::unique_ptr<Attributes> attr(
        Attributes::interleave(table, 0, 0, 0,
        std::vector<size_t>(), 0));
    CPPUNIT_ASSERT(verifyConstant(attr, TestAttributes::sTestColor0, mC0));
    CPPUNIT_ASSERT(verifyConstant(attr, TestAttributes::sTestColor1, mC1));
    CPPUNIT_ASSERT(verifyConstant(attr, TestAttributes::sTestFloat0, mF0));
    CPPUNIT_ASSERT(verifyConstant(attr, TestAttributes::sTestString0, mS0));
}

void TestRenderingPrimAttr::testPart0()
{
    PrimitiveAttributeTable table;
    size_t numParts = 2;
    table.addAttribute(TestAttributes::sTestColor0, RATE_PART, {mC0, mC1});
    std::unique_ptr<Attributes> attr(
        Attributes::interleave(
        table, numParts, 0, 0, std::vector<size_t>(), 0));
    CPPUNIT_ASSERT(verifyPart(attr, TestAttributes::sTestColor0, 0, mC0));
    CPPUNIT_ASSERT(verifyPart(attr, TestAttributes::sTestColor0, 1, mC1));
}

void TestRenderingPrimAttr::testPart1()
{
    // Multiple uniform attrs, different sizes
    PrimitiveAttributeTable table;
    size_t numParts = 2;
    table.addAttribute(TestAttributes::sTestColor0, RATE_PART, {mC0, mC2});
    table.addAttribute(TestAttributes::sTestFloat0, RATE_PART, {mF0, mF1});
    table.addAttribute(TestAttributes::sTestColor1, RATE_PART, {mC1, mC3});
    table.addAttribute(TestAttributes::sTestString0, RATE_PART, {mS0, mS1});
    std::unique_ptr<Attributes> attr(
        Attributes::interleave(
        table, numParts, 0, 0, std::vector<size_t>(), 0));

    CPPUNIT_ASSERT(verifyPart(attr, TestAttributes::sTestColor0, 0, mC0));
    CPPUNIT_ASSERT(verifyPart(attr, TestAttributes::sTestColor0, 1, mC2));
    CPPUNIT_ASSERT(verifyPart(attr, TestAttributes::sTestFloat0, 0, mF0));
    CPPUNIT_ASSERT(verifyPart(attr, TestAttributes::sTestFloat0, 1, mF1));
    CPPUNIT_ASSERT(verifyPart(attr, TestAttributes::sTestColor1, 0, mC1));
    CPPUNIT_ASSERT(verifyPart(attr, TestAttributes::sTestColor1, 1, mC3));
    CPPUNIT_ASSERT(verifyPart(attr, TestAttributes::sTestString0, 0, mS0));
    CPPUNIT_ASSERT(verifyPart(attr, TestAttributes::sTestString0, 1, mS1));
}

void TestRenderingPrimAttr::testPart2()
{
    // Rewriting to same attr
    PrimitiveAttributeTable table;
    size_t numParts = 2;
    table.addAttribute(TestAttributes::sTestColor0, RATE_PART, {mC0});
    table.addAttribute(TestAttributes::sTestFloat0, RATE_PART, std::vector<float>{mF0});
    table.addAttribute(TestAttributes::sTestColor1, RATE_PART, {mC1});
    table.getAttribute(TestAttributes::sTestColor1)[0] = mC2;
    table.addAttribute(TestAttributes::sTestString0, RATE_PART, {mS0});
    table.getAttribute(TestAttributes::sTestString0)[0] = mS1;

    table.getAttribute(TestAttributes::sTestColor0).push_back(mC2);
    table.getAttribute(TestAttributes::sTestFloat0).push_back(mF1);
    table.getAttribute(TestAttributes::sTestColor1).push_back(mC3);
    table.getAttribute(TestAttributes::sTestString0).push_back(mS1);
    table.getAttribute(TestAttributes::sTestString0)[1] = mS0;

    std::unique_ptr<Attributes> attr(
        Attributes::interleave(
        table, numParts, 0, 0, std::vector<size_t>(), 0));

    CPPUNIT_ASSERT(verifyPart(attr, TestAttributes::sTestColor0, 0, mC0));
    CPPUNIT_ASSERT(verifyPart(attr, TestAttributes::sTestColor0, 1, mC2));
    CPPUNIT_ASSERT(verifyPart(attr, TestAttributes::sTestFloat0, 0, mF0));
    CPPUNIT_ASSERT(verifyPart(attr, TestAttributes::sTestFloat0, 1, mF1));
    CPPUNIT_ASSERT(verifyPart(attr, TestAttributes::sTestColor1, 0, mC2));
    CPPUNIT_ASSERT(verifyPart(attr, TestAttributes::sTestColor1, 1, mC3));
    CPPUNIT_ASSERT(verifyPart(attr, TestAttributes::sTestString0, 0, mS1));
    CPPUNIT_ASSERT(verifyPart(attr, TestAttributes::sTestString0, 1, mS0));
}

void TestRenderingPrimAttr::testUniform0()
{
    PrimitiveAttributeTable table;
    table.addAttribute(TestAttributes::sTestColor0, RATE_UNIFORM, {mC0, mC1});
    size_t numFaces= 2;
    std::vector<size_t> numFaceVaryings(numFaces, 0);
    std::unique_ptr<Attributes> attr(
        Attributes::interleave(
        table, 0, numFaces, 0, numFaceVaryings, 0));
    CPPUNIT_ASSERT(verifyUniform(attr, TestAttributes::sTestColor0, 0, mC0));
    CPPUNIT_ASSERT(verifyUniform(attr, TestAttributes::sTestColor0, 1, mC1));
}

void TestRenderingPrimAttr::testUniform1()
{
    // Multiple uniform attrs, different sizes
    PrimitiveAttributeTable table;
    size_t numFaces= 2;
    table.addAttribute(TestAttributes::sTestColor0, RATE_UNIFORM, {mC0, mC2});
    table.addAttribute(TestAttributes::sTestFloat0, RATE_UNIFORM, {mF0, mF1});
    table.addAttribute(TestAttributes::sTestColor1, RATE_UNIFORM, {mC1, mC3});
    table.addAttribute(TestAttributes::sTestString0, RATE_UNIFORM, {mS0, mS1});
    std::vector<size_t> numFaceVaryings(numFaces, 0);
    std::unique_ptr<Attributes> attr(
        Attributes::interleave(
        table, 0, numFaces, 0, numFaceVaryings, 0));

    CPPUNIT_ASSERT(verifyUniform(attr, TestAttributes::sTestColor0, 0, mC0));
    CPPUNIT_ASSERT(verifyUniform(attr, TestAttributes::sTestColor0, 1, mC2));
    CPPUNIT_ASSERT(verifyUniform(attr, TestAttributes::sTestFloat0, 0, mF0));
    CPPUNIT_ASSERT(verifyUniform(attr, TestAttributes::sTestFloat0, 1, mF1));
    CPPUNIT_ASSERT(verifyUniform(attr, TestAttributes::sTestColor1, 0, mC1));
    CPPUNIT_ASSERT(verifyUniform(attr, TestAttributes::sTestColor1, 1, mC3));
    CPPUNIT_ASSERT(verifyUniform(attr, TestAttributes::sTestString0, 0, mS0));
    CPPUNIT_ASSERT(verifyUniform(attr, TestAttributes::sTestString0, 1, mS1));
}

void TestRenderingPrimAttr::testUniform2()
{
    // Rewriting to same attr
    PrimitiveAttributeTable table;
    size_t numFaces= 2;
    std::vector<size_t> numFaceVaryings(numFaces, 0);
    table.addAttribute(TestAttributes::sTestColor0, RATE_UNIFORM, {mC0});
    table.addAttribute(TestAttributes::sTestFloat0, RATE_UNIFORM, std::vector<float>{mF0});
    table.addAttribute(TestAttributes::sTestColor1, RATE_UNIFORM, {mC1});
    table.getAttribute(TestAttributes::sTestColor1)[0] = mC2;
    table.addAttribute(TestAttributes::sTestString0, RATE_UNIFORM, {mS0});
    table.getAttribute(TestAttributes::sTestString0)[0] = mS1;

    table.getAttribute(TestAttributes::sTestColor0).push_back(mC2);
    table.getAttribute(TestAttributes::sTestFloat0).push_back(mF1);
    table.getAttribute(TestAttributes::sTestColor1).push_back(mC3);
    table.getAttribute(TestAttributes::sTestString0).push_back(mS1);
    table.getAttribute(TestAttributes::sTestString0)[1] = mS0;

    std::unique_ptr<Attributes> attr(
        Attributes::interleave(
        table, 0, numFaces, 0, numFaceVaryings, 0));

    CPPUNIT_ASSERT(verifyUniform(attr, TestAttributes::sTestColor0, 0, mC0));
    CPPUNIT_ASSERT(verifyUniform(attr, TestAttributes::sTestColor0, 1, mC2));
    CPPUNIT_ASSERT(verifyUniform(attr, TestAttributes::sTestFloat0, 0, mF0));
    CPPUNIT_ASSERT(verifyUniform(attr, TestAttributes::sTestFloat0, 1, mF1));
    CPPUNIT_ASSERT(verifyUniform(attr, TestAttributes::sTestColor1, 0, mC2));
    CPPUNIT_ASSERT(verifyUniform(attr, TestAttributes::sTestColor1, 1, mC3));
    CPPUNIT_ASSERT(verifyUniform(attr, TestAttributes::sTestString0, 0, mS1));
    CPPUNIT_ASSERT(verifyUniform(attr, TestAttributes::sTestString0, 1, mS0));
}

void TestRenderingPrimAttr::testFaceVarying0()
{
    // Basic face-varying
    PrimitiveAttributeTable table;
    size_t numFaces= 1;
    std::vector<size_t> numFaceVaryings(numFaces, 1);
    table.addAttribute(TestAttributes::sTestFloat0, RATE_FACE_VARYING, std::vector<float>{mF0});
    table.addAttribute(TestAttributes::sTestColor0, RATE_FACE_VARYING, {mC0});
    std::unique_ptr<Attributes> attr(
        Attributes::interleave(
        table, 0, numFaces, 0, numFaceVaryings, 0));
    CPPUNIT_ASSERT(
        attr->getFaceVarying(TestAttributes::sTestFloat0,0,0) == mF0);
    CPPUNIT_ASSERT(
        attr->getFaceVarying(TestAttributes::sTestColor0,0,0) == mC0);
}

void TestRenderingPrimAttr::testFaceVarying1()
{
    // Multiple varyings
     PrimitiveAttributeTable table;
    size_t numFaces= 1;
    std::vector<size_t> numFaceVaryings(numFaces, 2);
    table.addAttribute(TestAttributes::sTestFloat0, RATE_FACE_VARYING,
        {mF0, mF1});
    table.addAttribute(TestAttributes::sTestColor0, RATE_FACE_VARYING,
        {mC0, mC1});
    std::unique_ptr<Attributes> attr(
        Attributes::interleave(
        table, 0, numFaces, 0, numFaceVaryings, 0));

    CPPUNIT_ASSERT(
        attr->getFaceVarying(TestAttributes::sTestFloat0,0,0) == mF0);
    CPPUNIT_ASSERT(
        attr->getFaceVarying(TestAttributes::sTestColor0,0,0) == mC0);
    CPPUNIT_ASSERT(
        attr->getFaceVarying(TestAttributes::sTestFloat0,0,1) == mF1);
    CPPUNIT_ASSERT(
        attr->getFaceVarying(TestAttributes::sTestColor0,0,1) == mC1);
}

void TestRenderingPrimAttr::testFaceVarying2()
{
    // Multiple faces, multiple varyings
    PrimitiveAttributeTable table;
    size_t numFaces= 2;
    std::vector<size_t> numFaceVaryings(numFaces, 2);
    table.addAttribute(TestAttributes::sTestFloat0, RATE_FACE_VARYING,
        {mF0, mF1, mF2, mF3});
    table.addAttribute(TestAttributes::sTestColor0, RATE_FACE_VARYING,
        {mC0, mC1, mC2, mC3});
    std::unique_ptr<Attributes> attr(
        Attributes::interleave(
        table, 0, numFaces, 0, numFaceVaryings, 0));

    CPPUNIT_ASSERT(
        attr->getFaceVarying(TestAttributes::sTestFloat0,0,0) == mF0);
    CPPUNIT_ASSERT(
        attr->getFaceVarying(TestAttributes::sTestColor0,0,0) == mC0);
    CPPUNIT_ASSERT(
        attr->getFaceVarying(TestAttributes::sTestFloat0,0,1) == mF1);
    CPPUNIT_ASSERT(
        attr->getFaceVarying(TestAttributes::sTestColor0,0,1) == mC1);
    CPPUNIT_ASSERT(
        attr->getFaceVarying(TestAttributes::sTestFloat0,1,0) == mF2);
    CPPUNIT_ASSERT(
        attr->getFaceVarying(TestAttributes::sTestColor0,1,0) == mC2);
    CPPUNIT_ASSERT(
        attr->getFaceVarying(TestAttributes::sTestFloat0,1,1) == mF3);
    CPPUNIT_ASSERT(
        attr->getFaceVarying(TestAttributes::sTestColor0,1,1) == mC3);
}

void TestRenderingPrimAttr::testFaceVarying3()
{
    // Varying numbers of vertices(spans) per face(curve)
    PrimitiveAttributeTable table;
    size_t numFaces= 4;
    std::vector<size_t> numFaceVaryings{1, 3, 0, 2};
    table.addAttribute(TestAttributes::sTestFloat0, RATE_FACE_VARYING,
        {mF0, mF2, mF3, mF3, mF3, mF2});
    table.addAttribute(TestAttributes::sTestColor0, RATE_FACE_VARYING,
        {mC0, mC2, mC3, mC0, mC1, mC0});
    std::unique_ptr<Attributes> attr(
        Attributes::interleave(
        table, 0, numFaces, 0, numFaceVaryings, 0));

    CPPUNIT_ASSERT(
        attr->getFaceVarying(TestAttributes::sTestFloat0,0,0) == mF0);
    CPPUNIT_ASSERT(
        attr->getFaceVarying(TestAttributes::sTestColor0,0,0) == mC0);

    CPPUNIT_ASSERT(
        attr->getFaceVarying(TestAttributes::sTestFloat0,1,0) == mF2);
    CPPUNIT_ASSERT(
        attr->getFaceVarying(TestAttributes::sTestColor0,1,0) == mC2);
    CPPUNIT_ASSERT(
        attr->getFaceVarying(TestAttributes::sTestFloat0,1,1) == mF3);
    CPPUNIT_ASSERT(
        attr->getFaceVarying(TestAttributes::sTestColor0,1,1) == mC3);
    CPPUNIT_ASSERT(
        attr->getFaceVarying(TestAttributes::sTestFloat0,1,2) == mF3);
    CPPUNIT_ASSERT(
        attr->getFaceVarying(TestAttributes::sTestColor0,1,2) == mC0);

    CPPUNIT_ASSERT(
        attr->getFaceVarying(TestAttributes::sTestFloat0,3,0) == mF3);
    CPPUNIT_ASSERT(
        attr->getFaceVarying(TestAttributes::sTestColor0,3,0) == mC1);
    CPPUNIT_ASSERT(
        attr->getFaceVarying(TestAttributes::sTestFloat0,3,1) == mF2);
    CPPUNIT_ASSERT(
        attr->getFaceVarying(TestAttributes::sTestColor0,3,1) == mC0);
}

void TestRenderingPrimAttr::testVarying0()
{
    // Basic varying
    PrimitiveAttributeTable table;
    size_t numVaryings= 1;
    table.addAttribute(TestAttributes::sTestFloat0, RATE_VARYING, std::vector<float>{mF0});
    table.addAttribute(TestAttributes::sTestColor0, RATE_VARYING, {mC0});
    std::unique_ptr<Attributes> attr(
        Attributes::interleave(table, 0, 0, numVaryings,
        std::vector<size_t>(), 0));

    CPPUNIT_ASSERT(attr->getVarying(TestAttributes::sTestFloat0,0) == mF0);
    CPPUNIT_ASSERT(attr->getVarying(TestAttributes::sTestColor0,0) == mC0);
}

void TestRenderingPrimAttr::testVertex0()
{
    // Basic vertex
    PrimitiveAttributeTable table;
    size_t numVertices= 1;
    table.addAttribute(TestAttributes::sTestFloat0, RATE_VERTEX, std::vector<float>{mF0});
    table.addAttribute(TestAttributes::sTestColor0, RATE_VERTEX, {mC0});
    std::unique_ptr<Attributes> attr(
        Attributes::interleave(table, 0, 0, 0,
        std::vector<size_t>(), numVertices));

    CPPUNIT_ASSERT(attr->getVertex(TestAttributes::sTestFloat0,0) == mF0);
    CPPUNIT_ASSERT(attr->getVertex(TestAttributes::sTestColor0,0) == mC0);
}

static void fillTestSubdMeshData(
        SubdivisionMesh::FaceVertexCount& faceVertexCount,
        SubdivisionMesh::VertexBuffer& vertices,
        SubdivisionMesh::IndexBuffer& indices) {

    faceVertexCount = {3, 4, 3, 3, 4, 3, 3, 4, 3};
    vertices.resize(8);
    float sqrt3 = sqrt(3.0f);
    vertices(0) = Vec3fa( 0.0f, 0.0f, -3.0f, 0.f);
    vertices(1) = Vec3fa(-2.0f, 2.0f / 3.0f * sqrt3, -2.0f, 0.f);
    vertices(2) = Vec3fa( 2.0f, 2.0f / 3.0f * sqrt3, -2.0f, 0.f);
    vertices(3) = Vec3fa(-2.0f, 2.0f / 3.0f * sqrt3,  2.0f, 0.f);
    vertices(4) = Vec3fa( 2.0f, 2.0f / 3.0f * sqrt3,  2.0f, 0.f);
    vertices(5) = Vec3fa( 0.0f, 0.0f,  3.0f, 0.f);
    vertices(6) = Vec3fa( 0.0f,-4.0f / 3.0f * sqrt3, -2.0f, 0.f);
    vertices(7) = Vec3fa( 0.0f,-4.0f / 3.0f * sqrt3,  2.0f, 0.f);

    indices = {
        0, 1, 2,
        1, 3, 4, 2,
        3, 5, 4,
        0, 2, 6,
        2, 4, 7, 6,
        4, 5, 7,
        0, 6, 1,
        6, 7, 3, 1,
        7, 5, 3};
}

void TestRenderingPrimAttr::testVertex1()
{
    scene_rdl2::rdl2::SceneContext ctx;
    ctx.setDsoPath(ctx.getDsoPath() +
        ":dso/geometry/TestGeometry:dso/material/TestMaterial");
    scene_rdl2::rdl2::Geometry* geom =
        ctx.createSceneObject("TestGeometry", "geom")->asA<scene_rdl2::rdl2::Geometry>();
    scene_rdl2::rdl2::Layer* layer =
        ctx.createSceneObject("Layer", "/seq/shot/layer")->asA<scene_rdl2::rdl2::Layer>();
    scene_rdl2::rdl2::Material* mtl =
        ctx.createSceneObject("TestMaterial", "mtl")->asA<scene_rdl2::rdl2::Material>();
    scene_rdl2::rdl2::LightSet* lgt =
        ctx.createSceneObject("LightSet", "lgt")->asA<scene_rdl2::rdl2::LightSet>();

    layer->beginUpdate();
    // LayerAssignmentID(0) below
    layer->assign(geom, "empty", mtl, lgt, nullptr, nullptr);
    layer->endUpdate();

    // Testing subdivision mesh vertex rate tessellation
    SubdivisionMesh::FaceVertexCount faceVertexCount;
    SubdivisionMesh::VertexBuffer vertices;
    SubdivisionMesh::IndexBuffer indices;
    fillTestSubdMeshData(faceVertexCount, vertices, indices);

    // insert a Vec3 primitive atribute that is a duplication of position data
    PrimitiveAttributeTable table;
    std::vector<Vec3f> pRefAttr1(vertices.size());
    std::vector<Vec3f> pRefAttr2(vertices.size());
    std::vector<Vec3f> pRefAttr3(vertices.size());
    for (size_t i = 0; i < vertices.size(); ++i) {
        pRefAttr1[i] = vertices(i);
        pRefAttr2[i] = vertices(i);
        pRefAttr3[i] = vertices(i);
    }
    TypedAttributeKey<Vec3f> pRef1Key("pRef1");
    TypedAttributeKey<Vec3f> pRef2Key("pRef2");
    TypedAttributeKey<Vec3f> pRef3Key("pRef3");
    table.addAttribute(pRef1Key, RATE_VERTEX, std::move(pRefAttr1));
    table.addAttribute(pRef2Key, RATE_VERTEX, std::move(pRefAttr2));
    table.addAttribute(pRef3Key, RATE_VERTEX, std::move(pRefAttr3));

    std::unique_ptr<internal::OpenSubdivMesh> subdMesh =
        fauxstd::make_unique<internal::OpenSubdivMesh>(
        SubdivisionMesh::Scheme::CATMULL_CLARK,
        std::move(faceVertexCount), std::move(indices), std::move(vertices),
        LayerAssignmentId(0), std::move(table));
    const scene_rdl2::math::Mat4d world2render;
    subdMesh->setMeshResolution(5);
    subdMesh->tessellate({layer, {}, world2render, false, false, false, nullptr});

    internal::Mesh::TessellatedMesh tessellatedMesh;
    subdMesh->getTessellatedMesh(tessellatedMesh);
    size_t mbSteps = tessellatedMesh.mVertexBufferDesc.size();
    CPPUNIT_ASSERT(mbSteps == 1);
    const float* vertexBuffer = reinterpret_cast<const float*>(
        tessellatedMesh.mVertexBufferDesc[0].mData);
    size_t offset = tessellatedMesh.mVertexBufferDesc[0].mOffset;
    size_t stride = tessellatedMesh.mVertexBufferDesc[0].mStride /
        sizeof(float);
    for (size_t v = 0; v < tessellatedMesh.mVertexCount; ++v) {
        const Vec3f& p = Vec3f(&vertexBuffer[offset + v * stride]);
        const Vec3f& pRef1 = subdMesh->getAttributes()->getVertex(pRef1Key, v);
        const Vec3f& pRef2 = subdMesh->getAttributes()->getVertex(pRef2Key, v);
        const Vec3f& pRef3 = subdMesh->getAttributes()->getVertex(pRef3Key, v);
        CPPUNIT_ASSERT(scene_rdl2::math::isEqual(p, pRef1));
        CPPUNIT_ASSERT(scene_rdl2::math::isEqual(p, pRef2));
        CPPUNIT_ASSERT(scene_rdl2::math::isEqual(p, pRef3));
    }
}

void TestRenderingPrimAttr::testVertex2()
{
    scene_rdl2::rdl2::SceneContext ctx;
    ctx.setDsoPath(ctx.getDsoPath() +
        ":dso/geometry/TestGeometry:dso/material/TestMaterial");
    scene_rdl2::rdl2::Geometry* geom =
        ctx.createSceneObject("TestGeometry", "geom")->asA<scene_rdl2::rdl2::Geometry>();
    scene_rdl2::rdl2::Layer* layer =
        ctx.createSceneObject("Layer", "/seq/shot/layer")->asA<scene_rdl2::rdl2::Layer>();
    scene_rdl2::rdl2::Material* mtl =
        ctx.createSceneObject("TestMaterial", "mtl")->asA<scene_rdl2::rdl2::Material>();
    scene_rdl2::rdl2::LightSet* lgt =
        ctx.createSceneObject("LightSet", "lgt")->asA<scene_rdl2::rdl2::LightSet>();

    layer->beginUpdate();
    // LayerAssignmentID(0) below
    layer->assign(geom, "empty", mtl, lgt, nullptr, nullptr);
    layer->endUpdate();

    // Testing subdivision mesh vertex rate tessellation
    SubdivisionMesh::FaceVertexCount faceVertexCount;
    SubdivisionMesh::VertexBuffer vertices;
    SubdivisionMesh::IndexBuffer indices;
    fillTestSubdMeshData(faceVertexCount, vertices, indices);

    // a relatively trickier test case
    // use combination of Vec2f and float to form a duplication of position data
    PrimitiveAttributeTable table;
    std::vector<Vec2f> pRefAttr1(vertices.size());
    std::vector<float> pRefAttr2(vertices.size());
    for (size_t i = 0; i < vertices.size(); ++i) {
        pRefAttr1[i][0] = vertices(i)[0];
        pRefAttr1[i][1] = vertices(i)[1];
        pRefAttr2[i] = vertices(i)[2];
    }
    TypedAttributeKey<Vec2f> pRefKey1("pRef1");
    TypedAttributeKey<float> pRefKey2("pRef2");
    table.addAttribute(pRefKey1, RATE_VERTEX, std::move(pRefAttr1));
    table.addAttribute(pRefKey2, RATE_VERTEX, std::move(pRefAttr2));

    std::unique_ptr<internal::OpenSubdivMesh> subdMesh =
        fauxstd::make_unique<internal::OpenSubdivMesh>(
        SubdivisionMesh::Scheme::CATMULL_CLARK,
        std::move(faceVertexCount), std::move(indices), std::move(vertices),
        LayerAssignmentId(0), std::move(table));
    const scene_rdl2::math::Mat4d world2render;
    subdMesh->setMeshResolution(5);
    subdMesh->tessellate({layer, {}, world2render, false, false, false, nullptr});

    internal::Mesh::TessellatedMesh tessellatedMesh;
    subdMesh->getTessellatedMesh(tessellatedMesh);
    size_t mbSteps = tessellatedMesh.mVertexBufferDesc.size();
    CPPUNIT_ASSERT(mbSteps == 1);
    const float* vertexBuffer = reinterpret_cast<const float*>(
        tessellatedMesh.mVertexBufferDesc[0].mData);
    size_t offset = tessellatedMesh.mVertexBufferDesc[0].mOffset;
    size_t stride = tessellatedMesh.mVertexBufferDesc[0].mStride /
        sizeof(float);
    for (size_t v = 0; v < tessellatedMesh.mVertexCount; ++v) {
        const Vec3f& p = Vec3f(&vertexBuffer[offset + v * stride]);
        const Vec2f& pXY = subdMesh->getAttributes()->getVertex(pRefKey1, v);
        const float& pZ = subdMesh->getAttributes()->getVertex(pRefKey2, v);
        const Vec3f pRef(pXY[0], pXY[1], pZ);
        CPPUNIT_ASSERT(scene_rdl2::math::isEqual(p, pRef));
    }
}

void TestRenderingPrimAttr::testTime0()
{
    PrimitiveAttributeTable table;
    size_t numVertices= 2;
    table.addAttribute(TestAttributes::sTestFloat0, RATE_VERTEX,
        {{mF0, mF1}, {mF2, mF3}});
    table.addAttribute(TestAttributes::sTestColor0, RATE_VERTEX,
        {{mC0, mC1}, {mC2, mC3}});
    std::unique_ptr<Attributes> attr(
        Attributes::interleave(table, 0, 0, 0,
        std::vector<size_t>(), numVertices));

    CPPUNIT_ASSERT(attr->getVertex(TestAttributes::sTestFloat0, 0, 0) == mF0);
    CPPUNIT_ASSERT(attr->getVertex(TestAttributes::sTestFloat0, 1, 0) == mF1);
    CPPUNIT_ASSERT(attr->getVertex(TestAttributes::sTestFloat0, 0, 1) == mF2);
    CPPUNIT_ASSERT(attr->getVertex(TestAttributes::sTestFloat0, 1, 1) == mF3);
    CPPUNIT_ASSERT(attr->getVertex(TestAttributes::sTestColor0, 0, 0) == mC0);
    CPPUNIT_ASSERT(attr->getVertex(TestAttributes::sTestColor0, 1, 0) == mC1);
    CPPUNIT_ASSERT(attr->getVertex(TestAttributes::sTestColor0, 0, 1) == mC2);
    CPPUNIT_ASSERT(attr->getVertex(TestAttributes::sTestColor0, 1, 1) == mC3);
}

void TestRenderingPrimAttr::testMixing0()
{
    size_t nCurves = 3;
    std::vector<size_t> nSpans(nCurves, mRNG.randomInt() % 10 + 1);
    std::vector<size_t> nFaceVarying(nCurves, 0);
    std::vector<float> varyingFloat;
    std::vector<scene_rdl2::math::Vec2f> faceVaryingVec2;
    std::vector<float> vertexFloat;
    for (size_t c = 0; c < nCurves; ++c) {
        for (size_t s = 0; s < nSpans[c]; ++s) {
            faceVaryingVec2.push_back(mRNG.randomVec2f());
            varyingFloat.push_back(mRNG.randomFloat());
            for (size_t v = 0; v < 3; ++v) {
                vertexFloat.push_back(mRNG.randomFloat());
            }
        }
        varyingFloat.push_back(mRNG.randomFloat());
        vertexFloat.push_back(mRNG.randomFloat());
        faceVaryingVec2.push_back(mRNG.randomVec2f());
        nFaceVarying[c] = nSpans[c] + 1;
    }

    // insert varying, face varying, vertex attributes in a mixing way
    PrimitiveAttributeTable table;
    table.addAttribute(TestAttributes::sTestFloat0, RATE_VARYING,
        std::vector<float>{varyingFloat});
    table.addAttribute(TestAttributes::sTestFloat1, RATE_VERTEX,
        std::vector<float>{vertexFloat});
    table.addAttribute(TestAttributes::sTestVec2f0, RATE_FACE_VARYING,
        std::vector<scene_rdl2::math::Vec2f>(faceVaryingVec2));
    std::unique_ptr<Attributes> attr(
        Attributes::interleave(table, 0, nCurves,
        varyingFloat.size(), nFaceVarying, vertexFloat.size()));

    for (size_t i = 0; i < varyingFloat.size(); ++i) {
        CPPUNIT_ASSERT(attr->getVarying(TestAttributes::sTestFloat0, i) ==
            varyingFloat[i]);
    }

    for (size_t i = 0; i < vertexFloat.size(); ++i) {
        CPPUNIT_ASSERT(attr->getVertex(TestAttributes::sTestFloat1, i) ==
            vertexFloat[i]);
    }

    size_t offset = 0;
    for (size_t c = 0; c < nCurves; ++c) {
        for (size_t s = 0; s <= nSpans[c]; ++s) {
            CPPUNIT_ASSERT(
                attr->getFaceVarying(TestAttributes::sTestVec2f0, c, s) ==
                faceVaryingVec2[offset++]);
        }
    }
}


void TestRenderingPrimAttr::testTransformAttributes0()
{
    // constant and uniform testing
    PrimitiveAttributeTable table;
    size_t numFaces= 3;
    Vec3f pRoot = mRNG.randomVec3f();
    Vec3f tRoot = mRNG.randomVec3f();
    std::vector<Vec3f> nFace(numFaces);
    std::vector<Vec3f> noTransform(numFaces);
    for (size_t f = 0; f < numFaces; ++f) {
        nFace[f] = mRNG.randomVec3f();
        noTransform[f] = mRNG.randomVec3f();
    }
    TypedAttributeKey<Vec3f> pKey = TypedAttributeKey<Vec3f>("position");
    TypedAttributeKey<Vec3f> tKey = TypedAttributeKey<Vec3f>("tangent");
    TypedAttributeKey<Vec3f> nKey = TypedAttributeKey<Vec3f>("normal");
    TypedAttributeKey<Vec3f> noTransformKey =
        TypedAttributeKey<Vec3f>("no_transform");
    table.addAttribute(pKey, RATE_CONSTANT, {pRoot});
    table.addAttribute(tKey, RATE_CONSTANT, {tRoot});
    table.addAttribute(nKey, RATE_UNIFORM, std::vector<Vec3f>{nFace});
    table.addAttribute(noTransformKey, RATE_UNIFORM,
        std::vector<Vec3f>{noTransform});

    std::unique_ptr<Attributes> attr(
        Attributes::interleave(table, 0, numFaces, 0,
        std::vector<size_t>(), 0));

    Vec3f rotateAxis(normalize(mRNG.randomVec3f()));
    scene_rdl2::math::Xform3f xform =
        scene_rdl2::math::Xform3f::translate(mRNG.randomVec3f()) *
        scene_rdl2::math::Xform3f::rotate(rotateAxis, mRNG.randomFloat()) *
        scene_rdl2::math::Xform3f::scale(mRNG.randomVec3f());

    attr->transformAttributes({xform}, 0.0f, 0.0f,
        {{pKey, Vec3Type::POINT},
        {tKey, Vec3Type::VECTOR},
        {nKey, Vec3Type::NORMAL}});

    // Originally, this line called transformPoint(). The result should be the same as the one generated by
    // transformAttributes() just above, because that's calling the same function, transformPoint(). However, that
    // function is inlined, and generates different code in the 2 places, leading to slightly different rounding
    // that causes a 1-ulp difference in this unit test. So here we get around that by having it call out to a
    // function in another module, which simply passes through to transformPoint(). That stops it being inlined here,
    // and brings the results back to being identical.
    CPPUNIT_ASSERT(verifyConstant(attr, pKey, transformPointUtil(xform, pRoot)));

    CPPUNIT_ASSERT(verifyConstant(attr, tKey, transformVector(xform, tRoot)));
    for (size_t f = 0; f < numFaces; ++f) {
        CPPUNIT_ASSERT(verifyUniform(attr, nKey, f,
            transformNormal(xform.inverse(), nFace[f])));
        CPPUNIT_ASSERT(verifyUniform(attr, noTransformKey, f,
            noTransform[f]));
    }
}

void TestRenderingPrimAttr::testTransformAttributes1()
{
    // vertex/varying testing
    PrimitiveAttributeTable table;
    size_t numVaryings = 2;
    size_t numVertices = 4;
    std::vector<Vec3f> p(numVaryings);
    std::vector<Vec3f> t(numVaryings);
    for (size_t v = 0; v < numVaryings; ++v) {
        p[v] = mRNG.randomVec3f();
        t[v] = mRNG.randomVec3f();
    }
    std::vector<Vec3f> n(numVertices);
    std::vector<Vec3f> noTransform(numVertices);
    for (size_t v = 0; v < numVertices; ++v) {
        n[v] = mRNG.randomVec3f();
        noTransform[v] = mRNG.randomVec3f();
    }

    TypedAttributeKey<Vec3f> pKey = TypedAttributeKey<Vec3f>("position");
    TypedAttributeKey<Vec3f> tKey = TypedAttributeKey<Vec3f>("tangent");
    TypedAttributeKey<Vec3f> nKey = TypedAttributeKey<Vec3f>("normal");
    TypedAttributeKey<Vec3f> noTransformKey =
        TypedAttributeKey<Vec3f>("no_transform");

    table.addAttribute(pKey, RATE_VARYING, std::vector<Vec3f>(p));
    table.addAttribute(tKey, RATE_VARYING, std::vector<Vec3f>(t));
    table.addAttribute(nKey, RATE_VERTEX, std::vector<Vec3f>(n));
    table.addAttribute(noTransformKey, RATE_VERTEX,
        std::vector<Vec3f>(noTransform));

    std::unique_ptr<Attributes> attr(
        Attributes::interleave(
        table, 0, 0, numVaryings, std::vector<size_t>(), numVertices));

    const Vec3f rotateAxis(normalize(mRNG.randomVec3f()));
    scene_rdl2::math::Xform3f xform =
        scene_rdl2::math::Xform3f::translate(mRNG.randomVec3f()) *
        scene_rdl2::math::Xform3f::rotate(rotateAxis, mRNG.randomFloat()) *
        scene_rdl2::math::Xform3f::scale(mRNG.randomVec3f());

    attr->transformAttributes({xform}, 0.0f, 0.0f,
        {{pKey, Vec3Type::POINT},
        {tKey, Vec3Type::VECTOR},
        {nKey, Vec3Type::NORMAL}});

    for (size_t v = 0; v < numVaryings; ++v) {
        CPPUNIT_ASSERT(scene_rdl2::math::isEqual(attr->getVarying(pKey, v),
            transformPoint(xform, p[v])));
        CPPUNIT_ASSERT(scene_rdl2::math::isEqual(attr->getVarying(tKey, v),
            transformVector(xform, t[v])));
    }

    for (size_t v = 0; v < numVertices; ++v) {
        CPPUNIT_ASSERT(scene_rdl2::math::isEqual(attr->getVertex(nKey, v),
            transformNormal(xform.inverse(), n[v])));
        CPPUNIT_ASSERT(scene_rdl2::math::isEqual(attr->getVertex(noTransformKey, v),
            noTransform[v]));
    }
}


void TestRenderingPrimAttr::testTransformAttributes2()
{
    // face varying testing
    PrimitiveAttributeTable table;
    size_t numFaces = 2;
    std::vector<size_t> numFaceVaryings = {3, 4};
    std::vector<Vec3f> p;
    std::vector<Vec3f> t;
    std::vector<Vec3f> n;
    std::vector<Vec3f> noTransform;
    for (size_t f = 0; f < numFaces; ++f) {
        for (size_t v = 0; v < numFaceVaryings[f]; ++v) {
            p.push_back(mRNG.randomVec3f());
            t.push_back(mRNG.randomVec3f());
            n.push_back(mRNG.randomVec3f());
            noTransform.push_back(mRNG.randomVec3f());
        }
    }

    TypedAttributeKey<Vec3f> pKey = TypedAttributeKey<Vec3f>("position");
    TypedAttributeKey<Vec3f> tKey = TypedAttributeKey<Vec3f>("tangent");
    TypedAttributeKey<Vec3f> nKey = TypedAttributeKey<Vec3f>("normal");
    TypedAttributeKey<Vec3f> noTransformKey =
        TypedAttributeKey<Vec3f>("no_transform");

    table.addAttribute(pKey, RATE_FACE_VARYING, std::vector<Vec3f>(p));
    table.addAttribute(tKey, RATE_FACE_VARYING, std::vector<Vec3f>(t));
    table.addAttribute(nKey, RATE_FACE_VARYING, std::vector<Vec3f>(n));
    table.addAttribute(noTransformKey, RATE_FACE_VARYING,
        std::vector<Vec3f>(noTransform));

    std::unique_ptr<Attributes> attr(
        Attributes::interleave(
        table, 0, numFaces, 0, numFaceVaryings, 0));

    const Vec3f rotateAxis(normalize(mRNG.randomVec3f()));
    scene_rdl2::math::Xform3f xform =
        scene_rdl2::math::Xform3f::translate(mRNG.randomVec3f()) *
        scene_rdl2::math::Xform3f::rotate(rotateAxis, mRNG.randomFloat()) *
        scene_rdl2::math::Xform3f::scale(mRNG.randomVec3f());

    attr->transformAttributes({xform}, 0.0f, 0.0f,
        {{pKey, Vec3Type::POINT},
        {tKey, Vec3Type::VECTOR},
        {nKey, Vec3Type::NORMAL}});

    size_t offset = 0;
    for (size_t f = 0; f < numFaces; ++f) {
        for (size_t v = 0; v < numFaceVaryings[f]; ++v) {
            CPPUNIT_ASSERT(
                scene_rdl2::math::isEqual(attr->getFaceVarying(pKey, f, v),
                transformPoint(xform, p[offset])));
            CPPUNIT_ASSERT(
                scene_rdl2::math::isEqual(attr->getFaceVarying(tKey, f, v),
                transformVector(xform, t[offset])));
            CPPUNIT_ASSERT(
                scene_rdl2::math::isEqual(attr->getFaceVarying(nKey, f, v),
                transformNormal(xform.inverse(), n[offset])));
            CPPUNIT_ASSERT(
                scene_rdl2::math::isEqual(attr->getFaceVarying(noTransformKey, f, v),
                noTransform[offset]));
            offset++;
        }
    }
}

void TestRenderingPrimAttr::testInstanceAttributes()
{
    PrimitiveAttributeTable table;
    TypedAttributeKey<float> fKey1("test_f1");
    TypedAttributeKey<float> fKey2("test_f2");
    TypedAttributeKey<int> iKey1("test_i1");
    TypedAttributeKey<int> iKey2("test_i2");
    TypedAttributeKey<bool> bKey1("test_b1");
    TypedAttributeKey<bool> bKey2("test_b2");
    TypedAttributeKey<std::string> sKey1("test_s1");
    TypedAttributeKey<std::string> sKey2("test_s2");
    TypedAttributeKey<Vec2f> v2Key("test_vec2");
    TypedAttributeKey<Vec3f> v3Key("test_vec3");
    TypedAttributeKey<scene_rdl2::math::Color> rgbKey("test_rgb");
    TypedAttributeKey<scene_rdl2::math::Color4> rgbaKey("test_rgba");

    table.addAttribute(fKey1, RATE_CONSTANT, std::vector<float>{0.12f});
    table.addAttribute(fKey2, RATE_CONSTANT, std::vector<float>{0.34f});
    table.addAttribute(iKey1, RATE_CONSTANT, std::vector<int>{555});
    table.addAttribute(iKey2, RATE_CONSTANT, std::vector<int>{666});
    table.addAttribute(bKey1, RATE_CONSTANT, std::vector<bool>{true});
    table.addAttribute(bKey2, RATE_CONSTANT, std::vector<bool>{false});
    table.addAttribute(sKey1, RATE_CONSTANT, {"test1"});
    table.addAttribute(sKey2, RATE_CONSTANT, {"test2"});
    table.addAttribute(v2Key, RATE_CONSTANT, {Vec2f(1.23f, 7.89f)});
    table.addAttribute(v3Key, RATE_CONSTANT, {Vec3f(987, 654, 321)});
    table.addAttribute(rgbKey, RATE_CONSTANT,
        {scene_rdl2::math::Color(0.26f, 0.94f, 0.78f)});
    table.addAttribute(rgbaKey, RATE_CONSTANT,
        {scene_rdl2::math::Color4(0.26f, 0.94f, 0.78f, 0.55f)});

    InstanceAttributes attr(std::move(table));
    CPPUNIT_ASSERT(attr.getAttribute(fKey1) == 0.12f);
    CPPUNIT_ASSERT(attr.getAttribute(fKey2) == 0.34f);
    CPPUNIT_ASSERT(attr.getAttribute(iKey1) == 555);
    CPPUNIT_ASSERT(attr.getAttribute(iKey2) == 666);
    CPPUNIT_ASSERT(attr.getAttribute(bKey1) == true);
    CPPUNIT_ASSERT(attr.getAttribute(bKey2) == false);
    CPPUNIT_ASSERT(attr.getAttribute(sKey1) == "test1");
    CPPUNIT_ASSERT(attr.getAttribute(sKey2) == "test2");
    CPPUNIT_ASSERT(attr.getAttribute(v2Key) == Vec2f(1.23f, 7.89f));
    CPPUNIT_ASSERT(attr.getAttribute(v3Key) == Vec3f(987, 654, 321));
    CPPUNIT_ASSERT(attr.getAttribute(rgbKey) ==
        scene_rdl2::math::Color(0.26f, 0.94f, 0.78f));
    CPPUNIT_ASSERT(attr.getAttribute(rgbaKey) ==
        scene_rdl2::math::Color4(0.26f, 0.94f, 0.78f, 0.55f));

    CPPUNIT_ASSERT(!attr.isSupported(TypedAttributeKey<int>("not_exist_attr")));
}

} // namespace unittest
} // namespace geom
} // namespace moonray

