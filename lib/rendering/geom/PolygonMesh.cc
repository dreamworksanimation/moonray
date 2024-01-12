// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file PolygonMesh.cc
/// $Id$
///

#include "PolygonMesh.h"

#include <moonray/rendering/geom/PrimitiveVisitor.h>
#include <moonray/rendering/geom/ProceduralContext.h>
#include <moonray/rendering/geom/State.h>

#include <scene_rdl2/render/util/stdmemory.h>
#include <moonray/rendering/geom/prim/TriMesh.h>
#include <moonray/rendering/geom/prim/QuadMesh.h>
#include <numeric>
#include <algorithm>

namespace moonray {
namespace geom {

struct PolygonMesh::Impl
{
    explicit Impl(internal::PolyMesh* polyMesh) : mPolyMesh(polyMesh) {}
    std::unique_ptr<internal::PolyMesh> mPolyMesh;
};

PolygonMesh::PolygonMesh(FaceVertexCount&& faceVertexCount,
        IndexBuffer&& indices,
        VertexBuffer&& vertices,
        LayerAssignmentId&& layerAssignmentId,
        shading::PrimitiveAttributeTable&& primitiveAttributeTable)
{
    // n-gon to triangle
    // nTriangle = n - 2
    // e.g. pentagon(5) -> 3 triangles; quad(4) -> 2 triangles
    size_t outputTriFaceCount = std::accumulate(
            faceVertexCount.begin(), faceVertexCount.end(), 0,
            [](const size_t& faceCount, size_t curFaceVerts) {
                return faceCount + (curFaceVerts - 2);
            });
    // n-gon to quad
    // nQuad = (n - 1) / 2
    // e.g. triangle(3) -> 1 quad
    //      pentagon(5) -> 1 quad + 1 triangle -> 2 quads
    //      hexagon (6) -> 2 quads
    //      heptagon(7) -> 2 quads+ 1 triangle -> 3 quads
    size_t outputQuadFaceCount = std::accumulate(
            faceVertexCount.begin(), faceVertexCount.end(), 0,
            [](const size_t& faceCount, size_t curFaceVerts) {
                return faceCount + ((curFaceVerts - 1) >> 1);
            });

    const size_t maxFVCount = *std::max_element(faceVertexCount.begin(),
                                                faceVertexCount.end());
    // If the maximum face vertex count is more than four we always create a
    // tri mesh for now since it can deal with concave ngons.   If not, we
    // create a quad mesh if it will generate less polygons than a trimesh.
    if (maxFVCount <= 4 && outputQuadFaceCount * 4 < outputTriFaceCount * 3) {
        mImpl = fauxstd::make_unique<Impl>(new internal::QuadMesh(outputQuadFaceCount,
                std::move(faceVertexCount), std::move(indices), std::move(vertices),
                std::move(layerAssignmentId), std::move(primitiveAttributeTable)));
    } else {
        mImpl = fauxstd::make_unique<Impl>(new internal::TriMesh(outputTriFaceCount,
                std::move(faceVertexCount), std::move(indices), std::move(vertices),
                std::move(layerAssignmentId), std::move(primitiveAttributeTable)));
    }
}

PolygonMesh::~PolygonMesh() = default;

void
PolygonMesh::accept(PrimitiveVisitor& v)
{
    v.visitPolygonMesh(*this);
}

Primitive::size_type
PolygonMesh::getFaceCount() const
{
    return mImpl->mPolyMesh->getTessellatedMeshFaceCount();
}

Primitive::size_type
PolygonMesh::getMemory() const
{
    return sizeof(PolygonMesh) + mImpl->mPolyMesh->getMemory();
}

Primitive::size_type
PolygonMesh::getMotionSamplesCount() const
{
    return mImpl->mPolyMesh->getMotionSamplesCount();
}

PolygonMesh::VertexBuffer&
PolygonMesh::getVertexBuffer() {
    return mImpl->mPolyMesh->getVertexBuffer();
}

void
PolygonMesh::setName(const std::string& name)
{
    mImpl->mPolyMesh->setName(name);
}

void
PolygonMesh::setMeshResolution(int meshResolution)
{
    mImpl->mPolyMesh->setMeshResolution(meshResolution);
}

void
PolygonMesh::setAdaptiveError(float adaptiveError)
{
    mImpl->mPolyMesh->setAdaptiveError(adaptiveError);
}

const std::string&
PolygonMesh::getName() const
{
    return mImpl->mPolyMesh->getName();
}

void
PolygonMesh::setParts(size_t partCount, FaceToPartBuffer&& faceToPart)
{
    mImpl->mPolyMesh->setParts(partCount, std::move(faceToPart));
}

void
PolygonMesh::setIsSingleSided(bool isSingleSided)
{
    mImpl->mPolyMesh->setIsSingleSided(isSingleSided);
}

bool
PolygonMesh::getIsSingleSided() const
{
    return mImpl->mPolyMesh->getIsSingleSided();
}

void
PolygonMesh::setIsNormalReversed(bool isNormalReversed)
{
    mImpl->mPolyMesh->setIsNormalReversed(isNormalReversed);
}

bool
PolygonMesh::getIsNormalReversed() const
{
    return mImpl->mPolyMesh->getIsNormalReversed();
}

void
PolygonMesh::setIsOrientationReversed(bool reverseOrientation)
{
    mImpl->mPolyMesh->setIsOrientationReversed(reverseOrientation);
}

bool
PolygonMesh::getIsOrientationReversed() const
{
    return mImpl->mPolyMesh->getIsOrientationReversed();
}

void
PolygonMesh::setSmoothNormal(bool smoothNormal)
{
    mImpl->mPolyMesh->setSmoothNormal(smoothNormal);
}

bool
PolygonMesh::getSmoothNormal() const
{
    return mImpl->mPolyMesh->getSmoothNormal();
}

void
PolygonMesh::setCurvedMotionBlurSampleCount(int count)
{
    mImpl->mPolyMesh->setCurvedMotionBlurSampleCount(count);
}

void
PolygonMesh::transformPrimitive(
        const MotionBlurParams& motionBlurParams,
        const shading::XformSamples& prim2render)
{
    size_t motionSamplesCount = getMotionSamplesCount();
    shading::XformSamples p2r = prim2render;
    if (motionSamplesCount > 1 && prim2render.size() == 1) {
        p2r.resize(motionSamplesCount, prim2render[0]);
    }
    const shading::PrimitiveAttributeTable* primAttrTab = mImpl->mPolyMesh->getPrimitiveAttributeTable();
    transformVertexBuffer(mImpl->mPolyMesh->getVertexBuffer(), p2r, motionBlurParams,
                          mImpl->mPolyMesh->getMotionBlurType(), mImpl->mPolyMesh->getCurvedMotionBlurSampleCount(),
                          primAttrTab);
    float shutterOpenDelta, shutterCloseDelta;
    motionBlurParams.getMotionBlurDelta(shutterOpenDelta, shutterCloseDelta);
    mImpl->mPolyMesh->setTransform(p2r, shutterOpenDelta, shutterCloseDelta);
}

internal::Primitive*
PolygonMesh::getPrimitiveImpl()
{
    return mImpl->mPolyMesh.get();
}

void
PolygonMesh::updateVertexData(const std::vector<float>& vertexData,
        const shading::XformSamples& prim2render)
{
    if (vertexData.size() > 0) {
        mImpl->mPolyMesh->updateVertexData(vertexData, prim2render);
    }
}

void
PolygonMesh::recomputeVertexNormals()
{
    mImpl->mPolyMesh->recomputeVertexNormals();
}

Primitive::size_type
PolygonMesh::getVertexCount() const
{
    return mImpl->mPolyMesh->getVertexCount();
}

} // namespace geom
} // namespace moonray

