// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file SubdivisionMesh.cc
/// $Id$
///

#include "SubdivisionMesh.h"

#include <moonray/rendering/geom/PrimitiveVisitor.h>
#include <moonray/rendering/geom/ProceduralContext.h>
#include <moonray/rendering/geom/State.h>

#include <moonray/rendering/geom/prim/OpenSubdivMesh.h>
#include <scene_rdl2/render/util/stdmemory.h>

namespace moonray {
namespace geom {

namespace internal {
class SubdMesh;
}

struct SubdivisionMesh::Impl
{
    explicit Impl(internal::SubdMesh* subdMesh) : mSubdMesh(subdMesh) {}
    std::unique_ptr<internal::SubdMesh> mSubdMesh;
};

SubdivisionMesh::SubdivisionMesh(Impl* impl) :
    mImpl(fauxstd::make_unique<Impl>(impl->mSubdMesh->copy()))
{}

SubdivisionMesh::~SubdivisionMesh() = default;

SubdivisionMesh::SubdivisionMesh(
        Scheme scheme,
        FaceVertexCount&& faceVertexCount,
        IndexBuffer&& indices,
        VertexBuffer&& vertices,
        LayerAssignmentId&& layerAssignmentId,
        shading::PrimitiveAttributeTable&& primitiveAttributeTable)
{
    mImpl = fauxstd::make_unique<Impl>(new internal::OpenSubdivMesh(
        scheme,
        std::move(faceVertexCount), std::move(indices),
        std::move(vertices), std::move(layerAssignmentId),
        std::move(primitiveAttributeTable)));
}

std::unique_ptr<SubdivisionMesh>
SubdivisionMesh::copy() const
{
    return fauxstd::make_unique<SubdivisionMesh>(mImpl.get());
}

void
SubdivisionMesh::accept(PrimitiveVisitor& v)
{
    v.visitSubdivisionMesh(*this);
}

Primitive::size_type
SubdivisionMesh::getSubdivideFaceCount() const
{
    return mImpl->mSubdMesh->getTessellatedMeshFaceCount();
}

Primitive::size_type
SubdivisionMesh::getSubdivideVertexCount() const
{
    return mImpl->mSubdMesh->getTessellatedMeshVertexCount();
}


Primitive::size_type
SubdivisionMesh::getMemory() const
{
    return sizeof(SubdivisionMesh) + mImpl->mSubdMesh->getMemory();
}

Primitive::size_type
SubdivisionMesh::getMotionSamplesCount() const
{
    return mImpl->mSubdMesh->getMotionSamplesCount();
}

SubdivisionMesh::VertexBuffer&
SubdivisionMesh::getControlVertexBuffer() {
    return mImpl->mSubdMesh->getControlVertexBuffer();
}

void
SubdivisionMesh::setName(const std::string& name)
{
    mImpl->mSubdMesh->setName(name);
}

void
SubdivisionMesh::setMeshResolution(int meshResolution)
{
    mImpl->mSubdMesh->setMeshResolution(meshResolution);
}

void
SubdivisionMesh::setAdaptiveError(float adaptiveError)
{
    mImpl->mSubdMesh->setAdaptiveError(adaptiveError);
}

const std::string&
SubdivisionMesh::getName() const
{
    return mImpl->mSubdMesh->getName();
}

void
SubdivisionMesh::setSubdBoundaryInterpolation(BoundaryInterpolation val)
{
    mImpl->mSubdMesh->setSubdBoundaryInterpolation(val);
}

SubdivisionMesh::BoundaryInterpolation
SubdivisionMesh::getSubdBoundaryInterpolation() const
{
    return mImpl->mSubdMesh->getSubdBoundaryInterpolation();
}

void
SubdivisionMesh::setSubdFVarLinearInterpolation(FVarLinearInterpolation val)
{
    mImpl->mSubdMesh->setSubdFVarLinearInterpolation(val);
}

SubdivisionMesh::FVarLinearInterpolation
SubdivisionMesh::getSubdFVarLinearInterpolation() const
{
    return mImpl->mSubdMesh->getSubdFVarLinearInterpolation();
}

void
SubdivisionMesh::setSubdCreases(IndexBuffer&&     creaseIndices,
                                SharpnessBuffer&& creaseSharpnesses)
{
    mImpl->mSubdMesh->setSubdCreases(std::move(creaseIndices),
                                     std::move(creaseSharpnesses));
}

bool
SubdivisionMesh::hasSubdCreases() const
{
    return mImpl->mSubdMesh->hasSubdCreases();
}

void
SubdivisionMesh::setSubdCorners(IndexBuffer&&     cornerIndices,
                                SharpnessBuffer&& cornerSharpnesses)
{
    mImpl->mSubdMesh->setSubdCorners(std::move(cornerIndices),
                                     std::move(cornerSharpnesses));
}

bool
SubdivisionMesh::hasSubdCorners() const
{
    return mImpl->mSubdMesh->hasSubdCorners();
}

void
SubdivisionMesh::setSubdHoles(IndexBuffer&& holeIndices)
{
    mImpl->mSubdMesh->setSubdHoles(std::move(holeIndices));
}

bool
SubdivisionMesh::hasSubdHoles() const
{
    return mImpl->mSubdMesh->hasSubdHoles();
}

void
SubdivisionMesh::setParts(size_t partCount, FaceToPartBuffer&& faceToPart)
{
    mImpl->mSubdMesh->setParts(partCount, std::move(faceToPart));
}

void
SubdivisionMesh::setIsSingleSided(bool isSingleSided)
{
    mImpl->mSubdMesh->setIsSingleSided(isSingleSided);
}

bool
SubdivisionMesh::getIsSingleSided() const
{
    return mImpl->mSubdMesh->getIsSingleSided();
}

void
SubdivisionMesh::setIsNormalReversed(bool isNormalReversed)
{
    mImpl->mSubdMesh->setIsNormalReversed(isNormalReversed);
}

bool
SubdivisionMesh::getIsNormalReversed() const
{
    return mImpl->mSubdMesh->getIsNormalReversed();
}

void
SubdivisionMesh::setIsOrientationReversed(bool reverseOrientation)
{
    mImpl->mSubdMesh->setIsOrientationReversed(reverseOrientation);
}

bool
SubdivisionMesh::getIsOrientationReversed() const
{
    return mImpl->mSubdMesh->getIsOrientationReversed();
}

void
SubdivisionMesh::setCurvedMotionBlurSampleCount(int count)
{
    mImpl->mSubdMesh->setCurvedMotionBlurSampleCount(count);
}

void
SubdivisionMesh::transformPrimitive(
        const MotionBlurParams& motionBlurParams,
        const shading::XformSamples& prim2render)
{
    size_t motionSamplesCount = getMotionSamplesCount();
    shading::XformSamples p2r = prim2render;
    if (motionSamplesCount > 1 && prim2render.size() == 1) {
        p2r.resize(motionSamplesCount, prim2render[0]);
    }
    const shading::PrimitiveAttributeTable* primAttrTab = &mImpl->mSubdMesh->getPrimitiveAttributeTable();
    transformVertexBuffer(mImpl->mSubdMesh->getControlVertexBuffer(), p2r, motionBlurParams,
                          mImpl->mSubdMesh->getMotionBlurType(), mImpl->mSubdMesh->getCurvedMotionBlurSampleCount(),
                          primAttrTab);
    float shutterOpenDelta, shutterCloseDelta;
    motionBlurParams.getMotionBlurDelta(shutterOpenDelta, shutterCloseDelta);
    mImpl->mSubdMesh->setTransform(p2r, shutterOpenDelta, shutterCloseDelta);
}

internal::Primitive*
SubdivisionMesh::getPrimitiveImpl()
{
    return mImpl->mSubdMesh.get();
}

void
SubdivisionMesh::updateVertexData(const std::vector<float>& vertexData,
        const shading::XformSamples& prim2render)
{
    if (vertexData.size() > 0) {
        mImpl->mSubdMesh->updateVertexData(vertexData, prim2render);
    }
}

} // namespace geom
} // namespace moonray

