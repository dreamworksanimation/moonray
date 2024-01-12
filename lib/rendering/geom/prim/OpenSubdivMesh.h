// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file OpenSubdivMesh.h
///

#pragma once

#include <moonray/rendering/geom/prim/Mesh.h>
#include <moonray/rendering/geom/prim/SubdMesh.h>

#include <moonray/rendering/bvh/shading/AttributeTable.h>
#include <moonray/rendering/bvh/shading/AttributeKey.h>
#include <moonray/rendering/bvh/shading/Intersection.h>
#include <moonray/rendering/bvh/shading/Xform.h>
#include <moonray/rendering/geom/SubdivisionMesh.h>

namespace moonray {
namespace geom {
namespace internal {

//------------------------------------------------------------------------------

struct LimitSurfaceSample;
struct DisplacementFootprint;
struct SubdQuadTopology;
struct FaceVaryingSeams;
struct SubdTessellationFactor;
class ControlMeshData;
class SubdTessellatedVertexLookup;
class SubdTopologyIdLookup;
class FaceVaryingAttributes;

class OpenSubdivMesh : public SubdMesh
{
public:
    OpenSubdivMesh(SubdivisionMesh::Scheme scheme,
            SubdivisionMesh::FaceVertexCount&& faceVertexCount,
            SubdivisionMesh::IndexBuffer&& indices,
            SubdivisionMesh::VertexBuffer&& vertices,
            LayerAssignmentId&& layerAssignmentId,
            shading::PrimitiveAttributeTable&& primitiveAttributeTable);

    ~OpenSubdivMesh(); 

    virtual size_t getMemory() const override;

    virtual size_t getMotionSamplesCount() const override;

    virtual bool canIntersect() const override
    {
        return false;
    }

    virtual void tessellate(const TessellationParams& tessellationParams) override;

    virtual void getTessellatedMesh(TessellatedMesh& tessMesh) const override;

    virtual void getBakedMesh(BakedMesh& bakedMesh) const override;

    virtual bool bakePosMap(int width, int height, int udim,
            shading::TypedAttributeKey<Vec2f> stKey,
            Vec3fa *posResult, Vec3f *nrmResult) const override;

    virtual SubdivisionMesh::VertexBuffer& getControlVertexBuffer() override;

    virtual size_t getTessellatedMeshFaceCount() const override
    {
        // the final intersection unit is quad
        return mTessellatedIndices.size() / sQuadVertexCount;
    }

    virtual size_t getTessellatedMeshVertexCount() const override
    {
        return mTessellatedVertices.size();
    }

    virtual bool hasAttribute(shading::AttributeKey key) const override;

    virtual void postIntersect(mcrt_common::ThreadLocalState& tls,
            const scene_rdl2::rdl2::Layer* pRdlLayer, const mcrt_common::Ray& ray,
            shading::Intersection& intersection) const override;

    virtual bool computeIntersectCurvature(const mcrt_common::Ray& ray,
            const shading::Intersection& intersection,
            Vec3f& dnds, Vec3f& dndt) const override;

    virtual SubdMesh* copy() const override;

    virtual void setMeshResolution(int meshResolution) override
    {
        // to avoid T junction caused crack, the meshResolution should always
        // be even number. The only exception case is meshResolution <= 1,
        // which we only render control cage and handle as special case
        if (meshResolution <= 1) {
            mMeshResolution = 1;
        } else {
            mMeshResolution = meshResolution % 2 == 0 ?
                meshResolution : meshResolution + 1;
        }
    }

    virtual void setTransform(const shading::XformSamples& xforms,
            float shutterOpenDelta, float shutterCloseDelta) override;

    virtual void setParts(size_t partCount,
            SubdivisionMesh::FaceToPartBuffer&& faceToPart) override
    {
        mPartCount = partCount;
        mFaceToPart = std::move(faceToPart);
    }

    virtual void updateVertexData(const std::vector<float>& vertexData,
            const std::vector<moonray::geom::Mat43> &prim2render) override;

    virtual BBox3f computeAABB() const override;
    virtual BBox3f computeAABBAtTimeStep(int timeStep) const override;

    virtual void getST(int tessFaceId, float u, float v, Vec2f& st) const override;
    virtual int getFaceAssignmentId(int tessFaceId) const override {
        return getControlFaceAssignmentId(mTessellatedToControlFace[tessFaceId]);
    }
    virtual void setRequiredAttributes(int primId, float time, float u, float v,
        float w, bool isFirst, shading::Intersection& intersection) const override;

    virtual scene_rdl2::math::Vec3f getVelocity(int tessFaceId, int vIndex, float time = 0.0f) const override
    {
        int controlFaceId = mTessellatedToControlFace[tessFaceId];
        return getAttribute(shading::StandardAttributes::sVelocity, controlFaceId, tessFaceId, vIndex);
    }

private:
    virtual int getIntersectionAssignmentId(int primID) const override;

    virtual void fillDisplacementAttributes(int tessFaceId, int vIndex,
            shading::Intersection& intersection) const override;

    std::vector<SubdTessellationFactor>
    computeSubdTessellationFactor(const scene_rdl2::rdl2::Layer* pRdlLayer,
            const std::vector<mcrt_common::Frustum>& frustums,
            bool enableDisplacement,
            bool noTessellation) const;

    std::vector<SubdQuadTopology> generateSubdQuadTopology(
            const scene_rdl2::rdl2::Layer* pRdlLayer,
            const SubdTopologyIdLookup& topologyIdLookup,
            const SubdivisionMesh::FaceVertexCount& faceVertexCount,
            const SubdivisionMesh::IndexBuffer& faceVaryingIndices,
            bool noTessellation) const;

    void generateIndexBufferAndSurfaceSamples(
            const std::vector<SubdQuadTopology>& quadTopologies,
            const SubdTessellatedVertexLookup& tessellatedVertexLookup,
            bool noTessellation,
            SubdivisionMesh::IndexBuffer& indices,
            std::vector<LimitSurfaceSample>& limitSurfaceSamples,
            std::vector<int>* tessellatedToControlFace = nullptr) const;

    void generateControlMeshIndexBufferAndSurfaceSamples(
            const std::vector<SubdQuadTopology>& quadTopologies,
            const SubdTessellatedVertexLookup& tessellatedVertexLookup,
            SubdivisionMesh::IndexBuffer& indices,
            std::vector<LimitSurfaceSample>& limitSurfaceSamples,
            std::vector<int>* tessellatedToControlFace) const;

    int getControlFaceAssignmentId(int controlFaceId) const
    {
        int assignmentId =
            mLayerAssignmentId.getType() == LayerAssignmentId::Type::CONSTANT ?
            mLayerAssignmentId.getConstId() :
            mLayerAssignmentId.getVaryingId()[controlFaceId];
        return assignmentId;
    }

    void computeAttributesDerivatives(const shading::AttributeTable* table,
            size_t vid1, size_t vid2, size_t vid3, float time,
            shading::Intersection& intersection) const;

    void displaceMesh(const scene_rdl2::rdl2::Layer *pRdlLayer,
            const std::vector<LimitSurfaceSample>& limitSurfaceSamples,
            const std::vector<DisplacementFootprint>& displacementFootprints,
            const SubdTessellatedVertexLookup& tessellatedVertexLookup,
            const FaceVaryingSeams& faceVaryingSeams,
            const mcrt_common::Frustum& frustum,
            const scene_rdl2::math::Mat4d& world2render);

    template <typename T>
    T getAttribute(const shading::TypedAttributeKey<T>& key,
            int controlFaceId, int tessFaceId, int vIndex) const;

    std::unique_ptr<BakedAttribute> getBakedAttribute(const shading::AttributeKey& key) const;
    std::unique_ptr<BakedAttribute> getFVBakedAttribute(const shading::AttributeKey& key) const;

    template <typename T>
    void* getBakedAttributeData(const shading::TypedAttributeKey<T>& key,
                                size_t vertexCount, size_t faceCount, size_t timeSamples,
                                size_t& numElements) const;

    template <typename T>
    void* getFVBakedAttributeData(const shading::TypedAttributeKey<T>& key,
                                  size_t& numElements) const;

private:
    // tessellated vertex/index/normal/st/dpds/dpdt buffer
    SubdivisionMesh::VertexBuffer mTessellatedVertices;
    SubdivisionMesh::IndexBuffer mTessellatedIndices;
    VertexBuffer<Vec3f, InterleavedTraits> mSurfaceNormal;
    VertexBuffer<Vec2f, InterleavedTraits> mSurfaceSt;
    VertexBuffer<Vec3f, InterleavedTraits> mSurfaceDpds;
    VertexBuffer<Vec3f, InterleavedTraits> mSurfaceDpdt;
    // mapping from tessellated face id to control face id
    std::vector<int> mTessellatedToControlFace;
    std::unique_ptr<FaceVaryingAttributes> mFaceVaryingAttributes;

    size_t mPartCount;
    SubdivisionMesh::FaceToPartBuffer mFaceToPart;
};

//------------------------------------------------------------------------------

} // namespace internal
} // namespace geom
} // namespace rendering


