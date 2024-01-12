// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file PolyMesh.h
/// $Id$
///
#pragma once

#include <moonray/rendering/geom/prim/Mesh.h>
#include <moonray/rendering/geom/prim/PolyMeshCalcNv.h>

#include <moonray/rendering/bvh/shading/AttributeKey.h>
#include <moonray/rendering/bvh/shading/Attributes.h>
#include <moonray/rendering/bvh/shading/PrimitiveAttribute.h>
#include <moonray/rendering/bvh/shading/Xform.h>
#include <moonray/rendering/geom/PolygonMesh.h>

namespace moonray {
namespace geom {
namespace internal {

struct PolyTessellationFactor;
struct PolyFaceTopology;
class PolyTopologyIdLookup;
class PolyTessellatedVertexLookup;

class PolyMeshData
{
public:
    PolyMeshData(shading::PrimitiveAttributeTable&& primitiveAttributeTable,
            size_t estiFaceCount,
            PolygonMesh::FaceVertexCount&& faceVertexCount):
        mPrimitiveAttributeTable(std::move(primitiveAttributeTable)),
        mShutterOpenDelta(0),
        mShutterCloseDelta(0),
        mEstiFaceCount(estiFaceCount),
        mFaceVertexCount(faceVertexCount)
    {}

    shading::PrimitiveAttributeTable mPrimitiveAttributeTable;
    shading::XformSamples mXforms;
    float mShutterOpenDelta;
    float mShutterCloseDelta;
    size_t mEstiFaceCount;
    PolygonMesh::FaceVertexCount mFaceVertexCount;
};

class PolyMesh : public Mesh
{
public:
    /// Constructor / Destructor
    PolyMesh(size_t estiFaceCount,
            PolygonMesh::FaceVertexCount&& faceVertexCount,
            PolygonMesh::IndexBuffer&& indices,
            PolygonMesh::VertexBuffer&& vertices,
            LayerAssignmentId&& layerAssignmentId,
            shading::PrimitiveAttributeTable&& primitiveAttributeTable):
        Mesh(std::move(layerAssignmentId)),
        mPartCount(0),
        mIndices(std::move(indices)),
        mVertices(std::move(vertices)),
        mIsTessellated(false),
        mPolyMeshData(std::unique_ptr<PolyMeshData>(new PolyMeshData(
                      std::move(primitiveAttributeTable), estiFaceCount,
                      std::move(faceVertexCount)))),
        mSmoothNormal(true),
        mCurvedMotionBlurSampleCount(0)
    {
    }

    virtual size_t getMemory() const override;

    virtual size_t getMotionSamplesCount() const override
    {
        return mVertices.get_time_steps();
    }

    virtual bool canIntersect() const override
    {
        return false;
    }

    virtual void tessellate(const TessellationParams& tessellationParams) override;

    virtual size_t getTessellatedMeshFaceCount() const override;

    virtual void getTessellatedMesh(TessellatedMesh& tessMesh) const override;

    virtual void getBakedMesh(BakedMesh &bakedMesh) const override;

    virtual int getIntersectionAssignmentId(int primID) const override;

    virtual void updateVertexData(const std::vector<float>& vertexData,
            const shading::XformSamples& prim2render) override;

    virtual BBox3f computeAABB() const override;
    virtual BBox3f computeAABBAtTimeStep(int timeStep) const override;

    const PolygonMesh::VertexBuffer& getVertexBuffer() const
    {
        return mVertices;
    }

    PolygonMesh::VertexBuffer& getVertexBuffer()
    {
        return mVertices;
    }

    size_t getVertexCount() const
    {
        return mVertices.size();
    }

    void setParts(size_t partCount, PolygonMesh::FaceToPartBuffer&& faceToPart)
    {
        mPartCount = partCount;
        mFaceToPart = std::move(faceToPart);
    }

    void setTransform(const shading::XformSamples& xforms,
            float shutterOpenDelta, float shutterCloseDelta)
    {
        mPolyMeshData->mXforms = xforms;
        mPolyMeshData->mShutterOpenDelta = shutterOpenDelta;
        mPolyMeshData->mShutterCloseDelta = shutterCloseDelta;

        // Needed for volume shader
        // We'll set this to the world2render xform in tessellate
        // if this is a shared primitive
        mPrimToRender = scene_rdl2::math::Mat4f(xforms[0]);
    }

    void setSmoothNormal(bool smoothNormal)
    {
        mSmoothNormal = smoothNormal;
    }

    bool getSmoothNormal() const
    {
        return mSmoothNormal;
    }

    virtual int getFaceAssignmentId(int tessFaceId) const override
    {
        int baseFaceId = getBaseFaceId(tessFaceId);
        int assignmentId =
            mLayerAssignmentId.getType() == LayerAssignmentId::Type::CONSTANT ?
            mLayerAssignmentId.getConstId() :
            mLayerAssignmentId.getVaryingId()[baseFaceId];
        return assignmentId;
    }

    void setupRecomputeVertexNormals(bool fixInvalid);

    void recomputeVertexNormals();

    /// Copy is disabled
    PolyMesh(const PolyMesh &other) = delete;

    const PolyMesh &operator=(const PolyMesh &other) = delete;

    const shading::PrimitiveAttributeTable* getPrimitiveAttributeTable() const
    {
        MNRY_ASSERT(mPolyMeshData != nullptr);
        return &mPolyMeshData->mPrimitiveAttributeTable;
    }

    void setCurvedMotionBlurSampleCount(int count)
    {
        mCurvedMotionBlurSampleCount = count;
    }

    uint32_t getCurvedMotionBlurSampleCount() const
    {
        return mCurvedMotionBlurSampleCount;
    }

    scene_rdl2::rdl2::MotionBlurType getMotionBlurType() const
    {
        return mMotionBlurType;
    }

protected:

    // SurfaceSample contains the ingredients to generate a final
    // surface sample point
    struct SurfaceSample
    {
        int mFaceId;
        Vec2f mUv;
    };

    int getBaseFaceId(int tessellatedFaceId) const
    {
        return mIsTessellated ?
            mTessellatedToBaseFace[tessellatedFaceId] : tessellatedFaceId;
    }

    virtual MeshIndexType getBaseFaceType() const = 0;

    virtual void splitNGons(size_t outputFaceCount,
                            const PolygonMesh::VertexBuffer& vertices,
                            PolygonMesh::FaceToPartBuffer& faceToPart,
                            const PolygonMesh::FaceVertexCount& faceVertexCount,
                            PolygonMesh::IndexBuffer& indices,
                            LayerAssignmentId& layerAssignmentId,
                            shading::PrimitiveAttributeTable& primitiveAttributeTable) = 0;

    virtual void generateIndexBufferAndSurfaceSamples(
            const std::vector<PolyFaceTopology>& quadTopologies,
            const PolyTessellatedVertexLookup& tessellatedVertexLookup,
            PolygonMesh::IndexBuffer& indices,
            std::vector<PolyMesh::SurfaceSample>& surfaceSamples,
            std::vector<int>& tessellatedToBaseFace,
            std::vector<Vec2f>* faceVaryingUv) const = 0;

    virtual PolygonMesh::VertexBuffer generateVertexBuffer(
            const PolygonMesh::VertexBuffer& baseVertices,
            const PolygonMesh::IndexBuffer& baseIndices,
            const std::vector<PolyMesh::SurfaceSample>& surfaceSamples) const = 0;

    // whether we should tessellate the input mesh
    bool shouldTessellate(bool enableDisplacement, const scene_rdl2::rdl2::Layer *pRdlLayer) const;

    std::vector<PolyTessellationFactor> computeTessellationFactor(
            const scene_rdl2::rdl2::Layer *pRdlLayer,
            const std::vector<mcrt_common::Frustum>& frustums,
            const PolyTopologyIdLookup& topologyIdLookup) const;

    std::vector<PolyFaceTopology> generatePolyFaceTopology(
            const PolyTopologyIdLookup& topologyIdLookup) const;

    void initAttributesAndDisplace(const scene_rdl2::rdl2::Layer *pRdlLayer,
            size_t baseFaceCount, size_t varyingsCount, bool enableDisplacement,
            bool realtimeMode, bool isBaking,
            const scene_rdl2::math::Mat4d& world2render);

    template <typename T> T
    getFaceVaryingAttribute(shading::TypedAttributeKey<T> key, int fid, int vIndex,
            float time = 0.0f) const
    {
        const shading::Attributes* attributes = getAttributes();
        MNRY_ASSERT(attributes->isSupported(key) &&
            attributes->getRate(key) == shading::RATE_FACE_VARYING);
        if (attributes->getTimeSampleCount(key) > 1 && !scene_rdl2::math::isZero(time)) {
            return attributes->getMotionBlurFaceVarying(key, fid, vIndex, time);
        } else {
            return attributes->getFaceVarying(key, fid, vIndex);
        }
    }

    template<typename T> void
    computeAttributeDerivatives(shading::TypedAttributeKey<T> key,
            uint32_t fid, uint32_t fvid1, uint32_t fvid2, uint32_t fvid3,
            uint32_t vid1, uint32_t vid2, uint32_t vid3, float time,
            const std::array<float, 4>& invA,
            shading::Intersection& intersection) const
    {
        switch (getAttributes()->getRate(key)) {
        case shading::RATE_VARYING:
            computeVaryingAttributeDerivatives(key, vid1, vid2, vid3,
                time, invA, intersection);
            break;
        case shading::RATE_FACE_VARYING:
            computeFaceVaryingAttributeDerivatives(key, fid,
                fvid1, fvid2, fvid3, time, invA, intersection);
            break;
        case shading::RATE_VERTEX:
            computeVertexAttributeDerivatives(key, vid1, vid2, vid3,
                time, invA, intersection);
            break;
        default:
            break;
        }
    }

    void displaceMesh(const scene_rdl2::rdl2::Layer *pRdlLayer,
                      const scene_rdl2::math::Mat4d& world2render);

    // helper for displaceMesh to query:
    // for a tessellated vertex with face id tessFaceId and
    // local index tessFaceVIndex, what is the tessellated vertex id,
    // st coordinate of itself and its neighbors
    virtual void getNeighborVertices(int baseFaceId, int tessFaceId,
            int tessFaceVIndex, int& vid, int& vid1, int& vid2, int& vid3,
            Vec2f& st, Vec2f& st1, Vec2f& st2, Vec2f& st3) const = 0;

    std::unique_ptr<BakedAttribute> getBakedAttribute(const shading::AttributeKey& key) const;

    template <typename T>
    void* getBakedAttributeData(const shading::TypedAttributeKey<T>& key,
                                size_t& numElements, shading::AttributeRate &newRate) const;

protected:
    size_t mPartCount;
    PolygonMesh::FaceToPartBuffer mFaceToPart;
    // mBaseIndices is empty if no tessellation for displacement happens
    // it is used to retrieve varying rate vertex id
    PolygonMesh::IndexBuffer mBaseIndices;
    PolygonMesh::IndexBuffer mIndices;
    PolygonMesh::VertexBuffer mVertices;
    // mapping from tessellated face id to base mesh face id
    std::vector<int> mTessellatedToBaseFace;
    // mapping from tessellated vertices to base mesh face surface uv
    // use this for varying/facevarying interpolation
    std::vector<Vec2f> mFaceVaryingUv;
    // whether the mesh is tessellated for displacement
    bool mIsTessellated;
    // utilities for calculating smooth normal
    std::unique_ptr<PolyMeshCalcNv> mPolyMeshCalcNv;
    std::unique_ptr<PolyMeshData> mPolyMeshData;
    // whether calculate smooth shading normal
    bool mSmoothNormal;
    uint32_t mCurvedMotionBlurSampleCount;
    scene_rdl2::rdl2::MotionBlurType mMotionBlurType;
};

//----------------------------------------------------------------------------

} // namespace internal
} // namespace geom
} // namespace moonray

