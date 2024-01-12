// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file QuadMesh.h
/// $Id$
///

#pragma once

#include <moonray/rendering/geom/prim/Mesh.h>
#include <moonray/rendering/geom/prim/PolyMesh.h>

#include <moonray/rendering/bvh/shading/AttributeKey.h>
#include <moonray/rendering/bvh/shading/AttributeTable.h>
#include <moonray/rendering/geom/Api.h>
#include <moonray/rendering/geom/PolygonMesh.h>

namespace moonray {
namespace geom {
namespace internal {

class QuadMesh : public PolyMesh
{
public:
    QuadMesh(size_t estiFaceCount,
            PolygonMesh::FaceVertexCount&& faceVertexCount,
            PolygonMesh::IndexBuffer&& indices,
            PolygonMesh::VertexBuffer&& vertices,
            LayerAssignmentId&& layerAssignmentId,
            shading::PrimitiveAttributeTable&& primitiveAttributeTable);

    virtual bool bakePosMap(int width, int height, int udim,
            shading::TypedAttributeKey<Vec2f> stKey,
            Vec3fa *posResult, Vec3f *nrmResult) const override;

    virtual void postIntersect(mcrt_common::ThreadLocalState& tls,
            const scene_rdl2::rdl2::Layer* pRdlLayer, const mcrt_common::Ray& ray,
            shading::Intersection& intersection) const override;

    virtual bool computeIntersectCurvature(const mcrt_common::Ray& ray,
            const shading::Intersection& intersection,
            Vec3f& dnds, Vec3f& dndt) const override;

    virtual void getST(int tessFaceId, float u, float v, Vec2f& st) const override;
    virtual void setRequiredAttributes(int primId, float time, float u, float v,
        float w, bool isFirst, shading::Intersection& intersection) const override;

    virtual scene_rdl2::math::Vec3f getVelocity(int tessFaceId, int vIndex, float time = 0.0f) const override
    {
        int baseFaceId = getBaseFaceId(tessFaceId);
        return getAttribute(shading::StandardAttributes::sVelocity, baseFaceId, tessFaceId, vIndex, time);
    }

private:
    virtual void getNeighborVertices(int baseFaceId, int tessFaceId,
            int tessFaceVIndex, int& vid, int& vid1, int& vid2, int& vid3,
            Vec2f& st, Vec2f& st1, Vec2f& st2, Vec2f& st3) const override;

    virtual MeshIndexType getBaseFaceType() const override {
        return MeshIndexType::QUAD;
    }

    // quadrangulate input data if there is any non quadrilateral exists
    virtual void splitNGons(size_t outputFaceCount,
                            const PolygonMesh::VertexBuffer& vertices,
                            PolygonMesh::FaceToPartBuffer& faceToPart,
                            const PolygonMesh::FaceVertexCount& faceVertexCount,
                            PolygonMesh::IndexBuffer& indices,
                            LayerAssignmentId& layerAssignmentId,
                            shading::PrimitiveAttributeTable& primitiveAttributeTable) override;

    virtual void generateIndexBufferAndSurfaceSamples(
            const std::vector<PolyFaceTopology>& quadTopologies,
            const PolyTessellatedVertexLookup& tessellatedVertexLookup,
            PolygonMesh::IndexBuffer& indices,
            std::vector<PolyMesh::SurfaceSample>& surfaceSamples,
            std::vector<int>& tessellatedToBaseFace,
            std::vector<Vec2f>* faceVaryingUv) const override;

    virtual PolygonMesh::VertexBuffer generateVertexBuffer(
            const PolygonMesh::VertexBuffer& baseVertices,
            const PolygonMesh::IndexBuffer& baseIndices,
            const std::vector<PolyMesh::SurfaceSample>& surfaceSamples) const override;

    virtual void fillDisplacementAttributes(int tessFaceId, int vIndex,
            shading::Intersection& intersection) const override;

    void computeAttributesDerivatives(const shading::AttributeTable* table,
            const Vec2f& st1, const Vec2f& st2, const Vec2f& st3,
            int baseFaceId, int tessFaceId, float time, bool isFirst,
            shading::Intersection& intersection) const;

    template <typename T>
    T getAttribute(const shading::TypedAttributeKey<T>& key,
            int baseFaceId, int tessFaceId, int vIndex,
            float time = 0.0f) const;

    // isFirst == true means 1st triangle in quad <0, 1, 3>
    // isFirst == false means 2nd triangle in quad <2, 3, 1>
    template <typename T>
    bool getQuadAttributes(const shading::TypedAttributeKey<T>& key,
            int baseFaceId, int tessFaceId, T& v1, T& v2, T& v3,
            float time, bool isFirst) const;

    void getQuadST(int baseFaceId, int tessFaceId,
            Vec2f& st1, Vec2f& st2, Vec2f& st3,
            bool isFirst) const;

    void getQuadNormal(int baseFaceId, int tessFaceId,
            Vec3f& n1, Vec3f& n2, Vec3f& n3,
            float time, bool isFirst) const;
};

//------------------------------------------------------------------------------

} // namespace internal
} // namespace geom
} // namespace moonray


