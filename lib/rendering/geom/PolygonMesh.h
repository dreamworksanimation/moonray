// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file PolygonMesh.h
/// $Id$
///
#pragma once
#ifndef POLYGONMESH_H
#define POLYGONMESH_H

#include <moonray/rendering/geom/LayerAssignmentId.h>
#include <moonray/rendering/geom/Primitive.h>
#include <moonray/rendering/geom/VertexBuffer.h>

#include <moonray/rendering/bvh/shading/PrimitiveAttribute.h>
#include <moonray/rendering/bvh/shading/Xform.h>
#include <moonray/rendering/geom/internal/InterleavedTraits.h>

namespace moonray {
namespace geom {

// forward declaration
class BakedAttribute;

// BakedMesh is used for geometry baking, i.e.
// RenderContext->bakeGeometry().

struct BakedMesh
{
    const scene_rdl2::rdl2::Geometry* mRdlGeometry;

    // Mesh name
    std::string mName;

    // Will be 3 or 4
    int mVertsPerFace;

    // Indices for tessellated faces.  Size of mIndexBuffer is numFaces * mVertsPerFace.
    // Obtain numFaces by dividing by mVertsPerFace.
    std::vector<unsigned int> mIndexBuffer;

    // Number of tessellated vertices
    size_t mVertexCount;

    // 0 if no motion blur
    size_t mMotionSampleCount;

    std::vector<float> mMotionFrames;

    // The vertex buffer always has the format v0_t0, v0_t1, v1_t0, v1_t1...
    // (if there are two motion samples t0 and t1.)
    std::vector<Vec3f> mVertexBuffer;

    // WARNING: Baked geometry vertices and normals are in RENDER SPACE.  You may want to
    // transform back to object space.  Be careful with transformNormal() as it uses
    // the inverse xform!

    // Mapping from tessellated face id to base face id.
    std::vector<int> mTessellatedToBaseFace;

    // Mapping of face to part.
    std::vector<int> mFaceToPart;

    // Mesh attributes.
    std::vector<std::unique_ptr<BakedAttribute>> mAttrs;
};


/// @class PolygonMesh
/// @brief Primitive that composed by a set of polygons
class PolygonMesh : public moonray::geom::Primitive
{
public:
    typedef typename moonray::geom::VertexBuffer<Vec3fa, InterleavedTraits> VertexBuffer;
    typedef std::vector<IndexType> FaceToPartBuffer;
    typedef std::vector<IndexType> IndexBuffer;
    typedef std::vector<size_type> FaceVertexCount;

    /// the constructor for ngons
    PolygonMesh(FaceVertexCount&& faceVertexCount,
            IndexBuffer&& indices, VertexBuffer&& vertices,
            LayerAssignmentId&& layerAssignmentId,
            shading::PrimitiveAttributeTable&& primitiveAttributeTable);

    ~PolygonMesh(); 

    virtual void accept(PrimitiveVisitor& v) override;

    size_type getFaceCount() const;

    virtual size_type getMemory() const override;

    virtual size_type getMotionSamplesCount() const override;

    /// get the control mesh vertex buffer
    PolygonMesh::VertexBuffer& getVertexBuffer();

    /// set the primitive name
    /// @param name the name of this primitive
    void setName(const std::string& name);

    /// get the primitive name
    /// @return name the name of this primitive
    const std::string& getName() const;

    // sets the face->part mapping
    void setParts(size_t partCount, FaceToPartBuffer&& faceToPart);

    /// set the mesh resolution (each edge in control face would be split
    /// into n segments when resolution is n)
    void setMeshResolution(int meshResolution);

    /// set the maximum allowable difference in pixels for
    /// adaptive tessellation (each final tessellated edge won't be longer
    /// than n pixels if adaptiveError is set to n)
    void setAdaptiveError(float adaptiveError);

    /// set whether the mesh is single sided
    void setIsSingleSided(bool isSingleSided);

    /// get whether the mesh is single sided
    bool getIsSingleSided() const;

    /// set whether normals are reversed
    void setIsNormalReversed(bool isNormalReversed);

    /// get whether normals are reversed
    bool getIsNormalReversed() const;

    /// set whether orientation is reversed
    void setIsOrientationReversed(bool);

    /// get whether orientation is reversed
    bool getIsOrientationReversed() const;

    /// set whether the mesh automatically calculates smooth shadint normal
    void setSmoothNormal(bool smoothNormal);

    /// get whether the mesh automatically calculates smooth shading normal
    bool getSmoothNormal() const;

    // TODO replace this function with proper VertexBuffer update mechanics
    void updateVertexData(const std::vector<float>& vertexData,
            const shading::XformSamples& prim2render);

    void recomputeVertexNormals();

    /// return the number of vertices in this polygon mesh
    size_type getVertexCount() const;

    void setCurvedMotionBlurSampleCount(int count);

private:
    /// @remark For renderer internal use, procedural should never call this
    /// @internal
    virtual internal::Primitive* getPrimitiveImpl() override;

    /// @remark For renderer internal use, procedural should never call this
    /// @internal
    virtual void transformPrimitive(
            const MotionBlurParams& motionBlurParams,
            const shading::XformSamples& prim2render) override;

private:
    struct Impl;
    std::unique_ptr<Impl> mImpl;
};

} // namespace geom
} // namespace moonray

#endif // POLYGONMESH_H
