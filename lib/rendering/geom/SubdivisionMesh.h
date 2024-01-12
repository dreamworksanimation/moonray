// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file SubdivisionMesh.h
/// $Id$
///
#pragma once
#ifndef SUBDIVISIONMESH_H
#define SUBDIVISIONMESH_H

#include <moonray/rendering/geom/LayerAssignmentId.h>
#include <moonray/rendering/geom/Primitive.h>
#include <moonray/rendering/geom/VertexBuffer.h>

#include <moonray/rendering/bvh/shading/PrimitiveAttribute.h>
#include <moonray/rendering/bvh/shading/Xform.h>
#include <moonray/rendering/geom/internal/InterleavedTraits.h>

namespace moonray {
namespace geom {

/// @class SubdivisionMesh
/// @brief Subdivision surface primitive defined by an input mesh, a subdivision
/// scheme and additional subdivision options.
class SubdivisionMesh : public moonray::geom::Primitive 
{
public:
    typedef moonray::geom::VertexBuffer<Vec3fa, InterleavedTraits> VertexBuffer;
    typedef std::vector<IndexType> FaceToPartBuffer;
    typedef std::vector<IndexType> IndexBuffer;
    typedef std::vector<float>     SharpnessBuffer;
    typedef std::vector<size_type> FaceVertexCount;

    enum class Scheme
    {
        BILINEAR,
        CATMULL_CLARK,
        NUM_SCHEMES
    };
    static bool isValidScheme(int x);

    enum class BoundaryInterpolation
    {
        NONE,
        EDGE_ONLY,
        EDGE_AND_CORNER,
        NUM_BOUNDARY_INTERPOLATIONS
    };
    static bool isValidBoundaryInterpolation(int x);

    enum class FVarLinearInterpolation
    {
        NONE,
        CORNERS_ONLY,
        CORNERS_PLUS1,
        CORNERS_PLUS2,
        BOUNDARIES,
        ALL,
        NUM_FVAR_LINEAR_INTERPOLATIONS
    };
    static bool isValidFVarLinearInterpolation(int x);

    struct Impl;

    explicit SubdivisionMesh(Impl* impl);

    SubdivisionMesh(Scheme scheme,
            FaceVertexCount&& faceVertexCount,
            IndexBuffer&& indices,
            VertexBuffer&& vertices,
            LayerAssignmentId&& layerAssignmentId,
            shading::PrimitiveAttributeTable&& primitiveAttributeTable);

    ~SubdivisionMesh();

    std::unique_ptr<SubdivisionMesh> copy() const;

    virtual void accept(PrimitiveVisitor& v) override;

    size_type getSubdivideFaceCount() const;

    size_type getSubdivideVertexCount() const;

    virtual size_type getMemory() const override;

    virtual size_type getMotionSamplesCount() const override;

    /// get the control mesh vertex buffer
    SubdivisionMesh::VertexBuffer& getControlVertexBuffer();

    /// set the primitive name
    /// @param name the name of this primitive
    void setName(const std::string& name);

    /// get the primitive name
    /// @return name the name of this primitive
    const std::string& getName() const;

    /// set the subdivision boundary interpolation option
    void setSubdBoundaryInterpolation(BoundaryInterpolation value);

    /// get the subdivision boundary interpolation option
    BoundaryInterpolation getSubdBoundaryInterpolation() const;

    /// set the subdivision face-varying linear-interpolation option
    void setSubdFVarLinearInterpolation(FVarLinearInterpolation value);

    /// get the subdivision face-varying linear-interpolation option
    FVarLinearInterpolation getSubdFVarLinearInterpolation() const;

    /// set the subdivision creasing data for edges
    void setSubdCreases(IndexBuffer&& creaseIndices,
                        SharpnessBuffer&& creaseSharpnesses);

    /// get whether the mesh has subd crease data
    bool hasSubdCreases() const;

    /// set the subdivision corner data for vertices
    void setSubdCorners(IndexBuffer&& cornerIndices,
                        SharpnessBuffer&& cornerSharpnesses);

    /// get whether the mesh has subd corner data
    bool hasSubdCorners() const;

    /// set the subdivision hole tag for faces
    void setSubdHoles(IndexBuffer&& holeIndices);

    /// get whether the mesh has subd face holes
    bool hasSubdHoles() const;

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

    // TODO replace this function with proper VertexBuffer update mechanics
    void updateVertexData(const std::vector<float>& vertexData,
            const shading::XformSamples& prim2render);

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
    std::unique_ptr<Impl> mImpl;
};

//
//  Inline static functions to validate enumerated types:
//
inline bool
SubdivisionMesh::isValidScheme(int x)
{
    return (x >= 0) &&
        (x < static_cast<int>(Scheme::NUM_SCHEMES));
}
inline bool
SubdivisionMesh::isValidBoundaryInterpolation(int x)
{
    return (x >= 0) &&
        (x < static_cast<int>(BoundaryInterpolation::NUM_BOUNDARY_INTERPOLATIONS));
}
inline bool
SubdivisionMesh::isValidFVarLinearInterpolation(int x)
{
    return (x >= 0) &&
        (x < static_cast<int>(FVarLinearInterpolation::NUM_FVAR_LINEAR_INTERPOLATIONS));
}

} // namespace geom
} // namespace moonray

#endif // SUBDIVISIONMESH_H
