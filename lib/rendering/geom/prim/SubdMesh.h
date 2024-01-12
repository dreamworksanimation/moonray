// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file SubdMesh.h
/// $Id$
///
#pragma once

#ifndef GEOM_SUBDMESH_HAS_BEEN_INCLUDED
#define GEOM_SUBDMESH_HAS_BEEN_INCLUDED

#include <moonray/rendering/geom/prim/Mesh.h>

#include <moonray/rendering/bvh/shading/AttributeKey.h>
#include <moonray/rendering/bvh/shading/PrimitiveAttribute.h>
#include <moonray/rendering/bvh/shading/Xform.h>
#include <moonray/rendering/geom/SubdivisionMesh.h>

#include <scene_rdl2/render/logging/logging.h>

namespace moonray {
namespace geom {
namespace internal {


// small helper struct that is used for tracking all the unique vertex values
// when face varying vertices are in the same position but have different values
template <typename T>
struct FVarVertex {
    FVarVertex(const T& data, int vertexId) :
        mData(data), mVertexId(vertexId) {}

    T mData;
    int mVertexId;
};


class ControlMeshData
{
public:
    ControlMeshData(SubdivisionMesh::Scheme scheme,
            SubdivisionMesh::FaceVertexCount&& faceVertexCount,
            SubdivisionMesh::IndexBuffer&& indices,
            SubdivisionMesh::VertexBuffer&& vertices,
            shading::PrimitiveAttributeTable&& primitiveAttributeTable):
        mScheme(scheme),
        mVertices(std::move(vertices)),
        mIndices(std::move(indices)),
        mFaceVertexCount(std::move(faceVertexCount)),
        mTextureRate(shading::RATE_UNKNOWN),
        mPrimitiveAttributeTable(std::move(primitiveAttributeTable)),
        mBoundaryInterpolation(SubdivisionMesh::BoundaryInterpolation::EDGE_AND_CORNER),
        mFVarLinearInterpolation(SubdivisionMesh::FVarLinearInterpolation::CORNERS_ONLY),
        mShutterOpenDelta(0.f), mShutterCloseDelta(0.f)
    {}

    bool initTextureSt()
    {
        mTextureRate = shading::RATE_UNKNOWN;
        const shading::TypedAttributeKey<scene_rdl2::math::Vec2f>& stKey = shading::StandardAttributes::sSurfaceST;
        if (!mPrimitiveAttributeTable.hasAttribute(stKey)) {
            return false;
        }
        const shading::PrimitiveAttribute<scene_rdl2::math::Vec2f>& textureSt =
            mPrimitiveAttributeTable.getAttribute(stKey);
        mTextureVertices.resize(mVertices.size());
        shading::AttributeRate stRate = textureSt.getRate();
        switch (stRate) {
        case shading::RATE_UNIFORM:
        //case shading::RATE_PART: // Unsupported as faceToPart(i) not accessible here
        case shading::RATE_FACE_VARYING: {
            mTextureRate = shading::RATE_FACE_VARYING;
            // This is exactly the same operation we did for other face varying
            // attributes during FaceVaryingAttributes::addAttributeBuffer
            std::vector<std::vector<FVarVertex<scene_rdl2::math::Vec2f>>> textureVertexGroups(
                mVertices.size());
            size_t indexOffset = 0;
            size_t faceCount = mFaceVertexCount.size();
            mTextureIndices.resize(mIndices.size());
            for (size_t f = 0; f < faceCount; ++f) {
                size_t nFv = mFaceVertexCount[f];
                for (size_t v = 0; v < nFv; ++v) {
                    int vid = mIndices[indexOffset + v];
                    size_t i;
                    switch (stRate) {
                    case shading::RATE_UNIFORM: i = f; break;
                    //case shading::RATE_PART: i = faceToPart(f); break;
                    default: i = indexOffset + v; break;
                    }
                    const scene_rdl2::math::Vec2f& st = textureSt[i];
                    if (textureVertexGroups[vid].empty()) {
                        int stVid = vid;
                        textureVertexGroups[vid].emplace_back(st, stVid);
                        mTextureVertices[stVid] = st;
                        mTextureIndices[indexOffset + v] = stVid;
                    } else {
                        bool foundDuplicatedSt = false;
                        for (const auto& textureVertex :
                            textureVertexGroups[vid]) {
                            if (scene_rdl2::math::isEqual(textureVertex.mData, st)) {
                                mTextureIndices[indexOffset + v] =
                                    textureVertex.mVertexId;
                                foundDuplicatedSt = true;
                                break;
                            }
                        }
                        if (!foundDuplicatedSt) {
                            int stVid = mTextureVertices.size();
                            mTextureVertices.push_back(st);
                            textureVertexGroups[vid].emplace_back(st, stVid);
                            mTextureIndices[indexOffset + v] = stVid;
                        }
                    }
                }
                indexOffset += nFv;
            }
            break;}
        case shading::RATE_VARYING:
        case shading::RATE_VERTEX:
            mTextureRate = shading::RATE_VERTEX;
            mTextureIndices = mIndices;
            MNRY_ASSERT_REQUIRE(textureSt.size() >= mVertices.size());
            for (size_t v = 0; v < mVertices.size(); ++v) {
                mTextureVertices[v] = textureSt[v];
            }
            break;
        case shading::RATE_CONSTANT:
            mTextureRate = shading::RATE_VERTEX;
            mTextureIndices = mIndices;
            for (size_t v = 0; v < mVertices.size(); ++v) {
                mTextureVertices[v] = textureSt[0];
            }
            break;
        default:
            mTextureRate = shading::RATE_UNKNOWN;
            scene_rdl2::Logger::error("unsupported rate for surface st");
            return false;
        }
        return true;
    }

    // this method is called when we do the reverse normal operation. Since the
    // main control mesh index buffer got reversed, the corresponding
    // texture index buffers will also need to be reversed
    void reverseTextureIndices()
    {
        size_t indexOffset = 0;
        for (size_t i = 0; i < mFaceVertexCount.size(); ++i) {
            int nFv = mFaceVertexCount[i];
            std::reverse(mTextureIndices.begin() + indexOffset,
                mTextureIndices.begin() + indexOffset + nFv);
            indexOffset += nFv;
        }
    }

    size_t getMemory() const
    {
        size_t result = 0;
        result += sizeof(mScheme);
        result += mVertices.get_memory_usage();
        result += scene_rdl2::util::getVectorMemory(mIndices);
        result += scene_rdl2::util::getVectorMemory(mFaceVertexCount);
        result += scene_rdl2::util::getVectorMemory(mTextureVertices);
        result += scene_rdl2::util::getVectorMemory(mTextureIndices);
        result += sizeof(mTextureRate);
        result += sizeof(mPrimitiveAttributeTable);
        result += scene_rdl2::util::getVectorMemory(mCreaseSharpness);
        result += scene_rdl2::util::getVectorMemory(mCreaseIndices);
        result += scene_rdl2::util::getVectorMemory(mCornerSharpness);
        result += scene_rdl2::util::getVectorMemory(mCornerIndices);
        result += scene_rdl2::util::getVectorMemory(mHoleIndices);
        result += scene_rdl2::util::getVectorMemory(mXforms);
        result += sizeof(mShutterOpenDelta);
        result += sizeof(mShutterCloseDelta);
        return result;
    }

    const shading::PrimitiveAttributeTable& getPrimitiveAttributeTable() const
    {
        return mPrimitiveAttributeTable;
    }

    // The subdivision rule (Catmull-Clark or Linear)
    SubdivisionMesh::Scheme mScheme;

    // Vertex buffer for control mesh
    SubdivisionMesh::VertexBuffer mVertices;
    // Index buffer for control mesh
    SubdivisionMesh::IndexBuffer mIndices;
    // face vertex count for control mesh
    SubdivisionMesh::FaceVertexCount mFaceVertexCount;

    // Optional vertex buffer for control mesh texture ST
    shading::Vector<scene_rdl2::math::Vec2f> mTextureVertices;
    // Optional index buffer for control mesh texture ST
    SubdivisionMesh::IndexBuffer mTextureIndices;
    // Attribute rate for optional control mesh texture ST
    shading::AttributeRate mTextureRate;
    shading::PrimitiveAttributeTable mPrimitiveAttributeTable;

    // Optional subdivision properties:
    SubdivisionMesh::BoundaryInterpolation mBoundaryInterpolation;
    SubdivisionMesh::FVarLinearInterpolation mFVarLinearInterpolation;

    // Optional sharpness and index buffers for creases (semi-sharp edges):
    SubdivisionMesh::SharpnessBuffer mCreaseSharpness;
    SubdivisionMesh::IndexBuffer mCreaseIndices;
    // Optional sharpness and index buffers for corners (semi-sharp vertices):
    SubdivisionMesh::SharpnessBuffer mCornerSharpness;
    SubdivisionMesh::IndexBuffer mCornerIndices;
    // Optional index buffer for face holes:
    SubdivisionMesh::IndexBuffer mHoleIndices;

    shading::XformSamples mXforms;
    float mShutterOpenDelta;
    float mShutterCloseDelta;
};

//----------------------------------------------------------------------------

///
/// @class SubdMesh SubdMesh.h <geom/prim/SubdMesh.h>
/// @brief base subdivision mesh primitive internal implementation
/// 
class SubdMesh : public Mesh
{
public:
    /// Constructor / Destructor
    SubdMesh(
        SubdivisionMesh::Scheme scheme,
        SubdivisionMesh::FaceVertexCount&& faceVertexCount,
        SubdivisionMesh::IndexBuffer&& indices,
        SubdivisionMesh::VertexBuffer&& vertices,
        LayerAssignmentId&& layerAssignmentId,
        shading::PrimitiveAttributeTable&& primitiveAttributeTable):
            Mesh(std::move(layerAssignmentId)),
            mControlMeshData(new ControlMeshData(scheme,
                                                 std::move(faceVertexCount),
                                                 std::move(indices),
                                                 std::move(vertices),
                                                 std::move(primitiveAttributeTable)))
        {}

    virtual ~SubdMesh() = default;

    virtual SubdMesh* copy() const = 0;

    virtual SubdivisionMesh::VertexBuffer& getControlVertexBuffer() = 0;

    virtual size_t getMemory() const {
        return sizeof(SubdMesh) - sizeof(Mesh) + Mesh::getMemory();
    }

    virtual size_t getTessellatedMeshVertexCount() const = 0;

    virtual void setTransform(const shading::XformSamples& xforms,
            float shutterOpenDelta, float shutterCloseDelta) = 0;

    virtual void setParts(size_t partCount,
            SubdivisionMesh::FaceToPartBuffer&& faceToPart) = 0;

    /// explicitely disable copy constructor and assignment operator
    SubdMesh(const SubdMesh &other) = delete;

    const SubdMesh &operator=(const SubdMesh &other) = delete;

    const shading::PrimitiveAttributeTable& getPrimitiveAttributeTable() const
    {
        return mControlMeshData->getPrimitiveAttributeTable();
    }

    void setSubdBoundaryInterpolation(SubdivisionMesh::BoundaryInterpolation val)
    {
        mControlMeshData->mBoundaryInterpolation = val;
    }

    void setSubdFVarLinearInterpolation(SubdivisionMesh::FVarLinearInterpolation val)
    {
        mControlMeshData->mFVarLinearInterpolation = val;
    }

    SubdivisionMesh::BoundaryInterpolation getSubdBoundaryInterpolation() const
    {
        return mControlMeshData->mBoundaryInterpolation;
    }

    SubdivisionMesh::FVarLinearInterpolation getSubdFVarLinearInterpolation() const
    {
        return mControlMeshData->mFVarLinearInterpolation;
    }

    void setSubdCreases(SubdivisionMesh::IndexBuffer&&     creaseIndices,
                        SubdivisionMesh::SharpnessBuffer&& creaseSharpnesses)
    {
        mControlMeshData->mCreaseIndices   = std::move(creaseIndices);
        mControlMeshData->mCreaseSharpness = std::move(creaseSharpnesses);
    }

    void setSubdCorners(SubdivisionMesh::IndexBuffer&&     cornerIndices,
                        SubdivisionMesh::SharpnessBuffer&& cornerSharpness)
    {
        mControlMeshData->mCornerIndices   = std::move(cornerIndices);
        mControlMeshData->mCornerSharpness = std::move(cornerSharpness);
    }

    void setSubdHoles(SubdivisionMesh::IndexBuffer&& holeIndices)
    {
        mControlMeshData->mHoleIndices = std::move(holeIndices);
    }

    bool hasSubdCreases() const
    {
        return mControlMeshData->mCreaseIndices.size() > 0;
    }

    bool hasSubdCorners() const
    {
        return mControlMeshData->mCornerIndices.size() > 0;
    }

    bool hasSubdHoles() const
    {
        return mControlMeshData->mHoleIndices.size() > 0;
    }

    void setCurvedMotionBlurSampleCount(int count)
    {
        mCurvedMotionBlurSampleCount = count;
    }

    int getCurvedMotionBlurSampleCount()
    {
        return mCurvedMotionBlurSampleCount;
    }

    scene_rdl2::rdl2::MotionBlurType getMotionBlurType()
    {
        return mMotionBlurType;
    }

protected:
    scene_rdl2::rdl2::MotionBlurType mMotionBlurType;
    uint32_t mCurvedMotionBlurSampleCount;
    std::unique_ptr<ControlMeshData> mControlMeshData;
};

//----------------------------------------------------------------------------

} // namespace internal
} // namespace geom
} // namespace moonray

#endif /* GEOM_SUBDMESH_HAS_BEEN_INCLUDED */

