// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file OpenSubdivMesh.cc
///

#include "OpenSubdivMesh.h"

#include <moonray/rendering/geom/prim/GeomTLState.h>
#include <moonray/rendering/geom/prim/MeshTessellationUtil.h>
#include <moonray/rendering/geom/prim/Util.h>

#include <moonray/rendering/bvh/shading/Attributes.h>
#include <moonray/rendering/bvh/shading/PrimitiveAttribute.h>
#include <moonray/rendering/bvh/shading/RootShader.h>
#include <moonray/rendering/bvh/shading/State.h>
#include <moonray/rendering/geom/BakedAttribute.h>
#include <moonray/rendering/mcrt_common/ThreadLocalState.h>

#include <opensubdiv/far/patchDescriptor.h>
#include <opensubdiv/far/patchMap.h>
#include <opensubdiv/far/patchTable.h>
#include <opensubdiv/far/patchTableFactory.h>
#include <opensubdiv/far/primvarRefiner.h>
#include <opensubdiv/far/topologyDescriptor.h>
#include <opensubdiv/far/topologyRefiner.h>
#include <opensubdiv/sdc/types.h>

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include "tbb/concurrent_vector.h"

#include <scene_rdl2/common/math/Vec2.h>

#include <limits>
#include <numeric>

namespace moonray {
namespace geom {
namespace internal {

using scene_rdl2::math::Vec2f;
using namespace mcrt_common;
using namespace shading;

static constexpr size_t sTextureStFVarChannel = 0;

// FaceVaryingAttributes extract out arbitrary face varying attributes stored
// in PrimitiveAttributeTable and store each of them with its own
// index/vertex buffer. After tessellation the control index/vertex buffers
// will be swapped with their final tessellated result, which is used during
// mcrt rendering time query/interpolation
class FaceVaryingAttributes
{
public:
    FaceVaryingAttributes(const int controlVertexCount,
            const SubdivisionMesh::FaceVertexCount& controlFaceVertexCount,
            const SubdivisionMesh::IndexBuffer& controlIndices,
            const PrimitiveAttributeTable& primitiveAttributeTable,
            const int firstFaceVaryingChannel)
    {
        int maxFvarKey = -1;
        std::vector<AttributeKey> fvarKeys;
        for (const auto& kv : primitiveAttributeTable) {
            AttributeKey key = kv.first;
            AttributeRate rate = primitiveAttributeTable.getRate(key);
            if (rate == RATE_FACE_VARYING) {
                // face varying before process should be the same size as
                // control index buffer
                MNRY_ASSERT_REQUIRE(
                    kv.second[0]->size() == controlIndices.size());
                fvarKeys.push_back(key);
                if (key > maxFvarKey) {
                    maxFvarKey = key;
                }
            }
        }
        mKeyToBufferIndex.resize(maxFvarKey + 1, -1);
        int channel = firstFaceVaryingChannel;
        for (const auto& key : fvarKeys) {
            switch (key.getType()) {
            case scene_rdl2::rdl2::TYPE_FLOAT:
                addAttributeBuffer(TypedAttributeKey<float>(key),
                    controlVertexCount, controlFaceVertexCount, controlIndices,
                    primitiveAttributeTable, channel);
                break;
            case scene_rdl2::rdl2::TYPE_RGB:
                addAttributeBuffer(TypedAttributeKey<scene_rdl2::math::Color>(key),
                    controlVertexCount, controlFaceVertexCount, controlIndices,
                    primitiveAttributeTable, channel);
                break;
            case scene_rdl2::rdl2::TYPE_RGBA:
                addAttributeBuffer(TypedAttributeKey<scene_rdl2::math::Color4>(key),
                    controlVertexCount, controlFaceVertexCount, controlIndices,
                    primitiveAttributeTable, channel);
                break;
            case scene_rdl2::rdl2::TYPE_VEC2F:
                addAttributeBuffer(TypedAttributeKey<Vec2f>(key),
                    controlVertexCount, controlFaceVertexCount, controlIndices,
                    primitiveAttributeTable, channel);
                break;
            case scene_rdl2::rdl2::TYPE_VEC3F:
                addAttributeBuffer(TypedAttributeKey<Vec3f>(key),
                    controlVertexCount, controlFaceVertexCount, controlIndices,
                    primitiveAttributeTable, channel);
                break;
            default:
                // we can only tessellate float based attributes
                MNRY_ASSERT_REQUIRE(false,
                    (std::string("unsupported attribute type ") +
                    std::string(attributeTypeName(key.getType())) +
                    std::string(" for face varying atttribute ") +
                    std::string(key.getName())).c_str());
                break;
            }
            channel++;
        }
    }

    struct AttributeBuffer
    {
        int getVertexCount() const { return mData.size() / mFloatPerVertex; }
        std::vector<float> mData;
        SubdivisionMesh::IndexBuffer mIndices;
        int mFloatPerVertex;
        int mChannel;
    };

    // this method is called during postIntersect stage. Interpolate requested
    // face varying attribute and fill the result to intersection
    void fillAttributes(Intersection& intersection, int faceId,
            float w1, float w2, float w3, float w4) const
    {
        for (const auto k : intersection.getTable()->getRequiredAttributes()) {
            if (hasAttribute(k)) {
                fillAttribute(k, intersection, faceId, w1, w2, w3, w4);
            }
        }
        for (const auto k : intersection.getTable()->getOptionalAttributes()) {
            if (hasAttribute(k)) {
                fillAttribute(k, intersection, faceId, w1, w2, w3, w4);
            }
        }
    }

    void fillAttribute(AttributeKey key, Intersection& intersection, int faceId,
            float w1, float w2, float w3, float w4) const
    {
        switch (key.getType()) {
        case scene_rdl2::rdl2::TYPE_FLOAT:
            fillAttribute(TypedAttributeKey<float>(key), intersection,
                faceId, w1, w2, w3, w4);
            break;
        case scene_rdl2::rdl2::TYPE_RGB:
            fillAttribute(TypedAttributeKey<scene_rdl2::math::Color>(key), intersection,
                faceId, w1, w2, w3, w4);
            break;
        case scene_rdl2::rdl2::TYPE_RGBA:
            fillAttribute(TypedAttributeKey<scene_rdl2::math::Color4>(key), intersection,
                faceId, w1, w2, w3, w4);
            break;
        case scene_rdl2::rdl2::TYPE_VEC2F:
            fillAttribute(TypedAttributeKey<Vec2f>(key), intersection,
                faceId, w1, w2, w3, w4);
            break;
        case scene_rdl2::rdl2::TYPE_VEC3F:
            fillAttribute(TypedAttributeKey<Vec3f>(key), intersection,
                faceId, w1, w2, w3, w4);
            break;
        default:
            // we can only tessellate float based attributes
            MNRY_ASSERT(false,
                (std::string("unsupported attribute type ") +
                std::string(attributeTypeName(key.getType())) +
                std::string(" for face varying atttribute ") +
                std::string(key.getName())).c_str());
            break;
        }
    }

    template <typename T> void
    fillAttribute(TypedAttributeKey<T> key, Intersection& intersection,
            int faceId, float w1, float w2, float w3, float w4) const
    {
        const auto& attributeBuffer = getAttributeBuffer(key);
        int vid1 = attributeBuffer.mIndices[sQuadVertexCount * faceId    ];
        int vid2 = attributeBuffer.mIndices[sQuadVertexCount * faceId + 1];
        int vid3 = attributeBuffer.mIndices[sQuadVertexCount * faceId + 2];
        int vid4 = attributeBuffer.mIndices[sQuadVertexCount * faceId + 3];
        const T* data = reinterpret_cast<const T*>(
            attributeBuffer.mData.data());
        intersection.setAttribute(key,
            w1 * data[vid1] + w2 * data[vid2] +
            w3 * data[vid3] + w4 * data[vid4]);
    }

    bool hasAttribute(AttributeKey key) const
    {
        return (size_t)key < mKeyToBufferIndex.size() &&
            mKeyToBufferIndex[key] != -1;
    }

    const AttributeBuffer& getAttributeBuffer(AttributeKey key) const
    {
        MNRY_ASSERT(hasAttribute(key));
        return mAttributeBuffers[mKeyToBufferIndex[key]];
    }

    AttributeBuffer& getAttributeBuffer(AttributeKey key)
    {
        MNRY_ASSERT(hasAttribute(key));
        return mAttributeBuffers[mKeyToBufferIndex[key]];
    }

    std::vector<AttributeKey> getAllKeys() const
    {
        std::vector<AttributeKey> keys;
        for (size_t i = 0; i < mKeyToBufferIndex.size(); ++i) {
            if (mKeyToBufferIndex[i] != -1) {
                keys.emplace_back(i);
            }
        }
        return keys;
    }

    size_t getMemory() const
    {
        size_t result = scene_rdl2::util::getVectorMemory(mAttributeBuffers) +
            scene_rdl2::util::getVectorMemory(mKeyToBufferIndex);
        for (const auto& attributeBuffer : mAttributeBuffers) {
            result += scene_rdl2::util::getVectorMemory(attributeBuffer.mData);
            result += scene_rdl2::util::getVectorMemory(attributeBuffer.mIndices);
        }
        return result;
    }

    // this method is called when we do the reverse normal operation. Since the
    // main control mesh index buffer got reversed, all the corresponding
    // face varying attribute index buffers will also need to be reversed
    void reverseControlIndices(
        const SubdivisionMesh::FaceVertexCount& controlFaceVertexCount)
    {
        size_t faceCount = controlFaceVertexCount.size();
        for (auto& buffer : mAttributeBuffers) {
            size_t indexOffset = 0;
            for (size_t i = 0; i < faceCount; ++i) {
                int nFv = controlFaceVertexCount[i];
                std::reverse(buffer.mIndices.begin() + indexOffset,
                    buffer.mIndices.begin() + indexOffset + nFv);
                indexOffset += nFv;
            }
        }
    }

private:
    template <typename T> void
    addAttributeBuffer(const TypedAttributeKey<T>& key,
            const int controlVertexCount,
            const SubdivisionMesh::FaceVertexCount controlFaceVertexCount,
            const SubdivisionMesh::IndexBuffer controlIndices,
            const PrimitiveAttributeTable& primitiveAttributeTable,
            const int channel)
    {
        AttributeBuffer buffer;
        size_t stride = sizeof(T) / sizeof(float);
        buffer.mFloatPerVertex = stride;
        buffer.mIndices.reserve(controlIndices.size());
        buffer.mData.resize(stride * controlVertexCount);
        buffer.mChannel = channel;
        const auto& attribute = primitiveAttributeTable.getAttribute(key);
        std::vector<std::vector<FVarVertex<T>>> fvarVertexGroups(
            controlVertexCount);
        // most of the face varying attributes still use the same data
        // on one vertex across all face sharing the vertex, so the fully
        // expanded face varying attribute is not optimal in memory usage
        // (it also caused incorrect topology analyzation since in that case
        // every face on face varying attribute is its own isolated face)
        // The following operation goes through all the vertex in face varying
        // attributes, locate the duplicated attributes and figure out proper
        // index buffer for input attribute with specified key
        size_t indexOffset = 0;
        size_t faceCount = controlFaceVertexCount.size();
        for (size_t f = 0; f < faceCount; ++f) {
            size_t nFv = controlFaceVertexCount[f];
            for (size_t v = 0; v < nFv; ++v) {
                int vid = controlIndices[indexOffset + v];
                const T& data = attribute[indexOffset + v];
                if (fvarVertexGroups[vid].empty()) {
                    int fvarVid = vid;
                    fvarVertexGroups[vid].emplace_back(data, fvarVid);
                    memcpy(&buffer.mData[stride * fvarVid], &data,
                        sizeof(T));
                    buffer.mIndices.push_back(fvarVid);
                } else {
                    bool foundDuplicatedVertex = false;
                    for (const auto& fvarVertex : fvarVertexGroups[vid]) {
                        if (scene_rdl2::math::isEqual(fvarVertex.mData, data)) {
                            buffer.mIndices.push_back(fvarVertex.mVertexId);
                            foundDuplicatedVertex = true;
                            break;
                        }
                    }
                    if (!foundDuplicatedVertex) {
                        int fvarVid = buffer.mData.size() / stride;
                        for (size_t d = 0; d < stride; ++d) {
                            buffer.mData.push_back(0.0f);
                        }
                        memcpy(&buffer.mData[stride * fvarVid], &data,
                            sizeof(T));
                        fvarVertexGroups[vid].emplace_back(data, fvarVid);
                        buffer.mIndices.push_back(fvarVid);
                    }
                }
            }
            indexOffset += nFv;
        }
        mKeyToBufferIndex[key] = mAttributeBuffers.size();
        mAttributeBuffers.push_back(std::move(buffer));
    }

private:
    std::vector<AttributeBuffer> mAttributeBuffers;
    std::vector<int> mKeyToBufferIndex;
};

OpenSubdivMesh::OpenSubdivMesh(SubdivisionMesh::Scheme scheme,
        SubdivisionMesh::FaceVertexCount&& faceVertexCount,
        SubdivisionMesh::IndexBuffer&& indices,
        SubdivisionMesh::VertexBuffer&& vertices,
        LayerAssignmentId&& layerAssignmentId,
        PrimitiveAttributeTable&& primitiveAttributeTable):
    SubdMesh(scheme,
             std::move(faceVertexCount),
             std::move(indices),
             std::move(vertices),
             std::move(layerAssignmentId),
             std::move(primitiveAttributeTable)),
    mPartCount(0)
{
}

OpenSubdivMesh::~OpenSubdivMesh() = default;

size_t
OpenSubdivMesh::getMemory() const
{
    size_t result = sizeof(OpenSubdivMesh) - sizeof(SubdMesh) + SubdMesh::getMemory();
    if (mControlMeshData) {
        result += mControlMeshData->getMemory();
    }
    result += scene_rdl2::util::getVectorElementsMemory(mTessellatedIndices);
    result += scene_rdl2::util::getVectorElementsMemory(mFaceToPart);
    result += mTessellatedVertices.get_memory_usage();
    result += mSurfaceNormal.get_memory_usage();
    result += mSurfaceSt.get_memory_usage();
    result += mSurfaceDpds.get_memory_usage();
    result += mSurfaceDpdt.get_memory_usage();
    result += scene_rdl2::util::getVectorElementsMemory(mTessellatedToControlFace);
    if (mFaceVaryingAttributes) {
        result += mFaceVaryingAttributes->getMemory();
    }
    return result;
}

size_t
OpenSubdivMesh::getMotionSamplesCount() const
{
    return mControlMeshData ?
        mControlMeshData->mVertices.get_time_steps() :
        mTessellatedVertices.get_time_steps();
}

static OpenSubdiv::Far::TopologyRefiner*
createTopologyRefiner(const ControlMeshData& controlMeshData,
        const FaceVaryingAttributes& faceVaryingAttributes)
{
    typedef OpenSubdiv::Far::TopologyDescriptor Descriptor;
    typedef OpenSubdiv::Far::TopologyRefinerFactory<Descriptor> RefinerFactory;

    // while data may be present to support all subdivision options, downstream
    // support for some may not be present -- mark those to be ignored here:
    //
    // similarly, include flags to revert to original defaults if production
    // data from multiple sources/procedurals cannot be trusted:
    const bool ignoreHoles = true;

    const bool useStudioDefaultBoundaryInterpolation = false;
    const bool useStudioDefaultFVarLinearInterpolation = false;

    // assign subdivision scheme
    OpenSubdiv::Sdc::SchemeType osdScheme = OpenSubdiv::Sdc::SCHEME_BILINEAR;
    switch (controlMeshData.mScheme) {
    case SubdivisionMesh::Scheme::BILINEAR:
        osdScheme = OpenSubdiv::Sdc::SCHEME_BILINEAR;
        break;
    case SubdivisionMesh::Scheme::CATMULL_CLARK:
        osdScheme = OpenSubdiv::Sdc::SCHEME_CATMARK;
        break;
    default:
        MNRY_ASSERT(false, "unknown subd scheme value");
        break;
    }

    // assign subdivision options
    OpenSubdiv::Sdc::Options::VtxBoundaryInterpolation osdBoundary =
        OpenSubdiv::Sdc::Options::VTX_BOUNDARY_EDGE_AND_CORNER;
    if (!useStudioDefaultBoundaryInterpolation) {
        switch (controlMeshData.mBoundaryInterpolation) {
        case SubdivisionMesh::BoundaryInterpolation::NONE:
            osdBoundary = ignoreHoles
                        ? OpenSubdiv::Sdc::Options::VTX_BOUNDARY_EDGE_AND_CORNER
                        : OpenSubdiv::Sdc::Options::VTX_BOUNDARY_NONE;
            break;
        case SubdivisionMesh::BoundaryInterpolation::EDGE_ONLY:
            osdBoundary = OpenSubdiv::Sdc::Options::VTX_BOUNDARY_EDGE_ONLY;
            break;
        case SubdivisionMesh::BoundaryInterpolation::EDGE_AND_CORNER:
            osdBoundary = OpenSubdiv::Sdc::Options::VTX_BOUNDARY_EDGE_AND_CORNER;
            break;
        default:
            MNRY_ASSERT(false, "unknown subd boundary interpolation value");
            break;
        }
    }

    OpenSubdiv::Sdc::Options::FVarLinearInterpolation osdFVarLinear =
        OpenSubdiv::Sdc::Options::FVAR_LINEAR_CORNERS_ONLY;
    if (!useStudioDefaultFVarLinearInterpolation) {
        switch (controlMeshData.mFVarLinearInterpolation) {
        case SubdivisionMesh::FVarLinearInterpolation::NONE:
            osdFVarLinear = OpenSubdiv::Sdc::Options::FVAR_LINEAR_NONE;
            break;
        case SubdivisionMesh::FVarLinearInterpolation::CORNERS_ONLY:
            osdFVarLinear = OpenSubdiv::Sdc::Options::FVAR_LINEAR_CORNERS_ONLY;
            break;
        case SubdivisionMesh::FVarLinearInterpolation::CORNERS_PLUS1:
            osdFVarLinear = OpenSubdiv::Sdc::Options::FVAR_LINEAR_CORNERS_PLUS1;
            break;
        case SubdivisionMesh::FVarLinearInterpolation::CORNERS_PLUS2:
            osdFVarLinear = OpenSubdiv::Sdc::Options::FVAR_LINEAR_CORNERS_PLUS2;
            break;
        case SubdivisionMesh::FVarLinearInterpolation::BOUNDARIES:
            osdFVarLinear = OpenSubdiv::Sdc::Options::FVAR_LINEAR_BOUNDARIES;
            break;
        case SubdivisionMesh::FVarLinearInterpolation::ALL:
            osdFVarLinear = OpenSubdiv::Sdc::Options::FVAR_LINEAR_ALL;
            break;
        default:
            MNRY_ASSERT(false, "unknown subd fvar linear interpolation value");
            break;
        }
    }

    OpenSubdiv::Sdc::Options osdOptions;
    osdOptions.SetVtxBoundaryInterpolation(osdBoundary);
    osdOptions.SetFVarLinearInterpolation(osdFVarLinear);

    // fill out topology specs
    Descriptor desc;
    desc.numVertices = controlMeshData.mVertices.size();
    desc.numFaces = controlMeshData.mFaceVertexCount.size();
    desc.numVertsPerFace = (const int*)controlMeshData.mFaceVertexCount.data();
    desc.vertIndicesPerFace = (const int*)controlMeshData.mIndices.data();

    const auto& fvarKeys = faceVaryingAttributes.getAllKeys();
    size_t fvarAttributesCount = fvarKeys.size();
    int textureVertexCount = controlMeshData.mTextureVertices.size();
    bool hasFaceVaryingTextureSt =
        controlMeshData.mTextureRate == RATE_FACE_VARYING;
    // if there is textureSt, it will occupies one extra face varying channel
    size_t fvarChannelCount = hasFaceVaryingTextureSt ?
        1 + fvarAttributesCount : fvarAttributesCount;
    std::vector<Descriptor::FVarChannel> channels(fvarChannelCount);
    if (hasFaceVaryingTextureSt) {
        channels[sTextureStFVarChannel].numValues = textureVertexCount;
        channels[sTextureStFVarChannel].valueIndices =
            (const int*)controlMeshData.mTextureIndices.data();
    }
    for (const auto& key : fvarKeys) {
        const auto& attribute = faceVaryingAttributes.getAttributeBuffer(key);
        channels[attribute.mChannel].numValues = attribute.getVertexCount();
        channels[attribute.mChannel].valueIndices =
            (const int*)attribute.mIndices.data();
    }
    desc.numFVarChannels = fvarChannelCount;
    desc.fvarChannels = channels.data();

    // Subdivision properties assigned to topology (creases, corners and holes):
    desc.numCreases = controlMeshData.mCreaseSharpness.size();
    if (desc.numCreases) {
        desc.creaseVertexIndexPairs =
            (const int*)controlMeshData.mCreaseIndices.data();
        desc.creaseWeights =
            (const float*)controlMeshData.mCreaseSharpness.data();
    }

    desc.numCorners = controlMeshData.mCornerSharpness.size();
    if (desc.numCorners) {
        desc.cornerVertexIndices =
            (const int*)controlMeshData.mCornerIndices.data();
        desc.cornerWeights =
            (const float*)controlMeshData.mCornerSharpness.data();
    }

    desc.numHoles = ignoreHoles ? 0 : controlMeshData.mHoleIndices.size();
    if (desc.numHoles) {
        desc.holeIndices =
            (const int*)controlMeshData.mHoleIndices.data();
    }

    // Instantiate a TopologyRefiner from the descripor
    OpenSubdiv::Far::TopologyRefiner* refiner = RefinerFactory::Create(desc,
        RefinerFactory::Options(osdScheme, osdOptions));
    return refiner;
}

// This is the control point OpenSubdiv use to weight sum final limit surface
// sample point for position data
struct PatchCV
{
    explicit PatchCV(size_t size)
    : mData(size)
    {
    }

    finline void Clear(void* = nullptr) {
        std::fill(mData.begin(), mData.end(), Vec3fa{0.0f});
        // We don't want to change the size of the vector, so we won't call 'clear()'
    }

    finline void AddWithWeight(const PatchCV& src, float weight) {
        for (size_t t = 0; t < mData.size(); ++t) {
            mData[t] += weight * src.mData[t];
        }
    }

    std::vector<Vec3fa> mData;
};

// This is the control point OpenSubdiv use to weight sum final limit surface
// sample point for texture st data
struct TextureCV
{
    finline void Clear() {
        memset(&mSt, 0, sizeof(Vec2f));
    }

    finline void AddWithWeight(const TextureCV& src, float weight) {
        mSt += weight * src.mSt;
    }

    Vec2f mSt;
};

// we don't know the primitive attribute size in compile time
// (user can feed in arbitrary sets of primitive attributes)
// so we need to use indirect reference way to implement
// OpenSubdiv required interface for primitive attributes
struct PrimVarCV
{
    finline void Clear() {
        memset(mData, 0, mFloatPerCV * sizeof(float));
    }

    finline void AddWithWeight(const PrimVarCV& src, float weight) {
        for (int i = 0; i < mFloatPerCV; ++i) {
            mData[i] += weight * src.mData[i];
        }
    }

    float* mData;
    int mFloatPerCV;
};

// Provide a list of cluster vertices. Each cluster should be in C0
// continuity during limit surface evaluation stage  but the continuity breaks
// after displacement due to the fact that each vertex in cluster can have
// different texture st coordinate. The current solution to avoid crack is
// simply glue all vertices in cluster to the average position
void
getOverlapVertexClusters(const FaceVaryingSeams& faceVaryingSeams,
        const SubdTessellatedVertexLookup& tessellatedVertexLookup,
        std::vector<std::vector<int>>& vertexClusters)
{
    // overlap vertices
    for (const auto& vertices : faceVaryingSeams.mOverlapVertices) {
        std::vector<int> vertexCluster;
        vertexCluster.reserve(vertices.second.size());
        for (const auto& cv : vertices.second) {
            int vid = tessellatedVertexLookup.getTessellatedVertexId(cv);
            // may encounter vertices in unassigned face
            if (vid < 0) {
                continue;
            }
            vertexCluster.push_back(vid);
        }
        if (vertexCluster.size() > 1) {
            vertexClusters.push_back(std::move(vertexCluster));
        }
    }
    // overlap edges
    for (const auto& edges : faceVaryingSeams.mOverlapEdges) {
        size_t clusterSize = edges.second.size();
        if (clusterSize == 0) {
            continue;
        }
        // this kind of edge vertex count mismatch can happen where
        // faces outside of camera frustum share edges with faces inside
        // camera frustum. In this kind of situation we just skip the
        // later displacement stitching
        bool mismatchEdgeVertexCount = false;
        int edge0VertexCount =
            tessellatedVertexLookup.getEdgeVertexCount(
            edges.second[0].mEdgeId0);
        int edge1VertexCount =
            tessellatedVertexLookup.getEdgeVertexCount(
            edges.second[0].mEdgeId1);
        for (size_t c = 1; c < clusterSize; ++c) {
            if (tessellatedVertexLookup.getEdgeVertexCount(
                edges.second[c].mEdgeId0) != edge0VertexCount) {
                mismatchEdgeVertexCount = true;
                break;
            }
            if (tessellatedVertexLookup.getEdgeVertexCount(
                edges.second[c].mEdgeId1) != edge1VertexCount) {
                mismatchEdgeVertexCount = true;
                break;
            }
        }
        if (mismatchEdgeVertexCount) {
            continue;
        }
        // edge0
        for (int v = 0; v < edge0VertexCount; ++v) {
            std::vector<int> vertexCluster;
            for (size_t n = 0; n < clusterSize; ++n) {
                // edge0 end point is midEdgeVertex
                int vid = tessellatedVertexLookup.getEdgeVertexId(
                    edges.second[n].mEdgeId0, v, true);
                // may encounter vertices in unassigned face
                if (vid < 0) {
                    continue;
                }
                vertexCluster.push_back(vid);
            }
            if (vertexCluster.size() > 1) {
                vertexClusters.push_back(std::move(vertexCluster));
            }
        }
        // midEdgeVertex
        std::vector<int> vertexCluster;
        for (size_t n = 0; n < clusterSize; ++n) {
            int vid = tessellatedVertexLookup.getTessellatedVertexId(
                edges.second[n].mMidEdgeVertexId);
            // may encounter vertices in unassigned face
            if (vid < 0) {
                continue;
            }
            vertexCluster.push_back(vid);
        }
        if (vertexCluster.size() > 1) {
            vertexClusters.push_back(std::move(vertexCluster));
        }
        // edge1
        for (int v = 0; v < edge1VertexCount; ++v) {
            std::vector<int> vertexCluster;
            for (size_t n = 0; n < clusterSize; ++n) {
                // edge1 start point is midEdgeVertex
                int vid = tessellatedVertexLookup.getEdgeVertexId(
                    edges.second[n].mEdgeId1, v, false);
                // may encounter vertices in unassigned face
                if (vid < 0) {
                    continue;
                }
                vertexCluster.push_back(vid);
            }
            if (vertexCluster.size() > 1) {
                vertexClusters.push_back(std::move(vertexCluster));
            }
        }
    }
}

std::vector<SubdTessellationFactor>
OpenSubdivMesh::computeSubdTessellationFactor(const scene_rdl2::rdl2::Layer *pRdlLayer,
        const std::vector<mcrt_common::Frustum>& frustums,
        bool enableDisplacement,
        bool noTessellation) const
{
    const SubdivisionMesh::FaceVertexCount& faceVertexCount =
        mControlMeshData->mFaceVertexCount;
    const SubdivisionMesh::VertexBuffer& vertices =
        mControlMeshData->mVertices;
    const SubdivisionMesh::IndexBuffer& indices =
        mControlMeshData->mIndices;
    std::vector<SubdTessellationFactor> tessellationFactors;
    // only do adaptive tessellation when adaptiveError > 0
    if (mAdaptiveError > scene_rdl2::math::sEpsilon &&
        !frustums.empty() && !noTessellation) {
        SubdTopologyIdLookup topologyIdLookup(vertices.size(), faceVertexCount,
            indices, noTessellation);
        tessellationFactors.reserve(
            topologyIdLookup.getQuadrangulatedFaceCount());

        float pixelsPerScreenHeight =
            frustums[0].mViewport[3] - frustums[0].mViewport[1] + 1;
        float pixelsPerEdge = mAdaptiveError;
        float edgesPerScreenHeight = pixelsPerScreenHeight / pixelsPerEdge;

        std::unordered_map<int, int> edgeTessellationFactor;
        size_t motionSampleCount = vertices.get_time_steps();
        int indexOffset = 0;
        for (size_t f = 0; f < faceVertexCount.size(); ++f) {
            int nFv = faceVertexCount[f];
            scene_rdl2::math::BBox3f bbox(vertices(indices[indexOffset]).asVec3f());
            for (int v = 0; v < nFv; ++v) {
                for (size_t t = 0; t < motionSampleCount; ++t) {
                    bbox.extend(vertices(indices[indexOffset + v], t));
                }
            }
            // enlarge the bounding box with its diag length and
            // optional user provide displacement bound padding to avoid
            // case that undisplaced face got culled out unintionally
            Vec3f pCenter = scene_rdl2::math::center(bbox);
            float padding = 0.0f;
            if (enableDisplacement) {
                int assignmentId = getControlFaceAssignmentId(f);
                if (assignmentId != -1) {
                    const scene_rdl2::rdl2::Displacement* displacement =
                        pRdlLayer->lookupDisplacement(assignmentId);
                    if (displacement) {
                        padding = displacement->get(
                            scene_rdl2::rdl2::Displacement::sBoundPadding);
                        if (padding < 0.0f) {
                            padding = 0.0f;
                        }
                    }
                }
            }
            Vec3f radius(0.5f * scene_rdl2::math::length(bbox.size()) + padding);
            bbox.lower = pCenter - radius;
            bbox.upper = pCenter + radius;
            // frustum culling test
            bool inFrustum = false;
            for (size_t i = 0; i < frustums.size(); ++i) {
                if (frustums[i].testBBoxOverlaps(bbox)) {
                    inFrustum = true;
                    break;
                }
            }

            if (nFv == sQuadVertexCount) {
                SubdTessellationFactor factor;
                // regular face
                for (size_t v = 0; v < sQuadVertexCount; ++v) {
                    int vid0 = indices[indexOffset + v];
                    int vid1 = indices[indexOffset + (v + 1) % nFv];
                    int eid = topologyIdLookup.getEdgeId(vid0, vid1);
                    int vidMid = topologyIdLookup.getEdgeChildVertex(eid);
                    factor.mEdgeId0[v] =
                        topologyIdLookup.getEdgeId(vid0, vidMid);
                    factor.mEdgeId1[v] =
                        topologyIdLookup.getEdgeId(vidMid, vid1);

                    for (size_t t = 0; t < motionSampleCount; ++t) {
                        const Vec3f& v0 = vertices(vid0, t).asVec3f();
                        const Vec3f& v1 = vertices(vid1, t).asVec3f();
                        Vec3f vMid = 0.5f * (v0 + v1);
                        int edge0Factor = 0;
                        int edge1Factor = 0;
                        if (inFrustum) {
                            edge0Factor = computeEdgeVertexCount(v0, vMid,
                                edgesPerScreenHeight, frustums[0].mC2S);
                            edge1Factor = computeEdgeVertexCount(vMid, v1,
                                edgesPerScreenHeight, frustums[0].mC2S);
                        }
                        edgeTessellationFactor[factor.mEdgeId0[v]] = scene_rdl2::math::max(
                            edge0Factor,
                            edgeTessellationFactor[factor.mEdgeId0[v]]);
                        edgeTessellationFactor[factor.mEdgeId1[v]] = scene_rdl2::math::max(
                            edge1Factor,
                            edgeTessellationFactor[factor.mEdgeId1[v]]);
                    }
                }
                tessellationFactors.push_back(factor);
            } else {
                // irregular face
                Vec3f vCenter(0.0f);
                for (int v = 0; v < nFv; ++v) {
                    vCenter += vertices(indices[indexOffset + v]).asVec3f();
                }
                vCenter /= (float)nFv;
                int vidCenter = topologyIdLookup.getFaceChildVertex(f);
                for (int v = 0; v < nFv; ++v) {
                    SubdTessellationFactor factor;
                    int vid0 = indices[indexOffset + v];
                    int vid1 = indices[indexOffset + (v + 1) % nFv];
                    int vidn_1 = indices[indexOffset + (v + nFv - 1) % nFv];
                    int eid0 = topologyIdLookup.getEdgeId(vid0, vid1);
                    int vidMid0 = topologyIdLookup.getEdgeChildVertex(eid0);
                    int eidn_1 = topologyIdLookup.getEdgeId(vid0, vidn_1);
                    int vidMidn_1 = topologyIdLookup.getEdgeChildVertex(eidn_1);
                    factor.mEdgeId0[0] =
                        topologyIdLookup.getEdgeId(vid0, vidMid0);
                    factor.mEdgeId0[1] =
                        topologyIdLookup.getEdgeId(vidMid0, vidCenter);
                    factor.mEdgeId0[2] =
                        topologyIdLookup.getEdgeId(vidCenter, vidMidn_1);
                    factor.mEdgeId0[3] =
                        topologyIdLookup.getEdgeId(vidMidn_1, vid0);

                    for (size_t t = 0; t < motionSampleCount; ++t) {
                        const Vec3f& v0 = vertices(vid0, t).asVec3f();
                        const Vec3f& v1 = vertices(vid1, t).asVec3f();
                        const Vec3f& vn_1 = vertices(vidn_1, t).asVec3f();
                        Vec3f vMid0 = 0.5f * (v0 + v1);
                        Vec3f vMidN_1 = 0.5f * (v0 + vn_1);
                        int edge0Factor = 0;
                        int edgeMidCenterFactor = 0;
                        int edgen_1Factor = 0;
                        if (inFrustum) {
                            edge0Factor = computeEdgeVertexCount(
                                v0, vMid0,
                                edgesPerScreenHeight, frustums[0].mC2S);
                            edgeMidCenterFactor = computeEdgeVertexCount(
                                vMid0, vCenter,
                                edgesPerScreenHeight, frustums[0].mC2S);
                            edgen_1Factor = computeEdgeVertexCount(
                                v0, vMidN_1,
                                edgesPerScreenHeight, frustums[0].mC2S);
                        }
                        edgeTessellationFactor[factor.mEdgeId0[0]] = scene_rdl2::math::max(
                            edge0Factor,
                            edgeTessellationFactor[factor.mEdgeId0[0]]);
                        edgeTessellationFactor[factor.mEdgeId0[1]] = scene_rdl2::math::max(
                            edgeMidCenterFactor,
                            edgeTessellationFactor[factor.mEdgeId0[1]]);
                        // the third edge that we omit here will be covered
                        // by neighbor quadrangulated quad so we can skip the
                        // effort calculating it twice
                        edgeTessellationFactor[factor.mEdgeId0[3]] = scene_rdl2::math::max(
                            edgen_1Factor,
                            edgeTessellationFactor[factor.mEdgeId0[3]]);
                    }
                    factor.mEdgeId1[0] = -1;
                    factor.mEdgeId1[1] = -1;
                    factor.mEdgeId1[2] = -1;
                    factor.mEdgeId1[3] = -1;
                    tessellationFactors.push_back(factor);
                }
            }
            indexOffset += nFv;
        }
        // Clamp the maximum tessellation factor based on user specified
        // mesh resolution. Otherwise the tessellation factor can get out
        // of control when the edge is extremely close to camera near plane
        int maxEdgeVertexCount = mMeshResolution / 2 - 1;
        for (size_t i = 0; i < tessellationFactors.size(); ++i) {
            for (size_t e = 0; e < sQuadVertexCount; ++e) {
                int eid0 = tessellationFactors[i].mEdgeId0[e];
                tessellationFactors[i].mEdge0Factor[e] = scene_rdl2::math::clamp(
                    edgeTessellationFactor[eid0], 0, maxEdgeVertexCount);
                int eid1 = tessellationFactors[i].mEdgeId1[e];
                tessellationFactors[i].mEdge1Factor[e] = scene_rdl2::math::clamp(
                    edgeTessellationFactor[eid1], 0, maxEdgeVertexCount);
            }
        }
    } else {
        tessellationFactors.reserve(std::accumulate(
            faceVertexCount.begin(), faceVertexCount.end(), 0,
            [](int quadCount, int nFv)->int {
                return nFv == sQuadVertexCount ? quadCount + 1 : quadCount + nFv;
            })
        );
        int edgeVertexCount = scene_rdl2::math::max(0, mMeshResolution / 2 - 1);
        for (size_t f = 0; f < faceVertexCount.size(); ++f) {
            int nFv = faceVertexCount[f];
            if (nFv == sQuadVertexCount) {
                SubdTessellationFactor factor;
                // regular face
                for (size_t v = 0; v < sQuadVertexCount; ++v) {
                    factor.mEdge0Factor[v] = edgeVertexCount;
                    factor.mEdge1Factor[v] = edgeVertexCount;
                }
                tessellationFactors.push_back(factor);
            } else {
                // irregular face
                for (int c = 0; c < nFv; ++c) {
                    SubdTessellationFactor factor;
                    for (size_t v = 0; v < sQuadVertexCount; ++v) {
                        factor.mEdge0Factor[v] = edgeVertexCount;
                        factor.mEdge1Factor[v] = 0;
                    }
                    tessellationFactors.push_back(factor);
                }
            }
        }
    }
    return tessellationFactors;
}

// loop through all control faces, quadrangulate n-gons and put all the
// needed topology info into SubdQuadTopology.
std::vector<SubdQuadTopology>
OpenSubdivMesh::generateSubdQuadTopology(
        const scene_rdl2::rdl2::Layer* pRdlLayer,
        const SubdTopologyIdLookup& topologyIdLookup,
        const SubdivisionMesh::FaceVertexCount& faceVertexCount,
        const SubdivisionMesh::IndexBuffer& faceVaryingIndices,
        bool noTessellation) const
{
    std::vector<SubdQuadTopology> quadTopologies;
    quadTopologies.reserve(topologyIdLookup.getQuadrangulatedFaceCount());
    // the code logic is similar to Far::PtexIndices::initializePtexIndices
    int indexOffset = 0;
    for (size_t f = 0; f < faceVertexCount.size(); ++f) {
        int assignmentId = getControlFaceAssignmentId(f);
        bool hasAssignment = assignmentId != -1 &&
            (pRdlLayer->lookupMaterial(assignmentId) != nullptr ||
             pRdlLayer->lookupVolumeShader(assignmentId) != nullptr);
        int nFv = faceVertexCount[f];
        if (nFv == sQuadVertexCount) {
            // regular face
            //
            // *--x--*  *: corner vertex
            // |     |  x: mid edge vertex
            // x     x  -: edge0, edge1
            // |     |
            // *--x--*
            SubdQuadTopology quadTopology;
            for (size_t v = 0; v < sQuadVertexCount; ++v) {
                int vid0 = faceVaryingIndices[indexOffset + v];
                int vid1 = faceVaryingIndices[indexOffset + (v + 1) % nFv];
                if (noTessellation) {
                    quadTopology.mCornerVertexId[v] = vid0;
                    quadTopology.mMidEdgeVertexId[v] = -1;
                    quadTopology.mEdgeId0[v] = -1;
                    quadTopology.mEdgeId1[v] = -1;
                } else {
                    int eid = topologyIdLookup.getEdgeId(vid0, vid1);
                    int vMidId = topologyIdLookup.getEdgeChildVertex(eid);
                    quadTopology.mCornerVertexId[v] = vid0;
                    quadTopology.mMidEdgeVertexId[v] = vMidId;
                    quadTopology.mEdgeId0[v] =
                        topologyIdLookup.getEdgeId(vid0, vMidId);
                    quadTopology.mEdgeId1[v] =
                        topologyIdLookup.getEdgeId(vMidId, vid1);
                }
            }
            quadTopology.mControlFaceId = f;
            quadTopology.mHasAssignment = hasAssignment;
            quadTopology.mNonQuadParent = false;
            quadTopologies.push_back(quadTopology);
        } else {
            // irregular face
            for (int c = 0; c < nFv; ++c) {
                SubdQuadTopology quadTopology;
                if (noTessellation) {
                    int cv0 = faceVaryingIndices[indexOffset + c];
                    quadTopology.mCornerVertexId[0] = cv0;
                    // explicitly assign invalid values
                    quadTopology.mCornerVertexId[1] = -1;
                    quadTopology.mCornerVertexId[2] = -1;
                    quadTopology.mCornerVertexId[3] = -1;
                    for (size_t i = 0; i < sQuadVertexCount; ++i) {
                        quadTopology.mMidEdgeVertexId[i] = -1;
                        quadTopology.mEdgeId0[i] = -1;
                        quadTopology.mEdgeId1[i] = -1;
                    }
                } else {
                    int cv0 = faceVaryingIndices[indexOffset + c];
                    int cv1 = faceVaryingIndices[indexOffset + (c + 1) % nFv];
                    int cvn_1 = faceVaryingIndices[
                        indexOffset + (c + nFv - 1) % nFv];
                    int eid0 = topologyIdLookup.getEdgeId(cv0, cv1);
                    int eidn_1 = topologyIdLookup.getEdgeId(cv0, cvn_1);
                    // *--*  *: corner vertex
                    // |  |  -: edge0
                    // *--*
                    int v[sQuadVertexCount];
                    v[0] = topologyIdLookup.getVertexChildVertex(cv0);
                    v[1] = topologyIdLookup.getEdgeChildVertex(eid0);
                    v[2] = topologyIdLookup.getFaceChildVertex(f);
                    v[3] = topologyIdLookup.getEdgeChildVertex(eidn_1);
                    int e[sQuadVertexCount];
                    e[0] = topologyIdLookup.getEdgeId(v[0], v[1]);
                    e[1] = topologyIdLookup.getEdgeId(v[1], v[2]);
                    e[2] = topologyIdLookup.getEdgeId(v[2], v[3]);
                    e[3] = topologyIdLookup.getEdgeId(v[3], v[0]);
                    for (size_t i = 0; i < sQuadVertexCount; ++i) {
                        quadTopology.mCornerVertexId[i] = v[i];
                        quadTopology.mEdgeId0[i] = e[i];
                        // explicitly assign invalid values
                        quadTopology.mMidEdgeVertexId[i] = -1;
                        quadTopology.mEdgeId1[i] = -1;
                    }
                }
                quadTopology.mControlFaceId = f;
                quadTopology.mHasAssignment = hasAssignment;
                quadTopology.mNonQuadParent = true;
                quadTopologies.push_back(quadTopology);
            }
        }
        indexOffset += nFv;
    }
    return quadTopologies;
}

// LimitSurfaceSample contains the ingredients to come up a final
// limit surface sample point. OpenSubdiv use these ingredient to find patch
// to evaluate
struct LimitSurfaceSample
{
    LimitSurfaceSample(): mFaceId(-1), mAssignmentId(-1) {}
    int mFaceId;
    int mAssignmentId;
    Vec2f mUv;
    Vec2f mUvDelta;
};

// Texture st coverage range for each tessellated vertex. We use this to
// determine the mip selection level during displacement stage. Similar to
// what ray footprint does during mcrt shading stage
struct DisplacementFootprint
{
    Vec2f mDst[2];
};

// A special case that all edge tessellation factor are 0 in the control quad
// We can generate more optimal index buffer by handling this case explicitly
void
generateZeroStitchIndexBufferAndSurfaceSamples(
        const SubdQuadTopology& quadTopology,
        const SubdTessellatedVertexLookup& tessellatedVertexLookup,
        int fid, int assignmentId, SubdivisionMesh::IndexBuffer& indices,
        std::vector<LimitSurfaceSample>& limitSurfaceSamples,
        std::vector<int>* tessellatedToControlFace)
{
    int v0 = tessellatedVertexLookup.getTessellatedVertexId(
        quadTopology.mCornerVertexId[0]);
    int v1 = tessellatedVertexLookup.getTessellatedVertexId(
        quadTopology.mCornerVertexId[1]);
    int v2 = tessellatedVertexLookup.getTessellatedVertexId(
        quadTopology.mCornerVertexId[2]);
    int v3 = tessellatedVertexLookup.getTessellatedVertexId(
        quadTopology.mCornerVertexId[3]);

    if (quadTopology.nonQuadParent()) {
        indices.push_back(v0);
        indices.push_back(v1);
        indices.push_back(v2);
        indices.push_back(v3);
        if (tessellatedToControlFace) {
            tessellatedToControlFace->push_back(quadTopology.mControlFaceId);
        }

        limitSurfaceSamples[v0].mUvDelta = Vec2f(1.0f, 1.0f);
        limitSurfaceSamples[v1].mUvDelta = Vec2f(1.0f, 1.0f);
        limitSurfaceSamples[v2].mUvDelta = Vec2f(1.0f, 1.0f);
        limitSurfaceSamples[v3].mUvDelta = Vec2f(1.0f, 1.0f);
    } else {
        int mv0 = tessellatedVertexLookup.getTessellatedVertexId(
            quadTopology.mMidEdgeVertexId[0]);
        int mv1 = tessellatedVertexLookup.getTessellatedVertexId(
            quadTopology.mMidEdgeVertexId[1]);
        int mv2 = tessellatedVertexLookup.getTessellatedVertexId(
            quadTopology.mMidEdgeVertexId[2]);
        int mv3 = tessellatedVertexLookup.getTessellatedVertexId(
            quadTopology.mMidEdgeVertexId[3]);
        int center = tessellatedVertexLookup.getInteriorVertexId(
            fid, 0, 0, 1);
        //  v3----mv2-----v2
        //  |      |      |
        //  |      |      |
        //  mv3---cen-----mv1
        //  |      |      |
        //  |      |      |
        //  v0----mv0----v1
        indices.push_back(v0);
        indices.push_back(mv0);
        indices.push_back(center);
        indices.push_back(mv3);

        indices.push_back(mv0);
        indices.push_back(v1);
        indices.push_back(mv1);
        indices.push_back(center);

        indices.push_back(center);
        indices.push_back(mv1);
        indices.push_back(v2);
        indices.push_back(mv2);

        indices.push_back(mv3);
        indices.push_back(center);
        indices.push_back(mv2);
        indices.push_back(v3);
        if (tessellatedToControlFace) {
            tessellatedToControlFace->push_back(quadTopology.mControlFaceId);
            tessellatedToControlFace->push_back(quadTopology.mControlFaceId);
            tessellatedToControlFace->push_back(quadTopology.mControlFaceId);
            tessellatedToControlFace->push_back(quadTopology.mControlFaceId);
        }

        limitSurfaceSamples[mv0].mFaceId = fid;
        limitSurfaceSamples[mv0].mAssignmentId = assignmentId;
        limitSurfaceSamples[mv0].mUv = Vec2f(0.5f, 0.0f);
        limitSurfaceSamples[mv0].mUvDelta = Vec2f(0.5f, 0.5f);
        limitSurfaceSamples[mv1].mFaceId = fid;
        limitSurfaceSamples[mv1].mAssignmentId = assignmentId;
        limitSurfaceSamples[mv1].mUv = Vec2f(1.0f, 0.5f);
        limitSurfaceSamples[mv1].mUvDelta = Vec2f(0.5f, 0.5f);
        limitSurfaceSamples[mv2].mFaceId = fid;
        limitSurfaceSamples[mv2].mAssignmentId = assignmentId;
        limitSurfaceSamples[mv2].mUv = Vec2f(0.5f, 1.0f);
        limitSurfaceSamples[mv2].mUvDelta = Vec2f(0.5f, 0.5f);
        limitSurfaceSamples[mv3].mFaceId = fid;
        limitSurfaceSamples[mv3].mAssignmentId = assignmentId;
        limitSurfaceSamples[mv3].mUv = Vec2f(0.0f, 0.5f);
        limitSurfaceSamples[mv3].mUvDelta = Vec2f(0.5f, 0.5f);
        limitSurfaceSamples[center].mFaceId = fid;
        limitSurfaceSamples[center].mAssignmentId = assignmentId;
        limitSurfaceSamples[center].mUv = Vec2f(0.5f, 0.5f);
        limitSurfaceSamples[center].mUvDelta = Vec2f(0.5f, 0.5f);

        limitSurfaceSamples[v0].mUvDelta = Vec2f(0.5f, 0.5f);
        limitSurfaceSamples[v1].mUvDelta = Vec2f(0.5f, 0.5f);
        limitSurfaceSamples[v2].mUvDelta = Vec2f(0.5f, 0.5f);
        limitSurfaceSamples[v3].mUvDelta = Vec2f(0.5f, 0.5f);
    }

    limitSurfaceSamples[v0].mFaceId = fid;
    limitSurfaceSamples[v0].mAssignmentId = assignmentId;
    limitSurfaceSamples[v0].mUv = Vec2f(0.0f, 0.0f);
    limitSurfaceSamples[v1].mFaceId = fid;
    limitSurfaceSamples[v1].mAssignmentId = assignmentId;
    limitSurfaceSamples[v1].mUv = Vec2f(1.0f, 0.0f);
    limitSurfaceSamples[v2].mFaceId = fid;
    limitSurfaceSamples[v2].mAssignmentId = assignmentId;
    limitSurfaceSamples[v2].mUv = Vec2f(1.0f, 1.0f);
    limitSurfaceSamples[v3].mFaceId = fid;
    limitSurfaceSamples[v3].mAssignmentId = assignmentId;
    limitSurfaceSamples[v3].mUv = Vec2f(0.0f, 1.0f);
}

// This method generated the final tessellated index buffer and a list
// of LimitSurfaceSample, which is the ingredient to cook out the final
// tessellated vertex buffer
void
OpenSubdivMesh::generateIndexBufferAndSurfaceSamples(
        const std::vector<SubdQuadTopology>& quadTopologies,
        const SubdTessellatedVertexLookup& tessellatedVertexLookup,
        bool noTessellation,
        SubdivisionMesh::IndexBuffer& indices,
        std::vector<LimitSurfaceSample>& limitSurfaceSamples,
        std::vector<int>* tessellatedToControlFace) const
{
    // simplest case, only render control cage
    if (noTessellation) {
        generateControlMeshIndexBufferAndSurfaceSamples(quadTopologies,
            tessellatedVertexLookup, indices, limitSurfaceSamples,
            tessellatedToControlFace);
        return;
    }
    // we need to use arena to allocate temporary data structure for
    // outer/interior ring stitching
    mcrt_common::ThreadLocalState *tls = mcrt_common::getFrameUpdateTLS();
    scene_rdl2::alloc::Arena *arena = &tls->mArena;

    size_t tessellatedVertexCount =
        tessellatedVertexLookup.getTessellatedVertexCount();
    limitSurfaceSamples.resize(tessellatedVertexCount);
    for (size_t f = 0; f < quadTopologies.size(); ++f) {
        const SubdQuadTopology& quadTopology = quadTopologies[f];
        if (!quadTopology.mHasAssignment) {
            continue;
        }
        int assignmentId = getControlFaceAssignmentId(
            quadTopology.mControlFaceId);
        if (tessellatedVertexLookup.noRingToStitch(quadTopology)) {
            // the case we can easily construct 4 quads for regular quad
            // control face and 1 quad for quadrangulated child quad from
            // irregular n-gons control face
            generateZeroStitchIndexBufferAndSurfaceSamples(quadTopology,
                tessellatedVertexLookup, f, assignmentId, indices,
                limitSurfaceSamples, tessellatedToControlFace);
            continue;
        }

        // interior region index buffer first.
        // *--------*--------*
        // |                 |
        // |  -------------  *
        // *  |   |   |   |  |
        // |  -------------  *
        // |  |   |   |   |  |
        // *  -------------  *
        // |  |   |   |   |  |
        // |  -------------  *
        // *  |   |   |   |  |
        // |  -------------  *
        // |                 |
        // *--*---*---*---*--*
        // index buffer
        int interiorRowVertexCount =
            tessellatedVertexLookup.getInteriorRowVertexCount(quadTopology);
        int interiorColVertexCount =
            tessellatedVertexLookup.getInteriorColVertexCount(quadTopology);
        for (int i = 0; i < interiorRowVertexCount - 1; ++i) {
            for (int j = 0; j < interiorColVertexCount - 1; ++j) {
                indices.push_back(tessellatedVertexLookup.getInteriorVertexId(
                    f, i, j, interiorColVertexCount));
                indices.push_back(tessellatedVertexLookup.getInteriorVertexId(
                    f, i, j + 1, interiorColVertexCount));
                indices.push_back(tessellatedVertexLookup.getInteriorVertexId(
                    f, i + 1, j + 1, interiorColVertexCount));
                indices.push_back(tessellatedVertexLookup.getInteriorVertexId(
                    f, i + 1, j, interiorColVertexCount));
                if (tessellatedToControlFace) {
                    tessellatedToControlFace->push_back(
                        quadTopology.mControlFaceId);
                }
            }
        }
        float du = 1.0f / (float)(1 + interiorColVertexCount);
        float dv = 1.0f / (float)(1 + interiorRowVertexCount);
        // limit surface samples
        for (int i = 0; i < interiorRowVertexCount; ++i) {
            for (int j = 0; j < interiorColVertexCount; ++j) {
                int vid = tessellatedVertexLookup.getInteriorVertexId(
                    f, i, j, interiorColVertexCount);
                limitSurfaceSamples[vid].mFaceId = f;
                limitSurfaceSamples[vid].mAssignmentId = assignmentId;
                limitSurfaceSamples[vid].mUv = Vec2f((j + 1) * du, (i + 1) * dv);
                limitSurfaceSamples[vid].mUvDelta = Vec2f(du, dv);
            }
        }

        // stitch bottom interior and outer ring edge
        //     *---*--*--*---*
        //    / \ / \ | / \ / \
        //   *---*----*----*---*
        {
            SCOPED_MEM(arena);
            StitchPoint* innerRing = nullptr;
            StitchPoint* outerRing = nullptr;
            int innerRingVertexCount, outerRingVertexCount;
            generateSubdStitchRings(arena, quadTopology, f, 0,
                tessellatedVertexLookup,
                interiorRowVertexCount, interiorColVertexCount,
                innerRing, innerRingVertexCount,
                outerRing, outerRingVertexCount);
            stitchRings(innerRing, innerRingVertexCount,
                outerRing, outerRingVertexCount, indices,
                quadTopology.mControlFaceId, tessellatedToControlFace);
            // limit surface samples
            for (int i = 0; i < outerRingVertexCount; ++i) {
                int vid = outerRing[i].mVertexId;
                limitSurfaceSamples[vid].mFaceId = f;
                limitSurfaceSamples[vid].mAssignmentId = assignmentId;
                limitSurfaceSamples[vid].mUv = Vec2f(outerRing[i].mT, 0.0f);
                limitSurfaceSamples[vid].mUvDelta = Vec2f(du, dv);
            }
        }
        // stitch right interior and outer ring edge
        //      *
        //     /|
        //    / |
        //   *--*
        //   |\ |
        //   *--*
        //   | /|
        //   *--*
        //   \  |
        //    \ |
        //      *
        {
            SCOPED_MEM(arena);
            StitchPoint* innerRing = nullptr;
            StitchPoint* outerRing = nullptr;
            int innerRingVertexCount, outerRingVertexCount;
            generateSubdStitchRings(arena, quadTopology, f, 1,
                tessellatedVertexLookup,
                interiorRowVertexCount, interiorColVertexCount,
                innerRing, innerRingVertexCount,
                outerRing, outerRingVertexCount);
            stitchRings(innerRing, innerRingVertexCount,
                outerRing, outerRingVertexCount, indices,
                quadTopology.mControlFaceId, tessellatedToControlFace);
            // limit surface samples
            for (int i = 0; i < outerRingVertexCount; ++i) {
                int vid = outerRing[i].mVertexId;
                limitSurfaceSamples[vid].mFaceId = f;
                limitSurfaceSamples[vid].mAssignmentId = assignmentId;
                limitSurfaceSamples[vid].mUv = Vec2f(1.0f , outerRing[i].mT);
                limitSurfaceSamples[vid].mUvDelta = Vec2f(du, dv);
            }
        }
        // stitch top interior and outer ring edge
        //   *---*----*----*---*
        //    \ / \ / | \ / \ /
        //     *---*--*--*---*
        {
            SCOPED_MEM(arena);
            StitchPoint* innerRing = nullptr;
            StitchPoint* outerRing = nullptr;
            int innerRingVertexCount, outerRingVertexCount;
            generateSubdStitchRings(arena, quadTopology, f, 2,
                tessellatedVertexLookup,
                interiorRowVertexCount, interiorColVertexCount,
                innerRing, innerRingVertexCount,
                outerRing, outerRingVertexCount);
            stitchRings(innerRing, innerRingVertexCount,
                outerRing, outerRingVertexCount, indices,
                quadTopology.mControlFaceId, tessellatedToControlFace);
            // limit surface samples
            for (int i = 0; i < outerRingVertexCount; ++i) {
                int vid = outerRing[i].mVertexId;
                limitSurfaceSamples[vid].mFaceId = f;
                limitSurfaceSamples[vid].mAssignmentId = assignmentId;
                limitSurfaceSamples[vid].mUv =
                    Vec2f(1.0f - outerRing[i].mT, 1.0f);
                limitSurfaceSamples[vid].mUvDelta = Vec2f(du, dv);
            }
        }
        // stitch left interior and outer ring edge
        //   *
        //   |\
        //   | \
        //   *--*
        //   | /|
        //   *--*
        //   |\ |
        //   *--*
        //   |  /
        //   | /
        //   *
        {
            SCOPED_MEM(arena);
            StitchPoint* innerRing = nullptr;
            StitchPoint* outerRing = nullptr;
            int innerRingVertexCount, outerRingVertexCount;
            generateSubdStitchRings(arena, quadTopology, f, 3,
                tessellatedVertexLookup,
                interiorRowVertexCount, interiorColVertexCount,
                innerRing, innerRingVertexCount,
                outerRing, outerRingVertexCount);
            stitchRings(innerRing, innerRingVertexCount,
                outerRing, outerRingVertexCount, indices,
                quadTopology.mControlFaceId, tessellatedToControlFace);
            // limit surface samples
            for (int i = 0; i < outerRingVertexCount; ++i) {
                int vid = outerRing[i].mVertexId;
                limitSurfaceSamples[vid].mFaceId = f;
                limitSurfaceSamples[vid].mAssignmentId = assignmentId;
                limitSurfaceSamples[vid].mUv =
                    Vec2f(0.0f, 1.0f - outerRing[i].mT);
                limitSurfaceSamples[vid].mUvDelta = Vec2f(du, dv);
            }
        }
    }
}

// A special case that we just render the control cage (the case that
// subd resolution <= 1). This is still a common production case when
// the mesh are simple enough or far from camera with minimum LOD required
void
OpenSubdivMesh::generateControlMeshIndexBufferAndSurfaceSamples(
        const std::vector<SubdQuadTopology>& quadTopologies,
        const SubdTessellatedVertexLookup& tessellatedVertexLookup,
        SubdivisionMesh::IndexBuffer& indices,
        std::vector<LimitSurfaceSample>& limitSurfaceSamples,
        std::vector<int>* tessellatedToControlFace) const
{
    size_t tessellatedVertexCount =
        tessellatedVertexLookup.getTessellatedVertexCount();
    limitSurfaceSamples.resize(tessellatedVertexCount);
    for (size_t f = 0; f < quadTopologies.size();) {
        const SubdQuadTopology& quadTopology = quadTopologies[f];
        if (!quadTopology.mHasAssignment) {
            ++f;
            continue;
        }
        int assignmentId = getControlFaceAssignmentId(
            quadTopology.mControlFaceId);

        if (quadTopology.nonQuadParent()) {
            size_t nFv = 1;
            int controlFaceId = quadTopology.mControlFaceId;
            while ((f + nFv) < quadTopologies.size() &&
                quadTopologies[f + nFv].mControlFaceId == controlFaceId) {
                ++nFv;
            }
            MNRY_ASSERT_REQUIRE(nFv >= 3);
            // split this n-gons into a series of degened quad (triangle)
            int v0 = tessellatedVertexLookup.getTessellatedVertexId(
                quadTopology.mCornerVertexId[0]);
            for (size_t offset = 1; offset <= (nFv - 2); ++offset) {
                int v1 = tessellatedVertexLookup.getTessellatedVertexId(
                    quadTopologies[f + offset].mCornerVertexId[0]);
                int v2 = tessellatedVertexLookup.getTessellatedVertexId(
                    quadTopologies[f + offset + 1].mCornerVertexId[0]);
                indices.push_back(v0);
                indices.push_back(v1);
                indices.push_back(v2);
                indices.push_back(v2);
                if (tessellatedToControlFace) {
                    tessellatedToControlFace->push_back(controlFaceId);
                }
            }
            for (size_t offset = 0; offset < nFv; ++offset) {
                int v = tessellatedVertexLookup.getTessellatedVertexId(
                    quadTopologies[f + offset].mCornerVertexId[0]);
                limitSurfaceSamples[v].mFaceId = f + offset;
                limitSurfaceSamples[v].mAssignmentId = assignmentId;
                limitSurfaceSamples[v].mUv = Vec2f(0.0f, 0.0f);
                limitSurfaceSamples[v].mUvDelta = Vec2f(1.0f, 1.0f);
            }
            f += nFv;
        } else {
            int v0 = tessellatedVertexLookup.getTessellatedVertexId(
                quadTopology.mCornerVertexId[0]);
            int v1 = tessellatedVertexLookup.getTessellatedVertexId(
                quadTopology.mCornerVertexId[1]);
            int v2 = tessellatedVertexLookup.getTessellatedVertexId(
                quadTopology.mCornerVertexId[2]);
            int v3 = tessellatedVertexLookup.getTessellatedVertexId(
                quadTopology.mCornerVertexId[3]);
            indices.push_back(v0);
            indices.push_back(v1);
            indices.push_back(v2);
            indices.push_back(v3);
            if (tessellatedToControlFace) {
                tessellatedToControlFace->push_back(quadTopology.mControlFaceId);
            }

            limitSurfaceSamples[v0].mFaceId = f;
            limitSurfaceSamples[v0].mAssignmentId = assignmentId;
            limitSurfaceSamples[v0].mUv = Vec2f(0.0f, 0.0f);
            limitSurfaceSamples[v0].mUvDelta = Vec2f(1.0f, 1.0f);
            limitSurfaceSamples[v1].mFaceId = f;
            limitSurfaceSamples[v1].mAssignmentId = assignmentId;
            limitSurfaceSamples[v1].mUv = Vec2f(1.0f, 0.0f);
            limitSurfaceSamples[v1].mUvDelta = Vec2f(1.0f, 1.0f);
            limitSurfaceSamples[v2].mFaceId = f;
            limitSurfaceSamples[v2].mAssignmentId = assignmentId;
            limitSurfaceSamples[v2].mUv = Vec2f(1.0f, 1.0f);
            limitSurfaceSamples[v2].mUvDelta = Vec2f(1.0f, 1.0f);
            limitSurfaceSamples[v3].mFaceId = f;
            limitSurfaceSamples[v3].mAssignmentId = assignmentId;
            limitSurfaceSamples[v3].mUv = Vec2f(0.0f, 1.0f);
            limitSurfaceSamples[v3].mUvDelta = Vec2f(1.0f, 1.0f);
            ++f;
        }
    }
}

void
generatePatchCvs(const OpenSubdiv::Far::TopologyRefiner* refiner,
        const OpenSubdiv::Far::PatchTable* patchTable,
        const SubdivisionMesh::VertexBuffer& controlVertices,
        const Vector<Vec2f>& textureVertices,
        AttributeRate textureRate,
        std::vector<PatchCV>& patchCvs,
        std::vector<TextureCV>& textureCvs,
        bool requireUniformFix,
        uint motionSampleCount)
{
    // compute the total number of points we need to evaluate patchTable
    // we use local points around extraordinary features.
    int nRefinerVertices = refiner->GetNumVerticesTotal();
    int nLocalPoints = patchTable->GetNumLocalPoints();
    // create a buffer to hold the position of the refined vertices and
    // local points, then copy the control vertices at the beginning.
    patchCvs.resize(nRefinerVertices + nLocalPoints, PatchCV(motionSampleCount));
    for (size_t i = 0; i < controlVertices.size(); ++i) {
        for (size_t t = 0; t < motionSampleCount; ++t) {
            patchCvs[i].mData[t] = controlVertices(i, t);
        }
    }
    // adaptive refinement may result in fewer levels than maxIsolation
    int nRefinedLevels = refiner->GetNumLevels();
    // interpolate patchCvs: they will be the control vertices
    // of the limit patches
    PatchCV* src = &patchCvs[0];
    for (int level = 1; level < nRefinedLevels; ++level) {
        PatchCV* dst = src + refiner->GetLevel(level - 1).GetNumVertices();
        OpenSubdiv::Far::PrimvarRefiner(*refiner).Interpolate(level, src, dst);
        src = dst;
    }

    // evaluate local points from interpolated patchCvs
    patchTable->ComputeLocalPointValues(&patchCvs[0],
        &patchCvs[nRefinerVertices]);

    // allocate and initialize textureCvs if we have texture st provided
    if (textureRate == RATE_FACE_VARYING) {
        int nFVarRefinerVertices = refiner->GetNumFVarValuesTotal(
            sTextureStFVarChannel);
        int nFVarLocalPoints = patchTable->GetNumLocalPointsFaceVarying(
            sTextureStFVarChannel);
        textureCvs.resize(nFVarRefinerVertices + nFVarLocalPoints);
        for (size_t i = 0; i < textureVertices.size(); ++i) {
            textureCvs[i].mSt = textureVertices[i];
        }

        TextureCV* src = &textureCvs[0];
        for (int level = 1; level < nRefinedLevels; ++level) {
            TextureCV* dst = src +
                refiner->GetLevel(level - 1).GetNumFVarValues(
                sTextureStFVarChannel);
            OpenSubdiv::Far::PrimvarRefiner(*refiner).InterpolateFaceVarying(
                level, src, dst, sTextureStFVarChannel);
            src = dst;
        }
        patchTable->ComputeLocalPointValuesFaceVarying(&textureCvs[0],
            &textureCvs[nFVarRefinerVertices], sTextureStFVarChannel);

        // see comment in variable initialized location for detail explanation
        // why we need to do this OpenSubdiv-3.1 workaround.
        if (requireUniformFix) {
            size_t fvarPatchPointOffset =
                refiner->GetLevel(0).GetNumFVarValues(sTextureStFVarChannel);
            for (size_t i = fvarPatchPointOffset; i < textureCvs.size(); ++i) {
                textureCvs[i - fvarPatchPointOffset] = textureCvs[i];
            }
        }
    } else if (textureRate == RATE_VARYING || textureRate == RATE_VERTEX) {
        textureCvs.resize(nRefinerVertices + nLocalPoints);
        for (size_t i = 0; i < textureVertices.size(); ++i) {
            textureCvs[i].mSt = textureVertices[i];
        }

        TextureCV* src = &textureCvs[0];
        for (int level = 1; level < nRefinedLevels; ++level) {
            TextureCV* dst = src +
                refiner->GetLevel(level - 1).GetNumVertices();
            OpenSubdiv::Far::PrimvarRefiner(*refiner).Interpolate(
                level, src, dst);
            src = dst;
        }
        patchTable->ComputeLocalPointValues(&textureCvs[0],
            &textureCvs[nRefinerVertices]);
    }
}

void
generatePrimVarCvs(const OpenSubdiv::Far::TopologyRefiner* refiner,
        const OpenSubdiv::Far::PatchTable* patchTable,
        Attributes* primitiveAttributes,
        FaceVaryingAttributes& faceVaryingAttributes,
        size_t controlVertexCount,
        std::vector<float>& varyingData,
        std::vector<PrimVarCV>& varyingPrimVarCvs,
        std::vector<float>& vertexData,
        std::vector<PrimVarCV>& vertexPrimVarCvs,
        std::unordered_map<int, std::vector<PrimVarCV>>& faceVaryingPrimVarCvs,
        bool requireUniformFix)
{
    // compute the total number of points we need to evaluate patchTable
    // we use local points around extraordinary features.
    int nRefinerVertices = refiner->GetNumVerticesTotal();
    int nRefinedLevels = refiner->GetNumLevels();
    if (primitiveAttributes->hasVaryingAttributes()) {
        int nLocalPoints = patchTable->GetNumLocalPointsVarying();
        size_t floatPerCV =
            primitiveAttributes->getVaryingAttributesStride() / sizeof(float);
        varyingData.resize(
            (nRefinerVertices + nLocalPoints) * floatPerCV);
        varyingPrimVarCvs.resize(nRefinerVertices + nLocalPoints);
        // this is the buffer we are going to write tessellated attributes into
        // but we need to keep its current content (per control vertex attributes)
        // during tessellation stage for patch evaluation, so we copy the
        // current content to a temporary buffer varyingData
        float* attributesData = primitiveAttributes->getVaryingAttributesData();
        std::copy(attributesData,
            attributesData + floatPerCV * controlVertexCount,
            varyingData.begin());
        for (size_t i = 0; i < varyingPrimVarCvs.size(); ++i) {
            varyingPrimVarCvs[i].mData = &(varyingData[i * floatPerCV]);
            varyingPrimVarCvs[i].mFloatPerCV= floatPerCV;
        }
        PrimVarCV* src = &varyingPrimVarCvs[0];
        for (int level = 1; level < nRefinedLevels; ++level) {
            PrimVarCV* dst = src +
                refiner->GetLevel(level - 1).GetNumVertices();
            OpenSubdiv::Far::PrimvarRefiner(*refiner).InterpolateVarying(level,
                src, dst);
            src = dst;
        }
        // evaluate local points from interpolated patchCvs
        patchTable->ComputeLocalPointValuesVarying(&varyingPrimVarCvs[0],
            &varyingPrimVarCvs[nRefinerVertices]);
    }
    if (primitiveAttributes->hasVertexAttributes()) {
        int nLocalPoints = patchTable->GetNumLocalPoints();
        size_t floatPerCV =
            primitiveAttributes->getVertexAttributesStride() / sizeof(float);
        vertexData.resize(
            (nRefinerVertices + nLocalPoints) * floatPerCV);
        vertexPrimVarCvs.resize(nRefinerVertices + nLocalPoints);
        // this is the buffer we are going to write tessellated attributes into
        // but we need to keep its current content (per control vertex attributes)
        // during tessellation stage for patch evaluation, so we copy the
        // current content to a temporary buffer vertexData
        float* attributesData = primitiveAttributes->getVertexAttributesData();
        std::copy(attributesData,
            attributesData + floatPerCV * controlVertexCount,
            vertexData.begin());
        for (size_t i = 0; i < vertexPrimVarCvs.size(); ++i) {
            vertexPrimVarCvs[i].mData = &(vertexData[i * floatPerCV]);
            vertexPrimVarCvs[i].mFloatPerCV= floatPerCV;
        }
        PrimVarCV* src = &vertexPrimVarCvs[0];
        for (int level = 1; level < nRefinedLevels; ++level) {
            PrimVarCV* dst = src +
                refiner->GetLevel(level - 1).GetNumVertices();
            OpenSubdiv::Far::PrimvarRefiner(*refiner).Interpolate(level,
                src, dst);
            src = dst;
        }
        // evaluate local points from interpolated patchCvs
        patchTable->ComputeLocalPointValues(&vertexPrimVarCvs[0],
            &vertexPrimVarCvs[nRefinerVertices]);
    }

    for (auto& key: faceVaryingAttributes.getAllKeys()) {
        auto& attributeBuffer = faceVaryingAttributes.getAttributeBuffer(key);
        size_t floatPerCV = attributeBuffer.mFloatPerVertex;
        int channel = attributeBuffer.mChannel;
        int nFVarRefinerVertices = refiner->GetNumFVarValuesTotal(channel);
        int nFVarLocalPoints = patchTable->GetNumLocalPointsFaceVarying(
            channel);
        int cvCount = nFVarRefinerVertices + nFVarLocalPoints;
        attributeBuffer.mData.resize(floatPerCV * cvCount);
        std::vector<PrimVarCV> primVarCVs(cvCount);
        for (size_t i = 0; i < primVarCVs.size(); ++i) {
            primVarCVs[i].mData = &(attributeBuffer.mData[i * floatPerCV]);
            primVarCVs[i].mFloatPerCV = floatPerCV;
        }
        PrimVarCV* src = &primVarCVs[0];
        for (int level = 1; level < nRefinedLevels; ++level) {
            PrimVarCV* dst = src +
                refiner->GetLevel(level - 1).GetNumFVarValues(channel);
            OpenSubdiv::Far::PrimvarRefiner(*refiner).InterpolateFaceVarying(
                level, src, dst, channel);
            src = dst;
        }
        patchTable->ComputeLocalPointValuesFaceVarying(&primVarCVs[0],
            &primVarCVs[nFVarRefinerVertices], channel);

        // see comment in variable initialized location for detail explanation
        // why we need to do this OpenSubdiv-3.1 workaround.
        if (requireUniformFix) {
            size_t fvarPatchPointOffset =
                refiner->GetLevel(0).GetNumFVarValues(channel);
            for (size_t i = fvarPatchPointOffset; i < primVarCVs.size(); ++i) {
                primVarCVs[i - fvarPatchPointOffset] = primVarCVs[i];
            }
        }
        faceVaryingPrimVarCvs.insert({channel, std::move(primVarCVs)});
    }
}

void
evalLimitSurface(const OpenSubdiv::Far::PatchTable* patchTable,
        const std::vector<LimitSurfaceSample>& limitSurfaceSamples,
        const std::vector<PatchCV>& patchCvs,
        const std::vector<TextureCV>& textureCvs,
        AttributeRate textureRate,
        SubdivisionMesh::VertexBuffer& tessellatedVertices,
        VertexBuffer<Vec3f, InterleavedTraits>& surfaceNormal,
        VertexBuffer<Vec2f, InterleavedTraits>& surfaceSt,
        VertexBuffer<Vec3f, InterleavedTraits>& surfaceDpds,
        VertexBuffer<Vec3f, InterleavedTraits>& surfaceDpdt,
        std::vector<DisplacementFootprint>& displacementFootprints,
        bool& hasBadDerivatives, bool requireUniformFix,
        uint motionSampleCount)
{
    // allocate tessellated vertex/index buffer to hold the evaluation result
    size_t tessellatedVertexCount = limitSurfaceSamples.size();
    tessellatedVertices = SubdivisionMesh::VertexBuffer(
        tessellatedVertexCount, motionSampleCount);
    surfaceNormal = VertexBuffer<Vec3f, InterleavedTraits>(
        tessellatedVertexCount, motionSampleCount);
    surfaceSt = VertexBuffer<Vec2f, InterleavedTraits>(
        tessellatedVertexCount, 1);
    surfaceDpds = VertexBuffer<Vec3f, InterleavedTraits>(
        tessellatedVertexCount, motionSampleCount);
    surfaceDpdt = VertexBuffer<Vec3f, InterleavedTraits>(
        tessellatedVertexCount, motionSampleCount);
    displacementFootprints.resize(tessellatedVertexCount);
    bool hasTextureSt = textureRate != RATE_UNKNOWN;
    hasBadDerivatives = false;
    OpenSubdiv::Far::PatchMap patchMap(*patchTable);
    tbb::blocked_range<size_t> range =
        tbb::blocked_range<size_t>(0, tessellatedVertexCount);
    tbb::parallel_for(range, [&](const tbb::blocked_range<size_t> &r) {
        for (size_t i = r.begin(); i < r.end(); ++i) {
            int fid = limitSurfaceSamples[i].mFaceId;
            if (fid == -1) {
                continue;
            }
            const Vec2f& uv = limitSurfaceSamples[i].mUv;
            const Vec2f& uvDelta = limitSurfaceSamples[i].mUvDelta;
            const OpenSubdiv::Far::PatchTable::PatchHandle* handle =
                patchMap.FindPatch(fid, uv[0], uv[1]);
            // 20 is the maximum patch control point number opensubdiv use
            // (Gregory patch in particular)
            float pWeights[20];
            float duWeights[20];
            float dvWeights[20];
            float duuWeights[20];
            float duvWeights[20];
            float dvvWeights[20];
            const auto& cvIndices = patchTable->GetPatchVertices(*handle);
            patchTable->EvaluateBasis(*handle, uv[0], uv[1], pWeights,
                duWeights, dvWeights, duuWeights, duvWeights, dvvWeights);
            Vec3fa pos[motionSampleCount];
            Vec3fa dpdu[motionSampleCount];
            Vec3fa dpdv[motionSampleCount];
            Vec3fa dpduu[motionSampleCount];
            Vec3fa dpduv[motionSampleCount];
            Vec3fa dpdvv[motionSampleCount];
            for (size_t t = 0; t < motionSampleCount; ++t) {
                pos[t] = Vec3fa(0.0f);
                dpdu[t] = Vec3fa(0.0f);
                dpdv[t] = Vec3fa(0.0f);
                dpduu[t] = Vec3fa(0.0f);
                dpduv[t] = Vec3fa(0.0f);
                dpdvv[t] = Vec3fa(0.0f);
            }
            for (int j = 0; j < cvIndices.size(); ++j) {
                for (size_t t = 0; t < motionSampleCount; ++t) {
                    const Vec3fa& cv = patchCvs[cvIndices[j]].mData[t];
                    pos[t] += pWeights[j] * cv;
                    dpdu[t] += duWeights[j] * cv;
                    dpdv[t] += dvWeights[j] * cv;
                    dpduu[t] += duuWeights[j] * cv;
                    dpduv[t] += duvWeights[j] * cv;
                    dpdvv[t] += dvvWeights[j] * cv;
                }
            }
            for (size_t t = 0; t < motionSampleCount; ++t) {
                tessellatedVertices(i, t) = pos[t];
                Vec3f N = scene_rdl2::math::cross(dpdu[t], dpdv[t]);
                float tolerance = 1e-5f *
                    (scene_rdl2::math::length(dpdu[t]) + scene_rdl2::math::length(dpdv[t]));
                if (scene_rdl2::math::length(N) <= tolerance) {
                    // degenerated normal caused by degenerated derivatives,
                    // which can happen when vertices on control face overlap
                    // or parallel neighbor edges on control face
                    // we need to approximate the normal for these edge cases
                    //
                    // The justification for this formula is that since
                    // N(0, 0) = 0, we are approximating N(0,0) in some
                    // neighborhood of N(0, 0) that we will call N(du, dv) --
                    // where (du, dv) is some small epsilon near (0, 0).
                    // The Taylor series for N(u,v) in this neighborhood --
                    // truncated to the second order term -- is:
                    // N(du, dv) ~=
                    // N +
                    // (dNdu * DU + dNdv * DV) +
                    // (dNdudu * DU^2 + 2*dNdudv * DU * DV + dNdvdv * DV^2) / 2
                    //
                    // So our decision to use N = dNdu + dNdv is a first order
                    // Taylor approximation to N(u,v) where N = 0.
                    // We just need to incorporate the sign reflected in the
                    // du and dv, as going "past the end" of the patch can flip
                    // the sign in some cases due to the behavior of the
                    // derivatives. Assuming the domain of our patches is
                    // [0,1] x [0,1]:
                    //
                    // du = (u == 1.0) ? -1.0 : 1.0;
                    // dv = (v == 1.0) ? -1.0 : 1.0;
                    // and we can use:
                    // N = normalize(dNdu * du + dNdv * dv);
                    //
                    Vec3fa dndu = scene_rdl2::math::cross(dpduu[t], dpdv[t]) +
                        scene_rdl2::math::cross(dpdu[t], dpduv[t]);
                    Vec3fa dndv = scene_rdl2::math::cross(dpduv[t], dpdv[t]) +
                        scene_rdl2::math::cross(dpdu[t], dpdvv[t]);
                    float uFlip = (uv[0] >= 1.0f) ? -1.0f : 1.0f;
                    float vFlip = (uv[1] >= 1.0f) ? -1.0f : 1.0f;
                    surfaceNormal(i, t) = normalize(
                        (dndu * uFlip + dndv * vFlip).asVec3f());
                    // TODO find a better way to fix the degenerated derivatives
                    patchTable->EvaluateBasis(*handle, 0.5f, 0.5f, pWeights,
                        duWeights, dvWeights);
                    dpdu[t] = Vec3fa(0.0f);
                    dpdv[t] = Vec3fa(0.0f);
                    for (int j = 0; j < cvIndices.size(); ++j) {
                        const Vec3fa& cv = patchCvs[cvIndices[j]].mData[t];
                        dpdu[t] += duWeights[j] * cv;
                        dpdv[t] += dvWeights[j] * cv;
                    }
                    // the above approximation still doesn't work, use the fixed
                    // derivatives to generate shading normal
                    if (!scene_rdl2::math::isFinite(surfaceNormal(i, t))) {
                        surfaceNormal(i, t) = normalize(
                            scene_rdl2::math::cross(dpdu[t], dpdv[t]));
                        // really don't know what to do at this point...
                        if (!scene_rdl2::math::isFinite(surfaceNormal(i, t))) {
                            dpdu[t] = Vec3fa(1, 0, 0, 0);
                            dpdv[t] = Vec3fa(0, 1, 0, 0);
                            surfaceNormal(i, t) = Vec3f(0, 0, 1);
                        }
                        hasBadDerivatives = true;
                    }
                } else {
                    surfaceNormal(i, t) = normalize(N);
                }
            }
            if (hasTextureSt) {
                // 20 is the maximum patch control point number opensubdiv uses
                // (Gregory patch in particular)
                float pWeights[20];
                float duWeights[20];
                float dvWeights[20];
                const auto& textureCvIndices =
                    textureRate == RATE_FACE_VARYING ?
                    patchTable->GetPatchFVarValues(*handle, sTextureStFVarChannel) :
                    patchTable->GetPatchVertices(*handle);

                if (!requireUniformFix) {
                    if (textureRate == RATE_FACE_VARYING) {
                        patchTable->EvaluateBasisFaceVarying(*handle,
                            uv[0], uv[1], pWeights, duWeights, dvWeights,
                            0, 0, 0, sTextureStFVarChannel);
                    } else {
                        patchTable->EvaluateBasis(*handle,
                            uv[0], uv[1], pWeights, duWeights, dvWeights);
                    }
                } else {
                    // see comment in variable initialized location for detail
                    // explanation why we need to do this OpenSubdiv-3.1
                    // workaround.
                    patchTable->EvaluateBasis(*handle, uv[0], uv[1],
                        pWeights, duWeights, dvWeights);
                }
                Vec2f st(0.0f);
                Vec2f dstdu(0.0f);
                Vec2f dstdv(0.0f);
                for (int j = 0; j < textureCvIndices.size(); ++j) {
                    const Vec2f& cv = textureCvs[textureCvIndices[j]].mSt;
                    st += pWeights[j] * cv;
                    dstdu += duWeights[j] * cv;
                    dstdv += dvWeights[j] * cv;
                }
                surfaceSt(i) = st;
                Vec3f dpds[motionSampleCount];
                Vec3f dpdt[motionSampleCount];
                for (size_t t = 0; t < motionSampleCount; ++t) {
                    computePartialsWithRespect2Texture(dpdu[t], dpdv[t],
                        dstdu, dstdv, dpds[t], dpdt[t]);
                    surfaceDpds(i, t) = dpds[t];
                    surfaceDpdt(i, t) = dpdt[t];
                }
                displacementFootprints[i].mDst[0] = 0.5f * dstdu * uvDelta[0];
                displacementFootprints[i].mDst[1] = 0.5f * dstdv * uvDelta[1];
            } else {
                surfaceSt(i) = uv;
                for (size_t t = 0; t < motionSampleCount; ++t) {
                    surfaceDpds(i, t) = dpdu[t];
                    surfaceDpdt(i, t) = dpdv[t];
                }
                displacementFootprints[i].mDst[0] =
                    Vec2f(0.5f * uvDelta[0], 0.0f);
                displacementFootprints[i].mDst[1] =
                    Vec2f(0.0f, 0.5f * uvDelta[1]);
            }
        }
    });
}

void
evalLimitAttributes(const OpenSubdiv::Far::PatchTable* patchTable,
        const std::vector<LimitSurfaceSample>& limitSurfaceSamples,
        const std::vector<PrimVarCV>& varyingPrimVarCvs,
        const std::vector<PrimVarCV>& vertexPrimVarCvs,
        const std::unordered_map<int, std::vector<LimitSurfaceSample>>&
        fvarLimitSamples,
        const std::unordered_map<int, std::vector<PrimVarCV>>&
        faceVaryingPrimVarCvs,
        std::unordered_map<int, SubdivisionMesh::IndexBuffer>&& fvarIndices,
        Attributes* primitiveAttributes,
        FaceVaryingAttributes& faceVaryingAttributes,
        bool requireUniformFix)
{
    OpenSubdiv::Far::PatchMap patchMap(*patchTable);
    size_t tessellatedVertexCount = limitSurfaceSamples.size();
    if (primitiveAttributes->hasVaryingAttributes()) {
         // allocate tessellated attributes buffer to hold the evaluation result
        primitiveAttributes->resizeVaryingAttributes(tessellatedVertexCount);
        float* attributesData = primitiveAttributes->getVaryingAttributesData();
        size_t floatPerCV =
            primitiveAttributes->getVaryingAttributesStride() / sizeof(float);
        tbb::blocked_range<size_t> range =
            tbb::blocked_range<size_t>(0, tessellatedVertexCount);
        tbb::parallel_for(range, [&](const tbb::blocked_range<size_t> &r) {
            for (size_t i = r.begin(); i < r.end(); ++i) {
                int fid = limitSurfaceSamples[i].mFaceId;
                if (fid == -1) {
                    continue;
                }
                const Vec2f& uv = limitSurfaceSamples[i].mUv;
                const OpenSubdiv::Far::PatchTable::PatchHandle* handle =
                    patchMap.FindPatch(fid, uv[0], uv[1]);
                float vWeights[20];
                const auto& cvIndices = patchTable->GetPatchVaryingVertices(
                    *handle);
                patchTable->EvaluateBasisVarying(
                    *handle, uv[0], uv[1], vWeights);
                float* result = attributesData + i * floatPerCV;
                memset(result, 0, sizeof(float) * floatPerCV);
                for (int j = 0; j < cvIndices.size(); ++j) {
                    const float* cv = varyingPrimVarCvs[cvIndices[j]].mData;
                    for (size_t f = 0; f < floatPerCV; ++f) {
                        result[f] += vWeights[j] * cv[f];
                    }
                }
            }
        });
    }

    if (primitiveAttributes->hasVertexAttributes()) {
        // allocate tessellated attributes buffer to hold the evaluation result
        primitiveAttributes->resizeVertexAttributes(tessellatedVertexCount);
        float* attributesData = primitiveAttributes->getVertexAttributesData();
        size_t floatPerCV =
            primitiveAttributes->getVertexAttributesStride() / sizeof(float);
        tbb::blocked_range<size_t> range =
            tbb::blocked_range<size_t>(0, tessellatedVertexCount);
        tbb::parallel_for(range, [&](const tbb::blocked_range<size_t> &r) {
            for (size_t i = r.begin(); i < r.end(); ++i) {
                int fid = limitSurfaceSamples[i].mFaceId;
                if (fid == -1) {
                    continue;
                }
                const Vec2f& uv = limitSurfaceSamples[i].mUv;
                const OpenSubdiv::Far::PatchTable::PatchHandle* handle =
                    patchMap.FindPatch(fid, uv[0], uv[1]);
                float vWeights[20];
                const auto& cvIndices = patchTable->GetPatchVertices(*handle);
                patchTable->EvaluateBasis(*handle, uv[0], uv[1], vWeights);
                float* result = attributesData + i * floatPerCV;
                memset(result, 0, sizeof(float) * floatPerCV);
                for (int j = 0; j < cvIndices.size(); ++j) {
                    const float* cv = vertexPrimVarCvs[cvIndices[j]].mData;
                    for (size_t f = 0; f < floatPerCV; ++f) {
                        result[f] += vWeights[j] * cv[f];
                    }
                }
            }
        });
    }

    for (auto& key: faceVaryingAttributes.getAllKeys()) {
        auto& attributeBuffer = faceVaryingAttributes.getAttributeBuffer(key);
        size_t floatPerCV = attributeBuffer.mFloatPerVertex;
        int channel = attributeBuffer.mChannel;
        const std::vector<LimitSurfaceSample>& limitSamples =
            fvarLimitSamples.find(channel)->second;
        const std::vector<PrimVarCV>& primVarCVs =
            faceVaryingPrimVarCvs.find(channel)->second;
        size_t fvarTessellatedVertexCount = limitSamples.size();
        std::vector<float> tessellatedData(
            fvarTessellatedVertexCount * floatPerCV);
        tbb::blocked_range<size_t> range =
            tbb::blocked_range<size_t>(0, fvarTessellatedVertexCount);
        tbb::parallel_for(range, [&](const tbb::blocked_range<size_t> &r) {
            for (size_t i = r.begin(); i < r.end(); ++i) {
                int fid = limitSamples[i].mFaceId;
                if (fid == -1) {
                    continue;
                }
                const Vec2f& uv = limitSamples[i].mUv;
                const OpenSubdiv::Far::PatchTable::PatchHandle* handle =
                    patchMap.FindPatch(fid, uv[0], uv[1]);
                float fvWeights[20];
                const auto& cvIndices = patchTable->GetPatchFVarValues(
                    *handle, channel);

                if (!requireUniformFix) {
                    patchTable->EvaluateBasisFaceVarying(*handle, uv[0], uv[1],
                        fvWeights, 0, 0, 0, 0, 0, channel);
                } else {
                    // see comment in variable initialized location for detail
                    // explanation why we need to do this OpenSubdiv-3.1
                    // workaround.
                    patchTable->EvaluateBasis(*handle, uv[0], uv[1], fvWeights);
                }
                float* result = &tessellatedData[i * floatPerCV];
                memset(result, 0, sizeof(float) * floatPerCV);
                for (int j = 0; j < cvIndices.size(); ++j) {
                    const float* cv = primVarCVs[cvIndices[j]].mData;
                    for (size_t f = 0; f < floatPerCV; ++f) {
                        result[f] += fvWeights[j] * cv[f];
                    }
                }
            }
        });
        // replace the primvar control points with final tesselalted result
        attributeBuffer.mData.swap(tessellatedData);
        attributeBuffer.mIndices.swap(fvarIndices.find(channel)->second);
    }
}

void
OpenSubdivMesh::tessellate(const TessellationParams& tessellationParams)
{
    if (mIsMeshFinalized) {
        return;
    }

    MNRY_ASSERT_REQUIRE(mControlMeshData);
    auto& primitiveAttributeTable = mControlMeshData->mPrimitiveAttributeTable;
    // process texture st if it is provided
    bool hasTextureSt = mControlMeshData->initTextureSt();
    if (hasTextureSt) {
        // we already explicitly store texture st in its own container,
        // remove it from primitiveAttributeTable to avoid duplicated
        // memory usage
        primitiveAttributeTable.erase(StandardAttributes::sSurfaceST);
    }
    int controlVertexCount = mControlMeshData->mVertices.size();
    // extract out face varying attributes. At this moment the Attributes API
    // can't handle different indexing for different face varying attributes,
    // and that's what happened in OpenSubdiv tessellation logic.
    // As the result, we need to handle face varying attributes tessellation
    // and interpolation separately from attributes with other rate.
    // if there is face varying texture st, it would occupy the first
    // face varying channel
    int firstFaceVaryingChannel =
        mControlMeshData->mTextureRate == RATE_FACE_VARYING ? 1 : 0;
    mFaceVaryingAttributes.reset(new FaceVaryingAttributes(controlVertexCount,
        mControlMeshData->mFaceVertexCount, mControlMeshData->mIndices,
        primitiveAttributeTable, firstFaceVaryingChannel));
    // Remove face varying attributes from primitiveAttributeTable
    // to avoid duplicated memory usage
    for (auto& key : mFaceVaryingAttributes->getAllKeys()) {
        primitiveAttributeTable.erase(key);
    }

    // interleave PrimitiveAttributeTable
    setAttributes(Attributes::interleave(primitiveAttributeTable,
        mPartCount,
        mControlMeshData->mFaceVertexCount.size(), controlVertexCount,
        mControlMeshData->mFaceVertexCount, controlVertexCount));

    getAttributes()->transformAttributes(mControlMeshData->mXforms,
                                         mControlMeshData->mShutterOpenDelta,
                                         mControlMeshData->mShutterCloseDelta,
                                         {{StandardAttributes::sNormal, Vec3Type::NORMAL},
                                         {StandardAttributes::sdPds, Vec3Type::VECTOR},
                                         {StandardAttributes::sdPdt, Vec3Type::VECTOR}});
 
    // reverse normals reverses orientation and negates normals
    if (mIsNormalReversed ^ mIsOrientationReversed) {
        reverseOrientation(mControlMeshData->mFaceVertexCount,
                         mControlMeshData->mIndices, mAttributes);
        if (hasTextureSt) {
            mControlMeshData->reverseTextureIndices();
        }
        // we extract all the face varying attributes to
        // mFaceVaryingAttributes now (instead of having them
        // live in mAttributes) so we need to manually reverse
        // their indices here
        mFaceVaryingAttributes->reverseControlIndices(
            mControlMeshData->mFaceVertexCount);
    }
    if (mIsNormalReversed) mAttributes->negateNormal();

    // get the rdl layer
    const scene_rdl2::rdl2::Layer* pRdlLayer = tessellationParams.mRdlLayer;

    // the case that we only render control cage
    bool noTessellation = mMeshResolution <= 1;
    // calculate edge tessellation factor based on either user specified
    // resolution (uniform) or camera frustum info (adaptive)
    std::vector<SubdTessellationFactor> tessellationFactors =
        computeSubdTessellationFactor(pRdlLayer, tessellationParams.mFrustums,
            tessellationParams.mEnableDisplacement, noTessellation);

    // analyze control faces and generate quadTopologies, which are used
    // for generating tessellated vertices and indices later

    // A bit confusing when texture st is using vertex rate instead of
    // face varying rate... the faceVaryingIndices is "potentially"
    // face varying that if you feed in a vertex rate it still works fine
    // With face varying rate it will locate edges that use different
    // vertex values in neighbor faces, with vertex rate we will not find
    // this kind of edges
    const SubdivisionMesh::IndexBuffer& faceVaryingIndices = hasTextureSt ?
        mControlMeshData->mTextureIndices : mControlMeshData->mIndices;
    size_t coarseVertexCount = hasTextureSt ?
        mControlMeshData->mTextureVertices.size() :
        mControlMeshData->mVertices.size();
    SubdTopologyIdLookup topologyIdLookup(coarseVertexCount,
        mControlMeshData->mFaceVertexCount, mControlMeshData->mIndices,
        faceVaryingIndices, noTessellation);
    std::vector<SubdQuadTopology> quadTopologies = generateSubdQuadTopology(
        pRdlLayer, topologyIdLookup, mControlMeshData->mFaceVertexCount,
        faceVaryingIndices, noTessellation);
    MNRY_ASSERT_REQUIRE(tessellationFactors.size() == quadTopologies.size());
    SubdTessellatedVertexLookup tessellatedVertexLookup(quadTopologies,
        topologyIdLookup, tessellationFactors, noTessellation);
    // generate the tessellated index buffer and
    // sample points for limit surface evaluation
    std::vector<LimitSurfaceSample> limitSurfaceSamples;
    mTessellatedToControlFace.clear();
    generateIndexBufferAndSurfaceSamples(quadTopologies,
        tessellatedVertexLookup, noTessellation, mTessellatedIndices,
        limitSurfaceSamples, &mTessellatedToControlFace);
    // for each face varying attribute, generate its own limitSurfaceSamples
    // and tessellated index buffer
    std::unordered_map<int, std::vector<LimitSurfaceSample>> fvarLimitSamples;
    std::unordered_map<int, SubdivisionMesh::IndexBuffer> fvarIndices;
    for (auto& key: mFaceVaryingAttributes->getAllKeys()) {
        auto& attributeBuffer = mFaceVaryingAttributes->getAttributeBuffer(key);
        SubdTopologyIdLookup fvarTopologyIdLookup(attributeBuffer.getVertexCount(),
            mControlMeshData->mFaceVertexCount, attributeBuffer.mIndices,
            noTessellation);
        std::vector<SubdQuadTopology> fvarQuadTopologies =
            generateSubdQuadTopology(pRdlLayer,
            fvarTopologyIdLookup, mControlMeshData->mFaceVertexCount,
            attributeBuffer.mIndices, noTessellation);
        MNRY_ASSERT_REQUIRE(
            tessellationFactors.size() == fvarQuadTopologies.size());
        SubdTessellatedVertexLookup fvarTessellatedVertexLookup(
            fvarQuadTopologies, fvarTopologyIdLookup,
            tessellationFactors, noTessellation);
        std::vector<LimitSurfaceSample> limitSamples;
        SubdivisionMesh::IndexBuffer tessellatedIndices;
        generateIndexBufferAndSurfaceSamples(fvarQuadTopologies,
            fvarTessellatedVertexLookup, noTessellation, tessellatedIndices,
            limitSamples);
        fvarLimitSamples.insert(
            {attributeBuffer.mChannel, std::move(limitSamples)});
        fvarIndices.insert(
            {attributeBuffer.mChannel, tessellatedIndices});
    }

    // generate a TopologyRefiner
    OpenSubdiv::Far::TopologyRefiner* refiner = createTopologyRefiner(
        *mControlMeshData, *mFaceVaryingAttributes);

    bool hasFaceVaryingAttributes =
        mControlMeshData->mTextureRate == RATE_FACE_VARYING ||
        mFaceVaryingAttributes->getAllKeys().size() > 0;

    // Unfortunate work around for OpenSubdiv-3.1...
    // There are two problems/bugs with the way a PatchTable is constructed
    // for Uniform refinement right now and we have to work around both (but
    // note that Uniform refinement can be avoided completely using 3.4).
    // 1) The first is related to this filed issue:
    // https://github.com/PixarAnimationStudios/OpenSubdiv/issues/737
    // about the expectations of what levels you have in your primvar buffers.
    // For historical reasons, for Uniform patches it is expected that vertex
    // for the cage and the last level are present.  But for face-varying data,
    // for some reason only the last level is expected --
    // not values for the cage.  For our case refining to level 1,
    // that means levels 0 and 1 are expected in the position buffers,
    // but only level 1 for the face-varying buffers.
    // 2) The second problem is unique to 3.1 when the face-varying patch
    // could differ from the vertex patch. There is a little information
    // about the patch (e.g. its depth) that needs to be stored for each patch,
    // and in 3.1 that needed to be made separate for the patches for
    // face-varying channels. It turns out this data is not
    // initialized correctly -- it should be assigned the same as
    // the vertex patch, but that was forgotten so it is default-initialized
    // on construction.
    // This leads to EvaluateBasisFaceVarying() not working properly since
    // (among other things) the depth of the patch is not stored correctly.
    // The mapping of a (u,v) coordinate of the base face to that of a sub face
    // is incorrect, so the weights we get back are all wrong.
    // All we need to do here is to call EvaluateBasis() explicitly instead of
    // EvaluateBasisFaceVarying() for face varying attributes and textureSt.
    //
    // So now what do we do to work around them...
    // 1) We initialize the face-varying buffers to NOT include the cage points.
    // We need to know the number of FVarValues in the cage (level 0), i.e.:
    // int fvarPatchPointOffset =
    //     refiner.GetLevel(0).GetNumFVarValues(fvarChannel);
    // 2) When the scheme is bilinear and RefineUniform,
    // We call EvaluateBasis()instead of EvaluateBasisFaceVarying()
    // for face varying primitive attributes and textureSt
    bool requireUniformFix = false;
    bool hasCreaseOrCorner = this->hasSubdCreases() || this->hasSubdCorners();
    if (mControlMeshData->mScheme != SubdivisionMesh::Scheme::CATMULL_CLARK) {
        refiner->RefineUniform(
           OpenSubdiv::Far::TopologyRefiner::UniformOptions(1));
        requireUniformFix = true;
    } else {
        // determine a suitably accurate depth for feature adaptive refinement
        // based on tessellation factors.  when creasing is present, use a deeper
        // level to capture crease accuracy but keep the "secondary level" low to
        // avoid over-representing irregular features beyond tessellation accuracy
        const int maxCreaseDepth = 6;

        int tessDepth = scene_rdl2::math::clamp(
            scene_rdl2::math::log2i(tessellatedVertexLookup.getMaxEdgeResolution()) + 1,
            1, 10);

        int creaseDepth = hasCreaseOrCorner ? maxCreaseDepth : tessDepth;

        int maxDepth = std::max(tessDepth, creaseDepth);
        int minDepth = std::min(tessDepth, creaseDepth);

        OpenSubdiv::Far::TopologyRefiner::AdaptiveOptions options(maxDepth);
        options.secondaryLevel = minDepth;
        options.useInfSharpPatch = hasCreaseOrCorner;
        options.considerFVarChannels = hasFaceVaryingAttributes;

        // adaptively refine the topology with an isolation level
        refiner->RefineAdaptive(options);
    }
    // generate PatchTable that we will use to evaluate the surface limit
    OpenSubdiv::Far::PatchTableFactory::Options patchOptions;
    patchOptions.SetEndCapType(
        OpenSubdiv::Far::PatchTableFactory::Options::ENDCAP_GREGORY_BASIS);
    patchOptions.generateFVarTables = hasFaceVaryingAttributes;
    patchOptions.generateFVarLegacyLinearPatches = false;
    patchOptions.useInfSharpPatch = hasCreaseOrCorner;
    const OpenSubdiv::Far::PatchTable* patchTable =
        OpenSubdiv::Far::PatchTableFactory::Create(*refiner, patchOptions);
    // evaluate limit surface samples to form the final tessellated mesh
    std::vector<DisplacementFootprint> displacementFootprints;
    size_t motionSampleCount = getMotionSamplesCount();
    std::vector<TextureCV> textureCvs;
    bool hasBadDerivatives = false;
    AttributeRate textureRate = mControlMeshData->mTextureRate;
    // generate patch cvs for sample point limit surface evaluation
    std::vector<PatchCV> patchCvs;
    generatePatchCvs(refiner, patchTable, mControlMeshData->mVertices,
        mControlMeshData->mTextureVertices, textureRate,
        patchCvs, textureCvs, requireUniformFix, motionSampleCount);
    evalLimitSurface(patchTable, limitSurfaceSamples, patchCvs,
        textureCvs, textureRate, mTessellatedVertices, mSurfaceNormal,
        mSurfaceSt, mSurfaceDpds, mSurfaceDpdt, displacementFootprints,
        hasBadDerivatives, requireUniformFix, motionSampleCount);
    if (hasBadDerivatives) {
        const scene_rdl2::rdl2::Geometry* pRdlGeometry = getRdlGeometry();
        MNRY_ASSERT(pRdlGeometry != nullptr);
        pRdlGeometry->debug("mesh ", getName(),
            " contains bad derivatives that may cause incorrect"
            " rendering result");
    }
    // tessellate primitive attributes
    Attributes* primitiveAttributes = getAttributes();
    // varyingData, vertexData hold the actual content PrimVarCV refer to
    std::vector<PrimVarCV> varyingPrimVarCvs;
    std::vector<float> varyingData;
    std::vector<PrimVarCV> vertexPrimVarCvs;
    std::vector<float> vertexData;
    std::unordered_map<int, std::vector<PrimVarCV>> faceVaryingPrimVarCvs;
    generatePrimVarCvs(refiner, patchTable,
        primitiveAttributes, *mFaceVaryingAttributes, controlVertexCount,
        varyingData, varyingPrimVarCvs,
        vertexData, vertexPrimVarCvs, faceVaryingPrimVarCvs,
        requireUniformFix);
    evalLimitAttributes(patchTable, limitSurfaceSamples,
        varyingPrimVarCvs, vertexPrimVarCvs,
        fvarLimitSamples, faceVaryingPrimVarCvs, std::move(fvarIndices),
        primitiveAttributes, *mFaceVaryingAttributes,
        requireUniformFix);

    // apply displacement
    if (tessellationParams.mEnableDisplacement && hasDisplacementAssignment(pRdlLayer)) {
        displaceMesh(pRdlLayer, limitSurfaceSamples, displacementFootprints,
            tessellatedVertexLookup, topologyIdLookup.getFaceVaryingSeams(),
            tessellationParams.mFrustums[0],
            tessellationParams.mWorld2Render);
    }

    // For the baked volume shader grid, we want to set the transform
    // to world2render if the primitive is shared.
    if (getIsReference()) {
        mPrimToRender = scene_rdl2::math::toFloat(tessellationParams.mWorld2Render);
    }

    // cleanup after tessellation
    // if we're baking, we need to keep mControlMeshData around
    if (!tessellationParams.mFastGeomUpdate && !tessellationParams.mIsBaking) {
        mControlMeshData.reset();
    }
    delete refiner;
    delete patchTable;

    mIsMeshFinalized = true;
}

void
OpenSubdivMesh::getTessellatedMesh(TessellatedMesh& tessMesh) const
{
    tessMesh.mIndexBufferType = MeshIndexType::QUAD;
    tessMesh.mFaceCount = getTessellatedMeshFaceCount();
    tessMesh.mIndexBufferDesc.mData =
        static_cast<const void*>(mTessellatedIndices.data());
    tessMesh.mIndexBufferDesc.mOffset = 0;
    tessMesh.mIndexBufferDesc.mStride =
        sQuadVertexCount * sizeof(geom::Primitive::IndexType);

    tessMesh.mVertexCount = getTessellatedMeshVertexCount();
    size_t motionSampleCount = getMotionSamplesCount();
    size_t vertexSize = sizeof(SubdivisionMesh::VertexBuffer::value_type);
    size_t vertexStride = motionSampleCount * vertexSize;
    const void* data = tessMesh.mVertexCount > 0  ?
        mTessellatedVertices.data() : nullptr;
    for (size_t t = 0; t < motionSampleCount; ++t) {
        size_t offset = t * vertexSize;
        tessMesh.mVertexBufferDesc.emplace_back(data, offset, vertexStride);
    }
}

template <typename T>
void*
OpenSubdivMesh::getBakedAttributeData(const TypedAttributeKey<T>& key,
                                      size_t vertexCount,
                                      size_t faceCount,
                                      size_t timeSamples,
                                      size_t& numElements) const
{
    Attributes *attributes = getAttributes();
    void* data = nullptr;

    switch (attributes->getRate(key)) {
    case RATE_CONSTANT:
    {
        numElements = timeSamples;
        T* tdata = new T[numElements];
        data = tdata;
        for (size_t t = 0; t < timeSamples; t++) {
            tdata[t] = attributes->getConstant(key, t);
        }
        break;
    }
    case RATE_PART:
    {
        numElements = faceCount * timeSamples;
        T* tdata = new T[numElements];
        data = tdata;
        for (size_t i = 0, dstIdx = 0; i < faceCount; i++) {
            for (size_t t = 0; t < timeSamples; t++) {
                int controlFaceIdx = mTessellatedToControlFace[i];
                int part = mFaceToPart[controlFaceIdx];
                tdata[dstIdx++] = attributes->getPart(key, part, t);
            }
        }
        break;
    }
    case RATE_UNIFORM:
    {
        numElements = faceCount * timeSamples;
        T* tdata = new T[numElements];
        data = tdata;
        for (size_t i = 0, dstIdx = 0; i < faceCount; i++) {
            for (size_t t = 0; t < timeSamples; t++) {
                int controlFaceIdx = mTessellatedToControlFace[i];
                tdata[dstIdx++] = attributes->getUniform(key, controlFaceIdx, t);
            }
        }
        break;
    }
    case RATE_VARYING:
    {
        numElements = vertexCount * timeSamples;
        T* tdata = new T[numElements];
        data = tdata;
        for (size_t i = 0, dstIdx = 0; i < vertexCount; i++) {
            for (size_t t = 0; t < timeSamples; t++) {
                tdata[dstIdx++] = attributes->getVarying(key, i, t);
            }
        }
        break;
    }
    case RATE_FACE_VARYING:
    {
        // We don't do this here as it's in mFaceVaryingAttributes and should not
        // be in the attribute table
        MNRY_ASSERT(false, "face varying attribute in attribute table");
        break;
    }
    case RATE_VERTEX:
    {
        numElements = vertexCount * timeSamples;
        T* tdata = new T[numElements];
        data = tdata;
        for (size_t i = 0, dstIdx = 0; i < vertexCount; i++) {
            for (size_t t = 0; t < timeSamples; t++) {
                tdata[dstIdx++] = attributes->getVertex(key, i, t);
            }
        }
        break;
    }
    default:
        MNRY_ASSERT(false, "unknown attribute rate");
        break;
    }

    return data;
}

std::unique_ptr<BakedAttribute>
OpenSubdivMesh::getBakedAttribute(const AttributeKey& key) const
{
    Attributes *attributes = getAttributes();
    size_t vertexCount = mTessellatedVertices.size();
    size_t faceCount = mTessellatedIndices.size() / 4;
    size_t timeSamples = attributes->getTimeSampleCount(key);

    std::unique_ptr<BakedAttribute> battr = fauxstd::make_unique<BakedAttribute>();

    battr->mName = key.getName();
    battr->mTimeSampleCount = timeSamples;
    battr->mType = key.getType();
    battr->mRate = attributes->getRate(key);
    battr->mData = nullptr;

    switch (battr->mType) {
    case AttributeType::TYPE_BOOL:
        battr->mData = getBakedAttributeData(TypedAttributeKey<bool>(key),
                                             vertexCount, faceCount, timeSamples,
                                             battr->mNumElements);
        break;
    case AttributeType::TYPE_INT:
        battr->mData = getBakedAttributeData(TypedAttributeKey<int>(key),
                                             vertexCount, faceCount, timeSamples,
                                             battr->mNumElements);
        break;
    case AttributeType::TYPE_LONG:
        battr->mData = getBakedAttributeData(TypedAttributeKey<long>(key),
                                             vertexCount, faceCount, timeSamples,
                                             battr->mNumElements);
        break;
    case AttributeType::TYPE_FLOAT:
        battr->mData = getBakedAttributeData(TypedAttributeKey<float>(key),
                                             vertexCount, faceCount, timeSamples,
                                             battr->mNumElements);
        break;
    case AttributeType::TYPE_DOUBLE:
        battr->mData = getBakedAttributeData(TypedAttributeKey<double>(key),
                                             vertexCount, faceCount, timeSamples,
                                             battr->mNumElements);
        break;
    case AttributeType::TYPE_STRING:
        battr->mData = getBakedAttributeData(TypedAttributeKey<std::string>(key),
                                             vertexCount, faceCount, timeSamples,
                                             battr->mNumElements);
        break;
    case AttributeType::TYPE_RGB:
        battr->mData = getBakedAttributeData(TypedAttributeKey<scene_rdl2::math::Color>(key),
                                             vertexCount, faceCount, timeSamples,
                                             battr->mNumElements);
        break;
    case AttributeType::TYPE_RGBA:
        battr->mData = getBakedAttributeData(TypedAttributeKey<scene_rdl2::math::Color4>(key),
                                             vertexCount, faceCount, timeSamples,
                                             battr->mNumElements);
        break;
    case AttributeType::TYPE_VEC2F:
        battr->mData = getBakedAttributeData(TypedAttributeKey<Vec2f>(key),
                                             vertexCount, faceCount, timeSamples,
                                             battr->mNumElements);
        break;
    case AttributeType::TYPE_VEC3F:
        battr->mData = getBakedAttributeData(TypedAttributeKey<Vec3f>(key),
                                             vertexCount, faceCount, timeSamples,
                                             battr->mNumElements);
        break;
    case AttributeType::TYPE_VEC4F:
        battr->mData = getBakedAttributeData(TypedAttributeKey<Vec4f>(key),
                                             vertexCount, faceCount, timeSamples,
                                             battr->mNumElements);
        break;
    case AttributeType::TYPE_MAT4F:
        battr->mData = getBakedAttributeData(TypedAttributeKey<scene_rdl2::math::Mat4f>(key),
                                             vertexCount, faceCount, timeSamples,
                                             battr->mNumElements);
        break;
    default:
        MNRY_ASSERT(false, (std::string("unsupported attribute type ") +
            std::string(attributeTypeName(key.getType())) +
            std::string(" for attribute ") + std::string(key.getName())).c_str());
        break;
    }

    return battr;
}

// Face-varying rate attributes are handled specially
template <typename T>
void*
OpenSubdivMesh::getFVBakedAttributeData(const TypedAttributeKey<T>& key,
                                        size_t& numElements) const
{
    numElements = mTessellatedIndices.size();
    size_t faceCount = numElements / 4;
    T* tdata = new T[numElements];

    const auto &attributeBuffer = mFaceVaryingAttributes->getAttributeBuffer(key);
    const T *srcdata = reinterpret_cast<const T*>(attributeBuffer.mData.data());

    for (size_t faceIdx = 0, dstIdx = 0; faceIdx < faceCount; faceIdx++) {
        for (size_t v = 0; v < 4; v++) {
            const int vtxIdx = attributeBuffer.mIndices[4 * faceIdx + v];
            tdata[dstIdx++] = srcdata[vtxIdx];
        }
    }

    return tdata;
}

std::unique_ptr<BakedAttribute>
OpenSubdivMesh::getFVBakedAttribute(const AttributeKey& key) const
{
    std::unique_ptr<BakedAttribute> battr = fauxstd::make_unique<BakedAttribute>();

    battr->mName = key.getName();
    battr->mTimeSampleCount = 1;  // only one time sample for face-varying data
    battr->mType = key.getType();
    battr->mRate = RATE_FACE_VARYING;
    battr->mData = nullptr;

    switch (battr->mType) {
    case AttributeType::TYPE_BOOL:
        battr->mData = getFVBakedAttributeData(TypedAttributeKey<bool>(key),
                                               battr->mNumElements);
        break;
    case AttributeType::TYPE_INT:
        battr->mData = getFVBakedAttributeData(TypedAttributeKey<int>(key),
                                               battr->mNumElements);
        break;
    case AttributeType::TYPE_LONG:
        battr->mData = getFVBakedAttributeData(TypedAttributeKey<long>(key),
                                               battr->mNumElements);
        break;
    case AttributeType::TYPE_FLOAT:
        battr->mData = getFVBakedAttributeData(TypedAttributeKey<float>(key),
                                               battr->mNumElements);
        break;
    case AttributeType::TYPE_DOUBLE:
        battr->mData = getFVBakedAttributeData(TypedAttributeKey<double>(key),
                                               battr->mNumElements);
        break;
    case AttributeType::TYPE_STRING:
        battr->mData = getFVBakedAttributeData(TypedAttributeKey<std::string>(key),
                                               battr->mNumElements);
        break;
    case AttributeType::TYPE_RGB:
        battr->mData = getFVBakedAttributeData(TypedAttributeKey<scene_rdl2::math::Color>(key),
                                               battr->mNumElements);
        break;
    case AttributeType::TYPE_RGBA:
        battr->mData = getFVBakedAttributeData(TypedAttributeKey<scene_rdl2::math::Color4>(key),
                                               battr->mNumElements);
        break;
    case AttributeType::TYPE_VEC2F:
        battr->mData = getFVBakedAttributeData(TypedAttributeKey<Vec2f>(key),
                                               battr->mNumElements);
        break;
    case AttributeType::TYPE_VEC3F:
        battr->mData = getFVBakedAttributeData(TypedAttributeKey<Vec3f>(key),
                                               battr->mNumElements);
        break;
    case AttributeType::TYPE_VEC4F:
        battr->mData = getFVBakedAttributeData(TypedAttributeKey<Vec4f>(key),
                                               battr->mNumElements);
        break;
    case AttributeType::TYPE_MAT4F:
        battr->mData = getFVBakedAttributeData(TypedAttributeKey<scene_rdl2::math::Mat4f>(key),
                                               battr->mNumElements);
        break;
    default:
        MNRY_ASSERT(false, (std::string("unsupported attribute type ") +
            std::string(attributeTypeName(key.getType())) +
            std::string(" for attribute ") + std::string(key.getName())).c_str());
        break;
    }

    return battr;
}

void
OpenSubdivMesh::getBakedMesh(BakedMesh& bakedMesh) const
{
    bakedMesh.mName = mName;

    bakedMesh.mVertsPerFace = 4;

    bakedMesh.mVertexCount = mTessellatedVertices.size();
    bakedMesh.mMotionSampleCount = getMotionSamplesCount();

    // Vertex buffer
    // After tessellation, there are vertices in the buffer that are
    // not connected to any face.   We don't want to include these in
    // the baked result so, as we iterate over the original mTessellatedVertices,
    // we check if the current index is referenced by the mTessellatedIndices
    // index buffer.   If it's not we skip it but we track the skipped indices 
    // for when we build the baked index buffer below.
    std::vector<size_t> skippedIndices;
    bakedMesh.mVertexBuffer.resize(bakedMesh.mVertexCount * bakedMesh.mMotionSampleCount);
    size_t actualVertexCount = 0;
    SubdivisionMesh::IndexBuffer sortedIndices = mTessellatedIndices;
    std::sort(sortedIndices.begin(), sortedIndices.end());
    for (size_t i = 0, dstIdx = 0; i < bakedMesh.mVertexCount; i++) {
        if (!std::binary_search(sortedIndices.begin(), sortedIndices.end(), i)) {
            // Skip vertices that are not referenced by the index buffer
            skippedIndices.push_back(i);
            continue;
        } else {
            actualVertexCount++;
            for (size_t t = 0; t < bakedMesh.mMotionSampleCount; t++) {
                bakedMesh.mVertexBuffer[dstIdx++] = mTessellatedVertices(i, t);
            }
            // these can be transformed from render space to object space with
            // scene_rdl2::math::Xform3f render2Obj = getRdlGeometry()->getRender2Object();
            // and then transformPoint()
        }
    }
    bakedMesh.mVertexCount = actualVertexCount;
    // Resize the vertex buffer to the actual vertex count
    bakedMesh.mVertexBuffer.resize(bakedMesh.mVertexCount * bakedMesh.mMotionSampleCount);

    // Index buffer
    std::vector<unsigned int> origIndexBuffer;
    bakedMesh.mIndexBuffer.resize(mTessellatedIndices.size());
    for (size_t i = 0; i < mTessellatedIndices.size(); i++) {
        if (!skippedIndices.empty()) {
            // Account for skipped vertices above by counting
            // the number of skipped indices that are less than
            // the current index and then subtracting this number
            // from the current index
            size_t numSkippedIndices = 0;
            for (size_t j = 0; j < skippedIndices.size(); j++) {
                if (skippedIndices[j] < mTessellatedIndices[i]) {
                    numSkippedIndices++;
                }
            }
            bakedMesh.mIndexBuffer[i] = mTessellatedIndices[i] - numSkippedIndices;
        } else {
            bakedMesh.mIndexBuffer[i] = mTessellatedIndices[i];
        }
    }

    {
        // grab the normal attr from mSurfaceNormal
        std::unique_ptr<BakedAttribute> normalAttr = fauxstd::make_unique<BakedAttribute>();
        normalAttr->mName = std::string("normal");
        normalAttr->mTimeSampleCount = getMotionSamplesCount();
        normalAttr->mType = AttributeType::TYPE_VEC3F;
        // convert from vertex rate to face varying as the standard for the lighting
        //  tools code that uses this API is face varying for normals
        normalAttr->mRate = RATE_FACE_VARYING;
        normalAttr->mNumElements = mTessellatedIndices.size() * normalAttr->mTimeSampleCount;
        Vec3f *normals = new Vec3f[normalAttr->mNumElements];
        for (size_t i = 0, dstIdx = 0; i < mTessellatedIndices.size(); i++) {
            for (size_t t = 0; t < normalAttr->mTimeSampleCount; t++) {
                normals[dstIdx++] = mSurfaceNormal(mTessellatedIndices[i], t);
                // to bake at vertex rate just use mSurfaceNormal directly
            }
        }
        normalAttr->mData = reinterpret_cast<char*>(normals);
        bakedMesh.mAttrs.push_back(std::move(normalAttr));
    }

    {
        // grab the surface_st attr from mSurfaceSt
        std::unique_ptr<BakedAttribute> stAttr = fauxstd::make_unique<BakedAttribute>();
        stAttr->mName = std::string("surface_st");
        stAttr->mTimeSampleCount = 1;
        stAttr->mType = AttributeType::TYPE_VEC2F;
        // convert from vertex rate to face varying, as that's the standard
        stAttr->mRate = RATE_FACE_VARYING;
        stAttr->mNumElements = mTessellatedIndices.size();
        Vec2f *sts = new Vec2f[stAttr->mNumElements];
        for (size_t i = 0; i < mTessellatedIndices.size(); i++) {
            sts[i] = mSurfaceSt(mTessellatedIndices[i]);
            // to bake at vertex rate just use mSurfaceSt directly
        }
        stAttr->mData = reinterpret_cast<char*>(sts);
        bakedMesh.mAttrs.push_back(std::move(stAttr));
    }

    bakedMesh.mTessellatedToBaseFace = mTessellatedToControlFace;

    bakedMesh.mFaceToPart.resize(mFaceToPart.size());
    for (size_t i = 0; i < mFaceToPart.size(); i++) {
        bakedMesh.mFaceToPart[i] = static_cast<int>(mFaceToPart[i]);
    }

    Attributes *attributes = getAttributes();
    const PrimitiveAttributeTable& patable = getPrimitiveAttributeTable();
    for (const auto& entry : patable) {
        const AttributeKey& key = entry.first;
        if (attributes->hasAttribute(key)) {
            bakedMesh.mAttrs.push_back(getBakedAttribute(key));
        } else if (mFaceVaryingAttributes->hasAttribute(key)) {
            bakedMesh.mAttrs.push_back(getFVBakedAttribute(key));
        }
    }
}

bool
OpenSubdivMesh::bakePosMap(int width, int height, int udim,
        TypedAttributeKey<Vec2f> stKey,
        Vec3fa *posResult, Vec3f *nrmResult) const
{
    const int faceVertexCount = 4; // tessellated mesh is always quads
    for (size_t faceId = 0; faceId < getTessellatedMeshFaceCount(); ++faceId) {
        // vertex ids of quad
        const uint vid0 = mTessellatedIndices[faceVertexCount * faceId];
        const uint vid1 = mTessellatedIndices[faceVertexCount * faceId + 1];
        const uint vid2 = mTessellatedIndices[faceVertexCount * faceId + 2];
        const uint vid3 = mTessellatedIndices[faceVertexCount * faceId + 3];

        // texture coordinates for this quad
        Vec2f st0, st1, st2, st3;
        if (stKey == StandardAttributes::sSurfaceST) {
            st0 = mSurfaceSt(vid0);
            st1 = mSurfaceSt(vid1);
            st2 = mSurfaceSt(vid2);
            st3 = mSurfaceSt(vid3);
        } else {
            if (mFaceVaryingAttributes->hasAttribute(stKey)) {
                const auto &attributeBuffer =
                    mFaceVaryingAttributes->getAttributeBuffer(stKey);
                const int stVid0 =
                    attributeBuffer.mIndices[faceVertexCount * faceId];
                const int stVid1 =
                    attributeBuffer.mIndices[faceVertexCount * faceId + 1];
                const int stVid2 =
                    attributeBuffer.mIndices[faceVertexCount * faceId + 2];
                const int stVid3 =
                    attributeBuffer.mIndices[faceVertexCount * faceId + 3];
                const Vec2f *data = reinterpret_cast<const Vec2f *>(
                    attributeBuffer.mData.data());
                st0 = data[stVid0];
                st1 = data[stVid1];
                st2 = data[stVid2];
                st3 = data[stVid3];
            } else {
                // hmmm.... not face varying, look it up in the Attributes
                Attributes *attributes = getAttributes();
                if (attributes->isSupported(stKey)) {
                    AttributeRate rate = attributes->getRate(stKey);
                    switch (rate) {
                    case RATE_CONSTANT:
                        st0 = st1 = st2 = st3 = attributes->getConstant(stKey);
                        break;
                    case RATE_UNIFORM:
                        st0 = st1 = st2 = st3 = attributes->getUniform(
                            stKey, faceId);
                        break;
                    case RATE_VARYING:
                        st0 = attributes->getVarying(stKey, vid0);
                        st1 = attributes->getVarying(stKey, vid1);
                        st2 = attributes->getVarying(stKey, vid2);
                        st3 = attributes->getVarying(stKey, vid3);
                        break;
                    case RATE_FACE_VARYING:
                        MNRY_ASSERT_REQUIRE(false,
                            "face varying attribute in attribute table");
                        break;
                    case RATE_VERTEX:
                        st0 = attributes->getVertex(stKey, vid0);
                        st1 = attributes->getVertex(stKey, vid1);
                        st2 = attributes->getVertex(stKey, vid2);
                        st3 = attributes->getVertex(stKey, vid3);
                        break;
                    default:
                        MNRY_ASSERT_REQUIRE(false,
                            "unknown attribute rate");
                        break;
                    }
                } else {
                    MNRY_ASSERT_REQUIRE(false,
                        "can't find uv attribute");
                }
            }
        }

        // check if this face exists in the udim and if so
        // transform th st coordinates into the normalized range
        if (udimxform(udim, st0) && udimxform(udim, st1) &&
            udimxform(udim, st2) && udimxform(udim, st3)) {

            // positions
            const Vec3f pos0 = mTessellatedVertices(vid0);
            const Vec3f pos1 = mTessellatedVertices(vid1);
            const Vec3f pos2 = mTessellatedVertices(vid2);
            const Vec3f pos3 = mTessellatedVertices(vid3);

            // normals
            const Vec3f *nrm0 = nullptr;
            const Vec3f *nrm1 = nullptr;
            const Vec3f *nrm2 = nullptr;
            const Vec3f *nrm3 = nullptr;
            Vec3f nrmData0, nrmData1, nrmData2, nrmData3;
            if (nrmResult) {
                nrmData0 = mSurfaceNormal(vid0);
                nrmData1 = mSurfaceNormal(vid1);
                nrmData2 = mSurfaceNormal(vid2);
                nrmData3 = mSurfaceNormal(vid3);
                nrm0 = &nrmData0;
                nrm1 = &nrmData1;
                nrm2 = &nrmData2;
                nrm3 = &nrmData3;
            }

            // first triangle: <0, 1, 3>
            // might be degenerate if face was originally a triangle
            if (vid0 != vid3) {
                scene_rdl2::math::BBox2f roiST(
                    Vec2f(std::min(st0.x, std::min(st1.x, st3.x)),
                          std::min(st0.y, std::min(st1.y, st3.y))),
                    Vec2f(std::max(st0.x, std::max(st1.x, st3.x)),
                          std::max(st0.y, std::max(st1.y, st3.y))));
                rasterizeTrianglePos(roiST, width, height, st0, st1, st3,
                    pos0, pos1, pos3, nrm0, nrm1, nrm3,
                    posResult, nrmResult);
            }
            // second triangle: <2, 3, 1>
            scene_rdl2::math::BBox2f roiST(
                Vec2f(std::min(st2.x, std::min(st3.x, st1.x)),
                      std::min(st2.y, std::min(st3.y, st1.y))),
                Vec2f(std::max(st2.x, std::max(st3.x, st1.x)),
                      std::max(st2.y, std::max(st3.y, st1.y))));
            rasterizeTrianglePos(roiST, width, height, st2, st3, st1,
                pos2, pos3, pos1, nrm2, nrm3, nrm1,
                posResult, nrmResult);
        }
    }

    return true;
}

SubdivisionMesh::VertexBuffer&
OpenSubdivMesh::getControlVertexBuffer()
{
    MNRY_ASSERT_REQUIRE(mControlMeshData,
        "control mesh has been deleted after tessellation");
    return mControlMeshData->mVertices;
}

bool
OpenSubdivMesh::hasAttribute(AttributeKey key) const
{
    // sSurfaceST got explicitly stored in mSurfaceSt
    // face varying attributes got explicitly stored in mFaceVaryingAttributes
    return (key == StandardAttributes::sSurfaceST) ||
        getAttributes()->hasAttribute(key) ||
        mFaceVaryingAttributes->hasAttribute(key);
}

void
OpenSubdivMesh::setRequiredAttributes(int primId, float time, float u, float v,
    float w, bool isFirst, Intersection& intersection) const
{
    // In Embree, quad is internally handled as a pair of
    // two triangles (v0, v1, v3) and (v2, v3, v1).
    // The (u', v') coordinates of the second triangle
    // is corrected by u = 1 - u' and v = 1 - v' to produce
    // a quad parameterization where u and v go from 0 to 1.
    // That's to say, if u + v > 1,
    // intersection happens in the second triangle,
    // and (u', v') in the second triangle should be (1 - u, 1 - v)
    size_t id1 = mTessellatedIndices[sQuadVertexCount * primId    ];
    size_t id2 = mTessellatedIndices[sQuadVertexCount * primId + 1];
    size_t id3 = mTessellatedIndices[sQuadVertexCount * primId + 2];
    size_t id4 = mTessellatedIndices[sQuadVertexCount * primId + 3];
    // weights for interpolator
    float wq[sQuadVertexCount];
    if (isFirst) {
        // First triangle in quad (0, 1, 3)
        wq[0] = w; wq[1] = u; wq[2] = 0; wq[3] = v;
        intersection.setIds(id1, id2, id4);
    } else {
        // Second triangle in quad (2, 3, 1)
        wq[0] = 0; wq[1] = v; wq[2] = w; wq[3] = u;
        intersection.setIds(id3, id4, id2);
    }

    int controlFaceId = mTessellatedToControlFace[primId];

    // If the control mesh data doesn't have face->part mapping,
    // just assume part = 0
    const int partId = (mFaceToPart.size() > 0) ? mFaceToPart[controlFaceId] : 0;

    // primitive attributes interpolation
    const Attributes* primitiveAttributes = getAttributes();
    MeshInterpolator interpolator(primitiveAttributes, time, partId,
        controlFaceId, id1, id2, id3, id4, wq[0], wq[1], wq[2], wq[3],
        primId, id1, id2, id3, id4, wq[0], wq[1], wq[2], wq[3]);
    intersection.setRequiredAttributes(interpolator);
    // explicitly handling face varying attributes here for now
    // constant/uniform/varying/vertex are handled by setRequiredAttributes
    mFaceVaryingAttributes->fillAttributes(intersection, primId,
        wq[0], wq[1], wq[2], wq[3]);

    // Add an interpolated N, dPds, and dPdt to the intersection
    // if they exist in the primitive attribute table
    setExplicitAttributes<MeshInterpolator>(interpolator,
                                            *primitiveAttributes,
                                            intersection);
}

void
OpenSubdivMesh::postIntersect(mcrt_common::ThreadLocalState& tls,
        const scene_rdl2::rdl2::Layer* pRdlLayer, const mcrt_common::Ray& ray,
        Intersection& intersection) const
{
    int primId = ray.primID;
    int controlFaceId = mTessellatedToControlFace[primId];

    // barycentric coordinate
    float u = ray.u;
    float v = ray.v;
    float w = 1.0f - u - v;
    bool isFirst = w > 0.f;
    if (!isFirst) {
        u = 1.0f - u;
        v = 1.0f - v;
        w = -w;
    }

    const int assignmentId = getControlFaceAssignmentId(controlFaceId);
    intersection.setLayerAssignments(assignmentId, pRdlLayer);

    const AttributeTable *table =
        intersection.getMaterial()->get<shading::RootShader>().getAttributeTable();
    intersection.setTable(&tls.mArena, table);
    const Attributes* primitiveAttributes = getAttributes();

    setRequiredAttributes(primId,
                          ray.time,
                          u, v, w,
                          isFirst,
                          intersection);

    uint32_t isecId1, isecId2, isecId3;
    intersection.getIds(isecId1, isecId2, isecId3);

    overrideInstanceAttrs(ray, intersection);

    // The St value is read from the explicit "surface_st" primitive
    // attribute on the control mesh if it exists.  The mSurfaceSt
    // member stores the tesselated value.
    Vec2f St = w * mSurfaceSt(isecId1) +
               u * mSurfaceSt(isecId2) +
               v * mSurfaceSt(isecId3);

    const Vec3f Ng = normalize(ray.getNg());

    Vec3f N, dPds, dPdt;
    const bool hasExplicitAttributes = getExplicitAttributes(*primitiveAttributes,
                                                             intersection,
                                                             N, dPds, dPdt);

    if (!hasExplicitAttributes) {
        if (isMotionBlurOn()) {
            const float i0PlusT = ray.time * static_cast<float>(getMotionSamplesCount() - 1);
            const int i0 = static_cast<int>(std::floor(i0PlusT));
            const int i1 = i0 + 1;
            const float t = i0PlusT - static_cast<float>(i0);

            Vec3f N0 = lerp(mSurfaceNormal(isecId1, i0), mSurfaceNormal(isecId1, i1), t);
            Vec3f N1 = lerp(mSurfaceNormal(isecId2, i0), mSurfaceNormal(isecId2, i1), t);
            Vec3f N2 = lerp(mSurfaceNormal(isecId3, i0), mSurfaceNormal(isecId3, i1), t);
            N = normalize(w * N0 + u * N1 + v * N2);

            Vec3f dPds0 = lerp(mSurfaceDpds(isecId1, i0), mSurfaceDpds(isecId1, i1), t);
            Vec3f dPds1 = lerp(mSurfaceDpds(isecId2, i0), mSurfaceDpds(isecId2, i1), t);
            Vec3f dPds2 = lerp(mSurfaceDpds(isecId3, i0), mSurfaceDpds(isecId3, i1), t);
            dPds = w * dPds0 + u * dPds1 + v * dPds2;

            Vec3f dPdt0 = lerp(mSurfaceDpdt(isecId1, i0), mSurfaceDpdt(isecId1, i1), t);
            Vec3f dPdt1 = lerp(mSurfaceDpdt(isecId2, i0), mSurfaceDpdt(isecId2, i1), t);
            Vec3f dPdt2 = lerp(mSurfaceDpdt(isecId3, i0), mSurfaceDpdt(isecId3, i1), t);
            dPdt = w * dPdt0 + u * dPdt1 + v * dPdt2;
        } else {
            N = normalize(w * mSurfaceNormal(isecId1) +
                          u * mSurfaceNormal(isecId2) +
                          v * mSurfaceNormal(isecId3));

            dPds = w * mSurfaceDpds(isecId1) +
                u * mSurfaceDpds(isecId2) +
                v * mSurfaceDpds(isecId3);
            dPdt = w * mSurfaceDpdt(isecId1) +
                u * mSurfaceDpdt(isecId2) +
                v * mSurfaceDpdt(isecId3);
        }
    }

    intersection.setDifferentialGeometry(Ng,
                                         N,
                                         St,
                                         dPds,
                                         dPdt,
                                         true); // has derivatives

    const scene_rdl2::rdl2::Geometry* geometry = intersection.getGeometryObject();
    MNRY_ASSERT(geometry != nullptr);
    intersection.setEpsilonHint( geometry->getRayEpsilon() );

    // calculate dfds/dfdt for primitive attributes that request differential
    if (table->requestDerivatives()) {
        computeAttributesDerivatives(table, isecId1, isecId2, isecId3, ray.time,
            intersection);
    }

    // polygon vertices for drawing wireframe
    if (table->requests(StandardAttributes::sNumPolyVertices)) {
        intersection.setAttribute(StandardAttributes::sNumPolyVertices, 4);
    }
    if (table->requests(StandardAttributes::sPolyVertexType)) {
        intersection.setAttribute(StandardAttributes::sPolyVertexType,
            static_cast<int>(StandardAttributes::POLYVERTEX_TYPE_POLYGON));
    }
    if (table->requests(StandardAttributes::sReversedNormals)) {
        intersection.setAttribute(StandardAttributes::sReversedNormals, mIsNormalReversed);
    }
    for (int iVert = 0; iVert < 4; iVert++) {
        if (table->requests(StandardAttributes::sPolyVertices[iVert])) {
            size_t id = mTessellatedIndices[sQuadVertexCount * primId + iVert];
            // may need to move the vertices to render space
            // for instancing object since they are ray traced in local space
            const Vec3f v = ray.isInstanceHit() ? transformPoint(ray.ext.l2r, mTessellatedVertices(id).asVec3f())
                                                : mTessellatedVertices(id).asVec3f();
            intersection.setAttribute(StandardAttributes::sPolyVertices[iVert], v);
        }
    }

    // we already explicitly store texture st in its own container
    if (table->requests(StandardAttributes::sSurfaceST)) {
        intersection.setAttribute(StandardAttributes::sSurfaceST, St);
    }
    // motion vectors
    if (table->requests(StandardAttributes::sMotion)) {
        const Vec3f motion = computeMotion(mTessellatedVertices, isecId1, isecId2, isecId3, w, u, v, ray);
        intersection.setAttribute(StandardAttributes::sMotion, motion);
    }
}

bool
OpenSubdivMesh::computeIntersectCurvature(const mcrt_common::Ray& ray,
        const Intersection& intersection, Vec3f& dnds, Vec3f& dndt) const
{
    uint32_t vid1, vid2, vid3;
    intersection.getIds(vid1, vid2, vid3);
    const Vec2f& st1 = mSurfaceSt(vid1);
    const Vec2f& st2 = mSurfaceSt(vid2);
    const Vec2f& st3 = mSurfaceSt(vid3);
    if (scene_rdl2::math::isEqual(st1, st2) && scene_rdl2::math::isEqual(st1, st3)) {
        return false;
    }
    Vec3f dn[2];
    if (isMotionBlurOn()) {
        float t = ray.time;
        Vec3f n1 = (1.0f - t) * mSurfaceNormal(vid1, 0) +
            t  * mSurfaceNormal(vid1, 1);
        Vec3f n2 = (1.0f - t) * mSurfaceNormal(vid2, 0) +
            t  * mSurfaceNormal(vid2, 1);
        Vec3f n3 = (1.0f - t) * mSurfaceNormal(vid3, 0) +
            t  * mSurfaceNormal(vid3, 1);
        if (!computeTrianglePartialDerivatives(n1, n2, n3, st1, st2, st3, dn)) {
            return false;
        }
        dnds = dn[0];
        dndt = dn[1];
    } else {
        const Vec3f& n1 = mSurfaceNormal(vid1, 0);
        const Vec3f& n2 = mSurfaceNormal(vid2, 0);
        const Vec3f& n3 = mSurfaceNormal(vid3, 0);
        if (!computeTrianglePartialDerivatives(n1, n2, n3, st1, st2, st3, dn)) {
            return false;
        }
        dnds = dn[0];
        dndt = dn[1];
    }
    return true;
}

SubdMesh*
OpenSubdivMesh::copy() const
{
    MNRY_ASSERT_REQUIRE(mControlMeshData);
    PrimitiveAttributeTable primitiveAttributeTable;
    mControlMeshData->mPrimitiveAttributeTable.copy(primitiveAttributeTable);
    OpenSubdivMesh *mesh = new OpenSubdivMesh(mControlMeshData->mScheme,
        SubdivisionMesh::FaceVertexCount(mControlMeshData->mFaceVertexCount),
        SubdivisionMesh::IndexBuffer(mControlMeshData->mIndices),
        mControlMeshData->mVertices.copy(),
        LayerAssignmentId(mLayerAssignmentId),
        std::move(primitiveAttributeTable));
    mesh->setParts(mPartCount, SubdivisionMesh::FaceToPartBuffer(mFaceToPart));
    return mesh;
}

void
OpenSubdivMesh::updateVertexData(const std::vector<float>& vertexData,
        const std::vector<Mat43>& prim2render)
{
    // TODO the original design of updateVertexData has been dated quite a lot
    // that we should spend some time redesign it. It doesn't support
    // motion blur now and using float buffer to carry Vector data seems lack
    // of type safety checks.
    MNRY_ASSERT(mControlMeshData);
    if (mControlMeshData->mIndices.size() == 0) {
        return;
    }
    // update control vertex position.
    size_t controlVertexCount = mControlMeshData->mVertices.size();
    if (vertexData.size() != 3 * controlVertexCount) {
        size_t updateVertexCount = vertexData.size() / 3;
        scene_rdl2::logging::Logger::error("SubdivisionMesh ", getName(), " contains ",
           controlVertexCount, " control vertices"
           " while update data contains ", updateVertexCount,
           " control vertices. Mesh topology should remain unchanged"
           " during geometry update.");
        controlVertexCount = std::min(controlVertexCount, updateVertexCount);
    }
    for (size_t i = 0; i < controlVertexCount; ++i) {
        Vec3f p(vertexData[3 * i    ],
                vertexData[3 * i + 1],
                vertexData[3 * i + 2]);
        mControlMeshData->mVertices(i) = Vec3fa(transformPoint(prim2render[0], p), 0.f);
    }
}

BBox3f
OpenSubdivMesh::computeAABB() const
{
    const SubdivisionMesh::VertexBuffer* pVertices = nullptr;
    if (mControlMeshData) {
        pVertices = &(mControlMeshData->mVertices);
    } else {
        pVertices = &mTessellatedVertices;
    }
    if (!pVertices || pVertices->empty()) {
        return BBox3f(scene_rdl2::math::zero);
    }
    const auto& vertices = *pVertices;
    BBox3f result(vertices(0).asVec3f());
    size_t motionSamplesCount = getMotionSamplesCount();
    for (size_t v = 0; v < vertices.size(); ++v) {
        for (size_t t = 0; t < motionSamplesCount; ++t) {
            result.extend(vertices(v, t));
        }
    }
    return result;
}

BBox3f
OpenSubdivMesh::computeAABBAtTimeStep(int timeStep) const
{
    const SubdivisionMesh::VertexBuffer* pVertices = nullptr;
    if (mControlMeshData) {
        pVertices = &(mControlMeshData->mVertices);
    } else {
        pVertices = &mTessellatedVertices;
    }
    if (!pVertices || pVertices->empty()) {
        return BBox3f(scene_rdl2::math::zero);
    }
    const auto& vertices = *pVertices;
    MNRY_ASSERT(timeStep >= 0 && timeStep < static_cast<int>(getMotionSamplesCount()), "timeStep out of range");
    BBox3f result(vertices(0, timeStep).asVec3f());
    for (size_t v = 1; v < vertices.size(); ++v) {
        result.extend(vertices(v, timeStep));
    }
    return result;
}

void
OpenSubdivMesh::getST(int tessFaceId, float u, float v, Vec2f& st) const
{
    size_t id1, id2, id3;
    const auto indices = &mTessellatedIndices[sQuadVertexCount * tessFaceId];

    float w = 1.0f - u - v;
    if (w > 0.0f) {
        // First triangle in quad (0, 1, 3)
        id1 = indices[0];
        id2 = indices[1];
        id3 = indices[3];
    } else {
        // Second triangle in quad (2, 3, 1)
        u = 1.0f - u;
        v = 1.0f - v;
        w = -w;

        id1 = indices[2];
        id2 = indices[3];
        id3 = indices[1];
    }

    st = w * mSurfaceSt(id1) +
         u * mSurfaceSt(id2) +
         v * mSurfaceSt(id3);
}

int
OpenSubdivMesh::getIntersectionAssignmentId(int primID) const
{
    return getControlFaceAssignmentId(mTessellatedToControlFace[primID]);
}

void
OpenSubdivMesh::computeAttributesDerivatives(const AttributeTable* table,
        size_t vid1, size_t vid2, size_t vid3, float time,
        Intersection& intersection) const
{
    std::array<float , 4> invA = {1.0f, 0.0f, 0.0f, 1.0f};
    computeStInverse(mSurfaceSt(vid1), mSurfaceSt(vid2), mSurfaceSt(vid3),
        invA);
    Attributes* attrs = getAttributes();
    for (auto key: table->getDifferentialAttributes()) {
        if (!attrs->isSupported(key)) {
            continue;
        }
        if (attrs->getRate(key) == RATE_VARYING) {
            switch (key.getType()) {
            case scene_rdl2::rdl2::TYPE_FLOAT:
                computeVaryingAttributeDerivatives(
                    TypedAttributeKey<float>(key),
                    vid1, vid2, vid3, time, invA, intersection);
                break;
            case scene_rdl2::rdl2::TYPE_RGB:
                computeVaryingAttributeDerivatives(
                    TypedAttributeKey<scene_rdl2::math::Color>(key),
                    vid1, vid2, vid3, time, invA, intersection);
                break;
            case scene_rdl2::rdl2::TYPE_RGBA:
                computeVaryingAttributeDerivatives(
                    TypedAttributeKey<scene_rdl2::math::Color4>(key),
                    vid1, vid2, vid3, time, invA, intersection);
                break;
            case scene_rdl2::rdl2::TYPE_VEC2F:
                computeVaryingAttributeDerivatives(
                    TypedAttributeKey<scene_rdl2::math::Vec2f>(key),
                    vid1, vid2, vid3, time, invA, intersection);
                break;
            case scene_rdl2::rdl2::TYPE_VEC3F:
                computeVaryingAttributeDerivatives(
                    TypedAttributeKey<scene_rdl2::math::Vec3f>(key),
                    vid1, vid2, vid3, time, invA, intersection);
                break;
            case scene_rdl2::rdl2::TYPE_MAT4F:
                computeVaryingAttributeDerivatives(
                    TypedAttributeKey<scene_rdl2::math::Mat4f>(key),
                    vid1, vid2, vid3, time, invA, intersection);
                break;
            default:
                break;
            }

        } else if (attrs->getRate(key) == RATE_VERTEX) {
            switch (key.getType()) {
            case scene_rdl2::rdl2::TYPE_FLOAT:
                computeVertexAttributeDerivatives(
                    TypedAttributeKey<float>(key),
                    vid1, vid2, vid3, time, invA, intersection);
                break;
            case scene_rdl2::rdl2::TYPE_RGB:
                computeVertexAttributeDerivatives(
                    TypedAttributeKey<scene_rdl2::math::Color>(key),
                    vid1, vid2, vid3, time, invA, intersection);
                break;
            case scene_rdl2::rdl2::TYPE_RGBA:
                computeVertexAttributeDerivatives(
                    TypedAttributeKey<scene_rdl2::math::Color4>(key),
                    vid1, vid2, vid3, time, invA, intersection);
                break;
            case scene_rdl2::rdl2::TYPE_VEC2F:
                computeVertexAttributeDerivatives(
                    TypedAttributeKey<scene_rdl2::math::Vec2f>(key),
                    vid1, vid2, vid3, time, invA, intersection);
                break;
            case scene_rdl2::rdl2::TYPE_VEC3F:
                computeVertexAttributeDerivatives(
                    TypedAttributeKey<scene_rdl2::math::Vec3f>(key),
                    vid1, vid2, vid3, time, invA, intersection);
                break;
            case scene_rdl2::rdl2::TYPE_MAT4F:
                computeVertexAttributeDerivatives(
                    TypedAttributeKey<scene_rdl2::math::Mat4f>(key),
                    vid1, vid2, vid3, time, invA, intersection);
                break;
            default:
                break;
            }
        }
    }
}

static inline bool 
outsideFrustum(uint8_t* outcodes, int vid0, int vid1, int vid2, int vid3)
{
    return ((outcodes[vid0] & outcodes[vid1] & outcodes[vid2] & outcodes[vid3]) != 0);
}

static inline bool 
insideFrustum(uint8_t* outcodes, int vid0, int vid1, int vid2, int vid3, uint8_t& outcode)
{
    outcode = (outcodes[vid0] | outcodes[vid1] | outcodes[vid2] | outcodes[vid3]);
    return (outcode == 0);
}

//----------------------------------------------------------------------------

void
OpenSubdivMesh::displaceMesh(const scene_rdl2::rdl2::Layer *pRdlLayer,
        const std::vector<LimitSurfaceSample>& limitSurfaceSamples,
        const std::vector<DisplacementFootprint>& displacementFootprints,
        const SubdTessellatedVertexLookup& tessellatedVertexLookup,
        const FaceVaryingSeams& faceVaryingSeams,
        const mcrt_common::Frustum& frustum,
        const scene_rdl2::math::Mat4d& world2render)
{
    size_t motionSampleCount = getMotionSamplesCount();
    size_t tessellatedVertexCount = getTessellatedMeshVertexCount();

    // need to map tessellated vertex id back to tessellated face and
    // local vertex index so that we can support primitive attribute
    // for displacment
    std::vector<int> vidToFaceId(tessellatedVertexCount, -1);
    std::vector<int> vidToFVIndex(tessellatedVertexCount, -1);
    for (size_t f = 0; f < getTessellatedMeshFaceCount(); ++f) {
        // vertex ids of quad
        int vid0 = mTessellatedIndices[sQuadVertexCount * f];
        vidToFaceId[vid0] = f;
        vidToFVIndex[vid0] = 0;
        int vid1 = mTessellatedIndices[sQuadVertexCount * f + 1];
        vidToFaceId[vid1] = f;
        vidToFVIndex[vid1] = 1;
        int vid2 = mTessellatedIndices[sQuadVertexCount * f + 2];
        vidToFaceId[vid2] = f;
        vidToFVIndex[vid2] = 2;
        int vid3 = mTessellatedIndices[sQuadVertexCount * f + 3];
        vidToFaceId[vid3] = f;
        vidToFVIndex[vid3] = 3;
    }

    // Displacement of Geometry....

    // tracking vertices that got displaced, we need this info for later
    // shading normal/derivatives computation
    bool* isDisplaced = scene_rdl2::util::alignedMallocArray<bool>(tessellatedVertexCount);
    MNRY_ASSERT_REQUIRE(isDisplaced);
    memset(isDisplaced, 0, sizeof(bool) * tessellatedVertexCount);

    tbb::blocked_range<size_t> range =
        tbb::blocked_range<size_t>(0, tessellatedVertexCount);
    tbb::parallel_for(range, [&](const tbb::blocked_range<size_t> &r) {
        mcrt_common::ThreadLocalState *tls = mcrt_common::getFrameUpdateTLS();
        shading::TLState *shadingTls = MNRY_VERIFY(tls->mShadingTls.get());
        Intersection isect;
        for (size_t v = r.begin(); v < r.end(); ++v) {
            int assignmentId = limitSurfaceSamples[v].mAssignmentId;
            if (assignmentId == -1) {
                continue;
            }
            const scene_rdl2::rdl2::Displacement* displacement =
                pRdlLayer->lookupDisplacement(assignmentId);
            if (displacement == nullptr) {
                continue;
            }

            // get primitive attribute table
            const AttributeTable* table = nullptr;
            if (displacement->hasExtension()) {
                table = displacement->
                        get<shading::RootShader>().getAttributeTable();
            }

            const scene_rdl2::rdl2::Geometry* geometry =
                pRdlLayer->lookupGeomAndPart(assignmentId).first;
            Vec2f st = mSurfaceSt(v);
            for (size_t t = 0; t < motionSampleCount; ++t) {
                Vec3f position = mTessellatedVertices(v, t);
                Vec3f normal = mSurfaceNormal(v, t);
                Vec3f dPds = mSurfaceDpds(v, t);
                Vec3f dPdt = mSurfaceDpdt(v, t);

                if (getIsReference()) {
                    // If this primitive is referenced by another geometry (i.e. instancing)
                    // then these attributes are given to us in local/object space.   Our 
                    // shaders expect all State attributes to be given in render space
                    // so we transform them here by world2render.   The local2world transform
                    // is assumed to be identity for references.
                    position = scene_rdl2::math::transformPoint(world2render, position);
                    dPds = scene_rdl2::math::transformVector(world2render, dPds);
                    dPdt = scene_rdl2::math::transformVector(world2render, dPdt);
                }

                // st footprint along the control face u param
                const Vec2f& dSt0 = displacementFootprints[v].mDst[0];
                // st footprint along the control face v param
                const Vec2f& dSt1 = displacementFootprints[v].mDst[1];
                isect.initDisplacement(tls, table, geometry, pRdlLayer,
                    assignmentId, position, normal, dPds, dPdt, st,
                    dSt0[0], dSt1[0], dSt0[1], dSt1[1]);
                fillDisplacementAttributes(vidToFaceId[v], vidToFVIndex[v],
                    isect);

                Vec3f displace;
                shading::displace(displacement, shadingTls, shading::State(&isect), &displace);
                mTessellatedVertices(v, t) += Vec3fa(displace, 0.f);
                isDisplaced[v] = true;
            }
        }
    });

    // stitch uv discontinuity area after displacement to avoid crack
    std::vector<std::vector<int>> vertexClusters;
    getOverlapVertexClusters(faceVaryingSeams, tessellatedVertexLookup,
        vertexClusters);
    range = tbb::blocked_range<size_t>(0, vertexClusters.size());
    tbb::parallel_for(range, [&](const tbb::blocked_range<size_t> &r) {
        for (size_t c = r.begin(); c < r.end(); ++c) {
            const std::vector<int>& vertexCluster = vertexClusters[c];
            bool needStitch = false;
            for (auto vid : vertexCluster) {
                if (isDisplaced[vid]) {
                    needStitch = true;
                    break;
                }
            }
            if (!needStitch) {
                continue;
            }
            for (size_t t = 0; t < motionSampleCount; ++t) {
                Vec3fa position(0.0f);
                for (auto vid : vertexCluster) {
                    position += mTessellatedVertices(vid, t);
                }
                position /= (float)vertexCluster.size();
                for (auto vid : vertexCluster) {
                    mTessellatedVertices(vid, t) = position;
                    isDisplaced[vid] = true;
                }
            }
        }
    });

    struct TempVertexData
    {
        scene_rdl2::math::Vec3d mNormal;
        scene_rdl2::math::Vec3d mDpds;
        scene_rdl2::math::Vec3d mDpdt;
        double mAreaWeight;
    };
    // accumulated normal/dpds/dpdt info with area weight sum that will be
    // normalized at the end
    TempVertexData* tempVertexData = scene_rdl2::util::alignedMallocArray<TempVertexData>(
        motionSampleCount * tessellatedVertexCount);
    MNRY_ASSERT_REQUIRE(tempVertexData);
    for (size_t i = 0; i < motionSampleCount * tessellatedVertexCount; ++i) {
        tempVertexData[i].mNormal = scene_rdl2::math::Vec3d(0.0);
        tempVertexData[i].mDpds = scene_rdl2::math::Vec3d(0.0);
        tempVertexData[i].mDpdt = scene_rdl2::math::Vec3d(0.0);
        tempVertexData[i].mAreaWeight = 0.0;
    }
    std::vector<tbb::spin_mutex> vertexMutex(tessellatedVertexCount);

    auto computeTriangleDerivatives = [&](int vid1, int vid2, int vid3) {
        const Vec2f& st1 = mSurfaceSt(vid1);
        const Vec2f& st2 = mSurfaceSt(vid2);
        const Vec2f& st3 = mSurfaceSt(vid3);
        for (size_t t = 0; t < motionSampleCount; ++t) {
            const Vec3fa& p1 = mTessellatedVertices(vid1, t);
            const Vec3fa& p2 = mTessellatedVertices(vid2, t);
            const Vec3fa& p3 = mTessellatedVertices(vid3, t);
            Vec2f dst1 = st2 - st1;
            Vec2f dst2 = st3 - st1;
            Vec3f dp1 = p2 - p1;
            Vec3f dp2 = p3 - p1;
            float det = dst1[0] * dst2[1] - dst2[0] * dst1[1];
            const float tolerance = 1.e-12f;
            const float condNum = tessCondNumber2x2SVD(
                dst1[0], dst1[1], // a, b
                dst2[0], dst2[1], // c, d
                tolerance);
            if (condNum >= gIllConditioned || scene_rdl2::math::abs(det) <= tolerance) {
                det = 1.0f;
            } else {
                det = 1.0f / det;
            }
            Vec3f dPdsf = det * ( dst2[1] * dp1 - dst1[1] * dp2);
            Vec3f dPdtf = det * (-dst2[0] * dp1 + dst1[0] * dp2);
            Vec3f normalf = scene_rdl2::math::cross(dp1, dp2);
            // intentionally uprez float to double to avoid
            // non determinant issue introduced by multithreading
            scene_rdl2::math::Vec3d normal(normalf[0], normalf[1], normalf[2]);
            scene_rdl2::math::Vec3d dPds(dPdsf[0], dPdsf[1], dPdsf[2]);
            scene_rdl2::math::Vec3d dPdt(dPdtf[0], dPdtf[1], dPdtf[2]);
            double area = scene_rdl2::math::length(normal);
            {
                tbb::spin_mutex::scoped_lock spinLock(vertexMutex[vid1]);
                TempVertexData& vertexData =
                    tempVertexData[motionSampleCount * vid1 + t];
                vertexData.mAreaWeight += area;
                vertexData.mNormal += normal;
                vertexData.mDpds += area * dPds;
                vertexData.mDpdt += area * dPdt;
            }
            {
                tbb::spin_mutex::scoped_lock spinLock(vertexMutex[vid2]);
                TempVertexData& vertexData =
                    tempVertexData[motionSampleCount * vid2 + t];
                vertexData.mAreaWeight += area;
                vertexData.mNormal += normal;
                vertexData.mDpds += area * dPds;
                vertexData.mDpdt += area * dPdt;
            }
            {
                tbb::spin_mutex::scoped_lock spinLock(vertexMutex[vid3]);
                TempVertexData& vertexData =
                    tempVertexData[motionSampleCount * vid3 + t];
                vertexData.mAreaWeight += area;
                vertexData.mNormal += normal;
                vertexData.mDpds += area * dPds;
                vertexData.mDpdt += area * dPdt;
            }
        }
    };
    // recompute the normal and derivatives after vertices got displaced
    range = tbb::blocked_range<size_t>(0, getTessellatedMeshFaceCount());
    tbb::parallel_for(range, [&](const tbb::blocked_range<size_t> &r) {
        for (size_t f = r.begin(); f < r.end(); ++f) {
            int vid1 = mTessellatedIndices[sQuadVertexCount * f    ];
            int vid2 = mTessellatedIndices[sQuadVertexCount * f + 1];
            int vid3 = mTessellatedIndices[sQuadVertexCount * f + 2];
            int vid4 = mTessellatedIndices[sQuadVertexCount * f + 3];
            if (!isDisplaced[vid1] && !isDisplaced[vid2] &&
                !isDisplaced[vid3] && !isDisplaced[vid4]) {
                continue;
            }
            if (vid1 == vid4) {
                computeTriangleDerivatives(vid1, vid2, vid3);
            } else {
                computeTriangleDerivatives(vid2, vid4, vid1);
                computeTriangleDerivatives(vid2, vid3, vid4);
            }
        }
    });
    // normalized recomputed normal and derivatives with area weight sum
    bool hasBadDisplacedNormal = false;
    range = tbb::blocked_range<size_t>(0, tessellatedVertexCount);
    tbb::parallel_for(range, [&](const tbb::blocked_range<size_t> &r) {
        for (size_t v = r.begin(); v < r.end(); ++v) {
            if (!isDisplaced[v]) {
                continue;
            }
            for (size_t t = 0; t < motionSampleCount; ++t) {
                const TempVertexData& vertexData =
                    tempVertexData[motionSampleCount * v + t];
                // skip unassigned part
                if (scene_rdl2::math::isZero(vertexData.mAreaWeight)) {
                    continue;
                }
                double invWeight = 1.0 / vertexData.mAreaWeight;
                scene_rdl2::math::Vec3d normal = normalize(vertexData.mNormal);
                scene_rdl2::math::Vec3d dPds = invWeight * vertexData.mDpds;
                scene_rdl2::math::Vec3d dPdt = invWeight * vertexData.mDpdt;
                if (scene_rdl2::math::isFinite(normal)) {
                    mSurfaceNormal(v, t) = Vec3f(normal[0], normal[1], normal[2]);
                    mSurfaceDpds(v, t) = Vec3f(dPds[0], dPds[1], dPds[2]);
                    mSurfaceDpdt(v, t) = Vec3f(dPdt[0], dPdt[1], dPdt[2]);
                } else {
                    // there are nasty cases like two duplicated faces
                    // with opposite orientations can result to a nan result
                    // assign a valid but meaningless value in this case then
                    mSurfaceDpds(v, t) =  Vec3f(1, 0, 0);
                    mSurfaceDpdt(v, t) =  Vec3f(0, 1, 0);
                    mSurfaceNormal(v, t) = Vec3f(0, 0, 1);
                    hasBadDisplacedNormal = true;
                }
            }
        }
    });

    if (hasBadDisplacedNormal) {
        const scene_rdl2::rdl2::Geometry* pRdlGeometry = getRdlGeometry();
        MNRY_ASSERT(pRdlGeometry != nullptr);
        pRdlGeometry->info("mesh ", getName(),
            " contains bad displaced normal that may cause incorrect"
            " rendering result");
    }

    scene_rdl2::util::alignedFreeArray(isDisplaced);
    scene_rdl2::util::alignedFreeArray(tempVertexData);
}

void
OpenSubdivMesh::fillDisplacementAttributes(int tessFaceId, int vIndex,
        Intersection& intersection) const
{
    const AttributeTable* table = intersection.getTable();
    if (table == nullptr) {
        return;
    }
    int controlFaceId = mTessellatedToControlFace[tessFaceId];
    const Attributes* attributes = getAttributes();
    for (const auto key : table->getRequiredAttributes()) {
        if (!attributes->isSupported(key)) {
            continue;
        }
        switch (key.getType()) {
        case scene_rdl2::rdl2::TYPE_FLOAT:
            {
            TypedAttributeKey<float> typedKey(key);
            intersection.setAttribute(typedKey,
                getAttribute(typedKey, controlFaceId, tessFaceId, vIndex));
            }
            break;
        case scene_rdl2::rdl2::TYPE_RGB:
            {
            TypedAttributeKey<scene_rdl2::math::Color> typedKey(key);
            intersection.setAttribute(typedKey,
                getAttribute(typedKey, controlFaceId, tessFaceId, vIndex));
            }
            break;
        case scene_rdl2::rdl2::TYPE_VEC2F:
            {
            TypedAttributeKey<Vec2f> typedKey(key);
            intersection.setAttribute(typedKey,
                getAttribute(typedKey, controlFaceId, tessFaceId, vIndex));
            }
            break;
        case scene_rdl2::rdl2::TYPE_VEC3F:
            {
            TypedAttributeKey<Vec3f> typedKey(key);
            intersection.setAttribute(typedKey,
                getAttribute(typedKey, controlFaceId, tessFaceId, vIndex));
            }
            break;
        default:
            break;
        }
    }
    for (const auto key : table->getOptionalAttributes()) {
        if (!attributes->isSupported(key)) {
            continue;
        }
        switch (key.getType()) {
        case scene_rdl2::rdl2::TYPE_FLOAT:
            {
            TypedAttributeKey<float> typedKey(key);
            intersection.setAttribute(typedKey,
                getAttribute(typedKey, controlFaceId, tessFaceId, vIndex));
            }
            break;
        case scene_rdl2::rdl2::TYPE_RGB:
            {
            TypedAttributeKey<scene_rdl2::math::Color> typedKey(key);
            intersection.setAttribute(typedKey,
                getAttribute(typedKey, controlFaceId, tessFaceId, vIndex));
            }
            break;
        case scene_rdl2::rdl2::TYPE_VEC2F:
            {
            TypedAttributeKey<Vec2f> typedKey(key);
            intersection.setAttribute(typedKey,
                getAttribute(typedKey, controlFaceId, tessFaceId, vIndex));
            }
            break;
        case scene_rdl2::rdl2::TYPE_VEC3F:
            {
            TypedAttributeKey<Vec3f> typedKey(key);
            intersection.setAttribute(typedKey,
                getAttribute(typedKey, controlFaceId, tessFaceId, vIndex));
            }
            break;
        default:
            break;
        }
    }
    if (mFaceVaryingAttributes) {
        // handle face varying attributes
        float w[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        w[vIndex] = 1.0f;
        mFaceVaryingAttributes->fillAttributes(intersection, tessFaceId,
            w[0], w[1], w[2], w[3]);
    }
}

template <typename T> T
OpenSubdivMesh::getAttribute(const TypedAttributeKey<T>& key,
        int controlFaceId, int tessFaceId, int vIndex) const
{
    const Attributes* attributes = getAttributes();
    MNRY_ASSERT(attributes->isSupported(key));
    T result;
    AttributeRate rate = attributes->getRate(key);
    switch (rate) {
    case RATE_CONSTANT:
        result = getConstantAttribute(key);
        break;
    case RATE_UNIFORM:
        result = getUniformAttribute(key, controlFaceId);
        break;
    case RATE_VARYING:
        result = getVaryingAttribute(
            key, mTessellatedIndices[sQuadVertexCount * tessFaceId + vIndex]);
        break;
    case RATE_FACE_VARYING:
        // handled by FaceVaryingAttributes
        break;
    case RATE_VERTEX:
        result = getVertexAttribute(
            key, mTessellatedIndices[sQuadVertexCount * tessFaceId + vIndex]);
        break;
    default:
        MNRY_ASSERT(false, "unknown attribute rate");
        break;
    }
    return result;
}

void
OpenSubdivMesh::setTransform(const XformSamples& xforms, float shutterOpenDelta,
                             float shutterCloseDelta)
{
    mControlMeshData->mXforms = xforms;
    mControlMeshData->mShutterOpenDelta = shutterOpenDelta;
    mControlMeshData->mShutterCloseDelta = shutterCloseDelta;

    // Needed for volume shader
    // We'll set this to the world2render xform in tessellate
    // if this is a shared primitive
    mPrimToRender = scene_rdl2::math::Mat4f(xforms[0]);
}

} // namespace internal
} // namespace geom
} // namespace moonray

