// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file MeshTessellationUtil.h
///

#pragma once

#include <moonray/rendering/geom/prim/Mesh.h>

#include <moonray/rendering/geom/PolygonMesh.h>
#include <moonray/rendering/geom/SubdivisionMesh.h>

#include <scene_rdl2/render/util/Arena.h>

namespace moonray {
namespace geom {
namespace internal {

// PolyFaceTopology describes a triangle or quad face in polygon mesh
// that is going to be further tessellated (for displacement purpose)
// with a set of vertex/edge id.
// It is used for input quad -> tessellated face lookup with the help of
// PolyTessellatedVertexLookup.
//
// The representation for quad face looks like:
//
//               corner[3]      corner[2]
//              *-------------*
//              |  edge[2]    |
//              |             |
//      edge[3] |             | edge[1]
//              |             |
//              |  edge[0]    |
//              *-------------*
//               corner[0]      corner[1]
//
// The presentation for triangle face looks like:
//
//                             corner[2]
//                           /*
//                         /  |
//               edge[2] /    |
//                     /      | edge[1]
//                   /        |
//                 /  edge[0] |
//               *------------*
//               corner[0]      corner[1]
//
struct PolyFaceTopology
{
    std::array<int, sQuadVertexCount> mCornerVertexId;
    std::array<int, sQuadVertexCount> mEdgeId;
};

// PolyTessellationFactor describes how many segments each edge on input face
// will be tessellated (the value can be different from face to face if we
// do adaptive tessellation) for polygon mesh
struct PolyTessellationFactor
{
    // edge id is only used during computation stage,
    // the computation result is stored in edge factor
    union {
        std::array<int, sQuadVertexCount> mEdgeId;
        std::array<int, sQuadVertexCount> mEdgeFactor;
    };
};


// SubdQuadTopology describes a control quad in subdivision mesh
// with a set of vertex/edge id.
// It is used for control quad -> tessellated face lookup with the help of
// SubdTessellatedVertexLookup.
// If a quad is regular control quad (major case for well modeled Catmull-Clark
// control mesh), its representation looks like:
//
//     corner[3]      mid[2]          corner[2]
//     *--------------x--------------*
//     |   edge1[2]       edge0[2]   |
//     |                             |
//     |                             |
//     | edge0[3]                    | edge1[1]
//     |                             |
//     |                             |
//     x mid[3]                      x mid[1]
//     |                             |
//     |                             |
//     | edge1[3]                    | edge0[1]
//     |                             |
//     |                             |
//     |    edge0[0]  mid[0]         |
//     *--------------x--------------*
//     corner[0]           edge1[0]   corner[1]
//
// If a quad comes from quadrangulated n-gons, its representation looks like:
//
//               corner[3]      corner[2]
//              *-------------*
//              |  edge0[2]   |
//              |             |
//     edge0[3] |             | edge0[1]
//              |             |
//              |  edge0[0]   |
//              *-------------*
//               corner[0]      corner[1]
struct SubdQuadTopology
{
    bool nonQuadParent() const
    {
        return mNonQuadParent;
    }
    std::array<int, sQuadVertexCount> mCornerVertexId;
    std::array<int, sQuadVertexCount> mEdgeId0;
    // only regular control quad (instead of quads from quadrangulated n-gons)
    // has valid values for mMidEdgeVertexId and mEdgeId1
    std::array<int, sQuadVertexCount> mMidEdgeVertexId;
    std::array<int, sQuadVertexCount> mEdgeId1;
    int mControlFaceId;
    bool mHasAssignment;
    bool mNonQuadParent;
};

// SubdTessellationFactor describes how many segments each edge on control quad
// will be tessellated (the value can be different from face to face if we
// do adaptive tessellation) for subdivision mesh
// The number of SubdTessellationFactor should match the number of
// SubdQuadTopology. The reason we separate these two helpers is
// because SubdTessellationFactor can be reused across multiple channels of
// face varying attributes, position and texture buffer, while SubdQuadTopology
// can be different for each of them since they use different index buffer
// (and thus different overlap edges/vertices)
struct SubdTessellationFactor
{
    // edge id is only used during computation stage,
    // the computation result is stored in edge factor
    union {
        std::array<int, sQuadVertexCount> mEdgeId0;
        std::array<int, sQuadVertexCount> mEdge0Factor;
    };
    // only regular control quad (instead of quads from quadrangulated n-gons)
    // has valid values for the following members
    union {
        std::array<int, sQuadVertexCount> mEdgeId1;
        std::array<int, sQuadVertexCount> mEdge1Factor;
    };
};


// see the following link for the heuristic detail explanation
// https://developer.nvidia.com/content/dynamic-hardware-tessellation-basics
// this is the same method OpenSubdiv GPU evaluator uses
// in util function OsdGetTessLevelsAdaptiveLimitPoints()
finline int
computeEdgeVertexCount(const Vec3f& v0, const Vec3f& v1,
        float edgesPerScreenHeight, const scene_rdl2::math::Mat4f& c2s)
{
    // calculate the bounding sphere of this edge
    Vec3f pCenter = 0.5f * (v0 + v1);
    float diameter = distance(v0, v1);
    // projection[1][1] = 1 / tan(fov / 2);
    // calculate edge tessellation factor based on frustum information
    int edgeVertexCount = 0.5f * edgesPerScreenHeight *
        scene_rdl2::math::abs(diameter * c2s[1][1] / pCenter.z) - 1;
    return edgeVertexCount;
}


// This helper struct stores 3 ids to represent a control edge.
//   edgeid0  edgeid1
// |---------x---------|
//           midEdgeVertexId
struct ControlEdge
{
    ControlEdge(int edgeId0, int midEdgeVertexId, int edgeId1):
        mEdgeId0(edgeId0), mMidEdgeVertexId(midEdgeVertexId), mEdgeId1(edgeId1)
    {}

    int mEdgeId0;
    int mMidEdgeVertexId;
    int mEdgeId1;
};

// The list of control edges/vertices overlap each other in face varying
// attributes discontinuity boundary. They will be mapped to a list of
// tessellation vertex ids that need to be processed after displacement
// to avoid cracks artifact
struct FaceVaryingSeams
{
    std::unordered_map<int, std::vector<int>> mOverlapVertices;
    std::unordered_map<int64_t, std::vector<ControlEdge>> mOverlapEdges;
};


// TopologyIdLookup provides edge id lookup for tessellation utilities usage
// by analyzing the input control face topology
class TopologyIdLookup
{
public:
    int getEdgeId(int v0, int v1) const
    {
        const auto& result = mEdgeIds.find(getEdgeKey(v0, v1));
        return result == mEdgeIds.end() ? -1 : result->second;
    }

    int getCoarseVertexCount() const
    {
        return mCoarseVertexCount;
    }

    int getCoarseEdgeCount() const
    {
        return mCoarseEdgeCount;
    }

protected:
    // TopologyIdLookup is not intended for direct use but holding shared data
    // across subclasses
    TopologyIdLookup(const int coarseVertexCount):
        mCoarseVertexCount(coarseVertexCount), mCoarseEdgeCount(0)
    {}

    virtual ~TopologyIdLookup() = default;

    int64_t getEdgeKey(int64_t v0, int64_t v1) const
    {
        return v0 < v1 ? (v1 << 32) + v0 : (v0 << 32) + v1;
    }

protected:
    std::unordered_map<int64_t, int> mEdgeIds;
    int mCoarseVertexCount;
    int mCoarseEdgeCount;
};

class PolyTopologyIdLookup : public TopologyIdLookup
{
public:
    PolyTopologyIdLookup(const int vertexCount, const int faceVertexCount,
            const PolygonMesh::IndexBuffer& indices);
};

// SubdTopologyIdLookup further provides edge/face/vertex id lookup for
// subdivision mesh specific tessellation utilities
class SubdTopologyIdLookup : public TopologyIdLookup
{
public:
    SubdTopologyIdLookup(const int faceVaryingVertexCount,
            const SubdivisionMesh::FaceVertexCount& faceVertexCount,
            const SubdivisionMesh::IndexBuffer& faceVaryingIndices,
            bool noTessellation);

    SubdTopologyIdLookup(const int faceVaryingVertexCount,
            const SubdivisionMesh::FaceVertexCount& faceVertexCount,
            const SubdivisionMesh::IndexBuffer& controlIndices,
            const SubdivisionMesh::IndexBuffer& faceVaryingIndices,
            bool noTessellation);

    const FaceVaryingSeams& getFaceVaryingSeams() const
    {
        return mFaceVaryingSeams;
    }

    int getVertexChildVertex(int vertexId) const
    {
        return vertexId;
    }

    int getEdgeChildVertex(int edgeId) const
    {
        return mCoarseVertexCount + edgeId;
    }

    int getFaceChildVertex(int faceId) const
    {
        return mCoarseVertexCount + mCoarseEdgeCount + faceId;
    }

    // for case that we actually tessellate mesh
    int getSplitVertexCount() const
    {
        return mSplitVertexCount;
    }

    int getSplitEdgeCount() const
    {
        return mSplitEdgeCount;
    }

    int getQuadrangulatedFaceCount() const
    {
        return mQuadrangulatedFaceCount;
    }

private:
    int getEdgeChildEdge(int edgeId, bool firstChild) const
    {
        return firstChild ? 2 * edgeId : 2 * edgeId + 1;
    }

private:
    int mCoarseFaceCount;
    int mSplitVertexCount;
    int mSplitEdgeCount;
    int mQuadrangulatedFaceCount;
    FaceVaryingSeams mFaceVaryingSeams;
};


// TessellatedVertexLookup provides getters to lookup the final
// tessellated vertex ids through edge/vertex/face ids
class TessellatedVertexLookup
{
public:
    int getEdgeVertexCount(int edgeId) const
    {
        return mOuterEdgeVertexIdsCount[edgeId];
    }

    int getTessellatedVertexCount() const
    {
        return mTotalVertexCount;
    }

    int getInteriorVertexId(int face, int row, int col,
            int interiorColVertexCount) const
    {
        return mInteriorVertexIdsStart[face] +
            interiorColVertexCount * row + col;
    }

    int getTessellatedVertexId(int controlVertexId) const
    {
        return mOuterRingVertexIds[controlVertexId];
    }

protected:
    // TessellatedVertexLookup is not intended for direct use but
    // holding shared data across subclasses
    TessellatedVertexLookup(): mTotalVertexCount(0)
    {}

    virtual ~TessellatedVertexLookup() = default;

protected:
    std::vector<int> mOuterRingVertexIds;
    std::vector<int> mOuterEdgeVertexIdsStart;
    std::vector<int> mOuterEdgeVertexIdsCount;
    std::vector<int> mInteriorVertexIdsStart;
    size_t mTotalVertexCount;
};


// PolyTessellatedVertexLookup implements polygon mesh specific vertex id
// lookup through PolyFaceTopology
class PolyTessellatedVertexLookup : public TessellatedVertexLookup
{
public:
    PolyTessellatedVertexLookup(
            const std::vector<PolyFaceTopology>& faceTopologies,
            const PolyTopologyIdLookup& topologyIdLookup,
            const std::vector<PolyTessellationFactor>& tessellationFactors,
            size_t faceVertexCount);

    // the convention for edge tessellation vertex id lookup is:
    // the edge vertex id increase from the cornervertex with smaller vertex id
    // for example:
    //       v0    v1    v2
    // x0----*-----*-----*-----x1
    //
    // x0 < x1
    // if reverseEdge:
    //     getEdgeVertexId(edgeId, 0, reverseEdge) = v2
    //     getEdgeVertexId(edgeId, 1, reverseEdge) = v1
    //     getEdgeVertexId(edgeId, 2, reverseEdge) = v0
    // else:
    //     getEdgeVertexId(edgeId, 0, reverseEdge) = v0
    //     getEdgeVertexId(edgeId, 1, reverseEdge) = v1
    //     getEdgeVertexId(edgeId, 2, reverseEdge) = v2
    //
    int getEdgeVertexId(int edgeId, int n, bool reverseEdge) const
    {
        MNRY_ASSERT(n < mOuterEdgeVertexIdsCount[edgeId]);
        int start = mOuterEdgeVertexIdsStart[edgeId];
        return reverseEdge ?
            start + mOuterEdgeVertexIdsCount[edgeId] - 1 - n : start + n;
    }

    // for quad face usage
    int getInteriorRowVertexCount(const PolyFaceTopology& faceTopology) const;

    // for quad face usage
    int getInteriorColVertexCount(const PolyFaceTopology& faceTopology) const;

    // for triangle face usage
    // note that edge vertex count doesn't include corner vertex,
    // so if an edge is tessellated to n segments, its edgeVertexCount is
    // n -1 instead of n + 1
    int getMaxEdgeVertexCount(const PolyFaceTopology& faceTopology) const;

    // for triangle face usage
    // given a base triangle with maxEdgeVertexCount
    // return the vertex id on ith interior ring, cth edge, nth vertex
    // i = interiorRingIndex
    // c = cornerIndex
    int getTriangleInteriorVertexId(int face, int maxEdgeVertexCount,
            int interiorRingIndex, int cornerIndex, int n) const;

    // for triangle face usage
    int getTriangleRingCount(const PolyFaceTopology& faceTopology) const;

    bool noRingToStitch(const PolyFaceTopology& faceTopology) const;

    size_t getEstimatedFaceCount() const
    {
        return mEstimatedFaceCount;
    }

private:
    size_t mFaceVertexCount;
    // rough estimation of the final tessellated face count
    size_t mEstimatedFaceCount;
};


// SubdTessellatedVertexLookup implements subdivision mesh specific vertex id
// lookup through SubdQuadTopology
class SubdTessellatedVertexLookup : public TessellatedVertexLookup
{
public:
    SubdTessellatedVertexLookup(
            const std::vector<SubdQuadTopology>& quadTopologies,
            const SubdTopologyIdLookup& topologyIdLookup,
            const std::vector<SubdTessellationFactor>& tessellationFactors,
            bool noTessellation);

    // the convention for edge tessellation vertex id lookup is:
    // the edge vertex id increase from the end point that is not on the
    // middle of control edge to the end point that is on the middle of
    // control edge
    // for example:
    //       v0    v1    v2
    // x0----*-----*-----*-----x1
    //
    // x1 is on the middle of control edge
    // v1 = v0 + 1
    // v2 = v1 + 1
    //
    // if endAtEdgeMiddle:
    //     getEdgeVertexId(edgeId, 0, endAtEdgeMiddle) = v0
    //     getEdgeVertexId(edgeId, 1, endAtEdgeMiddle) = v1
    //     getEdgeVertexId(edgeId, 2, endAtEdgeMiddle) = v2
    // else:
    //     getEdgeVertexId(edgeId, 0, endAtEdgeMiddle) = v2
    //     getEdgeVertexId(edgeId, 1, endAtEdgeMiddle) = v1
    //     getEdgeVertexId(edgeId, 2, endAtEdgeMiddle) = v0
    //
    int getEdgeVertexId(int edgeId, int n, bool endAtEdgeMiddle) const
    {
        MNRY_ASSERT(n < mOuterEdgeVertexIdsCount[edgeId]);
        int start = mOuterEdgeVertexIdsStart[edgeId];
        return endAtEdgeMiddle ?
            start + n : start + mOuterEdgeVertexIdsCount[edgeId] - 1 - n;
    }

    int getInteriorRowVertexCount(const SubdQuadTopology& quadTopology) const;

    int getInteriorColVertexCount(const SubdQuadTopology& quadTopology) const;

    bool noRingToStitch(const SubdQuadTopology& quadTopology) const;

    int getMaxEdgeResolution() const
    {
        return mMaxEdgeResolution;
    }

private:
    int mMaxEdgeResolution;
};

// Each StitchPoint represent a vertex that needed to be stitched between
// inner outer ring of control face. For example, the below illustrates
// a down row inner ring and outer ring of a control face
//
//     *---*--*--*---*    <- inner ring
//    / \ / \ | / \ / \
//   *---*----*----*---*  <- outer ring
struct StitchPoint
{
    int mVertexId;
    float mT;
    float mDeltaT;
};

// Given inner ring and outer ring StitchPoint:
//     *---*--*--*---*    <- inner ring
//   *---*----*----*---*  <- outer ring
//
// This function weave out the final tessellated faces index buffer:
//     *---*--*--*---*    <- inner ring
//    / \ / \ | / \ / \
//   *---*----*----*---*  <- outer ring
//
// By stitching vertex based on slope rate comparison. The idea is similar
// to Bresenham's line algorithm
void
stitchRings(const StitchPoint* innerRing, const Vec2f* innerRingUv,
        int innerRingVertexCount,
        const StitchPoint* outerRing, const Vec2f* outerRingUv,
        int outerRingVertexCount,
        std::vector<geom::Primitive::IndexType>& indices,
        const int controlFaceId, std::vector<int>* tessellatedToControlFace,
        std::vector<Vec2f>* faceVaryingUv);

void
stitchRings(const StitchPoint* innerRing, int innerRingVertexCount,
        const StitchPoint* outerRing, int outerRingVertexCount,
        std::vector<geom::Primitive::IndexType>& indices,
        const int controlFaceId, std::vector<int>* tessellatedToControlFace);

// output a part of inner ring and outer ring based on cornerIndex
// 0 for bottom side
// 1 for right side
// 2 for top side
// 3 for left side
// subdivision mesh version
void
generateSubdStitchRings(scene_rdl2::alloc::Arena* arena,
        const SubdQuadTopology& quadTopology, int fid, size_t cornerIndex,
        const SubdTessellatedVertexLookup& tessellatedVertexLookup,
        int interiorRowVertexCount, int interiorColVertexCount,
        StitchPoint* & innerRing, int& innerRingVertexCount,
        StitchPoint* & outerRing, int& outerRingVertexCount);

// quad polygon mesh version
void
generateQuadStitchRings(scene_rdl2::alloc::Arena* arena,
        const PolyFaceTopology& quadTopology, int fid, size_t cornerIndex,
        const PolyTessellatedVertexLookup& tessellatedVertexLookup,
        int interiorRowVertexCount, int interiorColVertexCount,
        StitchPoint* & innerRing, Vec2f* & innerRingUv,
        int& innerRingVertexCount,
        StitchPoint* & outerRing, Vec2f* & outerRingUv,
        int& outerRingVertexCount);

// output a part of inner ring and outer ring based on cornerIndex
// 0 for v0 -> v1 side
// 1 for v1 -> v2 side
// 2 for v2 -> v0 side
// triangle polygon mesh version
void
generateTriangleStitchRings(scene_rdl2::alloc::Arena* arena,
        const PolyFaceTopology& faceTopology, int fid, size_t cornerIndex,
        const PolyTessellatedVertexLookup& tessellatedVertexLookup,
        int maxEdgeVertexCount, float ringUvDelta,
        StitchPoint* & innerRing, Vec2f* & innerRingUv,
        int& innerRingVertexCount,
        StitchPoint* & outerRing, Vec2f* & outerRingUv,
        int& outerRingVertexCount);

// when the tessellation factors of bottom/up edges are identical and
// the tessellationfactors of left/right edges are identical, collect
// all the vertex ids in row by row order with this utility,
// which can be used to generate better topology index buffer
void
collectUniformTessellatedVertices(scene_rdl2::alloc::Arena *arena,
        const PolyFaceTopology& faceTopology, int baseFaceId,
        const PolyTessellatedVertexLookup& tessellatedVertexLookup,
        int* & vids, int& rowVertexCount, int& colVertexCount);

} // namespace internal
} // namespace geom
} // namespace moonray


