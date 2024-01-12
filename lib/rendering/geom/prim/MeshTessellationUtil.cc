// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file MeshTessellationUtil.cc
///

#include "MeshTessellationUtil.h"
#include <numeric>

namespace moonray {
namespace geom {
namespace internal {

PolyTopologyIdLookup::PolyTopologyIdLookup(
        const int vertexCount, const int faceVertexCount,
        const PolygonMesh::IndexBuffer& indices):
    TopologyIdLookup(vertexCount)
{
    MNRY_ASSERT_REQUIRE(faceVertexCount == sQuadVertexCount ||
        faceVertexCount == sTriangleVertexCount);
    MNRY_ASSERT_REQUIRE(indices.size() % faceVertexCount == 0);

    int faceCount = indices.size() / faceVertexCount;
    int indexOffset = 0;
    for (int f = 0; f < faceCount; ++f) {
        for (int v = 0; v < faceVertexCount; ++v) {
            int v0 = indices[indexOffset + v];
            int v1 = indices[indexOffset + (v + 1) % faceVertexCount];
            int64_t edgeKey = getEdgeKey(v0, v1);
            if (mEdgeIds.find(edgeKey) == mEdgeIds.end()) {
                mEdgeIds[edgeKey] = mCoarseEdgeCount++;
            }
        }
        indexOffset += faceVertexCount;
    }
}

SubdTopologyIdLookup::SubdTopologyIdLookup(const int faceVaryingVertexCount,
        const SubdivisionMesh::FaceVertexCount& faceVertexCount,
        const SubdivisionMesh::IndexBuffer& faceVaryingIndices,
        bool noTessellation):
    TopologyIdLookup(faceVaryingVertexCount),
    mCoarseFaceCount(faceVertexCount.size()),
    mSplitVertexCount(0),
    mSplitEdgeCount(0),
    mQuadrangulatedFaceCount(0)
{
    // loop through control faces to sort out the coarse edges id lookup
    int indexOffset = 0;
    for (int f = 0; f < mCoarseFaceCount; ++f) {
        int nFv = faceVertexCount[f];
        for (int v = 0; v < nFv; ++v) {
            int v0 = faceVaryingIndices[indexOffset + v];
            int v1 = faceVaryingIndices[indexOffset + (v + 1) % nFv];
            int64_t edgeKey = getEdgeKey(v0, v1);
            if (mEdgeIds.find(edgeKey) == mEdgeIds.end()) {
                mEdgeIds[edgeKey] = mCoarseEdgeCount++;
            }
        }
        indexOffset += nFv;
    }
    if (noTessellation) {
        mQuadrangulatedFaceCount = std::accumulate(
            faceVertexCount.begin(), faceVertexCount.end(), 0,
            [](int quadCount, int nFv) {
                return nFv == sQuadVertexCount ?
                    quadCount + 1 : quadCount + nFv;
            });
    } else {
        // every coarse edge map to one middle point vertex
        // every coarse face map to one face central vertex
        mSplitVertexCount = mCoarseVertexCount + mCoarseEdgeCount +
            mCoarseFaceCount;
        // splitEdgeCount will be continuously updated in the following face
        // traversal. Each non quad n-gons will add in n more split edges
        // for future lookup (these edges don't appear in coarseEdgeIds
        // so we need to book keeping by ourself)
        mSplitEdgeCount = 2 * mCoarseEdgeCount;
        // a non quad n-gons control face will be quadrangulated to
        // n control quads in opensubdiv (Catmull-Clark scheme)
        indexOffset = 0;
        for (int f = 0; f < mCoarseFaceCount; ++f) {
            int nFv = faceVertexCount[f];
            for (int v = 0; v < nFv; ++v) {
                int vid0 = faceVaryingIndices[indexOffset + v];
                int vid1 = faceVaryingIndices[indexOffset + (v + 1) % nFv];
                int eid = getEdgeId(vid0, vid1);
                int vMidId = getEdgeChildVertex(eid);
                int eid0, eid1;
                if (vid0 < vid1) {
                    eid0 = getEdgeChildEdge(eid, true);
                    eid1 = getEdgeChildEdge(eid, false);
                } else {
                    eid0 = getEdgeChildEdge(eid, false);
                    eid1 = getEdgeChildEdge(eid, true);
                }
                mEdgeIds[getEdgeKey(vid0, vMidId)] = eid0;
                mEdgeIds[getEdgeKey(vMidId, vid1)] = eid1;

                if (nFv != sQuadVertexCount) {
                    int vFaceId = getFaceChildVertex(f);
                    int64_t splitEdgeKey = getEdgeKey(vMidId, vFaceId);
                    if (mEdgeIds.find(splitEdgeKey) == mEdgeIds.end()) {
                        mEdgeIds[splitEdgeKey] = mSplitEdgeCount++;
                    }
                    mQuadrangulatedFaceCount += nFv;
                } else {
                    mQuadrangulatedFaceCount += 1;
                }
            }
            indexOffset += nFv;
        }
    }
}

SubdTopologyIdLookup::SubdTopologyIdLookup(const int faceVaryingVertexCount,
        const SubdivisionMesh::FaceVertexCount& faceVertexCount,
        const SubdivisionMesh::IndexBuffer& controlIndices,
        const SubdivisionMesh::IndexBuffer& faceVaryingIndices,
        bool noTessellation):
    SubdTopologyIdLookup(faceVaryingVertexCount, faceVertexCount,
    faceVaryingIndices, noTessellation)
{
    // loop through control faces to locate face varying seams
    std::unordered_map<int, std::unordered_set<int>> overlapVertices;
    std::unordered_map<int64_t, std::unordered_set<int64_t>> overlapEdges;
    int indexOffset = 0;
    for (int f = 0; f < mCoarseFaceCount; ++f) {
        int nFv = faceVertexCount[f];
        for (int v = 0; v < nFv; ++v) {
            int v0 = faceVaryingIndices[indexOffset + v];
            int v1 = faceVaryingIndices[indexOffset + (v + 1) % nFv];
            int64_t edgeKey = getEdgeKey(v0, v1);

            int seamV0 = controlIndices[indexOffset + v];
            int seamV1 = controlIndices[indexOffset + (v + 1) % nFv];
            int64_t seamId = getEdgeKey(seamV0, seamV1);
            overlapEdges[seamId].insert(edgeKey);
            overlapVertices[seamV0].insert(v0);
        }
        indexOffset += nFv;
    }

    for (const auto& vertices : overlapVertices) {
        if (vertices.second.size() > 1) {
            auto& overlapVertices =
                mFaceVaryingSeams.mOverlapVertices[vertices.first];
            overlapVertices.insert(overlapVertices.begin(),
                vertices.second.begin(), vertices.second.end());
        }
    }
    // in the case that a control edge will be further tessellated,
    // we need to record control edges that overlap each other for
    // stitching after displacement
    if (!noTessellation) {
        for (const auto& edges : overlapEdges) {
            if (edges.second.size() > 1) {
                mFaceVaryingSeams.mOverlapEdges[edges.first].clear();
            }
        }
        indexOffset = 0;
        for (int f = 0; f < mCoarseFaceCount; ++f) {
            int nFv = faceVertexCount[f];
            for (int v = 0; v < nFv; ++v) {
                int vid0 = faceVaryingIndices[indexOffset + v];
                int vid1 = faceVaryingIndices[indexOffset + (v + 1) % nFv];
                int eid = getEdgeId(vid0, vid1);
                int vMidId = getEdgeChildVertex(eid);
                int eid0, eid1;
                if (vid0 < vid1) {
                    eid0 = getEdgeChildEdge(eid, true);
                    eid1 = getEdgeChildEdge(eid, false);
                } else {
                    eid0 = getEdgeChildEdge(eid, false);
                    eid1 = getEdgeChildEdge(eid, true);
                }
                int seamV0 = controlIndices[indexOffset + v];
                int seamV1 = controlIndices[indexOffset + (v + 1) % nFv];
                int64_t seamId = getEdgeKey(seamV0, seamV1);
                auto it = mFaceVaryingSeams.mOverlapEdges.find(seamId);
                if (it != mFaceVaryingSeams.mOverlapEdges.end()) {
                    if (seamV0 < seamV1) {
                        it->second.emplace_back(eid0, vMidId, eid1);
                    } else {
                        it->second.emplace_back(eid1, vMidId, eid0);
                    }
                }
            }
            indexOffset += nFv;
        }
    }
}

PolyTessellatedVertexLookup::PolyTessellatedVertexLookup(
        const std::vector<PolyFaceTopology>& faceTopologies,
        const PolyTopologyIdLookup& topologyIdLookup,
        const std::vector<PolyTessellationFactor>& tessellationFactors,
        size_t faceVertexCount):
    TessellatedVertexLookup(),
    mFaceVertexCount(faceVertexCount),
    mEstimatedFaceCount(0)
{
    MNRY_ASSERT_REQUIRE(faceVertexCount == sQuadVertexCount ||
        faceVertexCount == sTriangleVertexCount);
    int coarseVertexCount = topologyIdLookup.getCoarseVertexCount();
    int coarseEdgeCount = topologyIdLookup.getCoarseEdgeCount();
    // cache the result when loop through every faceTopology's
    // vertices, edges and faces so shared vertices only got counted once
    mOuterRingVertexIds.resize(coarseVertexCount, -1);
    mOuterEdgeVertexIdsStart.resize(coarseEdgeCount, -1);
    mOuterEdgeVertexIdsCount.resize(coarseEdgeCount, -1);
    mInteriorVertexIdsStart.resize(faceTopologies.size(), -1);

    for (size_t f = 0; f < faceTopologies.size(); ++f) {
        const PolyFaceTopology& faceTopology = faceTopologies[f];
        // outer ring corner vertex
        for (size_t i = 0; i < faceVertexCount; ++i) {
            int vid = faceTopology.mCornerVertexId[i];
            if (mOuterRingVertexIds[vid] == -1) {
                mOuterRingVertexIds[vid] = mTotalVertexCount++;
            }
        }
        const PolyTessellationFactor& factor = tessellationFactors[f];
        // outer ring edge
        for (size_t i = 0; i < faceVertexCount; ++i) {
            int eid = faceTopology.mEdgeId[i];
            if (mOuterEdgeVertexIdsStart[eid] == -1) {
                int edgeVertexCount = factor.mEdgeFactor[i];
                mOuterEdgeVertexIdsStart[eid] = mTotalVertexCount;
                mOuterEdgeVertexIdsCount[eid] = edgeVertexCount;
                mTotalVertexCount += edgeVertexCount;
            }
        }
        int interiorVertexCount;
        if (faceVertexCount == sQuadVertexCount) {
            int rowVertexCount = getInteriorRowVertexCount(faceTopology);
            int colVertexCount = getInteriorColVertexCount(faceTopology);
            // quad face case
            interiorVertexCount = rowVertexCount * colVertexCount;
            mEstimatedFaceCount += (rowVertexCount + 1) * (colVertexCount + 1);
        } else {
            // triangle face case
            interiorVertexCount = 0;
            int maxEdgeVertexCount = getMaxEdgeVertexCount(faceTopology);
            if (maxEdgeVertexCount != 0) {
                // this boils down to a simple arithmetic series problem
                // 3 * ((n - 1) + (n - 3) + (n - 5) +.....)
                if (maxEdgeVertexCount % 2 == 0) {
                    interiorVertexCount =
                        3 * maxEdgeVertexCount * maxEdgeVertexCount / 4;
                    // 1 for the central triangle
                    // the rest is a (n + 0) * h / 2 formula
                    // where h = (n + 2) / 2
                    mEstimatedFaceCount += 1 +
                        3 * ((maxEdgeVertexCount * maxEdgeVertexCount) / 4 +
                        maxEdgeVertexCount / 2);
                } else {
                    interiorVertexCount = 1 +
                        3 * (maxEdgeVertexCount * maxEdgeVertexCount - 1) / 4;
                    // edge vertex count:
                    // 1 -> 3 * (1)
                    // 3 -> 3 * (1 + 3)
                    // 5 -> 3 * (1 + 3 + 5)
                    // (1 + n) * h / 2
                    // where h = (n + 1) / 2
                    mEstimatedFaceCount +=
                        3 * (maxEdgeVertexCount * maxEdgeVertexCount +
                        2 * maxEdgeVertexCount + 1) / 4;
                }
            } else {
                mEstimatedFaceCount += 1;
            }
        }
        mInteriorVertexIdsStart[f] = mTotalVertexCount;
        mTotalVertexCount += interiorVertexCount;
    }
}

int
PolyTessellatedVertexLookup::getInteriorRowVertexCount(
        const PolyFaceTopology& faceTopology) const
{
    MNRY_ASSERT(mFaceVertexCount == sQuadVertexCount);
    int interiorRowVertexCount = scene_rdl2::math::max(
        mOuterEdgeVertexIdsCount[faceTopology.mEdgeId[1]],
        mOuterEdgeVertexIdsCount[faceTopology.mEdgeId[3]]);
    return scene_rdl2::math::max(1, interiorRowVertexCount);
}

int
PolyTessellatedVertexLookup::getInteriorColVertexCount(
        const PolyFaceTopology& faceTopology) const
{
    MNRY_ASSERT(mFaceVertexCount == sQuadVertexCount);
    int interiorColVertexCount = scene_rdl2::math::max(
        mOuterEdgeVertexIdsCount[faceTopology.mEdgeId[0]],
        mOuterEdgeVertexIdsCount[faceTopology.mEdgeId[2]]);
    return scene_rdl2::math::max(1, interiorColVertexCount);
}

int
PolyTessellatedVertexLookup::getMaxEdgeVertexCount(
        const PolyFaceTopology& faceTopology) const
{
    MNRY_ASSERT(mFaceVertexCount == sTriangleVertexCount);
    return scene_rdl2::math::max(scene_rdl2::math::max(
        getEdgeVertexCount(faceTopology.mEdgeId[0]),
        getEdgeVertexCount(faceTopology.mEdgeId[1])),
        getEdgeVertexCount(faceTopology.mEdgeId[2]));
}

int
PolyTessellatedVertexLookup::getTriangleInteriorVertexId(
        int face, int maxEdgeVertexCount,
        int interiorRingIndex, int cornerIndex, int n) const
{
    MNRY_ASSERT(mFaceVertexCount == sTriangleVertexCount);
    int offset = 0;
    if (interiorRingIndex > 0) {
        offset += 3 * (maxEdgeVertexCount - interiorRingIndex) *
            interiorRingIndex;
    }
    int vertexPerEdge = maxEdgeVertexCount - 2 * interiorRingIndex;
    // otherwise it is a degened single vertex ring
    if (vertexPerEdge > 1) {
        offset += (cornerIndex * (vertexPerEdge - 1) + n) %
            (3 * (vertexPerEdge - 1));
    }
    return mInteriorVertexIdsStart[face] + offset;
}

int
PolyTessellatedVertexLookup::getTriangleRingCount(
        const PolyFaceTopology& faceTopology) const
{
    MNRY_ASSERT(mFaceVertexCount == sTriangleVertexCount);
    // the central vertex in odd maxEdgeVertexCount case
    // is also considered as a ring
    int maxEdgeVertexCount = getMaxEdgeVertexCount(faceTopology);
    return (maxEdgeVertexCount + 3) / 2;
}

bool
PolyTessellatedVertexLookup::noRingToStitch(
        const PolyFaceTopology& faceTopology) const
{
    if (mFaceVertexCount == sQuadVertexCount) {
        int edgeVertexCount0 = getEdgeVertexCount(faceTopology.mEdgeId[0]);
        int edgeVertexCount1 = getEdgeVertexCount(faceTopology.mEdgeId[1]);
        int edgeVertexCount2 = getEdgeVertexCount(faceTopology.mEdgeId[2]);
        int edgeVertexCount3 = getEdgeVertexCount(faceTopology.mEdgeId[3]);
        return edgeVertexCount0 == edgeVertexCount2 &&
            edgeVertexCount1 == edgeVertexCount3;
    } else {
        int edgeVertexCount0 = getEdgeVertexCount(faceTopology.mEdgeId[0]);
        int edgeVertexCount1 = getEdgeVertexCount(faceTopology.mEdgeId[1]);
        int edgeVertexCount2 = getEdgeVertexCount(faceTopology.mEdgeId[2]);
        return edgeVertexCount0 == edgeVertexCount1 &&
            edgeVertexCount0 == edgeVertexCount2;
    }
}

SubdTessellatedVertexLookup::SubdTessellatedVertexLookup(
        const std::vector<SubdQuadTopology>& quadTopologies,
        const SubdTopologyIdLookup& topologyIdLookup,
        const std::vector<SubdTessellationFactor>& tessellationFactors,
        bool noTessellation):
    TessellatedVertexLookup(), mMaxEdgeResolution(1)
{
    if (noTessellation) {
        int coarseVertexCount = topologyIdLookup.getCoarseVertexCount();
        mOuterRingVertexIds.resize(coarseVertexCount, -1);
    } else {
        int splitVertexCount = topologyIdLookup.getSplitVertexCount();
        int splitEdgeCount = topologyIdLookup.getSplitEdgeCount();
        // cache the result when loop through every quadTopology's
        // vertices, edges and faces so shared vertices only got counted once
        mOuterRingVertexIds.resize(splitVertexCount, -1);
        mOuterEdgeVertexIdsStart.resize(splitEdgeCount, -1);
        mOuterEdgeVertexIdsCount.resize(splitEdgeCount, -1);
        mInteriorVertexIdsStart.resize(quadTopologies.size(), -1);
    }

    for (size_t f = 0; f < quadTopologies.size(); ++f) {
        const SubdQuadTopology& quadTopology = quadTopologies[f];
        // unassigned control face won't contribute vertices in final
        // tessellated vertex buffer
        if (!quadTopology.mHasAssignment) {
            continue;
        }
        // outer ring corner vertex
        for (size_t i = 0; i < sQuadVertexCount; ++i) {
            int vid = quadTopology.mCornerVertexId[i];
            if (mOuterRingVertexIds[vid] == -1) {
                mOuterRingVertexIds[vid] = mTotalVertexCount++;
            }
        }
        // when rendering control cage only, we don't need to handle
        // following edge tessellation and interior vertices
        if (noTessellation) {
            continue;
        }
        const SubdTessellationFactor& factor = tessellationFactors[f];
        if (quadTopology.nonQuadParent()) {
            // outer ring edge
            for (size_t i = 0; i < sQuadVertexCount; ++i) {
                int eid = quadTopology.mEdgeId0[i];
                if (mOuterEdgeVertexIdsStart[eid] == -1) {
                    int edgeVertexCount = factor.mEdge0Factor[i];
                    mOuterEdgeVertexIdsStart[eid] = mTotalVertexCount;
                    mOuterEdgeVertexIdsCount[eid] = edgeVertexCount;
                    mTotalVertexCount += edgeVertexCount;
                    int controlEdgeResolution = 2 * (edgeVertexCount + 1);
                    if (controlEdgeResolution > mMaxEdgeResolution) {
                        mMaxEdgeResolution = controlEdgeResolution;
                    }
                }
            }
        } else {
            // outer ring mid edge vertex
            for (size_t i = 0; i < sQuadVertexCount; ++i) {
                int vid = quadTopology.mMidEdgeVertexId[i];
                if (mOuterRingVertexIds[vid] == -1) {
                    mOuterRingVertexIds[vid] = mTotalVertexCount++;
                }
            }
            // outer ring edge
            for (size_t i = 0; i < sQuadVertexCount; ++i) {
                int eid0 = quadTopology.mEdgeId0[i];
                if (mOuterEdgeVertexIdsStart[eid0] == -1) {
                    int edgeVertexCount = factor.mEdge0Factor[i];
                    mOuterEdgeVertexIdsStart[eid0] = mTotalVertexCount;
                    mOuterEdgeVertexIdsCount[eid0] = edgeVertexCount;
                    mTotalVertexCount += edgeVertexCount;
                }
                int eid1 = quadTopology.mEdgeId1[i];
                if (mOuterEdgeVertexIdsStart[eid1] == -1) {
                    int edgeVertexCount = factor.mEdge1Factor[i];
                    mOuterEdgeVertexIdsStart[eid1] = mTotalVertexCount;
                    mOuterEdgeVertexIdsCount[eid1] = edgeVertexCount;
                    mTotalVertexCount += edgeVertexCount;
                }
                int controlEdgeResolution = getEdgeVertexCount(eid0) +
                    getEdgeVertexCount(eid1) + 2;
                if (controlEdgeResolution > mMaxEdgeResolution) {
                    mMaxEdgeResolution = controlEdgeResolution;
                }
            }
        }
        int interiorVertexCount = getInteriorRowVertexCount(quadTopology) *
            getInteriorColVertexCount(quadTopology);
        mInteriorVertexIdsStart[f] = mTotalVertexCount;
        mTotalVertexCount += interiorVertexCount;
    }
}

int
SubdTessellatedVertexLookup::getInteriorRowVertexCount(
        const SubdQuadTopology& quadTopology) const
{
    int interiorRowVertexCount;
    if (quadTopology.nonQuadParent()) {
        interiorRowVertexCount = scene_rdl2::math::max(
            mOuterEdgeVertexIdsCount[quadTopology.mEdgeId0[1]],
            mOuterEdgeVertexIdsCount[quadTopology.mEdgeId0[3]]);
    } else {
        interiorRowVertexCount = 1 + scene_rdl2::math::max(
            (mOuterEdgeVertexIdsCount[quadTopology.mEdgeId0[1]] +
            mOuterEdgeVertexIdsCount[quadTopology.mEdgeId1[1]]),
            (mOuterEdgeVertexIdsCount[quadTopology.mEdgeId0[3]] +
            mOuterEdgeVertexIdsCount[quadTopology.mEdgeId1[3]]));
    }
    return scene_rdl2::math::max(1, interiorRowVertexCount);
}

int
SubdTessellatedVertexLookup::getInteriorColVertexCount(
        const SubdQuadTopology& quadTopology) const
{
    int interiorColVertexCount;
    if (quadTopology.nonQuadParent()) {
        interiorColVertexCount = scene_rdl2::math::max(
            mOuterEdgeVertexIdsCount[quadTopology.mEdgeId0[0]],
            mOuterEdgeVertexIdsCount[quadTopology.mEdgeId0[2]]);
    } else {
        interiorColVertexCount = 1 + scene_rdl2::math::max(
            (mOuterEdgeVertexIdsCount[quadTopology.mEdgeId0[0]] +
            mOuterEdgeVertexIdsCount[quadTopology.mEdgeId1[0]]),
            (mOuterEdgeVertexIdsCount[quadTopology.mEdgeId0[2]] +
            mOuterEdgeVertexIdsCount[quadTopology.mEdgeId1[2]]));
    }
    return scene_rdl2::math::max(1, interiorColVertexCount);
}

bool
SubdTessellatedVertexLookup::noRingToStitch(
        const SubdQuadTopology& quadTopology) const
{
    if (quadTopology.nonQuadParent()) {
        return mOuterEdgeVertexIdsCount[quadTopology.mEdgeId0[0]] == 0 &&
            mOuterEdgeVertexIdsCount[quadTopology.mEdgeId0[1]] == 0 &&
            mOuterEdgeVertexIdsCount[quadTopology.mEdgeId0[2]] == 0 &&
            mOuterEdgeVertexIdsCount[quadTopology.mEdgeId0[3]] == 0;

    } else {
        return mOuterEdgeVertexIdsCount[quadTopology.mEdgeId0[0]] == 0 &&
            mOuterEdgeVertexIdsCount[quadTopology.mEdgeId0[1]] == 0 &&
            mOuterEdgeVertexIdsCount[quadTopology.mEdgeId0[2]] == 0 &&
            mOuterEdgeVertexIdsCount[quadTopology.mEdgeId0[3]] == 0 &&
            mOuterEdgeVertexIdsCount[quadTopology.mEdgeId1[0]] == 0 &&
            mOuterEdgeVertexIdsCount[quadTopology.mEdgeId1[1]] == 0 &&
            mOuterEdgeVertexIdsCount[quadTopology.mEdgeId1[2]] == 0 &&
            mOuterEdgeVertexIdsCount[quadTopology.mEdgeId1[3]] == 0;
    }
}

void
stitchRings(const StitchPoint* innerRing, const Vec2f* innerRingUv,
        int innerRingVertexCount,
        const StitchPoint* outerRing, const Vec2f* outerRingUv,
        int outerRingVertexCount,
        std::vector<geom::Primitive::IndexType>& indices,
        const int controlFaceId, std::vector<int>* tessellatedToControlFace,
        std::vector<Vec2f>* faceVaryingUv)
{
    int innerCounter = 0;
    int outerCounter = 0;
    int v0 = innerRing[innerCounter].mVertexId;
    int v1 = outerRing[outerCounter].mVertexId;
    int v2;
    Vec2f v0Uv, v1Uv, v2Uv;
    if (faceVaryingUv && innerRingUv && outerRingUv) {
        v0Uv = innerRingUv[innerCounter];
        v1Uv = outerRingUv[outerCounter];
    }

    while (innerCounter < innerRingVertexCount - 1 &&
        outerCounter < outerRingVertexCount - 1) {
        const StitchPoint& s0 = innerRing[innerCounter];
        const StitchPoint& s1 = outerRing[outerCounter];
        // heuristic threshold that we can move both stitch points
        // forward to form a valid quad instead of a degened quad (triangle)
        float threshold = 0.5f * scene_rdl2::math::min(s0.mDeltaT, s1.mDeltaT);
        if (scene_rdl2::math::abs(s0.mT - s1.mT) < threshold &&
            scene_rdl2::math::abs(s0.mT + s0.mDeltaT - s1.mT - s1.mDeltaT) < threshold) {
            innerCounter++;
            outerCounter++;
            v2 = outerRing[outerCounter].mVertexId;
            int v3 = innerRing[innerCounter].mVertexId;
            indices.push_back(v0);
            indices.push_back(v1);
            indices.push_back(v2);
            indices.push_back(v3);
            if (tessellatedToControlFace) {
                tessellatedToControlFace->push_back(controlFaceId);
            }
            v0 = v3;
            v1 = v2;
            if (faceVaryingUv && innerRingUv && outerRingUv) {
                v2Uv = outerRingUv[outerCounter];
                Vec2f v3Uv = innerRingUv[innerCounter];
                faceVaryingUv->push_back(v0Uv);
                faceVaryingUv->push_back(v1Uv);
                faceVaryingUv->push_back(v2Uv);
                faceVaryingUv->push_back(v3Uv);
                v0Uv = v3Uv;
                v1Uv = v2Uv;
            }
        } else if ((s1.mT + s1.mDeltaT) - s0.mT >= (s0.mT + s0.mDeltaT) - s1.mT) {
            innerCounter++;
            v2 = innerRing[innerCounter].mVertexId;
            indices.push_back(v0);
            indices.push_back(v1);
            indices.push_back(v2);
            indices.push_back(v2);
            if (tessellatedToControlFace) {
                tessellatedToControlFace->push_back(controlFaceId);
            }
            v0 = v2;
            if (faceVaryingUv && innerRingUv && outerRingUv) {
                v2Uv = innerRingUv[innerCounter];
                faceVaryingUv->push_back(v0Uv);
                faceVaryingUv->push_back(v1Uv);
                faceVaryingUv->push_back(v2Uv);
                faceVaryingUv->push_back(v2Uv);
                v0Uv = v2Uv;
            }
        } else {
            outerCounter++;
            v2 = outerRing[outerCounter].mVertexId;
            indices.push_back(v0);
            indices.push_back(v1);
            indices.push_back(v2);
            indices.push_back(v2);
            if (tessellatedToControlFace) {
                tessellatedToControlFace->push_back(controlFaceId);
            }
            v1 = v2;
            if (faceVaryingUv && innerRingUv && outerRingUv) {
                v2Uv = outerRingUv[outerCounter];
                faceVaryingUv->push_back(v0Uv);
                faceVaryingUv->push_back(v1Uv);
                faceVaryingUv->push_back(v2Uv);
                faceVaryingUv->push_back(v2Uv);
                v1Uv = v2Uv;
            }
        }
    }

    if (innerCounter == innerRingVertexCount - 1) {
        outerCounter++;
        while (outerCounter < outerRingVertexCount) {
            v2 = outerRing[outerCounter].mVertexId;
            indices.push_back(v0);
            indices.push_back(v1);
            indices.push_back(v2);
            indices.push_back(v2);
            if (tessellatedToControlFace) {
                tessellatedToControlFace->push_back(controlFaceId);
            }
            v1 = v2;
            if (faceVaryingUv && innerRingUv && outerRingUv) {
                v2Uv = outerRingUv[outerCounter];
                faceVaryingUv->push_back(v0Uv);
                faceVaryingUv->push_back(v1Uv);
                faceVaryingUv->push_back(v2Uv);
                faceVaryingUv->push_back(v2Uv);
                v1Uv = v2Uv;
            }
            outerCounter++;
        }
    }
    if (outerCounter == outerRingVertexCount - 1) {
        innerCounter++;
        while (innerCounter < innerRingVertexCount) {
            v2 = innerRing[innerCounter].mVertexId;
            indices.push_back(v0);
            indices.push_back(v1);
            indices.push_back(v2);
            indices.push_back(v2);
            if (tessellatedToControlFace) {
                tessellatedToControlFace->push_back(controlFaceId);
            }
            v0 = v2;
            if (faceVaryingUv && innerRingUv && outerRingUv) {
                v2Uv = innerRingUv[innerCounter];
                faceVaryingUv->push_back(v0Uv);
                faceVaryingUv->push_back(v1Uv);
                faceVaryingUv->push_back(v2Uv);
                faceVaryingUv->push_back(v2Uv);
                v0Uv = v2Uv;
            }
            innerCounter++;
        }
    }
}

void
stitchRings(const StitchPoint* innerRing, int innerRingVertexCount,
        const StitchPoint* outerRing, int outerRingVertexCount,
        std::vector<geom::Primitive::IndexType>& indices,
        const int controlFaceId, std::vector<int>* tessellatedToControlFace)
{
    stitchRings(innerRing, nullptr, innerRingVertexCount,
        outerRing, nullptr, outerRingVertexCount,
        indices, controlFaceId, tessellatedToControlFace, nullptr);
}

void
generateSubdStitchRings(scene_rdl2::alloc::Arena *arena,
        const SubdQuadTopology& quadTopology, int fid, size_t cornerIndex,
        const SubdTessellatedVertexLookup& tessellatedVertexLookup,
        int interiorRowVertexCount, int interiorColVertexCount,
        StitchPoint* & innerRing, int& innerRingVertexCount,
        StitchPoint* & outerRing, int& outerRingVertexCount)
{
    MNRY_ASSERT(cornerIndex < sQuadVertexCount);
    float du = 1.0f / (float)(1 + interiorColVertexCount);
    float dv = 1.0f / (float)(1 + interiorRowVertexCount);
    // allocate and initialize inner ring
    if (cornerIndex == 0 || cornerIndex == 2) {
        innerRingVertexCount = interiorColVertexCount;
    } else {
        innerRingVertexCount = interiorRowVertexCount;
    }
    innerRing = arena->allocArray<StitchPoint>(innerRingVertexCount);
    if (cornerIndex == 0) {
        for (int i = 0; i < innerRingVertexCount; ++i) {
            int row = 0;
            int col = i;
            innerRing[i].mVertexId =
                tessellatedVertexLookup.getInteriorVertexId(
                fid, row, col, interiorColVertexCount);
            innerRing[i].mT = (i + 1) * du;
            innerRing[i].mDeltaT = du;
        }
    } else if (cornerIndex == 1) {
        for (int i = 0; i < innerRingVertexCount; ++i) {
            int row = i;
            int col = interiorColVertexCount - 1;
            innerRing[i].mVertexId =
                tessellatedVertexLookup.getInteriorVertexId(
                fid, row, col, interiorColVertexCount);
            innerRing[i].mT = (i + 1) * dv;
            innerRing[i].mDeltaT = dv;
        }
    } else if (cornerIndex == 2) {
        for (int i = 0; i < innerRingVertexCount; ++i) {
            int row = interiorRowVertexCount - 1;
            int col = interiorColVertexCount - 1 - i;
            innerRing[i].mVertexId =
                tessellatedVertexLookup.getInteriorVertexId(
                fid, row, col, interiorColVertexCount);
            innerRing[i].mT = (i + 1) * du;
            innerRing[i].mDeltaT = du;
        }
    } else if (cornerIndex == 3) {
        for (int i = 0; i < innerRingVertexCount; ++i) {
            int row = interiorRowVertexCount - 1 - i;
            int col = 0;
            innerRing[i].mVertexId =
                tessellatedVertexLookup.getInteriorVertexId(
                fid, row, col, interiorColVertexCount);
            innerRing[i].mT = (i + 1) * dv;
            innerRing[i].mDeltaT = dv;
        }
    }
    // allocate and initialize outer ring
    if (quadTopology.nonQuadParent()) {
        // the outer ring edge looks like this:
        //        x---*---*---*---x
        // x: corner vertex
        // *: edge vertex
        int edgeId = quadTopology.mEdgeId0[cornerIndex];
        int edgeVertexCount =
            tessellatedVertexLookup.getEdgeVertexCount(edgeId);
        // 2 stands for 2 corner vertices
        outerRingVertexCount = edgeVertexCount + 2;
        outerRing = arena->allocArray<StitchPoint>(outerRingVertexCount);
        float deltaT = 1.0f / (edgeVertexCount + 1);
        int vidStart = tessellatedVertexLookup.getTessellatedVertexId(
            quadTopology.mCornerVertexId[cornerIndex]);
        int vidEnd = tessellatedVertexLookup.getTessellatedVertexId(
            quadTopology.mCornerVertexId[(cornerIndex + 1) % sQuadVertexCount]);
        outerRing[0].mVertexId = vidStart;
        outerRing[0].mT = 0.0f;
        outerRing[0].mDeltaT = deltaT;
        // |\              for quadrangulated quads A, B, C from control face
        // |  \            they all follow the rule that
        // |C   \          edge0 end at control edge middle
        // |___ / \        edge1 end at qudrangulated center
        // |   |    \      edge2 end at control edge middle
        // | A |  B   \    edge3 end at control vertices
        // ------------
        bool endAtEdgeMiddle = (cornerIndex % 2) == 0;
        for (int i = 0; i < edgeVertexCount; ++i) {
            outerRing[i + 1].mVertexId =
                tessellatedVertexLookup.getEdgeVertexId(edgeId, i,
                endAtEdgeMiddle);
            outerRing[i + 1].mT = (i + 1) * deltaT;
            outerRing[i + 1].mDeltaT = deltaT;
        }
        outerRing[edgeVertexCount + 1].mVertexId = vidEnd;
        outerRing[edgeVertexCount + 1].mT = 1.0f;
        outerRing[edgeVertexCount + 1].mDeltaT = 0.0f;
    } else {
        int edgeId0 = quadTopology.mEdgeId0[cornerIndex];
        int edgeId1 = quadTopology.mEdgeId1[cornerIndex];
        int edgeVertexCount0 =
            tessellatedVertexLookup.getEdgeVertexCount(edgeId0);
        int edgeVertexCount1 =
            tessellatedVertexLookup.getEdgeVertexCount(edgeId1);
        // 3 = 2 corner vertices + 1 mid edge vertex
        outerRingVertexCount = edgeVertexCount0 + edgeVertexCount1 + 3;
        outerRing = arena->allocArray<StitchPoint>(outerRingVertexCount);
        float deltaT0 = 0.5f / (edgeVertexCount0 + 1);
        float deltaT1 = 0.5f / (edgeVertexCount1 + 1);
        int vidStart = tessellatedVertexLookup.getTessellatedVertexId(
            quadTopology.mCornerVertexId[cornerIndex]);
        int vidMid = tessellatedVertexLookup.getTessellatedVertexId(
            quadTopology.mMidEdgeVertexId[cornerIndex]);
        int vidEnd = tessellatedVertexLookup.getTessellatedVertexId(
            quadTopology.mCornerVertexId[(cornerIndex + 1) % sQuadVertexCount]);
        int offset = 0;
        outerRing[offset].mVertexId = vidStart;
        outerRing[offset].mT = 0.0f;
        outerRing[offset].mDeltaT = deltaT0;
        offset++;
        for (int i = 0; i < edgeVertexCount0; ++i) {
            // edge vidStart->vidMid ends at middle of the control edge
            outerRing[offset + i].mVertexId =
                tessellatedVertexLookup.getEdgeVertexId(
                edgeId0, i, true);
            outerRing[offset + i].mT = (i + 1) * deltaT0;
            outerRing[offset + i].mDeltaT = deltaT0;
        }
        offset += edgeVertexCount0;
        outerRing[offset].mVertexId = vidMid;
        outerRing[offset].mT = 0.5f;
        outerRing[offset].mDeltaT = deltaT1;
        offset++;
        for (int i = 0; i < edgeVertexCount1; ++i) {
            // edge vidMid->vidEnd ends at control vertices
            outerRing[offset + i].mVertexId =
                tessellatedVertexLookup.getEdgeVertexId(
                edgeId1, i, false);
            outerRing[offset + i].mT = 0.5f + (i + 1) * deltaT1;
            outerRing[offset + i].mDeltaT = deltaT1;
        }
        offset += edgeVertexCount1;
        outerRing[offset].mVertexId = vidEnd;
        outerRing[offset].mT = 1.0f;
        outerRing[offset].mDeltaT = 0.0f;
    }
}

void
generateQuadStitchRings(scene_rdl2::alloc::Arena *arena,
        const PolyFaceTopology& quadTopology, int fid, size_t cornerIndex,
        const PolyTessellatedVertexLookup& tessellatedVertexLookup,
        int interiorRowVertexCount, int interiorColVertexCount,
        StitchPoint* & innerRing, Vec2f* & innerRingUv,
        int& innerRingVertexCount,
        StitchPoint* & outerRing, Vec2f* & outerRingUv,
        int& outerRingVertexCount)
{
    MNRY_ASSERT(cornerIndex < sQuadVertexCount);
    float du = 1.0f / (float)(1 + interiorColVertexCount);
    float dv = 1.0f / (float)(1 + interiorRowVertexCount);
    // allocate and initialize inner ring
    if (cornerIndex == 0 || cornerIndex == 2) {
        innerRingVertexCount = interiorColVertexCount;
    } else {
        innerRingVertexCount = interiorRowVertexCount;
    }
    innerRing = arena->allocArray<StitchPoint>(innerRingVertexCount);
    innerRingUv = arena->allocArray<Vec2f>(innerRingVertexCount);
    if (cornerIndex == 0) {
        for (int i = 0; i < innerRingVertexCount; ++i) {
            int row = 0;
            int col = i;
            innerRing[i].mVertexId =
                tessellatedVertexLookup.getInteriorVertexId(
                fid, row, col, interiorColVertexCount);
            innerRing[i].mT = (i + 1) * du;
            innerRing[i].mDeltaT = du;
            innerRingUv[i] = Vec2f((i + 1) * du, dv);
        }
    } else if (cornerIndex == 1) {
        for (int i = 0; i < innerRingVertexCount; ++i) {
            int row = i;
            int col = interiorColVertexCount - 1;
            innerRing[i].mVertexId =
                tessellatedVertexLookup.getInteriorVertexId(
                fid, row, col, interiorColVertexCount);
            innerRing[i].mT = (i + 1) * dv;
            innerRing[i].mDeltaT = dv;
            innerRingUv[i] = Vec2f(1.0f - du, (i + 1) * dv);
        }
    } else if (cornerIndex == 2) {
        for (int i = 0; i < innerRingVertexCount; ++i) {
            int row = interiorRowVertexCount - 1;
            int col = interiorColVertexCount - 1 - i;
            innerRing[i].mVertexId =
                tessellatedVertexLookup.getInteriorVertexId(
                fid, row, col, interiorColVertexCount);
            innerRing[i].mT = (i + 1) * du;
            innerRing[i].mDeltaT = du;
            innerRingUv[i] = Vec2f(1.0f - (i + 1) * du, 1.0f - dv);
        }
    } else if (cornerIndex == 3) {
        for (int i = 0; i < innerRingVertexCount; ++i) {
            int row = interiorRowVertexCount - 1 - i;
            int col = 0;
            innerRing[i].mVertexId =
                tessellatedVertexLookup.getInteriorVertexId(
                fid, row, col, interiorColVertexCount);
            innerRing[i].mT = (i + 1) * dv;
            innerRing[i].mDeltaT = dv;
            innerRingUv[i] = Vec2f(du, 1.0f - (i + 1) * dv);
        }
    }
    // allocate and initialize outer ring
    // the outer ring edge looks like this:
    //        x---*---*---*---x
    // x: corner vertex
    // *: edge vertex
    int edgeId = quadTopology.mEdgeId[cornerIndex];
    int edgeVertexCount =
        tessellatedVertexLookup.getEdgeVertexCount(edgeId);
    // 2 stands for 2 corner vertices
    outerRingVertexCount = edgeVertexCount + 2;
    outerRing = arena->allocArray<StitchPoint>(outerRingVertexCount);
    outerRingUv = arena->allocArray<Vec2f>(outerRingVertexCount);
    float deltaT = 1.0f / (edgeVertexCount + 1);
    int vidStart = tessellatedVertexLookup.getTessellatedVertexId(
        quadTopology.mCornerVertexId[cornerIndex]);
    int vidEnd = tessellatedVertexLookup.getTessellatedVertexId(
        quadTopology.mCornerVertexId[(cornerIndex + 1) % sQuadVertexCount]);

    bool reverseEdge = quadTopology.mCornerVertexId[cornerIndex] >
        quadTopology.mCornerVertexId[(cornerIndex + 1) % sQuadVertexCount];

    int offset = 0;
    outerRing[offset].mVertexId = vidStart;
    outerRing[offset].mT = 0.0f;
    outerRing[offset].mDeltaT = deltaT;
    offset++;
    for (int i = 0; i < edgeVertexCount; ++i) {
        outerRing[offset + i].mVertexId =
            tessellatedVertexLookup.getEdgeVertexId(edgeId, i, reverseEdge);
        outerRing[offset + i].mT = (i + 1) * deltaT;
        outerRing[offset + i].mDeltaT = deltaT;
    }
    offset += edgeVertexCount;
    outerRing[offset].mVertexId = vidEnd;
    outerRing[offset].mT = 1.0f;
    outerRing[offset].mDeltaT = 0.0f;

    if (cornerIndex == 0) {
        for (int i = 0; i < outerRingVertexCount; ++i) {
            outerRingUv[i] = Vec2f(i * deltaT, 0.0f);
        }
    } else if (cornerIndex == 1) {
        for (int i = 0; i < outerRingVertexCount; ++i) {
            outerRingUv[i] = Vec2f(1.0f, i * deltaT);
        }
    } else if (cornerIndex == 2) {
        for (int i = 0; i < outerRingVertexCount; ++i) {
            outerRingUv[i] = Vec2f(1.0f - i * deltaT, 1.0f);
        }
    } else if (cornerIndex == 3) {
        for (int i = 0; i < outerRingVertexCount; ++i) {
            outerRingUv[i] = Vec2f(0.0f, 1.0f - i * deltaT);
        }
    }
}

void
generateTriangleStitchRings(scene_rdl2::alloc::Arena* arena,
        const PolyFaceTopology& faceTopology, int fid, size_t cornerIndex,
        const PolyTessellatedVertexLookup& tessellatedVertexLookup,
        int maxEdgeVertexCount, float ringUvDelta,
        StitchPoint* & innerRing, Vec2f* & innerRingUv,
        int& innerRingVertexCount,
        StitchPoint* & outerRing, Vec2f* & outerRingUv,
        int& outerRingVertexCount)
{
    MNRY_ASSERT(cornerIndex < sTriangleVertexCount);
    int index0 = cornerIndex;
    int index1 = (cornerIndex + 1) % sTriangleVertexCount;
    float innerDeltaT = 1.0f / (float)(1 + maxEdgeVertexCount);
    // use the same barycentric coordinate encoding embree does
    // (the weight of v1 and v2, so three corner vertices v0, v1, v2
    // with barycentric coordinates (1, 0, 0), (0, 1, 0), (0, 0, 1) are
    // encoded as (0, 0), (1, 0), (0, 1)
    const Vec2f cornerUv0[3] = {
        Vec2f(0.0f, 0.0f), Vec2f(1.0f, 0.0f), Vec2f(0.0f, 1.0f)};
    const Vec2f cornerUv1[3] = {
        Vec2f(ringUvDelta, ringUvDelta),
        Vec2f(1.0f - 2 * ringUvDelta, ringUvDelta),
        Vec2f(ringUvDelta, 1.0f - 2 * ringUvDelta)};
    // allocate and initialize inner ring
    innerRingVertexCount = maxEdgeVertexCount;
    float wInner = innerRingVertexCount == 1 ?
        0.0f: 1.0f / (float)(innerRingVertexCount - 1);
    innerRing = arena->allocArray<StitchPoint>(innerRingVertexCount);
    innerRingUv = arena->allocArray<Vec2f>(innerRingVertexCount);
    for (int i = 0; i < innerRingVertexCount; ++i) {
        innerRing[i].mVertexId =
            tessellatedVertexLookup.getTriangleInteriorVertexId(
            fid, maxEdgeVertexCount, 0, cornerIndex, i);
        innerRing[i].mT = (i + 1) * innerDeltaT;
        innerRing[i].mDeltaT = innerDeltaT;
        innerRingUv[i] = Vec2f(
            (1.0f - i * wInner) * cornerUv1[index0] +
            (       i * wInner) * cornerUv1[index1]);
    }
    // allocate and initialize outer ring
    // the outer ring edge looks like this:
    //        x---*---*---*---x
    // x: corner vertex
    // *: edge vertex
    int edgeId = faceTopology.mEdgeId[cornerIndex];
    int edgeVertexCount =
        tessellatedVertexLookup.getEdgeVertexCount(edgeId);
    // 2 = 2 corner vertices
    outerRingVertexCount = edgeVertexCount + 2;
    float wOuter = 1.0f / (float)(outerRingVertexCount - 1);
    outerRing = arena->allocArray<StitchPoint>(outerRingVertexCount);
    outerRingUv = arena->allocArray<Vec2f>(outerRingVertexCount);
    float outerDeltaT = 1.0f / (edgeVertexCount + 1);
    int vidStart = tessellatedVertexLookup.getTessellatedVertexId(
        faceTopology.mCornerVertexId[index0]);
    int vidEnd = tessellatedVertexLookup.getTessellatedVertexId(
        faceTopology.mCornerVertexId[index1]);
    bool reverseEdge = faceTopology.mCornerVertexId[index0] >
        faceTopology.mCornerVertexId[index1];

    int offset = 0;
    outerRing[offset].mVertexId = vidStart;
    outerRing[offset].mT = 0.0f;
    outerRing[offset].mDeltaT = outerDeltaT;
    outerRingUv[offset] = cornerUv0[index0];
    offset++;
    for (int i = 0; i < edgeVertexCount; ++i) {
        outerRing[offset + i].mVertexId =
            tessellatedVertexLookup.getEdgeVertexId(edgeId, i, reverseEdge);
        outerRing[offset + i].mT = (i + 1) * outerDeltaT;
        outerRing[offset + i].mDeltaT = outerDeltaT;
        outerRingUv[offset + i] =
            (1.0f - (i + 1) * wOuter) * cornerUv0[index0] +
            (       (i + 1) * wOuter) * cornerUv0[index1];
    }
    offset += edgeVertexCount;
    outerRing[offset].mVertexId = vidEnd;
    outerRing[offset].mT = 1.0f;
    outerRing[offset].mDeltaT = 0.0f;
    outerRingUv[offset] = cornerUv0[index1];
}

void
collectUniformTessellatedVertices(scene_rdl2::alloc::Arena *arena,
        const PolyFaceTopology& quadTopology, int baseFaceId,
        const PolyTessellatedVertexLookup& tessellatedVertexLookup,
        int* & vids, int& rowVertexCount, int& colVertexCount)
{
    MNRY_ASSERT_REQUIRE(tessellatedVertexLookup.noRingToStitch(quadTopology));
    bool reverseEdge0 =
        quadTopology.mCornerVertexId[0] > quadTopology.mCornerVertexId[1];
    bool reverseEdge1 =
        quadTopology.mCornerVertexId[1] > quadTopology.mCornerVertexId[2];
    bool reverseEdge2 =
        quadTopology.mCornerVertexId[2] > quadTopology.mCornerVertexId[3];
    bool reverseEdge3 =
        quadTopology.mCornerVertexId[3] > quadTopology.mCornerVertexId[0];
    int eid0 = quadTopology.mEdgeId[0];
    int eid1 = quadTopology.mEdgeId[1];
    int eid2 = quadTopology.mEdgeId[2];
    int eid3 = quadTopology.mEdgeId[3];
    colVertexCount = tessellatedVertexLookup.getEdgeVertexCount(eid0) + 2;
    rowVertexCount = tessellatedVertexLookup.getEdgeVertexCount(eid1) + 2;
    vids = arena->allocArray<int>(rowVertexCount * colVertexCount);
    int offset = 0;
    // bottom row
    vids[offset++] = tessellatedVertexLookup.getTessellatedVertexId(
        quadTopology.mCornerVertexId[0]);
    for (int i = 1; i < colVertexCount - 1; ++i) {
        vids[offset++] = tessellatedVertexLookup.getEdgeVertexId(
            eid0, i - 1, reverseEdge0);
    }
    vids[offset++] = tessellatedVertexLookup.getTessellatedVertexId(
        quadTopology.mCornerVertexId[1]);
    // middle
    for (int i = 1; i < rowVertexCount - 1; ++i) {
        vids[offset++] = tessellatedVertexLookup.getEdgeVertexId(
            eid3, i - 1, !reverseEdge3);
        for (int j = 1; j < colVertexCount - 1; ++j) {
            vids[offset++] = tessellatedVertexLookup.getInteriorVertexId(
                baseFaceId, i - 1, j - 1, colVertexCount - 2);
        }
        vids[offset++] = tessellatedVertexLookup.getEdgeVertexId(
            eid1, i - 1, reverseEdge1);
    }
    // top row
    vids[offset++] = tessellatedVertexLookup.getTessellatedVertexId(
        quadTopology.mCornerVertexId[3]);
    for (int i = 1; i < colVertexCount - 1; ++i) {
        vids[offset++] = tessellatedVertexLookup.getEdgeVertexId(
            eid2, i - 1, !reverseEdge2);
    }
    vids[offset++] = tessellatedVertexLookup.getTessellatedVertexId(
        quadTopology.mCornerVertexId[2]);
}

} // namespace internal
} // namespace geom
} // namespace rendering


