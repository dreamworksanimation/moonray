// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file TriMesh.h
/// $Id$
///

#include "TriMesh.h"

#include <moonray/rendering/geom/prim/MeshTessellationUtil.h>

#include <moonray/rendering/bvh/shading/AttributeKey.h>
#include <moonray/rendering/bvh/shading/Attributes.h>
#include <moonray/rendering/bvh/shading/PrimitiveAttribute.h>
#include <moonray/rendering/bvh/shading/RootShader.h>
#include <moonray/rendering/mcrt_common/ThreadLocalState.h>
#include <scene_rdl2/common/math/Mat3.h>
#include <scene_rdl2/common/math/Math.h>
#include <scene_rdl2/common/math/MathUtil.h>
#include <scene_rdl2/common/math/ReferenceFrame.h>
#include <scene_rdl2/render/logging/logging.h>

namespace moonray {
namespace geom {
namespace internal {

using namespace shading;
using namespace scene_rdl2::math;

TriMesh::TriMesh(size_t estiFaceCount,
        PolygonMesh::FaceVertexCount&& faceVertexCount,
        PolygonMesh::IndexBuffer&& indices,
        PolygonMesh::VertexBuffer&& vertices,
        LayerAssignmentId&& layerAssignmentId,
        PrimitiveAttributeTable&& primitiveAttributeTable):
        PolyMesh(estiFaceCount, std::move(faceVertexCount),
            std::move(indices), std::move(vertices),
            std::move(layerAssignmentId),
            std::move(primitiveAttributeTable))
{
}

bool
TriMesh::bakePosMap(int width, int height, int udim,
        TypedAttributeKey<Vec2f> stKey,
        Vec3fa *posResult, Vec3f *nrmResult) const
{
    const Attributes *primitiveAttributes = getAttributes();
    MNRY_ASSERT_REQUIRE(primitiveAttributes->isSupported(stKey));
    // throw out the anchor boys....
    if (mIsTessellated) {
        // triangle got tessellated to quads
        for (size_t faceId = 0; faceId < getTessellatedMeshFaceCount(); ++faceId) {
            int baseFaceId = getBaseFaceId(faceId);
            const uint32_t id0 = mIndices[faceId * sQuadVertexCount];
            const uint32_t id1 = mIndices[faceId * sQuadVertexCount + 1];
            const uint32_t id2 = mIndices[faceId * sQuadVertexCount + 2];
            const uint32_t id3 = mIndices[faceId * sQuadVertexCount + 3];

            const Vec3f pos0 = mVertices(id0).asVec3f();
            const Vec3f pos1 = mVertices(id1).asVec3f();
            const Vec3f pos2 = mVertices(id2).asVec3f();
            const Vec3f pos3 = mVertices(id3).asVec3f();

            // FIXME: we are forced to fetch st1 and st3 twice.
            Vec2f st0, st1, st2, st3;
            getTriangleAttributes(stKey, baseFaceId, faceId, st0, st1, st3,
                0.0f, true);
            getTriangleAttributes(stKey, baseFaceId, faceId, st2, st3, st1,
                0.0f, false);

            // check if this face exists in this udim and if so
            // transform the st coordinates into the normalized range
            if (udimxform(udim, st0) && udimxform(udim, st1) &&
                udimxform(udim, st2) && udimxform(udim, st3)) {
                const Vec3f *nrm0 = nullptr;
                const Vec3f *nrm1 = nullptr;
                const Vec3f *nrm2 = nullptr;
                const Vec3f *nrm3 = nullptr;
                Vec3f nrmData0, nrmData1, nrmData2, nrmData3;
                if (nrmResult) {
                    getTriangleNormal(baseFaceId, faceId,
                        nrmData0, nrmData1, nrmData3,
                        0.0f, /* isFirst = */ true);
                    getTriangleNormal(baseFaceId, faceId,
                        nrmData2, nrmData3, nrmData1,
                        0.0f, /* isFirst = */ false);
                    nrm0 = &nrmData0;
                    nrm1 = &nrmData1;
                    nrm2 = &nrmData2;
                    nrm3 = &nrmData3;
                }
                // first triangle: <0, 1, 3>
                scene_rdl2::math::BBox2f roiST(
                    Vec2f(std::min(st0.x, std::min(st1.x, st3.x)),
                          std::min(st0.y, std::min(st1.y, st3.y))),
                    Vec2f(std::max(st0.x, std::max(st1.x, st3.x)),
                          std::max(st0.y, std::max(st1.y, st3.y))));
                rasterizeTrianglePos(roiST, width, height, st0, st1, st3,
                    pos0, pos1, pos3, nrm0, nrm1, nrm3, posResult, nrmResult);
                // second triangle: <2, 3, 1>
                roiST = scene_rdl2::math::BBox2f(
                    Vec2f(std::min(st2.x, std::min(st3.x, st1.x)),
                          std::min(st2.y, std::min(st3.y, st1.y))),
                    Vec2f(std::max(st2.x, std::max(st3.x, st1.x)),
                          std::max(st2.y, std::max(st3.y, st1.y))));
                rasterizeTrianglePos(roiST, width, height, st2, st3, st1,
                    pos2, pos3, pos1, nrm2, nrm3, nrm1, posResult, nrmResult);
            }
        }
    } else {
        for (size_t faceId = 0; faceId < getTessellatedMeshFaceCount(); ++faceId) {
            const uint32_t id0 = mIndices[faceId * sTriangleVertexCount];
            const uint32_t id1 = mIndices[faceId * sTriangleVertexCount + 1];
            const uint32_t id2 = mIndices[faceId * sTriangleVertexCount + 2];

            const Vec3f pos0 = mVertices(id0).asVec3f();
            const Vec3f pos1 = mVertices(id1).asVec3f();
            const Vec3f pos2 = mVertices(id2).asVec3f();

            Vec2f st0, st1, st2;
            getTriangleAttributes(stKey, faceId, faceId, st0, st1, st2, 0.0f, true);

            // check if this face exists in this udim and if so,
            // transform the st coordinates into the normalized range
            if (udimxform(udim, st0) && udimxform(udim, st1) &&
                udimxform(udim, st2)) {
                scene_rdl2::math::BBox2f roiST(
                    Vec2f(std::min(st0.x, std::min(st1.x, st2.x)),
                          std::min(st0.y, std::min(st1.y, st2.y))),
                    Vec2f(std::max(st0.x, std::max(st1.x, st2.x)),
                          std::max(st0.y, std::max(st1.y, st2.y))));

                const Vec3f *nrm0 = nullptr;
                const Vec3f *nrm1 = nullptr;
                const Vec3f *nrm2 = nullptr;
                Vec3f nrmData0, nrmData1, nrmData2;
                if (nrmResult) {
                    getTriangleNormal(faceId, faceId,
                        nrmData0, nrmData1, nrmData2, 0.0f, true);
                    nrm0 = &nrmData0;
                    nrm1 = &nrmData1;
                    nrm2 = &nrmData2;
                }
                rasterizeTrianglePos(roiST, width, height, st0, st1, st2,
                    pos0, pos1, pos2, nrm0, nrm1, nrm2, posResult, nrmResult);
            }
        }
    }

    return true;
}

void
TriMesh::setRequiredAttributes(int primId, float time, float u, float v,
    float w, bool isFirst, Intersection& intersection) const
{
    int tessFaceId = primId;
    int baseFaceId = getBaseFaceId(tessFaceId);
    uint32_t id1, id2, id3, id4;
    uint32_t isecId1, isecId2, isecId3;
    uint32_t varyingId1, varyingId2, varyingId3;
    // weights for interpolator
    float varyingW[3];

    // isFirst only matters when triangle got tessellated to quads
    if (mIsTessellated) {
        // triangle got tessellated to quads
        id1 = mIndices[sQuadVertexCount * tessFaceId    ];
        id2 = mIndices[sQuadVertexCount * tessFaceId + 1];
        id3 = mIndices[sQuadVertexCount * tessFaceId + 2];
        id4 = mIndices[sQuadVertexCount * tessFaceId + 3];
        // the surface uv of intersection point on base face
        Vec2f uv;
        if (isFirst) {
            // Triangle (0,1,3)
            intersection.setIds(id1, id2, id4);
            // remap tessellated face barycentric coordinate to
            // base face barycentric coordinate
            uv = w * mFaceVaryingUv[sQuadVertexCount * tessFaceId + 0] +
                 u * mFaceVaryingUv[sQuadVertexCount * tessFaceId + 1] +
                 v * mFaceVaryingUv[sQuadVertexCount * tessFaceId + 3];
        } else {
            // Triangle (2,3,1)
            intersection.setIds(id3, id4, id2);
            // remap tessellated face barycentric coordinate to
            // base face barycentric coordinate
            uv = w * mFaceVaryingUv[sQuadVertexCount * tessFaceId + 2] +
                 u * mFaceVaryingUv[sQuadVertexCount * tessFaceId + 3] +
                 v * mFaceVaryingUv[sQuadVertexCount * tessFaceId + 1];
        }
        intersection.getIds(isecId1, isecId2, isecId3);
        varyingId1 = mBaseIndices[sTriangleVertexCount * baseFaceId    ];
        varyingId2 = mBaseIndices[sTriangleVertexCount * baseFaceId + 1];
        varyingId3 = mBaseIndices[sTriangleVertexCount * baseFaceId + 2];
        varyingW[0] = 1.0f - uv[0] - uv[1];
        varyingW[1] = uv[0];
        varyingW[2] = uv[1];
    } else {
        id1 = mIndices[sTriangleVertexCount * tessFaceId    ];
        id2 = mIndices[sTriangleVertexCount * tessFaceId + 1];
        id3 = mIndices[sTriangleVertexCount * tessFaceId + 2];
        intersection.setIds(id1, id2, id3);
        intersection.getIds(isecId1, isecId2, isecId3);
        varyingId1 = isecId1;
        varyingId2 = isecId2;
        varyingId3 = isecId3;
        varyingW[0] = w;
        varyingW[1] = u;
        varyingW[2] = v;
    }

    int partId = (mFaceToPart.size() > 0) ? mFaceToPart[baseFaceId] : 0;
    const Attributes* primitiveAttributes = getAttributes();
    // interpolate PrimitiveAttribute
    MeshInterpolator interpolator(primitiveAttributes,
        time, partId, baseFaceId,
        varyingId1, varyingId2, varyingId3,
        varyingW[0], varyingW[1], varyingW[2], tessFaceId,
        isecId1, isecId2, isecId3,
        w, u, v);
    intersection.setRequiredAttributes(interpolator);

    // Add an interpolated N, dPds, and dPdt to the intersection
    // if they exist in the primitive attribute table
    setExplicitAttributes<MeshInterpolator>(interpolator,
                                            *primitiveAttributes,
                                            intersection);
}

void
TriMesh::postIntersect(mcrt_common::ThreadLocalState& tls,
                       const scene_rdl2::rdl2::Layer* pRdlLayer, const mcrt_common::Ray& ray,
                       Intersection& intersection) const
{
    int tessFaceId = ray.primID;
    int baseFaceId = getBaseFaceId(tessFaceId);

    // barycentric coordinate
    float u = ray.u;
    float v = ray.v;
    float w = 1.0f - u - v;
    bool isFirst = w > 0.0f;
    if (!isFirst && mIsTessellated) {
        u = 1.0f - u;
        v = 1.0f - v;
        w = -w;
    }

    const int assignmentId = getFaceAssignmentId(tessFaceId);
    intersection.setLayerAssignments(assignmentId, pRdlLayer);

    const AttributeTable *table =
        intersection.getMaterial()->get<shading::RootShader>().getAttributeTable();
    intersection.setTable(&tls.mArena, table);
    const Attributes* primitiveAttributes = getAttributes();

    setRequiredAttributes(tessFaceId,
                          ray.time,
                          u, v, w,
                          isFirst,
                          intersection);

    uint32_t isecId1, isecId2, isecId3;
    intersection.getIds(isecId1, isecId2, isecId3);

    overrideInstanceAttrs(ray, intersection);

    // The St value is read from the explicit "surface_st" primitive
    // attribute if it exists.
    Vec2f st1, st2, st3;
    getTriangleST(baseFaceId, tessFaceId, st1, st2, st3, isFirst);
    const Vec2f St = w * st1 + u * st2 + v * st3;

    const Vec3f Ng = normalize(ray.getNg());

    Vec3f N, dPds, dPdt;
    const bool hasExplicitAttributes = getExplicitAttributes(*primitiveAttributes,
                                                             intersection,
                                                             N, dPds, dPdt);

    if (!hasExplicitAttributes) {
        if (primitiveAttributes->isSupported(StandardAttributes::sNormal)) {
            N = intersection.getN();
        } else {
            N = Ng;
        }
        if (isMotionBlurOn()) {
            float t = ray.time;
            const Vec3f& p10 = mVertices(isecId1, 0);
            const Vec3f& p20 = mVertices(isecId2, 0);
            const Vec3f& p30 = mVertices(isecId3, 0);
            // form a ReferenceFrame and use its tangent/binormal as dpds dpdt
            // if we fail to calculate the st partial derivatives
            Vec3f dp0[2];
            if (!computeTrianglePartialDerivatives(
                p10, p20, p30, st1, st2, st3, dp0)) {
                scene_rdl2::math::ReferenceFrame frame(N);
                dp0[0] = frame.getX();
                dp0[1] = frame.getY();
            }
            const Vec3f& p11 = mVertices(isecId1, 1);
            const Vec3f& p21 = mVertices(isecId2, 1);
            const Vec3f& p31 = mVertices(isecId3, 1);
            Vec3f dp1[2];
            if (!computeTrianglePartialDerivatives(
                p11, p21, p31, st1, st2, st3, dp1)) {
                scene_rdl2::math::ReferenceFrame frame(N);
                dp1[0] = frame.getX();
                dp1[1] = frame.getY();
            }
            dPds = (1.0f - t) * dp0[0] + t * dp1[0];
            dPdt = (1.0f - t) * dp0[1] + t * dp1[1];
        } else {
            const Vec3f& p1 = mVertices(isecId1);
            const Vec3f& p2 = mVertices(isecId2);
            const Vec3f& p3 = mVertices(isecId3);
            // form a ReferenceFrame and use its tangent/binormal as dpds dpdt
            // if we fail to calculate the st partial derivatives
            Vec3f dp[2];
            if (!computeTrianglePartialDerivatives(p1, p2, p3, st1, st2, st3, dp)) {
                scene_rdl2::math::ReferenceFrame frame(N);
                dp[0] = frame.getX();
                dp[1] = frame.getY();
            }
            dPds = dp[0];
            dPdt = dp[1];
        }
    }

    intersection.setDifferentialGeometry(Ng,
                                         N,
                                         St,
                                         dPds,
                                         dPdt,
                                         true); // has derivatives

    // calculate dfds/dfdt for primitive attributes that request differential
    if (table->requestDerivatives()) {
        computeAttributesDerivatives(table, st1, st2, st3,
            baseFaceId, tessFaceId, ray.time, isFirst, intersection);
    }

    // fill in ray epsilon to avoid self intersection
    const scene_rdl2::rdl2::Geometry* geometry = intersection.getGeometryObject();
    MNRY_ASSERT(geometry != nullptr);
    intersection.setEpsilonHint(geometry->getRayEpsilon());

    // polygon vertices
    int numVertices = mIsTessellated ? 4 : 3;
    if (table->requests(StandardAttributes::sNumPolyVertices)) {
        intersection.setAttribute(StandardAttributes::sNumPolyVertices, numVertices);
    }
    if (table->requests(StandardAttributes::sPolyVertexType)) {
        intersection.setAttribute(StandardAttributes::sPolyVertexType,
            static_cast<int>(StandardAttributes::POLYVERTEX_TYPE_POLYGON));
    }
    if (table->requests(StandardAttributes::sReversedNormals)) {
        intersection.setAttribute(StandardAttributes::sReversedNormals, mIsNormalReversed);
    }
    for (int iVert = 0; iVert < numVertices; iVert++) {
        if (table->requests(StandardAttributes::sPolyVertices[iVert])) {
            uint32_t id = mIndices[numVertices * tessFaceId + iVert];
            // may need to move the vertices to render space
            // for instancing object since they are ray traced in local space
            const Vec3f v = ray.isInstanceHit() ? transformPoint(ray.ext.l2r, mVertices(id).asVec3f())
                                                : mVertices(id).asVec3f();
            intersection.setAttribute(StandardAttributes::sPolyVertices[iVert], v);
        }
    }

    // motion vectors
    if (table->requests(StandardAttributes::sMotion)) {
        const Vec3f motion = computeMotion(mVertices, isecId1, isecId2, isecId3, w, u, v, ray);
        intersection.setAttribute(StandardAttributes::sMotion, motion);
    }
}

bool
TriMesh::computeIntersectCurvature(const mcrt_common::Ray& ray,
        const Intersection& intersection, Vec3f& dnds, Vec3f& dndt) const
{
    int tessFaceId = ray.primID;
    int baseFaceId = getBaseFaceId(tessFaceId);
    uint32_t id1, id2, id3;
    intersection.getIds(id1, id2, id3);
    bool isFirst = ray.u + ray.v <= 1.0f;
    Vec2f st1, st2, st3;
    getTriangleST(baseFaceId, tessFaceId, st1, st2, st3, isFirst);
    Vec3f n1, n2, n3;
    getTriangleNormal(baseFaceId, tessFaceId, n1, n2, n3, ray.time, isFirst);
    Vec3f dn[2];
    if (computeTrianglePartialDerivatives(n1, n2, n3, st1, st2, st3, dn)) {
        dnds = dn[0];
        dndt = dn[1];
        return true;
    } else {
        return false;
    }
}

void
TriMesh::getST(int tessFaceId, float u, float v, Vec2f& st) const
{
    int baseFaceId = getBaseFaceId(tessFaceId);
    bool isFirst = u + v <= 1.0f;
    float w = 1.0f - u - v;
    Vec2f st1, st2, st3;
    getTriangleST(baseFaceId, tessFaceId, st1, st2, st3, isFirst);
    st = w * st1 + u * st2 + v * st3;
}

void
TriMesh::getNeighborVertices(int baseFaceId, int tessFaceId,
        int tessFaceVIndex, int& vid, int& vid1, int& vid2, int& vid3,
        Vec2f& st, Vec2f& st1, Vec2f& st2, Vec2f& st3) const
{
    if (mIsTessellated) {
        size_t offset = tessFaceId * sQuadVertexCount;
        if (tessFaceVIndex == 2) {
            if (mIndices[offset + 2] == mIndices[offset + 3]) {
                // the degened quad introduced during tessellation
                // first triangle on quad (0, 1, 3), which is (0, 1, 2)
                getTriangleST(baseFaceId, tessFaceId, st1, st2, st3, true);
                vid1 = mIndices[offset + 0];
                vid2 = mIndices[offset + 1];
                vid3 = mIndices[offset + 2];
                vid = vid3;
                st = st3;
            } else {
                // second triangle on quad (2, 3, 1)
                getTriangleST(baseFaceId, tessFaceId, st1, st2, st3, false);
                vid1 = mIndices[offset + 2];
                vid2 = mIndices[offset + 3];
                vid3 = mIndices[offset + 1];
                vid = vid1;
                st = st1;
            }
        } else {
            // first triangle on quad (0, 1, 3)
            getTriangleST(baseFaceId, tessFaceId, st1, st2, st3, true);
            vid1 = mIndices[offset + 0];
            vid2 = mIndices[offset + 1];
            vid3 = mIndices[offset + 3];
            if (tessFaceVIndex == 0) {
                vid = vid1;
                st = st1;
            } else if (tessFaceVIndex == 1) {
                vid = vid2;
                st = st2;
            } else if (tessFaceVIndex == 3) {
                vid = vid3;
                st = st3;
            }
        }
    } else {
        size_t offset = tessFaceId * sTriangleVertexCount;
        getTriangleST(baseFaceId, tessFaceId, st1, st2, st3, true);
        vid1 = mIndices[offset + 0];
        vid2 = mIndices[offset + 1];
        vid3 = mIndices[offset + 2];
        if (tessFaceVIndex == 0) {
            vid = vid1;
            st = st1;
        } else if (tessFaceVIndex == 1) {
            vid = vid2;
            st = st2;
        } else if (tessFaceVIndex == 2) {
            vid = vid3;
            st = st3;
        }
    }
}

// This function is used to wrap the requested index if it
// is negative or exceeds the face vertex count
uint32_t
loopIndex(int requestedIndex,
          int fvCount)
{
    if (requestedIndex >= 0 && requestedIndex < fvCount) {
        // Within normal range (0-fvCount)
        return requestedIndex;
    } else if (requestedIndex >= fvCount) {
        // Above fvCount so wrap around to the beginning
        return requestedIndex % fvCount;
    } else {
        // Below fvCount so wrap around to the end
        return  requestedIndex % fvCount + fvCount;
    }
}

// Calculate the unit normal of plane defined by points a, b, and c
Vec3f
unitNormal(Vec3f a,
           Vec3f b,
           Vec3f c)
{
    const float x = Mat3f(1.0f, a.y, a.z,
                          1.0f, b.y, b.z,
                          1.0f, c.y, c.z).det();

    const float y = Mat3f(a.x, 1.0f, a.z,
                          b.x, 1.0f, b.z,
                          c.x, 1.0f, c.z).det();

    const float z = Mat3f(a.x, a.y, 1.0f,
                          b.x, b.y, 1.0f,
                          c.x, c.y, 1.0f).det();

    const float magnitude =
        scene_rdl2::math::max(sEpsilon,
                              scene_rdl2::math::sqrt(x*x + y*y + z*z));
    return Vec3f(x / magnitude,
                 y / magnitude,
                 z / magnitude);
}

float
calculateConcaveNGonArea(const PolygonMesh::VertexBuffer& vertices,
                         const PolygonMesh::IndexBuffer& indices,
                         size_t fvCount,
                         size_t inputIndexOffset)
{
    Vec3f total(0.f);
    for (size_t i = 0; i < fvCount; ++i) {
        const size_t index1 = indices[loopIndex(i + 0, fvCount) + inputIndexOffset];
        const size_t index2 = indices[loopIndex(i + 1, fvCount) + inputIndexOffset];
        const Vec3f vi1 = vertices(index1, 0);
        const Vec3f vi2 = vertices(index2, 0);
        total = total + cross(vi1, vi2);
    }

    const Vec3f un = unitNormal(vertices(indices[0 + inputIndexOffset], 0),
                                vertices(indices[1 + inputIndexOffset], 0),
                                vertices(indices[2 + inputIndexOffset], 0));
    const float result = dot(total, un);
    return result / 2.f;
}

// Calculate an ngon normal from the sum of the cross
// products of it's adjacent vertices.  If the area of
// the ngon is negative, this indicates a reverse winding
// order and we therefore flip the normal.
Vec3f
calculateConcaveNGonNormal(const PolygonMesh::VertexBuffer& vertices,
                           const PolygonMesh::IndexBuffer& indices,
                           size_t fvCount,
                           size_t inputIndexOffset)
{
    const float area = calculateConcaveNGonArea(vertices,
                                                indices,
                                                fvCount,
                                                inputIndexOffset);
    Vec3fa sum(0.f, 0.f, 0.f);
    for (size_t i = 0; i < fvCount; ++i) {
        const Vec3f va = vertices(indices[i + inputIndexOffset], 0);
        const Vec3f vb = vertices(indices[loopIndex(i + 1, fvCount) + inputIndexOffset], 0);
        sum = sum + cross(va, vb);
    }
    if (area < 0.f) {
        sum *= -1.f;
    }
    return normalize(sum);
}

bool
isPlanar(const PolygonMesh::VertexBuffer& vertices,
         const PolygonMesh::IndexBuffer& indices,
         size_t fvCount,
         size_t inputIndexOffset)
{
    // Calculate normal of first 3 points
    const Vec3f& va = vertices(indices[0 + inputIndexOffset], 0);
    const Vec3f& vb = vertices(indices[1 + inputIndexOffset], 0);
    const Vec3f& vc = vertices(indices[2 + inputIndexOffset], 0);
    const Vec3f N =  cross(va - vb, vc - vb);

    // Check that the other points lie in the plane of the first 3
    for (size_t i = 3; i < fvCount; ++i) {
        const Vec3f v = vertices(indices[i + inputIndexOffset], 0);
        if (!isZero(dot(normalize(va - v), N))) {
            return false;
        }
    }
    return true;
}

bool
pointInTriangle(const Vec3f& N,
                const Vec3fa& p,
                const Vec3fa& a,
                const Vec3fa& b,
                const Vec3fa& c)
{
    // Get barycentric coords
    const Vec3f v0 = b - a;
    const Vec3f v1 = c - a;
    const Vec3f v2 = p - a;
    const float d00 = dot(v0, v0);
    const float d01 = dot(v0, v1);
    const float d11 = dot(v1, v1);
    const float d20 = dot(v2, v0);
    const float d21 = dot(v2, v1);
    const float denom = d00 * d11 - d01 * d01;
    const float v = (d11 * d20 - d01 * d21) / denom;
    const float w = (d00 * d21 - d01 * d20) / denom;
    const float u = 1.0f - v - w;

    // The Point is outside triangle(false) if any
    // of the coords are less than zero.
    return u >= 0 && v >= 0 && w >= 0;
}

void
TriMesh::splitNGons(size_t outputFaceCount,
                    const PolygonMesh::VertexBuffer& vertices,
                    PolygonMesh::FaceToPartBuffer& faceToPart,
                    const PolygonMesh::FaceVertexCount& faceVertexCount,
                    PolygonMesh::IndexBuffer& indices,
                    LayerAssignmentId& layerAssignmentId,
                    PrimitiveAttributeTable& primitiveAttributeTable)
{
    size_t inputFaceCount = faceVertexCount.size();
    size_t inputIndexCount = indices.size();
    indices.resize(sTriangleVertexCount * outputFaceCount);
    if (faceToPart.size() > 0) {
        faceToPart.resize(outputFaceCount);
    }

    // Track original indices for face varying primitive attributes
    PolygonMesh::IndexBuffer indexRemapping;
    indexRemapping.resize(sTriangleVertexCount * outputFaceCount);

    size_t inputIndexOffset = inputIndexCount;
    size_t outputIndexOffset = indices.size() - 1;
    size_t outputF2POffset = faceToPart.size() - 1;
    // triangulate indices
    for (int f = inputFaceCount - 1; f >= 0; --f) {
        size_t fvCount = faceVertexCount[f];
        inputIndexOffset -= fvCount;

        // Check that the polygon has a non zero area and
        // planar before running the ear clipping algorithm
        const float area = calculateConcaveNGonArea(vertices,
                                                    indices,
                                                    fvCount,
                                                    inputIndexOffset);

        const bool nonZeroArea = !std::isnan(area) && area >= sEpsilon;

        bool planar = true;
        if (fvCount > 3 && nonZeroArea) {
            planar = isPlanar(vertices,
                              indices,
                              fvCount,
                              inputIndexOffset);
        }

        if (!nonZeroArea || !planar || fvCount <= 4) {
            for (size_t v = fvCount; v >= sTriangleVertexCount; --v) {
                // triangulate a ngon into triangles fan
                indexRemapping[outputIndexOffset] = inputIndexOffset + v - 1;
                indices[outputIndexOffset--] = indices[inputIndexOffset + v - 1];
                indexRemapping[outputIndexOffset] = inputIndexOffset + v - 2;
                indices[outputIndexOffset--] = indices[inputIndexOffset + v - 2];
                indexRemapping[outputIndexOffset] = inputIndexOffset;
                indices[outputIndexOffset--] = indices[inputIndexOffset];
                if (faceToPart.size() > 0) {
                    faceToPart[outputF2POffset--] = faceToPart[f];
                }
            }
        } else {
            PolygonMesh::IndexBuffer indexRemappingCopy = indexRemapping;
            size_t outputIndexOffsetCopy = outputIndexOffset;
            size_t outputF2POffsetCopy = outputF2POffset;

            // Ear clipping algorithm for concave ngons
            // https://en.wikipedia.org/wiki/Polygon_triangulation
            const Vec3f N = calculateConcaveNGonNormal(vertices,
                                                       indices,
                                                       fvCount,
                                                       inputIndexOffset);

            // Create a list of local indices to keep track
            // of which vertices have been ear clipped
            PolygonMesh::IndexBuffer localIndices;
            localIndices.resize(fvCount);
            for (uint32_t i = 0; i < fvCount; ++i) {
                localIndices[i] = i;
            }

            size_t numRemainingIndices = localIndices.size();
            bool success;
            while (numRemainingIndices > 3) {
                success = false;
                for (int i = numRemainingIndices - 1; i >= 0; --i) {
                    // Triangle vertices in clockwise winding order (b, a, c)
                    const uint32_t a = localIndices[i];
                    const uint32_t b = localIndices[loopIndex(i - 1, numRemainingIndices)];
                    const uint32_t c = localIndices[loopIndex(i + 1, numRemainingIndices)];

                    const Vec3fa& va = vertices(indices[inputIndexOffset + a], 0);
                    const Vec3fa& vb = vertices(indices[inputIndexOffset + b], 0);
                    const Vec3fa& vc = vertices(indices[inputIndexOffset + c], 0);

                    // Check vertex for concavity
                    const Vec3fa vab = vb - va;
                    const Vec3fa vac = vc - va;
                    const Vec3fa cp = cross(vac, vab);
                    if (dot(cp, N) < 0.f) {
                        continue;
                    }

                    // Check triangle for interior points
                    bool hasInteriorPoints = false;
                    for (size_t j = 0; j < numRemainingIndices; ++j) {
                        // Skip the points of the current triangle
                        if (localIndices[j] == a ||
                            localIndices[j] == b ||
                            localIndices[j] == c) continue;

                        const uint32_t indexP = indices[localIndices[j] + inputIndexOffset];
                        const Vec3fa& p = vertices(indexP, 0);
                        if (pointInTriangle(N, p, vb, va, vc)) {
                            hasInteriorPoints = true;
                            break;
                        }
                    }
                    if (hasInteriorPoints) {
                        continue;
                    }

                    // Add the triangle in reverse winding order since
                    // we are iterating from the end of the indices to
                    // the beginning
                    indexRemapping[outputIndexOffset] = inputIndexOffset + c;
                    indices[outputIndexOffset--] = indices[inputIndexOffset + c];
                    indexRemapping[outputIndexOffset] = inputIndexOffset + a;
                    indices[outputIndexOffset--] = indices[inputIndexOffset + a];
                    indexRemapping[outputIndexOffset] = inputIndexOffset + b;
                    indices[outputIndexOffset--] = indices[inputIndexOffset + b];
                    if (faceToPart.size() > 0) {
                        faceToPart[outputF2POffset--] = faceToPart[f];
                    }

                    // Remove this vertex from the list to check
                    localIndices.erase(localIndices.begin() + i);
                    numRemainingIndices -= 1;
                    success = true;
                    break;
                }

                if (!success) {
                    indexRemapping = indexRemappingCopy;
                    outputIndexOffset = outputIndexOffsetCopy;
                    outputF2POffset = outputF2POffsetCopy;
                    success = false;
                    break;
                }
            }

            if (success) {
                // Add the last triangle
                indexRemapping[outputIndexOffset] = inputIndexOffset + localIndices[2];
                indices[outputIndexOffset--] = indices[inputIndexOffset + localIndices[2]];
                indexRemapping[outputIndexOffset] = inputIndexOffset + localIndices[1];
                indices[outputIndexOffset--] = indices[inputIndexOffset + localIndices[1]];
                indexRemapping[outputIndexOffset] = inputIndexOffset + localIndices[0];
                indices[outputIndexOffset--] = indices[inputIndexOffset + localIndices[0]];
                if (faceToPart.size() > 0) {
                    faceToPart[outputF2POffset--] = faceToPart[f];
                }
            } else {
                // Revert to triangle fan
                for (size_t v = fvCount; v >= sTriangleVertexCount; --v) {
                    // triangulate a ngon into triangles fan
                    indexRemapping[outputIndexOffset] = inputIndexOffset + v - 1;
                    indices[outputIndexOffset--] = indices[inputIndexOffset + v - 1];
                    indexRemapping[outputIndexOffset] = inputIndexOffset + v - 2;
                    indices[outputIndexOffset--] = indices[inputIndexOffset + v - 2];
                    indexRemapping[outputIndexOffset] = inputIndexOffset;
                    indices[outputIndexOffset--] = indices[inputIndexOffset];
                    if (faceToPart.size() > 0) {
                        faceToPart[outputF2POffset--] = faceToPart[f];
                    }
                }
            }
        }
    }

    // triangulate layerAssignmentId if it's in per face frequency
    if (layerAssignmentId.getType() == LayerAssignmentId::Type::VARYING) {
        auto& faceAssignmentIds = layerAssignmentId.getVaryingId();
        faceAssignmentIds.resize(outputFaceCount);
        size_t outputFaceOffset = outputFaceCount - 1;
        for (int f = inputFaceCount - 1; f >= 0; --f) {
            size_t fvCount = faceVertexCount[f];
            for (size_t v = fvCount; v >= sTriangleVertexCount; --v) {
                faceAssignmentIds[outputFaceOffset--] = faceAssignmentIds[f];
            }
        }
    }

    // triangulate uniform and facevarying primitive attributes
    for (auto& kv : primitiveAttributeTable) {
        for (auto& attribute : kv.second) {
            AttributeRate rate = attribute->getRate();
            if (rate == RATE_UNIFORM) {
                attribute->resize(outputFaceCount);
                size_t outputFaceOffset = outputFaceCount - 1;
                for (int f = inputFaceCount - 1; f >= 0; --f) {
                    size_t fvCount = faceVertexCount[f];
                    for (size_t v = fvCount; v >= sTriangleVertexCount; --v) {
                        attribute->copyInPlace(f, outputFaceOffset--);
                    }
                }
            } else if (rate == RATE_FACE_VARYING) {
                attribute->resize(sTriangleVertexCount * outputFaceCount);
                size_t outputIndexOffset = indices.size() - 1;
                for (int f = inputFaceCount - 1; f >= 0; --f) {
                    size_t fvCount = faceVertexCount[f];
                    inputIndexOffset -= fvCount;
                    for (size_t v = fvCount; v >= sTriangleVertexCount; --v) {
                        attribute->copyInPlace(indexRemapping[outputIndexOffset],
                                               outputIndexOffset);
                        outputIndexOffset--;
                        attribute->copyInPlace(indexRemapping[outputIndexOffset],
                                               outputIndexOffset);
                        outputIndexOffset--;
                        attribute->copyInPlace(indexRemapping[outputIndexOffset],
                                               outputIndexOffset);
                        outputIndexOffset--;
                    }
                }
            }
        }
    }
}

void
TriMesh::generateIndexBufferAndSurfaceSamples(
        const std::vector<PolyFaceTopology>& faceTopologies,
        const PolyTessellatedVertexLookup& tessellatedVertexLookup,
        PolygonMesh::IndexBuffer& indices,
        std::vector<PolyMesh::SurfaceSample>& surfaceSamples,
        std::vector<int>& tessellatedToBaseFace,
        std::vector<Vec2f>* faceVaryingUv) const
{
    tessellatedToBaseFace.clear();
    // we need to use arena to allocate temporary data structure for
    // outer/interior ring stitching
    mcrt_common::ThreadLocalState *tls = mcrt_common::getFrameUpdateTLS();
    scene_rdl2::alloc::Arena *arena = &tls->mArena;

    size_t tessellatedVertexCount =
        tessellatedVertexLookup.getTessellatedVertexCount();
    surfaceSamples.resize(tessellatedVertexCount);
    // please refer to OpenGL spec for triangle tessellation detail
    // 11.2.2.1 Triangle Tessellation p371
    // https://www.khronos.org/registry/OpenGL/specs/gl/glspec44.core.pdf
    for (size_t f = 0; f < faceTopologies.size(); ++f) {
        const PolyFaceTopology& faceTopology = faceTopologies[f];
        int maxEdgeVertexCount =
            tessellatedVertexLookup.getMaxEdgeVertexCount(faceTopology);
        // simplest case
        if (maxEdgeVertexCount == 0) {
            int vid0 = tessellatedVertexLookup.getTessellatedVertexId(
                faceTopology.mCornerVertexId[0]);
            int vid1 = tessellatedVertexLookup.getTessellatedVertexId(
                faceTopology.mCornerVertexId[1]);
            int vid2 = tessellatedVertexLookup.getTessellatedVertexId(
                faceTopology.mCornerVertexId[2]);
            indices.push_back(vid0);
            indices.push_back(vid1);
            indices.push_back(vid2);
            indices.push_back(vid2);
            if (faceVaryingUv) {
                faceVaryingUv->emplace_back(0.0f, 0.0f);
                faceVaryingUv->emplace_back(1.0f, 0.0f);
                faceVaryingUv->emplace_back(0.0f, 1.0f);
                faceVaryingUv->emplace_back(0.0f, 1.0f);
            }
            tessellatedToBaseFace.push_back(f);
            surfaceSamples[vid0].mFaceId = f;
            surfaceSamples[vid0].mUv = Vec2f(0.0f, 0.0f);
            surfaceSamples[vid1].mFaceId = f;
            surfaceSamples[vid1].mUv = Vec2f(1.0f, 0.0f);
            surfaceSamples[vid2].mFaceId = f;
            surfaceSamples[vid2].mUv = Vec2f(0.0f, 1.0f);
            continue;
        }
        // TODO handle uniform tessellation in more optimized topology
        // interior region index buffer first.
        int interiorRingCount =
            tessellatedVertexLookup.getTriangleRingCount(faceTopology) - 1;
        // how many rings to pass through from outer vertex to central
        // with barycentric coordinate shifting from 0 to 1/3
        float ringUvDelta = 2.0f / (3.0f * (float)(maxEdgeVertexCount + 1));
        for (int r = 0; r < interiorRingCount - 1; ++r) {
            // use the same barycentric coordinate encoding embree does
            // (the weight of v1 and v2, so three corner vertices v0, v1, v2
            // with barycentric coordinates (1, 0, 0), (0, 1, 0), (0, 0, 1) are
            // encoded as (0, 0), (1, 0), (0, 1)

            // cornerUv0 is the corner barycentric coordinates in the
            // current interior ring
            // cornerUv1 is the corner barycentric coordinates in the
            // next interior ring (closer to center)
            Vec2f cornerUv0[3] = {
                Vec2f((r + 1) * ringUvDelta, (r + 1) * ringUvDelta),
                Vec2f(1.0f - 2 * (r + 1) * ringUvDelta, (r + 1) * ringUvDelta),
                Vec2f((r + 1) * ringUvDelta, 1.0f - 2 * (r + 1) * ringUvDelta)};
            Vec2f cornerUv1[3] = {
                Vec2f((r + 2) * ringUvDelta, (r + 2) * ringUvDelta),
                Vec2f(1.0f - 2 * (r + 2) * ringUvDelta, (r + 2) * ringUvDelta),
                Vec2f((r + 2) * ringUvDelta, 1.0f - 2 * (r + 2) * ringUvDelta)};
            int edgeVertexCount = maxEdgeVertexCount - 2 * (r + 1);
            float w0 = 1.0f / (float)(edgeVertexCount + 1);
            float w1 = 1.0f / (float)(edgeVertexCount - 1);
            for (size_t c = 0; c < sTriangleVertexCount; ++c) {
                // corner quad
                indices.push_back(
                    tessellatedVertexLookup.getTriangleInteriorVertexId(
                    f, maxEdgeVertexCount, r, c, 0));
                indices.push_back(
                    tessellatedVertexLookup.getTriangleInteriorVertexId(
                    f, maxEdgeVertexCount, r, c, 1));
                indices.push_back(
                    tessellatedVertexLookup.getTriangleInteriorVertexId(
                    f, maxEdgeVertexCount, r + 1, c, 0));
                indices.push_back(
                    tessellatedVertexLookup.getTriangleInteriorVertexId(
                    f, maxEdgeVertexCount, r, (c + 2) % sTriangleVertexCount,
                    edgeVertexCount));
                if (faceVaryingUv) {
                    faceVaryingUv->push_back(cornerUv0[c]);
                    faceVaryingUv->push_back(
                        (1.0f - w0) * cornerUv0[c] +
                        (       w0) * cornerUv0[(c + 1) % sTriangleVertexCount]);
                    faceVaryingUv->push_back(cornerUv1[c]);
                    faceVaryingUv->push_back(
                        (1.0f - w0) * cornerUv0[c] +
                        (       w0) * cornerUv0[(c + 2) % sTriangleVertexCount]);
                }
                tessellatedToBaseFace.push_back(f);
                // edge quads
                for (int n = 1; n < edgeVertexCount; ++n) {
                    indices.push_back(
                        tessellatedVertexLookup.getTriangleInteriorVertexId(
                        f, maxEdgeVertexCount, r, c, n));
                    indices.push_back(
                        tessellatedVertexLookup.getTriangleInteriorVertexId(
                        f, maxEdgeVertexCount, r, c, n + 1));
                    indices.push_back(
                        tessellatedVertexLookup.getTriangleInteriorVertexId(
                        f, maxEdgeVertexCount, r + 1, c, n));
                    indices.push_back(
                        tessellatedVertexLookup.getTriangleInteriorVertexId(
                        f, maxEdgeVertexCount, r + 1, c, n - 1));
                    if (faceVaryingUv) {
                        faceVaryingUv->push_back(
                            (1.0f - (n    ) * w0) * cornerUv0[c] +
                            (       (n    ) * w0) * cornerUv0[(c + 1) % sTriangleVertexCount]);
                        faceVaryingUv->push_back(
                            (1.0f - (n + 1) * w0) * cornerUv0[c] +
                            (       (n + 1) * w0) * cornerUv0[(c + 1) % sTriangleVertexCount]);
                        faceVaryingUv->push_back(
                            (1.0f - (n    ) * w1) * cornerUv1[c] +
                            (       (n    ) * w1) * cornerUv1[(c + 1) % sTriangleVertexCount]);
                        faceVaryingUv->push_back(
                            (1.0f - (n - 1) * w1) * cornerUv1[c] +
                            (       (n - 1) * w1) * cornerUv1[(c + 1) % sTriangleVertexCount]);
                    }
                    tessellatedToBaseFace.push_back(f);
                }
                for (int n = 0; n < edgeVertexCount + 1; ++n) {
                    int vid =
                        tessellatedVertexLookup.getTriangleInteriorVertexId(
                        f, maxEdgeVertexCount, r, c, n);
                    surfaceSamples[vid].mFaceId = f;
                    surfaceSamples[vid].mUv =
                        (1.0f - n * w0) * cornerUv0[c] +
                        (       n * w0) * cornerUv0[(c + 1) % sTriangleVertexCount];
                }
            }
        }
        // few remaining interior central edge cases
        if (maxEdgeVertexCount % 2 == 0) {
            int vid0 =
                tessellatedVertexLookup.getTriangleInteriorVertexId(
                f, maxEdgeVertexCount, interiorRingCount - 1, 0, 0);
            int vid1 =
                tessellatedVertexLookup.getTriangleInteriorVertexId(
                f, maxEdgeVertexCount, interiorRingCount - 1, 1, 0);
            int vid2 =
                tessellatedVertexLookup.getTriangleInteriorVertexId(
                f, maxEdgeVertexCount, interiorRingCount - 1, 2, 0);
            Vec2f uv0(
                interiorRingCount * ringUvDelta,
                interiorRingCount * ringUvDelta);
            Vec2f uv1(
                1.0f - 2 * interiorRingCount * ringUvDelta,
                interiorRingCount * ringUvDelta);
            Vec2f uv2(
                interiorRingCount * ringUvDelta,
                1.0f - 2 * interiorRingCount * ringUvDelta);
            // one more central triangle
            indices.push_back(vid0);
            indices.push_back(vid1);
            indices.push_back(vid2);
            indices.push_back(vid2);
            if (faceVaryingUv) {
                faceVaryingUv->push_back(uv0);
                faceVaryingUv->push_back(uv1);
                faceVaryingUv->push_back(uv2);
                faceVaryingUv->push_back(uv2);
            }
            tessellatedToBaseFace.push_back(f);

            surfaceSamples[vid0].mFaceId = f;
            surfaceSamples[vid0].mUv = uv0;
            surfaceSamples[vid1].mFaceId = f;
            surfaceSamples[vid1].mUv = uv1;
            surfaceSamples[vid2].mFaceId = f;
            surfaceSamples[vid2].mUv = uv2;
        } else {
            // one more central point
            int vid =
                tessellatedVertexLookup.getTriangleInteriorVertexId(
                f, maxEdgeVertexCount, interiorRingCount - 1, 0, 0);
            surfaceSamples[vid].mFaceId = f;
            surfaceSamples[vid].mUv = Vec2f(1.0f / 3.0f, 1.0f / 3.0f);
        }

        // stitch interior and outer ring edge
        //     *---*--*--*---*
        //    / \ / \ | / \ / \
        //   *---*----*----*---*
        const Vec2f cornerUv[3] = {
            Vec2f(0.0f, 0.0f), Vec2f(1.0f, 0.0f), Vec2f(0.0f, 1.0f)};
        for (size_t c = 0; c < sTriangleVertexCount; ++c) {
            SCOPED_MEM(arena);
            StitchPoint* innerRing = nullptr;
            StitchPoint* outerRing = nullptr;
            Vec2f* innerRingUv = nullptr;
            Vec2f* outerRingUv = nullptr;
            int innerRingVertexCount, outerRingVertexCount;
            generateTriangleStitchRings(arena, faceTopology, f, c,
                tessellatedVertexLookup,
                maxEdgeVertexCount, ringUvDelta,
                innerRing, innerRingUv, innerRingVertexCount,
                outerRing, outerRingUv, outerRingVertexCount);
            stitchRings(innerRing, innerRingUv, innerRingVertexCount,
                outerRing, outerRingUv, outerRingVertexCount, indices,
                f, &tessellatedToBaseFace, faceVaryingUv);
            // surface samples
            float w = 1.0f / (float)(outerRingVertexCount - 1);
            for (int v = 0; v < outerRingVertexCount - 1; ++v) {
                int vid = outerRing[v].mVertexId;
                surfaceSamples[vid].mFaceId = f;
                surfaceSamples[vid].mUv =
                    (1.0f - v * w) * cornerUv[c] +
                    (       v * w) * cornerUv[(c + 1) % sTriangleVertexCount];
            }
        }
    }
}

PolygonMesh::VertexBuffer
TriMesh::generateVertexBuffer(
        const PolygonMesh::VertexBuffer& baseVertices,
        const PolygonMesh::IndexBuffer& baseIndices,
        const std::vector<PolyMesh::SurfaceSample>& surfaceSamples) const
{
    // allocate tessellated vertex/index buffer to hold the evaluation result
    size_t tessellatedVertexCount = surfaceSamples.size();
    size_t motionSampleCount = baseVertices.get_time_steps();
    PolygonMesh::VertexBuffer tessellatedVertices(
        tessellatedVertexCount, motionSampleCount);
    tbb::blocked_range<size_t> range =
        tbb::blocked_range<size_t>(0, tessellatedVertexCount);
    tbb::parallel_for(range, [&](const tbb::blocked_range<size_t> &r) {
        for (size_t i = r.begin(); i < r.end(); ++i) {
            int fid = surfaceSamples[i].mFaceId;
            int vid0 = baseIndices[sTriangleVertexCount * fid + 0];
            int vid1 = baseIndices[sTriangleVertexCount * fid + 1];
            int vid2 = baseIndices[sTriangleVertexCount * fid + 2];
            // bilinear interpolate the new position
            Vec2f uv = surfaceSamples[i].mUv;
            float u = uv[0];
            float v = uv[1];
            float w = 1.0f - u - v;
            for (size_t t = 0; t < motionSampleCount; ++t) {
                tessellatedVertices(i, t) =
                    w * baseVertices(vid0, t) +
                    u * baseVertices(vid1, t) +
                    v * baseVertices(vid2, t);
            }
        }
    });
    return tessellatedVertices;
}

void
TriMesh::fillDisplacementAttributes(int tessFaceId, int vIndex,
        Intersection& intersection) const
{
    const AttributeTable* table = intersection.getTable();
    if (table == nullptr) {
        return;
    }
    int baseFaceId = getBaseFaceId(tessFaceId);
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
                getAttribute(typedKey, baseFaceId, tessFaceId, vIndex));
            }
            break;
        case scene_rdl2::rdl2::TYPE_RGB:
            {
            TypedAttributeKey<scene_rdl2::math::Color> typedKey(key);
            intersection.setAttribute(typedKey,
                getAttribute(typedKey, baseFaceId, tessFaceId, vIndex));
            }
            break;
        case scene_rdl2::rdl2::TYPE_VEC2F:
            {
            TypedAttributeKey<Vec2f> typedKey(key);
            intersection.setAttribute(typedKey,
                getAttribute(typedKey, baseFaceId, tessFaceId, vIndex));
            }
            break;
        case scene_rdl2::rdl2::TYPE_VEC3F:
            {
            TypedAttributeKey<Vec3f> typedKey(key);
            intersection.setAttribute(typedKey,
                getAttribute(typedKey, baseFaceId, tessFaceId, vIndex));
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
                getAttribute(typedKey, baseFaceId, tessFaceId, vIndex));
            }
            break;
        case scene_rdl2::rdl2::TYPE_RGB:
            {
            TypedAttributeKey<scene_rdl2::math::Color> typedKey(key);
            intersection.setAttribute(typedKey,
                getAttribute(typedKey, baseFaceId, tessFaceId, vIndex));
            }
            break;
        case scene_rdl2::rdl2::TYPE_VEC2F:
            {
            TypedAttributeKey<Vec2f> typedKey(key);
            intersection.setAttribute(typedKey,
                getAttribute(typedKey, baseFaceId, tessFaceId, vIndex));
            }
            break;
        case scene_rdl2::rdl2::TYPE_VEC3F:
            {
            TypedAttributeKey<Vec3f> typedKey(key);
            intersection.setAttribute(typedKey,
                getAttribute(typedKey, baseFaceId, tessFaceId, vIndex));
            }
            break;
        default:
            break;
        }
    }
}

void
TriMesh::computeAttributesDerivatives(const AttributeTable* table,
        const Vec2f& st1, const Vec2f& st2, const Vec2f& st3,
        int baseFaceId, int tessFaceId, float time, bool isFirst,
        Intersection& intersection) const
{
    std::array<float, 4> invA = {1.0f, 0.0f, 0.0f, 1.0f};
    computeStInverse(st1, st2, st3, invA);
    Attributes* attrs = getAttributes();
    for (auto key: table->getDifferentialAttributes()) {
        if (!attrs->isSupported(key)) {
            continue;
        }
        switch (key.getType()) {
        case scene_rdl2::rdl2::TYPE_FLOAT:
            {
            TypedAttributeKey<float> typedKey(key);
            float f1, f2, f3;
            getTriangleAttributes(typedKey, baseFaceId, tessFaceId,
                f1, f2, f3, time, isFirst);
            computeDerivatives(typedKey, f1, f2, f3, invA, intersection);
            break;
            }
        case scene_rdl2::rdl2::TYPE_RGB:
            {
            TypedAttributeKey<scene_rdl2::math::Color> typedKey(key);
            scene_rdl2::math::Color f1, f2, f3;
            getTriangleAttributes(typedKey, baseFaceId, tessFaceId,
                f1, f2, f3, time, isFirst);
            computeDerivatives(typedKey, f1, f2, f3, invA, intersection);
            break;
            }
        case scene_rdl2::rdl2::TYPE_VEC2F:
            {
            TypedAttributeKey<Vec2f> typedKey(key);
            Vec2f f1, f2, f3;
            getTriangleAttributes(typedKey, baseFaceId, tessFaceId,
                f1, f2, f3, time, isFirst);
            computeDerivatives(typedKey, f1, f2, f3, invA, intersection);
            break;
            }
        case scene_rdl2::rdl2::TYPE_VEC3F:
            {
            TypedAttributeKey<Vec3f> typedKey(key);
            Vec3f f1, f2, f3;
            getTriangleAttributes(typedKey, baseFaceId, tessFaceId,
                f1, f2, f3, time, isFirst);
            computeDerivatives(typedKey, f1, f2, f3, invA, intersection);
            break;
            }
        default:
            break;
        }
    }
}

template <typename T> T
TriMesh::getAttribute(const TypedAttributeKey<T>& key,
        int baseFaceId, int tessFaceId, int vIndex, float time) const
{
    const Attributes* attributes = getAttributes();
    MNRY_ASSERT(attributes->isSupported(key));

    T result;
    AttributeRate rate = attributes->getRate(key);
    switch (rate) {
    case RATE_CONSTANT:
        result = getConstantAttribute(key, time);
        break;
    case RATE_UNIFORM:
        result = getUniformAttribute(key, baseFaceId, time);
        break;
    case RATE_VARYING:
        {
        if (mIsTessellated) {
            size_t offset = sQuadVertexCount * tessFaceId;
            Vec2f uv = mFaceVaryingUv[offset + vIndex];
            size_t baseOffset = sTriangleVertexCount * baseFaceId;
            T a1 = getVaryingAttribute(
                key, mBaseIndices[baseOffset + 0], time);
            T a2 = getVaryingAttribute(
                key, mBaseIndices[baseOffset + 1], time);
            T a3 = getVaryingAttribute(
                key, mBaseIndices[baseOffset + 2], time);
            result = (1.0f - uv[0] - uv[1]) * a1 + uv[0] * a2 + uv[1] * a3;
        } else {
            result = getVaryingAttribute(
                key, mIndices[sTriangleVertexCount * baseFaceId + vIndex], time);
        }
        }
        break;
    case RATE_FACE_VARYING:
        {
        if (mIsTessellated) {
            size_t tessOffset = sQuadVertexCount * tessFaceId;
            Vec2f uv = mFaceVaryingUv[tessOffset + vIndex];
            T a1 = getFaceVaryingAttribute(key, baseFaceId, 0, time);
            T a2 = getFaceVaryingAttribute(key, baseFaceId, 1, time);
            T a3 = getFaceVaryingAttribute(key, baseFaceId, 2, time);
            result = (1.0f - uv[0] - uv[1]) * a1 + uv[0] * a2 + uv[1] * a3;
        } else {
            result = getFaceVaryingAttribute(key, baseFaceId, vIndex, time);
        }
        }
        break;
    case RATE_VERTEX:
        {
        if (mIsTessellated) {
            result = getVertexAttribute(
                key, mIndices[sQuadVertexCount * tessFaceId + vIndex], time);
        } else {
            result = getVertexAttribute(
                key, mIndices[sTriangleVertexCount * baseFaceId + vIndex], time);
        }
        }
        break;
    default:
        MNRY_ASSERT(false, "unknown attribute rate");
        break;
    }
    return result;
}

template <typename T> bool
TriMesh::getTriangleAttributes(const TypedAttributeKey<T>& key,
        int baseFaceId, int tessFaceId, T& v1, T& v2, T& v3,
        float time, bool isFirst) const
{
    int index1, index2, index3;
    if (mIsTessellated) {
        if (isFirst) {
            index1 = 0;
            index2 = 1;
            index3 = 3;
        } else {
            index1 = 2;
            index2 = 3;
            index3 = 1;
        }
    } else {
        index1 = 0;
        index2 = 1;
        index3 = 2;
    }
    const Attributes* attributes = getAttributes();
    if (attributes->isSupported(key)) {
        AttributeRate rate = attributes->getRate(key);
        switch (rate) {
        case RATE_CONSTANT:
            v3 = v2 = v1 = getConstantAttribute(key, time);
            break;
        case RATE_UNIFORM:
            v3 = v2 = v1 = getUniformAttribute(key, baseFaceId, time);
            break;
        case RATE_VARYING:
            {
            if (mIsTessellated) {
                size_t offset = sQuadVertexCount * tessFaceId;
                Vec2f uv1 = mFaceVaryingUv[offset + index1];
                Vec2f uv2 = mFaceVaryingUv[offset + index2];
                Vec2f uv3 = mFaceVaryingUv[offset + index3];
                size_t baseOffset = sTriangleVertexCount * baseFaceId;
                T a1 = getVaryingAttribute(
                    key, mBaseIndices[baseOffset + 0], time);
                T a2 = getVaryingAttribute(
                    key, mBaseIndices[baseOffset + 1], time);
                T a3 = getVaryingAttribute(
                    key, mBaseIndices[baseOffset + 2], time);
                v1 = (1.0f - uv1[0] - uv1[1]) * a1 +
                    uv1[0] * a2 + uv1[1] * a3;
                v2 = (1.0f - uv2[0] - uv2[1]) * a1 +
                    uv2[0] * a2 + uv2[1] * a3;
                v3 = (1.0f - uv3[0] - uv3[1]) * a1 +
                    uv3[0] * a2 + uv3[1] * a3;
            } else {
                size_t offset = sTriangleVertexCount * baseFaceId;
                v1 = getVaryingAttribute(
                    key, mIndices[offset + index1], time);
                v2 = getVaryingAttribute(
                    key, mIndices[offset + index2], time);
                v3 = getVaryingAttribute(
                    key, mIndices[offset + index3], time);
            }
            }
            break;
        case RATE_FACE_VARYING:
            {
            if (mIsTessellated) {
                size_t tessOffset = sQuadVertexCount * tessFaceId;
                Vec2f uv1 = mFaceVaryingUv[tessOffset + index1];
                Vec2f uv2 = mFaceVaryingUv[tessOffset + index2];
                Vec2f uv3 = mFaceVaryingUv[tessOffset + index3];
                T a1 = getFaceVaryingAttribute(key, baseFaceId, 0, time);
                T a2 = getFaceVaryingAttribute(key, baseFaceId, 1, time);
                T a3 = getFaceVaryingAttribute(key, baseFaceId, 2, time);
                v1 = (1.0f - uv1[0] - uv1[1]) * a1 +
                    uv1[0] * a2 + uv1[1] * a3;
                v2 = (1.0f - uv2[0] - uv2[1]) * a1 +
                    uv2[0] * a2 + uv2[1] * a3;
                v3 = (1.0f - uv3[0] - uv3[1]) * a1 +
                    uv3[0] * a2 + uv3[1] * a3;
            } else {
                v1 = getFaceVaryingAttribute(key, baseFaceId, 0, time);
                v2 = getFaceVaryingAttribute(key, baseFaceId, 1, time);
                v3 = getFaceVaryingAttribute(key, baseFaceId, 2, time);
            }
            }
            break;
        case RATE_VERTEX:
            {
            size_t offset = mIsTessellated ?
                sQuadVertexCount * tessFaceId :
                sTriangleVertexCount * tessFaceId;
            v1 = getVertexAttribute(key, mIndices[offset + index1], time);
            v2 = getVertexAttribute(key, mIndices[offset + index2], time);
            v3 = getVertexAttribute(key, mIndices[offset + index3], time);
            }
            break;
        default:
            MNRY_ASSERT(false, "unknown attribute rate");
            break;
        }
        return true;
    }
    return false;
}

void
TriMesh::getTriangleST(int baseFaceId, int tessFaceId,
        Vec2f& st1, Vec2f& st2, Vec2f& st3, bool isFirst) const
{
    if (!getTriangleAttributes(StandardAttributes::sSurfaceST,
        baseFaceId, tessFaceId, st1, st2, st3, 0.0f, isFirst)) {
        if (mIsTessellated) {
            if (isFirst) {
                st1 = Vec2f(0.0f, 0.0f);
                st2 = Vec2f(1.0f, 0.0f);
                st3 = Vec2f(0.0f, 1.0f);
            } else {
                st1 = Vec2f(1.0f, 1.0f);
                st2 = Vec2f(0.0f, 1.0f);
                st3 = Vec2f(1.0f, 0.0f);
            }
        } else {
            st1 = Vec2f(0.0f, 0.0f);
            st2 = Vec2f(1.0f, 0.0f);
            st3 = Vec2f(1.0f, 1.0f);
        }
    }
}

void
TriMesh::getTriangleNormal(int baseFaceId, int tessFaceId,
        Vec3f &n1, Vec3f &n2, Vec3f &n3, float time, bool isFirst) const
{
    if (!getTriangleAttributes(StandardAttributes::sNormal,
        baseFaceId, tessFaceId, n1, n2, n3, time, isFirst)) {
        // lacking any better choice, we'll compute a normal using
        // the cross product of the triangle verts
        uint32_t id1, id2, id3;
        if (mIsTessellated) {
            if (isFirst) {
                id1 = mIndices[tessFaceId * sQuadVertexCount];
                id2 = mIndices[tessFaceId * sQuadVertexCount + 1];
                id3 = mIndices[tessFaceId * sQuadVertexCount + 3];
            } else {
                id1 = mIndices[tessFaceId * sQuadVertexCount + 2];
                id2 = mIndices[tessFaceId * sQuadVertexCount + 3];
                id3 = mIndices[tessFaceId * sQuadVertexCount + 1];
            }
        } else {
            id1 = mIndices[tessFaceId * sTriangleVertexCount];
            id2 = mIndices[tessFaceId * sTriangleVertexCount + 1];
            id3 = mIndices[tessFaceId * sTriangleVertexCount + 2];
        }
        const Vec3f pos1 = mVertices(id1).asVec3f();
        const Vec3f pos2 = mVertices(id2).asVec3f();
        const Vec3f pos3 = mVertices(id3).asVec3f();
        n1 = normalize(scene_rdl2::math::cross(pos1 - pos2, pos3 - pos2));
        n2 = n3 = n1;
    }
}

} // namespace internal
} // namespace geom
} // namespace moonray


