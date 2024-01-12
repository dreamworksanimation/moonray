// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file QuadMesh.h
/// $Id$
///

#include "QuadMesh.h"

#include <moonray/rendering/geom/prim/MeshTessellationUtil.h>

#include <moonray/rendering/bvh/shading/AttributeKey.h>
#include <moonray/rendering/bvh/shading/Attributes.h>
#include <moonray/rendering/bvh/shading/PrimitiveAttribute.h>
#include <moonray/rendering/bvh/shading/RootShader.h>
#include <moonray/rendering/mcrt_common/ThreadLocalState.h>
#include <scene_rdl2/common/math/MathUtil.h>
#include <scene_rdl2/common/math/ReferenceFrame.h>
#include <scene_rdl2/render/logging/logging.h>

namespace moonray {
namespace geom {
namespace internal {

using namespace shading;

QuadMesh::QuadMesh(size_t estiFaceCount,
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
QuadMesh::bakePosMap(int width, int height, int udim,
        TypedAttributeKey<Vec2f> stKey,
        Vec3fa *posResult, Vec3f *nrmResult) const
{
    const Attributes *primitiveAttributes = getAttributes();
    MNRY_ASSERT_REQUIRE(primitiveAttributes->isSupported(stKey));
    // throw out the anchor boys....
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
        getQuadAttributes(stKey, baseFaceId, faceId, st0, st1, st3,
            0.0f, /* isFirst = */ true);
        getQuadAttributes(stKey, baseFaceId, faceId, st2, st3, st1,
            0.0f, /* isFrist = */ false);

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
                getQuadNormal(baseFaceId, faceId, nrmData0, nrmData1, nrmData3,
                    0.0f, /* isFirst = */ true);
                getQuadNormal(baseFaceId, faceId, nrmData2, nrmData3, nrmData1,
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

    return true;
}

void
QuadMesh::setRequiredAttributes(int primId, float time, float u, float v,
    float w, bool isFirst, Intersection& intersection) const
{
    int tessFaceId = primId;
    int baseFaceId = getBaseFaceId(tessFaceId);
    // In Embree, quad is internally handled as a pair of
    // two triangles (v0,v1,v3) and (v2,v3,v1).
    // The (u',v') coordinates of the second triangle
    // is corrected by u = 1-u' and v = 1-v' to produce
    // a quad parametrization where u and v go from 0 to 1.
    // That's to say, if u+v > 1,
    // intersection happens in the second triangle,
    // and (u',v') in the second triangle should be (1-u, 1-v)
    // In order to get face varying attributes by offset,
    // we still need to keep the unused vertex,
    // but set its weight to 0 to remove its influence
    uint32_t id1 = mIndices[sQuadVertexCount * tessFaceId    ];
    uint32_t id2 = mIndices[sQuadVertexCount * tessFaceId + 1];
    uint32_t id3 = mIndices[sQuadVertexCount * tessFaceId + 2];
    uint32_t id4 = mIndices[sQuadVertexCount * tessFaceId + 3];
    uint32_t isecId1, isecId2, isecId3;
    uint32_t varyingId1, varyingId2, varyingId3, varyingId4;
    // weights for interpolator
    float varyingW[4];
    float vertexW[4];

    if (isFirst) {
        // Triangle (0,1,3)
        vertexW[0] = w;
        vertexW[1] = u;
        vertexW[2] = 0.0f;
        vertexW[3] = v;
        intersection.setIds(id1, id2, id4);
    } else {
        // Triangle (2,3,1)
        vertexW[0] = 0.0f;
        vertexW[1] = v;
        vertexW[2] = w;
        vertexW[3] = u;
        intersection.setIds(id3, id4, id2);
    }

    if (mIsTessellated) {
        // the surface uv of intersection point on base face
        Vec2f uv =
            vertexW[0] * mFaceVaryingUv[sQuadVertexCount * tessFaceId    ] +
            vertexW[1] * mFaceVaryingUv[sQuadVertexCount * tessFaceId + 1] +
            vertexW[2] * mFaceVaryingUv[sQuadVertexCount * tessFaceId + 2] +
            vertexW[3] * mFaceVaryingUv[sQuadVertexCount * tessFaceId + 3];
        // bilinear interpolation weight
        varyingW[0] = (1.0f - uv[0]) * (1.0f - uv[1]);
        varyingW[1] = (       uv[0]) * (1.0f - uv[1]);
        varyingW[2] = (       uv[0]) * (       uv[1]);
        varyingW[3] = (1.0f - uv[0]) * (       uv[1]);
        varyingId1 = mBaseIndices[sQuadVertexCount * baseFaceId    ];
        varyingId2 = mBaseIndices[sQuadVertexCount * baseFaceId + 1];
        varyingId3 = mBaseIndices[sQuadVertexCount * baseFaceId + 2];
        varyingId4 = mBaseIndices[sQuadVertexCount * baseFaceId + 3];
    } else {
        varyingW[0] = vertexW[0];
        varyingW[1] = vertexW[1];
        varyingW[2] = vertexW[2];
        varyingW[3] = vertexW[3];
        varyingId1 = id1;
        varyingId2 = id2;
        varyingId3 = id3;
        varyingId4 = id4;
    }

    int partId = (mFaceToPart.size() > 0) ? mFaceToPart[baseFaceId] : 0;
    const Attributes* primitiveAttributes = getAttributes();
    // Interpolate PrimitiveAttribute
    // Interpolator get attributes by offsetting indices,
    // directly passing sub triangle vertex id doesn't guarantee
    // To support face varying attributes for quad,
    // id should be passed in face index rotation order,
    // as well as four weights.
    // e.g. intersect at second triangle in face(v0-v1-v2-v3),
    //      we should pass indices (vid0, vid1, vid2, vid3),
    //      and weights (0, v, w, u)
    MeshInterpolator interpolator(primitiveAttributes,
        time, partId, baseFaceId,
        varyingId1, varyingId2, varyingId3, varyingId4,
        varyingW[0], varyingW[1], varyingW[2], varyingW[3], tessFaceId,
        id1, id2, id3, id4,
        vertexW[0], vertexW[1], vertexW[2], vertexW[3]);
    intersection.setRequiredAttributes(interpolator);

    // Add an interpolated N, dPds, and dPdt to the intersection
    // if they exist in the primitive attribute table
    setExplicitAttributes<MeshInterpolator>(interpolator,
                                            *primitiveAttributes,
                                            intersection);
}

void
QuadMesh::postIntersect(mcrt_common::ThreadLocalState& tls,
        const scene_rdl2::rdl2::Layer* pRdlLayer, const mcrt_common::Ray& ray,
        Intersection& intersection) const
{
    int primId = ray.primID;
    int tessFaceId = primId;
    int baseFaceId = getBaseFaceId(tessFaceId);

    // barycentric coordinate
    float u = ray.u;
    float v = ray.v;
    float w = 1.0f - u - v;
    bool isFirst = w > 0.0f;
    if (!isFirst) {
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

    setRequiredAttributes(primId,
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
    getQuadST(baseFaceId, tessFaceId, st1, st2, st3, isFirst);
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
            if (!computeTrianglePartialDerivatives(
                    p1, p2, p3, st1, st2, st3, dp)) {
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

    // polyon vertices
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
            uint32_t id = mIndices[4 * tessFaceId + iVert];
            // may need to move the vertices to render space
            // for instancing object since they are ray traced in local space
            const Vec3f v = ray.isInstanceHit() ? transformPoint(ray.ext.l2r, mVertices(id).asVec3f())
                                                : mVertices(id).asVec3f();
            intersection.setAttribute(StandardAttributes::sPolyVertices[iVert], v);
        }
    }

    // motion vectors
    if (table->requests(StandardAttributes::sMotion)) {
        const Vec3f motion = computeMotion(mVertices, isecId1, isecId2, isecId3,
            w, u, v, ray);
        intersection.setAttribute(StandardAttributes::sMotion, motion);
    }
}

bool
QuadMesh::computeIntersectCurvature(const mcrt_common::Ray& ray,
        const Intersection& intersection, Vec3f& dnds, Vec3f& dndt) const
{
    int tessFaceId = ray.primID;
    int baseFaceId = getBaseFaceId(tessFaceId);
    uint32_t id1, id2, id3;
    intersection.getIds(id1, id2, id3);
    bool isFirst = ray.u + ray.v <= 1.0f;
    Vec2f st1, st2, st3;
    getQuadST(baseFaceId, tessFaceId, st1, st2, st3, isFirst);
    Vec3f n1, n2, n3;
    getQuadNormal(baseFaceId, tessFaceId, n1, n2, n3, ray.time, isFirst);
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
QuadMesh::getST(int tessFaceId, float u, float v, Vec2f& st) const
{
    int baseFaceId = getBaseFaceId(tessFaceId);
    float w = 1.0f - u - v;
    bool isFirst = w > 0.f;
    if (!isFirst) {
        // Triangle (2,3,1)
        u = 1.0f - u;
        v = 1.0f - v;
        w = -w;
    }
    Vec2f st1, st2, st3;
    getQuadST(baseFaceId, tessFaceId, st1, st2, st3, isFirst);
    st = w * st1 + u * st2 + v * st3;
}

void
QuadMesh::getNeighborVertices(int baseFaceId, int tessFaceId,
        int tessFaceVIndex, int& vid, int& vid1, int& vid2, int& vid3,
        Vec2f& st, Vec2f& st1, Vec2f& st2, Vec2f& st3) const
{
    size_t offset = tessFaceId * sQuadVertexCount;
    if (tessFaceVIndex == 2) {
        if (mIndices[offset + 2] == mIndices[offset + 3]) {
            // the degened quad introduced during tessellation
            // first triangle on quad (0, 1, 3), which is (0, 1, 2)
            getQuadST(baseFaceId, tessFaceId, st1, st2, st3, true);
            vid1 = mIndices[offset + 0];
            vid2 = mIndices[offset + 1];
            vid3 = mIndices[offset + 2];
            vid = vid3;
            st = st3;
        } else {
            // second triangle on quad (2, 3, 1)
            getQuadST(baseFaceId, tessFaceId, st1, st2, st3, false);
            vid1 = mIndices[offset + 2];
            vid2 = mIndices[offset + 3];
            vid3 = mIndices[offset + 1];
            vid = vid1;
            st = st1;
        }
    } else {
        // first triangle on quad (0, 1, 3)
        getQuadST(baseFaceId, tessFaceId, st1, st2, st3, true);
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
}

void
QuadMesh::splitNGons(size_t outputFaceCount,
                     const PolygonMesh::VertexBuffer& vertices,
                     PolygonMesh::FaceToPartBuffer& faceToPart,
                     const PolygonMesh::FaceVertexCount& faceVertexCount,
                     PolygonMesh::IndexBuffer& indices,
                     LayerAssignmentId& layerAssignmentId,
                     PrimitiveAttributeTable& primitiveAttributeTable)
{
    size_t inputFaceCount = faceVertexCount.size();
    size_t inputIndexCount = indices.size();
    indices.resize(sQuadVertexCount * outputFaceCount);
    if (faceToPart.size() > 0) {
        faceToPart.resize(outputFaceCount);
    }

    size_t inputIndexOffset = inputIndexCount;
    size_t outputIndexOffset = indices.size() - 1;
    size_t outputF2POffset = faceToPart.size() - 1;
    // quadrangulate indices
    for (int f = inputFaceCount - 1; f >= 0; --f) {
        size_t fvCount = faceVertexCount[f];
        inputIndexOffset -= fvCount;
        size_t v = fvCount;
        // when quadragulating an ngon with odd number n,
        // there will be one quad that has a duplicated vertex
        if (fvCount & 1) {
            indices[outputIndexOffset--] = indices[inputIndexOffset + v - 1];
            indices[outputIndexOffset--] = indices[inputIndexOffset + v - 1];
            indices[outputIndexOffset--] = indices[inputIndexOffset + v - 2];
            indices[outputIndexOffset--] = indices[inputIndexOffset];
            if (faceToPart.size() > 0) {
                faceToPart[outputF2POffset--] = faceToPart[f];
            }
            v -= 1;
        }
        for (; v >= sQuadVertexCount; v -= 2) {
            indices[outputIndexOffset--] = indices[inputIndexOffset + v - 1];
            indices[outputIndexOffset--] = indices[inputIndexOffset + v - 2];
            indices[outputIndexOffset--] = indices[inputIndexOffset + v - 3];
            indices[outputIndexOffset--] = indices[inputIndexOffset];
            if (faceToPart.size() > 0) {
                faceToPart[outputF2POffset--] = faceToPart[f];
            }
        }
    }

    // quadrangulate layerAssignmentId if it's in per face frequency
    if (layerAssignmentId.getType() == LayerAssignmentId::Type::VARYING) {
        auto& faceAssignmentIds = layerAssignmentId.getVaryingId();
        faceAssignmentIds.resize(outputFaceCount);
        size_t outputFaceOffset = outputFaceCount - 1;
        for (int f = inputFaceCount - 1; f >= 0; --f) {
            size_t quadCount = (faceVertexCount[f] - 1) >> 1;
            for (int v = quadCount; v > 0; --v) {
                faceAssignmentIds[outputFaceOffset--] = faceAssignmentIds[f];
            }
        }
    }

    // quadrangulate uniform and facevarying primitive attributes
    for (auto& kv : primitiveAttributeTable) {
        for (auto& attribute : kv.second) {
            AttributeRate rate = attribute->getRate();
            if (rate == RATE_UNIFORM) {
                attribute->resize(outputFaceCount);
                size_t outputFaceOffset = outputFaceCount - 1;
                for (int f = inputFaceCount - 1; f >= 0; --f) {
                    size_t quadCount = (faceVertexCount[f] - 1) >> 1;
                    for (size_t v = quadCount; v > 0; --v) {
                        attribute->copyInPlace(f, outputFaceOffset--);
                    }
                }
            } else if (rate == RATE_FACE_VARYING) {
                attribute->resize(sQuadVertexCount * outputFaceCount);
                inputIndexOffset = inputIndexCount;
                outputIndexOffset = indices.size() - 1;
                for (int f = inputFaceCount - 1; f >= 0; --f) {
                    size_t fvCount = faceVertexCount[f];
                    inputIndexOffset -= fvCount;
                    size_t v = fvCount;
                    // when quadragulating an ngon with odd number n,
                    // there will be one quad that has a duplicated vertex
                    if (fvCount & 1) {
                        attribute->copyInPlace(inputIndexOffset + v - 1,
                            outputIndexOffset--);
                        attribute->copyInPlace(inputIndexOffset + v - 1,
                            outputIndexOffset--);
                        attribute->copyInPlace(inputIndexOffset + v - 2,
                            outputIndexOffset--);
                        attribute->copyInPlace(inputIndexOffset,
                            outputIndexOffset--);
                        v -= 1;
                    }
                    for (; v >= sQuadVertexCount; v -= 2) {
                        // quadrangulate a ngon into quad fan
                        attribute->copyInPlace(inputIndexOffset + v - 1,
                            outputIndexOffset--);
                        attribute->copyInPlace(inputIndexOffset + v - 2,
                            outputIndexOffset--);
                        attribute->copyInPlace(inputIndexOffset + v - 3,
                            outputIndexOffset--);
                        attribute->copyInPlace(inputIndexOffset,
                            outputIndexOffset--);
                    }
                }
            }
        }
    }
}

void
QuadMesh::generateIndexBufferAndSurfaceSamples(
        const std::vector<PolyFaceTopology>& quadTopologies,
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
    for (size_t f = 0; f < quadTopologies.size(); ++f) {
        const PolyFaceTopology& quadTopology = quadTopologies[f];
        // handle uniform tessellation in more optimized topology
        if (tessellatedVertexLookup.noRingToStitch(quadTopology)) {
            SCOPED_MEM(arena);
            int colVertexCount;
            int rowVertexCount;
            int * vids = nullptr;
            collectUniformTessellatedVertices(arena,
                quadTopology, f, tessellatedVertexLookup,
                vids, rowVertexCount, colVertexCount);
            float du = 1.0f / (float)(colVertexCount - 1);
            float dv = 1.0f / (float)(rowVertexCount - 1);
            int offset = 0;
            for (int i = 0; i < rowVertexCount; ++i) {
                for (int j = 0; j < colVertexCount; ++j) {
                    int vid = vids[offset++];
                    surfaceSamples[vid].mFaceId = f;
                    surfaceSamples[vid].mUv = Vec2f(j * du, i * dv);
                }
            }
            for (int i = 0; i < rowVertexCount - 1; ++i) {
                for (int j = 0; j < colVertexCount - 1; ++j) {
                    indices.push_back(vids[ i      * colVertexCount + j    ]);
                    indices.push_back(vids[ i      * colVertexCount + j + 1]);
                    indices.push_back(vids[(i + 1) * colVertexCount + j + 1]);
                    indices.push_back(vids[(i + 1) * colVertexCount + j    ]);
                    if (faceVaryingUv) {
                        faceVaryingUv->emplace_back((j    ) * du, (i    ) * dv);
                        faceVaryingUv->emplace_back((j + 1) * du, (i    ) * dv);
                        faceVaryingUv->emplace_back((j + 1) * du, (i + 1) * dv);
                        faceVaryingUv->emplace_back((j    ) * du, (i + 1) * dv);
                    }
                    tessellatedToBaseFace.push_back(f);
                }
            }
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
        float du = 1.0f / (float)(1 + interiorColVertexCount);
        float dv = 1.0f / (float)(1 + interiorRowVertexCount);
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
                if (faceVaryingUv) {
                    faceVaryingUv->emplace_back((j + 1) * du, (i + 1) * dv);
                    faceVaryingUv->emplace_back((j + 2) * du, (i + 1) * dv);
                    faceVaryingUv->emplace_back((j + 2) * du, (i + 2) * dv);
                    faceVaryingUv->emplace_back((j + 1) * du, (i + 2) * dv);
                }
                tessellatedToBaseFace.push_back(f);
            }
        }
        // surface samples
        for (int i = 0; i < interiorRowVertexCount; ++i) {
            for (int j = 0; j < interiorColVertexCount; ++j) {
                int vid = tessellatedVertexLookup.getInteriorVertexId(
                    f, i, j, interiorColVertexCount);
                surfaceSamples[vid].mFaceId = f;
                surfaceSamples[vid].mUv = Vec2f((j + 1) * du, (i + 1) * dv);
            }
        }
        // stitch interior and outer ring edge
        //     *---*--*--*---*
        //    / \ / \ | / \ / \
        //   *---*----*----*---*
        const Vec2f cornerUv[4] = {
            Vec2f(0.0f, 0.0f), Vec2f(1.0f, 0.0f),
            Vec2f(1.0f, 1.0f), Vec2f(0.0f, 1.0f)};
        for (size_t c = 0; c < sQuadVertexCount; ++c) {
            SCOPED_MEM(arena);
            StitchPoint* innerRing = nullptr;
            StitchPoint* outerRing = nullptr;
            Vec2f* innerRingUv = nullptr;
            Vec2f* outerRingUv = nullptr;
            int innerRingVertexCount, outerRingVertexCount;
            generateQuadStitchRings(arena, quadTopology, f, c,
                tessellatedVertexLookup,
                interiorRowVertexCount, interiorColVertexCount,
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
                    (       v * w) * cornerUv[(c + 1) % 4];
            }
        }
    }
}

template <typename T>
T bilinearInterpolate(const Vec2f& uv,
        const T& a0, const T& a1, const T& a2, const T& a3)
{
    return (1.0f - uv[0]) * (1.0f - uv[1]) * a0 +
           (       uv[0]) * (1.0f - uv[1]) * a1 +
           (       uv[0]) * (       uv[1]) * a2 +
           (1.0f - uv[0]) * (       uv[1]) * a3;
}

PolygonMesh::VertexBuffer
QuadMesh::generateVertexBuffer(
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
            int vid0 = baseIndices[sQuadVertexCount * fid + 0];
            int vid1 = baseIndices[sQuadVertexCount * fid + 1];
            int vid2 = baseIndices[sQuadVertexCount * fid + 2];
            int vid3 = baseIndices[sQuadVertexCount * fid + 3];
            // bilinear interpolate the new position
            Vec2f uv = surfaceSamples[i].mUv;
            for (size_t t = 0; t < motionSampleCount; ++t) {
                tessellatedVertices(i, t) = bilinearInterpolate(uv,
                    baseVertices(vid0, t), baseVertices(vid1, t),
                    baseVertices(vid2, t), baseVertices(vid3, t));
            }
        }
    });
    return tessellatedVertices;
}

void
QuadMesh::fillDisplacementAttributes(int tessFaceId, int vIndex,
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
QuadMesh::computeAttributesDerivatives(const AttributeTable* table,
        const Vec2f& st1, const Vec2f& st2, const Vec2f& st3,
        int baseFaceId, int tessFaceId, float time, bool isFirst,
        Intersection& intersection) const
{
    std::array<float , 4> invA = {1.0f, 0.0f, 0.0f, 1.0f};
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
            getQuadAttributes(typedKey, baseFaceId, tessFaceId,
                f1, f2, f3, time, isFirst);
            computeDerivatives(typedKey, f1, f2, f3, invA, intersection);
            break;
            }
        case scene_rdl2::rdl2::TYPE_RGB:
            {
            TypedAttributeKey<scene_rdl2::math::Color> typedKey(key);
            scene_rdl2::math::Color f1, f2, f3;
            getQuadAttributes(typedKey, baseFaceId, tessFaceId,
                f1, f2, f3, time, isFirst);
            computeDerivatives(typedKey, f1, f2, f3, invA, intersection);
            break;
            }
        case scene_rdl2::rdl2::TYPE_VEC2F:
            {
            TypedAttributeKey<Vec2f> typedKey(key);
            Vec2f f1, f2, f3;
            getQuadAttributes(typedKey, baseFaceId, tessFaceId,
                f1, f2, f3, time, isFirst);
            computeDerivatives(typedKey, f1, f2, f3, invA, intersection);
            break;
            }
        case scene_rdl2::rdl2::TYPE_VEC3F:
            {
            TypedAttributeKey<Vec3f> typedKey(key);
            Vec3f f1, f2, f3;
            getQuadAttributes(typedKey, baseFaceId, tessFaceId,
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
QuadMesh::getAttribute(const TypedAttributeKey<T>& key,
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
            size_t baseOffset = sQuadVertexCount * baseFaceId;
            T a1 = getVaryingAttribute(
                key, mBaseIndices[baseOffset + 0], time);
            T a2 = getVaryingAttribute(
                key, mBaseIndices[baseOffset + 1], time);
            T a3 = getVaryingAttribute(
                key, mBaseIndices[baseOffset + 2], time);
            T a4 = getVaryingAttribute(
                key, mBaseIndices[baseOffset + 3], time);
            result = bilinearInterpolate(uv, a1, a2, a3, a4);
        } else {
            result = getVaryingAttribute(
                key, mIndices[sQuadVertexCount * baseFaceId + vIndex], time);
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
            T a4 = getFaceVaryingAttribute(key, baseFaceId, 3, time);
            result = bilinearInterpolate(uv, a1, a2, a3, a4);
        } else {
            result = getFaceVaryingAttribute(key, baseFaceId, vIndex, time);
        }
        }
        break;
    case RATE_VERTEX:
        result = getVertexAttribute(
            key, mIndices[sQuadVertexCount * tessFaceId + vIndex], time);
        break;
    default:
        MNRY_ASSERT(false, "unknown attribute rate");
        break;
    }
    return result;
}

template <typename T> bool
QuadMesh::getQuadAttributes(const TypedAttributeKey<T>& key,
        int baseFaceId, int tessFaceId, T& v1, T& v2, T& v3,
        float time, bool isFirst) const
{
    int index1, index2, index3;
    if (isFirst) {
        index1 = 0;
        index2 = 1;
        index3 = 3;
    } else {
        index1 = 2;
        index2 = 3;
        index3 = 1;
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
                size_t baseOffset = sQuadVertexCount * baseFaceId;
                T a1 = getVaryingAttribute(
                    key, mBaseIndices[baseOffset + 0], time);
                T a2 = getVaryingAttribute(
                    key, mBaseIndices[baseOffset + 1], time);
                T a3 = getVaryingAttribute(
                    key, mBaseIndices[baseOffset + 2], time);
                T a4 = getVaryingAttribute(
                    key, mBaseIndices[baseOffset + 3], time);
                v1 = bilinearInterpolate(uv1, a1, a2, a3, a4);
                v2 = bilinearInterpolate(uv2, a1, a2, a3, a4);
                v3 = bilinearInterpolate(uv3, a1, a2, a3, a4);
            } else {
                size_t offset = sQuadVertexCount * baseFaceId;
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
                T a4 = getFaceVaryingAttribute(key, baseFaceId, 3, time);
                v1 = bilinearInterpolate(uv1, a1, a2, a3, a4);
                v2 = bilinearInterpolate(uv2, a1, a2, a3, a4);
                v3 = bilinearInterpolate(uv3, a1, a2, a3, a4);
            } else {
                v1 = getFaceVaryingAttribute(key, baseFaceId, index1, time);
                v2 = getFaceVaryingAttribute(key, baseFaceId, index2, time);
                v3 = getFaceVaryingAttribute(key, baseFaceId, index3, time);
            }
            }
            break;
        case RATE_VERTEX:
            {
            size_t offset = sQuadVertexCount * tessFaceId;
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
QuadMesh::getQuadST(int baseFaceId, int tessFaceId,
        Vec2f& st1, Vec2f& st2, Vec2f& st3, bool isFirst) const
{
    if (!getQuadAttributes(StandardAttributes::sSurfaceST,
        baseFaceId, tessFaceId, st1, st2, st3, 0.0f, isFirst)) {
        if (isFirst) {
            st1 = Vec2f(0.0f, 0.0f);
            st2 = Vec2f(1.0f, 0.0f);
            st3 = Vec2f(0.0f, 1.0f);
        } else {
            st1 = Vec2f(1.0f, 1.0f);
            st2 = Vec2f(0.0f, 1.0f);
            st3 = Vec2f(1.0f, 0.0f);
        }
    }
}

void
QuadMesh::getQuadNormal(int baseFaceId, int tessFaceId,
        Vec3f &n1, Vec3f &n2, Vec3f &n3, float time, bool isFirst) const
{
    if (!getQuadAttributes(StandardAttributes::sNormal,
        baseFaceId, tessFaceId, n1, n2, n3, time, isFirst)) {
        // lacking any better choice, we'll compute a normal using
        // the cross product of the triangle verts
        uint32_t id1, id2, id3;
        if (isFirst) {
            id1 = mIndices[tessFaceId * sQuadVertexCount];
            id2 = mIndices[tessFaceId * sQuadVertexCount + 1];
            id3 = mIndices[tessFaceId * sQuadVertexCount + 3];
        } else {
            id1 = mIndices[tessFaceId * sQuadVertexCount + 2];
            id2 = mIndices[tessFaceId * sQuadVertexCount + 3];
            id3 = mIndices[tessFaceId * sQuadVertexCount + 1];
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


