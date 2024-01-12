// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file Box.cc
/// $Id$
///

#include "Box.h"

#include <moonray/rendering/geom/prim/BVHUserData.h>
#include <moonray/rendering/geom/prim/Util.h>

#include <moonray/rendering/bvh/shading/Attributes.h>
#include <moonray/rendering/bvh/shading/AttributeTable.h>
#include <moonray/rendering/bvh/shading/Interpolator.h>
#include <moonray/rendering/bvh/shading/RootShader.h>

namespace moonray {
namespace geom {
namespace internal {

using namespace shading;

// Needed to check on which face of the box a ray intersects
static float constexpr sEpsilonEdge = 1e-5f;

Box::Box(float length,
         float width,
         float height,
         LayerAssignmentId&& layerAssignmentId,
         PrimitiveAttributeTable&& primitiveAttributeTable)
: NamedPrimitive(std::move(layerAssignmentId))
, mL2P(scene_rdl2::math::one)
, mP2L(scene_rdl2::math::one)
, mLength(scene_rdl2::math::abs(length))
, mWidth(scene_rdl2::math::abs(width))
, mHeight(scene_rdl2::math::abs(height))
, mIsSingleSided(false)
, mIsNormalReversed(false)
{
    MNRY_ASSERT_REQUIRE(
        mLayerAssignmentId.getType() == LayerAssignmentId::Type::CONSTANT);
    // facevarying/varying/vertex are the same for Box:
    // bilinear interpolation on the intersection uv across 4 specified values
    for (auto& kv : primitiveAttributeTable) {
        for (auto& attribute : kv.second) {
            if (attribute->getRate() == RATE_VARYING ||
                attribute->getRate() == RATE_FACE_VARYING) {
                attribute->setRate(RATE_VERTEX);
            }
        }
    }
    setAttributes(Attributes::interleave(primitiveAttributeTable,
        0, 1, 4, std::vector<size_t>(), 4));
}

const scene_rdl2::rdl2::Material*
Box::getIntersectionMaterial(const scene_rdl2::rdl2::Layer* pRdlLayer,
        const mcrt_common::Ray& ray) const
{
    // there is only one part for Box, so the LayerAssignmentId should
    // always be CONSTANT
    int assignmentId = mLayerAssignmentId.getConstId();
    MNRY_ASSERT(assignmentId > -1);
    const scene_rdl2::rdl2::Material* pMaterial = MNRY_VERIFY(pRdlLayer->lookupMaterial(assignmentId));
    return pMaterial;
}

void
Box::postIntersect(mcrt_common::ThreadLocalState& tls,
        const scene_rdl2::rdl2::Layer* pRdlLayer, const mcrt_common::Ray& ray,
        Intersection& intersection) const
{
    intersection.setLayerAssignments(mLayerAssignmentId.getConstId(), pRdlLayer);

    const scene_rdl2::rdl2::Material* material = intersection.getMaterial();
    const AttributeTable *table =
        material->get<shading::RootShader>().getAttributeTable();
    intersection.setTable(&tls.mArena, table);
    intersection.setIds(ray.primID, 0, 0);
    overrideInstanceAttrs(ray, intersection);

    const Vec3f& nLocal = ray.Ng;
    Vec3f Ng = normalize(scene_rdl2::math::transformNormal(mP2L, nLocal));

    // calculate st, dpds, dpdt
    Vec2f st;
    Vec3f dpds;
    Vec3f dpdt;

    if (nLocal.x != 0.0) { // left or right face
        dpds = scene_rdl2::math::transformVector(mL2P, Vec3f(0.0, mHeight, 0.0));
        dpdt = scene_rdl2::math::transformVector(mL2P, Vec3f(0.0, 0.0, mWidth));
    } else if (nLocal.y != 0.0) { // top or bottom face
        dpds = scene_rdl2::math::transformVector(mL2P, Vec3f(mLength, 0.0, 0.0));
        dpdt = scene_rdl2::math::transformVector(mL2P, Vec3f(0.0, 0.0, mWidth));
    } else { // back or front face
        dpds = scene_rdl2::math::transformVector(mL2P, Vec3f(mLength, 0.0, 0.0));
        dpdt = scene_rdl2::math::transformVector(mL2P, Vec3f(0.0, mHeight, 0.0));
    }
    st = Vec2f(ray.u, ray.v);

    if (mIsNormalReversed) {
        Ng = -Ng;
    }
    intersection.setDifferentialGeometry(Ng, Ng, st, dpds, dpdt, true);

    // interpolate primitive attributes
    QuadricInterpolator interpolator(getAttributes(), ray.time, ray.u, ray.v);
    intersection.setRequiredAttributes(interpolator);

    const scene_rdl2::rdl2::Geometry* geometry = intersection.getGeometryObject();
    MNRY_ASSERT(geometry != nullptr);
    intersection.setEpsilonHint( geometry->getRayEpsilon() );

    // wireframe AOV
    if (table->requests(StandardAttributes::sNumPolyVertices)) {
        intersection.setAttribute(StandardAttributes::sNumPolyVertices, 4);
    }
    if (table->requests(StandardAttributes::sPolyVertexType)) {
        intersection.setAttribute(StandardAttributes::sPolyVertexType,
                                  static_cast<int>(StandardAttributes::POLYVERTEX_TYPE_POLYGON));
    }
    // When the box is not an instancing object,
    // mL2P stands for local space to render space, which
    // transforms box center from origin to world space, then
    // from world space to render space.
    // When the box is an instancing object,
    // mL2P stands for local space to parent space,
    // where parent space is then instanced around and ray.ext.l2r transforms
    // instance to render space.
    const Mat43 transform = ray.isInstanceHit() ? mL2P * ray.ext.l2r : mL2P;
    const Vec3f lower = getMinCorner();
    const Vec3f upper = getMaxCorner();
    const Vec3f verts[8] = {
        Vec3f(lower.x, lower.y, lower.z),   //     2---------3
        Vec3f(upper.x, lower.y, lower.z),   //    /|        /|
        Vec3f(lower.x, upper.y, lower.z),   //   6---------7 |
        Vec3f(upper.x, upper.y, lower.z),   //   | |       | |
        Vec3f(lower.x, lower.y, upper.z),   //   | |       | |
        Vec3f(upper.x, lower.y, upper.z),   //   | 0-------|-1
        Vec3f(lower.x, upper.y, upper.z),   //   |/        |/
        Vec3f(upper.x, upper.y, upper.z),   //   4---------5
    };
    const int indices[7][4] = {
        {3, 7, 5, 1},   // right face
        {6, 2, 0, 4},   // left face
        {3, 2, 6, 7},   // top face
        {4, 0, 1, 5},   // bottom face
        {7, 6, 4, 5},   // front face
        {3, 1, 0, 2},   // back face
        {0, 0, 0, 0}    // degenerate face used in case nLocal has a nonsensical value
    };
    int iFace = scene_rdl2::math::isEqual(nLocal.x,  1.0f) ? 0 :
                scene_rdl2::math::isEqual(nLocal.x, -1.0f) ? 1 :
                scene_rdl2::math::isEqual(nLocal.y,  1.0f) ? 2 :
                scene_rdl2::math::isEqual(nLocal.y, -1.0f) ? 3 :
                scene_rdl2::math::isEqual(nLocal.z,  1.0f) ? 4 :
                scene_rdl2::math::isEqual(nLocal.z, -1.0f) ? 5 : 6;
    for (int iVert = 0; iVert < 4; iVert++) {
        if (table->requests(StandardAttributes::sPolyVertices[iVert])) {
            int index = indices[iFace][iVert];
            Vec3f v = scene_rdl2::math::transformPoint(transform, verts[index]);
            intersection.setAttribute(StandardAttributes::sPolyVertices[iVert], v);
        }
    }

    // motion vectors
    if (table->requests(StandardAttributes::sMotion)) {
        // box has no motion slices, primitive motion can only come from instancing
        Vec3f motion(0.0f);
        if (ray.isInstanceHit()) {
            // Motion vectors only support a single instance level, hence we only care
            // about ray.instance0.
            motion = computePrimitiveMotion(intersection.getP(), nullptr, ray.getTime(),
                static_cast<const Instance *>(ray.ext.instance0OrLight));
        }
        intersection.setAttribute(StandardAttributes::sMotion, motion);
    }
}

BBox3f
Box::computeAABB() const
{
    BBox3f localBound(getMinCorner(), getMaxCorner());
    return scene_rdl2::math::transformBounds(getL2P(), localBound);
}

static void
boundsFunc(const RTCBoundsFunctionArguments* args)
{
    const BVHUserData* userData = (const BVHUserData*)args->geometryUserPtr;
    const Box* box = (const Box*)userData->mPrimitive;
    RTCBounds* output = args->bounds_o;
    BBox3f bound = box->computeAABB();
    output->lower_x = bound.lower.x;
    output->lower_y = bound.lower.y;
    output->lower_z = bound.lower.z;
    output->upper_x = bound.upper.x;
    output->upper_y = bound.upper.y;
    output->upper_z = bound.upper.z;
}

RTCBoundsFunction
Box::getBoundsFunction() const
{
    return &boundsFunc;
}

static bool
bboxTest(const Vec3f& org, const Vec3f& dir, float rayTnear, float rayTfar,
        const Vec3f& pMin, const Vec3f& pMax,
        bool isSingleSided, bool isNormalReversed, float& tHit)
{
    float t0 = scene_rdl2::math::neg_inf;
    float t1 = scene_rdl2::math::pos_inf;
    bool hitBBox = true;
    // update t interval for each plane of box
    // not that we don't need to verify whether dir[i] == 0.
    // if it is 0, the invDir will hold an infinite value, either
    // -inf or inf, and the rest of the algorithm still works correctly
    // (this assumes that the architecture being used supports
    // IEEE floating-point arithmetic)
    for (size_t i = 0; i < 3; i++) {
        float invDir = 1.0 / dir[i];
        float tnear = (pMin[i] - org[i]) * invDir;
        float tfar = (pMax[i] - org[i]) * invDir;
        if (tnear > tfar) {
            std::swap(tnear, tfar);
        }
        if (t0 > tfar || t1 < tnear) {
            hitBBox = false;
            break;
        }
        t0 = tnear > t0 ? tnear : t0;
        t1 = tfar < t1 ? tfar : t1;
    }
    // compute intersection distance along ray
    if (!hitBBox || t0 > rayTfar || t1 < rayTnear) {
        return false;
    }
    tHit = t0;
    if (t0 < rayTnear || (isSingleSided && isNormalReversed)) {
        tHit = t1;
        if (tHit > rayTfar || (isSingleSided && !isNormalReversed)) {
            return false;
        }
    }
    return true;
}

static void
intersectFunc(const RTCIntersectFunctionNArguments* args)
{
    int* valid = (int*)args->valid;
    unsigned int N = args->N;
    RTCRayHitN* rayhit = (RTCRayHitN*)args->rayhit;
    RTCRayN* rays = RTCRayHitN_RayN(rayhit, N);
    RTCHitN* hits = RTCRayHitN_HitN(rayhit, N);
    const BVHUserData* userData = (const BVHUserData*)args->geometryUserPtr;
    const Box* box = (const Box*)userData->mPrimitive;
    // get sidedness and reverse normals
    bool isSingleSided = box->getIsSingleSided();
    bool isNormalReversed = box->getIsNormalReversed();
    Vec3f pMin = box->getMinCorner();
    Vec3f pMax = box->getMaxCorner();
    const Mat43& P2L = box->getP2L();
    for (unsigned int index = 0; index < N; ++index) {
        if (valid[index] == 0) {
            continue;
        }
        // transform ray to object space
        Vec3f org = scene_rdl2::math::transformPoint(P2L, Vec3f(
            RTCRayN_org_x(rays, N, index),
            RTCRayN_org_y(rays, N, index),
            RTCRayN_org_z(rays, N, index)));
        Vec3f dir = scene_rdl2::math::transformVector(P2L, Vec3f(
            RTCRayN_dir_x(rays, N, index),
            RTCRayN_dir_y(rays, N, index),
            RTCRayN_dir_z(rays, N, index)));
        float& rayTnear = RTCRayN_tnear(rays, N, index);
        float& rayTfar = RTCRayN_tfar(rays, N, index);
        float tHit;
        if (!bboxTest(org, dir, rayTnear, rayTfar, pMin, pMax,
            isSingleSided, isNormalReversed, tHit)) {
            continue;
        }
        // compute hit position
        Vec3f pHit = org + tHit * dir;
        // local space normal will be transformed
        // to parent space at postIntersect
        if (pHit.x < pMin.x * (1 - sEpsilonEdge) ||
            pHit.x > pMax.x * (1 - sEpsilonEdge)) { // left or right face
            RTCHitN_Ng_x(hits, N, index) = scene_rdl2::math::sign(pHit.x);
            RTCHitN_Ng_y(hits, N, index) = 0.0f;
            RTCHitN_Ng_z(hits, N, index) = 0.0f;
            RTCHitN_u(hits, N, index) = (pHit.y - pMin.y) / (pMax.y - pMin.y);
            RTCHitN_v(hits, N, index) = (pHit.z - pMin.z) / (pMax.z - pMin.z);
        } else if (pHit.y < pMin.y * (1 - sEpsilonEdge) ||
                   pHit.y > pMax.y * (1 - sEpsilonEdge)) { // top or bottom face
            RTCHitN_Ng_x(hits, N, index) = 0.0f;
            RTCHitN_Ng_y(hits, N, index) = scene_rdl2::math::sign(pHit.y);
            RTCHitN_Ng_z(hits, N, index) = 0.0f;
            RTCHitN_u(hits, N, index) = (pHit.x - pMin.x) / (pMax.x - pMin.x);
            RTCHitN_v(hits, N, index) = (pHit.z - pMin.z) / (pMax.z - pMin.z);
        } else { // front or back face
            RTCHitN_Ng_x(hits, N, index) = 0.0f;
            RTCHitN_Ng_y(hits, N, index) = 0.0f;
            RTCHitN_Ng_z(hits, N, index) = scene_rdl2::math::sign(pHit.z);
            RTCHitN_u(hits, N, index) = (pHit.x - pMin.x) / (pMax.x - pMin.x);
            RTCHitN_v(hits, N, index) = (pHit.y - pMin.y) / (pMax.y - pMin.y);
        }
        // TODO call intersect filter functions
        rayTfar = tHit;
        RTCHitN_instID(hits, N, index, 0) = args->context->instID[0];
        RTCHitN_geomID(hits, N, index) = box->getGeomID();
        RTCHitN_primID(hits, N, index) = 0;
    }
}

RTCIntersectFunctionN
Box::getIntersectFunction() const
{
    return &intersectFunc;
}

static void
occludedFunc(const RTCOccludedFunctionNArguments* args)
{
    int* valid = (int*)args->valid;
    unsigned int N = args->N;
    RTCRayN* rays = args->ray;
    const BVHUserData* userData = (const BVHUserData*)args->geometryUserPtr;
    const Box* box = (const Box*)userData->mPrimitive;
    // get sidedness and reverse normals
    bool isSingleSided = box->getIsSingleSided();
    bool isNormalReversed = box->getIsNormalReversed();
    Vec3f pMin = box->getMinCorner();
    Vec3f pMax = box->getMaxCorner();
    const Mat43& P2L = box->getP2L();
    for (unsigned int index = 0; index < N; ++index) {
        if (valid[index] == 0) {
            continue;
        }
        // transform ray to object space
        Vec3f org = scene_rdl2::math::transformPoint(P2L, Vec3f(
            RTCRayN_org_x(rays, N, index),
            RTCRayN_org_y(rays, N, index),
            RTCRayN_org_z(rays, N, index)));
        Vec3f dir = scene_rdl2::math::transformVector(P2L, Vec3f(
            RTCRayN_dir_x(rays, N, index),
            RTCRayN_dir_y(rays, N, index),
            RTCRayN_dir_z(rays, N, index)));
        float& rayTnear = RTCRayN_tnear(rays, N, index);
        float& rayTfar = RTCRayN_tfar(rays, N, index);
        float tHit;
        if (!bboxTest(org, dir, rayTnear, rayTfar, pMin, pMax,
            isSingleSided, isNormalReversed, tHit)) {
            continue;
        }
        // TODO call occlude filter functions
        // mark the tfar negative is the official signal
        // for embree that the ray is occluded
        rayTfar = -FLT_MAX;
    }
}

RTCOccludedFunctionN
Box::getOccludedFunction() const
{
    return &occludedFunc;
}

} // namespace internal
} // namespace geom
} // namespace moonray

