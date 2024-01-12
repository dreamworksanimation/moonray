// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file Sphere.cc
/// $Id$
///

#include "Sphere.h"

#include <moonray/rendering/geom/prim/BVHUserData.h>
#include <moonray/rendering/geom/prim/Util.h>

#include <moonray/rendering/bvh/shading/AttributeKey.h>
#include <moonray/rendering/bvh/shading/Attributes.h>
#include <moonray/rendering/bvh/shading/Interpolator.h>
#include <moonray/rendering/bvh/shading/RootShader.h>

namespace moonray {
namespace geom {
namespace internal {

using namespace shading;

Sphere::Sphere(float radius,
               LayerAssignmentId&& layerAssignmentId,
               PrimitiveAttributeTable&& primitiveAttributeTable)
: NamedPrimitive(std::move(layerAssignmentId))
, mL2P(scene_rdl2::math::one)
, mP2L(scene_rdl2::math::one)
, mRadius(radius)
, mPhiMax(scene_rdl2::math::sTwoPi)
, mZMin(-radius)
, mZMax(radius)
, mThetaMin(scene_rdl2::math::sPi)
, mThetaMax(0.0f)
, mIsSingleSided(false)
, mIsNormalReversed(false)
{
    MNRY_ASSERT_REQUIRE(
        mLayerAssignmentId.getType() == LayerAssignmentId::Type::CONSTANT);
    // facevarying/varying/vertex are the same for Sphere:
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
Sphere::getIntersectionMaterial(const scene_rdl2::rdl2::Layer* pRdlLayer,
        const mcrt_common::Ray& ray) const
{
    // there is only one part for sphere, so the LayerAssignmentId should
    // always be CONSTANT
    int assignmentId = mLayerAssignmentId.getConstId();
    MNRY_ASSERT(assignmentId > -1);
    const scene_rdl2::rdl2::Material* pMaterial = MNRY_VERIFY(pRdlLayer->lookupMaterial(assignmentId));
    return pMaterial;
}

void
Sphere::postIntersect(mcrt_common::ThreadLocalState& tls,
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

    // we store the local space intersect point in Ng during ray tracing time,
    // which is the geometry normal in local space. (we need this local space
    // information to compute st/dpds/dpdt here)
    const Vec3f& pLocal = ray.Ng;
    Vec3f Ng = normalize(scene_rdl2::math::transformNormal(mP2L, pLocal));
    // calculate st, dpds, dpdt
    float theta = scene_rdl2::math::acos(scene_rdl2::math::clamp(pLocal.z / mRadius, -1.0f, 1.0f));
    Vec2f st(ray.u / mPhiMax, (theta - mThetaMin) / (mThetaMax - mThetaMin));
    Vec3f dpds = scene_rdl2::math::transformVector(mL2P,
        Vec3f(-mPhiMax * pLocal.y, mPhiMax * pLocal.x, 0.0f));
    float invZRadius = 1.0f / sqrtf(pLocal.x * pLocal.x + pLocal.y * pLocal.y);
    float cosPhi = pLocal.x * invZRadius;
    float sinPhi = pLocal.y * invZRadius;
    Vec3f dpdt = scene_rdl2::math::transformVector(mL2P, (mThetaMax - mThetaMin) *
        Vec3f(pLocal.z * cosPhi, pLocal.z * sinPhi, -mRadius * sinf(theta)));
    if (mIsNormalReversed) {
        Ng = -Ng;
    }
    intersection.setDifferentialGeometry(Ng, Ng, st, dpds, dpdt, true);

    // interpolate primitive attributes
    QuadricInterpolator interpolator(getAttributes(), ray.time, st[0], st[1]);
    intersection.setRequiredAttributes(interpolator);

    const scene_rdl2::rdl2::Geometry* geometry = intersection.getGeometryObject();
    MNRY_ASSERT(geometry != nullptr);
    intersection.setEpsilonHint( geometry->getRayEpsilon() );

    // wireframe AOV is blank
    if (table->requests(StandardAttributes::sNumPolyVertices)) {
        intersection.setAttribute(StandardAttributes::sNumPolyVertices, 0);
        intersection.setAttribute(StandardAttributes::sPolyVertexType,
            static_cast<int>(StandardAttributes::POLYVERTEX_TYPE_POLYGON));
    }

    // motion vectors
    if (table->requests(StandardAttributes::sMotion)) {
        // sphere has no motion slices, primitive motion can only come from instancing
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

bool
Sphere::computeIntersectCurvature(const mcrt_common::Ray &ray,
        const Intersection &intersection,
        Vec3f &dnds, Vec3f &dndt) const
{
    // p(s, t) = n(s, t) for local space unit sphere so reuse dpds/dpdt
    // with radius scaling should be sufficient enough for sphere case
    float invRadius = 1.0f / mRadius;
    dnds = intersection.getdPds() * invRadius;
    dndt = intersection.getdPdt() * invRadius;
    return true;
}

BBox3f
Sphere::computeAABB() const
{
    float r = getRadius();
    BBox3f localBound(Vec3f(-r), Vec3f(r));
    return scene_rdl2::math::transformBounds(getL2P(), localBound);
}

static void
boundsFunc(const RTCBoundsFunctionArguments* args)
{
    const BVHUserData* userData = (const BVHUserData*)args->geometryUserPtr;
    const Sphere* sphere = (const Sphere*)userData->mPrimitive;
    RTCBounds* output = args->bounds_o;
    BBox3f bound = sphere->computeAABB();
    output->lower_x = bound.lower.x;
    output->lower_y = bound.lower.y;
    output->lower_z = bound.lower.z;
    output->upper_x = bound.upper.x;
    output->upper_y = bound.upper.y;
    output->upper_z = bound.upper.z;
}

RTCBoundsFunction
Sphere::getBoundsFunction() const
{
    return &boundsFunc;
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
    const Sphere* sphere = (const Sphere*)userData->mPrimitive;
    // get sidedness and reverse normals
    bool isSingleSided = sphere->getIsSingleSided();
    bool isNormalReversed = sphere->getIsNormalReversed();
    float radius = sphere->getRadius();
    const Mat43& P2L = sphere->getP2L();
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
        // compute quadratic sphere coefficients
        const float A = dot(dir, dir);
        const float B = 2.0f * dot(org, dir);
        const float C = dot(org, org) - scene_rdl2::math::sqr(radius);
        // solve quadratic equation for t values
        const float D = B * B - 4.0f * A * C;
        if (D < 0.0f) {
            continue;
        }
        const float rootDiscrim = scene_rdl2::math::sqrt(D);
        float q = B < 0.0f ?
            -0.5f * (B - rootDiscrim) : -0.5f * (B + rootDiscrim);
        float t0 = q / A;
        float t1 = C / q;
        if (t0 > t1) {
            std::swap(t0, t1);
        }
        // compute intersection distance along ray
        if (t0 > rayTfar || t1 < rayTnear) {
            continue;
        }
        float tHit = t0;
        if (t0 < rayTnear || (isSingleSided && isNormalReversed)) {
            tHit = t1;
            if (tHit > rayTfar || (isSingleSided && !isNormalReversed)) {
                continue;
            }
        }
        // compute hit position and phi
        Vec3f pHit = org + tHit * dir;
        if (pHit.x == 0.0f && pHit.y == 0.0f) {
            pHit.x = 1e-5f * radius;
        }
        float phi = atan2f(pHit.y, pHit.x);
        if (phi < 0.0f) {
            phi += scene_rdl2::math::sTwoPi;
        }
        float zMin = sphere->getZMin();
        float zMax = sphere->getZMax();
        float phiMax = sphere->getPhiMax();
        // test sphere intersection against clipping parameters
        if (pHit.z < zMin || pHit.z > zMax || phi > phiMax) {
            if (tHit == t1) {
                continue;
            }
            if (t1 > rayTfar || (isSingleSided && !isNormalReversed)) {
                continue;
            }
            tHit = t1;
            pHit = org + tHit * dir;
            if (pHit.x == 0.0f && pHit.y == 0.0f) {
                pHit.x = 1e-5f * radius;
            }
            phi = atan2f(pHit.y, pHit.x);
            if (phi < 0.0f) {
                phi += scene_rdl2::math::sTwoPi;
            }
            if (pHit.z < zMin || pHit.z > zMax || phi > phiMax) {
                continue;
            }
        }
        // TODO call intersect filter functions
        rayTfar = tHit;
        RTCHitN_instID(hits, N, index, 0) = args->context->instID[0];
        RTCHitN_geomID(hits, N, index) = sphere->getGeomID();
        RTCHitN_primID(hits, N, index) = 0;
        // local space normal will be transformed
        // to parent space at postIntersect
        RTCHitN_Ng_x(hits, N, index) = pHit.x;
        RTCHitN_Ng_y(hits, N, index) = pHit.y;
        RTCHitN_Ng_z(hits, N, index) = pHit.z;
        // v will be computed at postIntersect
        RTCHitN_u(hits, N, index) = phi;
    }
}

RTCIntersectFunctionN
Sphere::getIntersectFunction() const
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
    const Sphere* sphere = (const Sphere*)userData->mPrimitive;
    // get sidedness and reverse normals
    bool isSingleSided = sphere->getIsSingleSided();
    bool isNormalReversed = sphere->getIsNormalReversed();
    float radius = sphere->getRadius();
    const Mat43& P2L = sphere->getP2L();
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
        // compute quadratic sphere coefficients
        const float A = dot(dir, dir);
        const float B = 2.0f * dot(org, dir);
        const float C = dot(org, org) - scene_rdl2::math::sqr(radius);
        // solve quadratic equation for t values
        const float D = B * B - 4.0f * A * C;
        if (D < 0.0f) {
            continue;
        }
        const float rootDiscrim = scene_rdl2::math::sqrt(D);
        float q = B < 0.0f ?
            -0.5f * (B - rootDiscrim) : -0.5f * (B + rootDiscrim);
        float t0 = q / A;
        float t1 = C / q;
        if (t0 > t1) {
            std::swap(t0, t1);
        }
        // compute intersection distance along ray
        if (t0 > rayTfar || t1 < rayTnear) {
            continue;
        }
        float tHit = t0;
        if (t0 < rayTnear || (isSingleSided && isNormalReversed)) {
            tHit = t1;
            if (tHit > rayTfar || (isSingleSided && !isNormalReversed)) {
                continue;
            }
        }
        // compute hit position and phi
        Vec3f pHit = org + tHit * dir;
        if (pHit.x == 0.0f && pHit.y == 0.0f) {
            pHit.x = 1e-5f * radius;
        }
        float phi = atan2f(pHit.y, pHit.x);
        if (phi < 0.0f) {
            phi += scene_rdl2::math::sTwoPi;
        }
        float zMin = sphere->getZMin();
        float zMax = sphere->getZMax();
        float phiMax = sphere->getPhiMax();
        // test sphere intersection against clipping parameters
        if (pHit.z < zMin || pHit.z > zMax || phi > phiMax) {
            if (tHit == t1) {
                continue;
            }
            if (t1 > rayTfar || (isSingleSided && !isNormalReversed)) {
                continue;
            }
            tHit = t1;
            pHit = org + tHit * dir;
            if (pHit.x == 0.0f && pHit.y == 0.0f) {
                pHit.x = 1e-5f * radius;
            }
            phi = atan2f(pHit.y, pHit.x);
            if (phi < 0.0f) {
                phi += scene_rdl2::math::sTwoPi;
            }
            if (pHit.z < zMin || pHit.z > zMax || phi > phiMax) {
                continue;
            }
        }
        // TODO call occlude filter functions
        // mark the tfar negative is the official signal
        // for embree that the ray is occluded
        rayTfar = -FLT_MAX;
    }
}

RTCOccludedFunctionN
Sphere::getOccludedFunction() const
{
    return &occludedFunc;
}

} // namespace internal
} // namespace geom
} // namespace moonray

