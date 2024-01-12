// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file Points.cc
/// $Id$
///

#include "Points.h"

#include <moonray/rendering/geom/prim/BVHUserData.h>
#include <moonray/rendering/geom/prim/Instance.h>
#include <moonray/rendering/geom/prim/MotionTransform.h>
#include <moonray/rendering/geom/prim/Util.h>

#include <moonray/rendering/bvh/shading/Attributes.h>
#include <moonray/rendering/bvh/shading/AttributeTable.h>
#include <moonray/rendering/bvh/shading/Interpolator.h>
#include <moonray/rendering/bvh/shading/RootShader.h>

namespace moonray {
namespace geom {
namespace internal {

using namespace shading;

// primitive attributes for Points don't involve interpolation actually
// the reason to use a Interpolator with weight 1.0f is just reusing
// existing interface for other primitive types
class PointsInterpolator : public shading::Interpolator
{
public:
    PointsInterpolator(const Attributes *attr, float time, int pointID):
        shading::Interpolator(attr,
                              time,
                              0,       // part
                              0,       // coarseFace
                              1,       // numVaryings
                              mIndex,  // varyings
                              mWeight, // varyingWeights
                              1,       // numFaceVarying
                              mIndex,  // faceVaryings
                              mWeight, // faceVaryingWeidhts
                              0,       // tessellatedFace
                              1,       // numVertices
                              mIndex,  // vertices
                              mWeight) // vertexWeights
    {
        mIndex[0] = pointID;
        mWeight[0] = 1.0f;
    }
private:
    int mIndex[1];
    float mWeight[1];
};


Points::Points(geom::Points::VertexBuffer&& position,
        geom::Points::RadiusBuffer&& radius,
        LayerAssignmentId&& layerAssignmentId,
        PrimitiveAttributeTable&& primitiveAttributeTable):
    NamedPrimitive(std::move(layerAssignmentId)),
    mPosition(std::move(position)), mRadius(std::move(radius)),
    mPrimitiveAttributeTable(std::move(primitiveAttributeTable))
{
    size_t pointsCount = mPosition.size();
    MNRY_ASSERT_REQUIRE(mRadius.size() == pointsCount);
    if (mLayerAssignmentId.getType() == LayerAssignmentId::Type::VARYING) {
        MNRY_ASSERT_REQUIRE(
            mLayerAssignmentId.getVaryingId().size() == pointsCount);
    }

    // facevarying/varying/vertex are the same thing for Points
    for (auto& kv : primitiveAttributeTable) {
        for (auto& attribute : kv.second) {
            if (attribute->getRate() == RATE_FACE_VARYING) {
                attribute->setRate(RATE_VARYING);
            }
        }
    }
    setAttributes(Attributes::interleave(mPrimitiveAttributeTable,
        (size_t)0, (size_t)1, pointsCount, std::vector<size_t>(), pointsCount));
}

const scene_rdl2::rdl2::Material*
Points::getIntersectionMaterial(const scene_rdl2::rdl2::Layer* pRdlLayer,
        const mcrt_common::Ray& ray) const
{
    int assignmentId =
        mLayerAssignmentId.getType() == LayerAssignmentId::Type::CONSTANT ?
        mLayerAssignmentId.getConstId() :
        mLayerAssignmentId.getVaryingId()[ray.primID];

    MNRY_ASSERT(assignmentId > -1);
    const scene_rdl2::rdl2::Material* pMaterial = MNRY_VERIFY(pRdlLayer->lookupMaterial(assignmentId));

    return pMaterial;
}

void
Points::postIntersect(mcrt_common::ThreadLocalState& tls,
        const scene_rdl2::rdl2::Layer* pRdlLayer, const mcrt_common::Ray& ray,
        Intersection& intersection) const
{
    int assignmentId =
        mLayerAssignmentId.getType() == LayerAssignmentId::Type::CONSTANT ?
        mLayerAssignmentId.getConstId() :
        mLayerAssignmentId.getVaryingId()[ray.primID];
    intersection.setLayerAssignments(assignmentId, pRdlLayer);

    const AttributeTable *table =
        intersection.getMaterial()->get<shading::RootShader>().getAttributeTable();
    intersection.setTable(&tls.mArena, table);
    intersection.setIds(ray.primID, 0, 0);
    const Attributes* primitiveAttributes = getAttributes();

    PointsInterpolator interpolator(primitiveAttributes,
                                    ray.time,
                                    ray.primID);

    intersection.setRequiredAttributes(interpolator);

    // Add an interpolated N, dPds, and dPdt to the intersection
    // if they exist in the primitive attribute table
    setExplicitAttributes<PointsInterpolator>(interpolator,
                                              *primitiveAttributes,
                                              intersection);

    overrideInstanceAttrs(ray, intersection);

    // The St value is read from the explicit "uv" primitive
    // attribute if it exists.
    Vec2f St(1.0f, 1.0f);
    if (primitiveAttributes->isSupported(shading::StandardAttributes::sUv)) {
        interpolator.interpolate(shading::StandardAttributes::sUv,
            reinterpret_cast<char*>(&St));
    }

    Vec3f N, dPds, dPdt;
    const bool hasExplicitAttributes = getExplicitAttributes(*primitiveAttributes,
                                                             intersection,
                                                             N, dPds, dPdt);

    if (!hasExplicitAttributes) {
        N = normalize(ray.getNg());
    }

    // The geometric and shading normal use the value of the StandardAttribute
    // if available and that means the point is not being treated as a real sphere.
    // The "P" is on the surface of the sphere, but the "Ng" is the normal of a
    // disk, so it's a weird hybrid geometry. This is ok because points are supposed
    // to be small in the scene, so any inconsistency should not be noticeable.
    intersection.setDifferentialGeometry(N, // geometric normal
                                         N, // shading normal
                                         St,
                                         dPds,
                                         dPdt,
                                         hasExplicitAttributes, // If it has explicit attributes it has derivatives
                                         hasExplicitAttributes);

    const scene_rdl2::rdl2::Geometry* geometry = intersection.getGeometryObject();
    MNRY_ASSERT(geometry != nullptr);
    intersection.setEpsilonHint(geometry->getRayEpsilon());

    // wireframe AOV is blank
    if (table->requests(StandardAttributes::sNumPolyVertices)) {
        intersection.setAttribute(StandardAttributes::sNumPolyVertices, 0);
        intersection.setAttribute(StandardAttributes::sPolyVertexType,
                              static_cast<int>(StandardAttributes::POLYVERTEX_TYPE_POLYGON));

    }

    // motion vectors
    if (table->requests(StandardAttributes::sMotion)) {
        Vec3f pos0, pos1;
        const Vec3f *pos1Ptr = nullptr;
        if (isMotionBlurOn()) {
            pos0 = lerp(mPosition(ray.primID, 0), mPosition(ray.primID, 1), ray.getTime() - sHalfDt);
            pos1 = lerp(mPosition(ray.primID, 0), mPosition(ray.primID, 1), ray.getTime() + sHalfDt);
            pos1Ptr = &pos1;
        } else {
            pos0 = mPosition(ray.primID, 0);
            pos1Ptr = nullptr;
        }

        // Motion vectors only support a single instance level, hence we only care
        // about ray.instance0.
        const Instance *instance = (ray.isInstanceHit())?
            static_cast<const Instance *>(ray.ext.instance0OrLight) : nullptr;
        const Vec3f motion = computePrimitiveMotion(pos0, pos1Ptr, ray.getTime(), instance);
        intersection.setAttribute(StandardAttributes::sMotion, motion);
    }
}

BBox3f
Points::computeAABB() const
{
    if (mPosition.empty()) {
        return BBox3f(scene_rdl2::math::zero);
    }
    BBox3f result(mPosition(0));
    float maxRadius = 0.0f;
    size_t motionSampleCount = getMotionSamplesCount();
    for (size_t v = 0; v < mPosition.size(); ++v) {
        maxRadius = std::max(maxRadius, mRadius[v]);
        for (size_t t = 0; t < motionSampleCount; ++t) {
            result.extend(mPosition(v, t));
        }
    }
    result.lower -= Vec3f(maxRadius);
    result.upper += Vec3f(maxRadius);
    return result;
}

BBox3f
Points::computeAABBAtTimeStep(int timeStep) const
{
    if (mPosition.empty()) {
        return BBox3f(scene_rdl2::math::zero);
    }
    MNRY_ASSERT(timeStep >= 0 && timeStep < static_cast<int>(getMotionSamplesCount()), "timeStep out of range");
    BBox3f result(scene_rdl2::util::empty);
    float maxRadius = 0.0f;
    for (size_t v = 0; v < mPosition.size(); ++v) {
        maxRadius = std::max(maxRadius, mRadius[v]);
        result.extend(mPosition(v, timeStep));
    }
    result.lower -= Vec3f(maxRadius);
    result.upper += Vec3f(maxRadius);
    return result;
}

static void
boundsFunc(const RTCBoundsFunctionArguments* args)
{
    const BVHUserData* userData = (const BVHUserData*)args->geometryUserPtr;
    const Points* points = (const Points*)userData->mPrimitive;
    RTCBounds* output = args->bounds_o;
    size_t item = args->primID;
    Vec3f pMin = points->getVertexBuffer()(item);
    Vec3f pMax = pMin;
    float r = points->getRadiusBuffer()[item];
    if (points->isMotionBlurOn()) {
        for (size_t i = 1; i < points->getVertexBuffer().get_time_steps(); i++) {
            const Vec3f p = points->getVertexBuffer()(item, i);
            pMin = min(pMin, p);
            pMax = max(pMax, p);
        }
    }
    output->lower_x = pMin.x - r;
    output->lower_y = pMin.y - r;
    output->lower_z = pMin.z - r;
    output->upper_x = pMax.x + r;
    output->upper_y = pMax.y + r;
    output->upper_z = pMax.z + r;
}

RTCBoundsFunction
Points::getBoundsFunction() const
{
    return &boundsFunc;
}

static finline Vec3fa
getPosition(const Points* points, size_t item, float rayTime)
{
    Vec3f p;
    if (points->isMotionBlurOn()) {
        const int numSegments = points->getVertexBuffer().get_time_steps() - 1;
        const float clampedRayTime = scene_rdl2::math::clamp(rayTime, 0.0f, 1.0f);
        const float idxPlusT = clampedRayTime * (float)numSegments;
        const int idx0 = (int)floor(idxPlusT);
        const int idx1 = idx0 + 1;
        const float t = idxPlusT - (float)idx0;
        p = lerp(points->getVertexBuffer()(item, idx0),
                 points->getVertexBuffer()(item, idx1),
                 t);
    } else {
        p = points->getVertexBuffer()(item);
    }
    return Vec3fa(p, 0.f);
}

static void
intersectFunc(const RTCIntersectFunctionNArguments* args)
{
    // A common use case for us is very tiny spheres relative to
    // ray length.  This makes using the "standard" ray/sphere
    // intersection problematic as it is highly suspectible to
    // catstrophic cancellation.  This implemtnation avoids this
    // problem nicely, and was explained by Mike Day in an email
    // thread.  I'm pasting in some of the more salient points
    // from that disucssion.
    //
    // Here is the somewhate paraphrased "standard" function
    //
    // // o = ray origin
    // // d = ray direction
    // // p = sphere centre
    // // r = sphere radius
    //
    // const Vec3fa v = o - p;
    //
    // const float A = dot(d, d);
    // const float B = 2.0f * dot(v, d);
    // const float C = dot(v, v) - r * r;
    // const float D = B * B - 4.0f * A * C;
    // if (D < 0.0f) {
    //     return;
    // }
    //
    // const float Q = scene_rdl2::math::sqrt(D);
    // const float rcpA = scene_rdl2::math::rcp(A);
    // const float t0 = 0.5f * rcpA * (-B - Q);
    // const float t1 = 0.5f * rcpA * (-B + Q);
    //
    // if ((ray.tnear < t0) && (t0 < ray.tfar)) {
    //     ... // set other ray members
    //     ray.Ng = o + t0 * d - p;
    // }
    // if ((ray.tnear < t1) && (t1 < ray.tfar)) {
    //     ... // set other ray members
    //     ray.Ng = o + t1 * d - p;
    // }
    //
    // This is all quite standard. Use the input values to set up
    // the coefficients A, B, C of a quadratic equation,
    // take the discriminant D, quit if D < 0
    // otherwise go on to solve the quadratic equation and compute the other
    // necessary intersection results.
    //
    // Let's see where the precision losses are occurring.
    // The first source of precision loss is in computing the discriminant,
    //
    //     const float D = B * B - 4.0f * A * C;
    //
    // Plugging in the expressions for A, B and C, this is computing
    // 4 ( (v.d)^2 - (d.d)(v.v - r^2) ). So it's taking the difference of
    // (v.d)^2 and (d.d)(v.v - r^2). When v and d are in very nearly the same
    // direction, as they will be when the ray points at a distant tiny
    // sphere, their dot product v.d is almost equal to the product of the 2
    // lengths, |v| * |d|, which is to say (v.d)^2 is close to |v|^2 * |d|^2.
    // And with r very small, the term (v.v - r^2) is close to v.v, which means
    // (d.d)(v.v - r^2) is also close to |v|^2 * |d|^2. So we've got 2 numbers
    // almost equal to each other, and we're taking their difference.
    // Alarm bells should go off at this point!
    // Catastrophic cancellation ensues!
    //
    // The result is that D will retain very few significant bits, i.e.
    // instead of having a mantissa which can vary over any of the possible
    // 2^23 values, it may only vary over a small handful of values.
    // In extreme cases, as with very small spheres indeed,
    // the cancellation is absolutely catastrophic
    // and we'll end up with D = 0, so all information about the structure
    // of the sphere has effectively been lost. It has become a black hole.
    //
    // The point to note here is that in floating point the dot product has
    // very poor resolution for vectors which are nearly parallel. It varies
    // with the cosine of the angle between the vectors - the cosine is 1
    // when the angle is zero (actually -1 in the code, because v and d point
    // in opposite directions), and because the gradient flattens to zero here,
    // very little change occurs as you vary the vector directions by small
    // amounts in this vicinity, so you can expect very little change in the
    // mantissa and hence a very 'steppy' value - lots of quantisation
    // instead of a smoothly changing value.
    //
    // We can save the day using the cross product instead of the dot product.
    // The cross product varies with the sine of the angle between the vectors,
    // and sine has maximum gradient when the angle is zero, so we can expect it
    // to have good resolution in floating point arithmetic in situations where
    // the vectors are nearly parallel. Here's how to change the expression for
    // D to make this possible:
    //
    //     D = B^2 - 4AC
    //       = 4 ( (v.d)^2 - (d.d)(v.v - r^2) )
    //       = 4 ( (d.d)r^2 - ((d.d)(v.v) - (v.d)^2) )
    //       = 4 ( (d.d)r^2 - (d x v)^2 )
    //
    // The last step here is just a consequence of the familiar
    // identity sin^2(theta) + cos^2(theta) = 1.
    //
    // In this revised version for D, we've replaced the difference of 2
    // nearly equal terms (v.d)^2 and (d.d)(v.v) by a single, more accurately
    // computed term. We can expect the result to have correspondingly
    // improved resolution in floating point.
    //
    // Even with this improvement, things still go badly wrong even for
    // spheres with a modicum of smallness.  In this case
    // the culprit is the line
    //
    //     ray.Ng = o + t0 * d - p;
    //
    // This involves more catastrophic cancellation for small radii.
    // It's computing the intersection point, o + t0*d,
    // and then subtracting off it a very nearby point, p.
    // The closeness of these 2 points is the cause of the trouble.
    // And changing the order of summation won't help, since in general we
    // can expect all 3 of the vectors o, t0*d, and p to be large compared to
    // the result. To fix things, let's examine the expression for the normal
    // more closely:
    //
    //     o + t0 * d - p = v + t0 * d
    //                    = v + d * (-B - Q) / (2A)
    //                    = v + d * (-2v.d - Q) / (2d.d)
    //                    = ( (d.d)v - (v.d)d - (Q/2)d ) / (d.d)
    //                    = ( d x (v x d) - (Q/2)d ) / (d.d)
    //
    // The last step uses a standard identity from vector algebra.
    // So once again, we've taken a dot-product-type expression involving
    // the difference of (vector) quantities which will be very similar
    // for nearly parallel vectors, (d.d)v - (v.d)d, and replaced it with a
    // cross-product-type expression, d x (d x v), which will retain
    // much greater accuracy in this situation.

    unsigned int N = args->N;
    unsigned int primID = args->primID;
    const BVHUserData* userData = (const BVHUserData*)args->geometryUserPtr;
    const Points* points = (const Points*)userData->mPrimitive;

    float r = points->getRadiusBuffer()[primID];
    float r2 = r * r;

    if (N == 1) {
        RTCRayHit& rayhit = *((RTCRayHit*)args->rayhit);
        RTCRay& ray = rayhit.ray;
        RTCHit& hit = rayhit.hit;

        Vec3fa p = getPosition(points, primID, ray.time);
        const Vec3fa o(ray.org_x, ray.org_y, ray.org_z, 0.f);
        const Vec3fa d(ray.dir_x, ray.dir_y, ray.dir_z, 0.f);
        const Vec3fa u = p - o;
        if (dot(d, u) < 0.0f) {
            // ray is travelling away from sphere centre
            return;
        }
        if (dot(u, u) < r2) {
            // ray origin is inside sphere
            return;
        }
        const Vec3fa v = cross(d, u);
        float v2 = dot(v, v);
        float d2 = dot(d, d);
        float r2d2 = r2 * d2;
        float D = r2d2 - v2;
        if (D < 0.0f) {
            // no intersections
            return;
        }
        float s = scene_rdl2::math::sqrt(D);
        const Vec3fa w = cross(d, v);
        const Vec3fa rn = scene_rdl2::math::rcp(d2) * (w - s * d);
        float t = length(u + rn) * scene_rdl2::math::rsqrt(d2);
        if (ray.tnear < t && t < ray.tfar) {
            ray.tfar = t;
            hit.instID[0] = args->context->instID[0];
            hit.geomID = points->getGeomID();
            hit.primID = primID;
            hit.u = 0.0f;
            hit.v = 0.0f;
            Vec3fa Ng = scene_rdl2::math::rcp(r) * rn;
            hit.Ng_x = Ng.x;
            hit.Ng_y = Ng.y;
            hit.Ng_z = Ng.z;
        }
    } else {
        int* valid = (int*)args->valid;
        RTCRayHitN* rayhit = (RTCRayHitN*)args->rayhit;
        RTCRayN* rays = RTCRayHitN_RayN(rayhit, N);
        RTCHitN* hits = RTCRayHitN_HitN(rayhit, N);
        for (unsigned int index = 0; index < N; ++index) {
            if (valid[index] == 0) {
                continue;
            }
            Vec3fa p = getPosition(points, primID,
                RTCRayN_time(rays, N, index));
            const Vec3fa o(
                RTCRayN_org_x(rays, N, index),
                RTCRayN_org_y(rays, N, index),
                RTCRayN_org_z(rays, N, index), 0.f);
            const Vec3fa d(
                RTCRayN_dir_x(rays, N, index),
                RTCRayN_dir_y(rays, N, index),
                RTCRayN_dir_z(rays, N, index), 0.f);
            float& rayTnear = RTCRayN_tnear(rays, N, index);
            float& rayTfar = RTCRayN_tfar(rays, N, index);
            const Vec3fa u = p - o;
            if (dot(d, u) < 0.0f) {
                // ray is travelling away from sphere centre
                continue;
            }
            if (dot(u, u) < r2) {
                // ray origin is inside sphere
                continue;
            }
            const Vec3fa v = cross(d, u);
            float v2 = dot(v, v);
            float d2 = dot(d, d);
            float r2d2 = r2 * d2;
            float D = r2d2 - v2;
            if (D < 0.0f) {
                // no intersections
                continue;
            }
            float s = scene_rdl2::math::sqrt(D);
            const Vec3fa w = cross(d, v);
            const Vec3fa rn = scene_rdl2::math::rcp(d2) * (w - s * d);
            float t = length(u + rn) * scene_rdl2::math::rsqrt(d2);
            if (rayTnear < t && t < rayTfar) {
                rayTfar = t;
                RTCHitN_instID(hits, N, index, 0) = args->context->instID[0];
                RTCHitN_geomID(hits, N, index) = points->getGeomID();
                RTCHitN_primID(hits, N, index) = primID;
                RTCHitN_u(hits, N, index) = 0.0f;
                RTCHitN_v(hits, N, index) = 0.0f;
                Vec3fa Ng = scene_rdl2::math::rcp(r) * rn;
                RTCHitN_Ng_x(hits, N, index) = Ng.x;
                RTCHitN_Ng_y(hits, N, index) = Ng.y;
                RTCHitN_Ng_z(hits, N, index) = Ng.z;
            }
        }
    }
}

RTCIntersectFunctionN
Points::getIntersectFunction() const
{
    return &intersectFunc;
}

static void
occludedFunc(const RTCOccludedFunctionNArguments* args)
{
    unsigned int N = args->N;
    unsigned int primID = args->primID;
    const BVHUserData* userData = (const BVHUserData*)args->geometryUserPtr;
    const Points* points = (const Points*)userData->mPrimitive;

    float r = points->getRadiusBuffer()[primID];
    float r2 = r * r;

    if (N == 1) {
        RTCRay& ray = *((RTCRay*)args->ray);
        Vec3fa p = getPosition(points, primID, ray.time);
        const Vec3fa o(ray.org_x, ray.org_y, ray.org_z, 0.f);
        const Vec3fa d(ray.dir_x, ray.dir_y, ray.dir_z, 0.f);
        const Vec3fa u = p - o;
        if (dot(d, u) < 0.0f) {
            // ray is travelling away from sphere centre
            return;
        }
        if (dot(u, u) < r2) {
            // ray origin is inside sphere
            return;
        }
        const Vec3fa v = cross(d, u);
        float v2 = dot(v, v);
        float d2 = dot(d, d);
        float r2d2 = r2 * d2;
        float D = r2d2 - v2;
        if (D < 0.0f) {
            // no intersections
            return;
        }
        float s = scene_rdl2::math::sqrt(D);
        const Vec3fa w = cross(d, v);
        const Vec3fa rn = scene_rdl2::math::rcp(d2) * (w - s * d);
        float t = length(u + rn) * scene_rdl2::math::rsqrt(d2);
        // TODO call occlude filter functions
        // mark the tfar negative is the official signal
        // for embree that the ray is occluded
        if (ray.tnear < t && t < ray.tfar) {
            ray.tfar = -FLT_MAX;
        }
    } else {
        int* valid = (int*)args->valid;
        RTCRayN* rays = args->ray;
        for (unsigned int index = 0; index < N; ++index) {
            if (valid[index] == 0) {
                continue;
            }
            Vec3fa p = getPosition(points, primID,
                RTCRayN_time(rays, N, index));
            const Vec3fa o(
                RTCRayN_org_x(rays, N, index),
                RTCRayN_org_y(rays, N, index),
                RTCRayN_org_z(rays, N, index), 0.f);
            const Vec3fa d(
                RTCRayN_dir_x(rays, N, index),
                RTCRayN_dir_y(rays, N, index),
                RTCRayN_dir_z(rays, N, index), 0.f);
            float& rayTnear = RTCRayN_tnear(rays, N, index);
            float& rayTfar = RTCRayN_tfar(rays, N, index);
            const Vec3fa u = p - o;
            if (dot(d, u) < 0.0f) {
                // ray is travelling away from sphere centre
                continue;
            }
            if (dot(u, u) < r2) {
                // ray origin is inside sphere
                continue;
            }
            const Vec3fa v = cross(d, u);
            float v2 = dot(v, v);
            float d2 = dot(d, d);
            float r2d2 = r2 * d2;
            float D = r2d2 - v2;
            if (D < 0.0f) {
                // no intersections
                continue;
            }
            float s = scene_rdl2::math::sqrt(D);
            const Vec3fa w = cross(d, v);
            const Vec3fa rn = scene_rdl2::math::rcp(d2) * (w - s * d);
            float t = length(u + rn) * scene_rdl2::math::rsqrt(d2);
            // TODO call occlude filter functions
            // mark the tfar negative is the official signal
            // for embree that the ray is occluded
            if (rayTnear < t && t < rayTfar) {
                rayTfar = -FLT_MAX;
            }
        }
    }
}

RTCOccludedFunctionN
Points::getOccludedFunction() const
{
    return &occludedFunc;
}

} // namespace internal
} // namespace geom
} // namespace moonray

