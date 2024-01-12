// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file CubicSpline.cc
///

#include "CubicSpline.h"

#include <moonray/rendering/geom/prim/BVHUserData.h>
#include <moonray/rendering/geom/prim/Instance.h>
#include <moonray/rendering/geom/prim/MotionTransform.h>
#include <moonray/rendering/geom/prim/Util.h>
#include <moonray/rendering/bvh/shading/RootShader.h>
#include <moonray/rendering/mcrt_common/ThreadLocalState.h>
#include <scene_rdl2/common/math/Math.h>
#include <scene_rdl2/common/math/ReferenceFrame.h>

namespace moonray {
namespace geom {
namespace internal {

using namespace scene_rdl2::math;
using namespace moonray::shading;

CubicSpline::CubicSpline(
        Curves::Type type,
        geom::Curves::SubType subtype,
        geom::Curves::CurvesVertexCount&& curvesVertexCount,
        geom::Curves::VertexBuffer&& vertices,
        LayerAssignmentId&& layerAssignmentId,
        shading::PrimitiveAttributeTable&& primitiveAttributeTable):
        Curves(type,
               static_cast<Curves::SubType>(subtype),
               std::move(curvesVertexCount),
               std::move(vertices),
               std::move(layerAssignmentId),
               std::move(primitiveAttributeTable))
{

}

template <typename CvType>
static __forceinline CvType evalCubicFromWeights(
        const float& w0, const float& w1, const float& w2, const float& w3,
        const CvType& cv0, const CvType& cv1, const CvType& cv2, const CvType& cv3)
{
    return w0 * cv0 + w1 * cv1 + w2 * cv2 + w3 * cv3;
}

void
CubicSpline::postIntersect(mcrt_common::ThreadLocalState &tls,
        const scene_rdl2::rdl2::Layer* layer, const mcrt_common::Ray& ray,
        shading::Intersection& intersection) const
{
    const int spanId = ray.primID;
    const IndexData& index = mIndexBuffer[spanId];
    const uint32_t chain = index.mChain;
    const uint32_t vertexOffset = index.mVertex;
    const uint32_t spansInChain = getSpansInChain(chain);
    const uint32_t indexInChain = index.mSpan;

    Vec3fa cv[4];
    if (!isMotionBlurOn()) {
        // Grab the CVs
        cv[0] = mVertexBuffer(vertexOffset + 0);
        cv[1] = mVertexBuffer(vertexOffset + 1);
        cv[2] = mVertexBuffer(vertexOffset + 2);
        cv[3] = mVertexBuffer(vertexOffset + 3);
    } else {
        // only support two time samples for motionblur at this moment
        MNRY_ASSERT(getMotionSamplesCount() == 2);
        float wt = ray.time;
        cv[0] = lerp(mVertexBuffer(vertexOffset + 0, 0),
                     mVertexBuffer(vertexOffset + 0, 1), ray.time);
        cv[1] = lerp(mVertexBuffer(vertexOffset + 1, 0),
                     mVertexBuffer(vertexOffset + 1, 1), ray.time);
        cv[2] = lerp(mVertexBuffer(vertexOffset + 2, 0),
                     mVertexBuffer(vertexOffset + 2, 1), ray.time);
        cv[3] = lerp(mVertexBuffer(vertexOffset + 3, 0),
                     mVertexBuffer(vertexOffset + 3, 1), ray.time);
    }

    const int assignmentId =
        mLayerAssignmentId.getType() == LayerAssignmentId::Type::CONSTANT ?
        mLayerAssignmentId.getConstId() :
        mLayerAssignmentId.getVaryingId()[chain];
    intersection.setLayerAssignments(assignmentId, layer);

    const AttributeTable *table =
        intersection.getMaterial()->get<shading::RootShader>().getAttributeTable();
    intersection.setTable(&tls.mArena, table);
    intersection.setIds(vertexOffset, 0, 0);
    const Attributes* primitiveAttributes = getAttributes();

    float w0, w1, w2, w3;
    evalWeights(ray.u, w0, w1, w2, w3);
    const int varyingOffset = chain + spanId;
    CubicSplineInterpolator interpolator(primitiveAttributes,
                                         ray.time,
                                         chain,
                                         varyingOffset,
                                         indexInChain,
                                         ray.u,
                                         vertexOffset,
                                         w0, w1, w2, w3);
    intersection.setRequiredAttributes(interpolator);

    // Add an interpolated N, dPds, and dPdt to the intersection
    // if they exist in the primitive attribute table
    setExplicitAttributes<CubicSplineInterpolator>(interpolator,
                                                   *primitiveAttributes,
                                                   intersection);

    overrideInstanceAttrs(ray, intersection);

    // The St value is read from the explicit "uv" primitive
    // attribute if it exists.
    Vec2f St;
    if (primitiveAttributes->isSupported(shading::StandardAttributes::sUv)) {
        interpolator.interpolate(shading::StandardAttributes::sUv,
            reinterpret_cast<char*>(&St));
    } else {
        const float stSpanRange = 1.0f / spansInChain;
        const float ststart = stSpanRange * indexInChain;
        St[0] = ststart + ray.u * stSpanRange;
        // ray.v is in (-1,1) so we need to remap to (0,1)
        St[1] = ray.v * 0.5f + 0.5f;
    }

    Vec3f N, dPds, dPdt;
    const bool hasExplicitAttributes = getExplicitAttributes(*primitiveAttributes,
                                                             intersection,
                                                             N, dPds, dPdt);

    if (!hasExplicitAttributes) {
        // Partial with respect to S is easy enough -- its the tangent vector
        // TODO: This isn't varying with R down the curve in the direction of
        //       the tangent vector, which it should.  We should slope down by R'.
        dPds = evalCubicDeriv(ray.u, cv[0], cv[1], cv[2], cv[3]);

        if (mSubType == SubType::RAY_FACING) {
            // Use the flipped ray direction to compute the normal. But, this may be an
            // instance, so we need to transform the ray direction into the local
            // space.  It gets transformed back to render space later on.
            // Note that it looks like we should be using r2l to get from render to
            // local space, but recall that when transforming normals we use the transpose
            // of the inverse, which is l2r transposed.  The transformNormal() function
            // uses the inverse of the transform (l2r) and applies the transpose.
            const Vec3f flippedRayDir = ray.isInstanceHit() ?
                                        normalize(transformNormal(ray.ext.l2r, -ray.dir)) :
                                        normalize(-ray.dir);

            // Lacking much better options, we'll construct dPdt wrt the other two
            // components of the ostensibly orthonormal frame we have
            dPdt = cross(dPds, flippedRayDir);
            if (unlikely(lengthSqr(dPdt) < scene_rdl2::math::sEpsilon)) {
                // Degened dPds or dPdt. The reason can be either clustered cv points
                // or ray direction is parallel to dPds. Use ReferenceFrame as fallback
                N = flippedRayDir;
                scene_rdl2::math::ReferenceFrame frame(N);
                dPds = frame.getX();
                dPdt = frame.getY();
            } else {
                // Cross the partials to get the normal
                N = cross(dPdt, dPds);
            }
            N = N.normalize();
        } else if (mSubType == SubType::ROUND || mSubType == SubType::NORMAL_ORIENTED) {
            // round curves just use the geom normal
            N = ray.getNg();
            N = N.normalize();

            dPdt = cross(dPds, N);
            if (unlikely(lengthSqr(dPdt) < scene_rdl2::math::sEpsilon)) {
                // Degened dPds or dPdt. The reason can be either clustered cv points
                // or ray direction is parallel to dPds. Use ReferenceFrame as fallback
                scene_rdl2::math::ReferenceFrame frame(N);
                dPds = frame.getX();
                dPdt = frame.getY();
            }
        }
    }

    // Geometric normal equals shading normal for a cubic spline
    intersection.setDifferentialGeometry(N, // geometric normal
                                         N, // shading normal
                                         St,
                                         dPds,
                                         dPdt,
                                         true, // has derivatives
                                         hasExplicitAttributes);


    const scene_rdl2::rdl2::Geometry* geometry = intersection.getGeometryObject();
    MNRY_ASSERT(geometry != nullptr);
    if (geometry->getRayEpsilon() <= 0.0f) {
        // Use 1.5 times the curve radius as the next ray epsilon hint.
        // This lets shadow and continuation rays escape the intersected
        // hair, instead of self-intersecting.
        float curveRadius = w0 * cv[0].w + w1 * cv[1].w + w2 * cv[2].w + w3 * cv[3].w;
        intersection.setEpsilonHint(1.5f * curveRadius);
    } else {
        intersection.setEpsilonHint(geometry->getRayEpsilon());
    }

    // calculate dfds/dfdt for primitive attributes that request derivatives
    if (table->requestDerivatives()) {
        // ds = stSpanRange -> 1 / ds = 1 / stSpanRange = spansInChain
        computeAttributesDerivatives(table,
                                     ray.u,
                                     static_cast<float>(spansInChain),
                                     chain,
                                     varyingOffset,
                                     indexInChain,
                                     vertexOffset,
                                     ray.time,
                                     intersection);
    }

    // For wireframe AOV/shaders
    if (table->requests(StandardAttributes::sNumPolyVertices)) {
        intersection.setAttribute(StandardAttributes::sNumPolyVertices, 4);
    }
    if (table->requests(StandardAttributes::sPolyVertexType)) {
        intersection.setAttribute(StandardAttributes::sPolyVertexType,
            static_cast<int>(StandardAttributes::POLYVERTEX_TYPE_CUBIC_SPLINE));
    }
    for (int iVert = 0; iVert < 4; iVert++) {
        if (table->requests(StandardAttributes::sPolyVertices[iVert])) {
            // may need to move the vertices to render space
            // for instancing object since they are ray traced in local space
            const Vec3f v = ray.isInstanceHit() ?
                            transformPoint(ray.ext.l2r, cv[iVert].asVec3f()) :
                            cv[iVert].asVec3f();
            intersection.setAttribute(StandardAttributes::sPolyVertices[iVert], v);
        }
    }

    // motion vectors
    if (table->requests(StandardAttributes::sMotion)) {
        Vec3f pos0, pos1;
        const Vec3f *pos1Ptr = nullptr;
        if (!isMotionBlurOn()) {
            const Vec3fa v0 = evalCubic(ray.u,
                mVertexBuffer(vertexOffset + 0, 0), mVertexBuffer(vertexOffset + 1, 0),
                mVertexBuffer(vertexOffset + 2, 0), mVertexBuffer(vertexOffset + 3, 0));
            pos0 = v0.asVec3f();
            pos1Ptr = nullptr;
        } else {
            float wc = ray.getTime() - sHalfDt;
            const Vec3fa v0 = evalCubic(ray.u,
                lerp(mVertexBuffer(vertexOffset + 0, 0), mVertexBuffer(vertexOffset + 0, 1), wc),
                lerp(mVertexBuffer(vertexOffset + 1, 0), mVertexBuffer(vertexOffset + 1, 1), wc),
                lerp(mVertexBuffer(vertexOffset + 2, 0), mVertexBuffer(vertexOffset + 2, 1), wc),
                lerp(mVertexBuffer(vertexOffset + 3, 0), mVertexBuffer(vertexOffset + 3, 1), wc));
            wc = ray.getTime() + sHalfDt;
            const Vec3fa v1 = evalCubic(ray.u,
                lerp(mVertexBuffer(vertexOffset + 0, 0), mVertexBuffer(vertexOffset + 0, 1), wc),
                lerp(mVertexBuffer(vertexOffset + 1, 0), mVertexBuffer(vertexOffset + 1, 1), wc),
                lerp(mVertexBuffer(vertexOffset + 2, 0), mVertexBuffer(vertexOffset + 2, 1), wc),
                lerp(mVertexBuffer(vertexOffset + 3, 0), mVertexBuffer(vertexOffset + 3, 1), wc));
            pos0 = v0.asVec3f();
            pos1 = v1.asVec3f();
            pos1Ptr = &pos1;
        }

        // Motion vectors only support a single instance level, hence we only care
        // about ray.instance0.
        const Instance *instance = (ray.isInstanceHit())?
            static_cast<const Instance *>(ray.ext.instance0OrLight) : nullptr;
        const Vec3f motion = computePrimitiveMotion(pos0, pos1Ptr, ray.getTime(), instance);
        intersection.setAttribute(StandardAttributes::sMotion, motion);
    }
}


bool
CubicSpline::computeIntersectCurvature(const mcrt_common::Ray& ray,
        const shading::Intersection& intersection, Vec3f& dNds, Vec3f& dNdt) const
{
    // dNds and dNdt are not geometrically useful because the cuve is a flat
    // ribbon whose normal faces the ray. This means N is dependent on
    // ray direction rather than on the geometry of the curve alone.
    dNds = Vec3f(0.0f, 0.0f, 0.0f);
    dNdt = Vec3f(0.0f, 0.0f, 0.0f);
    return true;
}

void
CubicSpline::computeAttributesDerivatives(const AttributeTable* table,
        float u, float invDs, int chain,
        int varyingOffset, int faceVaryingOffset, int vertexOffset,
        float time, shading::Intersection& intersection) const
{
    Attributes* attrs = getAttributes();
    for (auto key: table->getDifferentialAttributes()) {
        if (!attrs->isSupported(key)) {
            continue;
        }
        switch (key.getType()) {
        case scene_rdl2::rdl2::TYPE_FLOAT:
            computeAttributeDerivatives(TypedAttributeKey<float>(key),
                u, invDs, chain, varyingOffset, faceVaryingOffset, vertexOffset,
                time, intersection);
            break;
        case scene_rdl2::rdl2::TYPE_RGB:
            computeAttributeDerivatives(TypedAttributeKey<scene_rdl2::math::Color>(key),
                u, invDs, chain, varyingOffset, faceVaryingOffset, vertexOffset,
                time, intersection);
            break;
        case scene_rdl2::rdl2::TYPE_RGBA:
            computeAttributeDerivatives(TypedAttributeKey<scene_rdl2::math::Color4>(key),
                u, invDs, chain, varyingOffset, faceVaryingOffset, vertexOffset,
                time, intersection);
            break;
        case scene_rdl2::rdl2::TYPE_VEC2F:
            computeAttributeDerivatives(TypedAttributeKey<scene_rdl2::math::Vec2f>(key),
                u, invDs, chain, varyingOffset, faceVaryingOffset, vertexOffset,
                time, intersection);
            break;
        case scene_rdl2::rdl2::TYPE_VEC3F:
            computeAttributeDerivatives(TypedAttributeKey<scene_rdl2::math::Vec3f>(key),
                u, invDs, chain, varyingOffset, faceVaryingOffset, vertexOffset,
                time, intersection);
            break;
        case scene_rdl2::rdl2::TYPE_MAT4F:
            computeAttributeDerivatives(TypedAttributeKey<scene_rdl2::math::Mat4f>(key),
                u, invDs, chain, varyingOffset, faceVaryingOffset, vertexOffset,
                time, intersection);
            break;
        default:
            break;
        }
    }
}

} // namespace internal
} // namespace geom
} // namespace moonray

