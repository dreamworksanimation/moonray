// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file LineSegments.cc
///

#include "LineSegments.h"

#include <moonray/rendering/geom/prim/Instance.h>
#include <moonray/rendering/geom/prim/MotionTransform.h>
#include <moonray/rendering/geom/prim/Util.h>

#include <moonray/rendering/bvh/shading/Attributes.h>
#include <moonray/rendering/bvh/shading/AttributeTable.h>
#include <moonray/rendering/bvh/shading/Interpolator.h>
#include <moonray/rendering/mcrt_common/ThreadLocalState.h>
#include <moonray/rendering/shading/Material.h>
#include <scene_rdl2/common/math/ReferenceFrame.h>

namespace moonray {
namespace geom {
namespace internal {

using namespace moonray::shading;

// LineSegments is similar to nonperiodic linear curves in Renderman
// each span is a linear line segment with 2 control points
// one or more spans form a chain, each chain is a curve
// one or more chains form a LineSegments (Curves) primitive
// To interpolate primitive attributes in different rates:
// constant    : should supply a single value for the entire Curves primitive
// uniform     : should supply a total of nCurves values for the entire Curves primitive
// varying     : should supply sum(nSpansi + 1) values
// facevarying : should supply sum(nSpansi + 1) values
// vertex      : should supply sum(nVerticesi) values
//
// For example: a LineSegments with 2 chains (nCurves = 2)
// chain0 has 3 spans (nSpans0 = 3, nVertices0 = 3  + 1 = 4)
// chain1 has 1 spans (nSpans1 = 1, nVertices1 = 1 + 1 = 2)
// constant attribute should supply 1 value
// uniform attribute should supply 2 values
// varying/facevarying/vertex attribute should supply 4 + 2 = 6 values
class LineSegmentsInterpolator : public shading::Interpolator
{
public:
    LineSegmentsInterpolator(const Attributes *attr, float time,
            int chain, float u, int vertexOffset):
        shading::Interpolator(attr, time,
        0,              // part
        chain,          // coarseFace
        2,              // numVaryings
        mCurveVertices, // varyings
        mLinearWeights, // varyingWeights
        2,              // numFaceVaryings
        mCurveVertices, // faceVaryings
        mLinearWeights, // faceVaryingWeights
        chain,          // tessellatedFace
        2,              // numVertices
        mCurveVertices, // vertices
        mLinearWeights) // vertexWeights
        {
            mLinearWeights[0] = 1.0f - u;
            mLinearWeights[1] = u;
            mCurveVertices[0] = vertexOffset;
            mCurveVertices[1] = vertexOffset + 1;
        }
private:
    float mLinearWeights[2];
    int mCurveVertices[2];
};


LineSegments::LineSegments(
        Curves::Type type,
        geom::Curves::SubType subtype,
        geom::Curves::CurvesVertexCount&& curvesVertexCount,
        geom::Curves::VertexBuffer&& vertices,
        LayerAssignmentId&& layerAssignmentId,
        PrimitiveAttributeTable&& primitiveAttributeTable):
        Curves(type,
               static_cast<Curves::SubType>(subtype),
               std::move(curvesVertexCount),
               std::move(vertices),
               std::move(layerAssignmentId),
               std::move(primitiveAttributeTable))
{
    // allocate/calculate index buffer
    size_t curvesCount = getCurvesCount();
    std::vector<uint32_t> spansInChain(curvesCount, 0);
    size_t spanCount = 0;
    for (size_t i = 0; i < curvesCount; ++i) {
        spansInChain[i] = mCurvesVertexCount[i] - 1;
        spanCount += spansInChain[i];
    }
    mSpanCount = spanCount;
    // allocate/fill the index buffer
    mIndexBuffer.reserve(mSpanCount);
    size_t vertexOffset = 0;
    for (size_t i = 0; i < curvesCount; ++i) {
        for (size_t j = 0; j < spansInChain[i]; ++j) {
            mIndexBuffer.emplace_back(vertexOffset++, i, j);
        }
        vertexOffset += 1;
    }

    if (mLayerAssignmentId.getType() == LayerAssignmentId::Type::VARYING) {
        MNRY_ASSERT_REQUIRE(
            mLayerAssignmentId.getVaryingId().size() == curvesCount);
    }

    // facevarying/varying/vertex are the same thing for LineSegments
    for (auto& kv : mPrimitiveAttributeTable) {
        for (auto& attribute : kv.second) {
            if (attribute->getRate() == RATE_VARYING ||
                attribute->getRate() == RATE_FACE_VARYING) {
                attribute->setRate(RATE_VERTEX);
            }
        }
    }
    size_t vertexCount = getVertexCount();
    setAttributes(Attributes::interleave(mPrimitiveAttributeTable,
        0, curvesCount, vertexCount, std::vector<size_t>(), vertexCount));
}

void
LineSegments::postIntersect(mcrt_common::ThreadLocalState &tls,
        const scene_rdl2::rdl2::Layer* layer, const mcrt_common::Ray& ray,
        Intersection& intersection) const
{
    const int spanId = ray.primID;
    const IndexData& index = mIndexBuffer[spanId];
    const uint32_t chain = index.mChain;
    const uint32_t vertexOffset = index.mVertex;
    const uint32_t spansInChain = mCurvesVertexCount[chain] - 1;
    const uint32_t indexInChain = index.mSpan;

    Vec3fa cv[2];
    if (!isMotionBlurOn()) {
        // Grab the CVs
        cv[0] = mVertexBuffer(vertexOffset + 0);
        cv[1] = mVertexBuffer(vertexOffset + 1);
    } else {
        // only support two time samples for motionblur at this moment
        MNRY_ASSERT(getMotionSamplesCount() == 2);
        float w1 = ray.time;
        float w0 = 1.0f - w1;
        cv[0] = mVertexBuffer(vertexOffset + 0, 0) * w0 +
                mVertexBuffer(vertexOffset + 0, 1) * w1;

        cv[1] = mVertexBuffer(vertexOffset + 1, 0) * w0 +
                mVertexBuffer(vertexOffset + 1, 1) * w1;
    }

    const int assignmentId =
        mLayerAssignmentId.getType() == LayerAssignmentId::Type::CONSTANT ?
        mLayerAssignmentId.getConstId() :
        mLayerAssignmentId.getVaryingId()[chain];
    intersection.setLayerAssignments(assignmentId, layer);

    const AttributeTable *table =
        intersection.getMaterial()->get<shading::Material>().getAttributeTable();
    intersection.setTable(&tls.mArena, table);
    intersection.setIds(vertexOffset, 0, 0);
    const Attributes* primitiveAttributes = getAttributes();

    // Set the required attributes
    LineSegmentsInterpolator interpolator(primitiveAttributes,
                                          ray.time,
                                          chain,
                                          ray.u,
                                          vertexOffset);

    intersection.setRequiredAttributes(interpolator);

    // Add an interpolated N, dPds, and dPdt to the intersection
    // if they exist in the primitive attribute table
    setExplicitAttributes<LineSegmentsInterpolator>(interpolator,
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
        // Partial with respect to S is simply the tangent vector
        dPds = cv[1] - cv[0];

        // We now render curves as ray-aligned flat ribbons.
        // Use the flipped ray direction to compute the normal.  But, this may be an
        // instance, so we need to transform the ray direction into the local
        // space.  It gets transformed back to render space later on.
        // Note that it looks like we should be using r2l to get from render to
        // local space, but recall that when transforming normals we use the transpose
        // of the inverse, which is l2r transposed.  The transformNormal() function
        // takes the inverse of the transform (l2r) and applies the transpose.
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
    }

    // Geometric normal equals shading normal for a line segment
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
        // Use the 1.5 times the curve radius as the next ray epsilon hint.
        // This lets shadow and continuation rays escape
        // the intersected hair, instead of self-intersecting.
        float curveRadius = cv[0].w * (1.0f - ray.u) + cv[1].w * ray.u;
        intersection.setEpsilonHint(1.5f * curveRadius);
    } else {
        intersection.setEpsilonHint( geometry->getRayEpsilon() );
    }

    // calculate dfds/dfdt for primitive attributes that request differential
    if (table->requestDerivatives()) {
        // ds = stSpanRange -> 1 / ds = 1 / stSpanRange = spansInChain
        computeAttributesDerivatives(table,
                                     static_cast<float>(spansInChain),
                                     vertexOffset,
                                     ray.time,
                                     intersection);
    }

    // For wireframe AOV/shaders
    if (table->requests(StandardAttributes::sNumPolyVertices)) {
        intersection.setAttribute(StandardAttributes::sNumPolyVertices, 2);
    }
    if (table->requests(StandardAttributes::sPolyVertexType)) {
        intersection.setAttribute(StandardAttributes::sPolyVertexType,
            static_cast<int>(StandardAttributes::POLYVERTEX_TYPE_LINE));
    }
    for (int iVert = 0; iVert < 2; iVert++) {
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
        if (isMotionBlurOn()) {
            float wc = ray.getTime() - sHalfDt;
            const Vec3fa v0 = lerp(
                lerp(mVertexBuffer(vertexOffset + 0, 0), mVertexBuffer(vertexOffset + 0, 1), wc),
                lerp(mVertexBuffer(vertexOffset + 1, 0), mVertexBuffer(vertexOffset + 1, 1), wc),
                ray.u);
            wc = ray.getTime() + sHalfDt;
            const Vec3fa v1 = lerp(
                lerp(mVertexBuffer(vertexOffset + 0, 0), mVertexBuffer(vertexOffset + 0, 1), wc),
                lerp(mVertexBuffer(vertexOffset + 1, 0), mVertexBuffer(vertexOffset + 1, 1), wc),
                ray.u);
            pos0 = v0.asVec3f();
            pos1 = v1.asVec3f();
            pos1Ptr = &pos1;
        } else {
            const Vec3fa v0 = lerp(
                mVertexBuffer(vertexOffset + 0, 0), mVertexBuffer(vertexOffset + 1, 0), ray.u);
            pos0 = v0.asVec3f();
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

bool
LineSegments::computeIntersectCurvature(const mcrt_common::Ray& ray,
        const Intersection& intersection, Vec3f& dNds, Vec3f& dNdt) const
{
    uint32_t id1, id2, id3;
    intersection.getIds(id1, id2, id3);

    const uint vertexOffset = id1;

    Vec3fa cv0, cv1;
    if (!isMotionBlurOn()) {
        // Grab the CVs
        cv0 = mVertexBuffer(vertexOffset + 0);
        cv1 = mVertexBuffer(vertexOffset + 1);
    } else {
        // only support two time samples for motionblur at this moment
        MNRY_ASSERT(getMotionSamplesCount() == 2);
        float w1 = ray.time;
        float w0 = 1.0f-w1;
        cv0 = mVertexBuffer(vertexOffset + 0, 0) * w0 +
              mVertexBuffer(vertexOffset + 0, 1) * w1;

        cv1 = mVertexBuffer(vertexOffset + 1, 0) * w0 +
              mVertexBuffer(vertexOffset + 1, 1) * w1;
    }

    Vec3fa centerP = (1.0f - ray.u) * cv0 + ray.u * cv1;
    Vec3fa normal = Vec3fa(intersection.getP(), 0.f) - centerP;

    // Modelled locally as a cylinder, there is no variance of the normal
    // down the length of the cylinder
    dNds = Vec3f(0.0f, 0.0f, 0.0f);

    // We do vary as we extend closer to the edge
    Vec3fa posPlusDeltaT = Vec3fa(intersection.getP() + intersection.getdPdt(), 0.f);
    dNdt = (posPlusDeltaT - centerP) - normal;
    return true;
}

void
LineSegments::computeAttributesDerivatives(const AttributeTable* table,
        float invDs, int vertexOffset,
        float time, Intersection& intersection) const
{
    Attributes* attrs = getAttributes();
    for (auto key: table->getDifferentialAttributes()) {
        if (!attrs->isSupported(key)) {
            continue;
        }
        if (attrs->getRate(key) == RATE_VERTEX) {
            switch (key.getType()) {
            case scene_rdl2::rdl2::TYPE_FLOAT:
                computeVertexAttributeDerivatives(
                    TypedAttributeKey<float>(key),
                    invDs, vertexOffset, time, intersection);
                break;
            case scene_rdl2::rdl2::TYPE_RGB:
                computeVertexAttributeDerivatives(
                    TypedAttributeKey<scene_rdl2::math::Color>(key),
                    invDs, vertexOffset, time, intersection);
                break;
            case scene_rdl2::rdl2::TYPE_RGBA:
                computeVertexAttributeDerivatives(
                    TypedAttributeKey<scene_rdl2::math::Color4>(key),
                    invDs, vertexOffset, time, intersection);
                break;
            case scene_rdl2::rdl2::TYPE_VEC2F:
                computeVertexAttributeDerivatives(
                    TypedAttributeKey<scene_rdl2::math::Vec2f>(key),
                    invDs, vertexOffset, time, intersection);
                break;
            case scene_rdl2::rdl2::TYPE_VEC3F:
                computeVertexAttributeDerivatives(
                    TypedAttributeKey<scene_rdl2::math::Vec3f>(key),
                    invDs, vertexOffset, time, intersection);
                break;
            case scene_rdl2::rdl2::TYPE_MAT4F:
                computeVertexAttributeDerivatives(
                    TypedAttributeKey<scene_rdl2::math::Mat4f>(key),
                    invDs, vertexOffset, time, intersection);
                break;
            default:
                break;
            }
        }
    }
}

} // namespace internal
} // namespace geom
} // namespace moonray

