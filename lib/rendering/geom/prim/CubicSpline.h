// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file CubicSpline.h
///

#pragma once

#include <moonray/rendering/geom/Api.h>
#include <moonray/rendering/geom/prim/Curves.h>

#include <moonray/rendering/bvh/shading/AttributeKey.h>
#include <moonray/rendering/bvh/shading/AttributeTable.h>
#include <moonray/rendering/bvh/shading/Interpolator.h>
#include <moonray/rendering/bvh/shading/Intersection.h>
#include <moonray/rendering/bvh/shading/PrimitiveAttribute.h>

namespace moonray {
namespace geom {
namespace internal {

///
/// @class CubicSpline CubicSpline.h <geom/CubicSpline.h>
/// @brief The CubicSpline class is a Curves primitive constructed
///    from piecewise third-order polynomials. The CubicSpline
///    is divided into "spans" of four control points each.
///
class CubicSpline : public Curves
{
public:
    CubicSpline(Curves::Type type,
                geom::Curves::SubType subtype,
                geom::Curves::CurvesVertexCount&& curvesVertexCount,
                geom::Curves::VertexBuffer&& vertices,
                LayerAssignmentId&& layerAssignmentId,
                shading::PrimitiveAttributeTable&& primitiveAttributeTable);

    // BVH has native support bezier and b-spline intersection so we don't need to implement
    // the intersection kernel by ourself
    virtual bool canIntersect() const override { return false; }

    virtual void postIntersect(mcrt_common::ThreadLocalState& tls,
            const scene_rdl2::rdl2::Layer* pRdlLayer, const mcrt_common::Ray& ray,
            shading::Intersection& intersection) const override;

    virtual bool computeIntersectCurvature(const mcrt_common::Ray& ray,
            const shading::Intersection& intersection,
            scene_rdl2::math::Vec3f& dNds, scene_rdl2::math::Vec3f& dNdt) const override;

    static void intersectionFilter(const RTCFilterFunctionNArguments* args);

private:

    virtual uint32_t getSpansInChain(const uint32_t chain) const = 0;

    // get weights for each of the 4 control points of a span
    virtual void evalWeights(const float& t,
            float& w0, float& w1,
            float& w2, float& w3) const = 0;

    // get weights for the difference of neighboring controls points
    // to scale (p1 - p0), (p2 - p1), and (p3 - p2)
    virtual void evalDerivWeights(const float& t,
            float& w0, float& w1, float& w2) const = 0;

    template<typename CvType>
    CvType evalCubic(const float& t,
            const CvType& cv0, const CvType& cv1,
            const CvType& cv2, const CvType& cv3) const
    {
        float w0, w1, w2, w3;
        evalWeights(t, w0, w1, w2, w3);
        return w0 * cv0 + w1 * cv1 + w2 * cv2 + w3 * cv3;
    }

    template<typename CvType>
    CvType evalCubicDeriv(const float& t,
            const CvType& cv0, const CvType& cv1,
            const CvType& cv2, const CvType& cv3) const
    {
         float w0, w1, w2;
         evalDerivWeights(t, w0, w1, w2);
         return w0 * (cv1 - cv0) + w1 * (cv2 - cv1) + w2 * (cv3 - cv2);
    }

    void computeAttributesDerivatives(const shading::AttributeTable* table,
            float u, float invDs, int chain,
            int varyingOffset, int faceVaryingOffset, int vertexOffset,
            float time, shading::Intersection& intersection) const;

    template<typename T> void
    computeVaryingAttributeDerivatives(shading::TypedAttributeKey<T> key,
            float invDs, int varyingOffset, float time,
            shading::Intersection& intersection) const
    {
        int vid0 = varyingOffset;
        int vid1 = varyingOffset + 1;
        T dfds;
        if (mAttributes->getTimeSampleCount(key) > 1) {
            T f0 = mAttributes->getMotionBlurVarying(key, vid0, time);
            T f1 = mAttributes->getMotionBlurVarying(key, vid1, time);
            dfds = (f1 - f0) * invDs;
        } else {
            T f0 = mAttributes->getVarying(key, vid0);
            T f1 = mAttributes->getVarying(key, vid1);
            dfds = (f1 - f0) * invDs;
        }
        intersection.setdAttributeds(key, dfds);
    }

    template<typename T> void
    computeFaceVaryingAttributeDerivatives(shading::TypedAttributeKey<T> key,
            float invDs, int fid, int faceVaryingOffset, float time,
            shading::Intersection& intersection) const
    {
        int fvid0 = faceVaryingOffset;
        int fvid1 = faceVaryingOffset + 1;
        T dfds;
        if (mAttributes->getTimeSampleCount(key) > 1) {
            T f0 = mAttributes->getMotionBlurFaceVarying(key, fid, fvid0, time);
            T f1 = mAttributes->getMotionBlurFaceVarying(key, fid, fvid1, time);
            dfds = (f1 - f0) * invDs;
        } else {
            const T& f0 = mAttributes->getFaceVarying(key, fid, fvid0);
            const T& f1 = mAttributes->getFaceVarying(key, fid, fvid1);
            dfds = (f1 - f0) * invDs;
        }
        intersection.setdAttributeds(key, dfds);
    }

    template<typename T> void
    computeVertexAttributeDerivatives(shading::TypedAttributeKey<T> key,
            float u, int vertexOffset, float time,
            shading::Intersection& intersection) const
    {
        int vid0 = vertexOffset;
        int vid1 = vertexOffset + 1;
        int vid2 = vertexOffset + 2;
        int vid3 = vertexOffset + 3;
        T dfds;
        if (mAttributes->getTimeSampleCount(key) > 1) {
            T f0 = mAttributes->getMotionBlurVertex(key, vid0, time);
            T f1 = mAttributes->getMotionBlurVertex(key, vid1, time);
            T f2 = mAttributes->getMotionBlurVertex(key, vid2, time);
            T f3 = mAttributes->getMotionBlurVertex(key, vid3, time);
            dfds = evalCubicDeriv(u, f0, f1, f2, f3);
        } else {
            const T& f0 = mAttributes->getVertex(key, vid0);
            const T& f1 = mAttributes->getVertex(key, vid1);
            const T& f2 = mAttributes->getVertex(key, vid2);
            const T& f3 = mAttributes->getVertex(key, vid3);
            dfds = evalCubicDeriv(u, f0, f1, f2, f3);
        }
        intersection.setdAttributeds(key, dfds);
    }

    template<typename T> void
    computeAttributeDerivatives(shading::TypedAttributeKey<T> key,
            float u, float invDs, int chain,
            int varyingOffset, int faceVaryingOffset, int vertexOffset,
            float time, shading::Intersection& intersection) const
    {
        switch (getAttributes()->getRate(key)) {
        case shading::RATE_VARYING:
            computeVaryingAttributeDerivatives(key, invDs, varyingOffset,
                time, intersection);
            break;
        case shading::RATE_FACE_VARYING:
            computeFaceVaryingAttributeDerivatives(key, invDs, chain,
                faceVaryingOffset, time, intersection);
            break;
        case shading::RATE_VERTEX:
            computeVertexAttributeDerivatives(key, u, vertexOffset,
                time, intersection);
            break;
        default:
            break;
        }
    }

};

// CubicSpline is similar to nonperiodic cubic curves in Renderman
// each span has 4 control points
// one or more spans form a chain, each chain is a curve
// one or more chains form a CubicSpline (Curves) primitive
// To interpolate primitive attributes in different rates:
// constant    : should supply a single value for the entire Curves primitive
// uniform     : should supply a total of nCurves values for the entire Curves primitive
// varying     : should supply sum(nSpansi + 1) values
// facevarying : should supply sum(nSpansi + 1) values
// vertex      : should supply sum(nVerticesi) values
//
// For example: a CubicSpline with 2 chains (nCurves = 2)
// chain0 has 3 spans (nSpans0 = 3, nVertices0 = 3 * 3 + 1 = 10 for bezier or 3 + 3 = 6 for b-spline)
// chain1 has 1 spans (nSpans1 = 1, nVertices1 = 3 * 1 + 1 = 4 for bezier or 1 + 3 = 4 for b-spline)
// constant attribute should supply 1 value
// uniform attribute should supply 2 values
// varying/facevarying attribute should supply (3 + 1) + (1 + 1) = 6 values
// vertex attribute should supply 10 + 4 = 14 values for bezier or 6 + 4 = 10 values for b-spline
class CubicSplineInterpolator : public shading::Interpolator
{
public:
    CubicSplineInterpolator(const shading::Attributes *attr, float time,
            int chain, int varyingOffset, int faceVaryingOffset, float u,
            int vertexOffset, float w0, float w1, float w2, float w3):
        shading::Interpolator(attr, time, 0, chain, 2, mVaryingIndex, mLinearWeights,
        2, mFaceVaryingIndex, mLinearWeights,
        chain, 4, mCurveVertices, mCurveWeights)
        {

            mVaryingIndex[0] = varyingOffset;
            mVaryingIndex[1] = varyingOffset + 1;
            mFaceVaryingIndex[0] = faceVaryingOffset;
            mFaceVaryingIndex[1] = faceVaryingOffset + 1;
            mLinearWeights[0] = 1.0f - u;
            mLinearWeights[1] = u;
            for (int i = 0; i < 4; i++) {
                mCurveVertices[i] = vertexOffset + i;
            }
            mCurveWeights[0] = w0;
            mCurveWeights[1] = w1;
            mCurveWeights[2] = w2;
            mCurveWeights[3] = w3;
        }
private:
    int mVaryingIndex[2];
    int mFaceVaryingIndex[2];
    float mLinearWeights[2];
    int mCurveVertices[4];
    float mCurveWeights[4];
};

} // namespace internal
} // namespace geom
} // namespace moonray

