// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file BezierSpanChains.h
///

#pragma once

#include <moonray/rendering/geom/prim/CubicSpline.h>
#include <moonray/rendering/geom/prim/Curves.h>

#include <moonray/rendering/bvh/shading/Attributes.h>
#include <moonray/rendering/bvh/shading/AttributeKey.h>
#include <moonray/rendering/bvh/shading/AttributeTable.h>
#include <moonray/rendering/bvh/shading/Interpolator.h>
#include <moonray/rendering/geom/Api.h>

namespace moonray {

namespace shading { class Intersection; }

namespace geom {
namespace internal {

///
/// @class BezierSpanChains BezierSpanChains.h <geom/BezierSpanChains.h>
/// @brief The BezierSpanChains class is a CubicSpline primitive representing
///     group of "chains" of cubic Bezier spans.  They are a "chain"
///     because we deduplicate the CV linking two Bezier spans.
///
///    Here is a visual of the span layout:
///
///    *--*--*--*             span1
///             *--*--*--*    span2

class BezierSpanChains : public CubicSpline
{
public:
    BezierSpanChains(Curves::Type type,
                     geom::Curves::SubType subtype,
                     geom::Curves::CurvesVertexCount&& curvesVertexCount,
                     geom::Curves::VertexBuffer&& vertices,
                     LayerAssignmentId&& layerAssignmentId,
                     shading::PrimitiveAttributeTable&& primitiveAttributeTable);

private:

    virtual uint32_t getSpansInChain(const uint32_t chain) const override
    {
        return (mCurvesVertexCount[chain] - 1) / 3;
    }

    // get weights for each of the 4 control points of a span
    virtual void evalWeights(const float& t, float& w0, float& w1, float& w2, float& w3) const override
    {
        const float s = 1.f - t;
        w0 = s * s * s;
        w1 = 3.0f * s * s * t;
        w2 = 3.0f * s * t * t;
        w3 = t * t * t;
    }

    // get weights for the difference of neighboring controls points
    // to scale (p1 - p0), (p2 - p1), and (p3 - p2)
    virtual void evalDerivWeights(const float& t, float& w0, float& w1, float& w2) const override
    {
        const float s = 1.0f - t;
        w0 = 3.0f * s * s;
        w1 = 6.0f * s * t;
        w2 = 3.0f * t * t;
    }

};

} // namespace internal
} // namespace geom
} // namespace moonray

