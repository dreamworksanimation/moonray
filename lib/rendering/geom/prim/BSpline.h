// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file BSpline.h
///

#pragma once

#include <moonray/rendering/bvh/shading/PrimitiveAttribute.h>
#include <moonray/rendering/geom/Api.h>
#include <moonray/rendering/geom/prim/CubicSpline.h>

namespace moonray {
namespace geom {
namespace internal {

///
/// @class
/// @brief The BSpline class is a CubicSpline primitive with overlapping
///    spans. Each span shares 3 control points with the subsequent span.
///    We only handle uniform cubic b-splines.
///
///    Here is a visual of the span layout:
///
///    *--*--*--*       span1
///       *--*--*--*    span2

class BSpline : public CubicSpline
{
public:
    BSpline(Curves::Type type,
            geom::Curves::SubType subtype,
            geom::Curves::CurvesVertexCount&& curvesVertexCount,
            geom::Curves::VertexBuffer&& vertices,
            LayerAssignmentId&& layerAssignmentId,
            shading::PrimitiveAttributeTable&& primitiveAttributeTable);

private:

    virtual uint32_t getSpansInChain(const uint32_t chain) const override
    {
        return mCurvesVertexCount[chain] - 3;
    }

    // get weights for each of the 4 control points of a span
    virtual void evalWeights(const float& t, float& w0, float& w1, float& w2, float& w3) const override
    {
        const float s = 1.f - t;
        const float oneSixth(1.0f/6.0f);

        w0 = oneSixth * s * s * s;
        w1 = 0.5f * s * (s * t + 1.0f) + oneSixth;
        w2 = 0.5f * t * (s * t + 1.0f) + oneSixth;
        w3 = oneSixth * t * t * t;
    }

    // get weights for the difference of neighboring controls points
    // to scale (p1 - p0), (p2 - p1), and (p3 - p2)
    virtual void evalDerivWeights(const float& t, float& w0, float& w1, float& w2) const override
    {
        const float s = 1.f - t;
        w0 = 0.5f * s * s;
        w1 = s * t + 0.5f;
        w2 = 0.5f * t * t;
    }

};

} // namespace internal
} // namespace geom
} // namespace moonray


