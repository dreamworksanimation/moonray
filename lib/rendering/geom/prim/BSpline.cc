// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file BSpline.cc
///

#include "BSpline.h"

#include <moonray/rendering/geom/prim/BVHUserData.h>
#include <moonray/rendering/geom/prim/Instance.h>
#include <moonray/rendering/geom/prim/MotionTransform.h>
#include <moonray/rendering/geom/prim/Util.h>
#include <moonray/rendering/mcrt_common/ThreadLocalState.h>

#include <scene_rdl2/common/math/Math.h>

// using namespace scene_rdl2::math; // can't use this as it breaks openvdb in clang.
using namespace moonray::shading;

namespace moonray {
namespace geom {
namespace internal {

BSpline::BSpline(
        Curves::Type type,
        geom::Curves::SubType subtype,
        geom::Curves::CurvesVertexCount&& curvesVertexCount,
        geom::Curves::VertexBuffer&& vertices,
        LayerAssignmentId&& layerAssignmentId,
        PrimitiveAttributeTable&& primitiveAttributeTable):
        CubicSpline(type,
                    subtype,
                    std::move(curvesVertexCount),
                    std::move(vertices),
                    std::move(layerAssignmentId),
                    std::move(primitiveAttributeTable))
{
    // allocate/calculate index buffer
    size_t curvesCount = getCurvesCount();
    std::vector<uint32_t> spansInChain(curvesCount, 0);
    size_t varyingCount = 0;
    std::vector<size_t> faceVaryingCount;
    faceVaryingCount.reserve(curvesCount);
    size_t spanCount = 0;
    for (size_t i = 0; i < curvesCount; ++i) {
        spansInChain[i] = mCurvesVertexCount[i] - 3;
        spanCount += spansInChain[i];
        varyingCount += spansInChain[i] + 1;
        faceVaryingCount.push_back(spansInChain[i] + 1);
    }
    mSpanCount = spanCount;
    // allocate/fill the index buffer
    mIndexBuffer.reserve(mSpanCount);
    size_t vertexOffset = 0;
    for (size_t i = 0; i < curvesCount; ++i) {
        for (size_t j = 0; j < spansInChain[i]; ++j) {
            mIndexBuffer.emplace_back(vertexOffset, i, j);
            // A span has 4 control points, however it shares 3
            // control points with the subsequent span. The index
            // buffer points to the first vertex in a span, so
            // we only increment the vertexOffset by 1.
            vertexOffset += 1;
        }
        vertexOffset += 3;
    }

    if (mLayerAssignmentId.getType() == LayerAssignmentId::Type::VARYING) {
        MNRY_ASSERT_REQUIRE(
            mLayerAssignmentId.getVaryingId().size() == curvesCount);
    }
    setAttributes(Attributes::interleave(mPrimitiveAttributeTable,
        0, curvesCount, varyingCount, faceVaryingCount, getVertexCount()));
}

} // namespace internal
} // namespace geom
} // namespace moonray



