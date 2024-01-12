// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file Curves.cc
/// $Id$
///

#include "Curves.h"

#include <moonray/rendering/geom/ProceduralContext.h>
#include <moonray/rendering/geom/PrimitiveVisitor.h>
#include <moonray/rendering/geom/State.h>

#include <moonray/rendering/bvh/shading/PrimitiveAttribute.h>
#include <moonray/rendering/geom/prim/BezierSpanChains.h>
#include <moonray/rendering/geom/prim/BSpline.h>
#include <moonray/rendering/geom/prim/LineSegments.h>
#include <scene_rdl2/render/util/stdmemory.h>

namespace moonray {
namespace geom {

using namespace moonray::shading;

struct Curves::Impl
{
    Impl(internal::Curves* curves,
         Curves::Type type,
         Curves::SubType subtype,
         int tessellationRate) :
        mCurves(curves),
        mType(type),
        mSubType(subtype),
        mTessellationRate(tessellationRate) {}

    std::unique_ptr<internal::Curves> mCurves;
    Curves::Type mType;
    Curves::SubType mSubType;
    int mTessellationRate;
};

Curves::Curves(Type type,
               SubType subtype,
               int tessellationRate,
               CurvesVertexCount&& curvesVertexCount,
               VertexBuffer&& vertices,
               LayerAssignmentId&& layerAssignmentId,
               PrimitiveAttributeTable&& primitiveAttributeTable)
{
    if (type == Type::LINEAR) {
        mImpl = fauxstd::make_unique<Impl>(new internal::LineSegments(
            static_cast<internal::Curves::Type>(type),
            subtype,
            std::move(curvesVertexCount),
            std::move(vertices), std::move(layerAssignmentId),
            std::move(primitiveAttributeTable)),
            type, subtype, tessellationRate);
    } else if (type == Type::BEZIER) {
        mImpl = fauxstd::make_unique<Impl>(new internal::BezierSpanChains(
            static_cast<internal::Curves::Type>(type),
            subtype,
            std::move(curvesVertexCount),
            std::move(vertices), std::move(layerAssignmentId),
            std::move(primitiveAttributeTable)),
            type, subtype, tessellationRate);
    } else if (type == Type::BSPLINE) {
        mImpl = fauxstd::make_unique<Impl>(new internal::BSpline(
            static_cast<internal::Curves::Type>(type),
            subtype,
            std::move(curvesVertexCount),
            std::move(vertices), std::move(layerAssignmentId),
            std::move(primitiveAttributeTable)),
            type, subtype, tessellationRate);
    } else {
        // only support linear/bezier/bspline curve at this moment
        MNRY_ASSERT_REQUIRE(false);
    }
}

Curves::~Curves() = default;

void
Curves::accept(PrimitiveVisitor& v)
{
    v.visitCurves(*this);
}

Primitive::size_type
Curves::getCurvesCount() const
{
    return mImpl->mCurves->getCurvesCount();
}

Curves::Type
Curves::getCurvesType() const
{
    return mImpl->mType;
}

Curves::SubType
Curves::getCurvesSubType() const
{
    return mImpl->mSubType;
}

int
Curves::getTessellationRate() const
{
    return mImpl->mTessellationRate;
}

const Curves::CurvesVertexCount&
Curves::getCurvesVertexCount() const
{
    return mImpl->mCurves->getCurvesVertexCount();
}

Primitive::size_type
Curves::getMemory() const
{
    return sizeof(Curves) + mImpl->mCurves->getMemory();
}

Primitive::size_type
Curves::getMotionSamplesCount() const
{
    return mImpl->mCurves->getMotionSamplesCount();
}

Curves::VertexBuffer&
Curves::getVertexBuffer()
{
    return mImpl->mCurves->getVertexBuffer();
}

const Curves::VertexBuffer&
Curves::getVertexBuffer() const
{
    return mImpl->mCurves->getVertexBuffer();
}

void
Curves::setName(const std::string& name)
{
    mImpl->mCurves->setName(name);
}

const std::string&
Curves::getName() const
{
    return mImpl->mCurves->getName();
}

void
Curves::setCurvedMotionBlurSampleCount(int count)
{
    mImpl->mCurves->setCurvedMotionBlurSampleCount(count);
}

static Primitive::DataValidness
checkLineSegmentsData(
        const Curves::CurvesVertexCount& curvesVertexCount,
        const Curves::VertexBuffer& vertices,
        const PrimitiveAttributeTable& primitiveAttributeTable,
        std::string* message)
{
    size_t curvesCount = curvesVertexCount.size();
    size_t verticesCount = 0;
    for (size_t i = 0; i < curvesCount; ++i) {
        size_t n = curvesVertexCount[i];
        // need at leat 2 vertices to form a line segment
        if (n < 2) {
            if (message) {
                std::stringstream errMsg;
                errMsg << "Invalid number of vertices (" << n <<
                    ") in a curve. A linear line segment should contain at least"
                    " two vertices per curve.";
                *message = errMsg.str();
            }
            return Primitive::DataValidness::INVALID_TOPOLOGY;
        }
        verticesCount += n;
    }
    if (vertices.size() != verticesCount) {
        if (message) {
            std::stringstream errMsg;
            errMsg << "Vertex buffer size (" << vertices.size() <<
                ") doesn't match the total number of vertices " <<
                verticesCount << " index buffer specified.";
            *message = errMsg.str();
        }
        return Primitive::DataValidness::INVALID_VERTEX_BUFFER;
    }
    return Primitive::DataValidness::VALID;
}

static Primitive::DataValidness
checkCubicCurvesData(
        const Curves::CurvesVertexCount& curvesVertexCount,
        const Curves::VertexBuffer& vertices,
        const PrimitiveAttributeTable& primitiveAttributeTable,
        const Curves::Type type,
        const int tessellationRate,
        std::string* message)
{
    size_t curvesCount = curvesVertexCount.size();
    size_t varyingsCount = 0;
    size_t faceVaryingsCount = 0;
    size_t verticesCount = 0;
    // each bezier curve should have 3 * k + 1 vertices
    // since each span has 4 vertices and the neighbor
    // spans share one vertex
    for (size_t i = 0; i < curvesCount; ++i) {
        size_t n = curvesVertexCount[i];
        // need at least 4 vertices to form a cubic spline span
        if (type == Curves::Type::BEZIER && (n - 1) % 3 != 0) {
            if (message) {
                std::stringstream errMsg;
                errMsg << "Invalid number of vertices (" << n <<
                    ") in a curve. A Bezier curve should be 3 * k + 1 vertices.";
                *message = errMsg.str();
            }
            return Primitive::DataValidness::INVALID_TOPOLOGY;
        }
        if (n < 4) {
            if (message) {
                std::stringstream errMsg;
                errMsg << "Invalid number of vertices (" << n <<
                    ") in a curve. A CubicSpline curve should have at least 4 vertices.";
                *message = errMsg.str();
            }
            return Primitive::DataValidness::INVALID_TOPOLOGY;
        }
        size_t spansInCurve = 0;
        if (type == Curves::Type::BEZIER) {
            spansInCurve = (n - 1) / 3;
        } else if (type == Curves::Type::BSPLINE) {
            spansInCurve = n - 3;
        }
        varyingsCount += spansInCurve + 1;
        faceVaryingsCount += spansInCurve + 1;
        verticesCount += n;
    }
    if (vertices.size() != verticesCount) {
        if (message) {
            std::stringstream errMsg;
            errMsg << "Vertex buffer size (" << vertices.size() <<
                ") doesn't match the total number of vertices " <<
                verticesCount << " index buffer specified.";
            *message = errMsg.str();
        }
        return Primitive::DataValidness::INVALID_VERTEX_BUFFER;
    }
    if (tessellationRate <= 0) {
        if (message) {
            std::stringstream errMsg;
            errMsg << "Invalid tessellation rate (" << tessellationRate <<
                "). Must be > 0.";
            *message = errMsg.str();
        }
        return Primitive::DataValidness::INVALID_TOPOLOGY;
    }
    return Primitive::DataValidness::VALID;
}

Primitive::DataValidness
Curves::checkPrimitiveData(Type type,
        SubType subtype,
        int tessellationRate,
        const CurvesVertexCount& curvesVertexCount,
        const VertexBuffer& vertices,
        const PrimitiveAttributeTable& primitiveAttributeTable,
        std::string* message)
{
    Primitive::DataValidness result;
    switch (type) {
    case Type::LINEAR:
        result = checkLineSegmentsData(curvesVertexCount, vertices,
            primitiveAttributeTable, message);
        break;
    case Type::BEZIER:
    case Type::BSPLINE:
        result = checkCubicCurvesData(curvesVertexCount, vertices,
            primitiveAttributeTable, type, tessellationRate, message);
        break;
    default:
        {
            if (message) {
                *message = "Invalid curves type";
            }
            result = DataValidness::INVALID_TOPOLOGY;
        }
        break;
    }

    return result;
}

void
Curves::transformPrimitive(
        const MotionBlurParams& motionBlurParams,
        const XformSamples& prim2render)
{
    size_t motionSamplesCount = getMotionSamplesCount();
    XformSamples p2r = prim2render;
    if (motionSamplesCount > 1 && prim2render.size() == 1) {
        p2r.resize(motionSamplesCount, prim2render[0]);
    }

    const PrimitiveAttributeTable* primAttrTab = mImpl->mCurves->getPrimitiveAttributeTable();
    transformVertexBuffer(mImpl->mCurves->getVertexBuffer(), p2r, motionBlurParams,
                          mImpl->mCurves->getMotionBlurType(), mImpl->mCurves->getCurvedMotionBlurSampleCount(),
                          primAttrTab);

    float shutterOpenDelta, shutterCloseDelta;
    motionBlurParams.getMotionBlurDelta(shutterOpenDelta, shutterCloseDelta);
    mImpl->mCurves->getAttributes()->transformAttributes(p2r,
                                                         shutterOpenDelta,
                                                         shutterCloseDelta,
                                                         {{StandardAttributes::sNormal, Vec3Type::NORMAL},
                                                         {StandardAttributes::sdPds, Vec3Type::VECTOR},
                                                         {StandardAttributes::sdPdt, Vec3Type::VECTOR}});
}

internal::Primitive*
Curves::getPrimitiveImpl()
{
    return mImpl->mCurves.get();
}

} // namespace geom
} // namespace moonray

