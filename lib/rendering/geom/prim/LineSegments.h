// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file LineSegments.h
///

#pragma once

#include <moonray/rendering/geom/Api.h>
#include <moonray/rendering/geom/prim/Curves.h>

#include <moonray/rendering/bvh/shading/AttributeKey.h>

namespace moonray {
namespace geom {
namespace internal {

/// @brief The LineSegments class is a Curves primitive representing
///     group of "chains" of linear line spans. Each chain can have
///     varying number of vertices, and adjacent vertices in a chain
///     form a line span
class LineSegments : public Curves
{
public:
    LineSegments(Curves::Type type,
                 geom::Curves::SubType subtype,
                 geom::Curves::CurvesVertexCount&& curvesVertexCount,
                 geom::Curves::VertexBuffer&& vertices,
                 LayerAssignmentId&& layerAssignmentId,
                 shading::PrimitiveAttributeTable&& primitiveAttributeTable);

    // BVH has native support line intersection so we don't need to implement
    // the intersection kernel by ourself
    virtual bool canIntersect() const override { return false; }

    virtual void postIntersect(mcrt_common::ThreadLocalState& tls,
            const scene_rdl2::rdl2::Layer* pRdlLayer, const mcrt_common::Ray& ray,
            shading::Intersection& intersection) const override;

    virtual bool computeIntersectCurvature(const mcrt_common::Ray& ray,
            const shading::Intersection& intersection,
            scene_rdl2::math::Vec3f& dNds, scene_rdl2::math::Vec3f& dNdt) const override;

private:

    void computeAttributesDerivatives(const shading::AttributeTable* table,
            float invDs, int vertexOffset,
            float time, shading::Intersection& intersection) const;

    template<typename T> void
    computeVertexAttributeDerivatives(shading::TypedAttributeKey<T> key,
            float invDs, int vertexOffset,
            float time, shading::Intersection& intersection) const
    {
        int vid1 = vertexOffset;
        int vid2 = vertexOffset + 1;
        T dfds;
        if (mAttributes->getTimeSampleCount(key) > 1) {
            T f1 = mAttributes->getMotionBlurVertex(key, vid1, time);
            T f2 = mAttributes->getMotionBlurVertex(key, vid2, time);
            dfds = (f2 - f1) * invDs;
        } else {
            const T& f1 = mAttributes->getVertex(key, vid1);
            const T& f2 = mAttributes->getVertex(key, vid2);
            dfds = (f2 - f1) * invDs;
        }
        intersection.setdAttributeds(key, dfds);
    }
};

} // namespace internal
} // namespace geom
} // namespace moonray

