// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file Points.h
///

#pragma once

#include <moonray/rendering/geom/prim/NamedPrimitive.h>

#include <moonray/rendering/geom/Points.h>

namespace moonray {
namespace geom {
namespace internal {


class Points : public NamedPrimitive
{
public:
    Points(geom::Points::VertexBuffer&& position,
            geom::Points::RadiusBuffer&& radius,
            LayerAssignmentId&& layerAssignmentId,
            shading::PrimitiveAttributeTable&& primitiveAttributeTable);

    virtual PrimitiveType getType() const override
    {
        return QUADRIC;
    }

    virtual int getIntersectionAssignmentId(int primID) const override
    {
        int assignmentId =
            mLayerAssignmentId.getType() == LayerAssignmentId::Type::CONSTANT ?
            mLayerAssignmentId.getConstId() :
            mLayerAssignmentId.getVaryingId()[primID];
        return assignmentId;
    }

    virtual const scene_rdl2::rdl2::Material* getIntersectionMaterial(
            const scene_rdl2::rdl2::Layer *pRdlLayer,
            const mcrt_common::Ray &ray) const override;

    virtual void postIntersect(mcrt_common::ThreadLocalState& tls,
            const scene_rdl2::rdl2::Layer* pRdlLayer, const mcrt_common::Ray& ray,
            shading::Intersection& intersection) const override;

    virtual BBox3f computeAABB() const override;
    virtual BBox3f computeAABBAtTimeStep(int timeStep) const override;

    virtual size_t getMemory() const override
    {
        size_t mem = sizeof(Points) - sizeof(NamedPrimitive) + NamedPrimitive::getMemory();

        mem += mPosition.get_memory_usage();
        mem += scene_rdl2::util::getVectorElementsMemory(mRadius);
        return  mem;
    }

    virtual size_t getMotionSamplesCount() const override
    {
        return mPosition.get_time_steps();
    }

    virtual bool canIntersect() const override
    {
        return true;
    }

    virtual size_t getSubPrimitiveCount() const override
    {
        return mPosition.size();
    }

    virtual RTCBoundsFunction getBoundsFunction() const override;

    virtual RTCIntersectFunctionN getIntersectFunction() const override;

    virtual RTCOccludedFunctionN getOccludedFunction() const override;

    geom::Points::VertexBuffer& getVertexBuffer()
    {
        return mPosition;
    }

    const geom::Points::VertexBuffer& getVertexBuffer() const
    {
        return mPosition;
    }

    const geom::Points::RadiusBuffer& getRadiusBuffer() const
    {
        return mRadius;
    }

    const shading::PrimitiveAttributeTable* getPrimitiveAttributeTable() const { return &mPrimitiveAttributeTable; }

    void setCurvedMotionBlurSampleCount(uint32_t count)
    {
        mCurvedMotionBlurSampleCount = count;
    }

    int getCurvedMotionBlurSampleCount()
    {
        return mCurvedMotionBlurSampleCount;
    }

    scene_rdl2::rdl2::MotionBlurType getMotionBlurType()
    {
        return mMotionBlurType;
    }

private:
    // TODO maybe interleave position and radius? (as Vec3fa buffer)
    geom::Points::VertexBuffer mPosition;
    geom::Points::RadiusBuffer mRadius;

protected:
    shading::PrimitiveAttributeTable mPrimitiveAttributeTable;
    uint32_t mCurvedMotionBlurSampleCount;
    scene_rdl2::rdl2::MotionBlurType mMotionBlurType;
};

} // namespace internal
} // namespace geom
} // namespace moonray

