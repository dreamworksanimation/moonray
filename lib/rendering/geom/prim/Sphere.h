// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file Sphere.h
///

#pragma once

#include <moonray/rendering/geom/prim/NamedPrimitive.h>

namespace moonray {
namespace geom {
namespace internal {

class Sphere : public NamedPrimitive
{
public:
    Sphere(float radius, LayerAssignmentId&& layerAssignmentId,
            shading::PrimitiveAttributeTable&& primitiveAttributeTable);

    virtual PrimitiveType getType() const override
    {
        return QUADRIC;
    }

    virtual int getIntersectionAssignmentId(int primID) const override
    {
        return mLayerAssignmentId.getConstId();
    }

    virtual const scene_rdl2::rdl2::Material* getIntersectionMaterial(
            const scene_rdl2::rdl2::Layer *pRdlLayer,
            const mcrt_common::Ray &ray) const override;

    virtual void postIntersect(mcrt_common::ThreadLocalState& tls,
            const scene_rdl2::rdl2::Layer* pRdlLayer, const mcrt_common::Ray& ray,
            shading::Intersection& intersection) const override;

    virtual BBox3f computeAABB() const override;

    virtual size_t getMemory () const override
    {
        return sizeof(Sphere) - sizeof(NamedPrimitive) + NamedPrimitive::getMemory();
    }

    virtual size_t getMotionSamplesCount() const override
    {
        return 1;
    }

    virtual bool canIntersect() const override
    {
        return true;
    }

    virtual size_t getSubPrimitiveCount() const override
    {
        return 1;
    }

    virtual RTCBoundsFunction getBoundsFunction() const override;

    virtual RTCIntersectFunctionN getIntersectFunction() const override;

    virtual RTCOccludedFunctionN getOccludedFunction() const override;

    virtual bool computeIntersectCurvature(const mcrt_common::Ray &ray,
            const shading::Intersection &intersection,
            Vec3f &dnds, Vec3f &dndt) const override;

    float getRadius() const
    {
        return mRadius;
    }

    float getZMin() const
    {
        return mZMin;
    }

    float getZMax() const
    {
        return mZMax;
    }

    float getPhiMax() const
    {
        return mPhiMax;
    }

    const Mat43& getL2P() const
    {
        return mL2P;
    }

    const Mat43& getP2L() const
    {
        return mP2L;
    }

    void setClippingRange(float zMin, float zMax, float sweepAngle)
    {
        mZMin = scene_rdl2::math::clamp(scene_rdl2::math::min(zMin, zMax), -mRadius, mRadius);
        mZMax = scene_rdl2::math::clamp(scene_rdl2::math::max(zMin, zMax), -mRadius, mRadius);
        mThetaMin = acos(scene_rdl2::math::clamp(mZMin / mRadius, -1.0f, 1.0f));
        mThetaMax = acos(scene_rdl2::math::clamp(mZMax / mRadius, -1.0f, 1.0f));
        mPhiMax = scene_rdl2::math::degreesToRadians(scene_rdl2::math::clamp(sweepAngle, 0.0f, 360.0f));
    }

    void setTransform(const Mat43& xform)
    {
        mL2P = xform;
        mP2L = mL2P.inverse();
    }

    void setIsSingleSided(bool isSingleSided)
    {
        mIsSingleSided = isSingleSided;
    }

    bool getIsSingleSided() const
    {
        return mIsSingleSided;
    }

    void setIsNormalReversed(bool isNormalReversed)
    {
        mIsNormalReversed = isNormalReversed;
    }

    bool getIsNormalReversed() const
    {
        return mIsNormalReversed;
    }

private:
    Mat43 mL2P;
    Mat43 mP2L;
    float mRadius;
    float mPhiMax;
    float mZMin;
    float mZMax;
    float mThetaMin;
    float mThetaMax;
    bool mIsSingleSided;
    bool mIsNormalReversed;
};

} // namespace internal
} // namespace geom
} // namespace moonray

