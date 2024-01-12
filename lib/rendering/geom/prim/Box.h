// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file Box.h
///

#pragma once

#include <moonray/rendering/geom/prim/NamedPrimitive.h>

#include <moonray/rendering/bvh/shading/Intersection.h>

namespace moonray {
namespace geom {
namespace internal {

class Box : public NamedPrimitive
{
public:
    Box(float length, float width, float height,
            LayerAssignmentId&& layerAssignmentId,
            shading::PrimitiveAttributeTable&& primitiveAttributeTable);

    virtual PrimitiveType getType() const override
    {
        return QUADRIC;
    }

    virtual const scene_rdl2::rdl2::Material* getIntersectionMaterial(
            const scene_rdl2::rdl2::Layer *pRdlLayer,
            const mcrt_common::Ray &ray) const override;

    virtual int getIntersectionAssignmentId(int primID) const override
    {
        return mLayerAssignmentId.getConstId();
    }

    virtual void postIntersect(mcrt_common::ThreadLocalState& tls,
            const scene_rdl2::rdl2::Layer* pRdlLayer, const mcrt_common::Ray& ray,
            shading::Intersection& intersection) const override;

    virtual BBox3f computeAABB() const override;

    virtual size_t getMemory () const override
    {
        return sizeof(Box) - sizeof(NamedPrimitive) + NamedPrimitive::getMemory();
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

    float getLength() const
    {
        return mLength;
    }

    float getWidth() const
    {
        return mWidth;
    }

    float getHeight() const
    {
        return mHeight;
    }

    Vec3f getSize() const
    {
        return Vec3f(mLength, mWidth, mHeight);
    }

    // box should be centered at (0,0,0)
    Vec3f getMinCorner() const
    {
        return Vec3f(-mLength/2, -mHeight/2, -mWidth/2);
    }

    Vec3f getMaxCorner() const
    {
        return Vec3f(mLength/2, mHeight/2, mWidth/2);
    }

    const Mat43& getL2P() const
    {
        return mL2P;
    }

    const Mat43& getP2L() const
    {
        return mP2L;
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
    float mLength;
    float mWidth;
    float mHeight;
    bool mIsSingleSided;
    bool mIsNormalReversed;
};

} // namespace internal
} // namespace geom
} // namespace moonray

