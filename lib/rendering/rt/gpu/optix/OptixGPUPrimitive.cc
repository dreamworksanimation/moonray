// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#include "OptixGPUPrimitive.h"
#include "OptixGPUInstance.h"

namespace moonray {
namespace rt {

template <typename T>
void clearMemory(std::vector<T>& v)
{
    std::vector<T>{}.swap(v);
}

void
OptixGPUBox::getPrimitiveAabbs(std::vector<OptixAabb>* aabbs) const
{
    OptixAabb localAabb = {-mLength / 2.f, -mHeight / 2.f, -mWidth / 2.f,
                            mLength / 2.f, mHeight / 2.f, mWidth / 2.f};
    aabbs->push_back(mL2P.transformAabb(localAabb));
}

void
OptixGPUCurve::getPrimitiveAabbs(std::vector<OptixAabb>* aabbs) const
{
    aabbs->resize(mHostIndices.size());

    int cpPerCurve = (mBasis == LINEAR) ? 2 : 4;

    for (size_t i = 0; i < mHostIndices.size(); i++) {
        OptixAabb &bb = (*aabbs)[i];
        bb = emptyAabb();
        for (int ms = 0; ms < mMotionSamplesCount; ms++) {
            int idx = ms * mNumControlPoints + mHostIndices[i];
            for (int j = 0; j < cpPerCurve; j++) {
                float4 p = mHostControlPoints[idx + j];
                float3 center = make_float3(p);
                float radius = p.w;
                bb = expandAabb(bb, center - radius);
                bb = expandAabb(bb, center + radius);
            }
        }
    }
}

void
OptixGPUCurve::freeHostMemory()
{
    clearMemory(mHostIndices);
    clearMemory(mHostControlPoints);
}

void
OptixGPUPoints::getPrimitiveAabbs(std::vector<OptixAabb>* aabbs) const
{
    aabbs->resize(mHostPoints.size() / mMotionSamplesCount);

    for (size_t i = 0; i < mHostPoints.size() / mMotionSamplesCount; i++) {
        OptixAabb &bb = (*aabbs)[i];
        bb = emptyAabb();
        for (int ms = 0; ms < mMotionSamplesCount; ms++) {
            size_t idx = i * mMotionSamplesCount + ms;
            float3 point = make_float3(mHostPoints[idx]);
            float radius = mHostPoints[idx].w;
            OptixAabb sampleBB = {point.x - radius, point.y - radius, point.z - radius,
                                  point.x + radius, point.y + radius, point.z + radius};
            bb = combineAabbs(bb, sampleBB);
        }
    }
}

void
OptixGPUPoints::freeHostMemory()
{
    clearMemory(mHostPoints);
}

void
OptixGPUSphere::getPrimitiveAabbs(std::vector<OptixAabb>* aabbs) const
{
    OptixAabb localAabb = {-mRadius, -mRadius, mZMin,
                            mRadius, mRadius, mZMax};
    // TODO: This will create a poor bounding box if the transform has any
    // rotation.  CPU-side Sphere::computeAABB() does the same thing but it's
    // even worse because it doesn't consider z-clipping of the sphere.
    aabbs->push_back(mL2P.transformAabb(localAabb));
}

} // namespace rt
} // namespace moonray

