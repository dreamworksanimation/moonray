// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file EmissionDistribution.cc
///

#include "EmissionDistribution.h"

using namespace scene_rdl2;
namespace moonray {
namespace geom {
namespace internal {

float
EmissionDistribution::pdf(const Transform &xform, const Vec3f& pRender, const Vec3f& wiRender,
        float& tEnd, float time) const
{
    // we do the DDA traversal in grid space
    Vec3f p, wi;
    if (mIsMotionBlurOn) {
        p  = scene_rdl2::math::transformPoint( xform.getRenderToDist(time), pRender);
        wi = scene_rdl2::math::transformVector(xform.getRenderToDist(time), wiRender);
    } else {
        p  = scene_rdl2::math::transformPoint( xform.getRenderToDist()[0], pRender);
        wi = scene_rdl2::math::transformVector(xform.getRenderToDist()[0], wiRender);
    }
    float lengthScale = 1.0f / wi.length();
    wi *= lengthScale;

    // bounding box intersection test
    float tStart = 0.0f;
    tEnd = FLT_MAX;
    for (int axis = 0; axis < 3; ++axis) {
        float invDir = 1.0f / wi[axis];
        float tnear = (0 - p[axis]) * invDir;
        float tfar  = (mRes[axis] - p[axis]) * invDir;
        if (tnear > tfar) {
            std::swap(tnear, tfar);
        }
        tStart = tnear > tStart ? tnear : tStart;
        tEnd = tfar < tEnd ? tfar : tEnd;
        // the ray doesn't hit bounding box
        if (tStart > tEnd) {
            return 0.0f;
        }
    }
    // the tEnd used for later line integral computation is in render space
    tEnd *= lengthScale;
    // the tStart used for grid traversal is in local grid space
    p = p + tStart * wi;
    float nextT[3];
    float deltaT[3];
    int step[3];
    int out[3];
    scene_rdl2::math::Vec3i pos;
    for (int axis = 0; axis < 3; ++axis) {
        // the clamping below prevents resolved intersection point falls out of
        // bounding box due to float point precision issue
        pos[axis] = scene_rdl2::math::clamp(static_cast<int>(p[axis]), 0, mRes[axis] - 1);
        if (wi[axis] >= 0) {
            nextT[axis] = tStart + (pos[axis] + 1 - p[axis]) / wi[axis];
            deltaT[axis] = 1.0f / wi[axis];
            step[axis] = 1;
            out[axis] = mRes[axis];
        } else {
            nextT[axis] = tStart + (pos[axis] - p[axis]) / wi[axis];
            deltaT[axis] = -1.0f / wi[axis];
            step[axis] = -1;
            out[axis] = -1;
        }
    }

    float tCurrent = tStart;
    float accumedPdf = 0.0f;
    while (true) {
        float pdfD = pdfDiscrete(pos);
        int stepAxis =
            ((nextT[1] < nextT[0]) | (nextT[2] < nextT[0])) <<
            (nextT[2] < nextT[1]);
        if (pdfD > 0.f) {
            float t0 = tCurrent * lengthScale;
            float t1 = nextT[stepAxis] * lengthScale;
            // the (t1^3 - t0^3) / 3 part stands for integrating 3d space pdf
            // along the line segment enter/exit grid voxel with Jacobian transform
            // For detail reference see:
            // "Line Integration for Rendering Heterogeneous Emissive Volume"
            // EGSR2017 Florian Simon
            // equation (15) and Algorithm 2
            accumedPdf += pdfD * xform.getInvUnitVolume() *
                (t1 * t1 * t1 - t0 * t0 * t0) / 3.0f;
        }
        tCurrent = nextT[stepAxis];
        pos[stepAxis] += step[stepAxis];
        // getting out of the bounding box
        if (pos[stepAxis] == out[stepAxis]) {
            break;
        }
        nextT[stepAxis] += deltaT[stepAxis];
    }
    return accumedPdf;
}

void
DenseEmissionDistribution::sample(const Transform &xform, const Vec3f& p, float u1, float u2, float u3,
        Vec3f& wi, float& pdfWi, float& tEnd, float time) const
{
    float pdfIndex;
    float uxRemapped, uyRemapped, uzRemapped;
    scene_rdl2::math::Vec3i coord = mDistribution->sampleDiscrete(u1, u2, u3,
        &pdfIndex, &uxRemapped, &uyRemapped, &uzRemapped);
    Vec3f pDist(
        coord[0] + uxRemapped,
        coord[1] + uyRemapped,
        coord[2] + uzRemapped);
    Vec3f pt = mIsMotionBlurOn ? scene_rdl2::math::transformPoint(xform.getDistToRender(time), pDist) :
                                 scene_rdl2::math::transformPoint(xform.getDistToRender()[0], pDist);
    wi = normalize(pt - p);
    pdfWi = pdf(xform, p, wi, tEnd, time);
}

Vec3f
DenseEmissionDistribution::sample(const Transform &xform, float u1, float u2, float u3, float time) const
{
    float pdfIndex;
    float uxRemapped, uyRemapped, uzRemapped;
    scene_rdl2::math::Vec3i coord = mDistribution->sampleDiscrete(u1, u2, u3,
        &pdfIndex, &uxRemapped, &uyRemapped, &uzRemapped);
    Vec3f pDist(
        coord[0] + uxRemapped,
        coord[1] + uyRemapped,
        coord[2] + uzRemapped);
    return mIsMotionBlurOn ? scene_rdl2::math::transformPoint(xform.getDistToRender(time), pDist) :
                             scene_rdl2::math::transformPoint(xform.getDistToRender()[0], pDist);
}

float
DenseEmissionDistribution::pdfDiscrete(const scene_rdl2::math::Vec3i& pos) const
{
    return mDistribution->pdfDiscrete(pos);
}

} // namespace internal
} // namespace geom
} // namespace moonray

