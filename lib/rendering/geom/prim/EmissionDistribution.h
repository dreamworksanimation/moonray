// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file EmissionDistribution.h
///
#pragma once

#include <moonray/rendering/geom/Types.h>
#include <moonray/rendering/geom/prim/Util.h>
#include <scene_rdl2/common/math/Mat4.h>

namespace moonray {
namespace geom {
namespace internal {

// EmissionDistribution provides a 3D pdf lookup table to sample a
// point based on emission energy distribution.
// Note the pdf it returns is not in 3D space but solid angle space,
// which describes the accumulated pdf of sampling a direction wi
// formed by connecting input p and sampled point
// The reason for this is we integrate volume contribution in
// solid angle domain instead of Cartesian domain
// (the total emission energy contribution along the wi direction
// instead of just the sampled point)
class EmissionDistribution
{
public:
    // EmissionDistributon::Transform stores transform information
    // between distribution space and render space.
    class Transform
    {
    public:
        Transform(const scene_rdl2::math::Mat4f distToRender[2], float invUnitVolume):
            mInvUnitVolume(invUnitVolume)
        {
            mDistToRender[0] = distToRender[0];
            mDistToRender[1] = distToRender[1];
            mRenderToDist[0] = distToRender[0].inverse();
            mRenderToDist[1] = distToRender[1].inverse();
        }
        const scene_rdl2::math::Mat4f *getDistToRender() const { return mDistToRender; }
        const scene_rdl2::math::Mat4f *getRenderToDist() const { return mRenderToDist; }
        const scene_rdl2::math::Mat4f getDistToRender(float time) const { return lerp(mDistToRender[0], mDistToRender[1], time); }
        const scene_rdl2::math::Mat4f getRenderToDist(float time) const { return lerp(mRenderToDist[0], mRenderToDist[1], time); }
        float getInvUnitVolume() const { return mInvUnitVolume; }
    private:
        scene_rdl2::math::Mat4f mDistToRender[2];
        scene_rdl2::math::Mat4f mRenderToDist[2];
        float mInvUnitVolume;
    };

    EmissionDistribution(const scene_rdl2::math::Vec3i& res,
            const scene_rdl2::math::Mat4f distToRender[2],
            float invUnitVolume):
        mTransform(distToRender, invUnitVolume),
        mRes(res)
    {
        mIsMotionBlurOn = (distToRender[0] != distToRender[1]);
    }

    virtual ~EmissionDistribution() = default;

    // What is the size of the distribution, i.e. how many buckets in
    // the histogram.
    virtual int count() const = 0;

    // given a shading point p, draw a direction wi based on the emission energy
    // distribution represented here and return corresponding solid angle pdf
    // in pdfWi, t interval exiting this emission region in tEnd
    virtual void sample(const Transform &xform, const scene_rdl2::math::Vec3f& p, float u1, float u2, float u3,
                        scene_rdl2::math::Vec3f& wi, float& pdfWi, float& tEnd, float time) const = 0;

    // sample a position based on emission energy distribution represented here
    virtual scene_rdl2::math::Vec3f sample(const Transform &xform, float u1, 
                                           float u2, float u3, float time) const = 0;

    // given a shading point pRender, evaluate the accumulated pdf along
    // direction wiRender extended from pRender and the t interval
    // exiting this emission region (tEnd)
    float pdf(const Transform &xform, const scene_rdl2::math::Vec3f& pRender, 
              const scene_rdl2::math::Vec3f& wiRender, float& tEnd, float time) const;

    // Get the transform between distribution space and render space.
    // If this distribution is a shared reference (i.e. an instancing ref) then
    // the transform will just take us between the distribution and the world origin.
    const Transform &getTransform() const { return mTransform; }

protected:
    // Evaluate the discrete pdf at a point pos
    virtual float pdfDiscrete(const scene_rdl2::math::Vec3i& pos) const = 0;

    // transform between distribution space and render space
    bool mIsMotionBlurOn;
    Transform mTransform;
    scene_rdl2::math::Vec3i mRes;
};


// DenseEmissionDistribution uses the bounding box around the volume.
// For VdbVolumes, the dimensions of the bounding box are voxels_wide * voxels_long * voxels_high.

class DenseEmissionDistribution : public EmissionDistribution
{
public:
    DenseEmissionDistribution(const scene_rdl2::math::Vec3i& res,
            const scene_rdl2::math::Mat4f distToRender[2],
            float invUnitVolume,
            const std::vector<float>& histogram):
        EmissionDistribution(res, distToRender, invUnitVolume)

    {
        mDistribution.reset(new Distribution3D(histogram.data(),
            mRes[0], mRes[1], mRes[2]));
    }

    virtual ~DenseEmissionDistribution() = default;

    virtual int count() const override
    {
        return mDistribution->count();
    }

    // given a shading point p, draw a direction wi based on the emission energy
    // distribution represented here and return corresponding solid angle pdf
    // in pdfWi, t interval exiting this emission region in tEnd
    void sample(const Transform &xform, const scene_rdl2::math::Vec3f& p, float u1, float u2, float u3,
            Vec3f& wi, float& pdfWi, float& tEnd, float time) const override;

    // sample a position based on emission energy distribution represented here
    scene_rdl2::math::Vec3f sample(const Transform &xform, float u1, float u2, float u3, float time) const override;

private:
    // Evaluate the discrete pdf at a point pos
    float pdfDiscrete(const scene_rdl2::math::Vec3i& pos) const override;

    std::unique_ptr<Distribution3D> mDistribution;
};

} // namespace internal
} // namespace geom
} // namespace moonray

