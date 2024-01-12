// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file PathGuide.h
#pragma once

#include <scene_rdl2/common/math/BBox.h>
#include <scene_rdl2/common/math/Color.h>
#include <scene_rdl2/common/math/Vec3.h>

#include <memory>

namespace scene_rdl2 {
namespace rdl2 { class SceneVariables; }
}

namespace moonray {
namespace pbr {

class PathGuide
{
public:
    PathGuide();
    PathGuide(const PathGuide &) = delete;
    PathGuide &operator=(const PathGuide &) = delete;
    ~PathGuide();

    // Initialize path guide for a new frame.  Bbox is an aabb of the scene.
    void startFrame(const scene_rdl2::math::BBox3f &bbox, const scene_rdl2::rdl2::SceneVariables &vars);

    // A path guided render should be broken into a series of passes where
    // each pass covers the entire frame and each new pass should contain roughly
    // twice the number of samples as the previous.
    //
    // This method is not thread safe.  It should be called only from a single
    // thread when no other thread could possibly call any other path guide method.
    void passReset();

    // Record a radiance value as seen from point p in direction dir.
    // This method is "const" because it is thread-safe, not because
    // it leaves the object unchanged.  It obviously changes object
    // internals.
    // p: point that receives radiance
    // dir: direction into the scene, away from p
    // radiance: the amount of radiance received
    void recordRadiance(const scene_rdl2::math::Vec3f &p, const scene_rdl2::math::Vec3f &dir,
                        const scene_rdl2::math::Color &radiance) const;

    // Given a point and direction, what is the pdf value?
    float getPdf(const scene_rdl2::math::Vec3f &p, const scene_rdl2::math::Vec3f &dir) const;

    // Return a guided sample direction and optional pdf.
    scene_rdl2::math::Vec3f sampleDirection(const scene_rdl2::math::Vec3f &p, float r1, float r2, float *pdf) const;

    // Is path guiding enabled?
    bool isEnabled() const;

    // Is sampling ready?  This includes a check for isEnabled()
    bool canSample() const;

    // What percentage of samples should use path guiding?
    float getPercentage() const;

private:
    class Impl;
    std::unique_ptr<Impl> mImpl;
};

} // namespace pbr
} // namespace moonray

