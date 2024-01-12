// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file VolumePhase.h
///

#include <moonray/rendering/shading/Util.h>
#include <scene_rdl2/common/math/Math.h>
#include <scene_rdl2/common/math/ReferenceFrame.h>
#include <scene_rdl2/common/math/Vec3.h>

#pragma once

namespace moonray {
namespace pbr {

/// The phase function is based on Henyey Greenstein model that is quite
/// standard for production rendering. We can extend to support other models
/// if there is actual need.
class VolumePhase
{
public:
    explicit VolumePhase(float g): mG(g) {}

    float eval(const scene_rdl2::math::Vec3f& wo, const scene_rdl2::math::Vec3f& wi) const {
        if (scene_rdl2::math::abs(mG) <= 1e-3f) {
            return scene_rdl2::math::sOneOverFourPi;
        } else {
            float cosTheta = scene_rdl2::math::dot(wo, wi);
            float denom = 1 + mG * mG + 2 * mG * cosTheta;
            return scene_rdl2::math::isZero(denom) ? 0.0f :
                scene_rdl2::math::sOneOverFourPi * (1 - mG * mG) / (denom * scene_rdl2::math::sqrt(denom));
        }
    }

    scene_rdl2::math::Vec3f sample(const scene_rdl2::math::Vec3f& wo, float u1, float u2) const {
        if (scene_rdl2::math::abs(mG) <= 1e-3f) {
            // isotropic case
            return shading::sampleSphereUniform(u1, u2);
        } else {
            // anisotropic case
            // Henyey Greenstein model can be analytically inversed
            u1 = scene_rdl2::math::clamp(u1, scene_rdl2::math::sEpsilon);
            float sqrTerm = (1.0f - mG * mG) / (1.0f - mG + 2.0f * mG * u1);
            float cosTheta = (1.0f + mG * mG - sqrTerm * sqrTerm) / (2.0f * mG);
            float sinTheta = scene_rdl2::math::sqrt(scene_rdl2::math::max(0.0f,
                1.0f - cosTheta * cosTheta));
            float phi = scene_rdl2::math::sTwoPi * u2;
            float cosPhi, sinPhi;
            scene_rdl2::math::sincos(phi, &sinPhi, &cosPhi);
            scene_rdl2::math::Vec3f localDir(sinTheta * cosPhi, sinTheta * sinPhi, cosTheta);
            scene_rdl2::math::ReferenceFrame frame(-wo);
            return frame.localToGlobal(localDir);
        }
    }

    float pdf(const scene_rdl2::math::Vec3f& wo, const scene_rdl2::math::Vec3f& wi) const {
        return eval(wo, wi);
    }

private:
    float mG;
};

} // namespace pbr
} // namespace moonray

