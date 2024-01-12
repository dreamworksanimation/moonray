// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <moonray/rendering/pbr/Types.h>

#include <scene_rdl2/common/math/Color.h>

namespace moonray {
namespace pbr {

struct VolumeTransmittance
{
    // Transmittance due to regular extinction.
    scene_rdl2::math::Color mTransmittanceE;
    // Holdouts (cutouts), Alpha, and Minimum volume transmittance
    // For a full explanation, see :
    // "Production Volume Rendering, Design and Implementation" by Wrenninge.
    // Section 10.8, specifically Code 10.6.
    scene_rdl2::math::Color mTransmittanceH;
    scene_rdl2::math::Color mTransmittanceAlpha;
    scene_rdl2::math::Color mTransmittanceMin;

    void reset()
    {
        mTransmittanceE = scene_rdl2::math::Color(1.f);
        mTransmittanceH = scene_rdl2::math::Color(1.f);
        mTransmittanceAlpha = scene_rdl2::math::Color(1.f);
        mTransmittanceMin = scene_rdl2::math::Color(0.f);
    }

    scene_rdl2::math::Color transmittance() const
    {
        // total transmittance is the combination of the regular extinction
        // transmittance and the holdouts/cutouts transmittance, since both
        // block light
        return mTransmittanceE * mTransmittanceH;
    }

    void update(const scene_rdl2::math::Color& tr, const scene_rdl2::math::Color& trh)
    {
        // update transmittance and holdouts transmittance
        mTransmittanceE *= tr;
        mTransmittanceH *= trh;
        // See Wrenninge pg. 201.  Update co-dependent minimum transmittance and
        // alpha transmittance.  Note the Wrenninge book has an error, we must use
        // the previous transmittanceM value when computing transmittanceA.
        scene_rdl2::math::Color oldTm = mTransmittanceMin;
        mTransmittanceMin = trh * mTransmittanceMin + (scene_rdl2::math::Color(1.f) - trh) * mTransmittanceAlpha;
        mTransmittanceAlpha = tr * mTransmittanceAlpha + (scene_rdl2::math::Color(1.f) - tr) * oldTm;
    }
};

} // namespace pbr
} // namespace moonray

