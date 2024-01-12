// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

//
#pragma once

#include "EllipticalFalloff.hh"
#include <moonray/rendering/pbr/light/LightUtil.h>

#include <scene_rdl2/common/math/Color.h>
#include <scene_rdl2/common/platform/HybridUniformData.h>


// Forward declaration of the ISPC types
namespace ispc {
    struct EllipticalFalloff;
}


namespace moonray {
namespace pbr {

//----------------------------------------------------------------------------

/// 
/// @class EllipticalFalloff
/// @brief Class which implements super elliptical falloff.
///

class EllipticalFalloff
{
public:
    EllipticalFalloff();

    void init( float roundness = 1.0f,   // 0 = square, 1 = circular
               float elliptical = 0.0f,  // 0 = symmetrical, negative = compress in u, positive = compress in v
               OldFalloffCurveType curveType = OLD_FALLOFF_CURVE_TYPE_NATURAL,
               float exp = 1.0f );
    /// This is called by the spotlight class to set its current inner and outer
    /// fields of view. A field of view spans the entire angle from side to side
    /// as opposed to a cone angle which is half the field of view.
    void setFov(float innerFov, float outerFov);


    /// HUD validation and type casting
    static uint32_t hudValidation(bool verbose) {
        ELLIPTICAL_FALLOFF_VALIDATION;
    }
    HUD_AS_ISPC_METHODS(EllipticalFalloff);


    /// Returns true if the supplied uvs represent a valid location on the function.
    /// TODO: Replace this with a sampling function...
    bool intersect(float u, float v) const;

    /// This function is responsible for modifying the incoming color anyway
    /// it sees fit. UV space is square, any further transformation should be
    /// done in here.
    void eval(float u, float v, scene_rdl2::math::Color *color) const;


protected:
    float           mRoundness;
    float           mElliptical;
    OldFalloffCurve mOldFalloffCurve;

    float           mInnerW;
    float           mInnerH;
    float           mOuterW;
    float           mOuterH;
};

//----------------------------------------------------------------------------

} // namespace pbr
} // namespace moonray

