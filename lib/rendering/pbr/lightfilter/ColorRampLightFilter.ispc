// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#include "LightFilter.isph"
#include <scene_rdl2/common/platform/IspcUtil.isph>
#include <moonray/rendering/mcrt_common/Util.isph>


//----------------------------------------------------------------------------

ISPC_UTIL_EXPORT_UNIFORM_STRUCT_TO_HEADER(ColorRampLightFilter);

export uniform uint32_t
ColorRampLightFilter_hudValidation(uniform bool verbose)
{
    COLOR_RAMP_LIGHT_FILTER_VALIDATION;
}


//----------------------------------------------------------------------------

// Helper methods for color ramp evaluation
varying bool
ColorRampLightFilter_canIlluminate(const uniform LightFilter * uniform lif,
                                   const varying CanIlluminateData &data)
{
    return true;
}

void
ColorRampLightFilter_eval(const uniform LightFilter * uniform lif,
                          const varying EvalData& data,
                          varying Color * uniform filterValue)
{
    const uniform ColorRampLightFilter * uniform lightfilter = (const uniform ColorRampLightFilter * uniform)lif;
    
    float dist;
    switch (lightfilter->mMode) {
    case RADIAL:
        if (lightfilter->mUseXform) {
            Vec3f xformedPoint0 = transformPoint(lightfilter->mXform[0], data.shadingPointPosition);
            Vec3f xformedPoint1 = transformPoint(lightfilter->mXform[1], data.shadingPointPosition);
            dist = lerp(length(xformedPoint0), length(xformedPoint1), data.time);
        } else {
            dist = data.isect->distance;
        }
    break;
    case DIRECTIONAL:
        if (lightfilter->mUseXform) {
            Vec3f xformedPoint0 = transformPoint(lightfilter->mXform[0], data.shadingPointPosition);
            Vec3f xformedPoint1 = transformPoint(lightfilter->mXform[1], data.shadingPointPosition);
            Vec3f xformedPoint = lerp(xformedPoint0, xformedPoint1, data.time);
            dist = -xformedPoint.z;
        } else {
            Plane plane;
            Plane_init(&plane, data.lightPosition, data.lightDirection);
            dist = Plane_getDistance(&plane, data.shadingPointPosition);
            if (dot(data.shadingPointPosition - data.lightPosition, data.lightDirection) < 0.f) {
                // Restores the sign of dist because plane.getDistance() returns the abs distance to the plane
                dist *= -1.f;
            }
        }
        
        switch (lightfilter->mWrapMode) {
        case EXTEND:
            // Causes f(dist) = f(0) for dist < 0
            if (dist < 0.f) dist = 0.f;
        break;
        case MIRROR:
            // Causes f(dist) = f(-dist), mirroring about dist = 0
            dist = abs(dist);
        break;
        default:
            MNRY_ASSERT(false);
        }

    break;
    default:
        MNRY_ASSERT(false);
    }


    *filterValue = ColorRampControl_eval1D(dist, &lightfilter->mColorRamp);    

    // Scale the filter color value by the intensity
    *filterValue = *filterValue * lightfilter->mIntensity;

    // Apply density scaling to allow partial light filtering
    *filterValue = Color_ctor(1.f - lightfilter->mDensity) + *filterValue * lightfilter->mDensity;
}

//----------------------------------------------------------------------------

export void
ColorRampLightFilter_init(uniform ColorRampLightFilter * uniform lightfilter)
{
    LightFilter_init((uniform LightFilter * uniform)lightfilter, 
            ColorRampLightFilter_canIlluminate, ColorRampLightFilter_eval);
}


//----------------------------------------------------------------------------

