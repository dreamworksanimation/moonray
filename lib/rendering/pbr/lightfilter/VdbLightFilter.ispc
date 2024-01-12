// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#include "LightFilter.isph"
#include <moonray/rendering/pbr/core/Util.isph>
#include <moonray/rendering/shading/ispc/Shading.isph>
#include <moonray/rendering/shading/ispc/Util.isph>
#include <scene_rdl2/common/platform/IspcUtil.isph>

//----------------------------------------------------------------------------

ISPC_UTIL_EXPORT_UNIFORM_STRUCT_TO_HEADER(VdbLightFilter);

export uniform uint32_t
VdbLightFilter_hudValidation(uniform bool verbose)
{
    VDB_LIGHT_FILTER_VALIDATION;
}

//----------------------------------------------------------------------------

varying bool
VdbLightFilter_canIlluminate(const uniform LightFilter * uniform ,
                             const varying CanIlluminateData &)
{
    return true;
}

void
VdbLightFilter_eval(const uniform LightFilter * uniform lif,
                    const varying EvalData& data,
                    varying Color * uniform filterValue)
{
    const uniform VdbLightFilter * uniform lf = (const uniform VdbLightFilter * uniform)lif;
   
    // We need the vdb space position of the render space point.
    Vec3f vdbSpaceP0 = transformPoint(lf->mVdbR2V[0], data.shadingPointPosition);
    Vec3f vdbSpaceP1 = transformPoint(lf->mVdbR2V[1], data.shadingPointPosition);
    Vec3f vdbSpaceP = lerp(vdbSpaceP0, vdbSpaceP1, data.time);    

    float filterRadius = lf->mBlurValue; // aka blur value
    
    Vec3f offset = Vec3f_ctor(0.f);

    if (lf->mBlurValue > 0.f) {
        switch (lf->mBlurType) {
            
        case VDBLIGHTFILTERBLURTYPE_GAUSSIAN:
            // Quadratic bspline "gaussian" filter, which is the same filter the regular camera uses.
            offset.x = quadraticBSplineWarp(data.randVar.r3.x) * filterRadius;
            offset.y = quadraticBSplineWarp(data.randVar.r3.y) * filterRadius;
            offset.z = quadraticBSplineWarp(data.randVar.r3.z) * filterRadius;
            break;
        case VDBLIGHTFILTERBLURTYPE_SPHERICAL:
            //get random point on sphere
            offset = sampleSphereUniform(data.randVar.r3.x, data.randVar.r3.y);
            //scale it by cuberoot(radius) for uniform distribution within sphere
            offset.x *= filterRadius * pow(data.randVar.r3.z, 1.f/3.f);
            offset.y *= filterRadius * pow(data.randVar.r3.z, 1.f/3.f);
            offset.z *= filterRadius * pow(data.randVar.r3.z, 1.f/3.f);
            break;
        default:
            MNRY_ASSERT(0);
        }
    }

    Vec3f P = {vdbSpaceP.x + offset.x, vdbSpaceP.y + offset.y, vdbSpaceP.z + offset.z};
    varying float densitySample;
    
    foreach_active(lane) {
        const uniform float Px = extract(P.x, lane);
        const uniform float Py = extract(P.y, lane);
        const uniform float Pz = extract(P.z, lane);

        typedef void (* varying SampleVdbFn_t)(const uniform LightFilter * uniform lif,
                                               uniform ShadingTLState * uniform tls,
                                               const uniform float Px,
                                               const uniform float Py,
                                               const uniform float Pz,
                                               uniform float * uniform outDensity);
        SampleVdbFn_t sampleFn = (SampleVdbFn_t) lf->mSampleFn;

        uniform float d;    
        sampleFn(lif, data.tls, Px, Py, Pz, &d);
        densitySample = insert(densitySample, lane, d);
    }

    // Invert the density sample before the remap
    if (lf->mInvertDensity) {
        densitySample = 1.f - densitySample;
    }
    //now remap the filter   
    densitySample = FloatRampControl_eval1D(densitySample, &lf->mDensityRamp);    
    
    //clamp density to 0
    densitySample = max(densitySample, 0.f);

    // density sample controls how much light should be unaffected. less density means
    // we shift towards the tinted color     
    *filterValue = lerp(lf->mColorTint, Color_ctor(1.0f), densitySample);
}

//----------------------------------------------------------------------------

export void
VdbLightFilter_init(uniform VdbLightFilter * uniform lightfilter)
{
    LightFilter_init((uniform LightFilter * uniform)lightfilter, 
            VdbLightFilter_canIlluminate, VdbLightFilter_eval);
}

//----------------------------------------------------------------------------

