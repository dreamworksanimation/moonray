// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file BaseVolume.cc
///

#include "attributes.cc"

#include <moonray/rendering/shading/MaterialApi.h>
#include <moonray/rendering/shading/RampControl.h>

using namespace scene_rdl2::rdl2;
using namespace scene_rdl2::math;
using namespace moonray::shading;

RDL2_DSO_CLASS_BEGIN(BaseVolume, VolumeShader)

public:
    BaseVolume(const SceneClass& sceneClass,
               const std::string& name);

    finline unsigned getProperties() const override
    {
        return mProp;
    }

/// ------------------------------------ Ramp Helper Functions -----------------------------------------------------

/* BaseVolume has options to pick attenuation/albedo colors or densities from a ramp, where the start of the 
 * ramp represents the thinnest part of the volume and the end of the ramp represents the deepest part of the
 * volume. In other words, you can change the look of a volume based on its depth. With the strategy we're using,
 * the volume actually remains a homogenous volume, since we evaluate the "depth" of the volume at a certain
 * point based on the distance between the point at which the ray enters the volume and the point at which it leaves.
 */

    /* With the given ramp parameters, evaluate the color ramp at rayVolumeDepth, where minDepth represents
     * the depth at which the ramp interpolation starts and maxDepth represents the depth at which the ramp 
     * interpolation ends */
    scene_rdl2::math::Color evalColorRamp(const moonray::shading::ColorRampControl &ramp,
                                          float minDepth, float maxDepth, float rayVolumeDepth) const
    {
        // The ramp will gradiate from minDepth --> maxDepth, and t should be 0 --> 1
        // based on the given rayVolumeDepth (which represents how far the ray travels through the volume)
        float t = scene_rdl2::math::max(rayVolumeDepth - minDepth, 0.f);
        float interpDepth = maxDepth - minDepth;
        t = interpDepth == 0.f ? 0.f : t / interpDepth;

        scene_rdl2::math::Color color = ramp.eval1D(t);
        return color;
    }

    float evalFloatRamp(const moonray::shading::FloatRampControl &ramp,
                        float minDepth, float maxDepth, float rayVolumeDepth) const
    {
        // The ramp will gradiate from minDepth --> maxDepth, and t should be 0 --> 1
        // based on the given rayVolumeDepth (which represents how far the ray travels through the volume)
        float t = scene_rdl2::math::max(rayVolumeDepth - minDepth, 0.f);
        float interpDepth = maxDepth - minDepth;
        t = interpDepth == 0.f ? 0.f : t / interpDepth;
        return ramp.eval1D(t);
    }

    scene_rdl2::math::Color getAttenuationColor(float rayVolumeDepth, 
                                                moonray::shading::TLState *tls, 
                                                const moonray::shading::State &state) const
    {
        scene_rdl2::math::Color color = get(attrMatchAlbedo) ? evalColor(this, attrAlbedoColor, tls, state)    
                                                             : evalColor(this, attrAttenuationColor, tls, state);
        bool useRamp =                  get(attrMatchAlbedo) ? get(attrUseAlbedoRamp)  : get(attrUseAttenuationRamp);     
        float minDepth =                get(attrMatchAlbedo) ? get(attrAlbedoMinDepth) : get(attrAttenuationMinDepth);
        float maxDepth =                get(attrMatchAlbedo) ? get(attrAlbedoMaxDepth) : get(attrAttenuationMaxDepth);
        moonray::shading::ColorRampControl ramp = get(attrMatchAlbedo) ? mAlbedoRamp : mAttenuationRamp;

        if (useRamp && rayVolumeDepth >= 0.f) { // -1 indicates ramp is not supported      
            color = evalColorRamp(ramp, minDepth, maxDepth, rayVolumeDepth);
        }
        return get(attrInvertAttenuationColor) ? 1.f - color : color;
    }

    scene_rdl2::math::Color getAlbedoColor(float rayVolumeDepth, 
                                           moonray::shading::TLState *tls, 
                                           const moonray::shading::State &state) const 
    {
        scene_rdl2::math::Color color = evalColor(this, attrAlbedoColor, tls, state);

        if (get(attrUseAlbedoRamp) && rayVolumeDepth >= 0.f) { // -1 indicates ramp is not supported
            color = evalColorRamp(mAlbedoRamp, get(attrAlbedoMinDepth), get(attrAlbedoMaxDepth), rayVolumeDepth);
        }
        return color;
    }

    // Get the density of the volume. This will be the input density multiplied
    // by the ramp values (which are by default 1.f)
    scene_rdl2::math::Color getDensity(const scene_rdl2::math::Color& density, float rayVolumeDepth) const
    {
        if (rayVolumeDepth == -1.f) return density; // -1 indicates ramp is not supported
        return density * evalFloatRamp(mDensityRamp, get(attrDensityMinDepth), get(attrDensityMaxDepth), rayVolumeDepth);
    }
/// ---------------------------------- End Ramp Helper Functions ---------------------------------------------------

    virtual scene_rdl2::math::Color extinct(moonray::shading::TLState *tls,
                                            const moonray::shading::State &state,
                                            const scene_rdl2::math::Color& density,
                                            float rayVolumeDepth) const override
    {
        scene_rdl2::math::Color attenuationColor = getAttenuationColor(rayVolumeDepth, tls, state);  
        scene_rdl2::math::Color bvDensity = getDensity(density, rayVolumeDepth);
        float attenuationIntensity = evalFloat(this, attrAttenuationIntensity, tls, state);
        float attenuationFactor = evalFloat(this, attrAttenuationFactor, tls, state);
        
        scene_rdl2::math::Color result = bvDensity * attenuationIntensity * attenuationColor * attenuationFactor;  
        // Map shader can produce negative values. An extinction value cannot be negative.
        result = scene_rdl2::math::max(result, sBlack);
        return result;
    }

    virtual scene_rdl2::math::Color albedo(moonray::shading::TLState *tls,
                                           const moonray::shading::State &state,
                                           const scene_rdl2::math::Color& density,
                                           float rayVolumeDepth) const override
    {
        scene_rdl2::math::Color albedoColor = getAlbedoColor(rayVolumeDepth, tls, state);
        scene_rdl2::math::Color bvDensity = getDensity(density, rayVolumeDepth);

        scene_rdl2::math::Color result = bvDensity * albedoColor;
        // Map shader can produce negative values. An albedo value cannot be negative.
        result = scene_rdl2::math::max(result, sBlack);
        return result;
    }

    Color emission(moonray::shading::TLState *tls,
                   const moonray::shading::State &state,
                   const Color& density) const override
    {
        Color result = density * evalFloat(this, attrEmissionIntensity, tls, state) *
            evalColor(this, attrEmissionColor, tls, state);
        // Map shader can produce negative values. An emission value cannot be negative.
        result = scene_rdl2::math::max(result, sBlack);
        return result;
    }

    float anisotropy(moonray::shading::TLState *tls,
                     const moonray::shading::State &state) const override
    {
        return scene_rdl2::math::clamp(evalFloat(this, attrAnisotropy, tls, state), -1.0f, 1.0f);
    }

    bool hasExtinctionMapBinding() const override
    {
        return getBinding(attrAttenuationIntensity) != nullptr ||
               getBinding(attrAttenuationFactor) != nullptr ||
               getBinding(attrAttenuationColor) != nullptr;
    }

    bool updateBakeRequired() const override
    {
        bool updateMapBinding = false;
        if (getBinding(attrAttenuationIntensity) != nullptr) {
            updateMapBinding |= getBinding(attrAttenuationIntensity)->updateRequired();
        }
        if (getBinding(attrAttenuationFactor) != nullptr) {
            updateMapBinding |= getBinding(attrAttenuationFactor)->updateRequired();
        }
        if (getBinding(attrAttenuationColor) != nullptr) {
            updateMapBinding |= getBinding(attrAttenuationColor)->updateRequired();
        }
        return updateMapBinding ||
               hasChanged(VolumeShader::sBakeResolutionMode) ||
               hasChanged(VolumeShader::sBakeDivisions) ||
               hasChanged(VolumeShader::sBakeVoxelSize) ||
               hasBindingChanged(attrAttenuationIntensity) ||
               hasBindingChanged(attrAttenuationFactor) ||
               hasBindingChanged(attrAttenuationColor) ||
               // If any map bindings are added or removed, we must rebake the volume shader
               // in case velocity is being applied. We don't know if velocity is applied until
               // we bake the volume shader, so this is a precaution.
               hasBindingChanged(attrAlbedoColor) ||
               hasBindingChanged(attrEmissionIntensity) ||
               hasBindingChanged(attrEmissionColor) ||
               hasBindingChanged(attrAnisotropy);
    }

protected:
    void update() override;

private:
    unsigned mProp;
    moonray::shading::ColorRampControl mAttenuationRamp;
    moonray::shading::ColorRampControl mAlbedoRamp;
    moonray::shading::FloatRampControl mDensityRamp;

RDL2_DSO_CLASS_END(BaseVolume)


BaseVolume::BaseVolume(const scene_rdl2::rdl2::SceneClass& sceneClass,
                       const std::string& name) : Parent(sceneClass, name) 
{     
}

void
BaseVolume::update()
{
    // If map binding does not exist, then property is homogenous.
    mProp = 0x0;
    if (getBinding(attrAlbedoColor)) {
        mProp |= IS_SCATTERING;
    } else if (!isBlack(get(attrAlbedoColor)) || get(attrUseAlbedoRamp)) {
        mProp |= IS_SCATTERING | HOMOGENOUS_ALBEDO;
    }

    if (get(attrAnisotropy) == 0.0f && getBinding(attrAnisotropy) == nullptr) {
        mProp |= ISOTROPIC_PHASE;
    }

    bool heterogeneousExtinction =
        getBinding(attrAttenuationIntensity) ||
        getBinding(attrAttenuationFactor) ||
        getBinding(attrAttenuationColor);

    scene_rdl2::math::Color extinction = get(attrMatchAlbedo) ? get(attrAlbedoColor) : get(attrAttenuationColor);
    extinction = get(attrInvertAttenuationColor) ? (1.f - extinction) : extinction;
    extinction = get(attrAttenuationIntensity) * get(attrAttenuationFactor) * extinction;
    
    bool useRamp = get(attrUseAttenuationRamp) || (get(attrMatchAlbedo) && get(attrUseAlbedoRamp));
    if (heterogeneousExtinction) {
        mProp |= IS_EXTINCTIVE;
    } else if (!isBlack(extinction) || (useRamp && get(attrAttenuationIntensity) != 0.0f)) {
        mProp |= IS_EXTINCTIVE | HOMOGENOUS_EXTINC;
    }

    bool heterogeneousEmission = getBinding(attrEmissionIntensity) || getBinding(attrEmissionColor);
    Color emission = get(attrEmissionIntensity) * get(attrEmissionColor);
    if (heterogeneousEmission) {
        mProp |= IS_EMISSIVE;
    } else if (!isBlack(emission)) {
        mProp |= IS_EMISSIVE | HOMOGENOUS_EMISS;
    }
  
    // Initialize/update ramp controls
    mAttenuationRamp.init(get(attrAttenuationDistances).size(), 
                          get(attrAttenuationDistances).data(), 
                          get(attrAttenuationColors).data(),
                          reinterpret_cast<const ispc::RampInterpolatorMode*>(
                            get(attrAttenuationInterpolationTypes).data()
                          ),
                          ispc::COLOR_RAMP_CONTROL_SPACE_RGB);

    mAlbedoRamp.init(get(attrAlbedoDistances).size(), 
                     get(attrAlbedoDistances).data(), 
                     get(attrAlbedoColors).data(),
                     reinterpret_cast<const ispc::RampInterpolatorMode*>(get(attrAlbedoInterpolationTypes).data()),
                     ispc::COLOR_RAMP_CONTROL_SPACE_RGB);
    
    mDensityRamp.init(get(attrDensityDistances).size(), 
                      get(attrDensityDistances).data(), 
                      get(attrDensities).data(),
                      reinterpret_cast<const ispc::RampInterpolatorMode*>(get(attrDensityInterpolationTypes).data()));
}

