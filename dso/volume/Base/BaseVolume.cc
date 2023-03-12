// Copyright 2023 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file BaseVolume.cc
///

#include "attributes.cc"

#include <moonray/rendering/shading/MaterialApi.h>

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

    Color extinct(moonray::shading::TLState *tls,
                  const moonray::shading::State &state,
                  const Color& density) const override
    {
        Color result = density *
            evalFloat(this, attrAttenuationIntensity, tls, state) *
            evalFloat(this, attrAttenuationFactor, tls, state) *
            evalColor(this, attrAttenuationColor, tls, state);
        // Map shader can produce negative values. An extinction value cannot be negative.
        result = scene_rdl2::math::max(result, sBlack);
        return result;
    }

    Color albedo(moonray::shading::TLState *tls,
                 const moonray::shading::State &state,
                 const Color& density) const override
    {
        Color result = density * evalColor(this, attrAlbedo, tls, state);
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
               hasBindingChanged(attrAlbedo) ||
               hasBindingChanged(attrEmissionIntensity) ||
               hasBindingChanged(attrEmissionColor) ||
               hasBindingChanged(attrAnisotropy);
    }

protected:
    void update() override;

private:
    unsigned mProp;

RDL2_DSO_CLASS_END(BaseVolume)


BaseVolume::BaseVolume(const SceneClass& sceneClass,
        const std::string& name) : Parent(sceneClass, name)
{
}

void
BaseVolume::update()
{
    // If map binding does not exist, then property is homogenous.
    mProp = 0x0;
    if (getBinding(attrAlbedo)) {
        mProp |= IS_SCATTERING;
    } else if (!isBlack(get(attrAlbedo))) {
        mProp |= IS_SCATTERING | HOMOGENOUS_ALBEDO;
    }

    if (get(attrAnisotropy) == 0.0f && getBinding(attrAnisotropy) == nullptr) {
        mProp |= ISOTROPIC_PHASE;
    }

    bool heterogeneousExtinction =
        getBinding(attrAttenuationIntensity) ||
        getBinding(attrAttenuationFactor) ||
        getBinding(attrAttenuationColor);
    Color extinction = get(attrAttenuationIntensity) *
                       get(attrAttenuationFactor) *
                       get(attrAttenuationColor);
    if (heterogeneousExtinction) {
        mProp |= IS_EXTINCTIVE;
    } else if (!isBlack(extinction)) {
        mProp |= IS_EXTINCTIVE | HOMOGENOUS_EXTINC;
    }

    bool heterogeneousEmission = getBinding(attrEmissionIntensity) || getBinding(attrEmissionColor);
    Color emission = get(attrEmissionIntensity) * get(attrEmissionColor);
    if (heterogeneousEmission) {
        mProp |= IS_EMISSIVE;
    } else if (!isBlack(emission)) {
        mProp |= IS_EMISSIVE | HOMOGENOUS_EMISS;
    }
}

