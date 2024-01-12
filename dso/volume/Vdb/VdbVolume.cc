// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file VdbVolume.cc
///

#include "attributes.cc"

#include <moonray/rendering/shading/MaterialApi.h>

using namespace scene_rdl2;

RDL2_DSO_CLASS_BEGIN(VdbVolume, scene_rdl2::rdl2::VolumeShader)

public:
    VdbVolume(const scene_rdl2::rdl2::SceneClass& sceneClass,
            const std::string& name);

    ~VdbVolume();

    virtual finline unsigned getProperties() const override
    {
        return mProp;
    }

    virtual scene_rdl2::math::Color extinct(moonray::shading::TLState *tls,
                                const moonray::shading::State &state,
                                const scene_rdl2::math::Color& density,
                                float /*rayVolumeDepth*/) const override
    {
        scene_rdl2::math::Color result = density * evalColor(this, attrExtinctionGainMult, tls, state);
        // Map shader can produce negative values. An extinction value cannot be negative.
        result = scene_rdl2::math::max(result, scene_rdl2::math::sBlack);
        return result;
    }

    virtual scene_rdl2::math::Color albedo(moonray::shading::TLState *tls,
                               const moonray::shading::State &state,
                               const scene_rdl2::math::Color& density,
                               float /*rayVolumeDepth*/) const override
    {
        scene_rdl2::math::Color result = density * evalColor(this, attrAlbedoMult, tls, state);
        // Map shader can produce negative values. An albedo value cannot be negative.
        result = scene_rdl2::math::max(result, scene_rdl2::math::sBlack);
        return result;
    }

    virtual scene_rdl2::math::Color emission(moonray::shading::TLState *tls,
                                 const moonray::shading::State &state,
                                 const scene_rdl2::math::Color& density) const override
    {
        scene_rdl2::math::Color result = density * evalColor(this, attrIncandGainMult, tls, state);
        // Map shader can produce negative values. An emission value cannot be negative.
        result = scene_rdl2::math::max(result, scene_rdl2::math::sBlack);
        return result;
    }

    virtual float anisotropy(moonray::shading::TLState *tls,
                             const moonray::shading::State &state) const override
    {
        return scene_rdl2::math::clamp(evalFloat(this, attrAnisotropy, tls, state), -1.0f, 1.0f);
    }

    virtual bool hasExtinctionMapBinding() const override
    {
        return getBinding(attrExtinctionGainMult) != nullptr;
    }

    virtual bool updateBakeRequired() const override
    {
        bool updateMapBinding = false;
        if (hasExtinctionMapBinding()) {
            updateMapBinding |= getBinding(attrExtinctionGainMult)->updateRequired();
        }
        return updateMapBinding ||
               hasChanged(scene_rdl2::rdl2::VolumeShader::sBakeResolutionMode) ||
               hasChanged(scene_rdl2::rdl2::VolumeShader::sBakeDivisions) ||
               hasChanged(scene_rdl2::rdl2::VolumeShader::sBakeVoxelSize) ||
               hasBindingChanged(attrExtinctionGainMult) ||
               // If any map bindings are added or removed, we must rebake the volume shader
               // in case velocity is being applied. We don't know if velocity is applied until
               // we bake the volume shader, so this is a precaution.
               hasBindingChanged(attrAlbedoMult) ||
               hasBindingChanged(attrIncandGainMult) ||
               hasBindingChanged(attrAnisotropy);
    }

protected:
    virtual void update() override;

private:
    unsigned mProp;

RDL2_DSO_CLASS_END(VdbVolume)


VdbVolume::VdbVolume(const scene_rdl2::rdl2::SceneClass& sceneClass,
        const std::string& name) : Parent(sceneClass, name)
{
}

VdbVolume::~VdbVolume()
{
}

void
VdbVolume::update()
{
    // If map binding does not exist, then property is homogenous.
    mProp = 0x0;
    if (getBinding(attrExtinctionGainMult)) {
        mProp |= IS_EXTINCTIVE;
    } else if (!isBlack(get(attrExtinctionGainMult))) {
        mProp |= IS_EXTINCTIVE | HOMOGENOUS_EXTINC;
    }

    if (getBinding(attrAlbedoMult)) {
        mProp |= IS_SCATTERING;
    } else if (!isBlack(get(attrAlbedoMult))) {
        mProp |= IS_SCATTERING | HOMOGENOUS_ALBEDO;
    }

    if (getBinding(attrIncandGainMult)) {
        mProp |= IS_EMISSIVE;
    } else if (!isBlack(get(attrIncandGainMult))) {
        mProp |= IS_EMISSIVE | HOMOGENOUS_EMISS;
    }

    if (get(attrAnisotropy) == 0.0f && getBinding(attrAnisotropy) == nullptr) {
        mProp |= ISOTROPIC_PHASE;
    }
}

