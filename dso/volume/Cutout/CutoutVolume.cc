// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#include <scene_rdl2/scene/rdl2/rdl2.h>
#include <scene_rdl2/scene/rdl2/Types.h>

#include "attributes.cc"

using namespace scene_rdl2;

/*
 * Moonray's volume cutouts (a.k.a. holdouts) are based on the book:
 * "Production Volume Rendering, Design and Implementation" by Wrenninge.
 * See Section 10.8, specifically Code 10.6.
 *
 * The CutoutVolume works the same way as the CutoutMaterial.  It has an
 * indirect volume attribute that is a regular volume shader.  This shader is
 * used for non-primary rays (that we don't cutout), ensuring correct indirect
 * illumination.  CutoutVolume just wraps the indirect volume shader and overrides
 * the isCutout() virtual function to tell Moonray's PathIntegratorVolume to
 * invoke cutout behavior when integrating / ray marching.
 */

RDL2_DSO_CLASS_BEGIN(CutoutVolume, scene_rdl2::rdl2::VolumeShader)

public:
    CutoutVolume(const scene_rdl2::rdl2::SceneClass& sceneClass,
                 const std::string& name);

    virtual finline unsigned getProperties() const override
    {
        return (mIndirectVolume != nullptr) ?
            mIndirectVolume->getProperties() :
            0;
    }

    virtual scene_rdl2::math::Color extinct(moonray::shading::TLState *tls,
                                const moonray::shading::State &state,
                                const scene_rdl2::math::Color& density,
                                float /*rayVolumeDepth*/) const override
    {
        return (mIndirectVolume != nullptr) ?
            mIndirectVolume->extinct(tls, state, density, /*rayVolumeDepth*/ -1) :
            scene_rdl2::math::Color(0.f, 0.f, 0.f);
    }

    virtual scene_rdl2::math::Color albedo(moonray::shading::TLState *tls,
                               const moonray::shading::State &state,
                               const scene_rdl2::math::Color& density,
                               float /*rayVolumeDepth*/) const override
    {
        return (mIndirectVolume != nullptr) ?
            mIndirectVolume->albedo(tls, state, density, /*rayVolumeDepth*/ -1) :
            scene_rdl2::math::Color(0.f, 0.f, 0.f);
    }

    virtual scene_rdl2::math::Color emission(moonray::shading::TLState *tls,
                                 const moonray::shading::State &state,
                                 const scene_rdl2::math::Color& density) const override
    {
        return (mIndirectVolume != nullptr) ?
            mIndirectVolume->emission(tls, state, density) :
            scene_rdl2::math::Color(0.f, 0.f, 0.f);
    }

    virtual float anisotropy(moonray::shading::TLState *tls,
                             const moonray::shading::State &state) const override
    {
        return (mIndirectVolume != nullptr) ?
            mIndirectVolume->anisotropy(tls, state) :
            0.f;
    }

    virtual bool hasExtinctionMapBinding() const override
    {
        return (mIndirectVolume != nullptr) ?
                mIndirectVolume->hasExtinctionMapBinding() : false;
    }

    virtual bool updateBakeRequired() const override
    {
        return (mIndirectVolume != nullptr) ?
                mIndirectVolume->updateBakeRequired() : false;
    }

    virtual bool isCutout() const override
    {
        return true;
    }

protected:
    void update() override;

private:
    scene_rdl2::rdl2::VolumeShader* mIndirectVolume;  // always check for nullptr before using


RDL2_DSO_CLASS_END(CutoutVolume)


CutoutVolume::CutoutVolume(const scene_rdl2::rdl2::SceneClass& sceneClass,
                           const std::string& name) :
    Parent(sceneClass, name),
    mIndirectVolume(nullptr)
{
}

void
CutoutVolume::update()
{
    SceneObject *obj = get(attrIndirectVolume);
    if (obj) {
        mIndirectVolume = obj->asA<scene_rdl2::rdl2::VolumeShader>();
    }
}

