// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#include "attributes.cc"
#include "RaySwitchMaterial_ispc_stubs.h"

#include <moonray/rendering/shading/MaterialApi.h>

#include <string>

using namespace scene_rdl2::math;
using namespace moonray::shading;


//---------------------------------------------------------------------------

RDL2_DSO_CLASS_BEGIN(RaySwitchMaterial, scene_rdl2::rdl2::Material)

public:
    RaySwitchMaterial(const scene_rdl2::rdl2::SceneClass& sceneClass, const std::string& name);
    ~RaySwitchMaterial();

    virtual void update() override;

    const scene_rdl2::rdl2::Material* raySwitch(const scene_rdl2::rdl2::RaySwitchContext& ctx) const override;

private:
    static void shade(const scene_rdl2::rdl2::Material* self,
                      moonray::shading::TLState *tls,
                      const State& state,
                      BsdfBuilder& bsdfBuilder);

    static float presence(const scene_rdl2::rdl2::Material* self,
                          moonray::shading::TLState *tls,
                          const moonray::shading::State& state);

    static float ior(const scene_rdl2::rdl2::Material* self,
                     moonray::shading::TLState *tls,
                     const moonray::shading::State& state);

    ispc::RaySwitchMaterial mIspc;

RDL2_DSO_CLASS_END(RaySwitchMaterial)


//---------------------------------------------------------------------------

RaySwitchMaterial::RaySwitchMaterial(const scene_rdl2::rdl2::SceneClass& sceneClass,
                                     const std::string& name) :
    Parent(sceneClass, name)
{
    mShadeFunc = RaySwitchMaterial::shade;
    mShadeFuncv = (scene_rdl2::rdl2::ShadeFuncv) ispc::RaySwitchMaterial_getShadeFunc();
    mPresenceFunc = RaySwitchMaterial::presence;
    mIspc.mDefaultMaterialObj = 0;
    mIspc.mCutoutCameraRays = false;
}

RaySwitchMaterial::~RaySwitchMaterial()
{
}

void
RaySwitchMaterial::update()
{
    mIspc.mDefaultMaterialObj = (intptr_t) get(attrDefaultMaterial);
    mIspc.mCutoutCameraRays = get(attrCutoutCameraRays);
}

void
RaySwitchMaterial::shade(const scene_rdl2::rdl2::Material* self,
                         moonray::shading::TLState *tls,
                         const State& state,
                         BsdfBuilder& bsdfBuilder)
{
    // If this material is cutting out, terminate the ray
    if (!state.isIndirect() && self->get(attrCutoutCameraRays)) {
        bsdfBuilder.setEarlyTermination();
        return;
    }

    // We have already performed substitutions in the integrator so we don't need to switch here.
    // If there was no substitution, use the default material.

    const SceneObject* defaultMaterialObj = self->get(attrDefaultMaterial);
    if (defaultMaterialObj) {
        moonray::shading::shade(self, defaultMaterialObj->asA<scene_rdl2::rdl2::Material>(), tls, state, bsdfBuilder);
    }
}

float
RaySwitchMaterial::presence(const scene_rdl2::rdl2::Material* self,
                            moonray::shading::TLState *tls,
                            const moonray::shading::State& state)
{
    const SceneObject* defaultMaterialObj = self->get(attrDefaultMaterial);
    if (defaultMaterialObj) {
        return moonray::shading::presence(self, defaultMaterialObj->asA<scene_rdl2::rdl2::Material>(), tls, state);
    }

    return 1.f;
}

float
RaySwitchMaterial::ior(const scene_rdl2::rdl2::Material* self,
                       moonray::shading::TLState *tls,
                       const moonray::shading::State& state)
{
    const SceneObject* defaultMaterialObj = self->get(attrDefaultMaterial);
    if (defaultMaterialObj) {
        return moonray::shading::ior(self, defaultMaterialObj->asA<scene_rdl2::rdl2::Material>(), tls, state);
    }

    return 1.f;
}

const scene_rdl2::rdl2::Material*
RaySwitchMaterial::raySwitch(const scene_rdl2::rdl2::RaySwitchContext& ctx) const
{
    switch (ctx.mRayType) {
    case scene_rdl2::rdl2::RaySwitchContext::RayType::CameraRay:
    {
        if(get(attrCutoutCameraRays)) {
            // If cutting out camera rays, return this material.
            // Calling shade on this material will trigger early
            // termination of the rays.
            return this;
        }
        const SceneObject* materialObj = get(attrCameraRayMaterial);
        if (materialObj) {
            return materialObj->asA<scene_rdl2::rdl2::Material>();
        }
    }
    break;
    case scene_rdl2::rdl2::RaySwitchContext::RayType::IndirectMirrorRay:
    {
        const SceneObject* materialObj = get(attrIndirectMirrorRayMaterial);
        if (materialObj) {
            return materialObj->asA<scene_rdl2::rdl2::Material>();
        }
    }
    break;
    case scene_rdl2::rdl2::RaySwitchContext::RayType::IndirectGlossyRay:
    {
        const SceneObject* materialObj = get(attrIndirectGlossyRayMaterial);
        if (materialObj) {
            return materialObj->asA<scene_rdl2::rdl2::Material>();
        }
    }
    break;
    case scene_rdl2::rdl2::RaySwitchContext::RayType::IndirectDiffuseRay:
    {
        const SceneObject* materialObj = get(attrIndirectDiffuseRayMaterial);
        if (materialObj) {
            return materialObj->asA<scene_rdl2::rdl2::Material>();
        }
    }
    break;
    }

    return this;
}

