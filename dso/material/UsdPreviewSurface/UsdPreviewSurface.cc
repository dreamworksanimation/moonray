// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file UsdPreviewSurface.cc

#include "attributes.cc"
#include "labels.h"
#include "UsdPreviewSurface_ispc_stubs.h"

#include <moonray/rendering/shading/MaterialApi.h>

using namespace moonray::shading;

namespace {
void
addMirrorDielectricLobe(BsdfBuilder& bsdfBuilder,
                        const scene_rdl2::math::Vec3f& N,
                        const float ior,
                        const float weight,
                        const float opacity,
                        const int reflectionLabel=aovSpecular,
                        const int transmissionLabel=aovSpecularTransmission,
                        const ispc::BsdfBuilderBehavior behavior=ispc::BSDFBUILDER_PHYSICAL)
{
    const moonray::shading::MirrorBSDF dielectricBSDF(N, ior,
                                                    scene_rdl2::math::sWhite,    // transmission color
                                                    0.0f,            // abbe number
                                                    ior,
                                                    weight,
                                                    1.0f - opacity); // transmission
    bsdfBuilder.addMirrorBSDF(dielectricBSDF,
                              1.0f, // weight
                              behavior,
                              reflectionLabel,
                              transmissionLabel);
}

void
addMicrofacetDielectricLobe(BsdfBuilder& bsdfBuilder,
                            const scene_rdl2::math::Vec3f& N,
                            const float ior,
                            const float roughness,
                            const float weight,
                            const float opacity,
                            const int reflectionLabel=aovSpecular,
                            const int transmissionLabel=aovSpecularTransmission,
                            const ispc::BsdfBuilderBehavior behavior=ispc::BSDFBUILDER_PHYSICAL)
{
    const MicrofacetIsotropicBSDF dielectricBSDF(N,
                                                 ior,
                                                 roughness,
                                                 ispc::MICROFACET_DISTRIBUTION_GGX,
                                                 ispc::MICROFACET_GEOMETRIC_SMITH,
                                                 scene_rdl2::math::sWhite,    // transmission color
                                                 0.0f,            // abbe number
                                                 ior,
                                                 weight,
                                                 1.0f - opacity); // transmission

    bsdfBuilder.addMicrofacetIsotropicBSDF(dielectricBSDF,
                                           1.0f, // weight
                                           behavior,
                                           reflectionLabel,
                                           transmissionLabel);
}

void
addMirrorConductorLobe(BsdfBuilder& bsdfBuilder,
                       const scene_rdl2::math::Vec3f& N,
                       const scene_rdl2::math::Color& color,
                       const float weight,
                       const int label=aovSpecular,
                       const ispc::BsdfBuilderBehavior behavior=ispc::BSDFBUILDER_PHYSICAL)
{
    const moonray::shading::MirrorBRDF conductorBRDF(N, color, color);
    bsdfBuilder.addMirrorBRDF(conductorBRDF, weight, behavior, label);
}

void
addMicrofacetConductorLobe(BsdfBuilder& bsdfBuilder,
                           const scene_rdl2::math::Vec3f& N,
                           const scene_rdl2::math::Color& color,
                           const float roughness,
                           const float weight,
                           const int label=aovSpecular,
                           const ispc::BsdfBuilderBehavior behavior=ispc::BSDFBUILDER_PHYSICAL)
{
    const moonray::shading::MicrofacetIsotropicBRDF conductorBRDF(N, color, color, roughness, 
                                                                ispc::MICROFACET_DISTRIBUTION_GGX,
                                                                ispc::MICROFACET_GEOMETRIC_SMITH);
    bsdfBuilder.addMicrofacetIsotropicBRDF(conductorBRDF, weight, behavior, label);
}

} // end anonymous namespace


RDL2_DSO_CLASS_BEGIN(UsdPreviewSurface, scene_rdl2::rdl2::Material)

public:
    UsdPreviewSurface(const scene_rdl2::rdl2::SceneClass& sceneClass, const std::string& name);
    ~UsdPreviewSurface() { }

    void update() override;

    static void shade(const scene_rdl2::rdl2::Material* self,
                      moonray::shading::TLState *tls,
                      const State& state,
                      BsdfBuilder& bsdfBuilder);

    static float presence(const scene_rdl2::rdl2::Material* self,
                          moonray::shading::TLState *tls,
                          const moonray::shading::State& state);
private:

    ispc::UsdPreviewSurface mIspc; // needs to be first member

RDL2_DSO_CLASS_END(UsdPreviewSurface)

UsdPreviewSurface::UsdPreviewSurface(const scene_rdl2::rdl2::SceneClass& sceneClass,
                                                     const std::string& name) :
    Parent(sceneClass, name)
{
    mShadeFunc = UsdPreviewSurface::shade;
    mShadeFuncv = (scene_rdl2::rdl2::ShadeFuncv) ispc::UsdPreviewSurface_getShadeFunc();
    mPresenceFunc = UsdPreviewSurface::presence;
}

void
UsdPreviewSurface::update()
{
    // Get whether or not the normals have been reversed
    mOptionalAttributes.push_back(moonray::shading::StandardAttributes::sReversedNormals);
    mIspc.mReversedNormalsIndx = StandardAttributes::sReversedNormals.getIndex();
}

float
UsdPreviewSurface::presence(const scene_rdl2::rdl2::Material* self,
                                    moonray::shading::TLState *tls,
                                    const moonray::shading::State& state)
{
    const UsdPreviewSurface* me = static_cast<const UsdPreviewSurface*>(self);
    const float opacityThreshold = evalFloat(me, attrOpacityThreshold, tls, state);
    const float opacity = evalFloat(me, attrOpacity, tls, state);
    return opacity >= opacityThreshold ? 1.0f : 0.0f;
}

void
UsdPreviewSurface::shade(const scene_rdl2::rdl2::Material* self,
                                 moonray::shading::TLState *tls,
                                 const State& state,
                                 BsdfBuilder& bsdfBuilder)
{
    const UsdPreviewSurface* me =
        static_cast<const UsdPreviewSurface*>(self);


    // Transform inputNormal from tangent space to shade space
    scene_rdl2::math::Vec3f inputNormal = evalVec3f(me, attrNormal, tls, state);
    const scene_rdl2::math::Vec3f &stateN = state.getN();
    bool reversedNormals = false;
    if (state.isProvided(moonray::shading::StandardAttributes::sReversedNormals)) {
        reversedNormals = state.getAttribute(moonray::shading::StandardAttributes::sReversedNormals);
    }

    scene_rdl2::math::Vec3f N;
    if (scene_rdl2::math::isZero(scene_rdl2::math::length(state.getdPds()))) {
        N = stateN;
    } else {
        const scene_rdl2::math::Vec3f statedPds = reversedNormals ? state.getdPds() * -1.0f : state.getdPds() * 1.0f;
        const scene_rdl2::math::ReferenceFrame frame(stateN, scene_rdl2::math::normalize(statedPds));
        N = scene_rdl2::math::normalize(frame.localToGlobal(inputNormal));
    }

    const scene_rdl2::math::Color albedo = evalColor(me, attrDiffuseColor, tls, state);
    const float ior = evalFloat(me, attrIor, tls, state);
    const float roughness = scene_rdl2::math::saturate(evalFloat(me, attrRoughness, tls, state));
    const float opacityThreshold = scene_rdl2::math::saturate(evalFloat(me, attrOpacityThreshold, tls, state));
    const float opacity = opacityThreshold > 0.0f ? 1.0f : scene_rdl2::math::saturate(evalFloat(me, attrOpacity, tls, state));

    // Emission
    const scene_rdl2::math::Color emissiveColor = evalColor(me, attrEmissiveColor, tls, state);
    if (!scene_rdl2::math::isBlack(emissiveColor)) {
        bsdfBuilder.addEmission(evalColor(me, attrEmissiveColor, tls, state));
    }

    // Clearcoat
    const float clearcoat = evalFloat(me, attrClearcoat, tls, state);
    if (!scene_rdl2::math::isZero(clearcoat)) {
        const float clearcoatRoughness = evalFloat(me, attrClearcoatRoughness, tls, state);
        if (scene_rdl2::math::isZero(clearcoatRoughness)) { // mirror
            addMirrorDielectricLobe(bsdfBuilder, N, ior,
                                    clearcoat, // weight
                                    1.0f, // opacity
                                    aovOuterSpecular, aovSpecularTransmission,
                                    ispc::BSDFBUILDER_ADDITIVE);
        } else { // microfacet
            addMicrofacetDielectricLobe(bsdfBuilder, N, ior,
                                        clearcoatRoughness, clearcoat,
                                        1.0f, // opacity
                                        aovOuterSpecular, aovSpecularTransmission,
                                        ispc::BSDFBUILDER_ADDITIVE);
        }
    }

    const float weight = 1.0f;

    // Specular
    const ispc::SpecularWorkflow specularWorkflow =
        static_cast<ispc::SpecularWorkflow>(me->get(attrUseSpecularWorkflow));
    if (specularWorkflow == ispc::SPECULAR_WORKFLOW_SPECULAR) {
        // We can't support this until we implement Schlick fresnel so do nothing for now.
        // Phase 2: implement Schlick which takes specular color directly
    } else { // SPECULAR_WORKFLOW_METALNESS
        const float metallic = evalFloat(me, attrMetallic, tls, state);
        if (scene_rdl2::math::isZero(metallic)) {
            // If "metallic" is zero then use dielectric
            if (scene_rdl2::math::isZero(roughness)) {
                addMirrorDielectricLobe(bsdfBuilder, N, ior, weight, opacity);
            } else {
                addMicrofacetDielectricLobe(bsdfBuilder, N, ior, roughness, weight, opacity);
            }
        } else if (scene_rdl2::math::isOne(metallic)) {
            // If "metallic" is one then use artist friendly conductor
            if (roughness < scene_rdl2::math::sEpsilon) {
                addMirrorConductorLobe(bsdfBuilder, N, albedo, weight);
            } else {
                addMicrofacetConductorLobe(bsdfBuilder, N, albedo, roughness, weight);
            }
            return; // Exit early for fully metallic surfaces
        } else {
            // If "metallic" is between zero and one then blend dielectric
            // and conductor based with "metallic" as the weight
            if (roughness < scene_rdl2::math::sEpsilon) {
                addMirrorConductorLobe(bsdfBuilder, N, albedo, metallic);
                addMirrorDielectricLobe(bsdfBuilder, N, ior, weight, opacity);
            } else {
                addMicrofacetConductorLobe(bsdfBuilder, N, albedo, roughness, metallic);
                addMicrofacetDielectricLobe(bsdfBuilder, N, ior, roughness, weight, opacity);
            }
        }
    }

    // Diffuse
    if (!scene_rdl2::math::isBlack(albedo)) {
        const moonray::shading::LambertianBRDF diffuseRefl(N, albedo);
        bsdfBuilder.addLambertianBRDF(diffuseRefl,
                                      weight,
                                      ispc::BSDFBUILDER_PHYSICAL,
                                      aovDiffuse);
    }
}

