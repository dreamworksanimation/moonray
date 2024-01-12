// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#include "attributes.cc"
#include "BaseMaterial_ispc_stubs.h"
#include "labels.h"

#include <moonray/common/mcrt_macros/moonray_static_check.h>
#include <moonray/rendering/shading/bsdf/cook_torrance/BsdfCookTorrance.h>
#include <moonray/rendering/shading/bsdf/BsdfLambert.h>
#include <moonray/rendering/shading/bsdf/BsdfRetroreflection.h>
#include <moonray/rendering/shading/bsdf/BsdfMirror.h>
#include <moonray/rendering/shading/bsdf/BsdfIridescence.h>
#include <moonray/rendering/shading/bsdf/under/BsdfUnder.h>
#include <moonray/rendering/shading/bssrdf/Dipole.h>
#include <moonray/rendering/shading/Ior.h>

#include <moonray/rendering/shading/MaterialApi.h>
#include <moonray/rendering/shading/ShadingUtil.h>

#include <string>

using namespace scene_rdl2::math;
using namespace moonray::shading;


#define USE_SCHLICK 1


//---------------------------------------------------------------------------

RDL2_DSO_CLASS_BEGIN(BaseMaterial, Material)

public:
[[deprecated]] BaseMaterial(const SceneClass& sceneClass, const std::string& name);

    virtual void update();

private:
    static void shade(const Material* self,
                      moonray::shading::TLState *tls,
                      const State& state,
                      BsdfBuilder& bsdfBuilder);

RDL2_DSO_CLASS_END(BaseMaterial)


//---------------------------------------------------------------------------

BaseMaterial::BaseMaterial(const SceneClass& sceneClass, const std::string& name) :
    Parent(sceneClass, name)
{
    mShadeFunc = BaseMaterial::shade;
    mShadeFuncv = (ShadeFuncv) ispc::BaseMaterial_getShadeFunc();
}

void
BaseMaterial::update()
{
}

void
BaseMaterial::shade(const Material* self, moonray::shading::TLState *tls,
                    const State& state, BsdfBuilder& bsdfBuilder)
{
    const BaseMaterial* me = static_cast<const BaseMaterial*>(self);
    moonray::shading::Bsdf *bsdf = const_cast<moonray::shading::Bsdf*>(bsdfBuilder.getBsdf());

    scene_rdl2::alloc::Arena *arena = getArena(tls);

    // Fully opaque and no specular or other non-diffuse terms by default
    float opacityFactor = 1.0f;
    Color directionalDiffuseColor(0.0f);
    Color specularColor(0.0f);
    float specularRoughness = 0.0f;
    float retroreflectivity(0.0f);
    Color transmissionColor(0.0f);
    Color omTransmissionColor(1.0f);

    Vec3f anisotropicDirection(0.0f);
    float anisotropy = evalFloat(me, attrAnisotropy, tls, state);

    // Get minimum roughness used to apply roughness clamping.
    const Vec2f minRoughnessAniso = state.getMinRoughness();
    const float minRoughness = min(minRoughnessAniso.x, minRoughnessAniso.y);

    // Only do opacity, specular, etc. if we are not on a caustics path or unless
    // we really want caustics
    if (!state.isCausticPath()  ||  me->get(attrCastsCaustics)) {

        if (me->get(attrOpacity)) {
            // Transparency (== 1 - opacity) takes priority over everything else. All
            // the other terms of the Bsdf will be proportional to opacity
            opacityFactor = evalFloat(me, attrOpacityFactor, tls, state);
            if (!isEqual(opacityFactor, 1.0f)) {
                auto lobe = arena->allocWithArgs<moonray::shading::MirrorTransmissionBsdfLobe>(state.getN(),
                                                                                  1.f,
                                                                                  1.f,
                                                                                  sWhite);
                lobe->setScale(Color(1.0f - opacityFactor));
                bsdf->addLobe(lobe);
            }

            // Early out if fully transparent
            if (isZero(opacityFactor)) {
                return;
            }
        }

        // Evaluate specular and other caustic-sensitive components
        specularColor = evalColorComponent(me, attrSpecular, attrSpecularFactor,
                attrSpecularColor, tls, state);

        specularRoughness = evalFloat(me, attrSpecularRoughness, tls, state);

        retroreflectivity = evalFloat(me, attrRetroreflectivity, tls, state);

        directionalDiffuseColor = evalColorComponent(me, attrDirectionalDiffuse,
                attrDirectionalDiffuseFactor, attrDirectionalDiffuseColor, tls, state);

        transmissionColor = evalColorComponent(me, attrTransmission,
                attrTransmissionFactor, attrTransmissionColor, tls, state);
        omTransmissionColor = sWhite - transmissionColor;

        // Anisotropic direction
        if (!isZero(anisotropy)) {
            const Vec3f T = normalize(state.getdPds());
            const ReferenceFrame frame(state.getN(), T);
            const Vec2f dir = evalVec2f(me, attrAnisotropicDirection, tls, state);
            anisotropicDirection = frame.localToGlobal(
                    normalize(Vec3f(dir.x, dir.y, 0.0f)));
        }
    }


    // We use a top specular lobe and use it to attenuate energy from lower lobes
    moonray::shading::Fresnel *omSpecFresnel = nullptr;

    // TODO: avoid doing these eval() calls if the result is unused
    const Vec3f N = evalNormal(me, attrInputNormal, attrInputNormalDial, 
        attrInputNormalSpace, tls, state);
    const float fresnelFactor = (me->get(attrUseFresnel)  ?
            evalFloat(me, attrFresnelFactor, tls, state)  :  0.0);
    const ShaderIor ior(state, me->get(attrIndexOfRefraction));

    const bool isTransmissive = !isBlack(transmissionColor);
    // Energy Compensation Params
    float favg, favgInv;
    moonray::shading::averageFresnelReflectance(
            ior.getTransmitted()/ior.getIncident(),
            favg, favgInv);

    // Top Specular lobe
    if (!isBlack(specularColor) && (retroreflectivity < 1.0f)) {
        moonray::shading::BsdfLobe *lobe;

        // Apply roughness clamping
        float roughness = max(specularRoughness, minRoughness);

        if (roughness < sEpsilon) {
            lobe = arena->allocWithArgs<moonray::shading::MirrorReflectionBsdfLobe>(N);
        } else {
            if (isZero(anisotropy)) {
                lobe = arena->allocWithArgs<moonray::shading::CookTorranceBsdfLobe>(N, roughness,
                                                                           sWhite*favg, sWhite*favgInv,
                                                                           ior.getIncident(),
                                                                           ior.getTransmitted(),
                                                                           isTransmissive); //coupledWithTransmission

            } else {
                // Compute anisotropy using un-clamped roughness, then clamp
                float uRoughness = anisotropy > 0.0f ?
                        specularRoughness * (1.0f - anisotropy) : specularRoughness;
                float vRoughness = anisotropy < 0.0f ?
                        specularRoughness * (1.0f + anisotropy) : specularRoughness;
                uRoughness = max(uRoughness, minRoughness);
                vRoughness = max(vRoughness, minRoughness);
                lobe = arena->allocWithArgs<moonray::shading::AnisoCookTorranceBsdfLobe>(N,
                        anisotropicDirection, uRoughness, vRoughness);
                roughness = 0.5f * (uRoughness + vRoughness);
            }
        }

#if USE_SCHLICK
        auto specFresnel = arena->allocWithArgs<moonray::shading::SchlickFresnel>(
                specularColor, fresnelFactor, ior.getRatio());
#else
        auto specFresnel = arena->allocWithArgs<moonray::shading::DielectricFresnel>(
                ior.getIncident(), ior.getTransmitted());
#endif
        lobe->setFresnel(specFresnel);

        // Energy conservation: dim specular based on opacity/retro-reflectivity
        lobe->setScale(Color((1.0f - retroreflectivity) * opacityFactor));
#if USE_SCHLICK
        omSpecFresnel = arena->allocWithArgs<moonray::shading::OneMinusRoughSchlickFresnel>(
                specFresnel, roughness);
#else
        omSpecFresnel = arena->allocWithArgs<moonray::shading::OneMinusFresnel>(specFresnel);
#endif
        if (me->get(attrIridescence)) {
            const float iridescenceFac = me->get(attrIridescenceFactor);
            const Color iridescencePrimary = me->get(attrIridescencePrimaryColor);
            const Color iridescenceSecondary = me->get(attrIridescenceSecondaryColor);
            const bool iridescenceFlipHue = me->get(attrIridescenceFlipHueDirection);
            const float iridescenceThickness = evalFloat(me, attrIridescenceThickness, tls, state);
            const float iridescenceExponent = evalFloat(me, attrIridescenceExponent, tls, state);
            const float iridescenceAt0 = evalFloat(me, attrIridescenceAt0, tls, state);

            // Pass along the mirror/glossy lobe created above to the iridescence lobe
            lobe = arena->allocWithArgs<moonray::shading::IridescenceBsdfLobe>(lobe,
                                                                      N,
                                                                      iridescenceFac,
                                                                      static_cast<ispc::SHADING_IridescenceColorMode>(0), // use hue interpolation
                                                                      iridescencePrimary,
                                                                      iridescenceSecondary,
                                                                      iridescenceFlipHue,
                                                                      static_cast<ispc::ColorRampControlSpace>(0),    // ramp interpolation mode
                                                                      0,            // ramp num points
                                                                      nullptr,      // ramp positions
                                                                      nullptr,      // ramp interpolators
                                                                      nullptr,      // ramp colors
                                                                      iridescenceThickness,
                                                                      iridescenceExponent,
                                                                      iridescenceAt0,
                                                                      1.0f);
        }

        lobe->setLabel(aovSpecular);
        bsdf->addLobe(lobe);
    }

    // Retroreflection Lobe
    if (retroreflectivity > 0.0f) {
        moonray::shading::BsdfLobe *lobe;
        const float roughness = max(specularRoughness, minRoughness);
        if (roughness < sEpsilon) {
            lobe = arena->allocWithArgs<moonray::shading::MirrorRetroreflectionBsdfLobe>(N);
        } else {
            lobe = arena->allocWithArgs<moonray::shading::RetroreflectionBsdfLobe>(N, roughness);
        }
        lobe->setScale(Color(opacityFactor * retroreflectivity));
        lobe->setLabel(aovSpecular);
        bsdf->addLobe(lobe);
    }

    // Directional Diffuse
    if (!isBlack(directionalDiffuseColor)) {
        moonray::shading::BsdfLobe *lobe;

        // Evaluate roughness, then apply roughness clamping
        const float ddRoughness = evalFloat(me, attrDirectionalDiffuseRoughness,
                tls, state);
        float roughness = max(ddRoughness, minRoughness);

        if (roughness < sEpsilon) {
            lobe = arena->allocWithArgs<moonray::shading::MirrorReflectionBsdfLobe>(N);
        } else {
            if (isZero(anisotropy)) {
                lobe = arena->allocWithArgs<moonray::shading::CookTorranceBsdfLobe>(N, roughness);
            } else {
                // Compute anisotropy using un-clamped roughness, then clamp
                float uRoughness = anisotropy > 0.0f ?
                        ddRoughness * (1.0f - anisotropy) : ddRoughness;
                float vRoughness = anisotropy < 0.0f ?
                        ddRoughness * (1.0f + anisotropy) : ddRoughness;
                uRoughness = max(uRoughness, minRoughness);
                vRoughness = max(vRoughness, minRoughness);
                lobe = arena->allocWithArgs<moonray::shading::AnisoCookTorranceBsdfLobe>(N,
                        anisotropicDirection, uRoughness, vRoughness);
            }
        }
        auto fresnel = arena->allocWithArgs<moonray::shading::SchlickFresnel>(
                directionalDiffuseColor, fresnelFactor, ior.getRatio());
        lobe->setFresnel(fresnel);

        // Energy conservation: dim based on transmission and opacity
        lobe->setScale(opacityFactor * omTransmissionColor);

        if (omSpecFresnel) {
            // Energy conservation: dim Dd based on 1-F(top lobe)
            moonray::shading::BsdfLobe *underLobe = arena->allocWithArgs<moonray::shading::UnderBsdfLobe>(lobe, N);
            underLobe->setFresnel(omSpecFresnel);
            underLobe->setLabel(aovDirectionalDiffuse);
            bsdf->addLobe(underLobe);
        } else {
            lobe->setLabel(aovDirectionalDiffuse);
            bsdf->addLobe(lobe);
        }
    }

    // Translucency
    const Color translucencyColor = evalColorComponent(me, attrTranslucency,
            attrTranslucencyFactor, attrTranslucencyColor, tls, state);
    if (!isBlack(translucencyColor)) {
        // Check diffuse_depth and use the a lambert bsdf approximation
        // TODO: Should we check if the ray footprint is larger than the
        // scattering radius to use the approx instead ?
        if (!state.isHifi()) {

            // TODO: implement and use Jensen et al. 2001 Brdf approximation
            // TODO: Apply omSpecFresnel
            auto lobe = arena->allocWithArgs<moonray::shading::LambertBsdfLobe>(N, translucencyColor, true);
            // Energy conservation: dim based on transmission and opacity
            lobe->setScale(opacityFactor * omTransmissionColor);
            if (omSpecFresnel) {
                // Energy conservation: dim translucency based on 1-F(top lobe)
                lobe->setFresnel(omSpecFresnel);
            }
            lobe->setLabel(aovTranslucency);
            bsdf->addLobe(lobe);
        } else {
            // TODO: modify DipoleBssrdf xtor to accommodate this;
            const Color translucencyRadius =
                    evalColor(me, attrTranslucencyFalloff, tls, state) *
                    evalFloat(me, attrTranslucencyRadius, tls, state);
            // Energy conservation: dim based on transmission and opacity
            auto bssrdf = moonray::shading::createBSSRDF(ispc::SUBSURFACE_DIPOLE_DIFFUSION, arena, N,
                    opacityFactor * omTransmissionColor * translucencyColor,
                    translucencyRadius, me, nullptr);
#if 0
            // Use physical parameters interface
            bssrdf = new DipoleBssrdf(N, 1.0f, translucencyColor,
                    translucencyRadius, 0.001);

            // Skin1 from Jensen et al. 2001.
            bssrdf = new DipoleBssrdf(N, 1.0f,
                     Color3s(0.032f, 0.17f, 0.48f), Color3s(0.74f, 0.88f, 1.01f), 0.001);
#endif
            if (omSpecFresnel) {
                // Energy conservation: dim translucency based on 1-F(top lobe)
                bssrdf->setTransmissionFresnel(omSpecFresnel);
            }
            bssrdf->setLabel(aovTranslucency);
            bsdf->setBssrdf(bssrdf);
        }
    }

    // Transmission
    if (isTransmissive) {

        moonray::shading::BsdfLobe *lobe;
        // Specular Transmission
        // Apply roughness clamping
        float roughness = max(specularRoughness, minRoughness);

        if (roughness < sEpsilon  ||  isOne(ior.getRatio())) {
            lobe = arena->allocWithArgs<moonray::shading::MirrorTransmissionBsdfLobe>(N,
                                                                             ior.getIncident(),
                                                                             ior.getTransmitted(),
                                                                             transmissionColor);
        } else {
            lobe = arena->allocWithArgs<moonray::shading::TransmissionCookTorranceBsdfLobe>(
                    N, roughness,
                    ior.getIncident(),
                    ior.getTransmitted(),
                    transmissionColor,
                    favg, favgInv);
        }
        if (omSpecFresnel) {
            // Energy conservation: dim transmission based on 1-F(top lobe)
            lobe->setFresnel(omSpecFresnel);
        }
        // Energy conservation: dim based on opacity
        lobe->setScale(Color(opacityFactor));
        lobe->setLabel(aovTransmission);
        bsdf->addLobe(lobe);
    }

    // Diffuse
    const Color diffuseColor = evalColorComponent(me, attrDiffuse,
            attrDiffuseFactor, attrDiffuseColor, tls, state);
    if (!isBlack(diffuseColor)) {
        auto lobe = arena->allocWithArgs<moonray::shading::LambertBsdfLobe>(N, diffuseColor, true);
        if (omSpecFresnel) {
            // Energy conservation: dim diffuse based on 1-F(top lobe)
            lobe->setFresnel(omSpecFresnel);
        }
        // Energy conservation: dim based on transmission and opacity
        lobe->setScale(opacityFactor * omTransmissionColor);
        lobe->setLabel(aovDiffuse);
        bsdf->addLobe(lobe);
    }

    // Translucent Diffuse
    // NOTE - We are 'adding' this lobe directly to support legacy material
    // conversions where a translucent diffuse material could be 'added' with
    // a base material.
    // Ideally, there'd be support for energy conservation similar to the 'specular transmission'
    // lobe in terms of reducing the lambertian diffuse by 'oneMinusColor'
    const Color translucentDiffuseColor = evalColorComponent(me,
                                                             attrTranslucentDiffuse,
                                                             attrTranslucentDiffuseFactor,
                                                             attrTranslucentDiffuseColor,
                                                             tls,
                                                             state);
    if (!isBlack(translucentDiffuseColor)) {
        auto lobe = arena->allocWithArgs<moonray::shading::LambertBsdfLobe>(-N, translucentDiffuseColor, false);
        if (omSpecFresnel) {
            // Energy conservation: dim transmission based on 1-F(top lobe)
            lobe->setFresnel(omSpecFresnel);
        }
        lobe->setScale(Color(opacityFactor));
        bsdf->addLobe(lobe);
    }


    // Self-emission
    const Color emissionColor = evalColorComponent(me, attrEmission,
            attrEmissionFactor, attrEmissionColor, tls, state);
    // Energy conservation: dim based on opacity
    bsdf->setSelfEmission(opacityFactor * emissionColor);
}

//---------------------------------------------------------------------------

