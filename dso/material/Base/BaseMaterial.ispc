// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file BaseMaterial.ispc

#warning "BaseMaterial.ispc is deprecated!"

#include "attributes.isph"
#include "labels.isph"

#include <moonray/rendering/shading/ispc/MaterialApi.isph>
#include <moonray/rendering/shading/ispc/Closure.isph>
#include <moonray/rendering/shading/ispc/bsdf/Fresnel.isph>
#include <moonray/rendering/shading/ispc/Ior.isph>

static void
shade(const uniform Material *      uniform  me,
            uniform ShadingTLState *uniform  tls,
      const varying State                    &state,
            varying BsdfBuilder              &bsdfBuilder)
{
    varying Closure * uniform closure = BsdfBuilder_getClosure(bsdfBuilder);

    // Fully opaque and no specular or other non-diffuse terms by default
    float opacityFactor = 1.0f;
    Color directionalDiffuseColor = Color_ctor(0.0f);
    Color specularColor = Color_ctor(0.0f);
    float specularRoughness = 0.0f;
    float retroreflectivity = 0.0f;
    Color transmissionColor = Color_ctor(0.0f);
    Color omTransmissionColor = Color_ctor(1.0f);

    Vec3f anisotropicDirection = Vec3f_ctor(0.0f);
    varying float anisotropy = evalAttrAnisotropy(me, tls, state);

    // Get minimum roughness used to apply roughness clamping.
    const Vec2f minRoughnessAniso = getMinRoughness(state);
    const float minRoughness = min(minRoughnessAniso.x, minRoughnessAniso.y);

    // Only do opacity, specular, etc. if we are not on a caustics path or unless
    // we really want caustics
    if (!isCausticPath(state) || getAttrCastsCaustics(me)) {

        if (getAttrOpacity(me)) {
            // Transparency (== 1 - opacity) takes priority over everything else. All
            // the other terms of the closure will be proportional to opacity
            opacityFactor = evalAttrOpacityFactor(me, tls, state);
            if (!isEqual(opacityFactor, 1.0f)) {
                Closure_addMirrorTransmissionBsdfLobe(closure, tls, state,
                                                /* scale = */ Color_ctor(1.0f - opacityFactor),
                                                /* fresnel = */ NULL,
                                                /* underFresnel = */ NULL,
                                                /* normal = */ state.mN,
                                                /* etaI = */ 1.f,
                                                /* etaT = */ 1.f,
                                                /* abbe = */ 0.0f,
                                                /* labels = */ 0);
            }

            // Early out if fully transparent
            if (isEqual(opacityFactor, 0.0f)) {
                return;
            }
        }

        // Evaluate specular and other caustic-sensitive components
        specularColor = evalCompSpecular(me, tls, state);

        specularRoughness = evalAttrSpecularRoughness(me, tls, state);

        retroreflectivity = evalAttrRetroreflectivity(me, tls, state);

        directionalDiffuseColor = evalCompDirectionalDiffuse(me, tls, state);

        transmissionColor = evalCompTransmission(me, tls, state);
        omTransmissionColor = Color_ctor(1.0f) - transmissionColor;

        // Anisotropic direction
        if (!isZero(anisotropy)) {
            const Vec3f T = normalize(state.mdPds);
            ReferenceFrame frame;
            ReferenceFrame_init(frame, state.mN, T);
            const Vec2f dir = evalAttrAnisotropicDirection(me, tls, state);
            anisotropicDirection =
                localToGlobal(frame, normalize(Vec3f_ctor(dir.x, dir.y, 0.0f)));
        }
    }

    // We use a top specular lobe and use it to attenuate energy from lower lobes
    varying Fresnel * uniform omSpecFresnel = NULL;

    // TODO: avoid doing these eval() calls if the result is unused
    const Vec3f N = evalNormalInput(me, tls, state);
    const float fresnelFactor = (getAttrUseFresnel(me) ?
                                 evalAttrFresnelFactor(me, tls, state) : 0.0);
    ShaderIor ior;
    ShaderIor_init(state, getAttrIndexOfRefraction(me), &ior, false);
    float favg, favgInv;
    averageFresnelReflectance(ior.mTransmitted/ior.mIncident,
                              favg, favgInv);


    // Top Specular lobe
    if (!isBlack(specularColor) && (retroreflectivity < 1.0f)) {

        // Apply roughness clamping
        float roughness = max(specularRoughness, minRoughness);

        float uRoughness = anisotropy > 0.0f ?
                specularRoughness * (1.0f - anisotropy) : specularRoughness;
        float vRoughness = anisotropy < 0.0f ?
                specularRoughness * (1.0f + anisotropy) : specularRoughness;
        uRoughness = max(uRoughness, minRoughness);
        vRoughness = max(vRoughness, minRoughness);

        // Prepare the fresnel
        varying Fresnel * uniform specFresnel = Fresnel_createSchlick(tls, state,
                specularColor, fresnelFactor, ior.mRatio);

        // Energy conservation: dim specular based on opacity/retroreflectivity
        const float lobeScale = (1.0f - retroreflectivity) * opacityFactor;

        if (!getAttrIridescence(me)) {
            if (roughness < sEpsilon) {
                Closure_addMirrorReflectionBsdfLobe(closure, tls, state,
                        /* scale = */ Color_ctor(lobeScale),
                        /* fresnel = */ specFresnel,
                        /* underFresnel = */ NULL,
                        /* normal = */ N,
                        /* labels = */ aovSpecular);
            } else {
                if (isZero(anisotropy)) {
                    Closure_addCookTorranceBsdfLobe(closure, tls, state,
                            /* scale = */ Color_ctor(lobeScale),
                            /* fresnel = */ specFresnel,
                            /* underFresnel = */ NULL,
                            /* normal = */ N,
                            /* roughness = */ roughness,
                            /* favg = */ sWhite*favg,
                            /* favgInv = */ sWhite*favgInv,
                            /* labels = */ aovSpecular);
                } else {
                    // Compute anisotropy using un-clamped roughness, then clamp
                    Closure_addAnisoCookTorranceBsdfLobe(closure, tls, state,
                            /* scale = */ Color_ctor(lobeScale),
                            /* fresnel = */ specFresnel,
                            /* underFresnel = */ NULL,
                            /* normal = */ N,
                            /* anisotropicDirection = */ anisotropicDirection,
                            /* uRoughness = */ uRoughness,
                            /* vRoughness = */ vRoughness,
                            /* labels = */ aovSpecular);
                    roughness = 0.5f * (uRoughness + vRoughness);
                }
            }
        } else { // iridescence
            const varying float iridescenceFac = evalAttrIridescenceFactor(me, tls, state);
            const varying Color primary = getAttrIridescencePrimaryColor(me);
            const varying Color secondary = getAttrIridescenceSecondaryColor(me);
            const varying bool  flipHue = getAttrIridescenceFlipHueDirection(me);
            const varying float thickness = evalAttrIridescenceThickness(me, tls, state);
            const varying float exponent = evalAttrIridescenceExponent(me, tls, state);
            const varying float iridescenceAt0 = evalAttrIridescenceAt0(me, tls, state);

            Closure_addIridescenceBsdfLobe(closure, tls, state,
                                           /* scale = */ Color_ctor(lobeScale),
                                           /* fresnel = */ specFresnel,
                                           /* underFresnel = */ NULL,
                                           /* normal = */ N,
                                           /* roughness = */ roughness,
                                           /* anisotropy = */ anisotropy,
                                           /* anisotropicDirection = */ anisotropicDirection,
                                           /* uRoughness = */ uRoughness,
                                           /* vRoughness = */ vRoughness,
                                           /* iridescence fac = */ iridescenceFac,
                                           /* primary = */ primary,
                                           /* secondary = */ secondary,
                                           /* flip hue = */ flipHue,
                                           /* thickness = */ thickness,
                                           /* exponent = */ exponent,
                                           /* iridescenceAt0 = */ iridescenceAt0,
                                           /* iridescenceAt90 = */ 1.0f,
                                           /* labels = */ aovSpecular);
            // If any of the lanes have anisotropy, iridescence closure will add anisotropic lobes
            // in all lanes
            if (any(!isZero(anisotropy))) {
                roughness = 0.5f * (uRoughness + vRoughness);
            }
        }
        omSpecFresnel = Fresnel_createOneMinusRoughSchlick(tls, state,
                specFresnel, roughness);
    }

    // Retroreflectivity
    if (retroreflectivity > 0.0f) {
        // Energy conservation: dim specular based on opacity/retroreflectivity
        const float lobeScale = retroreflectivity * opacityFactor;
        const float roughness = max(specularRoughness, minRoughness);
        if (roughness < sEpsilon) {
            Closure_addMirrorRetroreflectionBsdfLobe(closure, tls, state,
                    /* scale = */ Color_ctor(lobeScale),
                    /* fresnel = */ NULL,
                    /* underFresnel = */ NULL,
                    /* normal = */ N,
                    /* labels = */ aovSpecular);
        } else {
            Closure_addRetroreflectionBsdfLobe(closure, tls, state,
                    // Energy conservation: dim specular based on opacity
                    /* scale = */ Color_ctor(lobeScale),
                    /* fresnel = */ NULL,
                    /* underFresnel = */ NULL,
                    /* normal = */ N,
                    /* roughness = */ roughness,
                    /* labels = */ aovSpecular);
        }
    }

    // Directional Diffuse
    if (!isBlack(directionalDiffuseColor)) {
        // Evaluate roughness, then apply roughness clamping
        const float ddRoughness = evalAttrDirectionalDiffuseRoughness(me, tls, state);
        float roughness = max(ddRoughness, minRoughness);

        // Prepare the fresnel
        varying Fresnel * uniform fresnel =
            Fresnel_createSchlick(tls, state, directionalDiffuseColor,
                    fresnelFactor, ior.mRatio);

        // Energy conservation: dim based on transmission and opacity
        const Color scale = opacityFactor * omTransmissionColor;

        if (roughness < sEpsilon) {
            Closure_addMirrorReflectionBsdfLobe(closure, tls, state,
                    /* scale = */ scale,
                    /* fresnel = */ fresnel,
                    /* underFresnel = */ omSpecFresnel,
                    /* normal = */ N,
                    /* labels = */ aovDirectionalDiffuse);
        } else {
            if (isZero(anisotropy)) {
                Closure_addCookTorranceBsdfLobe(closure, tls, state,
                        /* scale = */ scale,
                        /* fresnel = */ fresnel,
                        /* underFresnel = */ omSpecFresnel,
                        /* normal = */ N,
                        /* roughness = */ roughness,
                        /* favg = */ sWhite*favg,
                        /* favgInv = */ sWhite*favgInv,
                        /* labels = */ aovDirectionalDiffuse);
            } else {
                // Compute anisotropy using un-clamped roughness, then clamp
                float uRoughness = anisotropy > 0.0f ?
                        ddRoughness * (1.0f - anisotropy) : ddRoughness;
                float vRoughness = anisotropy < 0.0f ?
                        ddRoughness * (1.0f + anisotropy) : ddRoughness;
                uRoughness = max(uRoughness, minRoughness);
                vRoughness = max(vRoughness, minRoughness);
                Closure_addAnisoCookTorranceBsdfLobe(closure, tls, state,
                        /* scale = */ scale,
                        /* fresnel = */ fresnel,
                        /* underFresnel = */ omSpecFresnel,
                        /* normal = */ N,
                        /* anisotropicDirection = */ anisotropicDirection,
                        /* uRoughness = */ uRoughness,
                        /* vRoughness = */ vRoughness,
                        /* labels = */ aovDirectionalDiffuse);
            }
        }
    }

    // Translucency
    const Color translucencyColor = evalCompTranslucency(me, tls, state);
    if (!isBlack(translucencyColor)) {
        // Energy conservation: dim based on transmission and opacity
        const Color scale = opacityFactor * omTransmissionColor;

        // Check diffuse_depth and use the a lambert bsdf approximation
        if (!isHifi(state)) {
            // TODO: implement and use Jensen et al. 2001 Brdf approximation
            Closure_addLambertBsdfLobe(closure, tls, state,
                                       /* scale = */ scale,
                                       // Energy conservation: dim translucency based on 1-F(top lobe)
                                       /* fresnel = */ omSpecFresnel,
                                       /* underFresnel = */ NULL,
                                       /* normal = */ N,
                                       /* albedo = */ translucencyColor,
                                       /* isReflection = */ true,
                                       /* labels = */ aovTranslucency);
        } else {
            // TODO: modify Bssrdh xtor to accommodate this;
            // TODO: Apply omSpecFresnel
            // since both falloff and radius are bindable,
            // these attrs should be eval not get right?
            const Color translucencyRadius = evalAttrTranslucencyFalloff(me, tls, state) *
                                             evalAttrTranslucencyRadius(me, tls, state);

            // Closure_addDipoleBssrdf expects 'scale' arg to be varying
            const varying Color white = sWhite;
            Closure_addDipoleBssrdf(closure, tls, state,
                                    /* scale = */ white,
                                    // Energy conservation: dim translucency based on 1-F(top lobe)
                                    /* fresnel = */ omSpecFresnel,
                                    /* normal = */ N,
                                    /* translucentColor = */ scale * translucencyColor,
                                    /* translucencyRadius = */ translucencyRadius,
                                    /* traceSet = */ nullptr,
                                    /* material = */ nullptr, 
                                    /* evalNormalMapFn = */ 0,
                                    /* labels = */ aovTranslucency);
        }
    }

    // Transmission
    if (!isBlack(transmissionColor)) {
        const Color lobeScale = opacityFactor * transmissionColor;
        // Apply roughness clamping
        float roughness = max(specularRoughness, minRoughness);
        cif (roughness < sEpsilon  ||  isOne(ior.mRatio)) {
            Closure_addMirrorTransmissionBsdfLobe(closure, tls, state,
                    /* scale = */ lobeScale,
                    // Energy conservation: dim transmission based on 1-F(top lobe)
                    /* fresnel = */ omSpecFresnel,
                    /* underFresnel = */ NULL,
                    /* normal = */ N,
                    /* etaI = */ ior.mIncident,
                    /* etaT = */ ior.mTransmitted,
                    /* abbeNumber = */ 0.0f,
                    /* labels = */ aovTransmission);
        } else {
            Closure_addTransmissionCookTorranceBsdfLobe(closure, tls, state,
                    /* scale = */ lobeScale,
                    // Energy conservation: dim transmission based on 1-F(top lobe)
                    /* fresnel = */ omSpecFresnel,
                    /* underFresnel = */ NULL,
                    /* normal = */ N,
                    /* roughness = */ roughness,
                    /* iorIncident = */ ior.mIncident,
                    /* iorTransmitted = */ ior.mTransmitted,
                    /* favg = */ favg,
                    /* favgInv = */ favgInv,
                    /* abbeNumber = */ 0.0f,
                    /* labels = */ aovTransmission);
        }
    }

    // Diffuse
    const Color diffuseColor = evalCompDiffuse(me, tls, state);
    if (!isBlack(diffuseColor)) {
        Closure_addLambertBsdfLobe(closure, tls, state,
                                   /* scale = */ opacityFactor * omTransmissionColor,
                                   // Energy conservation: dim diffuse based on 1-F(top lobe)
                                   /* fresnel = */ omSpecFresnel,
                                   /* underFresnel = */ NULL,
                                   /* normal = */ N,
                                   /* albedo = */ diffuseColor,
                                   /* isReflection = */ true,
                                   /* labels = */ aovDiffuse);

    }

    // Tranlucent diffuse lobe
    const Color translucentDiffuse = evalCompTranslucentDiffuse(me, tls, state);
    if (!isBlack(translucentDiffuse)) {
        Closure_addLambertBsdfLobe(closure, tls, state,
                                   /* scale = */ Color_ctor(opacityFactor),
                                   // Energy conservation: dim transmission based on 1-F(top lobe)
                                   /* fresnel = */ omSpecFresnel,
                                   /* underFresnel = */ NULL,
                                   /* normal = */ Vec3f_ctor(-N.x, -N.y, -N.z),
                                   /* albedo = */ translucentDiffuse,
                                   /* isReflection = */ false,
                                   /* labels = */ 0);
    }

    // Self-emission
    const Color emissionColor = evalCompEmission(me, tls, state);
    // Energy conservation: dim based on opacity
    Closure_setSelfEmission(closure, opacityFactor * emissionColor);
}

DEFINE_MATERIAL_SHADER(BaseMaterial, shade)

