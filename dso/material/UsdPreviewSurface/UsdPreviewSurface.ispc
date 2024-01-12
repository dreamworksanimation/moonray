// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

// TM and (c) 202102022 DreamWorks Animation LLC.  All Rights Reserved.

/// @file UsdPreviewSurface.ispc

#include "attributes.isph"
#include "labels.isph"

#include <moonray/rendering/shading/ispc/MaterialApi.isph>

void
addMirrorDielectricLobe(varying BsdfBuilder& bsdfBuilder,
                        const varying Vec3f& N,
                        const varying float ior,
                        const varying float weight,
                        const varying float opacity,
                        const uniform int reflectionLabel=aovSpecular,
                        const uniform int transmissionLabel=aovSpecularTransmission,
                        const uniform BsdfBuilderBehavior behavior=BSDFBUILDER_PHYSICAL)
{
    const varying Col3f tint = sWhite;
    MirrorBSDF dielectricBSDF;
    MirrorBSDF_init(dielectricBSDF,
                    N, ior,
                    tint, // transmission color
                    0.0f, // abbe number
                    ior,
                    weight,
                    1.0f - opacity); // transmission

    BsdfBuilder_addMirrorBSDF(bsdfBuilder,
                              dielectricBSDF,
                              1.0f, // weight
                              behavior,
                              reflectionLabel,
                              transmissionLabel);
}

void
addMicrofacetDielectricLobe(varying BsdfBuilder& bsdfBuilder,
                            const varying Vec3f& N,
                            const varying float ior,
                            const varying float roughness,
                            const varying float weight,
                            const varying float opacity,
                            const uniform int reflectionLabel=aovSpecular,
                            const uniform int transmissionLabel=aovSpecular,
                            const uniform BsdfBuilderBehavior behavior=BSDFBUILDER_PHYSICAL)
{
    const varying Col3f tint = sWhite;
    MicrofacetIsotropicBSDF dielectricBSDF;
    MicrofacetIsotropicBSDF_init(dielectricBSDF,
                                 N, ior, roughness,
                                 MICROFACET_DISTRIBUTION_GGX,
                                 MICROFACET_GEOMETRIC_SMITH,
                                 tint, // transmission color
                                 0.0f, // abbe number
                                 ior,
                                 weight,
                                 1.0f - opacity); // transmission

    BsdfBuilder_addMicrofacetIsotropicBSDF(bsdfBuilder,
                                           dielectricBSDF,
                                           1.0f, // weight
                                           behavior,
                                           reflectionLabel,
                                           transmissionLabel);
}

void
addMirrorConductorLobe(varying BsdfBuilder& bsdfBuilder,
                       const varying Vec3f& N,
                       const varying Color& color,
                       const varying float weight,
                       const uniform int label=aovSpecular,
                       const uniform BsdfBuilderBehavior behavior=BSDFBUILDER_PHYSICAL)
{
    MirrorBRDF conductorBRDF;
    MirrorBRDF_init(conductorBRDF, N, color, color);
    BsdfBuilder_addMirrorBRDF(bsdfBuilder, conductorBRDF,
                              weight,
                              behavior,
                              label);
}

void
addMicrofacetConductorLobe(varying BsdfBuilder& bsdfBuilder,
                           const varying Vec3f& N,
                           const varying Color& color,
                           const varying float roughness,
                           const varying float weight,
                           const uniform int label=aovSpecular,
                           const uniform BsdfBuilderBehavior behavior=BSDFBUILDER_PHYSICAL)
{
    MicrofacetIsotropicBRDF conductorBRDF;
    MicrofacetIsotropicBRDF_init(conductorBRDF, N, color, color, roughness,
                                 MICROFACET_DISTRIBUTION_GGX,
                                 MICROFACET_GEOMETRIC_SMITH);
    BsdfBuilder_addMicrofacetIsotropicBRDF(bsdfBuilder,
                                           conductorBRDF,
                                           weight,
                                           behavior,
                                           label);
}

enum SpecularWorkflow {
    SPECULAR_WORKFLOW_METALNESS,
    SPECULAR_WORKFLOW_SPECULAR
};

ISPC_UTIL_EXPORT_ENUM_TO_HEADER(SpecularWorkflow);

struct UsdPreviewSurface
{
    const uniform NormalMap * uniform mNormalMap;
    uniform intptr_t mSampleNormalFunc;
    uniform int mReversedNormalsIndx;
};
ISPC_UTIL_EXPORT_UNIFORM_STRUCT_TO_HEADER(UsdPreviewSurface);

static void
shade(const uniform Material * uniform  me,
      uniform ShadingTLState * uniform tls,
      const varying State &state,
      varying BsdfBuilder &bsdfBuilder)
{
    const uniform UsdPreviewSurface * uniform usdmtl =
        MATERIAL_GET_ISPC_CPTR(UsdPreviewSurface, me);

    // Transform inputNormal from tangent space to shade space
    const Vec3f inputNormal = evalAttrNormal(me, tls, state);
    varying bool reversedNormals = false;
    if (isProvided(state, usdmtl->mReversedNormalsIndx)) {
        reversedNormals = getBoolAttribute(tls, state, usdmtl->mReversedNormalsIndx);
    }

    Vec3f N;
    const Vec3f statedPds = reversedNormals ? state.mdPds * -1.0f : state.mdPds * 1.0f;
    if (isZero(length(statedPds))) {
        N = getN(state);
    } else {
        const ReferenceFrame frame = ReferenceFrame_ctor(getN(state), normalize(statedPds));
        N = normalize(localToGlobal(frame, inputNormal));
    }

    const Color albedo = evalAttrDiffuseColor(me, tls, state);
    const float ior = evalAttrIor(me, tls, state);
    const float roughness = saturate(evalAttrRoughness(me, tls, state));
    const float opacityThreshold = saturate(evalAttrOpacityThreshold(me, tls, state));
    const float opacity = opacityThreshold > 0.0f ? 1.0f : saturate(evalAttrOpacity(me, tls, state));

    // Emission
    const Color emissiveColor = evalAttrEmissiveColor(me, tls, state);
    if (!isBlack(emissiveColor)) {
        BsdfBuilder_addEmission(bsdfBuilder, emissiveColor);
    }

    // Clearcoat
    const float clearcoat = evalAttrClearcoat(me, tls, state);
    if (!isZero(clearcoat)) {
        const float clearcoatRoughness = evalAttrClearcoatRoughness(me, tls, state);
        if (isZero(clearcoatRoughness)) { // mirror
            addMirrorDielectricLobe(bsdfBuilder, N, ior, clearcoat,
                                    1.0f, // opacity
                                    aovOuterSpecular,
                                    aovSpecularTransmission,
                                    BSDFBUILDER_ADDITIVE);
        } else { // microfacet
            addMicrofacetDielectricLobe(bsdfBuilder, N, ior,
                                        clearcoatRoughness, clearcoat,
                                        1.0f, // opacity
                                        aovOuterSpecular,
                                        aovSpecularTransmission,
                                        BSDFBUILDER_ADDITIVE);
        }
    }

    const float weight = 1.0f;

    // Specular
    const uniform SpecularWorkflow specularWorkflow = (SpecularWorkflow)getAttrUseSpecularWorkflow(me);
    if (specularWorkflow == SPECULAR_WORKFLOW_SPECULAR) {
        // We can't support this until we implement Schlick fresnel so do nothing for now.
        // Phase 2: implement Schlick which takes specular color directly
    } else { // SPECULAR_WORKFLOW_METALNESS
        const float metallic = evalAttrMetallic(me, tls, state);
        if (isZero(metallic)) {
            // If "metallic" is zero then use dielectric
            if (isZero(roughness)) {
                addMirrorDielectricLobe(bsdfBuilder, N, ior, weight, opacity);
            } else {
                addMicrofacetDielectricLobe(bsdfBuilder, N, ior, roughness, weight, opacity);
            }
        } else if (isOne(metallic)) {
            // If "metallic" is one then use artist friendly conductor
            if (isZero(roughness)) {
                addMirrorConductorLobe(bsdfBuilder, N, albedo, weight);
            } else {
                addMicrofacetConductorLobe(bsdfBuilder, N, albedo, roughness, weight);
            }
            return; // Exit early for fully metallic surfaces
        } else {
            // If "metallic" is between zero and one then blend dielectric
            // and conductor based with "metallic" as the weight
            if (isZero(roughness)) {
                addMirrorConductorLobe(bsdfBuilder, N, albedo, metallic);
                addMirrorDielectricLobe(bsdfBuilder, N, ior, weight, opacity);
            } else {
                addMicrofacetConductorLobe(bsdfBuilder, N, albedo, roughness, metallic);
                addMicrofacetDielectricLobe(bsdfBuilder, N, ior, roughness, weight, opacity);
            }
        }
    }

    // Diffuse
    if (!isBlack(albedo)) {
        LambertianBRDF diffuseRefl;
        LambertianBRDF_init(diffuseRefl, N, albedo);
        BsdfBuilder_addLambertianBRDF(bsdfBuilder, diffuseRefl,
                                      1.0f, // weight
                                      BSDFBUILDER_PHYSICAL,
                                      aovDiffuse);
    }
}
DEFINE_MATERIAL_SHADER(UsdPreviewSurface, shade)

