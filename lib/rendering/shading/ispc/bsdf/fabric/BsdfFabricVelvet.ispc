// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file BsdfFabricVelvet.ispc

#include "BsdfFabricVelvet.isph"
#include "VelvetAlbedo.isph"
#include <moonray/rendering/shading/ispc/bsdf/BsdfUtil.isph>
#include <moonray/rendering/shading/ispc/ShaderMacros.isph>
#include <moonray/rendering/shading/ispc/Util.isph>

ISPC_UTIL_EXPORT_ENUM_TO_HEADER(VelvetConstants);

// Precomputed Normalization Integrals for the fuzz model -
// pow(sin(N, H), e) * cosThetaWi
// Roughness from 0.05 to 1.0f
uniform const float
BsdfFabricVelvet_precomputedNormalizationTable[VELVET_NORMALIZATION_TABLE_SIZE] = {
                                                                                   0.20789f, 0.43665f, 0.62931f, 0.78663f, 0.91573f,
                                                                                   1.02585f, 1.10919f, 1.18676f, 1.25177f, 1.30365f,
                                                                                   1.35732f, 1.39961f, 1.43324f, 1.48466f, 1.50755f,
                                                                                   1.53596f, 1.56253f, 1.58541f, 1.60406f, 1.62693f};

export void* uniform
BsdfFabricVelvet_normalizationTable() { return BsdfFabricVelvet_precomputedNormalizationTable; }

varying Color
FabricVelvetBsdfLobe_eval(const varying BsdfLobe * uniform lobe,
                          const varying BsdfSlice &slice,
                          const Vec3f &wi,
                          varying float * uniform pdf)
{
    const varying FabricVelvetBsdfLobe * uniform fab =
        (const varying FabricVelvetBsdfLobe * uniform) lobe;

    if (pdf != NULL) {
        *pdf = 0.0f;
    }

    const Vec3f N = getN(fab->mFrame);

    float cosNO = dot(N, slice.mWo);
    if (cosNO <= sEpsilon) return sBlack;

    float cosNI = dot(N, wi);
    if (cosNI <= sEpsilon) return sBlack;

    Vec3f H;
    if (!computeNormalizedHalfVector(slice.mWo, wi, H)) {
        return sBlack;
    }

    // This is the Velvet BRDF for grazing angle highlights
    // BRDF = sin(N, H)^exponent
    const float cosNH = clamp(dot(N, H), -0.99f, 0.99f);
    const float sinSquared = (1.0f - cosNH*cosNH);
    const float sinNDotH   = sqrt(sinSquared);

    float velvetFactor = fab->mNormalizationFactor * pow(sinNDotH, fab->mSpecularExponent);

    // MOONSHINE-921
    // Add an extra cosineThetaWi term to soften the shadow terminator
    // Note - this breaks bidirectionality but artists prefer this look.
    velvetFactor = velvetFactor * cosNI;

    // Cosine hemi-spherical sampling
    if (pdf != NULL) {
        *pdf = cosNI * sOneOverPi;
    }

    Color f = BsdfLobe_computeScaleAndFresnel(lobe, cosNO) *
              velvetFactor * fab->mColor *
              (slice.mIncludeCosineTerm  ?  cosNI  :  1.0f);

    return f;
}

varying Color
FabricVelvetBsdfLobe_sample(const varying BsdfLobe * uniform lobe,
                            const varying BsdfSlice &slice,
                            varying float r1, varying float r2,
                            varying Vec3f &wi, varying float &pdf)
{
    const varying FabricVelvetBsdfLobe * uniform fab =
        (const varying FabricVelvetBsdfLobe * uniform) lobe;
    wi = localToGlobal(fab->mFrame, sampleLocalHemisphereCosine(r1, r2));
    return FabricVelvetBsdfLobe_eval(lobe, slice, wi, &pdf);
}

varying Color
FabricVelvetBsdfLobe_albedo(const varying BsdfLobe * uniform lobe,
                            const varying BsdfSlice &slice)
{
    const varying FabricVelvetBsdfLobe * uniform fab =
        (const varying FabricVelvetBsdfLobe * uniform) lobe;

    float cosThetaWo = max(dot(getN(fab->mFrame), slice.mWo), 0.0f);
    return  BsdfLobe_computeScaleAndFresnel(lobe, cosThetaWo) * fab->mColor *
            VelvetAlbedo_at(cosThetaWo, fab->mRoughness);
}

void
FabricVelvetBsdfLobe_differentials(const varying BsdfLobe * uniform lobe,
                                   const varying Vec3f &wo,
                                   const varying Vec3f &wi,
                                   varying float r1, varying float r2,
                                   const varying Vec3f &dNdx,
                                   const varying Vec3f &dNdy,
                                   varying Vec3f &dDdx, varying Vec3f &dDdy)
{
    const varying FabricVelvetBsdfLobe * uniform fab =
        (const varying FabricVelvetBsdfLobe * uniform) lobe;

    // It's complex to capture the full derivative. Instead we use the
    // derivative of a mirror reflection about the H vector, and scale the
    // length of the directional derivative proportionally with roughness.
    Vec3f H;
    if (!computeNormalizedHalfVector(wo, wi, H)) {
        H = getN(fab->mFrame);
    }
    computeReflectionDirectionDifferentialNoCurvature(wo, wi, H, dDdx, dDdy);
    // TODO calculate the right factors for fabric
    // These are taken from the hair lobe
    dDdx = dDdx * fab->mdDFactor;
    dDdy = dDdy * fab->mdDFactor;
}

uniform bool
FabricVelvetBsdfLobe_getProperty(const varying BsdfLobe * uniform lobe,
                                 uniform BsdfLobeProperty property,
                                 varying float * uniform dest)
{
    const varying FabricVelvetBsdfLobe * uniform fab =
        (const varying FabricVelvetBsdfLobe * uniform) lobe;

    uniform bool handled = true;

    switch (property)
    {
    case BSDF_LOBE_PROPERTY_COLOR:
    {
        *dest       = fab->mColor.r;
        *(dest+1)   = fab->mColor.g;
        *(dest+2)   = fab->mColor.b;
        break;
    }
    case BSDF_LOBE_PROPERTY_ROUGHNESS:
    {
        *dest       = fab->mRoughness;
        *(dest+1)   = fab->mRoughness;
        break;
    }
    case BSDF_LOBE_PROPERTY_NORMAL:
        {
            const Vec3f N = getN(fab->mFrame);
            *dest       = N.x;
            *(dest + 1) = N.y;
            *(dest + 2) = N.z;
        }
        break;
    default:
        handled = false;
        break;
    }

    return handled;
}

inline void
FabricVelvetBsdfLobe_show(const varying BsdfLobe * const uniform lobe,
                          const uniform uint8_t ntab)
{
    const varying FabricVelvetBsdfLobe * uniform me =
        (const varying FabricVelvetBsdfLobe * uniform) lobe;

    const varying Fresnel * const uniform fresnel = me->mFresnel;

    BsdfUtil_indent(ntab);   print("[FabricVelvetBsdfLobe] mask: %\n", lobe->mMask);

    BsdfUtil_indent(ntab+1); print("scale: % % %\n", BSDF_UTIL_EXTRACT(me->mScale.r),
                                                     BSDF_UTIL_EXTRACT(me->mScale.g),
                                                     BSDF_UTIL_EXTRACT(me->mScale.b));
    BsdfUtil_indent(ntab+1); print("color: % % %\n", BSDF_UTIL_EXTRACT(me->mColor.r),
                                                     BSDF_UTIL_EXTRACT(me->mColor.g),
                                                     BSDF_UTIL_EXTRACT(me->mColor.b));
    BsdfUtil_indent(ntab+1); print("roughness: %\n", BSDF_UTIL_EXTRACT(me->mRoughness));
    BsdfUtil_indent(ntab+1); print("specular exponent: %\n", BSDF_UTIL_EXTRACT(me->mSpecularExponent));

    if (fresnel) {
        Fresnel_show(fresnel, ntab+1);
    }
}

void
FabricVelvetBsdfLobe_init(varying FabricVelvetBsdfLobe * uniform lobe,
                          const varying Vec3f &normal,
                          const varying float roughness,
                          const varying Color& color)
{
    BsdfLobe_init((varying BsdfLobe * uniform) lobe,
                  /* name = */ BSDF_LOBE_FABRIC_VELVET,
                  /* type = */ BSDF_LOBE_TYPE_REFLECTION | BSDF_LOBE_TYPE_GLOSSY,
                  /* differentialFlags = */ BSDF_LOBE_IGNORES_INCOMING_DIFFERENTIALS,
                  /* isSpherical = */ false,
                  /* propertyFlags = */ BSDF_LOBE_PROPERTY_COLOR | BSDF_LOBE_PROPERTY_NORMAL | BSDF_LOBE_PROPERTY_ROUGHNESS,
                  /* evalFn = */ FabricVelvetBsdfLobe_eval,
                  /* sampleFn = */ FabricVelvetBsdfLobe_sample,
                  /* albedoFn = */ FabricVelvetBsdfLobe_albedo,
                  /* differentialsFn = */ FabricVelvetBsdfLobe_differentials,
                  /* getPropertyFn = */ FabricVelvetBsdfLobe_getProperty,
                  /* showFn = */ FabricVelvetBsdfLobe_show);

    lobe->mRoughness = clamp(roughness, 0.05f, 1.0f);
    lobe->mColor = color;
    // Init FabricVelvetBsdfLobe components
    ReferenceFrame_init(lobe->mFrame, normal);

    lobe->mSpecularExponent = 1.0f / lobe->mRoughness;
    const int tableIndex = (int)floor(lobe->mRoughness*VELVET_NORMALIZATION_TABLE_SIZE - 1);
    lobe->mNormalizationFactor = 1.0f / BsdfFabricVelvet_precomputedNormalizationTable[tableIndex];

    // Derive a directional differential scale that varies according to width
    // TODO Figure out if the hair shader defaults work well or we need new ones for fabric
    // Convert specular exponent to roughness
    // Walter et al, EGSR 2007, "Microfacet Models for Refraction through Rough Surfaces"
    // Derive a directional differential scale that varies according to width
    lobe->mdDFactor = FabricV_sdDFactorMin + lobe->mRoughness * FabricV_sdDFactorSlope;
}

