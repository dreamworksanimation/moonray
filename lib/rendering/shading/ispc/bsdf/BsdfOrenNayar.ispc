// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file BsdfOrenNayar.ispc

#include "BsdfOrenNayar.isph"
#include "BsdfSlice.isph"
#include <moonray/rendering/shading/ispc/Util.isph>
#include <moonray/rendering/shading/ispc/PbrValidity.isph>

#include <scene_rdl2/common/platform/IspcUtil.isph>

// See Substance for Unity: Chapter 01-01 Understanding PBR
// https://youtu.be/ueC2qGzWrgQ?t=914
#define OREN_NAYAR_PBR_VALIDITY_MIN 0.031896
#define OREN_NAYAR_PBR_VALIDITY_MAX 0.871367

varying float
getOrenNayarIntensity(const varying Vec3f& n,
                      const varying Vec3f& v,
                      const varying Vec3f& l,
                      const varying float a,
                      const varying float b)
{
    // We are using the Fujii implementation or Oren-Nayar.
    // The PBRT implementation was also tested but resulted
    // in more energy loss.
    // https://mimosa-pudica.net/improved-oren-nayar.html
    const float nl = max(dot(n, l), 0.0f);
    const float nv = max(dot(n, v), 0.0f);
    float s = dot(l, v) - nl * nv;

    float t = 1.0f;
    if (s > 0.0f) {
        t = max(nl, nv) + sEpsilon;
    }

    return nl * (a + b * s/t);
}

varying Color
OrenNayarBsdfLobe_eval(const varying BsdfLobe * uniform lobe,
                       const varying BsdfSlice &slice,
                       const Vec3f &wi,
                       varying float * uniform pdf)
{

    const varying OrenNayarBsdfLobe * uniform orennayar =
        (const varying OrenNayarBsdfLobe * uniform) lobe;

    const float cosThetaWi = getOrenNayarIntensity(getN(orennayar->mFrame),
                                                   slice.mWo, wi,
                                                   orennayar->mA, orennayar->mB);
    if (pdf != NULL) {
        *pdf = cosThetaWi * sOneOverPi;
    }

    // Note: we assume this lobe has been setup with a OneMinus*Fresnel
    // as we want to use 1 - specular_fresnel. Also notice we use
    // cosThetaWo to evaluate the fresnel term, as an approximation of what
    // hDotWi would be for the specular lobe.
    float cosThetaWo = 1.0f;
    if (isActive(BsdfLobe_getFresnel(lobe))) {
        const Vec3f N = (BsdfLobe_matchesFlag(lobe, BSDF_LOBE_TYPE_REFLECTION)) ?
                        getN(orennayar->mFrame) :
                        neg(getN(orennayar->mFrame));
        cosThetaWo = max(dot(N, slice.mWo), 0.0f);
    }

    // Soften hard shadow terminator due to shading normals
    const float Gs = (BsdfLobe_matchesFlag(lobe, BSDF_LOBE_TYPE_REFLECTION)) ?
                     BsdfSlice_computeShadowTerminatorFix(&slice, getN(orennayar->mFrame), wi) :
                     1.0f;

    return orennayar->mAlbedo * sOneOverPi * Gs *
           BsdfLobe_computeScaleAndFresnel(lobe, cosThetaWo) *
           (slice.mIncludeCosineTerm ? cosThetaWi : 1.0f);
}

varying Color
OrenNayarBsdfLobe_sample(const varying BsdfLobe * uniform lobe,
                         const varying BsdfSlice &slice,
                         varying float r1, varying float r2,
                         varying Vec3f &wi, varying float &pdf)
{
    const varying OrenNayarBsdfLobe * uniform orennayar =
        (const varying OrenNayarBsdfLobe * uniform) lobe;

    wi = localToGlobal(orennayar->mFrame, sampleLocalHemisphereCosine(r1, r2));
    return OrenNayarBsdfLobe_eval(lobe, slice, wi, &pdf);
}

varying Color
OrenNayarBsdfLobe_albedo(const varying BsdfLobe * uniform lobe,
                         const varying BsdfSlice &slice)
{
    const varying OrenNayarBsdfLobe * uniform orennayar =
        (const varying OrenNayarBsdfLobe * uniform) lobe;

    float cosThetaWo = max(dot(getN(orennayar->mFrame), slice.mWo), 0.0f);
    return BsdfLobe_computeScaleAndFresnel(lobe, cosThetaWo) * orennayar->mAlbedo;
}

void
OrenNayarBsdfLobe_differentials(const varying BsdfLobe * uniform lobe,
                                const varying Vec3f &wo,
                                const varying Vec3f &wi,
                                varying float r1, varying float r2,
                                const varying Vec3f &dNdx,
                                const varying Vec3f &dNdy,
                                varying Vec3f &dDdx, varying Vec3f &dDdy)
{
    // The hemisphere cosine sampling direction derivative seems like a very
    // good approximation to the full derivative for diffuse sampling. This
    // is why we ignore the input differentials.

    // TODO: How can we avoid computing these twice (once here, once in
    // sample()->sampleLocalHemisphereCosine()) ?
    localHemisphereCosineDifferentials(r1, r2, dDdx, dDdy);

    // The differentials form rectangles which get long and thin close
    // to the hemisphere pole and equator. We prefer having ray
    // differentials that form a square, but preserve the ray footprint.
    squarifyRectangle(dDdx, dDdy);

    const varying OrenNayarBsdfLobe * uniform orennayar =
        (const varying OrenNayarBsdfLobe * uniform) lobe;

    dDdx = localToGlobal(orennayar->mFrame, dDdx);
    dDdy = localToGlobal(orennayar->mFrame, dDdy);
}

uniform bool
OrenNayarBsdfLobe_getProperty(const varying BsdfLobe * uniform lobe,
                              uniform BsdfLobeProperty property,
                              varying float * uniform dest)
{
    const varying OrenNayarBsdfLobe * uniform orennayar =
        (const varying OrenNayarBsdfLobe * uniform) lobe;

    uniform bool handled = true;

    switch (property)
    {
    case BSDF_LOBE_PROPERTY_ROUGHNESS:
        *dest       = 1.0f;
        *(dest + 1) = 1.0f;
        break;
    case BSDF_LOBE_PROPERTY_NORMAL:
        {
            const Vec3f N = getN(orennayar->mFrame);
            *dest       = N.x;
            *(dest + 1) = N.y;
            *(dest + 2) = N.z;
        }
        break;
    case BSDF_LOBE_PROPERTY_PBR_VALIDITY:
        {
            Color res = computeAlbedoPbrValidity(orennayar->mAlbedo);
            *dest       = res.r;
            *(dest + 1) = res.g;
            *(dest + 2) = res.b;
        }
        break;
    case BSDF_LOBE_PROPERTY_COLOR:
        {
            varying Color c = orennayar->mAlbedo * BsdfLobe_computeScaleAndFresnel(lobe, 1.f);
            *dest       = c.r;
            *(dest + 1) = c.g;
            *(dest + 2) = c.b;
        }
        break;
    default:
        handled = false;
        break;
    }

    return handled;
}

void
OrenNayarBsdfLobe_show(const varying BsdfLobe * const uniform lobe,
                       const uniform uint8_t ntab)
{
    const varying OrenNayarBsdfLobe * uniform me =
        (const varying OrenNayarBsdfLobe * uniform) lobe;

    const varying Vec3f N = getN(me->mFrame);
    const varying Fresnel * const uniform fresnel = me->mFresnel;

    if (BsdfLobe_matchesFlag(lobe, BSDF_LOBE_TYPE_REFLECTION)) {
        BsdfUtil_indent(ntab);   print("[OrenNayarBsdfLobe] (reflection) mask: %\n", lobe->mMask);
    } else {
        BsdfUtil_indent(ntab);   print("[OrenNayarBsdfLobe] (transmission) mask: %\n", lobe->mMask);
    }

    BsdfUtil_indent(ntab+1); print("scale: % % %\n", BSDF_UTIL_EXTRACT(me->mScale.r),
                                                     BSDF_UTIL_EXTRACT(me->mScale.g),
                                                     BSDF_UTIL_EXTRACT(me->mScale.b));
    BsdfUtil_indent(ntab+1); print("N: % % %\n", BSDF_UTIL_EXTRACT(N.x),
                                                 BSDF_UTIL_EXTRACT(N.y),
                                                 BSDF_UTIL_EXTRACT(N.z));
    BsdfUtil_indent(ntab+1); print("albedo: % % %\n", BSDF_UTIL_EXTRACT(me->mAlbedo.r),
                                                      BSDF_UTIL_EXTRACT(me->mAlbedo.g),
                                                      BSDF_UTIL_EXTRACT(me->mAlbedo.b));
    BsdfUtil_indent(ntab+1); print("diffuse_roughness: %\n", BSDF_UTIL_EXTRACT(me->mRoughness));
    if (fresnel) {
        Fresnel_show(fresnel, ntab+1);
    }
}

/// @brief intialize a oren nayar lobe
void
OrenNayarBsdfLobe_init(varying OrenNayarBsdfLobe * uniform lobe,
                       const varying Vec3f &normal,
                       const varying Color &albedo,
                       const varying float roughness,
                       const uniform bool reflection)
{
    BsdfLobe_init((varying BsdfLobe * uniform) lobe,
                  /* name = */ BSDF_LOBE_OREN_NAYAR,
                  /* type = */
                  reflection?
                  BSDF_LOBE_TYPE_REFLECTION   | BSDF_LOBE_TYPE_DIFFUSE :
                  BSDF_LOBE_TYPE_TRANSMISSION | BSDF_LOBE_TYPE_DIFFUSE,
                  /* differentialFlags = */ BSDF_LOBE_IGNORES_INCOMING_DIFFERENTIALS,
                  /* isSpherical = */ false,
                  /* propertyFlags = */ BSDF_LOBE_PROPERTY_NORMAL | BSDF_LOBE_PROPERTY_ROUGHNESS |
                                        BSDF_LOBE_PROPERTY_PBR_VALIDITY,
                  /* evalFn = */ OrenNayarBsdfLobe_eval,
                  /* sampleFn = */ OrenNayarBsdfLobe_sample,
                  /* albedoFn = */ OrenNayarBsdfLobe_albedo,
                  /* differentialsFn = */ OrenNayarBsdfLobe_differentials,
                  /* getPropertyFn = */ OrenNayarBsdfLobe_getProperty,
                  /* showFn = */ OrenNayarBsdfLobe_show);

    ReferenceFrame_init(lobe->mFrame, normal);
    lobe->mAlbedo = albedo;
    lobe->mRoughness = roughness;

    float s = deg2rad(roughness * 90.0f);
    // No sqr() function in ispc
    s = s * s;
    lobe->mA = 1.0f - (0.5f * (s / (s + 0.33f)));
    lobe->mB = 0.45f * (s / (s + 0.09f));
}

#pragma ignore warning(all)
ISPC_UTIL_EXPORT_STRUCT_TO_HEADER(OrenNayarBsdfLobe);

