// Copyright 2023 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file BsdfToon.ispc

#include "BsdfToon.isph"
#include <moonray/rendering/shading/ispc/bsdf/BsdfSlice.isph>
#include <moonray/rendering/shading/ispc/Util.isph>
#include <moonray/rendering/shading/ispc/PbrValidity.isph>

#include <scene_rdl2/common/platform/IspcUtil.isph>

varying Color
ToonBsdfLobe_eval(const varying BsdfLobe * uniform lobe,
                  const varying BsdfSlice &slice,
                  const Vec3f &wi,
                  varying float * uniform pdf)
{
    const varying ToonBsdfLobe * uniform toon =
        (const varying ToonBsdfLobe * uniform) lobe;
    Vec3f N = getN(toon->mFrame);
    if (toon->mExtendRamp && dot(N, wi) < 0) {
        // Make shading normal perpendicular to light direction
        // and nudge towards light with lerp
        N = normalize(lerp(cross(wi, cross(N, wi)),
                           wi,
                           sEpsilon));  // Must be > 1e-9
    }

    const float cosThetaWi = max(dot(N, wi), 0.0f);

    if (pdf != NULL) {
        *pdf = cosThetaWi * sOneOverPi;
    }

    // Note: we assume this lobe has been setup with a OneMinus*Fresnel
    // as we want to use 1 - specular_fresnel. Also notice we use
    // cosThetaWo to evaluate the fresnel term, as an approximation of what
    // hDotWi would be for the specular lobe.
    float cosThetaWo = 1.0f;

    if (isActive(BsdfLobe_getFresnel(lobe))) {
        cosThetaWo = max(dot(N, slice.mWo), 0.0f);
    }

    const Color ramp = ColorRampControl_eval1D(1.0f - cosThetaWi, &toon->mRampControl);

    const Color albedo = clamp(toon->mAlbedo * ramp, 0.0f, 1.0f);
    // Soften hard shadow terminator due to shading normals
    const float Gs = BsdfSlice_computeShadowTerminatorFix(&slice, N, wi);

    return Gs * albedo *
            BsdfLobe_computeScaleAndFresnel(lobe, cosThetaWo) * sOneOverTwoPi;
}

varying Color
ToonBsdfLobe_sample(const varying BsdfLobe * uniform lobe,
                    const varying BsdfSlice &slice,
                    varying float r1, varying float r2,
                    varying Vec3f &wi, varying float &pdf)
{
    const varying ToonBsdfLobe * uniform toon =
        (const varying ToonBsdfLobe * uniform) lobe;

    wi = localToGlobal(toon->mFrame, sampleLocalHemisphereCosine(r1, r2));
    return ToonBsdfLobe_eval(lobe, slice, wi, &pdf);
}

#pragma ignore warning(all)
ISPC_UTIL_EXPORT_STRUCT_TO_HEADER(ToonBsdfLobe);

//
// HairToonSpecularBsdfLobe
//
varying Color
HairToonSpecularBsdfLobe_sample(const varying BsdfLobe * uniform lobe,
                                const varying BsdfSlice &slice,
                                varying float r1, varying float r2,
                                varying Vec3f &wi, varying float &pdf)
{
    const varying HairToonSpecularBsdfLobe * uniform hairToonSpecLobe = (const varying HairToonSpecularBsdfLobe * uniform)lobe;
    if (hairToonSpecLobe->mIndirectHairLobe != NULL) {
        const varying BsdfLobe * uniform indirectHairLobe = (const varying BsdfLobe * uniform)hairToonSpecLobe->mIndirectHairLobe;
        return HairBsdfLobe_sample(indirectHairLobe, slice, r1, r2, wi, pdf) *
               hairToonSpecLobe->mIndirectReflectionsIntensity;
    } else {
        pdf = 0.0f;
        return sBlack;
    }
}

varying Color
HairToonSpecularBsdfLobe_eval(const varying BsdfLobe * uniform lobe,
                              const varying BsdfSlice &slice,
                              const Vec3f &wi,
                              varying float * uniform pdf)
{
    const varying HairToonSpecularBsdfLobe * uniform hairToonSpecLobe =
        (const varying HairToonSpecularBsdfLobe * uniform) lobe;
    const varying BsdfLobe * uniform hairLobe = (const varying BsdfLobe * uniform)hairToonSpecLobe->mDirectHairLobe;
    const Color hairEval = HairBsdfLobe_eval(hairLobe, slice, wi, pdf);
    const float ramp = FloatRampControl_eval1D(luminance(hairEval), &hairToonSpecLobe->mRampControl);
    return hairToonSpecLobe->mTint * ramp * hairToonSpecLobe->mIntensity * sOneOverPi;
}

varying Color
HairToonSpecularBsdfLobe_albedo(const varying BsdfLobe * uniform lobe,
                            const varying BsdfSlice &slice)
{
    const varying HairToonSpecularBsdfLobe * uniform ct =
        (const varying HairToonSpecularBsdfLobe * uniform) lobe;

    // This approximation is the same a cook torrance
    float cosThetaWo = max(dot(getN(ct->mFrame), slice.mWo), 0.0f);
    // TODO: Improve this approximation!
    return BsdfLobe_computeScaleAndFresnel(lobe, cosThetaWo);
}


void
HairToonSpecularBsdfLobe_differentials(const varying BsdfLobe * uniform lobe,
                                   const varying Vec3f &wo,
                                   const varying Vec3f &wi,
                                   varying float r1, varying float r2,
                                   const varying Vec3f &dNdx,
                                   const varying Vec3f &dNdy,
                                   varying Vec3f &dDdx, varying Vec3f &dDdy)
{
    // This function is only called if indirect reflections are enabled.
    // If that's the case we call the internal hair lobe's differentials
    // function.
    const varying HairToonSpecularBsdfLobe * uniform hairToonSpecLobe = (const varying HairToonSpecularBsdfLobe * uniform)lobe;
    if (hairToonSpecLobe->mIndirectHairLobe != NULL) {
        const varying BsdfLobe * uniform indirectHairLobe = (const varying BsdfLobe * uniform)hairToonSpecLobe->mIndirectHairLobe;
        HairBsdfLobe_differentials(indirectHairLobe, wo, wi, r1, r2, dNdx, dNdy, dDdx, dDdy);
    }
}


uniform bool
HairToonSpecularBsdfLobe_getProperty(const varying BsdfLobe * uniform lobe,
                                 uniform BsdfLobeProperty property,
                                 varying float * uniform dest)
{
    const varying HairToonSpecularBsdfLobe * uniform ct =
        (const varying HairToonSpecularBsdfLobe * uniform) lobe;

    uniform bool handled = true;

    switch (property)
    {
    default:
        handled = false;
        break;
    }

    return handled;
}

void
HairToonSpecularBsdfLobe_show(const varying BsdfLobe * const uniform lobe,
                          const uniform uint8_t ntab)
{
    const varying HairToonSpecularBsdfLobe * uniform me =
        (const varying HairToonSpecularBsdfLobe * uniform) lobe;

    const varying Vec3f N = getN(me->mFrame);
    const varying Fresnel * const uniform fresnel = me->mFresnel;

    BsdfUtil_indent(ntab);   print("[HairToonSpecularBsdfLobe] mask: %\n", lobe->mMask);

    BsdfUtil_indent(ntab+1); print("scale: % % %\n", BSDF_UTIL_EXTRACT(me->mScale.r),
                                                     BSDF_UTIL_EXTRACT(me->mScale.g),
                                                     BSDF_UTIL_EXTRACT(me->mScale.b));
    BsdfUtil_indent(ntab+1); print("N: % % %\n", BSDF_UTIL_EXTRACT(N.x),
                                                 BSDF_UTIL_EXTRACT(N.y),
                                                 BSDF_UTIL_EXTRACT(N.z));
    BsdfUtil_indent(ntab+1); print("intensity: %\n", BSDF_UTIL_EXTRACT(me->mIntensity));
    BsdfUtil_indent(ntab+1); print("tint: % % %\n", BSDF_UTIL_EXTRACT(me->mTint.r),
                                                    BSDF_UTIL_EXTRACT(me->mTint.g),
                                                    BSDF_UTIL_EXTRACT(me->mTint.b));
    if (fresnel) {
        Fresnel_show(fresnel, ntab+1);
    }
    if (me->mDirectHairLobe) {
        HairRBsdfLobe_show((const varying BsdfLobe * uniform)me->mDirectHairLobe, ntab+1);
    }
    if (me->mIndirectHairLobe) {
        HairRBsdfLobe_show((const varying BsdfLobe * uniform)me->mIndirectHairLobe, ntab+1);
    }
}

//
// ToonSpecularBsdfLobe
//
varying Color
ToonSpecularBsdfLobe_sample(const varying BsdfLobe * uniform lobe,
                            const varying BsdfSlice &slice,
                            varying float r1, varying float r2,
                            varying Vec3f &wi, varying float &pdf)
{
    const varying ToonSpecularBsdfLobe * uniform toonSpecLobe = (const varying ToonSpecularBsdfLobe * uniform)lobe;
    if (toonSpecLobe->mIndirectLobe != NULL) {
        const varying BsdfLobe * uniform indirectLobe = (const varying BsdfLobe * uniform)toonSpecLobe->mIndirectLobe;
        return GGXCookTorranceBsdfLobe_sample(indirectLobe, slice, r1, r2, wi, pdf) *
               toonSpecLobe->mIndirectReflectionsIntensity;
    } else {
        pdf = 0.0f;
        return sBlack;
    }
}

// Rodrigues' rotation formula, assume axis is normalized
static Vec3f
rotateVector(const Vec3f& v, const Vec3f& axis, float theta) {
    float ct, st;
    sincos(theta, &st, &ct);
    return ct * v + st * cross(axis, v) + dot(axis, v) * (1.f - ct) * axis;
}

varying Color
ToonSpecularBsdfLobe_eval(const varying BsdfLobe * uniform lobe,
                          const varying BsdfSlice &slice,
                          const Vec3f &wi,
                          varying float * uniform pdf)
{
    const varying ToonSpecularBsdfLobe * uniform toonSpecLobe =
        (const varying ToonSpecularBsdfLobe * uniform) lobe;

    Vec3f N = getN(toonSpecLobe->mFrame);
    const Vec3f wo = slice.mWo;
    Vec3f R = wi - 2.0f * dot(wi, N) * N;

    // Rotate N to "stretch" the specular highlight
    const float dot_u_l = dot(R, toonSpecLobe->mdPds);
    const float dot_u_c = dot(wo, toonSpecLobe->mdPds);
    const float dot_u = dot_u_l + dot_u_c;
    const float rot_u = clamp(toonSpecLobe->mStretchU * dot_u, -0.5f, 0.5f);
    N = rotateVector(N, toonSpecLobe->mdPdt, rot_u);

    const float dot_v_l = dot(R, toonSpecLobe->mdPdt);
    const float dot_v_c = dot(wo, toonSpecLobe->mdPdt);
    const float dot_v = dot_v_l + dot_v_c;
    const float rot_v = clamp(-toonSpecLobe->mStretchV * dot_v, -0.5f, 0.5f);
    N = rotateVector(N, toonSpecLobe->mdPds, rot_v);

    const float cosNO = dot(N, wo);
    const float cosNI = dot(N, wi);

    // Prepare for early exit
    if (pdf != NULL) {
        *pdf = 0.0f;
    }

    if (cosNO <= 0.0f || cosNI <= 0.0f) {
        return sBlack;
    }

    // https://docs.google.com/presentation/d/10lYRt3FpQEOu8vrCoGdc_ngJnRXkzFi2B69cEUFkM48/edit
    // Uses acos to get a more linear response for easier ramp remapping
    R = wi - 2.0f * dot(wi, N) * N;
    float specAngle = max(0.0f, dot(wo * -1.0f, R));
    specAngle = 1.0f - 2.0f * (acos(clamp(specAngle, -1.0f, 1.0f)) / sPi);
    const float exponent = (1.0f - toonSpecLobe->mRoughness) * 100.0f;
    specAngle = pow(specAngle, exponent);

    if (specAngle <= 0.0f) {
        return sBlack;
    }

    if (pdf != NULL) {
        *pdf = 0.5f * sOneOverPi * specAngle;
    }
    
    const float ramp = FloatRampControl_eval1D(specAngle, &toonSpecLobe->mRampControl);
    return toonSpecLobe->mTint * ramp * toonSpecLobe->mIntensity * sOneOverPi;
}

varying Color
ToonSpecularBsdfLobe_albedo(const varying BsdfLobe * uniform lobe,
                            const varying BsdfSlice &slice)
{
    const varying ToonSpecularBsdfLobe * uniform ct =
        (const varying ToonSpecularBsdfLobe * uniform) lobe;

    // This approximation is the same a cook torrance
    float cosThetaWo = max(dot(getN(ct->mFrame), slice.mWo), 0.0f);
    // TODO: Improve this approximation!
    return BsdfLobe_computeScaleAndFresnel(lobe, cosThetaWo);
}


void
ToonSpecularBsdfLobe_differentials(const varying BsdfLobe * uniform lobe,
                                   const varying Vec3f &wo,
                                   const varying Vec3f &wi,
                                   varying float r1, varying float r2,
                                   const varying Vec3f &dNdx,
                                   const varying Vec3f &dNdy,
                                   varying Vec3f &dDdx, varying Vec3f &dDdy)
{
    // This function is only called if indirect reflections are enabled.
    // If that's the case we call the internal cook torrance lobe's
    // differentials function.
    const varying ToonSpecularBsdfLobe * uniform toonSpecLobe = (const varying ToonSpecularBsdfLobe * uniform)lobe;
    if (toonSpecLobe->mIndirectLobe != NULL) {
        const varying BsdfLobe * uniform indirectLobe = (const varying BsdfLobe * uniform)toonSpecLobe->mIndirectLobe;
        CookTorranceBsdfLobe_differentials(indirectLobe, wo, wi, r1, r2, dNdx, dNdy, dDdx, dDdy);
    }
}


uniform bool
ToonSpecularBsdfLobe_getProperty(const varying BsdfLobe * uniform lobe,
                                 uniform BsdfLobeProperty property,
                                 varying float * uniform dest)
{
    const varying ToonSpecularBsdfLobe * uniform ct =
        (const varying ToonSpecularBsdfLobe * uniform) lobe;

    uniform bool handled = true;

    switch (property)
    {
    case BSDF_LOBE_PROPERTY_ROUGHNESS:
        *dest =  ct->mRoughness;
        *(dest + 1) =  ct->mRoughness;
        break;
    case BSDF_LOBE_PROPERTY_NORMAL:
        {
            const Vec3f N = getN(ct->mFrame);
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

void
ToonSpecularBsdfLobe_show(const varying BsdfLobe * const uniform lobe,
                          const uniform uint8_t ntab)
{
    const varying ToonSpecularBsdfLobe * uniform me =
        (const varying ToonSpecularBsdfLobe * uniform) lobe;

    const varying Vec3f N = getN(me->mFrame);
    const varying Fresnel * const uniform fresnel = me->mFresnel;

    BsdfUtil_indent(ntab);   print("[ToonSpecularBsdfLobe] mask: %\n", lobe->mMask);

    BsdfUtil_indent(ntab+1); print("scale: % % %\n", BSDF_UTIL_EXTRACT(me->mScale.r),
                                                     BSDF_UTIL_EXTRACT(me->mScale.g),
                                                     BSDF_UTIL_EXTRACT(me->mScale.b));
    BsdfUtil_indent(ntab+1); print("N: % % %\n", BSDF_UTIL_EXTRACT(N.x),
                                                 BSDF_UTIL_EXTRACT(N.y),
                                                 BSDF_UTIL_EXTRACT(N.z));
    BsdfUtil_indent(ntab+1); print("intensity: %\n", BSDF_UTIL_EXTRACT(me->mIntensity));
    BsdfUtil_indent(ntab+1); print("roughness: %\n", BSDF_UTIL_EXTRACT(me->mRoughness));
    BsdfUtil_indent(ntab+1); print("tint: % % %\n", BSDF_UTIL_EXTRACT(me->mTint.r),
                                                    BSDF_UTIL_EXTRACT(me->mTint.g),
                                                    BSDF_UTIL_EXTRACT(me->mTint.b));
    BsdfUtil_indent(ntab+1); print("stretch u: %\n", BSDF_UTIL_EXTRACT(me->mStretchU));
    BsdfUtil_indent(ntab+1); print("stretch v: %\n", BSDF_UTIL_EXTRACT(me->mStretchV));
    BsdfUtil_indent(ntab+1); print("dPds: % % %\n", BSDF_UTIL_EXTRACT(me->mdPds.x),
                                                    BSDF_UTIL_EXTRACT(me->mdPds.y),
                                                    BSDF_UTIL_EXTRACT(me->mdPds.z));
    BsdfUtil_indent(ntab+1); print("dPdt: % % %\n", BSDF_UTIL_EXTRACT(me->mdPdt.x),
                                                    BSDF_UTIL_EXTRACT(me->mdPdt.y),
                                                    BSDF_UTIL_EXTRACT(me->mdPdt.z));

    if (fresnel) {
        Fresnel_show(fresnel, ntab+1);
    }
    if (me->mIndirectLobe) {
        CookTorranceBsdfLobe_show((const varying BsdfLobe * uniform)me->mIndirectLobe, ntab+1);
    }
}

