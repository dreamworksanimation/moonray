// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file BsdfMirror.ispc

#include "BsdfMirror.isph"
#include "BsdfSlice.isph"

#include <scene_rdl2/common/platform/IspcUtil.isph>

//-----------------------------------------------------------------------------

//
// MirrorReflectionBsdfLobe
//

varying Color
MirrorReflectionBsdfLobe_eval(const varying BsdfLobe * uniform lobe,
                              const varying BsdfSlice &slice,
                              const Vec3f &wi,
                              varying float * uniform pdf)
{
    if (pdf != NULL) {
        *pdf = 0.0f;
    }
    return Color_ctor(0.0f, 0.0f, 0.0f);
}

varying Color
MirrorReflectionBsdfLobe_sample(const varying BsdfLobe * uniform lobe,
                                const varying BsdfSlice &slice,
                                varying float r1, varying float r2,
                                varying Vec3f &wi, varying float &pdf)
{
    const varying MirrorReflectionBsdfLobe * uniform mirror =
        (const varying MirrorReflectionBsdfLobe * uniform) lobe;

    // Reflect wo
    float cosTheta = computeReflectionDirection(mirror->mN, slice.mWo, wi);
    cosTheta = abs(cosTheta);

    MNRY_ASSERT(isNormalized(wi));

    // Probability of sampling the delta function is 1.
    pdf = 1.0f;

    // Compute contribution
    return BsdfLobe_computeScaleAndFresnel(lobe, cosTheta) *
            (slice.mIncludeCosineTerm  ?  1.0f  :  1.0f / cosTheta);
}

varying Color
MirrorReflectionBsdfLobe_albedo(const varying BsdfLobe * uniform lobe,
                                const varying BsdfSlice &slice)
{
    const varying MirrorReflectionBsdfLobe * uniform mirror =
        (const varying MirrorReflectionBsdfLobe * uniform) lobe;

    float cosThetaWo = dot(mirror->mN, slice.mWo);
    return BsdfLobe_computeScaleAndFresnel(lobe, abs(cosThetaWo));
}

void
MirrorReflectionBsdfLobe_differentials(const varying BsdfLobe * uniform lobe,
                                       const varying Vec3f &wo,
                                       const varying Vec3f &wi,
                                       varying float r1, varying float r2,
                                       const varying Vec3f &dNdx,
                                       const varying Vec3f &dNdy,
                                       varying Vec3f &dDdx, varying Vec3f &dDdy)
{
    const varying MirrorReflectionBsdfLobe * uniform mirror =
        (const varying MirrorReflectionBsdfLobe * uniform) lobe;

    computeReflectionDirectionDifferential(wo, wi, mirror->mN, dNdx, dNdy, dDdx, dDdy);
}

uniform bool
MirrorReflectionBsdfLobe_getProperty(const varying BsdfLobe * uniform lobe,
                                     uniform BsdfLobeProperty property,
                                     varying float * uniform dest)
{
    const varying MirrorReflectionBsdfLobe * uniform mirror =
        (const varying MirrorReflectionBsdfLobe * uniform) lobe;

    uniform bool handled = true;

    switch (property)
    {
    case BSDF_LOBE_PROPERTY_ROUGHNESS:
        *dest       = 0.0f;
        *(dest + 1) = 0.0f;
        break;
    case BSDF_LOBE_PROPERTY_NORMAL:
        *dest       = mirror->mN.x;
        *(dest + 1) = mirror->mN.y;
        *(dest + 2) = mirror->mN.z;
        break;
    default:
        handled = false;
        break;
    }

    return handled;
}

//-----------------------------------------------------------------------------

//
// MirrorTransmissionBsdfLobe
//
varying Color
MirrorTransmissionBsdfLobe_eval(const varying BsdfLobe * uniform lobe,
                                const varying BsdfSlice &slice,
                                const Vec3f &wi,
                                varying float * uniform pdf)
{
    if (pdf != NULL) {
        *pdf = 0.0f;
    }
    return Color_ctor(0.0f, 0.0f, 0.0f);
}

varying Color
MirrorTransmissionBsdfLobe_sample(const varying BsdfLobe * uniform lobe,
                                  const varying BsdfSlice &slice,
                                  varying float r1, varying float r2,
                                  varying Vec3f &wi, varying float &pdf)
{
    const varying MirrorTransmissionBsdfLobe * uniform mirror =
        (const varying MirrorTransmissionBsdfLobe * uniform) lobe;

    float neta;
    Color dispersionColor = sWhite;
    if (mirror->mAllowDispersion) {
        float sampledEta;
        sampleSpectralIOR(r1,
                          mirror->mEtaR,
                          mirror->mEtaG,
                          mirror->mEtaB,
                          sampledEta,
                          pdf,
                          dispersionColor);
        neta = mirror->mEtaI / sampledEta;
    } else {
        // Probability of sampling the delta function is 1.
        neta = mirror->mEtaI / mirror->mEtaT;
        pdf = 1.0f;
    }

    // Compute the transmission direction
    float cosThetaWo, cosThetaWi;
    if (!computeRefractionDirection(mirror->mN,
                                    slice.mWo,
                                    neta,
                                    wi,
                                    cosThetaWo,
                                    cosThetaWi)) {

        // Total internal reflection is handled by the reflection lobe
        // fresnel reflectance
        pdf = 0.0f;
        return Color_ctor(0.0f, 0.0f, 0.0f);
    }

    // Note: we assume this lobe has been setup with a OneMinus*Fresnel
    // as we want to use 1 - specular_fresnel. Also notice we use
    // cosThetaWo to evaluate the fresnel term, as an approximation of what
    // hDotWi would be for the specular lobe.
    return mirror->mTint * BsdfLobe_computeScaleAndFresnel(lobe, cosThetaWo) * dispersionColor *
            (slice.mIncludeCosineTerm  ?  1.0f  :  1.0f / cosThetaWi);
}

varying Color
MirrorTransmissionBsdfLobe_albedo(const varying BsdfLobe * uniform lobe,
                                  const varying BsdfSlice &slice)
{
    const varying MirrorTransmissionBsdfLobe * uniform mirror =
        (const varying MirrorTransmissionBsdfLobe * uniform) lobe;

    float cosThetaWo = dot(mirror->mN, slice.mWo);
    return mirror->mTint * BsdfLobe_computeScaleAndFresnel(lobe, abs(cosThetaWo));
}

void
MirrorTransmissionBsdfLobe_differentials(const varying BsdfLobe * uniform lobe,
                                         const varying Vec3f &wo,
                                         const varying Vec3f &wi,
                                         varying float r1, varying float r2,
                                         const varying Vec3f &dNdx,
                                         const varying Vec3f &dNdy,
                                         varying Vec3f &dDdx, varying Vec3f &dDdy)
{
    const varying MirrorTransmissionBsdfLobe * uniform mirror =
        (const varying MirrorTransmissionBsdfLobe * uniform) lobe;

    computeRefractionDirectionDifferential((mirror->mEtaI/mirror->mEtaT), wo, wi, mirror->mN,
            dNdx, dNdy, dDdx, dDdy);
}

uniform bool
MirrorTransmissionBsdfLobe_getProperty(const varying BsdfLobe * uniform lobe,
                                       uniform BsdfLobeProperty property,
                                       varying float * uniform dest)
{
    const varying MirrorTransmissionBsdfLobe * uniform mirror =
        (const varying MirrorTransmissionBsdfLobe * uniform) lobe;

    uniform bool handled = true;

    switch (property)
    {
    case BSDF_LOBE_PROPERTY_ROUGHNESS:
        *dest       = 0.0f;
        *(dest + 1) = 0.0f;
        break;
    case BSDF_LOBE_PROPERTY_NORMAL:
        *dest       = mirror->mN.x;
        *(dest + 1) = mirror->mN.y;
        *(dest + 2) = mirror->mN.z;
        break;
    default:
        handled = false;
        break;
    }

    return handled;
}

//-----------------------------------------------------------------------------

//
// MirrorRetroreflectionBsdfLobe
//

varying Color
MirrorRetroreflectionBsdfLobe_eval(const varying BsdfLobe * uniform lobe,
                                   const varying BsdfSlice &slice,
                                   const Vec3f &wi,
                                   varying float * uniform pdf)
{
    if (pdf != NULL) {
        *pdf = 0.0f;
    }
    return Color_ctor(0.0f, 0.0f, 0.0f);
}

varying Color
MirrorRetroreflectionBsdfLobe_sample(const varying BsdfLobe * uniform lobe,
                                     const varying BsdfSlice &slice,
                                     varying float r1, varying float r2,
                                     varying Vec3f &wi, varying float &pdf)
{
    const varying MirrorRetroreflectionBsdfLobe * uniform mirror =
        (const varying MirrorRetroreflectionBsdfLobe * uniform) lobe;

    //sample retroreflectively
    wi = slice.mWo;

    // Probability of sampling the delta function is 1.
    pdf = 1.0f;

    float cosTheta = dot(mirror->mN, slice.mWo);
    cosTheta = abs(cosTheta);

    // Compute contribution
    return BsdfLobe_computeScaleAndFresnel(lobe, cosTheta) *
            (slice.mIncludeCosineTerm  ?  1.0 :  1.0f / cosTheta);
}

varying Color
MirrorRetroreflectionBsdfLobe_albedo(const varying BsdfLobe * uniform lobe,
                                     const varying BsdfSlice &slice)
{
    const varying MirrorRetroreflectionBsdfLobe * uniform mirror =
            (const varying MirrorRetroreflectionBsdfLobe * uniform) lobe;

    float cosThetaWo = dot(mirror->mN, slice.mWo);
    return BsdfLobe_computeScaleAndFresnel(lobe, abs(cosThetaWo));
}

void
MirrorRetroreflectionBsdfLobe_differentials(const varying BsdfLobe * uniform lobe,
                                            const varying Vec3f &wo,
                                            const varying Vec3f &wi,
                                            varying float r1, varying float r2,
                                            const varying Vec3f &dNdx,
                                            const varying Vec3f &dNdy,
                                            varying Vec3f &dDdx, varying Vec3f &dDdy)
{
    //reverse differentials on a retroreflective bounce
    dDdx = neg(dDdx);
    dDdy = neg(dDdy);
}

uniform bool
MirrorRetroreflectionBsdfLobe_getProperty(const varying BsdfLobe * uniform lobe,
                                          uniform BsdfLobeProperty property,
                                          varying float * uniform dest)
{
    const varying MirrorRetroreflectionBsdfLobe * uniform mirror =
        (const varying MirrorRetroreflectionBsdfLobe * uniform) lobe;

    uniform bool handled = true;

    switch (property)
    {
    case BSDF_LOBE_PROPERTY_ROUGHNESS:
        *dest       = 0.0f;
        *(dest + 1) = 0.0f;
        break;
    case BSDF_LOBE_PROPERTY_NORMAL:
        *dest       = mirror->mN.x;
        *(dest + 1) = mirror->mN.y;
        *(dest + 2) = mirror->mN.z;
        break;
    default:
        handled = false;
        break;
    }
    return handled;
}

//-----------------------------------------------------------------------------

#pragma ignore warning(all)
ISPC_UTIL_EXPORT_STRUCT_TO_HEADER(MirrorReflectionBsdfLobe);
#pragma ignore warning(all)
ISPC_UTIL_EXPORT_STRUCT_TO_HEADER(MirrorTransmissionBsdfLobe);
#pragma ignore warning(all)
ISPC_UTIL_EXPORT_STRUCT_TO_HEADER(MirrorRetroreflectionBsdfLobe);

