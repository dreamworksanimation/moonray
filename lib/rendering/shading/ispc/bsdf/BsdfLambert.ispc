// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file BsdfLambert.ispc

#include "BsdfLambert.isph"
#include "BsdfSlice.isph"
#include <moonray/rendering/shading/ispc/Util.isph>
#include <moonray/rendering/shading/ispc/PbrValidity.isph>

#include <scene_rdl2/common/platform/IspcUtil.isph>

// See Substance for Unity: Chapter 01-01 Understanding PBR
// https://youtu.be/ueC2qGzWrgQ?t=914
#define LAMBERT_PBR_VALIDITY_MIN 0.031896
#define LAMBERT_PBR_VALIDITY_MAX 0.871367

varying Color
LambertBsdfLobe_eval(const varying BsdfLobe * uniform lobe,
                     const varying BsdfSlice &slice,
                     const Vec3f &wi,
                     varying float * uniform pdf)
{
    const varying LambertBsdfLobe * uniform lambert =
        (const varying LambertBsdfLobe * uniform) lobe;
    const Vec3f N = getN(lambert->mFrame);

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
        const Vec3f Nr = (BsdfLobe_matchesFlag(lobe, BSDF_LOBE_TYPE_REFLECTION)) ?
                            N : neg(N);
        cosThetaWo = max(dot(Nr, slice.mWo), 0.0f);
    }

    // Soften hard shadow terminator due to shading normals
    const float Gs = (BsdfLobe_matchesFlag(lobe, BSDF_LOBE_TYPE_REFLECTION)) ?
                     BsdfSlice_computeShadowTerminatorFix(&slice, getN(lambert->mFrame), wi) :
                     1.0f;

    return Gs * lambert->mAlbedo *
            BsdfLobe_computeScaleAndFresnel(lobe, cosThetaWo) * sOneOverPi *
            (slice.mIncludeCosineTerm ? cosThetaWi : 1.0f);
}

varying Color
LambertBsdfLobe_sample(const varying BsdfLobe * uniform lobe,
                       const varying BsdfSlice &slice,
                       varying float r1, varying float r2,
                       varying Vec3f &wi, varying float &pdf)
{
    const varying LambertBsdfLobe * uniform lambert =
        (const varying LambertBsdfLobe * uniform) lobe;

    wi = localToGlobal(lambert->mFrame, sampleLocalHemisphereCosine(r1, r2));
    return LambertBsdfLobe_eval(lobe, slice, wi, &pdf);
}

varying Color
LambertBsdfLobe_albedo(const varying BsdfLobe * uniform lobe,
                       const varying BsdfSlice &slice)
{
    const varying LambertBsdfLobe * uniform lambert =
        (const varying LambertBsdfLobe * uniform) lobe;

    float cosThetaWo = max(dot(getN(lambert->mFrame), slice.mWo), 0.0f);
    return BsdfLobe_computeScaleAndFresnel(lobe, cosThetaWo) * lambert->mAlbedo;
}

void
LambertBsdfLobe_differentials(const varying BsdfLobe * uniform lobe,
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

    const varying LambertBsdfLobe * uniform lambert =
        (const varying LambertBsdfLobe * uniform) lobe;

    dDdx = localToGlobal(lambert->mFrame, dDdx);
    dDdy = localToGlobal(lambert->mFrame, dDdy);
}

uniform bool
LambertBsdfLobe_getProperty(const varying BsdfLobe * uniform lobe,
                            uniform BsdfLobeProperty property,
                            varying float * uniform dest)
{
    const varying LambertBsdfLobe * uniform lambert =
        (const varying LambertBsdfLobe * uniform) lobe;

    uniform bool handled = true;

    switch (property)
    {
    case BSDF_LOBE_PROPERTY_ROUGHNESS:
        *dest       = 1.0f;
        *(dest + 1) = 1.0f;
        break;
    case BSDF_LOBE_PROPERTY_NORMAL:
        {
            const Vec3f N = getN(lambert->mFrame);
            *dest       = N.x;
            *(dest + 1) = N.y;
            *(dest + 2) = N.z;
        }
        break;
    case BSDF_LOBE_PROPERTY_PBR_VALIDITY:
        {
            Color res = computeAlbedoPbrValidity(lambert->mAlbedo);
            *dest       = res.r;
            *(dest + 1) = res.g;
            *(dest + 2) = res.b;
        }
        break;
    case BSDF_LOBE_PROPERTY_COLOR:
        {
            // Special case handling for lambert color property since albedo isn't included
            // in mScale anymore.
            varying Color c = lambert->mAlbedo * BsdfLobe_computeScaleAndFresnel(lobe, 1.f);
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


#pragma ignore warning(all)
ISPC_UTIL_EXPORT_STRUCT_TO_HEADER(LambertBsdfLobe);

