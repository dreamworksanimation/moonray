// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file BsdfFlatDiffuse.ispc

#include <moonray/rendering/shading/ispc/bsdf/BsdfOrenNayar.isph>
#include "BsdfFlatDiffuse.isph"
#include <moonray/rendering/shading/ispc/bsdf/BsdfSlice.isph>
#include <moonray/rendering/shading/ispc/Util.isph>

#include <scene_rdl2/common/platform/IspcUtil.isph>

varying float
getOrenNayarFalloffIntensity(const varying Vec3f& n,
                             const varying Vec3f& v,
                             const varying Vec3f& l,
                             const varying float a,
                             const varying float b,
                             const varying float terminatorShift)
{
    // We are using the Fujii implementation or Oren-Nayar.
    // The PBRT implementation was also tested but resulted
    // in more energy loss.
    // https://mimosa-pudica.net/improved-oren-nayar.html
    // Note: For the flat diffuse lobe type we introduce an 
    // additional terminatorShift control that reshapes the n dot l
    // response
    const float nl = max(0.0f, cos(min(sHalfPi, dw_acos(clamp(dot(n, l), -1.0f, 1.0f)) * terminatorShift)));
    const float nv = max(dot(n, v), 0.0f);
    const float s = dot(l, v) - nl * nv;

    float t = 1.0f;
    if (s > 0.0f) {
        t = max(nl, nv) + sEpsilon;
    }

    return nl * (a + b * s/t);
}

varying Color
FlatDiffuseBsdfLobe_eval(const varying BsdfLobe * uniform lobe,
                         const varying BsdfSlice &slice,
                         const Vec3f &wi,
                         varying float * uniform pdf)
{
    const varying FlatDiffuseBsdfLobe * uniform flatDiffuse =
        (const varying FlatDiffuseBsdfLobe * uniform) lobe;

    // When flatness approaches 1.0 it tends to produce an extremely harsh terminator,
    // which can lead to polygonal artifacts where self-shadowing occurs.  This remapping
    // attempts to apply a targeted softening effect at the terminator - similar to the
    // "shadow terminator fix" that we used to soften bump mapping

    // Calculate the flatness falloff off by remapping
    // the angle between the light and the normal.
    // Desmos expression:
    // https://www.desmos.com/calculator/z1iufecqmf
    const float terminatorShift = flatDiffuse->mTerminatorShift + 1.0f;
    const float theta = min(sHalfPi, terminatorShift *
                        dw_acos(clamp(dot(getN(flatDiffuse->mOrenNayar.mFrame), wi), -1.0f, 1.0f)));
    const float a = 1.0f - flatDiffuse->mFlatnessFalloff;
    // The power of "a" controls the linearity of the falloff parameter.
    // Raising to the power of 4 slightly biases the control towards
    // the low end giving more control over a low falloff value.  The
    // constant 100.0 controls how sharply the falloff occurs when the
    // parameter is set to 1.0.
    const float b = 1.0f + a * a * a * a * 100.0f;
    const float c = sHalfPi / b;
    const float d = c - sHalfPi;
    const float t = (theta > sHalfPi - c) ? cos((theta + d) * b) : 1.0f;
    const float flatness = flatDiffuse->mFlatness * t;
    const Vec3f flatN = normalize(lerp(getN(flatDiffuse->mOrenNayar.mFrame), wi, flatness));

    const float cosThetaWi = getOrenNayarFalloffIntensity(flatN,
                                                          slice.mWo, wi,
                                                          flatDiffuse->mOrenNayar.mA,
                                                          flatDiffuse->mOrenNayar.mB,
                                                          terminatorShift);
    if (pdf != NULL) {
        *pdf = cosThetaWi * sOneOverPi;
    }

    // Note: we assume this lobe has been setup with a OneMinus*Fresnel
    // as we want to use 1 - specular_fresnel. Also notice we use
    // cosThetaWo to evaluate the fresnel term, as an approximation of what
    // hDotWi would be for the specular lobe.
    float cosThetaWo = 1.0f;
    if (isActive(BsdfLobe_getFresnel(lobe))) {
        const Vec3f N = (BsdfLobe_matchesFlag(lobe, BSDF_LOBE_TYPE_REFLECTION)) ?  flatN : neg(flatN);
        cosThetaWo = max(dot(N, slice.mWo), 0.0f);
    }

    // Soften hard shadow terminator due to shading normals
    const float Gs = (BsdfLobe_matchesFlag(lobe, BSDF_LOBE_TYPE_REFLECTION)) ?
                     BsdfSlice_computeShadowTerminatorFix(&slice, flatN, wi) :
                     1.0f;

    const Color result = flatDiffuse->mOrenNayar.mAlbedo * sOneOverPi * Gs *
           BsdfLobe_computeScaleAndFresnel(lobe, cosThetaWo) *
           (slice.mIncludeCosineTerm ? cosThetaWi : 1.0f);

    // This is an ad hoc factor that tries to minimize energy
    // loss while also minimizing the difference in brightness
    // compared with lambertian (i.e. flatnesss = 0).
    const float normalizationFactor = lerp(1.0f,
                                           0.75f,
                                           bias(flatDiffuse->mFlatness, 0.75f));

    return result * normalizationFactor;
}

varying Color
FlatDiffuseBsdfLobe_sample(const varying BsdfLobe * uniform lobe,
                           const varying BsdfSlice &slice,
                           varying float r1, varying float r2,
                           varying Vec3f &wi, varying float &pdf)
{
    const varying FlatDiffuseBsdfLobe * uniform flatDiffuse =
        (const varying FlatDiffuseBsdfLobe * uniform) lobe;

    wi = localToGlobal(flatDiffuse->mOrenNayar.mFrame, sampleLocalHemisphereCosine(r1, r2));
    return FlatDiffuseBsdfLobe_eval(lobe, slice, wi, &pdf);
}

varying Color
FlatDiffuseBsdfLobe_albedo(const varying BsdfLobe * uniform lobe,
                           const varying BsdfSlice &slice)
{
    return OrenNayarBsdfLobe_albedo(lobe, slice);
}

void
FlatDiffuseBsdfLobe_differentials(const varying BsdfLobe * uniform lobe,
                                  const varying Vec3f &wo,
                                  const varying Vec3f &wi,
                                  varying float r1, varying float r2,
                                  const varying Vec3f &dNdx,
                                  const varying Vec3f &dNdy,
                                  varying Vec3f &dDdx, varying Vec3f &dDdy)
{
    OrenNayarBsdfLobe_differentials(lobe, wo, wi, r1, r2, dNdx, dNdy, dDdx, dDdy);
}

uniform bool
FlatDiffuseBsdfLobe_getProperty(const varying BsdfLobe * uniform lobe,
                                uniform BsdfLobeProperty property,
                                varying float * uniform dest)
{
    return OrenNayarBsdfLobe_getProperty(lobe, property, dest);
}

#pragma ignore warning(all)
ISPC_UTIL_EXPORT_STRUCT_TO_HEADER(FlatDiffuseBsdfLobe);

