// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file BsdfFlakes.isph

#include "BsdfStochasticFlakes.isph"
#include <moonray/rendering/shading/ispc/Ior.isph>
#include <moonray/rendering/shading/ispc/Util.isph>

#include <scene_rdl2/common/platform/IspcUtil.isph>

#define UNIFORM_SAMPLING 0

struct FlakeCaches {
    Vec3f reflections[StochasticFlakes_sMaxFlakes];
    float weights[StochasticFlakes_sMaxFlakes];
    Color scales[StochasticFlakes_sMaxFlakes];
    float wCDF[StochasticFlakes_sMaxFlakes];
    Vec3f flakeNormals[StochasticFlakes_sMaxFlakes];
    unsigned int size;
};

// Compute the D term as explained in "Reflection Cache" section of [2016]
// This is refactored into a function since we compute D twice for the bsdf
// calculation, once with Wo and again with Wi and use the average
varying float
StochasticFlakesBsdfLobe_computeD(varying StochasticFlakesBsdfLobe * uniform lobe,
                                  const varying FlakeCaches& caches,
                                  const varying Vec3f& w,
                                  varying Color& weightedColor,
                                  varying float * uniform reflectionWeight = nullptr,
                                  varying float * uniform totalWeights = nullptr)
{
    if (totalWeights != nullptr) {
        *totalWeights = 0.0f;
    }
    float reflectWt = 0.0f;
    weightedColor = sBlack;

    for (unsigned int i = 0; i < caches.size; ++i) {
        // Formulating the pdf and brdf weights calculation according to Chaos Group paper
        float wDotRj = dot(w, caches.reflections[i]);
        if(wDotRj >= lobe->mCosGamma) {
            reflectWt += caches.weights[i];
            weightedColor = weightedColor + (caches.scales[i] * caches.weights[i]);
        }
        if (totalWeights != nullptr) {
            *totalWeights += caches.weights[i];
        }
    }

    if (reflectionWeight != nullptr) {
        *reflectionWeight = reflectWt; // k for the pdf in [2016]
    }
    if (!isZero(reflectWt)) {
        weightedColor = weightedColor * rcp(reflectWt); // normalize the weighted color
    }
    return reflectWt / (float)caches.size; // Return discrete D
}

inline varying float
StochasticFlakeBsdfLobe_computeGeometryTerm(const varying float cosNO,
                                            const varying float cosNI,
                                            const varying float randomness)
{
    const float invCosNO = 1.0f / cosNO;
    const float invCosNI = 1.0f / cosNI;
    const float ao = 1.0f / (randomness * sqrt((1.0f - cosNO * cosNO) * (invCosNO * invCosNO)));
    const float ai = 1.0f / (randomness * sqrt((1.0f - cosNI * cosNI) * (invCosNI * invCosNI)));
    const float G1o = (ao < 1.6f) ? (3.535f * ao + 2.181f * ao * ao) /
                            (1.0f + 2.276f * ao + 2.577f * ao * ao) : 1.0f;
    const float G1i = (ai < 1.6f) ? (3.535f * ai + 2.181f * ai * ai) /
                            (1.0f + 2.276f * ai + 2.577f * ai * ai) : 1.0f;
    return G1o * G1i;
}

varying Vec3f * varying
getFlakeNormals(varying StochasticFlakesBsdfLobe * uniform lobe)
{
    return (varying Vec3f * varying)lobe->mFlakeNormals;
}


varying Color * varying
getFlakeColors(varying StochasticFlakesBsdfLobe * uniform lobe)
{
    return (varying Color * varying)lobe->mFlakeColors;
}
// ------------------------------------------------------------------------------------------------

/// Supporting functions for eval and sample
varying Color
StochasticFlakesBsdfLobe_computeBrdf(varying StochasticFlakesBsdfLobe * uniform lobe,
                                     const varying BsdfSlice& slice,
                                     const varying Vec3f& wi,
                                     const varying FlakeCaches& cachesWo,
                                     const varying FlakeCaches& cachesWi,
                                     varying float * uniform pdf = NULL)
{
    // calculate pdf as
    // Sum(weight[i], such that dot(wi,reflection[i]) < cos(gamma)) / pi*(1-cos(gamma))*sum(all weights)
    //
    // calculate return color value as
    // sum(scale[i]*weight[i], such that dot(wi,reflection[i]) < cos(gamma)) / sum(all weights)

    // prepare for possible early return
    if(pdf != NULL) {
        *pdf = 0.0f;
    }

    Vec3f m;
    if (!computeNormalizedHalfVector(slice.mWo, wi, m)) {
        return sBlack;
    }

    float kIntersect = 0.0f; // k in [2016] - "optimal importance sampling"
    float kTotalSum = 0.0f;  // c_n in [2016] - "optimal importance sampling"
    Color weightedColor1 = sBlack, weightedColor2 = sBlack;

    // **Cheat** for reciprocity. The D term evaluates how many of the flakes
    // reflect light from Wo into a cone around Wi. We however, also compute D
    // by flipping Wo and Wi and averaging the two results.
    // kIntersect and kTotalSum are only used for evaluating the pdf,
    // so we only accumulate them when we compute D as per the talk
    // (we retain the importance sampling scheme and thus the pdf from the talk)
    float discreteD = 0.5f *
            (StochasticFlakesBsdfLobe_computeD(lobe, cachesWo, wi, weightedColor1, &kIntersect, &kTotalSum) +
            StochasticFlakesBsdfLobe_computeD(lobe, cachesWi, slice.mWo, weightedColor2));

    const float cosNI = min(dot(getN(lobe->mFrame), wi), sOneMinusEpsilon);
    if (cosNI <= sEpsilon) return sBlack;

    const float cosNO = min(dot(getN(lobe->mFrame), slice.mWo), sOneMinusEpsilon);
    if (cosNO <= sEpsilon) return sBlack;

    const float cosMI = dot(m, wi);
    if (cosMI <= sEpsilon) return sBlack;

    // Computing D as described in the Chaos Group paper
    //
    // D is broken up into two parts. D~ and factor 4/(a(A)*sigma(SolidAngle))
    //
    // D = (D~)*4/(a(A)*sigma(SolidAngle))
    //
    // a(A) is the area of the pixel footprint
    // sigma(SolidAngle) is the solid angle area
    //
    // and
    // D~ = (1/N)*(Sum of all weights contributing to the reflection direction)
    //
    // Sum of all weights = kIntersect
    // N = number of flakes per unit area, i.e. it is constant
    // See Jakob, Haan, et al "Discrete Stochastic Micofacet Models" for a clear description of N
    // since it is not clear in the Chaos group paper and can be confused with the number of flakes
    // *within* the footprint
    //
    // So then
    //
    // D = 4/(a(A)*sigma(SolidAngle))*(1/N)*kIntersect
    // Re-organizing, moving N into the denominator...
    // D = 4/(N*a(A)*sigma(SolidAngle))*kIntersect
    //
    // Note the factor N*a(A) = E(n), which is the expected value for the number of flakes
    // E(n) is then the number of flakes within the pixel footprint, mFlakeCount
    //
    // So D = (4/(mFlakeCount*SolidAngleArea))*kIntersect
    //
    // Since the CookTorrance model divides by 4, the factor of 4 is left out in
    // both parts since they cancel.

    const float solidAngleArea = sTwoPi * (1.0f - lobe->mCosGamma); // must be 2*pi, the Chaos paper has only pi and is incorrect.

    const float D = discreteD * rcp(solidAngleArea);

    Color F = BsdfLobe_computeScaleAndFresnel((const varying BsdfLobe * uniform)lobe,
                                              cosMI);

    // compute G for CookTorrance
    const float G = StochasticFlakeBsdfLobe_computeGeometryTerm(cosNO, cosNI, lobe->mFlakeRandomness);

    const float invBrdfDenom = rcp( cosNO * (slice.mIncludeCosineTerm  ?  1.0f  :  cosNI) );
    Color flakesBrdf = F*D*G*invBrdfDenom;

    if(pdf != NULL) {
#if UNIFORM_SAMPLING
        *pdf = sOneOverTwoPi;
#else
        *pdf = kIntersect * rcp(solidAngleArea * kTotalSum);
#endif
    }
    // Create a conductor fresnel object using the weighted (reflection cache weights)
    // color of the flakes that contribute to the final bsdf
    // **Note: This is a special case use of the fresnel object inside the bsdf lobe itself -
    // since we know the color to use only after evaluating the reflection cache
    Color avgWeightedColor = min(Color_ctor(0.999f), 0.5f * (weightedColor1+weightedColor2));
    Color eta = ShaderComplexIor_computeEta(avgWeightedColor, avgWeightedColor);
    Color k = ShaderComplexIor_computeK(avgWeightedColor, eta);

    varying ConductorFresnel fresnel;
    ConductorFresnel_init(&fresnel, eta, k);
    Color result = flakesBrdf * ConductorFresnel_eval((const varying Fresnel * uniform)&fresnel, cosMI);
    return result;
}

varying bool
StochasticFlakesBsdfLobe_computeReflectionCache(varying StochasticFlakesBsdfLobe * uniform lobe,
                                                const varying Vec3f& w,
                                                varying FlakeCaches& caches,
                                                const uniform bool buildCDF)
{
    Vec3f reflection;
    float weight;

    unsigned int k = 0;
    float totalWeight = 0;
    varying Vec3f * varying normals = getFlakeNormals(lobe);
    varying Color * varying colors = getFlakeColors(lobe);
    for (unsigned int i = 0; i < lobe->mFlakeCount; ++i) {
        Vec3f normal = normals[i];
        weight = max(0.0f,computeReflectionDirection(normal, w, reflection));
        caches.weights[k] = weight;
        caches.reflections[k] = reflection;
        caches.scales[k] = colors[i];
        caches.flakeNormals[k] = normal;
        totalWeight += weight;
        k++;
    }

    if (totalWeight > 0) {
        totalWeight = rcp(totalWeight);
    } else {
        return false;
    }

    if (buildCDF) {
        float cdf = 0.0f;
        for (unsigned int i = 0; i < k; i++) {
            cdf += caches.weights[i] * totalWeight;
            caches.wCDF[i] = cdf;
        }
    }

    caches.size = k;
    return true;
}

/// Utility function to importance sample the CDF
#if !UNIFORM_SAMPLING
varying unsigned int
StochasticFlakesBsdfLobe_selectIndexFromCdf(const varying float * uniform cdfs,
                                            varying unsigned int count,
                                            varying const float r,
                                            varying float &remapped_r)
{
    // -------------------------------------------------------
    // upper_bound adopted from Embree Distribution1D__sample()
    // scalar code uses std::upper_bound()
    // -------------------------------------------------------
    const float val = r;
    int first = 0;
    int len = count;

    while (len > 0) {
        const int __half = len >> 1;
        const int __middle = first + __half;
        if (val < cdfs[__middle]) {
            len = __half;
        } else {
            first = __middle+1;
            len = len - __half - 1;
        }
    };
    const int index = clamp(first, 0, (int)count-1);
    float cdfRange = cdfs[index];
    if (index >= 1) {
        cdfRange = cdfs[index] - cdfs[index-1];
        remapped_r = (r-cdfs[index-1]) * rcp(cdfRange);
    } else {
        remapped_r = r * rcp(cdfRange);
    }
    return index;
}
#endif

// ------------------------------------------------------------------------------------------------

/// Implementation of the StochasticBsdfLobe API

varying Color
StochasticFlakesBsdfLobe_eval(const varying BsdfLobe * uniform lobe,
                              const varying BsdfSlice &slice,
                              const Vec3f &wi,
                              varying float * uniform pdf)
{
    varying StochasticFlakesBsdfLobe * uniform sfLobe =
            (varying StochasticFlakesBsdfLobe * uniform) lobe;
    // The bsdf from [2016] is not reciprocal (evaluating reflection cache based on Wo)
    // Using an additional reflection cache computed using Wi and
    // averaging the D term in the bsdf is a cheat we use to make
    // this bsdf reciprocal.
    varying FlakeCaches evalCachesWo, evalCachesWi;
    varying bool validCache = StochasticFlakesBsdfLobe_computeReflectionCache(sfLobe, slice.mWo, evalCachesWo, false);
    validCache &= StochasticFlakesBsdfLobe_computeReflectionCache(sfLobe, wi, evalCachesWi, false);

    if (!validCache) {
        if (pdf) *pdf = 0.0f;
        return sBlack;
    }

    return StochasticFlakesBsdfLobe_computeBrdf(sfLobe, slice, wi, evalCachesWo, evalCachesWi, pdf);
}

varying Color
StochasticFlakesBsdfLobe_sample(const varying BsdfLobe * uniform lobe,
                                const varying BsdfSlice &slice,
                                varying float r1, varying float r2,
                                varying Vec3f &wi, varying float &pdf)
{
    varying StochasticFlakesBsdfLobe * uniform sfLobe =
            (varying StochasticFlakesBsdfLobe * uniform) lobe;

    varying FlakeCaches sampleCachesWo;
    // generate reflection cache and weights with weighted CDF
    bool validCache = StochasticFlakesBsdfLobe_computeReflectionCache(sfLobe, slice.mWo, sampleCachesWo, true);

    if (!validCache) {
        pdf = 0.0f;
        return sBlack;
    }

#if UNIFORM_SAMPLING
    varying Vec3f wiLocal = sampleLocalHemisphereUniform(r1, r2);
    wi = localToGlobal(sfLobe->mFrame, wiLocal);
#else
    if (sampleCachesWo.size == 0) {
        varying Vec3f wiLocal = sampleLocalHemisphereUniform(r1, r2);
        wi = localToGlobal(sfLobe->mFrame, wiLocal);
    } else {
        varying float remapped_r1 = r1;
        varying unsigned int reflIdx =
                StochasticFlakesBsdfLobe_selectIndexFromCdf(sampleCachesWo.wCDF, sampleCachesWo.size, r1, remapped_r1);
        varying Vec3f reflSample = sampleCachesWo.reflections[reflIdx];

        // Sample the roughness of the individual flake
        // generate local wi using cos(gamma) as cut off
        varying Vec3f wiLocal = sampleLocalSphericalCapUniform(remapped_r1, r2, sfLobe->mCosGamma);

        // transform local wi to reflection sample reference frame
        varying ReferenceFrame reflReferenceFrame;
        ReferenceFrame_init(reflReferenceFrame, reflSample);
        wi = localToGlobal(reflReferenceFrame, wiLocal);
    }
#endif

    // Same as in eval() - we use an additional reflection cache created
    // using Wi to enforce reciprocity in bsdf computation
    varying FlakeCaches sampleCachesWi;
    validCache = StochasticFlakesBsdfLobe_computeReflectionCache(sfLobe, wi, sampleCachesWi, false);

    if (!validCache) {
        pdf = 0.0f;
        return sBlack;
    }

    return StochasticFlakesBsdfLobe_computeBrdf(sfLobe, slice, wi, sampleCachesWo, sampleCachesWi, &pdf);
}

varying Color
StochasticFlakesBsdfLobe_albedo(const varying BsdfLobe * uniform lobe,
                                const varying BsdfSlice &slice)
{
    varying StochasticFlakesBsdfLobe * uniform sfLobe =
            (varying StochasticFlakesBsdfLobe * uniform) lobe;

    float cosThetaWo = max(dot(getN(sfLobe->mFrame), slice.mWo), 0.0f);
    return BsdfLobe_computeScaleAndFresnel(lobe, cosThetaWo);
}

void
StochasticFlakesBsdfLobe_differentials(const varying BsdfLobe * uniform lobe,
                                       const varying Vec3f &wo,
                                       const varying Vec3f &wi,
                                       varying float r1, varying float r2,
                                       const varying Vec3f &dNdx,
                                       const varying Vec3f &dNdy,
                                       varying Vec3f &dDdx, varying Vec3f &dDdy)
{
    varying StochasticFlakesBsdfLobe * uniform sfLobe =
            (varying StochasticFlakesBsdfLobe * uniform) lobe;

    Vec3f H;
    if (!computeNormalizedHalfVector(wo, wi, H)) {
        H = getN(sfLobe->mFrame);
    }
    computeReflectionDirectionDifferential(wo, wi, H, dNdx, dNdy, dDdx, dDdy);
}

uniform bool
StochasticFlakesBsdfLobe_getProperty(const varying BsdfLobe * uniform lobe,
                                     uniform BsdfLobeProperty property,
                                     varying float * uniform dest)
{
    varying StochasticFlakesBsdfLobe * uniform sfLobe =
            (varying StochasticFlakesBsdfLobe * uniform) lobe;

    uniform bool handled = true;

    switch (property)
    {
    case BSDF_LOBE_PROPERTY_ROUGHNESS:
        *dest       = sfLobe->mFlakeRandomness;
        *(dest + 1) = sfLobe->mFlakeRandomness;
        break;
    case BSDF_LOBE_PROPERTY_NORMAL:
        {
            const Vec3f N = getN(sfLobe->mFrame);
            *dest       = N.x;
            *(dest + 1) = N.y;
            *(dest + 2) = N.z;
        }
        break;
    case BSDF_LOBE_PROPERTY_PBR_VALIDITY:
        // Forward call to fresnel
        {
            varying Color res = sBlack;
            if (isActive(lobe->mFresnel)) {
                res = Fresnel_computePbrValidity(lobe->mFresnel);
            }
            *dest       = res.r;
            *(dest + 1) = res.g;
            *(dest + 2) = res.b;
        }
        break;
    default:
        handled = false;
        break;
    }

    return handled;
}

// ------------------------------------------------------------------------------------------------

#pragma ignore warning(all)
ISPC_UTIL_EXPORT_STRUCT_TO_HEADER(StochasticFlakesBsdfLobe);

