// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file BsdfFlakes.h
///

#pragma once

#include "Bsdf.h"
#include "BsdfSlice.h"
#include <moonray/rendering/shading/Util.h>

#include <moonray/rendering/shading/ispc/bsdf/BsdfStochasticFlakes_ispc_stubs.h>

#include <scene_rdl2/common/math/Color.h>
#include <scene_rdl2/common/math/ReferenceFrame.h>
#include <scene_rdl2/common/math/Vec2.h>
#include <scene_rdl2/common/math/Vec3.h>
#include <scene_rdl2/common/platform/IspcUtil.h>

#define UNIFORM_SAMPLING 0

namespace moonray {
namespace shading {

namespace {

#if !UNIFORM_SAMPLING
unsigned int
selectIndexFromCdf(const float* cdfs, unsigned int count, const float r, float &remapped_r)
{
    const float *ptr = std::upper_bound(cdfs, cdfs+count, r);
    unsigned int index = ptr - cdfs;
    index =  scene_rdl2::math::min(index, count - 1);
    float cdfRange = cdfs[index];
    if (index >= 1) {
        cdfRange = cdfs[index] - cdfs[index-1];
        remapped_r = (r-cdfs[index-1]) * scene_rdl2::math::rcp(cdfRange);
    } else {
        remapped_r = r * scene_rdl2::math::rcp(cdfRange);
    }
    return index;
}
#endif

} // anonymous namespace

//----------------------------------------------------------------------------

///
/// @class StochasticFlakesBsdfLobe BsdfFlakes.h <shading/BsdfFlakes.h>
/// @brief Bsdf lobe for discrete flake surfaces
///
// This bsdf implementation is based on a Siggraph talk:
// ATANASOV, A., AND KOYLAZOV, V. 2016. A practical stochastic
// algorithm for rendering mirror-like flakes. In ACM SIGGRAPH
// 2016 Talks, ACM, New York, NY, USA, SIGGRAPH ’16, 67:1–
// 67:2.
// Referring to this talk as [2016] in the rest of the code
class StochasticFlakesBsdfLobe : public BsdfLobe
{
public:
    // Constructor / Destructor
    StochasticFlakesBsdfLobe(const scene_rdl2::math::Vec3f& N,
                             const scene_rdl2::math::Vec3f* flakeNormals,
                             const scene_rdl2::math::Color* flakeColors,
                             const size_t flakeCount,
                             float roughness,
                             float inputFlakeRandomness) :
        BsdfLobe(Type(REFLECTION | GLOSSY),
                 DifferentialFlags(0),
                 false,
                 PROPERTY_NORMAL | PROPERTY_ROUGHNESS |
                 PROPERTY_PBR_VALIDITY),
        mFrame(N),
        mFlakeRandomness(scene_rdl2::math::max(0.001f, inputFlakeRandomness*inputFlakeRandomness)),
        mCosGamma(scene_rdl2::math::min(0.999f, 1.0f - roughness*roughness)),
        mFlakeCount(scene_rdl2::math::min(flakeCount, (size_t)sMaxFlakes)),
        mFlakeNormals(flakeNormals),
        mFlakeColors(flakeColors)
        { }

    // BsdfLobe API
    finline scene_rdl2::math::Color eval(const BsdfSlice &slice, const scene_rdl2::math::Vec3f &wi, float *pdf = NULL) const override
    {
        // The bsdf from [2016] is not reciprocal (evaluating reflection cache based on Wo)
        // Using an additional reflection cache computed using Wi and
        // averaging the D term in the bsdf is a cheat we use to make
        // this bsdf reciprocal.
        FlakeCaches evalCachesWo, evalCachesWi;
        bool validCache = computeReflectionCache(slice.getWo(), evalCachesWo, false);
        validCache &= computeReflectionCache(wi, evalCachesWi, false);

        if (!validCache) {
            if (pdf) *pdf = 0.0f;
            return scene_rdl2::math::sBlack;
        }

        return computeBrdf(slice, wi, evalCachesWo, evalCachesWi, pdf);
    }
    
    finline scene_rdl2::math::Color sample(const BsdfSlice &slice, float r1, float r2,
            scene_rdl2::math::Vec3f &wi, float &pdf) const override
    {
        FlakeCaches sampleCachesWo;

        // generate reflection cache and weights with weighted CDF
        bool validCache = computeReflectionCache(slice.getWo(), sampleCachesWo, true);

        if (!validCache) {
            pdf = 0.0f;
            return scene_rdl2::math::sBlack;
        }
#if UNIFORM_SAMPLING
        scene_rdl2::math::Vec3f wiLocal = sampleLocalHemisphereUniform(r1, r2);
        wi = mFrame.localToGlobal(wiLocal);

#else 
        if (sampleCachesWo.size == 0) {
            scene_rdl2::math::Vec3f wiLocal = sampleLocalHemisphereUniform(r1, r2);
            wi = mFrame.localToGlobal(wiLocal);
        } else {
            float remapped_r1 = r1;
            unsigned int reflIdx = selectIndexFromCdf(sampleCachesWo.wCDF, sampleCachesWo.size, r1, remapped_r1);

            scene_rdl2::math::Vec3f reflSample = sampleCachesWo.reflections[reflIdx];

            // Sample the roughness of the individual flake
            // generate local wi using cos(gamma) as cut off
            scene_rdl2::math::Vec3f wiLocal = sampleLocalSphericalCapUniform(remapped_r1, r2, mCosGamma);

            // transform local wi to reflection sample reference frame
            scene_rdl2::math::ReferenceFrame reflReferenceFrame(reflSample);
            wi = reflReferenceFrame.localToGlobal(wiLocal);
        }
#endif

        // Same as in eval() - we use an additional reflection cache created
        // using Wi to enforce reciprocity in bsdf computation
        FlakeCaches sampleCachesWi;
        validCache = computeReflectionCache(wi, sampleCachesWi, false);

        if (!validCache) {
            pdf = 0.0f;
            return scene_rdl2::math::sBlack;
        }

        return computeBrdf(slice, wi, sampleCachesWo, sampleCachesWi, &pdf);
    }


    finline scene_rdl2::math::Color albedo(const BsdfSlice &slice) const override
    {
        float cosThetaWo = scene_rdl2::math::max(dot(mFrame.getN(), slice.getWo()), 0.0f);
        return computeScaleAndFresnel(scene_rdl2::math::abs(cosThetaWo));
    }

    void differentials(const scene_rdl2::math::Vec3f &wo, const scene_rdl2::math::Vec3f &wi,
            float r1, float r2, const scene_rdl2::math::Vec3f &dNdx, const scene_rdl2::math::Vec3f &dNdy,
            scene_rdl2::math::Vec3f &dDdx, scene_rdl2::math::Vec3f &dDdy) const override
    {
        scene_rdl2::math::Vec3f H;
        if (!computeNormalizedHalfVector(wo, wi, H)) {
            H = mFrame.getN();
        }
        computeReflectionDirectionDifferential(wo, wi, H, dNdx, dNdy, dDdx, dDdy);
    }

    bool getProperty(Property property, float *dest) const override
    {
        bool handled = true;

        switch (property) {
        case PROPERTY_ROUGHNESS:
            {
                *dest       = mFlakeRandomness;
                *(dest + 1) = mFlakeRandomness;
            }
            break;
        case PROPERTY_NORMAL:
            {
                const scene_rdl2::math::Vec3f &N = mFrame.getN();
                *dest       = N.x;
                *(dest + 1) = N.y;
                *(dest + 2) = N.z;
            }
            break;
        case PROPERTY_PBR_VALIDITY:
            {
                const Fresnel* fresnel = getFresnel();
                // Get the property from the fresnel object
                scene_rdl2::math::Color res = scene_rdl2::math::sBlack;
                if (fresnel) {
                    res = fresnel->computePbrValidity();
                }
                *dest       = res.r;
                *(dest + 1) = res.g;
                *(dest + 2) = res.b;
            }
            break;
        default:
            handled = BsdfLobe::getProperty(property, dest);
        }

        return handled;
    }

    void show(std::ostream& os, const std::string& indent) const override
    {
        const scene_rdl2::math::Color& scale = getScale();
        const Fresnel * const fresnel = getFresnel();
        os << indent << "[StochasticFlakesBsdfLobe]\n";
        os << indent << "    " << "scale: "
            << scale.r << " " << scale.g << " " << scale.b << "\n";
        if (fresnel) {
            fresnel->show(os, indent + "    ");
        }
    }

private:

    scene_rdl2::math::ReferenceFrame mFrame;
    float mFlakeRandomness;
    float mCosGamma;

    static const int sMaxFlakes = 2000;

    // Supporting members for reflection cache calculation and evaluation
    struct FlakeCaches {
        scene_rdl2::math::Vec3f reflections[sMaxFlakes];
        float weights[sMaxFlakes];
        scene_rdl2::math::Color scales[sMaxFlakes];
        float wCDF[sMaxFlakes];
        scene_rdl2::math::Vec3f flakeNormals[sMaxFlakes];
        unsigned int size;
    };

    size_t mFlakeCount;
    const scene_rdl2::math::Vec3f* mFlakeNormals; // pointer to array containing normals and colors
    const scene_rdl2::math::Color* mFlakeColors;

    // Compute the D term as explained in "Reflection Cache" section of [2016]
    // This is refactored into a function since we compute D twice for the bsdf
    // calculation, once with Wo and again with Wi and use the average
    finline float computeD(const FlakeCaches& caches, const scene_rdl2::math::Vec3f& w,
                           scene_rdl2::math::Color& weightedColor,
                           float* reflectionWeight = nullptr,
                           float* totalWeights = nullptr) const
    {
        if (totalWeights != nullptr) {
            *totalWeights = 0.0f;
        }
        float reflectWt = 0.0f;
        weightedColor = scene_rdl2::math::sBlack;

        for(unsigned int i = 0; i < caches.size; ++i) {
            // Formulating the pdf and brdf weights calculation according to Chaos Group paper
            float wDotRj = dot(w, caches.reflections[i]);
            if(wDotRj >= mCosGamma) {
                reflectWt += caches.weights[i];
                weightedColor += caches.scales[i] * caches.weights[i];
            }
            if (totalWeights != nullptr) {
                *totalWeights += caches.weights[i];
            }
        }

        if (reflectionWeight != nullptr) {
            *reflectionWeight = reflectWt; // k for the pdf in [2016]
        }
        if (!scene_rdl2::math::isZero(reflectWt)) {
            weightedColor /= reflectWt; // normalize the weighted color
        }
        return reflectWt / static_cast<float>(caches.size); // Return discrete D
    }

    /// Supporting private member functions
    finline scene_rdl2::math::Color computeBrdf(
        const BsdfSlice& slice, const scene_rdl2::math::Vec3f& wi,
        const FlakeCaches& cachesWo, const FlakeCaches& cachesWi, float *pdf = NULL) const
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

        scene_rdl2::math::Vec3f m;
        if (!computeNormalizedHalfVector(slice.getWo(), wi, m)) {
            return scene_rdl2::math::sBlack;
        }

        float kIntersect = 0.0f; // k in [2016] - "optimal importance sampling"
        float kTotalSum = 0.0f;  // c_n in [2016] - "optimal importance sampling"
        scene_rdl2::math::Color weightedColor1 = scene_rdl2::math::sBlack, weightedColor2 = scene_rdl2::math::sBlack;

        // **Cheat** for reciprocity. The D term evaluates how many of the flakes
        // reflect light from Wo into a cone around Wi. We however, also compute D
        // by flipping Wo and Wi and averaging the two results.
        // kIntersect and kTotalSum are only used for evaluating the pdf,
        // so we only accumulate them when we compute D as per the talk
        // (we retain the importance sampling scheme and thus the pdf from the talk)
        float discreteD = 0.5f * (computeD(cachesWo, wi, weightedColor1, &kIntersect, &kTotalSum) +
                computeD(cachesWi, slice.getWo(), weightedColor2, nullptr, nullptr));

        const float cosNI = std::min(dot(mFrame.getN(), wi), scene_rdl2::math::sOneMinusEpsilon);
        if (cosNI <= scene_rdl2::math::sEpsilon) return scene_rdl2::math::sBlack;

        const float cosNO = std::min(dot(mFrame.getN(), slice.getWo()), scene_rdl2::math::sOneMinusEpsilon);
        if (cosNO <= scene_rdl2::math::sEpsilon) return scene_rdl2::math::sBlack;

        const float cosMI = dot(m, wi);
        if (cosMI <= scene_rdl2::math::sEpsilon) return scene_rdl2::math::sBlack;

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

        const float solidAngleArea = scene_rdl2::math::sTwoPi * (1.0 - mCosGamma); // must be 2*pi, the Chaos paper has only pi and is incorrect.

        const float D = discreteD * scene_rdl2::math::rcp(solidAngleArea);

        scene_rdl2::math::Color F = computeScaleAndFresnel(cosMI);

        // compute G for CookTorrance
        const float G = computeGeometryTerm(cosNO, cosNI);

        const float invBrdfDenom = scene_rdl2::math::rcp( cosNO * (slice.getIncludeCosineTerm()  ?  1.0f  :  cosNI) );
        scene_rdl2::math::Color flakesBrdf = F*D*G*invBrdfDenom;

        if(pdf != NULL) {
#if UNIFORM_SAMPLING
            *pdf = scene_rdl2::math::sOneOverTwoPi;
#else 
            *pdf = kIntersect * scene_rdl2::math::rcp(solidAngleArea * kTotalSum);
#endif
        }
        // Create a conductor fresnel object using the weighted (reflection cache weights)
        // color of the flakes that contribute to the final bsdf
        // **Note: This is a special case use of the fresnel object inside the bsdf lobe itself -
        // since we know the color to use only after evaluating the reflection cache
        scene_rdl2::math::Color avgWeightedColor = scene_rdl2::math::min(
            scene_rdl2::math::Color(0.999f), 0.5f * (weightedColor1+weightedColor2));
        scene_rdl2::math::Color eta = computeEta(avgWeightedColor, avgWeightedColor);
        scene_rdl2::math::Color k = computeK(avgWeightedColor, eta);
        ConductorFresnel fresnel(eta, k);
        scene_rdl2::math::Color result = flakesBrdf * fresnel.eval(cosMI);
        return result;
    }

    finline bool computeReflectionCache(const scene_rdl2::math::Vec3f& w, FlakeCaches& caches,
                                        bool buildCDF) const
    {
        scene_rdl2::math::Vec3f reflection;
        float weight;

        unsigned int k = 0;
        float totalWeight = 0;
        for(unsigned int i = 0; i < mFlakeCount; ++i) {
            weight = scene_rdl2::math::max(0.0f, computeReflectionDirection(mFlakeNormals[i], w, reflection));
            caches.weights[k] = weight;
            caches.reflections[k] = reflection;
            caches.scales[k] = mFlakeColors[i];
            caches.flakeNormals[k] = mFlakeNormals[i];
            totalWeight += weight;
            k++;
        }

        if (totalWeight > 0) {
            totalWeight = scene_rdl2::math::rcp(totalWeight);
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

    finline float
    computeGeometryTerm(const float cosNO, const float cosNI) const
    {
        const float invCosNO = 1.0f / cosNO;
        const float invCosNI = 1.0f / cosNI;
        const float ao = 1.0f / (mFlakeRandomness * scene_rdl2::math::sqrt((1.0f - cosNO * cosNO) *
                                            (invCosNO * invCosNO)));
        const float ai = 1.0f / (mFlakeRandomness * scene_rdl2::math::sqrt((1.0f - cosNI * cosNI) *
                                            (invCosNI * invCosNI)));
        const float G1o = (ao < 1.6f) ? (3.535f * ao + 2.181f * ao * ao) /
                                (1.0f + 2.276f * ao + 2.577f * ao * ao) : 1.0f;
        const float G1i = (ai < 1.6f) ? (3.535f * ai + 2.181f * ai * ai) /
                                (1.0f + 2.276f * ai + 2.577f * ai * ai) : 1.0f;
        return G1o * G1i;
    }

    // ** Note: The below functions have been duplicated from
    // ShaderComplexIor class in lib/shaders/shading_util/Ior.h
    // and we cannot make the StochasticFlakesBsdfLobe class dependent on a
    // class in lib/shaders/shading_util.
    // Also, they cannot be moved to lib/rendering/pbr because of their dependency
    // on lib/rendering/shading (becomes cyclic).
    // UPDATE: revisit this note, I think this is no longer an issue,
    // can these duplicate functions be removed?

    // Functions for computing complex IOR values for conductor Fresnel
    // from 'reflectivity' and 'edge tint' colors.
    // See paper: "Artist Friendly Metallic Fresnel", by Ole Gulbrandsen
    // from Framestore, published at JCGT in 2014 (http://jcgt.org)
    finline static scene_rdl2::math::Color
    nMin(const scene_rdl2::math::Color &r)
    {
        return (scene_rdl2::math::sWhite - r) / (scene_rdl2::math::sWhite + r);
    }

    finline static scene_rdl2::math::Color
    nMax(const scene_rdl2::math::Color &r)
    {
        scene_rdl2::math::Color rSqrt = sqrt(r);
        return (scene_rdl2::math::sWhite + rSqrt) / (scene_rdl2::math::sWhite - rSqrt);
    }

    finline static scene_rdl2::math::Color
    computeEta(const scene_rdl2::math::Color &r, const scene_rdl2::math::Color &g)
    {
        return g * nMin(r) + (scene_rdl2::math::sWhite - g) * nMax(r);
    }

    finline static scene_rdl2::math::Color
    computeK(const scene_rdl2::math::Color &r, const scene_rdl2::math::Color &n)
    {
        const scene_rdl2::math::Color a = n + scene_rdl2::math::sWhite;
        const scene_rdl2::math::Color b = n - scene_rdl2::math::sWhite;
        // Take an abs() here to get rid of any numerical -0 etc
        const scene_rdl2::math::Color nr = max(scene_rdl2::math::sBlack, r * a * a - b * b);
        return sqrt(nr / (scene_rdl2::math::sWhite - r));
    }
};


//----------------------------------------------------------------------------
//ISPC_UTIL_TYPEDEF_STRUCT(StochasticFlakesBsdfLobe, StochasticFlakesBsdfLobev);

} // namespace shading
} // namespace moonray


