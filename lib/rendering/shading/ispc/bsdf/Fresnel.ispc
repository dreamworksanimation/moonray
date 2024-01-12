// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file Fresnel.ispc

#include "Fresnel.isph"

#include <scene_rdl2/common/platform/IspcUtil.isph>

#pragma ignore warning(all)
ISPC_UTIL_EXPORT_STRUCT_TO_HEADER(Fresnel);
#pragma ignore warning(all)
ISPC_UTIL_EXPORT_STRUCT_TO_HEADER(SchlickFresnel);
#pragma ignore warning(all)
ISPC_UTIL_EXPORT_STRUCT_TO_HEADER(OneMinusRoughSchlickFresnel);
#pragma ignore warning(all)
ISPC_UTIL_EXPORT_STRUCT_TO_HEADER(DielectricFresnel);
#pragma ignore warning(all)
ISPC_UTIL_EXPORT_STRUCT_TO_HEADER(ConductorFresnel);
#pragma ignore warning(all)
ISPC_UTIL_EXPORT_STRUCT_TO_HEADER(OneMinusFresnel);
#pragma ignore warning(all)
ISPC_UTIL_EXPORT_STRUCT_TO_HEADER(OneMinusVelvetFresnel);
#pragma ignore warning(all)
ISPC_UTIL_EXPORT_STRUCT_TO_HEADER(OneMinusRoughFresnel);
#pragma ignore warning(all)
ISPC_UTIL_EXPORT_STRUCT_TO_HEADER(MultipleTransmissionFresnel);
#pragma ignore warning(all)
ISPC_UTIL_EXPORT_ENUM_TO_HEADER(FresnelType);


float
DielectricFresnel_evalPerp(float etaI,
                           float etaT,
                           float cosThetaI,
                           float cosThetaT)
{
    const float etaIcosThetaI = etaI * cosThetaI;
    const float etaTcosThetaT = etaT  * cosThetaT;
    const float perp = ((etaIcosThetaI) - (etaTcosThetaT)) *
        rcp((etaIcosThetaI) + (etaTcosThetaT));
    return (perp*perp);
}

float
DielectricFresnel_evalParallel(float etaI,
                               float etaT,
                               float cosThetaI,
                               float cosThetaT)
{
    const float etaTcosThetaI = etaT * cosThetaI;
    const float etaIcosThetaT = etaI * cosThetaT;
    const float parallel = ((etaTcosThetaI) - (etaIcosThetaT)) *
            rcp((etaTcosThetaI) + (etaIcosThetaT));
    return (parallel*parallel);
}


// Reflectance from a single layer
// Eqn (6)
float
LayeredDielectricFresnel_evalSingleLayer(float etaI,
                                         float etaPerp,
                                         float etaParallel,
                                         float cosThetaI,
                                         float cosThetaT)
{
    const float Fs = DielectricFresnel_evalPerp(etaI, etaPerp,
                                                cosThetaI, cosThetaT);
    const float Fp = DielectricFresnel_evalParallel(etaI, etaParallel,
                                                    cosThetaI, cosThetaT);

    const float t1 = Fs + (1.0f - Fs)*(1.0f - Fs) * Fs / (1.0f - Fs*Fs);
    const float t2 = Fp + (1.0f - Fp)*(1.0f - Fp) * Fp / (1.0f - Fp*Fp);

    return (0.5f*t1 + 0.5f*t2);
}

varying Color
LayeredDielectricFresnel_evalFresnel(const varying float etaI,
                                     const varying float etaT,
                                     const varying float etaPerp,
                                     const varying float etaParallel,
                                     const varying float hDotI,
                                     const varying float numLayers)
{
    // Make sure we don't nan out down below
    float cosThetaI = max(hDotI, -1.0f);
    cosThetaI = min(cosThetaI, 1.0f);

    // Compute Snell law
    const float eta = 1.0f * rcp(etaT);
    const float sinThetaTSqr = eta * eta * (1.0f - cosThetaI*cosThetaI);

    if (sinThetaTSqr >= 1.0f) {
        // Total internal reflection
        return sWhite;
    }

    const float cosThetaT = sqrt(1.0f - sinThetaTSqr);
    cosThetaI = abs(cosThetaI);

    // Eqn (6)
    const float F = LayeredDielectricFresnel_evalSingleLayer(etaI,
                                                             etaPerp,
                                                             etaParallel,
                                                             cosThetaI,
                                                             cosThetaT);

    // Eqn (7)
    const float n = numLayers * F;
    const float d = 1.0f + (numLayers - 1.0f) * F;

    const float layeredRefl = clamp(n/d, 0.0f, 1.0f);
    return Color_ctor(layeredRefl);
}

/// @brief LayeredDielectricFresnel Fresnel evaluation
varying Color
LayeredDielectricFresnel_eval(const varying Fresnel * uniform fresnel,
                              varying float cosThetaI)
{
    const varying LayeredDielectricFresnel * uniform lyrF =
        (const varying LayeredDielectricFresnel * uniform) fresnel;

    return lyrF->mWeight*LayeredDielectricFresnel_evalFresnel(
            lyrF->mEtaI, lyrF->mEtaT,
            lyrF->mEtaT, lyrF->mEtaT,
            cosThetaI,
            lyrF->mNumLayers);
}


