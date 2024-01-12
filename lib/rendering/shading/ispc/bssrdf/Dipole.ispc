// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file Dipole.ispc

#include "Dipole.isph"
#include <moonray/rendering/shading/ispc/PbrValidity.isph>

#include <scene_rdl2/common/math/ispc/asA.isph>
#include <scene_rdl2/common/platform/IspcUtil.isph>


#define BSSRDF_SAMPLE_UNIFORM 0
#define BSSRDF_SAMPLE_WIDEST_TAIL 0


static void finishInit(varying DipoleBssrdf * uniform bssrdf);
static varying float findAlphaP(varying float reflectance, varying float A);


//----------------------------------------------------------------------------

inline varying float
totalDiffuseReflectance(varying float alphaP, varying float A)
{
    float e = sqrt(3.0f * (1.0f - alphaP));
    return 0.5f * alphaP * (1.0f + exp(-4.0f / 3.0f * A * e)) * exp(-e);
}

inline varying Color
mmInvToWorldInv(const varying Color &x, float sceneScale)
{
    // One unit in world space == sceneScale meters
    // Formally:
    //     world space unit * sceneScale = meter
    //     world space unit * (1000 * sceneScale) = mm
    //     mm-1 * (1000 * sceneScale) = world-unit-1
    return x * (1000.0f * sceneScale);
}

void
DipoleBssrdf_show(const varying Bssrdf * const uniform sss,
                  const uniform uint8_t ntab)
{
    const varying DipoleBssrdf * uniform me =
        (const varying DipoleBssrdf * uniform) sss;

    const varying Vec3f N = getN(me->mFrame);
    const varying Fresnel * const uniform fresnel = me->mFresnel;

    BsdfUtil_indent(ntab);   print("[DipoleBssrdf] mask: %\n", sss->mMask);
    BsdfUtil_indent(ntab+1); print("scale: % % %\n", BSDF_UTIL_EXTRACT(me->mScale.r),
                                                BSDF_UTIL_EXTRACT(me->mScale.g),
                                                BSDF_UTIL_EXTRACT(me->mScale.b));
    BsdfUtil_indent(ntab+1); print("N: % % %\n", BSDF_UTIL_EXTRACT(N.x),
                                            BSDF_UTIL_EXTRACT(N.y),
                                            BSDF_UTIL_EXTRACT(N.z));
    BsdfUtil_indent(ntab+1); print("albedo: % % %\n", BSDF_UTIL_EXTRACT(me->mAlbedo.r),
                                                 BSDF_UTIL_EXTRACT(me->mAlbedo.g),
                                                 BSDF_UTIL_EXTRACT(me->mAlbedo.b));
    if (fresnel) {
        Fresnel_show(fresnel, ntab+1);
    }
}

// FIXME: This constructor does not support PBR validity for now
// (note that BSSRDF_PROPERTY_PBR_VALIDITY isn't being initialized in the Bssrdf object).
// In order to support pbr validity, albedo would need to be computed from the inputs
void
DipoleBssrdf_init(varying DipoleBssrdf * uniform bssrdf,
            const varying Vec3f &N, varying float eta, const varying Color &sigmaA,
            const varying Color &sigmaSP, varying float sceneScale,
            const uniform Material *uniform  material,
            const uniform intptr_t evalNormalFn)
{
    // Initialize base class
    Bssrdf_init((varying Bssrdf * uniform)bssrdf, DipoleBssrdf_eval,
            DipoleBssrdf_sampleLocal, DipoleBssrdf_pdfLocal, DipoleBssrdf_getProperty, DipoleBssrdf_show,
            SUBSURFACE_DIPOLE_DIFFUSION, N, SUBSURFACE_PROPERTY_COLOR | SUBSURFACE_PROPERTY_RADIUS,
            material, evalNormalFn);

    ReferenceFrame_init(bssrdf->mFrame, N);

    bssrdf->mSigmaA = mmInvToWorldInv(sigmaA, sceneScale);
    bssrdf->mSigmaSP = mmInvToWorldInv(sigmaSP, sceneScale);

    float fdr = diffuseFresnelReflectance(eta);
    bssrdf->mA = (1.0f + fdr) * rcp(1.0f - fdr);

    bssrdf->mSigmaTP = bssrdf->mSigmaA + bssrdf->mSigmaSP;
    bssrdf->mSigmaTr = sqrt(3.0f * bssrdf->mSigmaA * bssrdf->mSigmaTP);
    bssrdf->mAlphaP = bssrdf->mSigmaSP * rcp(bssrdf->mSigmaTP);

    // Keep a note of the total diffuse reflectance
    bssrdf->mDiffuseReflectance = Color_ctor(totalDiffuseReflectance(bssrdf->mAlphaP.r, bssrdf->mA),
                                             totalDiffuseReflectance(bssrdf->mAlphaP.g, bssrdf->mA),
                                             totalDiffuseReflectance(bssrdf->mAlphaP.b, bssrdf->mA));

    bssrdf->mAlbedo = sBlack;

    finishInit(bssrdf);
}


void
DipoleBssrdf_init(varying DipoleBssrdf * uniform bssrdf,
            const varying Vec3f &N,
            const varying Color &translucentColor, const varying Color &radius,
            const uniform Material *uniform  material,
            const uniform intptr_t evalNormalFn)
{
    // Initialize base class
    Bssrdf_init((varying Bssrdf * uniform)bssrdf, DipoleBssrdf_eval,
            DipoleBssrdf_sampleLocal, DipoleBssrdf_pdfLocal, DipoleBssrdf_getProperty, DipoleBssrdf_show,
            SUBSURFACE_DIPOLE_DIFFUSION, N, SUBSURFACE_PROPERTY_COLOR | SUBSURFACE_PROPERTY_RADIUS |
            SUBSURFACE_PROPERTY_PBR_VALIDITY, material, evalNormalFn);

    ReferenceFrame_init(bssrdf->mFrame, N);

    float fdr = diffuseFresnelReflectance(1.0f);
    bssrdf->mA = (1.0f + fdr) / (1.0f - fdr);

    // Convert from artist controls to physical controls as described in [3]
    const varying Color nearBlack = Color_ctor(0.001f, 0.001f, 0.001f);
    Color r = max(radius, nearBlack);
    bssrdf->mSigmaTr = rcp(r);
    for (uniform int c = 0; c < 3; ++c) {
        float reflectance = max((asArray(translucentColor))[c], 0.0f);
        reflectance = min(reflectance, 1.0f);
        (asArray(bssrdf->mAlphaP))[c] = findAlphaP(reflectance, bssrdf->mA);
    }

    bssrdf->mDiffuseReflectance = Color_ctor(totalDiffuseReflectance(bssrdf->mAlphaP.r, bssrdf->mA),
                                             totalDiffuseReflectance(bssrdf->mAlphaP.g, bssrdf->mA),
                                             totalDiffuseReflectance(bssrdf->mAlphaP.b, bssrdf->mA));

    bssrdf->mSigmaTP = bssrdf->mSigmaTr * rcp(sqrt(3.0f * (Color_ctor(1.f) - bssrdf->mAlphaP)));
    bssrdf->mSigmaSP = bssrdf->mAlphaP * bssrdf->mSigmaTP;
    bssrdf->mSigmaA = bssrdf->mSigmaTP - bssrdf->mSigmaSP;

    bssrdf->mAlbedo = translucentColor;

    finishInit(bssrdf);
}


static void
finishInit(varying DipoleBssrdf * uniform bssrdf)
{
    bssrdf->mZr = rcp(bssrdf->mSigmaTP);
    bssrdf->mZv = bssrdf->mZr * (1.0f + (4.0f / 3.0f) * bssrdf->mA);

    // Evaluate Rd at the mean free path radius. This corresponds to the radius
    // passed in the second constructor, which is also 1 / sigmaTr
    Color mfp = Color_ctor(rcp(bssrdf->mSigmaTr.r),
                           rcp(bssrdf->mSigmaTr.g),
                           rcp(bssrdf->mSigmaTr.b));
    float maxMfp = max(mfp.r, mfp.g);
    maxMfp = max(maxMfp, mfp.b);

    // At 3x the mean free path Rd is very small.
    bssrdf->mMaxRadius = 3.0f * maxMfp;

#if !BSSRDF_SAMPLE_UNIFORM
    // Compute sampling ratio and pdf normalization terms
    float normalize = 0.0f;
    for (uniform int c = 0; c < 3; ++c) {
        float sigmaTr = (asArray(bssrdf->mSigmaTr))[c];
        float expTermR = exp(-sigmaTr * (asArray(bssrdf->mZr))[c]);
        float expTermV = exp(-sigmaTr * (asArray(bssrdf->mZv))[c]);
        float sumExpTerms = expTermR + expTermV;
        float reflectance = (asArray(bssrdf->mAlphaP))[c] * 0.5f * sumExpTerms;
        (asArray(bssrdf->mRatio))[c] = expTermR * rcp(sumExpTerms);
        (asArray(bssrdf->mPdfNormFactor))[c] = rcp(reflectance);
        (asArray(bssrdf->mChannelCdf))[c] = max(reflectance, sEpsilon);
        normalize += (asArray(bssrdf->mChannelCdf))[c];
    }

#if BSSRDF_SAMPLE_WIDEST_TAIL
    // The highest Rd value at maxMfp has the widest tail
    Color rd = DipoleBssrdf_eval(bssrdf, maxMfp);
    normalize = 1.0f;
    if (rd[0] >= rd[1]) {
        if (rd[0] >= rd[2]) {
            bssrdf->mChannelCdf = Color_ctor(1.0f, 0.0f, 0.0f);
        } else {
            bssrdf->mChannelCdf = Color_ctor(0.0f, 0.0f, 1.0f);
        }
    } else {
        if (rd[1] >= rd[2]) {
            bssrdf->mChannelCdf = Color_ctor(0.0f, 1.0f, 0.0f);
        } else {
            bssrdf->mChannelCdf = Color_ctor(0.0f, 0.0f, 1.0f);
        }
    }
#endif

    // Normalize and integrate the cdf, all in one pass
    normalize = rcp(normalize);
    float cdf = 0.0f;
    for (uniform int c = 0; c < 3; ++c) {
        (asArray(bssrdf->mChannelCdf))[c] *= normalize;
        (asArray(bssrdf->mChannelCdf))[c] += cdf;
        cdf = (asArray(bssrdf->mChannelCdf))[c];
    }
    MNRY_ASSERT(isEqual(cdf, 1.f));
#endif
}


//----------------------------------------------------------------------------

static varying float
findAlphaP(varying float reflectance, varying float A)
{
    // TODO: Use lookup table as in Christophe's sketch
    float low = 0.0f;
    float high = 1.0f;

#ifdef DEBUG
    float kdLow = totalDiffuseReflectance(low, A);
    float kdHigh = totalDiffuseReflectance(high, A); 
#endif

    for (uniform int i = 0; i < 16; ++i) {
#ifdef DEBUG
        MNRY_ASSERT(kdLow <= reflectance  &&  kdHigh >= reflectance);
#endif
        float mid = (low + high) * 0.5f;
        float kd = totalDiffuseReflectance(mid, A);
        if (kd < reflectance) {
            low = mid;
#ifdef DEBUG
            kdLow = kd;
#endif
        } else {
            high = mid;
#ifdef DEBUG
            kdHigh = kd;
#endif
        }
    }
    return (low + high) * 0.5f;
}


//----------------------------------------------------------------------------

// Careful, the formula in [1] is incorrect. See the other referenced papers,
// which all have the corrected formula.
inline varying float
Rd(varying float r2, varying float zr, varying float zv, varying float alphaP, varying float sigmaTr)
{
    float dr = sqrt(r2 + zr * zr);
    float dv = sqrt(r2 + zv * zv);
    float rd = (alphaP * sOneOverFourPi) *
        ((zr * (dr * sigmaTr + 1.0f) * exp(-sigmaTr * dr)) / (dr * dr * dr) +
         (zv * (dv * sigmaTr + 1.0f) * exp(-sigmaTr * dv)) / (dv * dv * dv));

    // return rd < 0.0f  ?  0.0f  :  rd;
    return rd;
}


varying Color
DipoleBssrdf_eval(const varying Bssrdf * uniform bss, varying float r,
        uniform bool global)
{
    const varying DipoleBssrdf * uniform bssrdf = (const varying DipoleBssrdf * uniform)bss;

    float r2 = r * r;
    Color rd;
    for (int c = 0; c < 3; ++c) {
        (asArray(rd))[c] = Rd(r2, (asArray(bssrdf->mZr))[c],
                              (asArray(bssrdf->mZv))[c],
                              (asArray(bssrdf->mAlphaP))[c],
                              (asArray(bssrdf->mSigmaTr))[c]);
    }

    return rd;
};


//----------------------------------------------------------------------------

// Newton solve for u (see [2] for details)
static varying float
solveU(varying float r1, varying float sigmaZ)
{
    float logfR1Term = log(1.0f - r1);
    float u = 1.0f;

    static const uniform int maxSteps = 10;
    int step = 0;
    while (step < maxSteps) {
        float f = sigmaZ * (u - 1) + log(u) + logfR1Term;
        step = (abs(f) < 0.001f  ?  maxSteps  :  step + 1);
        // TODO: Why do we hit this assert ?
        MNRY_ASSERT(u > sEpsilon);
        u -= f * rcp(sigmaZ + 1.0f * rcp(u));
    }

    return u;
}

#if !BSSRDF_SAMPLE_UNIFORM

varying float
DipoleBssrdf_sampleLocal(const varying Bssrdf * uniform bss,
                   varying float r1, varying float r2, varying Vec3f &dPi,
                   varying float &r)
{
    const varying DipoleBssrdf * uniform bssrdf = (const varying DipoleBssrdf * uniform)bss;

    // Importance sample the channel and remap r2
    int c = (r2 < (asArray(bssrdf->mChannelCdf))[0]  ?  0  :
             (r2 < (asArray(bssrdf->mChannelCdf))[1]  ?  1  :  2));
    r2 = (c == 0  ?  r2 * rcp((asArray(bssrdf->mChannelCdf))[0])  :
          (r2 - (asArray(bssrdf->mChannelCdf))[c-1]) *
          rcp(((asArray(bssrdf->mChannelCdf))[c] - (asArray(bssrdf->mChannelCdf))[c-1])));


    // Pick real or virtual source and remap r1
    float z;
    if (r1 < (asArray(bssrdf->mRatio))[c]) {
        z = (asArray(bssrdf->mZr))[c];
        r1 *= rcp((asArray(bssrdf->mRatio))[c]);
    } else {
        z = (asArray(bssrdf->mZv))[c];
        r1 = (r1 - (asArray(bssrdf->mRatio))[c]) * rcp((1.0f - (asArray(bssrdf->mRatio))[c]));
    }

    // Solve for u using r1
    float u = solveU(r1, (asArray(bssrdf->mSigmaTr))[c] * z);

    // Compute radius r
    r = z * sqrt(u * u - 1);

    // Note because the above two operations correspond to statistical branching
    // it is better to use the remapped r1 and r2 to preserve a good discrepancy of the
    // samples drawn from each distribution out of the 2*3=6 possible
    // distributions.
    // TODO: Try using a good quality 3D sampler instead that down-projects well
    // to lower dimensions.
    float phi = r2 * sTwoPi;

    // Compute position
    float sinPhi;
    float cosPhi;
    sincos(phi, &sinPhi, &cosPhi);
    dPi.x = r * cosPhi;
    dPi.y = r * sinPhi;
    dPi.z = 0.0f;

    return DipoleBssrdf_pdfLocal(bss, r);
}


varying float
DipoleBssrdf_pdfLocal(const varying Bssrdf * uniform bss, varying float r)
{
    const varying DipoleBssrdf * uniform bssrdf = (const varying DipoleBssrdf * uniform)bss;

    // We compute the estimator with MIS, under Veach's one-sample model.
    // See paper: "Optimally combining sampling techniques for MC rendering"
    // Here we use estimator (16) and therefore: pdf = ci * pi / wi
    // With the balance heuristic it amounts to the weighted average of the
    // pdf for each channel: pdf = sum(ci * pi)
    float pdf = 0.0f;
    for (uniform int i = 0; i < 3; ++i) {
        // Compute ci, the discrete probability of picking this channel
        float ci = (i == 0  ?  (asArray(bssrdf->mChannelCdf))[i]  :
                    (asArray(bssrdf->mChannelCdf))[i] - (asArray(bssrdf->mChannelCdf))[i-1]);

        // Accumulate ci * pi, where pi is the pdf of sampling dPi
        pdf += (ci < sEpsilon  ?  0.0f  :
                ci * (asArray(bssrdf->mPdfNormFactor))[i] *
                Rd(r * r, (asArray(bssrdf->mZr))[i], (asArray(bssrdf->mZv))[i],
                   (asArray(bssrdf->mAlphaP))[i], (asArray(bssrdf->mSigmaTr))[i]));
    }

    return pdf;
}

varying Color
DipoleBssrdf_diffuseReflectance(const varying Bssrdf * uniform bss)
{
    const varying DipoleBssrdf * uniform bssrdf =
            (const varying DipoleBssrdf * uniform)bss;

    Color dR;
    dR.r = totalDiffuseReflectance(bssrdf->mAlphaP.r, bssrdf->mA);
    dR.g = totalDiffuseReflectance(bssrdf->mAlphaP.g, bssrdf->mA);
    dR.b = totalDiffuseReflectance(bssrdf->mAlphaP.b, bssrdf->mA);

    return dR;
}

#else

// Uniform sampling
varying float
DipoleBssrdf_sampleLocal(const varying Bssrdf * uniform bss,
                   varying float r1, varying float r2, varying Vec3f &dPi,
                   varying float &r)
{
    const varying DipoleBssrdf * uniform bssrdf = (const varying DipoleBssrdf * uniform)bs;

    Vec2f p;
    p.x = (r1 - 0.5f) * 2.0f * bssrdf->mMaxRadius;
    p.y = (r2 - 0.5f) * 2.0f * bssrdf->mMaxRadius;
    r = p.length();

    // Compute position
    dPi = p.x * bssrdf->mFrame.mX + p.y * bssrdf->mFrame.mY;

    float pdf = rcp(4.0f * bssrdf->mMaxRadius * bssrdf->mMaxRadius);
    return pdf;
}


varying float
DipoleBssrdf_pdfLocal(const varying Bssrdf * uniform bss, varying float r)
{
    const varying DipoleBssrdf * uniform bssrdf = (const varying DipoleBssrdf * uniform)bs;

    float pdf = rcp(4.0f * mMaxRadius * mMaxRadius);
    return pdf;
}


#endif

uniform bool
DipoleBssrdf_getProperty(const varying Bssrdf * uniform bss,
                         uniform SubsurfaceProperty property,
                         varying float * uniform dest)
{
    const varying DipoleBssrdf * uniform bssrdf = (const varying DipoleBssrdf * uniform) bss;

    uniform bool handled = true;

    switch (property)
    {
    case SUBSURFACE_PROPERTY_COLOR:
        *dest       = totalDiffuseReflectance(bssrdf->mAlphaP.r, bssrdf->mA);
        *(dest + 1) = totalDiffuseReflectance(bssrdf->mAlphaP.g, bssrdf->mA);
        *(dest + 2) = totalDiffuseReflectance(bssrdf->mAlphaP.b, bssrdf->mA);
        break;
    case SUBSURFACE_PROPERTY_RADIUS:
        *dest       = rcp(bssrdf->mSigmaTr.r);
        *(dest + 1) = rcp(bssrdf->mSigmaTr.g);
        *(dest + 2) = rcp(bssrdf->mSigmaTr.b);
        break;
    case SUBSURFACE_PROPERTY_PBR_VALIDITY:
        {
            Color res = computeAlbedoPbrValidity(bssrdf->mAlbedo);
            *dest       = res.r;
            *(dest + 1) = res.g;
            *(dest + 2) = res.b;
        }
        break;
    default:
        handled = false;
    }

    return handled;

}

//----------------------------------------------------------------------------

#pragma ignore warning(all)
ISPC_UTIL_EXPORT_STRUCT_TO_HEADER(DipoleBssrdf);

