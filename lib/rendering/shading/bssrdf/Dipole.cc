// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
///

#include "Dipole.h"

#include <moonray/rendering/shading/bsdf/Fresnel.h>
#include <moonray/rendering/shading/PbrValidity.h>
#include <moonray/rendering/shading/ispc/Shadingv.h>
#include <moonray/rendering/shading/Util.h>

#include <scene_rdl2/common/math/Color.h>
#include <scene_rdl2/common/math/MathUtil.h>


using namespace scene_rdl2::math;

namespace moonray {
namespace shading {


#define BSSRDF_SAMPLE_UNIFORM 0
#define BSSRDF_SAMPLE_WIDEST_TAIL 0

static finline float
totalDiffuseReflectance(float alphaP, float A)
{
    const float e = scene_rdl2::math::sqrt(3.0f * (1.0f - alphaP));
    return 0.5f * alphaP * (1.0f + scene_rdl2::math::exp(-4.0f / 3.0f * A * e)) * scene_rdl2::math::exp(-e);
}

static float findAlphaP(float reflectance, float A);

namespace {

inline Color
mmInvToWorldInv(Color x, float sceneScale)
{
    // One unit in world space == sceneScale meters
    // Formally:
    //     world space unit * sceneScale = meter
    //     world space unit * (1000 * sceneScale) = mm
    //     mm-1 * (1000 * sceneScale) = world-unit-1
    return x * (1000.0f * sceneScale);
}

inline Color
colorvToColor(const Colorv &col, int lane)
{
    return Color(col.r[lane], col.g[lane], col.b[lane]);
}

}   // End of anon namespace.

//----------------------------------------------------------------------------
// FIXME: This constructor does not support PBR validity for now
// (note that PROPERTY_PBR_VALIDITY isn't being initialized in the Bssrdf object).
// In order to support pbr validity, albedo would need to be computed from the inputs
DipoleBssrdf::DipoleBssrdf(const Vec3f &N,
                           float eta,
                           const Color &sigmaA,
                           const Color &sigmaSP,
                           float sceneScale,
                           const scene_rdl2::rdl2::Material* material,
                           const scene_rdl2::rdl2::EvalNormalFunc evalNormalFn) :
    Bssrdf(N, PROPERTY_COLOR | PROPERTY_RADIUS, material, evalNormalFn),
    mSigmaA(mmInvToWorldInv(sigmaA, sceneScale)),
    mSigmaSP(mmInvToWorldInv(sigmaSP, sceneScale)),
    mAlbedo(scene_rdl2::math::sBlack)
{
    float fdr = diffuseFresnelReflectance(eta);
    mA = (1.0f + fdr) / (1.0f - fdr);

    mSigmaTP = mSigmaA + mSigmaSP;
    mSigmaTr = scene_rdl2::math::sqrt(3.0f * mSigmaA * mSigmaTP);
    mAlphaP = mSigmaSP / mSigmaTP;

    // Keep a note of the total diffuse reflectance
    mDiffuseReflectance = Color(totalDiffuseReflectance(mAlphaP[0], mA),
                                totalDiffuseReflectance(mAlphaP[1], mA),
                                totalDiffuseReflectance(mAlphaP[2], mA));
    finishInit();
}

DipoleBssrdf::DipoleBssrdf(const Vec3f &N,
                           float eta,
                           const Color &translucentColor,
                           const Color &radius,
                           const scene_rdl2::rdl2::Material* material,
                           const scene_rdl2::rdl2::EvalNormalFunc evalNormalFn) :
    Bssrdf(N, PROPERTY_COLOR | PROPERTY_RADIUS | PROPERTY_PBR_VALIDITY, material, evalNormalFn),
    mAlbedo(translucentColor)
{
    float fdr = diffuseFresnelReflectance(eta);
    mA = (1.0f + fdr) / (1.0f - fdr);

    // Convert from artist controls to physical controls as described in [3]
    Color r = max(radius, Color(0.001f, 0.001f, 0.001f));
    mSigmaTr = rcp(r);
    for (int c = 0; c < 3; ++c) {
        float reflectance = max(translucentColor[c], 0.0f);
        reflectance = min(reflectance, 1.0f);
        mAlphaP[c] = findAlphaP(reflectance, mA);
    }

    mDiffuseReflectance = Color(totalDiffuseReflectance(mAlphaP[0], mA),
                                totalDiffuseReflectance(mAlphaP[1], mA),
                                totalDiffuseReflectance(mAlphaP[2], mA));

    mSigmaTP = mSigmaTr / scene_rdl2::math::sqrt(3.0f * (sWhite - mAlphaP));
    mSigmaSP = mAlphaP * mSigmaTP;
    mSigmaA = mSigmaTP - mSigmaSP;

    finishInit();
}

DipoleBssrdf::DipoleBssrdf(scene_rdl2::alloc::Arena *arena, const DipoleBssrdfv &bssrdfv, int lane)
  : Bssrdf(arena, (const Bssrdfv &)bssrdfv, lane),
    mSigmaA(colorvToColor(bssrdfv.mSigmaA, lane)),
    mSigmaSP(colorvToColor(bssrdfv.mSigmaSP, lane)),
    mA(bssrdfv.mA[lane]),
    mSigmaTP(colorvToColor(bssrdfv.mSigmaTP, lane)),
    mSigmaTr(colorvToColor(bssrdfv.mSigmaTr, lane)),
    mAlphaP(colorvToColor(bssrdfv.mAlphaP, lane)),
    mZr(colorvToColor(bssrdfv.mZr, lane)),
    mZv(colorvToColor(bssrdfv.mZv, lane)),
    mChannelCdf(colorvToColor(bssrdfv.mChannelCdf, lane)),
    mRatio(colorvToColor(bssrdfv.mRatio, lane)),
    mPdfNormFactor(colorvToColor(bssrdfv.mPdfNormFactor, lane)),
    mDiffuseReflectance(colorvToColor(bssrdfv.mDiffuseReflectance, lane))
{
}

void
DipoleBssrdf::finishInit()
{
    mZr = rcp(mSigmaTP);
    mZv = mZr * (1.0f + (4.0f / 3.0f) * mA);

    // Evaluate Rd at the mean free path radius
    Color mfp = getDiffuseMeanFreePath();
    float maxMfp = max(mfp[0], mfp[1]);
    maxMfp = max(maxMfp, mfp[2]);

    // At 3x the mean free path Rd is very small.
    mMaxRadius = 3.0f * maxMfp;

#if !BSSRDF_SAMPLE_UNIFORM
    // Compute sampling ratio and pdf normalization terms
    float normalize = 0.0f;
    for (int c=0; c < 3; c++) {
        float sigmaTr = mSigmaTr[c];
        float expTermR = scene_rdl2::math::exp(-sigmaTr * mZr[c]);
        float expTermV = scene_rdl2::math::exp(-sigmaTr * mZv[c]);
        float sumExpTerms = expTermR + expTermV;
        float reflectance = mAlphaP[c] * 0.5f * sumExpTerms;
        mRatio[c] = expTermR / sumExpTerms;
        mPdfNormFactor[c] = 1.0f / reflectance;
        mChannelCdf[c] = max(reflectance, sEpsilon);
        normalize += mChannelCdf[c];
    }

#if BSSRDF_SAMPLE_WIDEST_TAIL
    // The highest Rd value at maxMfp has the widest tail
    Color rd = eval(maxMfp);
    normalize = 1.0f;
    if (rd[0] >= rd[1]) {
        if (rd[0] >= rd[2]) {
            mChannelCdf = Color(1.0f, 0.0f, 0.0f);
        } else {
            mChannelCdf = Color(0.0f, 0.0f, 1.0f);
        }
    } else {
        if (rd[1] >= rd[2]) {
            mChannelCdf = Color(0.0f, 1.0f, 0.0f);
        } else {
            mChannelCdf = Color(0.0f, 0.0f, 1.0f);
        }
    }
#endif

    // Normalize and integrate the cdf, all in one pass
    normalize = 1.0f / normalize;
    float cdf = 0.0f;
    for (int c=0; c < 3; c++) {
        mChannelCdf[c] *= normalize;
        mChannelCdf[c] += cdf;
        cdf = mChannelCdf[c];
    }
    MNRY_ASSERT(isOne(cdf));
#endif
}


//----------------------------------------------------------------------------

static float
findAlphaP(float reflectance, float A)
{
    // TODO: Use lookup table as in Christophe's sketch
    float low = 0.0f;
    float high = 1.0f;

    MNRY_DURING_ASSERTS(
        float kdLow = totalDiffuseReflectance(low, A);
        float kdHigh = totalDiffuseReflectance(high, A);
    );

    for (int i = 0; i < 16; i++) {
        MNRY_ASSERT(kdLow <= reflectance  &&  kdHigh >= reflectance);
        const float mid = (low + high) * 0.5f;
        const float kd = totalDiffuseReflectance(mid, A);
        if (kd < reflectance) {
            low = mid;
            MNRY_DURING_ASSERTS(kdLow = kd);
        } else {
            high = mid;
            MNRY_DURING_ASSERTS(kdHigh = kd);
        }
    }
    return (low + high) * 0.5f;
}


//----------------------------------------------------------------------------

// Careful, the formula in [1] is incorrect. See the other referenced papers,
// which all have the corrected formula.
static finline float
Rd(float r2, float zr, float zv, float alphaP, float sigmaTr)
{
    const float dr = scene_rdl2::math::sqrt(r2 + zr * zr);
    const float dv = scene_rdl2::math::sqrt(r2 + zv * zv);
    const float rd = (alphaP * sOneOverFourPi) *
        ((zr * (dr * sigmaTr + 1.0f) * scene_rdl2::math::exp(-sigmaTr * dr)) / (dr * dr * dr) +
         (zv * (dv * sigmaTr + 1.0f) * scene_rdl2::math::exp(-sigmaTr * dv)) / (dv * dv * dv));

    // return rd < 0.0f  ?  0.0f  :  rd;
    return rd;
}


Color
DipoleBssrdf::eval(float r, bool /* global */) const
{
    const float r2 = r * r;
    Color rd;
    for (int c=0; c < 3; c++) {
        rd[c] = Rd(r2, mZr[c], mZv[c], mAlphaP[c], mSigmaTr[c]);
    }

    return rd;
}


//----------------------------------------------------------------------------

// Newton solve for u (see [2] for details)
static float
solveU(float r1, float sigmaZ)
{
    float logfR1Term = scene_rdl2::math::log(1.0f - r1);
    float u = 1.0f;

    static const int maxSteps = 10;
    int step = 0;
    while (step < maxSteps) {
        float f = sigmaZ * (u - 1) + scene_rdl2::math::log(u) + logfR1Term;
        step = (scene_rdl2::math::abs(f) < 0.001f  ?  maxSteps  :  step + 1);
        // TODO: Why do we hit this assert ?
        MNRY_ASSERT(u > sEpsilon);
        u -= f / (sigmaZ + 1.0f / u);
    }

    return u;
}


#if !BSSRDF_SAMPLE_UNIFORM

float
DipoleBssrdf::sampleLocal(float r1, float r2, Vec3f &dPi, float &r) const
{
    // Importance sample the channel and remap r2
    int c = (r2 < mChannelCdf[0]  ?  0  :
            (r2 < mChannelCdf[1]  ?  1  :  2));
    r2 = (c == 0  ?  r2 / mChannelCdf[0]  :
            (r2 - mChannelCdf[c-1]) / (mChannelCdf[c] - mChannelCdf[c-1]));

//    if (c != 0) {
//        dPi = Vec3f(zero);
//        r = 0.0f;
//        return 0.0f;
//    }

    // Pick real or virtual source and remap r1
    float z;
    if (r1 < mRatio[c]) {
        z = mZr[c];
        r1 /= mRatio[c];
    } else {
        z = mZv[c];
        r1 = (r1 - mRatio[c]) / (1.0f - mRatio[c]);
    }

    // Solve for u using r1
    float u = solveU(r1, mSigmaTr[c] * z);

    // Compute radius r
    r = z * scene_rdl2::math::sqrt(u * u - 1);

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

    dPi[0] = r * cosPhi;
    dPi[1] = r * sinPhi;
    dPi[2] = 0.0f;

    return pdfLocal(r);
}


float
DipoleBssrdf::pdfLocal(float r) const
{
    // We compute the estimator with MIS, under Veach's one-sample model.
    // See paper: "Optimally combining sampling techniques for MC rendering"
    // Here we use estimator (16) and therefore: pdf = ci * pi / wi
    // With the balance heuristic it amounts to the weighted average of the
    // pdf for each channel: pdf = sum(ci * pi)
    float pdf = 0.0f;
    for (int i = 0; i < 3; i++) {
        // Compute ci, the discrete probability of picking this channel
        float ci = (i == 0  ?  mChannelCdf[i]  :
                               mChannelCdf[i] - mChannelCdf[i-1]);

        // Accumulate ci * pi, where pi is the pdf of sampling dPi
        pdf += (ci < sEpsilon  ?  0.0f  :
                    ci * mPdfNormFactor[i] *
                    Rd(r * r, mZr[i], mZv[i], mAlphaP[i], mSigmaTr[i]));
    }

    return pdf;
}

#else

// Uniform sampling
float
DipoleBssrdf::sampleLocal(float r1, float r2, Vec3f &dPi, float &r) const
{
    dPi[0] = (r1 - 0.5f) * 2.0f * mMaxRadius;
    dPi[1] = (r2 - 0.5f) * 2.0f * mMaxRadius;
    dPi[2] = 0.0f;
    r = dPi.length();

    float pdf = 1.0f / (4.0f * mMaxRadius * mMaxRadius);
    return pdf;
}


float
DipoleBssrdf::pdfLocal(float r) const
{
    float pdf = 1.0f / (4.0f * mMaxRadius * mMaxRadius);
    return pdf;
}


#endif

bool
DipoleBssrdf::getProperty(Property property, float *dest) const
{
    bool handled = true;

    switch (property)
    {
    case PROPERTY_COLOR:
        *dest       = totalDiffuseReflectance(mAlphaP[0], mA);
        *(dest + 1) = totalDiffuseReflectance(mAlphaP[1], mA);
        *(dest + 2) = totalDiffuseReflectance(mAlphaP[2], mA);
        break;
    case PROPERTY_RADIUS:
        *dest       = 1.f / mSigmaTr[0];
        *(dest + 1) = 1.f / mSigmaTr[1];
        *(dest + 2) = 1.f / mSigmaTr[2];
        break;
    case PROPERTY_PBR_VALIDITY:
        {
            Color res = computeAlbedoPbrValidity(mAlbedo);
            *dest       = res.r;
            *(dest + 1) = res.g;
            *(dest + 2) = res.b;
        }
    break;
    default:
        handled = Bssrdf::getProperty(property, dest);
    }

    return handled;
}

void
DipoleBssrdf::show(std::ostream& os, const std::string& indent) const
{
    const Color& scale = getScale();
    const scene_rdl2::math::Vec3f& N = mFrame.getN();

    os << indent << "[DipoleBssrdf]\n";
    os << indent << "    " << "scale: "
        << scale.r << " " << scale.g << " " << scale.b << "\n";
    os << indent << "    " << "N: "
        << N.x << " " << N.y << " " << N.z << "\n";
    os << indent << "    " << "albedo: "
        << mAlbedo.r << " " << mAlbedo.g << " " << mAlbedo.b << "\n";
    if (mFresnel) {
        mFresnel->show(os, indent + "    ");
    }
}

//----------------------------------------------------------------------------

} // namespace shading
} // namespace moonray

