// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
///

// Normalize diffusion is based on the Siggraph paper by Per Christensen and Brent Burley, 
// reference [1]
//
// The CDF inversion computation in developed by Feng Xie using math equation rewrite and 
// using a reference for root solving for cubic polynomial :
// http://www-history.mcs.st-andrews.ac.uk/history/HistTopics/Quadratic_etc_equations.html [2]

#include "NormalizedDiffusion.h"
#include <moonray/rendering/shading/bsdf/Fresnel.h>
#include <moonray/rendering/shading/PbrValidity.h>

#include <scene_rdl2/common/math/MathUtil.h>
#include <cmath>
#include <iomanip>

#define BSSRDF_SAMPLE_UNIFORM 0

using namespace scene_rdl2::math;

namespace {

// cdf for normalized diffusion, eq(11) in [1]
finline float diffusionCDF(float r, float d) __attribute__((__unused__));

finline float diffusionCDF(float r, float d)
{
    float x = -r/(d*3.f);
    float e1 = expf(x);
    float e2 = e1 * e1 * e1;
    
    //original diffusion equation is: eq(11)
    //  1.0f - 0.25f * exp(-r/d) - 0.75f * exp(-r/(3.f*d));
    //but simplified for evaluation below
    return 1.0f - 0.25f * e2 - 0.75f * e1;
}

float cubeRoot(float y)
{
    return cbrtf(y); 
}

float inverseDiffusionCDF(float y, float d)
{
    //-- Contrary to the claim of paper [1], our CDF actually
    //-- can be inverted analytically as follows
    //-- derivation by Feng

    // eq 1: y = 1-exp(-r/d)/4 - exp(-r/3d) * 3/4
    // let x = exp(-r/3d), then x^3 = exp(-r/d)
    // eq1 can be rewritten as 
    // eq 2:  4y = 4 - (x^3 +3x)
    // then again rewritten as 
    // eq 3: 4(1-y) = x^3 + 3x, 
    // the cubic polynomial has valid roots
    // using reference 2 as follows
    // for solving cubic polynomial of the form
    // x^3 + 3b^2x = 2c^3
    // where in our case, b is 1 and 2c^3 = 4(1-y)
    // so we solve our polynomial as follows:
    // let z = 4(1-y)/2
    // let c^3 = z
    // let e^3 = (c^3) (+/-) sqrt(1+c^6), 
    // then x = (e^2 - 1) /e,  is the root of eq 2 
    // r = -ln(x) * 3 * d,  is the root of eq 1

    // the code below implement the last 5 lines of the derivation comments
    float z = (1.f -y ) * 2.0f;
    float sqrtTerm = sqrtf(1.0f + z*z);   
    float eCubeA = z + sqrtTerm;
    float eA = cubeRoot(eCubeA);
    float x = (eA*eA-1.0f)/eA;
    float r = -logf(x) * 3.f * d;
    MNRY_ASSERT(fabs(diffusionCDF(r, d) - y) < .0001f);
    return r;
}

// Feng: max radius is computed by solving for radius st CDF(r)==.995
// this means the loss of energy in ignoring scattering contribution outside of 
// max radius is less than .5%

float findMaxR(float d)
{
    float x = inverseDiffusionCDF(.995, d);
    return x;  
} 

//Diffusion kernel, eq (2) of 1
finline float Diffusion(float r, float d)
{
    static const float sPiTimes8 = scene_rdl2::math::sPi * 8.0f;
    if (r < .001) r = .001;
    float e1 = expf(-r/(d*3.f));
    
    //diffusion kernal:  
    // (expf(-r/d) + expf(-r/(3.0f*d))) / (sPiTimes8*d*r);
    // simplified for evaluation
    return  (e1*e1*e1 + e1) / (sPiTimes8*d*r);
}

// reflectance profile approximation,
// from [1] equation (2),(3)
// scattering reflectance for normalized diffusion
// is albedo * Diffusion 
float Reflectance(float r, float albedo, float d)
{
    return albedo * Diffusion(r, d);
}

} // namespace


namespace moonray {
namespace shading {


//----------------------------------------------------------------------------

NormalizedDiffusionBssrdf::NormalizedDiffusionBssrdf(const Vec3f &N,
        const Color &albedo, const Color &radius,
        const scene_rdl2::rdl2::Material* material,
        const scene_rdl2::rdl2::EvalNormalFunc evalNormalFn)
    : Bssrdf(N, PROPERTY_COLOR | PROPERTY_RADIUS | PROPERTY_PBR_VALIDITY, material, evalNormalFn)
    , mAlbedo(max(albedo, Color(0.001f)))
    , mDmfp(max(radius, Color(0.001f)))
{
    // Compute 'd' shaping parameter from surface albedo, surface dmfp (radius)
    // see [1], sections (3) and (5),
    // Compute 's' scaling factor for searchlight config as described in [1], equation (8)
    for (size_t c = 0; c < 3; ++c) {
        const float tmp = mAlbedo[c] - 0.33f;
        const float tmp2 = tmp*tmp;
        const float s = 3.5f + 100.f * tmp2*tmp2;
        mD[c] = mDmfp[c]/s;
    }

    // compute sampling ratio and pdf normalization terms
    float normalize = 0.0f;
    for (int c=0; c < 3; c++) {
        mChannelCdf[c] = max(mAlbedo[c], sEpsilon);
        normalize += mChannelCdf[c];
        mMaxR[c] = findMaxR(mD[c]);
        MNRY_ASSERT(mMaxR[c] > 0.0f);
    }

    // Normalize and integrate the cdf, all in one pass
    normalize = 1.0f / normalize;
    float cdf = 0.0f;
    for (int c=0; c < 3; c++) {
        mChannelCdf[c] *= normalize;
        mChannelCdf[c] += cdf;
        cdf = mChannelCdf[c];
    }
    MNRY_ASSERT(isOne(cdf));

    mMaxRadius = max(max(mMaxR[0], mMaxR[1]), mMaxR[2]);
}

    NormalizedDiffusionBssrdf::NormalizedDiffusionBssrdf(scene_rdl2::alloc::Arena *arena,
                                                     const NormalizedDiffusionBssrdfv &bssrdfv,
                                                     int lane)
  : Bssrdf(arena, (const Bssrdfv &)bssrdfv, lane)
  , mAlbedo(bssrdfv.mAlbedo.r[lane], bssrdfv.mAlbedo.g[lane], bssrdfv.mAlbedo.b[lane])
  , mDmfp(bssrdfv.mDmfp.r[lane], bssrdfv.mDmfp.g[lane], bssrdfv.mDmfp.b[lane])
  , mD(bssrdfv.mD.r[lane], bssrdfv.mD.g[lane], bssrdfv.mD.b[lane])
  , mChannelCdf(bssrdfv.mChannelCdf.r[lane], bssrdfv.mChannelCdf.g[lane], bssrdfv.mChannelCdf.b[lane])
  , mMaxR(bssrdfv.mMaxR.r[lane], bssrdfv.mMaxR.g[lane], bssrdfv.mMaxR.b[lane])
{
}

//----------------------------------------------------------------------------

Color
NormalizedDiffusionBssrdf::eval(float r, bool /* global */) const
{
    Color rd;
    for (size_t c = 0; c < 3; ++c) {
        rd[c] = Reflectance(r, mAlbedo[c], mD[c]);
    }
    return rd;
}

#if !BSSRDF_SAMPLE_UNIFORM

float
NormalizedDiffusionBssrdf::sampleLocal(float r1, float r2, Vec3f &dPi, float &r) const
{
    // Importance sample the channel and remap r2
    int c = (r2 < mChannelCdf[0]  ?  0  :
            (r2 < mChannelCdf[1]  ?  1  :  2));
    r2 = (c == 0  ?  r2 / mChannelCdf[0]  :
            (r2 - mChannelCdf[c-1]) / (mChannelCdf[c] - mChannelCdf[c-1]));

    // Note because the above operation correspond to statistical branching
    // it is better to use the remapped r2 to preserve a good discrepancy of
    // the samples drawn from each of the 3 possible distributions.
    // TODO: Try using a good quality 3D sampler instead that down-projects well
    // to lower dimensions.
    const float phi = r2 * sTwoPi;

    // remap r1 from [0, 1] to [.005, .995] to avoid wasting samples thru rejection
    r1 = r1 * .99 + .005;
    
    // solve for sample "r" by inverting the CDF equal to random value "r1"
    r = inverseDiffusionCDF(r1, mD[c]);

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
NormalizedDiffusionBssrdf::pdfLocal(float r) const
{
    // We compute the estimator with MIS, under Veach's one-sample model.
    // See paper: "Optimally combining sampling techniques for MC rendering"
    // Here we use estimator (16) and therefore: pdf = ci * pi / wi
    // With the balance heuristic it amounts to the weighted average of the
    // pdf for each channel: pdf = sum(ci * pi)
    float pdf = 0.0f;
    for (int i = 0; i < 3; ++i) {
        // Compute ci, the discrete probability of picking this channel
        float ci = (i == 0  ?  mChannelCdf[i]  :
                               mChannelCdf[i] - mChannelCdf[i-1]);
        pdf += ci * Diffusion(r, mD[i]);
    }

    return pdf;
}

#else

// Uniform sampling
float
NormalizedDiffusionBssrdf::sampleLocal(float r1, float r2, Vec3f &dPi, float &r) const
{
    dPi[0] = (r1 - 0.5f) * 2.0f * mMaxRadius;
    dPi[1] = (r2 - 0.5f) * 2.0f * mMaxRadius;
    dPi[2] = 0.0f;
    r = dPi.length();

    float pdf = 1.0f / (4.0f * mMaxRadius * mMaxRadius);
    return pdf;
}

float
NormalizedDiffusionBssrdf::pdfLocal(float r) const
{
    float pdf = 1.0f / (4.0f * mMaxRadius * mMaxRadius);
    return pdf;
}
#endif

bool
NormalizedDiffusionBssrdf::getProperty(Property property, float *dest) const
{
    bool handled = true;

    switch (property)
    {
    case PROPERTY_COLOR:
        *dest       = mAlbedo[0];
        *(dest + 1) = mAlbedo[1];
        *(dest + 2) = mAlbedo[2];
        break;
    case PROPERTY_RADIUS:
        *dest       = mDmfp[0];
        *(dest + 1) = mDmfp[1];
        *(dest + 2) = mDmfp[2];
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
NormalizedDiffusionBssrdf::show(std::ostream& os, const std::string& indent) const
{
    const Color& scale = getScale();
    const scene_rdl2::math::Vec3f& N = mFrame.getN();

    os << indent << "[NormalizedDiffusionBssrdf]\n";
    os << indent << "    " << "scale: "
        << scale.r << " " << scale.g << " " << scale.b << "\n";
    os << indent << "    " << "N: "
        << N.x << " " << N.y << " " << N.z << "\n";
    os << indent << "    " << "albedo: "
        << mAlbedo.r << " " << mAlbedo.g << " " << mAlbedo.b << "\n";
    os << indent << "    " << "DMFP: "
        << mDmfp.r << " " << mDmfp.g << " " << mDmfp.b << "\n";
    if (mFresnel) {
        mFresnel->show(os, indent + "    ");
    }
}

//----------------------------------------------------------------------------

} // namespace shading
} // namespace moonray

