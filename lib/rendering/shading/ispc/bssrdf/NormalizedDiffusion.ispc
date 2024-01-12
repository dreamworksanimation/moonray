// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file NormalizedDiffusion.ispc

#include "NormalizedDiffusion.isph"
#include <moonray/rendering/shading/ispc/PbrValidity.isph>

#include <scene_rdl2/common/math/ispc/asA.isph>
#include <scene_rdl2/common/platform/IspcUtil.isph>


#define BSSRDF_SAMPLE_UNIFORM 0
#define BSSRDF_SAMPLE_WIDEST_TAIL 0


//----------------------------------------------------------------------------
// cdf for normalized diffusion, eq(11) in [1]
inline float diffusionCDF(float r,
                          float d)
{
    float x = -1 * r * rcp(d*3.f);
    float e1 = exp(x);
    float e2 = e1 * e1 * e1;

    //original diffusion equation is: eq(11)
    //  1.0f - 0.25f * exp(-r/d) - 0.75f * exp(-r/(3.f*d));
    //but simplified for evaluation below
    return 1.0f - 0.25f * e2 - 0.75f * e1;
}

inline float cubeRoot(float y)
{
    return pow(y, rcp(3.0f));
}

float inverseDiffusionCDF(float y,
                          float d)
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
    float sqrtTerm = sqrt(1.0f + z*z);
    float eCubeA = z + sqrtTerm;
    float eA = cubeRoot(eCubeA);
    float x = (eA*eA-1.0f) * rcp(eA);
    float r = -log(x) * 3.f * d;
    float valueCheck = diffusionCDF(r, d);
    assert(abs(valueCheck - y) < .0001f);
    return r;
}

// Feng: max radius is computed by solving for radius st CDF(r)==.995
// this means the loss of energy in ignoring scattering contribution outside of
// max radius is less than .5%

float findMaxR(float d)
{
    float x = inverseDiffusionCDF(.995, d);
    MNRY_ASSERT(x > 0.0f);
    return  x;
}

Color findMaxR(const Color& d)
{
    Color x;
    x.r = inverseDiffusionCDF(.995, d.r);
    x.g = inverseDiffusionCDF(.995, d.g);
    x.b = inverseDiffusionCDF(.995, d.b);
    return  x;
}

//Diffusion kernel, eq (2) of 1
inline float Diffusion(float r,
                       float d)
{
    static const float sPiTimes8 = sPi * 8.0f;
    if (r < .001) r = .001f;
    float e1 = exp(-r * rcp(d*3.f));

    //diffusion kernal:
//     (expf(-r/d) + expf(-r/(3.0f*d))) / (sPiTimes8*d*r);
    // simplified for evaluation
    return  (e1*e1*e1 + e1) * rcp(sPiTimes8*d*r);
    return d;
}

// reflectance profile approximation,
// from [1] equation (2),(3)
// scattering reflectance for normalized diffusion
// is albedo * Diffusion
Color
Reflectance(float r,
            Color albedo,
            Color d)
{
    Color reflectance;

    reflectance.r = albedo.r * Diffusion(r, d.r);
    reflectance.g = albedo.g * Diffusion(r, d.g);
    reflectance.b = albedo.b * Diffusion(r, d.b);

    return reflectance;
}

void
NormalizedDiffusionBssrdf_show(const varying Bssrdf * const uniform sss,
                               const uniform uint8_t ntab)
{
    const varying NormalizedDiffusionBssrdf * uniform me =
        (const varying NormalizedDiffusionBssrdf * uniform) sss;

    const varying Vec3f N = getN(me->mFrame);
    const varying Fresnel * const uniform fresnel = me->mFresnel;

    BsdfUtil_indent(ntab);   print("[NormalizedDiffusionBssrdf] mask: %\n", sss->mMask);
    BsdfUtil_indent(ntab+1); print("scale: % % %\n", BSDF_UTIL_EXTRACT(me->mScale.r),
                                                BSDF_UTIL_EXTRACT(me->mScale.g),
                                                BSDF_UTIL_EXTRACT(me->mScale.b));
    BsdfUtil_indent(ntab+1); print("N: % % %\n", BSDF_UTIL_EXTRACT(N.x),
                                            BSDF_UTIL_EXTRACT(N.y),
                                            BSDF_UTIL_EXTRACT(N.z));
    BsdfUtil_indent(ntab+1); print("albedo: % % %\n", BSDF_UTIL_EXTRACT(me->mAlbedo.r),
                                                 BSDF_UTIL_EXTRACT(me->mAlbedo.g),
                                                 BSDF_UTIL_EXTRACT(me->mAlbedo.b));
    BsdfUtil_indent(ntab+1); print("DMFP: % % %\n", BSDF_UTIL_EXTRACT(me->mDmfp.r),
                                               BSDF_UTIL_EXTRACT(me->mDmfp.g),
                                               BSDF_UTIL_EXTRACT(me->mDmfp.b));
    if (fresnel) {
        Fresnel_show(fresnel, ntab+1);
    }
}

void
NormalizedDiffusionBssrdf_init(varying NormalizedDiffusionBssrdf * uniform bssrdf,
                               const varying Vec3f &N,
                               const varying Color &albedo,
                               const varying Color &radius,
                               const uniform Material *uniform  material,
                               const uniform intptr_t evalNormalFn)
{
    // Initialize base class
    Bssrdf_init((varying Bssrdf * uniform)bssrdf,
                NormalizedDiffusionBssrdf_eval,
                NormalizedDiffusionBssrdf_sampleLocal,
                NormalizedDiffusionBssrdf_pdfLocal,
                NormalizedDiffusionBssrdf_getProperty,
                NormalizedDiffusionBssrdf_show,
                SUBSURFACE_NORMALIZED_DIFFUSION,
                N,
                SUBSURFACE_PROPERTY_COLOR | SUBSURFACE_PROPERTY_RADIUS | SUBSURFACE_PROPERTY_PBR_VALIDITY,
                material, evalNormalFn);

    ReferenceFrame_init(bssrdf->mFrame, N);

    const Color minAlbedo = Color_ctor(0.001f);
    const Color minDfmp = Color_ctor(0.001f);
    bssrdf->mAlbedo = max(albedo, minAlbedo);
    bssrdf->mDmfp = max(radius, minDfmp);

    // Compute 'd' shaping parameter from surface albedo, surface dmfp (radius)
    // see [1], sections (3) and (5),
    // Compute 's' scaling factor for searchlight config as described in [1], equation (8)
    // for (int c = 0; c < 3; ++c) {
    const Color tmp = bssrdf->mAlbedo - Color_ctor(0.33f);
    const Color tmp2 = tmp*tmp;
    const Color s = Color_ctor(3.5f) + 100.f * tmp2*tmp2;
    bssrdf->mD = bssrdf->mDmfp * rcp(s);

    // compute sampling ratio and pdf normalization terms
    bssrdf->mChannelCdf = max(bssrdf->mAlbedo, Color_ctor(sEpsilon));
    bssrdf->mMaxR = findMaxR(bssrdf->mD);

    float sumCDF = bssrdf->mChannelCdf.r +
                   bssrdf->mChannelCdf.g +
                   bssrdf->mChannelCdf.b;

    // Normalize
    if (!isZero(sumCDF))
        bssrdf->mChannelCdf = bssrdf->mChannelCdf * rcp(sumCDF);

    // Integrate the cdf
    bssrdf->mChannelCdf.g += bssrdf->mChannelCdf.r;
    bssrdf->mChannelCdf.b += bssrdf->mChannelCdf.g;
    float cdf = bssrdf->mChannelCdf.b;
    MNRY_ASSERT(isOne(cdf));

    bssrdf->mMaxRadius = max(max(bssrdf->mMaxR.r, bssrdf->mMaxR.g), bssrdf->mMaxR.b);
}

varying Color
NormalizedDiffusionBssrdf_eval(const varying Bssrdf * uniform bss,
                               varying float r,
                               uniform bool global)
{
    const varying NormalizedDiffusionBssrdf * uniform bssrdf = (const varying NormalizedDiffusionBssrdf * uniform)bss;

    Color rd = Reflectance(r,
                           bssrdf->mAlbedo,
                           bssrdf->mD);

    return rd;
};


//----------------------------------------------------------------------------

#if !BSSRDF_SAMPLE_UNIFORM

varying float
NormalizedDiffusionBssrdf_sampleLocal(const varying Bssrdf * uniform bss,
                                      varying float r1, varying float r2,
                                      varying Vec3f &dPi,
                                      varying float &r)
{
    const varying NormalizedDiffusionBssrdf * uniform bssrdf =
            (const varying NormalizedDiffusionBssrdf * uniform)bss;

    // remap r1 from [0, 1] to [.005, .995] to avoid wasting samples thru rejection
    r1 = r1 * .99 + .005;

    // Importance sample the channel and remap r2
    if (r2 < bssrdf->mChannelCdf.r) {
        // solve for sample "r" by inverting the CDF equal to random value "r1"
        r = inverseDiffusionCDF(r1, bssrdf->mD.r);
        r2 = r2 / bssrdf->mChannelCdf.r;
    }
    else if(r2 < bssrdf->mChannelCdf.g) {
        // solve for sample "r" by inverting the CDF equal to random value "r1"
        r = inverseDiffusionCDF(r1, bssrdf->mD.g);
        r2 = (r2 - bssrdf->mChannelCdf.r) / (bssrdf->mChannelCdf.g - bssrdf->mChannelCdf.r);
    }
    else if(r2 < bssrdf->mChannelCdf.b) {
        // solve for sample "r" by inverting the CDF equal to random value "r1"
        r = inverseDiffusionCDF(r1, bssrdf->mD.b);
        r2 = (r2 - bssrdf->mChannelCdf.g) / (bssrdf->mChannelCdf.b - bssrdf->mChannelCdf.g);
    }

    // Note because the above operation correspond to statistical branching
    // it is better to use the remapped r2 to preserve a good discrepancy of
    // the samples drawn from each of the 3 possible distributions.
    // TODO: Try using a good quality 3D sampler instead that down-projects well
    // to lower dimensions.
    const float phi = r2 * sTwoPi;

    // Compute position
    float sinPhi, cosPhi;
    sincos(phi, &sinPhi, &cosPhi);
    dPi.x = r * cosPhi;
    dPi.y = r * sinPhi;
    dPi.z = 0.0f;

    return NormalizedDiffusionBssrdf_pdfLocal(bss, r);
}

varying float
NormalizedDiffusionBssrdf_pdfLocal(const varying Bssrdf * uniform bss, varying float r)
{
    const varying NormalizedDiffusionBssrdf * uniform bssrdf =
            (const varying NormalizedDiffusionBssrdf * uniform)bss;

    // We compute the estimator with MIS, under Veach's one-sample model.
    // See paper: "Optimally combining sampling techniques for MC rendering"
    // Here we use estimator (16) and therefore: pdf = ci * pi / wi
    // With the balance heuristic it amounts to the weighted average of the
    // pdf for each channel: pdf = sum(ci * pi)
    float pdf = bssrdf->mChannelCdf.r * Diffusion(r, bssrdf->mD.r);
    pdf      += (bssrdf->mChannelCdf.g - bssrdf->mChannelCdf.r) * Diffusion(r, bssrdf->mD.g);
    pdf      += (bssrdf->mChannelCdf.b - bssrdf->mChannelCdf.g) * Diffusion(r, bssrdf->mD.b);

    return pdf;
}

/// For NormalizedDiffusion, albedo is the reflectance
varying Color
NormalizedDiffusionBssrdf_diffuseReflectance(const varying Bssrdf * uniform bss)
{
    const varying NormalizedDiffusionBssrdf * uniform bssrdf =
            (const varying NormalizedDiffusionBssrdf * uniform)bss;

    return (bssrdf->mAlbedo);
}

#else
// Uniform sampling
varying float
NormalizedDiffusionBssrdf_sampleLocal(const varying Bssrdf * uniform bss,
                                      varying float r1, varying float r2,
                                      varying Vec3f &dPi,
                                      varying float &r)
{
    const varying NormalizedDiffusionBssrdf * uniform bssrdf = (const varying NormalizedDiffusionBssrdf * uniform)bs;

    Vec2f p;
    p[0] = (r1 - 0.5f) * 2.0f * mMaxRadius;
    p[1] = (r2 - 0.5f) * 2.0f * mMaxRadius;
    r = p.length();

    // Compute position
    dPi = p[0] * mFrame.getX() + p[1] * mFrame.getY();

    float pdf = 1.0f / (4.0f * mMaxRadius * mMaxRadius);
    return pdf;
}


varying float
NormalizedDiffusionBssrdf_pdfLocal(const varying Bssrdf * uniform bss, varying float r)
{
    const varying NormalizedDiffusionBssrdf * uniform bssrdf = (const varying NormalizedDiffusionBssrdf * uniform)bs;

    float pdf = rcp(4.0f * mMaxRadius * mMaxRadius);
    return pdf;
}


#endif

uniform bool
NormalizedDiffusionBssrdf_getProperty(const varying Bssrdf * uniform bss,
                                      uniform SubsurfaceProperty property,
                                      varying float * uniform dest)
{
    const varying NormalizedDiffusionBssrdf * uniform bssrdf =
            (const varying NormalizedDiffusionBssrdf * uniform) bss;

    uniform bool handled = true;

    switch (property)
    {
    case SUBSURFACE_PROPERTY_COLOR:
        *dest       = bssrdf->mAlbedo.r;
        *(dest + 1) = bssrdf->mAlbedo.g;
        *(dest + 2) = bssrdf->mAlbedo.b;
        break;
    case SUBSURFACE_PROPERTY_RADIUS:
        *dest       = bssrdf->mDmfp.r;
        *(dest + 1) = bssrdf->mDmfp.g;
        *(dest + 2) = bssrdf->mDmfp.b;
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
ISPC_UTIL_EXPORT_STRUCT_TO_HEADER(NormalizedDiffusionBssrdf);

