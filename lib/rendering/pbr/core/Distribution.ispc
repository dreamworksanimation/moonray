// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#include "Distribution.isph"

#include <scene_rdl2/common/math/ispc/Math.isph>
#include <scene_rdl2/common/platform/IspcUtil.isph>

//----------------------------------------------------------------------------

ISPC_UTIL_EXPORT_UNIFORM_STRUCT_TO_HEADER(TextureFilterType);
ISPC_UTIL_EXPORT_UNIFORM_STRUCT_TO_HEADER(GuideDistribution1D);
ISPC_UTIL_EXPORT_UNIFORM_STRUCT_TO_HEADER(Distribution2D);
ISPC_UTIL_EXPORT_UNIFORM_STRUCT_TO_HEADER(ImageDistribution);


export uniform uint32_t
GuideDistribution1D_hudValidation(uniform bool verbose)
{
    GUIDE_DISTRIBUTION_1D_VALIDATION;
}

export uniform uint32_t
Distribution2D_hudValidation(uniform bool verbose)
{
    DISTRIBUTION_2D_VALIDATION;
}

export uniform uint32_t
ImageDistribution_hudValidation(uniform bool verbose)
{
    IMAGE_DISTRIBUTION_VALIDATION;
}


inline void
intAndFrac(varying float x, varying int * uniform xInt, varying float * uniform xFrac)
{
    MNRY_ASSERT(xInt != nullptr && xFrac != nullptr);
    float xFloor = floor(x);
    *xInt = (int)xFloor;
    *xFrac = x - xFloor;
}

//----------------------------------------------------------------------------

varying float
GuideDistribution1D_pdfDiscrete(const uniform GuideDistribution1D * uniform dis, varying uint32_t index)
{
    const float prevCdf = (index > 0)  ?  dis->mCdf[index - 1]  :  0.0f;
    return (dis->mCdf[index] - prevCdf) * dis->mInvTotalWeight;
}

varying float
GuideDistribution1D_pdfDiscrete(const uniform GuideDistribution1D * varying dis, varying uint32_t index)
{
    const float prevCdf = (index > 0)  ?  dis->mCdf[index - 1]  :  0.0f;
    return (dis->mCdf[index] - prevCdf) * dis->mInvTotalWeight;
}


varying float
GuideDistribution1D_pdfContinuous(const uniform GuideDistribution1D * uniform dis, varying float u)
{
    uint32_t i = (uint32_t)(u * dis->mSizeCdf);
    i = min(i, dis->mSizeCdf - 1);
    const float prevCdf = (i > 0)  ?  dis->mCdf[i - 1]  :  0.0f;
    return (dis->mCdf[i] - prevCdf) * dis->mInvTotalWeight * dis->mSizeCdf;
}

varying float
GuideDistribution1D_pdfContinuous(const uniform GuideDistribution1D * varying dis, varying float u)
{
    uint32_t i = (uint32_t)(u * dis->mSizeCdf);
    i = min(i, dis->mSizeCdf - 1);
    const float prevCdf = (i > 0)  ?  dis->mCdf[i - 1]  :  0.0f;
    return (dis->mCdf[i] - prevCdf) * dis->mInvTotalWeight * dis->mSizeCdf;
}


varying float
GuideDistribution1D_pdfLinear(const uniform GuideDistribution1D * uniform dis, varying float u)
{
    if (u < 0.0f || u > 1.0f) return 0.0f;

    uniform int size = dis->mSizeCdf;
    uniform float * uniform cdf = dis->mCdf;

    int i;
    float t;

    // Here we must add 0.5 to the texture coordinate (after scaling it to be in units of texels). This is because
    // we're evaluating the texture as a piecewise-linear function, with a new linear segment for each texel, but in
    // particular the segments join at the texel centers, whose positions have a fractional part of 0.5. Thus we want
    // to advance the index i whenever the fractional part reaches 0.5. (Compare this with the situation for nearest
    // neighbour textures, where the segments join a the texel boundaries, whose positions have a fractional part of
    // 0.0.)
    intAndFrac(u * size + 0.5f, &i, &t);

    float p;
    if (i == 0) {
        // Clamp to lower border
        p = cdf[0];
    }
    else if (i == size) {
        // Clamp to upper border
        p = cdf[size-1] - cdf[size-2];
    }
    else {
        // Lerp between 2 values
        float p0 = (i == 1) ? cdf[0] : cdf[i-1] - cdf[i-2];
        float p1 = cdf[i] - cdf[i-1];
        p  = lerp(p0, p1, t);
    }

    return p * dis->mInvTotalWeight * size;
}

varying float
GuideDistribution1D_pdfLinear(const uniform GuideDistribution1D * varying dis, varying float u)
{
    if (u < 0.0f || u > 1.0f) return 0.0f;

    varying int size = dis->mSizeCdf;
    uniform float * varying cdf = dis->mCdf;

    int i;
    float t;

    // Here we must add 0.5 to the texture coordinate (after scaling it to be in units of texels). This is because
    // we're evaluating the texture as a piecewise-linear function, with a new linear segment for each texel, but in
    // particular the segments join at the texel centers, whose positions have a fractional part of 0.5. Thus we want
    // to advance the index i whenever the fractional part reaches 0.5. (Compare this with the situation for nearest
    // neighbour textures, where the segments join a the texel boundaries, whose positions have a fractional part of
    // 0.0.)
    intAndFrac(u * size + 0.5f, &i, &t);

    float p;
    if (i == 0) {
        // Clamp to lower border
        p = cdf[0];
    }
    else if (i == size) {
        // Clamp to upper border
        p = cdf[size-1] - cdf[size-2];
    }
    else {
        // Lerp between 2 values
        float p0 = (i == 1) ? cdf[0] : cdf[i-1] - cdf[i-2];
        float p1 = cdf[i] - cdf[i-1];
        p  = lerp(p0, p1, t);
    }

    return p * dis->mInvTotalWeight * size;
}


uint32_t
GuideDistribution1D_sampleDiscrete(const uniform GuideDistribution1D * uniform dis,
        varying float r, varying float * uniform pdf)
{
    varying uint32_t i = getGuideIndex(dis, r);
    float rw = r * dis->mTotalWeight;   // Scaling by the total weight is needed because the cdf is non-normalized
    while (dis->mCdf[i] < rw) {
        ++i;
    }
    MNRY_ASSERT(i < dis->mSizeCdf);

    if (pdf) *pdf = GuideDistribution1D_pdfDiscrete(dis, i);

    return i;
}


float
GuideDistribution1D_sampleContinuous(const uniform GuideDistribution1D * uniform dis,
        varying float r, varying float * uniform pdf,
        varying uint32_t * uniform index)
{
    varying uint32_t i = getGuideIndex(dis, r);
    float rw = r * dis->mTotalWeight;   // Scaling by the total weight is needed because the cdf is non-normalized
    while (dis->mCdf[i] < rw) {
        ++i;
    }
    MNRY_ASSERT(i < dis->mSizeCdf);

    if (index) *index = i;

    const float prevCdf = (i > 0)  ?  dis->mCdf[i-1]  :  0.0f;

    // Lerp inverted cdf value
    // (Note here that an expression of the form lerp(i,i+1,t) has been replaced with the simpler and more
    // efficient expression i+t)
    float invCdf = ((float)i + (rw - prevCdf) / (dis->mCdf[i] - prevCdf)) * dis->mInvSizeCdf;

    // May reach 1.0 due to float imprecision
    invCdf = min(invCdf, sOneMinusEpsilon);

    if (pdf) *pdf = (dis->mCdf[i] - prevCdf) * dis->mInvTotalWeight * dis->mSizeCdf;

    return invCdf;
}


float
GuideDistribution1D_sampleContinuous(const uniform GuideDistribution1D * varying dis,
        varying float r, varying float * uniform pdf,
        varying uint32_t * uniform index)
{
    varying uint32_t i = getGuideIndex(dis, r);
    float rw = r * dis->mTotalWeight;   // Scaling by the total weight is needed because the cdf is non-normalized
    while (dis->mCdf[i] < rw) {
        ++i;
    }
    MNRY_ASSERT(i < dis->mSizeCdf);

    if (index) *index = i;

    // Lerp inverted cdf value
    const float prevCdf = (i > 0)  ?  dis->mCdf[i-1]  :  0.0f;
    float invCdf = ((float)i + (rw - prevCdf) / (dis->mCdf[i] - prevCdf)) * dis->mInvSizeCdf;
    // May reach 1.0 due to float imprecision
    invCdf = min(invCdf, sOneMinusEpsilon);

    if (pdf) *pdf = (dis->mCdf[i] - prevCdf) * dis->mInvTotalWeight * dis->mSizeCdf;

    return invCdf;
}


// Helper function to solve for the parameter t in [0,1] where a given quadratic curve segment takes on the 
// given value k. (2*k is actually passed in.)
//
// The calling code will have looked up 3 consecutive cdf values, c0, c1, c2. We are interested in the curve
// segment covering the range of values from (c0+c1)/2 to (c1+c2)/2, corresponding to the interval between
// texel centers.
//
// The curve segment is a quadratic Bezier curve with control points (c0+c1)/2, c1, (c1+c2)/2. It is easy to 
// show that this has expression q(t) = a*t^2 + 2b*t + c, with
// a = ((c2 - c1) - (c1-c0)) / 2
// b = (c1 - c0) / 2
// c = (c0 + c1) / 2
// 
// We wish to solve for t the equation q(t) = k, or Q(t) = A*t^2 + 2*B*t + C = 0, with
// A = (c2 - c1) - (c1 - c0)
// B = c1 - c0
// C = c0 + c1 - 2k
// 
// This is easy to solve via an application of the quadratic formula. The quadratic has 2 roots and we want to
// return a particular one - the lower root, if Q(t) is convex upwards (A > 0), and the upper root if it is
// convex downwards (A < 0). Note that it can also degenerate into a linear equation when all 3 points c0, c1, c2
// lie in a straight line (A = 0).
// 
// If A > 0, the upper root must be (-B + sqrt(discrim)) / A
// And if A < 0, the lower root must be (-B + sqrt(discrim)) / A, i.e. the same expression.
// Since B >= 0, -B and sqrt(discrim) have opposite signs, so to avoid possible catastrophic cancellation
// (which will happen if A*C is close to zero), we swith to alternate expression C / (-B - sqrt(discrim))
// in both cases.
// 
// Notice that this conveniently caters to the degenerate linear case, too, since if A = 0 the equation
// reduces to 2*B*t + C = 0 with solution -C/(2B), and since sqrt(discrim) will equal B, this is exactly
// the solution generated.
//
// Note also that the discriminant is evaluated as B^2-AC, rather than the probably more familiar B^2-4AC, and there's
// a corresponding removal of a factor 2 in the final result (the denominator 2A -> A for the "standard" form, or the
// numerator 2C -> C for the form used below). This is because the factor 2 has been explicitly pulled out of the
// linear term of Q(t) = A*t^2 + 2*B*t + C = 0, allowing the various cancellations to ensue.

inline float
quadraticHelper(float c0, float c1, float c2, float twoK, float &p)
{
    float p0 = c1 - c0;
    float p1 = c2 - c1;

    float A = p1 - p0;
    float B = p0;
    float C = c0 + c1 - twoK;

    float discrim = B*B - A*C;
    float denom = -B - sqrt(discrim);

    float t = (denom != 0.0f) ? C / denom : 0.0f;
    p = lerp(p0, p1, t);

    return t;
}


// Sampling from a distribution which is a linearly-interpolated 1D texture. The cdf in this case is made up 
// of a tangent-continuous set of quadratic Bezier curves, with linear portions in the first and last half-texel.
// The reason it is piecewise quadratic is that the cdf is the integral of the pdf, which in this case is a
// piecewise-linear function. And the reason the cdf is linear over the first and last half-texels is that the pdf
// is clamped to a constant border value over those intervals.
// We proceed as in the nearest-neighbour case by finding the texel span which the given r-value intersects, and
// then computing the instersection.

varying float
GuideDistribution1D_sampleLinear(const uniform GuideDistribution1D * uniform dis,
        varying float r, varying float * uniform pdf)
{
    MNRY_ASSERT(r >= 0.0f  &&  r <= 1.0f);
  
    // Linear interpolation in 1st half-texel
    if (r <= dis->mThresholdLow) {
        return min(r * dis->mLinearCoeffLow, sOneMinusEpsilon);
    }

    // Linear interpolation in last half-texel
    if (r >= dis->mThresholdHigh) {
        return min(1.0f - (1.0f - r) * dis->mLinearCoeffHigh, sOneMinusEpsilon);
    }

    // Look up in guide table.
    // The guide table entries are optimal for nearest-neighbour textures, not for linear ones. It would be
    // possible to add in a second table, but the added memory cost and code support would outweigh the very
    // slight performance gain, so we just use the existing table.
    varying int i = getGuideIndex(dis, r);

    // Early-out in 1st linear case above means we don't need to search starting at i=0
    i = max(i, 1);

    varying float twoWr = 2.0f * dis->mTotalWeight * r;

    // Step index until we find the curve segment which r intersects with
    varying float c0 = (i >= 2) ? dis->mCdf[i-2] : 0.0f;
    varying float c1 = dis->mCdf[i-1];
    varying float c2 = dis->mCdf[i];
    while (c1 + c2 < twoWr)  // We use precomputed 2*wr to avoid multiplying c1+c2 here by 0.5.
    {
        ++i;
        MNRY_ASSERT(i < dis->mSizeCdf);
        c0 = c1;
        c1 = c2;
        c2 = dis->mCdf[i];
    }

    // Solve for intersection with selected curve segment.
    // The curve segments are necessarily quadratic, because the cdf is the integral of the pdf
    // which in this case is piecewise linear.
    varying float p;
    varying float t = quadraticHelper(c0, c1, c2, twoWr, p);

    if (pdf) *pdf = p * dis->mInvTotalWeight * dis->mSizeCdf;

    // Return location which is a fraction t between texel centers i, i+1
    return min(((float)i + 0.5f + t) * dis->mInvSizeCdf, sOneMinusEpsilon);
}


varying float
GuideDistribution1D_sampleLinear(const uniform GuideDistribution1D * varying dis,
        varying float r, varying float * uniform pdf)
{
    MNRY_ASSERT(r >= 0.0f  &&  r <= 1.0f);
  
    // Linear interpolation in 1st half-texel
    if (r <= dis->mThresholdLow) {
        return min(r * dis->mLinearCoeffLow, sOneMinusEpsilon);
    }

    // Linear interpolation in last half-texel
    if (r >= dis->mThresholdHigh) {
        return min(1.0f - (1.0f - r) * dis->mLinearCoeffHigh, sOneMinusEpsilon);
    }

    // Look up in guide table.
    // The guide table entries are optimal for nearest-neighbour textures, not for linear ones. It would be
    // possible to add in a second table, but the added memory cost and code support would outweigh the very
    // slight performance gain, so we just use the existing table.
    varying int i = getGuideIndex(dis, r);

    // Early-out in 1st linear case above means we don't need to search starting at i=0
    i = max(i, 1);

    varying float twoWr = 2.0f * dis->mTotalWeight * r;

    // Step index until we find the curve segment which r intersects with
    varying float c0 = (i >= 2) ? dis->mCdf[i-2] : 0.0f;
    varying float c1 = dis->mCdf[i-1];
    varying float c2 = dis->mCdf[i];
    while (c1 + c2 < twoWr)  // We use precomputed 2*wr to avoid multiplying c1+c2 here by 0.5.
    {
        ++i;
        MNRY_ASSERT(i < dis->mSizeCdf);
        c0 = c1;
        c1 = c2;
        c2 = dis->mCdf[i];
    }

    // Solve for intersection with selected curve segment.
    // The curve segments are necessarily quadratic, because the cdf is the integral of the pdf
    // which in this case is piecewise linear.
    varying float p;
    varying float t = quadraticHelper(c0, c1, c2, twoWr, p);

    if (pdf) *pdf = p * dis->mInvTotalWeight * dis->mSizeCdf;

    // Return location which is a fraction t between texel centers i, i+1
    return min(((float)i + 0.5f + t) * dis->mInvSizeCdf, sOneMinusEpsilon);
}


//----------------------------------------------------------------------------

varying float
Distribution2D_pdfNearest(const uniform Distribution2D * uniform dis, varying float u, varying float v)
{
    uint32_t vSize = GuideDistribution1D_getSize(dis->mMarginal);
    uint32_t vIndex = (uint32_t)(v * vSize);
    vIndex = min(vIndex, vSize - 1);

    return GuideDistribution1D_pdfContinuous(dis->mMarginal, v) *
           // Careful, here the distribution is a (uniform Dist * varying dist)
           GuideDistribution1D_pdfContinuous(dis->mConditional[vIndex], u);
}


varying float
Distribution2D_pdfNearest(const uniform Distribution2D * varying dis, varying float u, varying float v)
{
    uint32_t vSize = GuideDistribution1D_getSize(dis->mMarginal);
    uint32_t vIndex = (uint32_t)(v * vSize);
    vIndex = min(vIndex, vSize - 1);

    return GuideDistribution1D_pdfContinuous(dis->mMarginal, v) *
           // Careful, here the distribution is a (uniform Dist * varying dist)
           GuideDistribution1D_pdfContinuous(dis->mConditional[vIndex], u);
}


varying float
Distribution2D_pdfBilinear(const uniform Distribution2D * uniform dis, varying float u, varying float v)
{
    if (u < 0.0f || u >= 1.0f || v < 0.0f || v >= 1.0f) return 0.0f;

    varying float pMarg = GuideDistribution1D_pdfContinuous(dis->mMarginal, v);

    int n = dis->mMarginal->mSizeCdf;

    int i1;
    float s;
    intAndFrac(v * n + 0.5f, &i1, &s);
    int i0 = i1 - 1;

    i0 = max(i0, 0);
    i1 = min(i1, n-1);

    float pCond0 = GuideDistribution1D_pdfContinuous(dis->mConditional[i0], u);
    float pCond1 = GuideDistribution1D_pdfContinuous(dis->mConditional[i1], u);

    float pCond = lerp(pCond0, pCond1, s);

    return pMarg * pCond;
}


varying float
Distribution2D_pdfBilinear(const uniform Distribution2D * varying dis, varying float u, varying float v)
{
    if (u < 0.0f || u >= 1.0f || v < 0.0f || v >= 1.0f) return 0.0f;

    varying float pMarg = GuideDistribution1D_pdfContinuous(dis->mMarginal, v);

    int n = dis->mMarginal->mSizeCdf;

    int i1;
    float s;
    intAndFrac(v * n + 0.5f, &i1, &s);
    int i0 = i1 - 1;

    i0 = max(i0, 0);
    i1 = min(i1, n-1);

    float pCond0 = GuideDistribution1D_pdfContinuous(dis->mConditional[i0], u);
    float pCond1 = GuideDistribution1D_pdfContinuous(dis->mConditional[i1], u);

    float pCond = lerp(pCond0, pCond1, s);

    return pMarg * pCond;
}


void
Distribution2D_sampleNearest(const uniform Distribution2D * uniform dis, varying float ru, varying float rv,
                             varying Vec2f * uniform uv, varying float * uniform pdf)
{
    MNRY_ASSERT(uv != nullptr);
    uint32_t v;
    float pdfMarg, pdfCond;
    uv->y = GuideDistribution1D_sampleContinuous(dis->mMarginal, rv, &pdfMarg, &v);
    // Careful, here the distribution is a (uniform Dist * varying dist)
    uv->x = GuideDistribution1D_sampleContinuous(dis->mConditional[v], ru, &pdfCond, nullptr);

    if (pdf) *pdf = pdfMarg * pdfCond;
}


void
Distribution2D_sampleNearest(const uniform Distribution2D * varying dis, varying float ru, varying float rv,
                             varying Vec2f * uniform uv, varying float * uniform pdf)
{
    MNRY_ASSERT(uv != nullptr);
    uint32_t v;
    float pdfMarg, pdfCond;
    uv->y = GuideDistribution1D_sampleContinuous(dis->mMarginal, rv, &pdfMarg, &v);
    // Careful, here the distribution is a (uniform Dist * varying dist)
    uv->x = GuideDistribution1D_sampleContinuous(dis->mConditional[v], ru, &pdfCond, nullptr);

    if (pdf) *pdf = pdfMarg * pdfCond;
}


// A helper function that works like GuideDistribution1D_sampleLinear() except that it linearly blends
// the coefficients of 2 cdfs (presumably 2 adjacent rows of the conditional cdf table) by the parameter s
varying float
sampleLinearBlended(varying float r, varying float *uniform pdf,
                    uniform GuideDistribution1D *varying cond0,
                    uniform GuideDistribution1D *varying cond1, varying float s)
{
    MNRY_ASSERT(r >= 0.0f  &&  r <= 1.0f);
  
    // Linear interpolation in 1st half-texel
    if (r <= lerp(cond0->mThresholdLow, cond1->mThresholdLow, s)) {
        return min(r * lerp(cond0->mLinearCoeffLow, cond1->mLinearCoeffLow, s), sOneMinusEpsilon);
    }

    // Linear interpolation in last half-texel
    if (r >= lerp(cond0->mThresholdHigh, cond1->mThresholdHigh, s)) {
        return min(1.0f - (1.0f - r) * lerp(cond0->mLinearCoeffHigh, cond1->mLinearCoeffHigh, s), sOneMinusEpsilon);
    }
      
    // Look up in guide tables
    varying int i = min(getGuideIndex(cond0, r), getGuideIndex(cond1, r));

    // Early-out in 1st linear case above means we don't need to search starting at i=0
    i = max(i, 1);

    varying float w = lerp(cond0->mTotalWeight, cond1->mTotalWeight, s);
    varying float twoWr = 2.0f * w * r;
    const uniform float *varying cdf0 = cond0->mCdf;
    const uniform float *varying cdf1 = cond1->mCdf;

    // Step index until we find the curve segment which r intersects with
    varying float c0 = (i >= 2) ? lerp(cdf0[i-2], cdf1[i-2], s) : 0.0f;
    varying float c1 = lerp(cdf0[i-1], cdf1[i-1], s);
    varying float c2 = lerp(cdf0[i], cdf1[i], s);
    const int n = cond0->mSizeCdf;
    while (c1 + c2 < twoWr)
    {
        ++i;
        if (i >= n) {
            // When the true sample result is very close to the boundary between 2 table entries, rounding can
            // cause i to step too high by 1, and if this happens at the top end of the array it will be out of
            // bounds. If this case is detected we simply fall back on the linear interpolation in the uppermost
            // half-texel.
            return min(1.0f - (1.0f - r) * lerp(cond0->mLinearCoeffHigh, cond1->mLinearCoeffHigh, s),
                       sOneMinusEpsilon);
        }
        c0 = c1;
        c1 = c2;
        c2 = lerp(cdf0[i], cdf1[i], s);
    }

    // Solve for intersection with selected curve segment.
    // The curve segments are necessarily quadratic, because the cdf is the integral of the pdf
    // which in this case is piecewise linear.
    varying float p;
    varying float t = quadraticHelper(c0, c1, c2, twoWr, p);

    if (pdf) *pdf = p * cond0->mInvSizeCdf / w;

    // Return location which is a fraction t between texel centers i, i+1
    return min(((float)i + 0.5f + t) * cond0->mInvSizeCdf, sOneMinusEpsilon);
}


void
Distribution2D_sampleBilinear(const uniform Distribution2D * uniform dis, varying float ru, varying float rv,
                              varying Vec2f * uniform uv, varying float * uniform pdf)
{
    MNRY_ASSERT(uv != nullptr);

    varying float u, v, pMarg, pCond;

    v = GuideDistribution1D_sampleLinear(dis->mMarginal, rv, &pMarg);

    uniform int n = dis->mMarginal->mSizeCdf;

    varying int i1;
    varying float s;
    intAndFrac(v * n, &i1, &s);
    varying int i0 = i1 - 1;

    i0 = max(i0, 0);
    i1 = min(i1, n-1);

    u = sampleLinearBlended(ru, &pCond, dis->mConditional[i0], dis->mConditional[i1], s);

    uv->x = u;
    uv->y = v;
    if (pdf) *pdf = pMarg * pCond;
}


void
Distribution2D_sampleBilinear(const uniform Distribution2D * varying dis, varying float ru, varying float rv,
                              varying Vec2f * uniform uv, varying float * uniform pdf)
{
    MNRY_ASSERT(uv != nullptr);

    varying float u, v, pMarg, pCond;

    v = GuideDistribution1D_sampleLinear(dis->mMarginal, rv, &pMarg);

    varying int n = dis->mMarginal->mSizeCdf;

    varying int i1;
    varying float s;
    intAndFrac(v * n, &i1, &s);
    varying int i0 = i1 - 1;

    i0 = max(i0, 0);
    i1 = min(i1, n-1);

    u = sampleLinearBlended(ru, &pCond, dis->mConditional[i0], dis->mConditional[i1], s);

    uv->x = u;
    uv->y = v;
    if (pdf) *pdf = pMarg * pCond;
}



//----------------------------------------------------------------------------

// Lookup pixel by a pair of indices and a mip level.
// This function is called by both textureLookupDirect() and textureLookupTransformed(),
// and assumes that the work of treating u,v values outside [0,1] has already been done.
// This ensures that converting (u,v) to a texel index will generate a good result. The
// only exceptions are the edge cases u=1 and v=1, hence the testing against mWidth and
// mHeight.

varying Color
lookup(const uniform ImageDistribution * uniform dis, const varying int xi, const varying int yi,
       const varying int mipLevel)
{
    varying int mipWidth  = dis->mWidth  >> mipLevel;
    varying int mipHeight = dis->mHeight >> mipLevel;

    varying int xi_ = (xi < 0) ? 0 : (xi >= mipWidth ) ? mipWidth  - 1 : xi;
    varying int yi_ = (yi < 0) ? 0 : (yi >= mipHeight) ? mipHeight - 1 : yi;

    varying int index = (yi_ * mipWidth + xi_) * 3;

    // TODO: make mPixelBuffer be of type Color[] and then we won't need this clunky 3-float copy operaiton
    varying Color color;
    color.r = dis->mPixelBuffer[mipLevel][index];
    color.g = dis->mPixelBuffer[mipLevel][index + 1];
    color.b = dis->mPixelBuffer[mipLevel][index + 2];

    return color;
}

varying Color
ImageDistribution_filterNearest(const uniform ImageDistribution * uniform dis, 
                                const varying float u, const varying float v)
{
    MNRY_ASSERT(u >= 0.0f  &&  u <= 1.0f);
    MNRY_ASSERT(v >= 0.0f  &&  v <= 1.0f);

    varying float x = u * dis->mWidth;
    varying float y = v * dis->mHeight;
    varying int xi = (varying int)floor(x);
    varying int yi = (varying int)floor(y);

    return lookup(dis, xi, yi, 0);
}

varying Color
ImageDistribution_filterBilinear(const uniform ImageDistribution * uniform dis,
                                 const varying float u, const varying float v)
{
    MNRY_ASSERT(u >= 0.0f  &&  u <= 1.0f);
    MNRY_ASSERT(v >= 0.0f  &&  v <= 1.0f);

    varying int xi, yi;
    varying float xf, yf;
    intAndFrac(u * dis->mWidth  - 0.5f, &xi, &xf);
    intAndFrac(v * dis->mHeight - 0.5f, &yi, &yf);

    varying Color color00 = lookup(dis, xi,   yi,   0);
    varying Color color10 = lookup(dis, xi+1, yi,   0);
    varying Color color01 = lookup(dis, xi,   yi+1, 0);
    varying Color color11 = lookup(dis, xi+1, yi+1, 0);

    return bilerp(color00, color10, color01, color11, xf, yf);
}

varying Color
ImageDistribution_filterNearestMipNearest(const uniform ImageDistribution * uniform dis, 
                                          const varying float u, const varying float v, const varying float mipLevel)
{
    MNRY_ASSERT(u >= 0.0f  &&  u <= 1.0f);
    MNRY_ASSERT(v >= 0.0f  &&  v <= 1.0f);

    varying float m = clamp(mipLevel, 0.0f, (varying float)(dis->mNumMipLevels-1));
    varying int mi = (varying int)round(m);

    varying int mipWidth  = dis->mWidth  >> mi;
    varying int mipHeight = dis->mHeight >> mi;

    varying float x = u * mipWidth;
    varying float y = v * mipHeight;
    varying int xi = (varying int)floor(x);
    varying int yi = (varying int)floor(y);

    return lookup(dis, xi, yi, mi);

}

varying Color
ImageDistribution_filterBilinearMipNearest(const uniform ImageDistribution * uniform dis, 
                                           const varying float u, const varying float v, const varying float mipLevel)
{
    MNRY_ASSERT(u >= 0.0f  &&  u <= 1.0f);
    MNRY_ASSERT(v >= 0.0f  &&  v <= 1.0f);

    varying float m = clamp(mipLevel, 0.0f, (varying float)(dis->mNumMipLevels-1));
    varying int mi = (varying int)round(m);

    varying int mipWidth  = dis->mWidth  >> mi;
    varying int mipHeight = dis->mHeight >> mi;

    varying int xi, yi;
    varying float xf, yf;
    intAndFrac(u * mipWidth  - 0.5f, &xi, &xf);
    intAndFrac(v * mipHeight - 0.5f, &yi, &yf);

    varying Color color00 = lookup(dis, xi,   yi,   mi);
    varying Color color10 = lookup(dis, xi+1, yi,   mi);
    varying Color color01 = lookup(dis, xi,   yi+1, mi);
    varying Color color11 = lookup(dis, xi+1, yi+1, mi);

    return bilerp(color00, color10, color01, color11, xf, yf);
}


varying Color
ImageDistribution_applyTexFilter(const uniform ImageDistribution * uniform dis,
                                 const varying float u, const varying float v, const varying float mipLevel,
                                 const uniform TextureFilterType texFilter)
{
    switch (texFilter) {
    case TEXTURE_FILTER_NEAREST:
        return ImageDistribution_filterNearest(dis, u, v);
    case TEXTURE_FILTER_BILINEAR:
        return ImageDistribution_filterBilinear(dis, u, v);
    case TEXTURE_FILTER_NEAREST_MIP_NEAREST:
        return ImageDistribution_filterNearestMipNearest(dis, u, v, mipLevel);
    case TEXTURE_FILTER_BILINEAR_MIP_NEAREST:
        return ImageDistribution_filterBilinearMipNearest(dis, u, v, mipLevel);
    default:
        return ImageDistribution_filterNearest(dis, u, v);
    }
}


// Lookup from untransformed texture
varying Color
ImageDistribution_textureLookupDirect(const uniform ImageDistribution * uniform dis,
                                      const varying float u, const varying float v, const varying float mipLevel,
                                      const uniform TextureFilterType texFilter)
{
    // Apply border color outside the texture
    if (u < 0.0f || u > 1.0f || v < 0.0f || v > 1.0f) {
        return dis->mBorderColor;
    }

    // Ensure there's a texture
    if (dis->mPixelBuffer == nullptr) {
        return sWhite;
    }

    return ImageDistribution_applyTexFilter(dis, u, v, mipLevel, texFilter);
}


// Lookup from transformed texture
varying Color
ImageDistribution_textureLookupTransformed(const uniform ImageDistribution * uniform dis,
                                           const varying float uUntransformed, const varying float vUntransformed,
                                           const varying float mipLevel,
                                           const uniform TextureFilterType texFilter)
{
    // Apply transformation
    varying Vec3f uv = Vec3f_ctor(uUntransformed, vUntransformed, 1.0f) * dis->mTransformation;
    varying float u = uv.x;
    varying float v = uv.y;

    // Apply border color outside the texture
    if (u < 0.0f || u > dis->mRepsU || v < 0.0f || v > dis->mRepsV) {
        return dis->mBorderColor;
    }

    // Ensure there's a texture
    if (dis->mPixelBuffer == nullptr) {
        return sWhite;
    }

    // Perform wrapping and/or mirroring
    varying float uFloor = floor(u);
    varying float vFloor = floor(v);
    u -= uFloor;
    v -= vFloor;
    if (dis->mMirrorU && ((int)uFloor & 1)) {
        u = 1.0f - u;
    }
    if (dis->mMirrorV && ((int)vFloor & 1)) {
        v = 1.0f - v;
    }

    return ImageDistribution_applyTexFilter(dis, u, v, mipLevel, texFilter);
}


varying Color
ImageDistribution_eval(const uniform ImageDistribution * uniform dis,
                       const varying float u, const varying float v, const varying float mipLevel,
                       const uniform TextureFilterType texFilter)
{
    MNRY_ASSERT(ImageDistribution_isValid(dis));
    if (dis->mIsTransformed) {
        return ImageDistribution_textureLookupTransformed(dis, u, v, mipLevel, texFilter);
    } else {
        return ImageDistribution_textureLookupDirect(dis, u, v, mipLevel, texFilter);
    }
}


varying float
ImageDistribution_pdfNearest(const uniform ImageDistribution * uniform dis,
                             const varying float u, const varying float v)
{
    // We look up from the finest lod, which is mip level 0
    return Distribution2D_pdfNearest(dis->mDistribution[0], u, v);
}


varying float
ImageDistribution_pdfBilinear(const uniform ImageDistribution * uniform dis,
                              const varying float u, const varying float v)
{
    // We look up from the finest lod, which is mip level 0
    return Distribution2D_pdfBilinear(dis->mDistribution[0], u, v);
}


varying float
ImageDistribution_pdfNearestMipNearest(const uniform ImageDistribution * uniform dis,
                                       const varying float u, const varying float v, const varying float mipLevel)
{
    varying float m = clamp(mipLevel, 0.0f, (varying float)(dis->mNumMipLevels-1));
    varying int mi = (varying int)round(m);
    return Distribution2D_pdfNearest(dis->mDistribution[mi], u, v);
}


varying float
ImageDistribution_pdfBilinearMipNearest(const uniform ImageDistribution * uniform dis,
                                        const varying float u, const varying float v, const varying float mipLevel)
{
    varying float m = clamp(mipLevel, 0.0f, (varying float)(dis->mNumMipLevels-1));
    varying int mi = (varying int)round(m);
    return Distribution2D_pdfBilinear(dis->mDistribution[mi], u, v);
}


varying float
ImageDistribution_pdf(const uniform ImageDistribution * uniform dis, const varying float u, const varying float v,
                      const varying float mipLevel, const uniform TextureFilterType texFilter)
{
    MNRY_ASSERT(ImageDistribution_isValid(dis));
    switch (texFilter) {
    case TEXTURE_FILTER_NEAREST:
        return ImageDistribution_pdfNearest(dis, u, v);
    case TEXTURE_FILTER_BILINEAR:
        return ImageDistribution_pdfBilinear(dis, u, v);
    case TEXTURE_FILTER_NEAREST_MIP_NEAREST:
        return ImageDistribution_pdfNearestMipNearest(dis, u, v, mipLevel);
    case TEXTURE_FILTER_BILINEAR_MIP_NEAREST:
        return ImageDistribution_pdfBilinearMipNearest(dis, u, v, mipLevel);
    default:
        return ImageDistribution_pdfNearest(dis, u, v);
    }
}


void
ImageDistribution_sampleNearest(const uniform ImageDistribution * const uniform dis,
                                const varying float ru, const varying float rv,
                                varying Vec2f * uniform uv, varying float * uniform pdf)
{
    // We sample the finest lod, which is mip level 0
    Distribution2D_sampleNearest(dis->mDistribution[0], ru, rv, uv, pdf);
}


void
ImageDistribution_sampleBilinear(const uniform ImageDistribution * const uniform dis,
                                 const varying float ru, const varying float rv,
                                 varying Vec2f * uniform uv, varying float * uniform pdf)
{
    // We sample the finest lod, which is mip level 0
    Distribution2D_sampleBilinear(dis->mDistribution[0], ru, rv, uv, pdf);
}


void
ImageDistribution_sampleNearestMipNearest(const uniform ImageDistribution * const uniform dis,
                                          const varying float ru, const varying float rv, const varying float mipLevel,
                                          varying Vec2f * uniform uv, varying float * uniform pdf)
{
    varying float m = clamp(mipLevel, 0.0f, (varying float)(dis->mNumMipLevels-1));
    varying int mi = (varying int)round(m);
    Distribution2D_sampleNearest(dis->mDistribution[mi], ru, rv, uv, pdf);
}


void
ImageDistribution_sampleBilinearMipNearest(const uniform ImageDistribution * const uniform dis,
                                           const varying float ru, const varying float rv, const varying float mipLevel,
                                           varying Vec2f * uniform uv, varying float * uniform pdf)
{
    varying float m = clamp(mipLevel, 0.0f, (varying float)(dis->mNumMipLevels-1));
    varying int mi = (varying int)round(m);
    Distribution2D_sampleBilinear(dis->mDistribution[mi], ru, rv, uv, pdf);
}


void
ImageDistribution_sample(const uniform ImageDistribution * const uniform dis,
                         const varying float ru, const varying float rv,
                         const varying float mipLevel,
                         varying Vec2f * uniform uv, varying float * uniform pdf,
                         const uniform TextureFilterType texFilter)
{
    MNRY_ASSERT(ImageDistribution_isValid(dis));
    switch (texFilter) {
    case TEXTURE_FILTER_NEAREST:
        ImageDistribution_sampleNearest(dis, ru, rv, uv, pdf);
        break;
    case TEXTURE_FILTER_BILINEAR:
        ImageDistribution_sampleBilinear(dis, ru, rv, uv, pdf);
        break;
    case TEXTURE_FILTER_NEAREST_MIP_NEAREST:
        ImageDistribution_sampleNearestMipNearest(dis, ru, rv, mipLevel, uv, pdf);
        break;
    case TEXTURE_FILTER_BILINEAR_MIP_NEAREST:
        ImageDistribution_sampleBilinearMipNearest(dis, ru, rv, mipLevel, uv, pdf);
        break;
    default:
        ImageDistribution_sampleNearest(dis, ru, rv, uv, pdf);
        break;
    }
}






//----------------------------------------------------------------------------

