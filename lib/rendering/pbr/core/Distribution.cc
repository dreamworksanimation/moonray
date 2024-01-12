// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file Distribution.cc
/// $Id$
///
#include "Distribution.h"
#include "Util.h"

#include <moonray/rendering/pbr/core/Distribution_ispc_stubs.h>
#include <moonray/rendering/pbr/core/ImageColorCorrect_ispc_stubs.h>
#include <moonray/rendering/pbr/core/Util_ispc_stubs.h>
#include <moonray/rendering/rndr/OiioUtils.h>

#include <scene_rdl2/common/except/exceptions.h>
#include <scene_rdl2/common/math/MathUtil.h>
#include <scene_rdl2/render/logging/logging.h>
#include <scene_rdl2/render/util/Memory.h>
#include <scene_rdl2/common/platform/HybridUniformData.h>
#include <OpenImageIO/imageio.h>
#include <OpenImageIO/imagebuf.h>
#include <OpenImageIO/imagebufalgo.h>
#include <tbb/parallel_for.h>

#include <cstring>

// TODO: rethink the idea of recovering pdf values by diffing cdf values. Jeff Mahovsky points out that it has the
// potential for highly imprecise pdf values due to catastrophic cancellation.

// TODO: strip out the unused Distribution classes, since we'll probably never use them.


namespace moonray {
namespace pbr {

using namespace scene_rdl2::math;
using scene_rdl2::logging::Logger;

static const int sRangeDivider = 80;


finline void
intAndFrac(float x, int *xInt, float *xFrac)
{
    MNRY_ASSERT(xInt != nullptr && xFrac != nullptr);
    float xFloor = scene_rdl2::math::floor(x);
    *xInt = static_cast<int>(xFloor);
    *xFrac = x - xFloor;
}


//----------------------------------------------------------------------------

HUD_VALIDATOR(GuideDistribution1D);

// Note: the guide table size is arbitrary and doesn't have to be the same as
// the cdf table. If its size is K * size, then the expected number of
// comparisons in the while() loop in the sample() methods is provably
//     1 + 1 / K
// See: Devroye, L. (1986). Non-uniform Random Variate Generation. Springer.
GuideDistribution1D::GuideDistribution1D(size_type size) :
GuideDistribution1D(size, nullptr, nullptr)
{
    // Or pass these allocations in instead of _nullptr_ ...
    mCdf = scene_rdl2::util::alignedMallocArray<float>(mSizeCdf);
    mGuide = scene_rdl2::util::alignedMallocArray<size_type>(mSizeGuide);
    mOwnsArrays = true;
}


// Note: the cdf will be tabulated in-place (by calling tabulateCdf()) in the memory occupied by the input pdf.
// Both are stored non-normalised, i.e. the pdf values don't need to sum to 1 (essentially they are
// just the function values, which the true pdf is proportional to)
GuideDistribution1D::GuideDistribution1D(const size_type size, float *const pdf, uint32_t *const guide) :
    mSizeCdf(size),
    mSizeGuide(size),
    mInvSizeCdf(1.0f / mSizeCdf),
    mInvSizeGuide(1.0f / mSizeGuide),
    mTotalWeight(1.0f),
    mInvTotalWeight(1.0f),
    mThresholdLow(0.0f),
    mThresholdHigh(1.0f),
    mLinearCoeffLow(0.0f),
    mLinearCoeffHigh(0.0f),
    mCdf(pdf),
    mGuide(guide),
    mOwnsArrays(false)
{
}


GuideDistribution1D::GuideDistribution1D(GuideDistribution1D&& other) :
    mSizeCdf(other.mSizeCdf),
    mSizeGuide(other.mSizeGuide),
    mInvSizeCdf(other.mInvSizeCdf),
    mInvSizeGuide(other.mInvSizeGuide),
    mTotalWeight(other.mTotalWeight),
    mInvTotalWeight(other.mInvTotalWeight),
    mThresholdLow(other.mThresholdLow),
    mThresholdHigh(other.mThresholdHigh),
    mLinearCoeffLow(other.mLinearCoeffLow),
    mLinearCoeffHigh(other.mLinearCoeffHigh),
    mCdf(other.mCdf),
    mGuide(other.mGuide)
{
    other.mSizeCdf = 0;
    other.mSizeGuide = 0;
    other.mInvSizeCdf = 0.0f;
    other.mInvSizeGuide = 0.0f;
    other.mTotalWeight = 0.0f;
    other.mInvTotalWeight = 0.0f;
    other.mThresholdLow = 0.0f;
    other.mThresholdHigh = 0.0f;
    other.mLinearCoeffLow = 0.0f;
    other.mLinearCoeffHigh = 0.0f;
    other.mCdf = nullptr;
    other.mGuide = nullptr;
}


GuideDistribution1D& GuideDistribution1D::operator=(GuideDistribution1D&& other)
{
    mSizeCdf = other.mSizeCdf;
    mSizeGuide = other.mSizeGuide;
    mInvSizeCdf = other.mInvSizeCdf;
    mInvSizeGuide = other.mInvSizeGuide;
    mTotalWeight = other.mTotalWeight;
    mInvTotalWeight = other.mInvTotalWeight;
    mThresholdLow = other.mThresholdLow;
    mThresholdHigh = other.mThresholdHigh;
    mLinearCoeffLow = other.mLinearCoeffLow;
    mLinearCoeffHigh = other.mLinearCoeffHigh;
    mCdf = other.mCdf;
    mGuide = other.mGuide;

    other.mSizeCdf = 0;
    other.mSizeGuide = 0;
    other.mInvSizeCdf = 0.0f;
    other.mInvSizeGuide = 0.0f;
    other.mTotalWeight = 0.0f;
    other.mInvTotalWeight = 0.0f;
    other.mThresholdLow = 0.0f;
    other.mThresholdHigh = 0.0f;
    other.mLinearCoeffLow = 0.0f;
    other.mLinearCoeffHigh = 0.0f;
    other.mCdf = nullptr;
    other.mGuide = nullptr;

    return *this;
}


GuideDistribution1D::~GuideDistribution1D()
{
    if (mOwnsArrays) {
        scene_rdl2::util::alignedFreeArray<float> (mCdf);
        mCdf = nullptr;
        scene_rdl2::util::alignedFreeArray<uint32_t> (mGuide);
        mGuide = nullptr;
    }
}


//----------------------------------------------------------------------------

float
GuideDistribution1D::tabulateCdf()
{
    // Find sum of weights
    float sum = 0.0f;
    for (size_type i = 0; i < mSizeCdf; i++) {
        MNRY_ASSERT(mCdf[i] >= 0.0f);
        sum += mCdf[i];
    }
 
    // Input function is represented as weights stored in the elements of mCdf.
    // Accumulate weights to make a homogeneous cdf.
    // (Regular cdf can be recovered by multiplying by mInvTotalWeight.)
    if (sum != 0.0f) {
        float partialSum = 0.0f;
        for (size_type i=0; i < mSizeCdf-1; i++) {
            partialSum += mCdf[i];
            if (partialSum > sum) {
                // This is unlikely to occur, but it is theoretically possible due to rounding (because of differences
                // in object code generated for this loop versus the one that computed sum)
                partialSum = sum;
            }
            mCdf[i] = partialSum;
        }
    } else {
        // Cater to edge case of all-zero weights
        for (size_type i = 0; i < mSizeCdf-1; i++) {
            mCdf[i] = (i + 1) * mInvSizeCdf;
        }
        sum = 1.0f;
    }

    // Store sum as final table entry. (Note that we don't include this line in the loop that builds the cdf,
    // becuse that loop actually produces a slightly different final sum due to differences in object code generated.)
    mCdf[mSizeCdf-1] = sum;

    // Store total weight and its reciprocal
    mTotalWeight = sum;
    mInvTotalWeight = 1.0f / sum;

    // Compute guide table
    size_type i = 0;
    for (size_type j=0; j < mSizeGuide; ++j) {

        // Find min i such that cdf[i] >= j * mTotalWeight / mSizeGuide
        const float tau = j * mTotalWeight * mInvSizeGuide;
        const size_type iBound = mSizeCdf - 1;
        while (i < iBound  &&  mCdf[i] < tau) {
            ++i;
        }
        mGuide[j] = i;
    }

    // Set helper values for sampleLinear() function.
    // The cdf curve is piecewise quadratic, but it has linear segments at the left and right ends.
    // The values here are for the early-outs in sampleLinear() that deal with these linear cases.
    // mThresholdLow is the threshold value which the input random variable r must not exceed for it to fall within
    // the first half-texel, for which a linear segment is used. The gradient of that segment is then stored in
    // mLinearCoeffLow for easy evaluation in that case.
    // The values mThresholdHigh and mLinearCoeffHigh are used in a corresponding way for the final half-texel.
    mThresholdLow  = 0.5f * mCdf[0] * mInvTotalWeight;
    mThresholdHigh = (mSizeCdf > 1) ? 0.5f * (mCdf[mSizeCdf-2] + mCdf[mSizeCdf-1]) * mInvTotalWeight
                                    : mThresholdLow;
    mLinearCoeffLow  = (mThresholdLow  != 0.0f) ? 0.5f * mInvSizeCdf / mThresholdLow : 0.0f;
    mLinearCoeffHigh = (mThresholdHigh != 1.0f) ? 0.5f * mInvSizeCdf / (1.0f - mThresholdHigh) :  0.0f;

    // Return the integral of the input function
    return sum * mInvSizeCdf;
}


float 
GuideDistribution1D::pdfDiscrete(size_type index) const
{
    const float prevCdf = (index > 0)  ?  mCdf[index - 1]  :  0.0f;
    return (mCdf[index] - prevCdf) * mInvTotalWeight;
}


float
GuideDistribution1D::pdfContinuous(float u) const
{
    size_type i = static_cast<size_type>(u * mSizeCdf);
    i = scene_rdl2::math::min(i, mSizeCdf - 1);
    const float prevCdf = (i > 0)  ?  mCdf[i - 1]  :  0.0f;
    return (mCdf[i] - prevCdf) * mInvTotalWeight * mSizeCdf;
}


float
GuideDistribution1D::pdfLinear(float u) const
{
    if (u < 0.0f || u > 1.0f) return 0.0f;
    if (mSizeCdf == 1) return 1.0f;

    int i;
    float t;

    // Here we must add 0.5 to the texture coordinate (after scaling it to be in units of texels). This is because
    // we're evaluating the texture as a piecewise-linear function, with a new linear segment for each texel, but in
    // particular the segments join at the texel centers, whose positions have a fractional part of 0.5. Thus we want
    // to advance the index i whenever the fractional part reaches 0.5. (Compare this with the situation for nearest
    // neighbour textures, where the segments join a the texel boundaries, whose positions have a fractional part of
    // 0.0.)
    intAndFrac(u * mSizeCdf + 0.5f, &i, &t);

    float p;
    if (i == 0) {
        // Clamp to lower border
        p = mCdf[0];
    }
    else if (i == mSizeCdf) {
        // Clamp to upper border
        p = mCdf[mSizeCdf-1] - mCdf[mSizeCdf-2];
    }
    else {
        // Lerp between 2 values
        float p0 = (i == 1) ? mCdf[0] : mCdf[i-1] - mCdf[i-2];
        float p1 = mCdf[i] - mCdf[i-1];
        p  = lerp(p0, p1, t);
    }

    return p * mInvTotalWeight * mSizeCdf;
}


// Returns the index of the interval in the discrete distribution where the value r falls.
// rRemapped represents the fractional distance of r along that interval. 
GuideDistribution1D::size_type
GuideDistribution1D::sampleDiscrete(float r, float *const pdf, float *const rRemapped) const
{
    size_type i = getGuideIndex(r);
    float rw = r * mTotalWeight;        // Scale here because cdf is non-normalized
    while (mCdf[i] < rw) {
        ++i;
    }
    MNRY_ASSERT(i < mSizeCdf);
    const float prevCdf = (i > 0) ? mCdf[i - 1] : 0.0f;
    const float funcVal = mCdf[i] - prevCdf;
    if (pdf) *pdf = funcVal * mInvTotalWeight;
    if (rRemapped) *rRemapped = (rw - prevCdf) / funcVal;
    return i;
}


float
GuideDistribution1D::sampleContinuous(float r, float *pdf, size_type *index) const
{
    size_type i = getGuideIndex(r);
    float rw = r * mTotalWeight;        // Scale here because cdf is non-normalized
    while (mCdf[i] < rw) {
        ++i;
    }
    MNRY_ASSERT(i < mSizeCdf);

    if (index) *index = i;

    const float prevCdf = (i > 0)  ?  mCdf[i-1]  :  0.0f;
    const float funcVal = mCdf[i] - prevCdf;

    // Lerp inverted cdf value
    // (Note here that an expression of the form lerp(i,i+1,t) has been replaced with the simpler and more
    // efficient expression i+t)
    float invCdf = (float(i) + (rw - prevCdf) / funcVal) * mInvSizeCdf;

    // May reach 1.0 due to float imprecision
    invCdf = min(invCdf, sOneMinusEpsilon);

    if (pdf) *pdf = funcVal * mInvTotalWeight * mSizeCdf;

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
    float denom = -B - scene_rdl2::math::sqrt(discrim);

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

float
GuideDistribution1D::sampleLinear(float r, float *pdf) const
{
    MNRY_ASSERT(r >= 0.0f  &&  r <= 1.0f);
  
    // Linear interpolation in 1st half-texel
    if (r <= mThresholdLow) {
        return min(r * mLinearCoeffLow, sOneMinusEpsilon);
    }

    // Linear interpolation in last half-texel
    if (r >= mThresholdHigh) {
        return min(1.0f - (1.0f - r) * mLinearCoeffHigh, sOneMinusEpsilon);
    }

    // Look up in guide table.
    // The guide table entries are optimal for nearest-neighbour textures, not for linear ones. It would be
    // possible to add in a second table, but the added memory cost and code support would outweigh the very
    // slight performance gain, so we just use the existing table.
    int i = getGuideIndex(r);
    MNRY_ASSERT(i < mSizeCdf);

    // Early-out in 1st linear case above means we don't need to search starting at i=0
    i = max(i, 1);

    float twoWr = 2.0f * mTotalWeight * r;

    // Step index until we find the curve segment which r intersects with
    float c0 = (i >= 2) ? mCdf[i-2] : 0.0f;
    float c1 = mCdf[i-1];
    float c2 = mCdf[i];
    while (c1 + c2 < twoWr)  // We use precomputed 2*wr to avoid multiplying c1+c2 here by 0.5.
    {
        ++i;
        MNRY_ASSERT(i < mSizeCdf);
        c0 = c1;
        c1 = c2;
        c2 = mCdf[i];
    }

    // Solve for intersection with selected curve segment.
    // The curve segments are necessarily quadratic, because the cdf is the integral of the pdf
    // which in this case is piecewise linear.
    float p;
    float t = quadraticHelper(c0, c1, c2, twoWr, p);

    if (pdf) *pdf = p * mInvTotalWeight * mSizeCdf;

    // Return location which is a fraction t between texel centers i, i+1
    return min((static_cast<float>(i) + 0.5f + t) * mInvSizeCdf, sOneMinusEpsilon);
}



//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

HUD_VALIDATOR(Distribution2D);

Distribution2D::Distribution2D(size_type sizeU, size_type sizeV) :
    mSizeV(sizeV),
    mConditional(NULL),
    mMarginal(NULL)
{
    // Allocate 1D distributions
    mConditional = scene_rdl2::util::alignedMallocArray<Distribution1D*>(mSizeV);
    for (size_type i = 0; i < mSizeV; ++i) {
        mConditional[i] = scene_rdl2::util::alignedMallocCtorArgs<Distribution1D>(DEFAULT_MEMORY_ALIGNMENT, sizeU);
    }
    mMarginal = scene_rdl2::util::alignedMallocCtorArgs<Distribution1D>(DEFAULT_MEMORY_ALIGNMENT, sizeV);
}


Distribution2D::~Distribution2D()
{
    for (size_type i = 0; i < mSizeV; ++i) {
        scene_rdl2::util::alignedFreeDtor<Distribution1D>(mConditional[i]);
    }
    scene_rdl2::util::alignedFreeArray<Distribution1D*> (mConditional);
    mConditional = nullptr;
    scene_rdl2::util::alignedFreeDtor<Distribution1D> (mMarginal);
    mMarginal = nullptr;
}


//----------------------------------------------------------------------------

void
Distribution2D::tabulateCdf(Mapping mapping)
{
    size_type sizeU = getSizeU();
    size_type sizeV = getSizeV();

    // Modify weights to respect mappings right before we compute the final CDF.
    switch (mapping) {
    case SPHERICAL: {
        // Note: We are slighly pulling in the extents so we don't get full black for
        //       the top and bottom scanlines
        float sclV, ofsV;
        getScaleOffset(-0.5f, float(sizeV) - 0.5f, 0.0f, sPi, &sclV, &ofsV);
        tbb::parallel_for(tbb::blocked_range<size_type>(0, sizeV, sizeV / sRangeDivider),
                          [&](const tbb::blocked_range<size_type> range) {
            for (size_type y = range.begin(); y < range.end(); ++y) {
                // The sinTheta term accounts for the distortion of the image mapping
                // onto the sphere into the sampling pdf: two pixels of equal brightness
                // should not be sampled similarly if one is close to the sphere pole
                // and the other close to the equator (see pbrt section 14.6.5)
                float sinTheta = scene_rdl2::math::sin(float(y) * sclV + ofsV);
                for (size_type x = 0; x < sizeU; ++x) {
                    scaleWeight(x, y, max(sinTheta, sEpsilon));
                }
            }
        });
        break;
    }

    case HEMISPHERICAL: {
        // Note: We are slighly pulling in the extents so we don't get full black for
        //       the top and bottom scanlines
        float sclV, ofsV;
        getScaleOffset(-0.5f, float(sizeV) - 0.5f, 0.0f, sPi, &sclV, &ofsV);
        unsigned halfSizeV = sizeV / 2;

        tbb::parallel_for(tbb::blocked_range<size_type>(0, sizeV, sizeV / sRangeDivider),
                          [&](const tbb::blocked_range<size_type> range) {
            for (size_type y = range.begin(); y < range.end(); ++y) {
                // The sinTheta term accounts for the distortion of the image mapping
                // onto the sphere into the sampling pdf: two pixels of equal brightness
                // should not be sampled similarly if one is close to the sphere pole
                // and the other close to the equator (see pbrt section 14.6.5)
                float sinTheta = (y >= halfSizeV) ? scene_rdl2::math::sin(float(y) * sclV + ofsV) : sEpsilon;
                for (size_type x = 0; x < sizeU; ++x) {
                    scaleWeight(x, y, sinTheta);
                }
            }
        });
        break;
    }

    case PLANAR: {
        // Nothing to do!
        break;
    }

    case CIRCULAR: {
        // Weights outside of the maximal inscribed circle are set to black so
        // we don't (or rarely) sample them.
        // note: We are slighly pushing out the extents so we don't get full
        //       black for the edge scanlines.
        float sclU, ofsU, sclV, ofsV;
        getScaleOffset(-0.5f, float(sizeU) - 0.5f, -1.0f, 1.0f, &sclU, &ofsU);
        getScaleOffset(-0.5f, float(sizeV) - 0.5f, -1.0f, 1.0f, &sclV, &ofsV);

        tbb::parallel_for(tbb::blocked_range<size_type>(0, sizeV, sizeV / sRangeDivider),
                          [&](const tbb::blocked_range<size_type> range) {
            for (size_type y = range.begin(); y < range.end(); ++y) {
                float t = float(y) * sclV + ofsV;
                float t2 = t * t;
                for (size_type x = 0; x < sizeU; ++x) {
                    float s = float(x) * sclU + ofsU;
                    if (s*s + t2 > 1.0f) {
                        setWeight(x, y, sEpsilon);
                    }
                }
            }
        });
        break;
    }
    default: {
        MNRY_ASSERT(false, "Undefined mapping in Distribution2D::tabulateCdf()");
    }
    }

    // Now compute CDF.
    tbb::parallel_for(tbb::blocked_range<size_type>(0, sizeV, sizeV / sRangeDivider),
                      [&](const tbb::blocked_range<size_type> range) {
        for (size_type y = range.begin(); y < range.end(); ++y) {
            float integral = mConditional[y]->tabulateCdf();
            mMarginal->setWeight(y, integral);
        }
    });
    mMarginal->tabulateCdf();
}


float
Distribution2D::pdfNearest(float u, float v) const
{
    size_type vSize = mMarginal->getSize();
    size_type vIndex = size_type(v * vSize);
    vIndex = scene_rdl2::math::min(vIndex, vSize - 1);

    return mMarginal->pdfContinuous(v) *
           mConditional[vIndex]->pdfContinuous(u);
}


float
Distribution2D::pdfBilinear(float u, float v) const
{
    if (u < 0.0f || u >= 1.0f || v < 0.0f || v >= 1.0f) return 0.0f;

    float pMarg = mMarginal->pdfContinuous(v);

    int n = mMarginal->getSize();

    int i1;
    float s;
    intAndFrac(v * n + 0.5f, &i1, &s);
    int i0 = i1 - 1;

    i0 = scene_rdl2::math::max(i0, 0);
    i1 = scene_rdl2::math::min(i1, n-1);

    float pCond0 = mConditional[i0]->pdfContinuous(u);
    float pCond1 = mConditional[i1]->pdfContinuous(u);

    float pCond = lerp(pCond0, pCond1, s);

    return pMarg * pCond;
}


void
Distribution2D::sampleNearest(float ru, float rv, scene_rdl2::math::Vec2f *uv, float *pdf) const
{
    MNRY_ASSERT(uv != nullptr);
    size_type v;
    float pdfMarg, pdfCond;
    uv->y = mMarginal->sampleContinuous(rv, &pdfMarg, &v);
    uv->x = mConditional[v]->sampleContinuous(ru, &pdfCond);

    if (pdf) *pdf = pdfMarg * pdfCond;
}


// A helper function that works like GuideDistribution1D::sampleLinear() except that it linearly blends
// the coefficients of 2 cdfs (presumably 2 adjacent rows of the conditional cdf table) by the parameter s
float
sampleLinearBlended(float r, float *pdf, const GuideDistribution1D *cond0, const GuideDistribution1D *cond1, float s)
{
    MNRY_ASSERT(r >= 0.0f  &&  r <= 1.0f);
  
    // Linear interpolation in 1st half-texel
    if (r <= lerp(cond0->getThresholdLow(), cond1->getThresholdLow(), s)) {
        return min(r * lerp(cond0->getLinearCoeffLow(), cond1->getLinearCoeffLow(), s), sOneMinusEpsilon);
    }

    // Linear interpolation in last half-texel
    if (r >= lerp(cond0->getThresholdHigh(), cond1->getThresholdHigh(), s)) {
        return min(1.0f - (1.0f - r) * lerp(cond0->getLinearCoeffHigh(), cond1->getLinearCoeffHigh(), s),
                   sOneMinusEpsilon);
    }
      
    // Look up in guide tables
    int i = min(cond0->getGuideIndex(r), cond1->getGuideIndex(r));
    MNRY_ASSERT(i < cond0->getSize());

    // Early-out in 1st linear case above means we don't need to search starting at i=0
    i = max(i, 1);

    float w = lerp(cond0->getTotalWeight(), cond1->getTotalWeight(), s);
    float twoWr = 2.0f * w * r;
    const float *cdf0 = cond0->getCdf();
    const float *cdf1 = cond1->getCdf();

    // Step index until we find the curve segment which r intersects with
    float c0 = (i >= 2) ? lerp(cdf0[i-2], cdf1[i-2], s) : 0.0f;
    float c1 = lerp(cdf0[i-1], cdf1[i-1], s);
    float c2 = lerp(cdf0[i], cdf1[i], s);
    const int n = cond0->getSize();
    while (c1 + c2 < twoWr) {
        ++i;
        if (i >= n) {
            // When the true sample result is very close to the boundary between 2 table entries, rounding can
            // cause i to step too high by 1, and if this happens at the top end of the array it will be out of
            // bounds. If this case is detected we simply fall back on the linear interpolation in the uppermost
            // half-texel.
            return min(1.0f - (1.0f - r) * lerp(cond0->getLinearCoeffHigh(), cond1->getLinearCoeffHigh(), s),
                       sOneMinusEpsilon);
        }
        c0 = c1;
        c1 = c2;
        c2 = lerp(cdf0[i], cdf1[i], s);
    }

    // Solve for intersection with selected curve segment.
    // The curve segments are necessarily quadratic, because the cdf is the integral of the pdf
    // which in this case is piecewise linear.
    float p;
    float t = quadraticHelper(c0, c1, c2, twoWr, p);

    if (pdf) *pdf = p * cond0->getInvSize() / w;

    // Return location which is a fraction t between texel centers i, i+1
    return min((static_cast<float>(i) + 0.5f + t) * cond0->getInvSize(), sOneMinusEpsilon);
}


void
Distribution2D::sampleBilinear(float ru, float rv, scene_rdl2::math::Vec2f *uv, float *pdf) const
{
    MNRY_ASSERT(uv != nullptr);

    float u, v, pMarg, pCond;

    v = mMarginal->sampleLinear(rv, &pMarg);

    int n = mMarginal->getSize();

    int i1;
    float s;
    intAndFrac(v * n, &i1, &s);
    int i0 = i1 - 1;

    i0 = scene_rdl2::math::max(i0, 0);
    i1 = scene_rdl2::math::min(i1, n-1);

    u = sampleLinearBlended(ru, &pCond, mConditional[i0], mConditional[i1], s);

    uv->x = u;
    uv->y = v;
    if (pdf) *pdf = pMarg * pCond;
}


//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

HUD_VALIDATOR(ImageDistribution);

ImageDistribution::ImageDistribution(const std::string &mapFilename,
                  Distribution2D::Mapping mapping) :
    mDistribution(NULL),
    mWidth(0),
    mHeight(0),
    mPixelBuffer(NULL)
{
    init(mapFilename, mapping,
         sWhite, sWhite, sWhite, sWhite, sBlack, Vec3f(zero), 0.0f, Vec2f(0.0f, 0.0f), Vec2f(1.0f, 1.0f), 1, 1, false,
         false, sWhite);
}

ImageDistribution::ImageDistribution(const std::string &mapFilename,
                                     Distribution2D::Mapping mapping,
                                     const Color& gamma,
                                     const Color& contrast,
                                     const Color& saturation,
                                     const Color& gain,
                                     const Color& offset,
                                     const Vec3f& temperature,
                                     const float  rotationAngle,
                                     const Vec2f& translation,
                                     const Vec2f& coverage,
                                     const float  repsU,
                                     const float  repsV,
                                     const bool   mirrorU,
                                     const bool   mirrorV,
                                     const Color& borderColor) :
    mDistribution(NULL),
    mWidth(0),
    mHeight(0),
    mPixelBuffer(NULL)
{
    init(mapFilename, mapping, gamma, contrast, saturation, gain, offset,
         temperature, rotationAngle, translation, coverage,
         repsU, repsV, mirrorU, mirrorV, borderColor);
}

void
ImageDistribution::init(const std::string &mapFilename,
                         Distribution2D::Mapping mapping,
                         const Color& gamma,
                         const Color& contrast,
                         const Color& saturation,
                         const Color& gain,
                         const Color& offset,
                         const Vec3f& temperature,
                         const float  rotationAngle,
                         const Vec2f& translation,
                         const Vec2f& coverage,
                         const float  repsU,
                         const float  repsV,
                         const bool   mirrorU,
                         const bool   mirrorV,
                         const Color& borderColor)
{
    // Get unique_ptr to ImageInput - cleanup will be handled automatically
    rndr::OiioUtils::ImageInputUqPtr in = rndr::OiioUtils::openFile(mapFilename);

    // Throw exception if 'in' is null
    if (in == nullptr) {
        throw scene_rdl2::except::IoError("Image file \"" + mapFilename + "\" does not exist. Using black radiance");
    }

    // Get ImageSpec
    OIIO::ImageSpec inputImageSpec = in->spec();

    const int simdWidth = ispc::PBR_getSIMDWidth();
    float* inputPixelBuffer;
    std::vector<float *> pixelBuffers;

    // We'll need to check each level dims against those of the level above.
    // (It doesn't matter what we initialise these values to.)
    int widthLevelAbove  = 0;
    int heightLevelAbove = 0;

    // Loop over mip levels present in the file
    for (int mipLevel = 0; ; ++mipLevel) {

        // Seek mip level
        if ( ! in->seek_subimage(0, mipLevel, inputImageSpec)) {
            // mip level not found
            break;
        }

        const int inputLevelWidth  = inputImageSpec.width;
        const int inputLevelHeight = inputImageSpec.height;

        if (inputLevelWidth < 1 || inputLevelHeight < 1) break;

        // Temp workaround: 1x1 mip levels cause problems.
        // TODO: solve the problems and remove this workaround!
        if (inputLevelWidth == 1 && inputLevelHeight == 1) {
            Logger::warn("1x1 mip level not supported at present");
            break;
        }

        const int numPixels = inputLevelWidth * inputLevelHeight;

        // Note: we are not making any assumptions about the presence or absence of padding in the image itself;
        // we simply require the memory buffer to be a multiple of simdWidth so that simd writes work correctly
        const int paddedNumPixels = (numPixels + simdWidth - 1) & -simdWidth;

        if (mipLevel == 0) {
            // Record info for base level
            mWidth  = inputLevelWidth;
            mHeight = inputLevelHeight;

            // Allocate buffer for base level - all smaller levels can reuse it
            inputPixelBuffer = scene_rdl2::util::alignedMallocArray<float>(paddedNumPixels * inputImageSpec.nchannels);
        } else {
            // For smaller levels, verify the size is what we expect.
            // In the usual case, the dimensions should be exactly half those of the level above.
            // However, occasionally the level above has an odd dimension, in which case we should allow the
            // smaller size to be either rounded down or rounded up.
            bool widthOK = false;
            if (inputLevelWidth == widthLevelAbove >> 1) {
                // Standard case, or odd level above getting rounded down
                widthOK = true;
            }
            if (widthLevelAbove & 1) {
                if (inputLevelWidth == (widthLevelAbove >> 1) + 1) {
                    // Odd level above getting rounded up
                    widthOK = true;
                }
            }

            bool heightOK = false;
            if (inputLevelHeight == heightLevelAbove >> 1) {
                // Standard case, or odd level above getting rounded down
                heightOK = true;
            }
            if (heightLevelAbove & 1) {
                if (inputLevelHeight == (heightLevelAbove >> 1) + 1) {
                    // Odd level above getting rounded up
                    heightOK = true;
                }
            }

            if (!widthOK || !heightOK) {
                Logger::warn("Unexpected mip level dimensions in texture file.");
                break;
            }
        }

        // Use current dims to check dims of next level on the next pass
        widthLevelAbove  = inputLevelWidth;
        heightLevelAbove = inputLevelHeight;

        // Read image in floating point format
        bool readOk = in->read_image(OIIO::TypeDesc::FLOAT, inputPixelBuffer);

        // Error if it wasn't read properly
        if (!readOk) {
            scene_rdl2::util::alignedFreeArray<float> (inputPixelBuffer);
            inputPixelBuffer = nullptr;
            throw scene_rdl2::except::IoError("Cannot read image file: \"" + mapFilename +
                          "\" (" + in->geterror() + ")");
        }

        // Need to change the input format spec since we asked OIIO to convert it to float
        inputImageSpec.format = OIIO::TypeDesc::FLOAT;

        // The input image might not have 3 channels; specify 3 channels for the final image spec
        OIIO::ImageSpec finalImageSpec = inputImageSpec;
        finalImageSpec.nchannels = 3;

        // Allocate 3-channel pixel buffer to permanently hold the texture data for this mip level
        float *pixelBuffer = scene_rdl2::util::alignedMallocArray<float>(paddedNumPixels * 3);
        std::fill(pixelBuffer, pixelBuffer + paddedNumPixels * 3, 0.0f);
        pixelBuffers.push_back(pixelBuffer);

        // Corresponding OIIO image buffers
        OIIO::ImageBuf inputImageBuf("input", inputImageSpec, inputPixelBuffer);
        OIIO::ImageBuf finalImageBuf("final", finalImageSpec, pixelBuffer);

        // Flip the image (our frame buffers are upside down compared to what OIIO is expecting).
        // Set flipped output to 3 channel 32-bit floating point color.
        OIIO::ImageBufAlgo::flip(finalImageBuf, inputImageBuf);
    }

    // Clean up
    scene_rdl2::util::alignedFreeArray<float> (inputPixelBuffer);
    inputPixelBuffer = nullptr;
    in->close();

    // Exit if we failed to find the base mip level
    mNumMipLevels = pixelBuffers.size();
    if (mNumMipLevels == 0) {
        throw scene_rdl2::except::IoError("Cannot seek into image file: \"" + mapFilename +
                      "\" (" + in->geterror() + ")");
    }

    // Set up array of pointers, one to each mip level's pixel buffer
    mPixelBuffer = scene_rdl2::util::alignedMallocArray<float *>(mNumMipLevels);
    memcpy(mPixelBuffer, pixelBuffers.data(), mNumMipLevels * sizeof(float *));

    // We'll also need a corresponding distribution for each mip level
    mDistribution = scene_rdl2::util::alignedMallocArray<Distribution2D *>(mNumMipLevels);

    // Set flag to say whether there's a non-trivial transformation
    mIsTransformed = (rotationAngle != 0.0f         ||
                      translation   != Vec2f(zero)  ||
                      coverage      != Vec2f(one)   ||
                      repsU         != 1.0f         ||
                      repsV         != 1.0f);

    // Compute affine transformation, unless it's the identity.
    // Note that transforming a texture by a given matrix involves applying the inverse of that matrix
    // to the texture coordinates. The forward transformation is
    //    rotate(angle) * scale(factor) * translate(vector)
    // and so the inverse translation would be
    //     translate(-vector) * scale(1/factor) * rotate(-angle)
    // and this is the matrix we build here. The scale factor is computed as coverage/reps, so we take the reciprocal
    // of that. Also note that the minus sign is missing from the rotation angle because the clockwise rotation
    // requested by the artists is already mathematically negative.
    if (mIsTransformed) {
        // translation
        float tx = -translation.x;
        float ty = -translation.y;

        // scale
        float sx = repsU / coverage.x;
        float sy = repsV / coverage.y;

        // rotation
        float theta = rotationAngle * (sPi / 180.0f);
        float c = scene_rdl2::math::cos(theta);
        float s = scene_rdl2::math::sin(theta);

        // mat =  trn * scl * rot
        //
        //        (1  0  0)  (sx 0  0)  ( c  s  0)
        //     =  (0  1  0)  (0  sy 0)  (-s  c  0)
        //        (tx ty 1)  (0  0  1)  ( 0  0  1)
        //
        //        (1  0  0)  ( c*sx  s*sx  0)
        //     =  (0  1  0)  (-s*sy  c*sy  0)
        //        (tx ty 1)  (  0     0    1)
        //
        //        (     c*sx             s*sx        0)
        //     =  (    -s*sy             c*sy        0)
        //        (c*sx*tx-s*sy*ty  s*sx*tx+c*sy*ty  1)
        //
        float csx = c * sx;
        float ssx = s * sx;
        float csy = c * sy;
        float ssy = s * sy;
        mTransformation = Mat3f(csx, ssx, 0.0f, -ssy, csy, 0.0f, csx*tx-ssy*ty,  ssx*tx+csy*ty,  1.0f);
    } else {
        mTransformation = Mat3f(1.0f);
    }

    // Copy remaining data
    mBorderColor = borderColor;
    mRepsU       = repsU;
    mRepsV       = repsV;
    mMirrorU     = mirrorU;
    mMirrorV     = mirrorV;

    bool gammaOn              = (gamma != sWhite);
    bool saturationOn         = (saturation != sWhite);
    bool contrastOn           = (contrast != sWhite);
    bool gainOffsetOn         = ((gain != sWhite) || (offset != sBlack));
    bool temperatureControlOn = (temperature != Vec3f(zero));

    bool doColorCorrect = gammaOn || saturationOn || contrastOn || gainOffsetOn || temperatureControlOn;

    // Loop over mipmap levels
    for (int mipLevel = 0; mipLevel < mNumMipLevels; ++mipLevel) {

        // Dimensions of mip level
        int mipWidth  = mWidth  >> mipLevel;
        int mipHeight = mHeight >> mipLevel;

        // Create a sampling distribution from the luminance of the image
        mDistribution[mipLevel] = scene_rdl2::util::alignedMallocCtorArgs<Distribution2D>(DEFAULT_MEMORY_ALIGNMENT,
                                                                              mipWidth, mipHeight);
        const int paddedNumPixels = (mipWidth * mipHeight + simdWidth - 1) & -simdWidth;
        const int parallelForRange = paddedNumPixels / simdWidth;

        // Color Correct the input image
        // if any color correct attribute values are non-default, perform color correction...
        if (doColorCorrect) {
            tbb::parallel_for(tbb::blocked_range<int>( 0 , parallelForRange,
                                                       parallelForRange / 64),
                              [&](const tbb::blocked_range<int> t) {
                                  ispc::PBR_colorCorrectFormatArray(t.end() - t.begin(),
                                      gammaOn, gamma.r, gamma.g, gamma.b,
                                      gainOffsetOn, gain.r, gain.g, gain.b,
                                                      offset.r, offset.g, offset.b,
                                      contrastOn, contrast.r, contrast.g, contrast.b,
                                      saturationOn, saturation.r, saturation.g, saturation.b,
                                      temperatureControlOn, temperature.x, temperature.y, temperature.z,
                                      &(mPixelBuffer[mipLevel][t.begin() * simdWidth * 3]));
                              });
        }

        // Update the Distribution with Image's luminance value
        if (mIsTransformed) {
            // Compute a suitable sampling rate in each direction, based on the the upper-left
            // 2x2 portion of the transformation matrix, which we can interpret as derivatives
            const Mat3f &m = mTransformation;
            float uCount = scene_rdl2::math::ceil(2.0f * max(scene_rdl2::math::abs(m[0][0]), scene_rdl2::math::abs(m[0][1])));    // # super-samples per texel in u direction
            float vCount = scene_rdl2::math::ceil(2.0f * max(scene_rdl2::math::abs(m[1][0]), scene_rdl2::math::abs(m[1][1])));    // # super-samples per texel in v direction
            float uCountRcp = 1.0f / uCount;                                // reciprocals, for speed
            float vCountRcp = 1.0f / vCount;
            float uOffset = 0.5f * (uCount - 1.0f);                         // offsets to center super-samples around
            float vOffset = 0.5f * (vCount - 1.0f);                         //   texel center

            // Sample the texture at the determined rate, (uCount x vCount) samples per texel
            tbb::parallel_for(tbb::blocked_range<size_type>(0, mipHeight, mipHeight / sRangeDivider),
                              [&](const tbb::blocked_range<size_type> range) {
                for (size_type y = range.begin(); y < range.end(); ++y) {
                    float v = (static_cast<float>(y) + 0.5f) / mipHeight;
                    for (size_type x = 0; x < mipWidth; ++x) {
                        float u = (static_cast<float>(x) + 0.5f) / mipWidth;

                        // Accumulate samples for current texel
                        Color color(0.f);
                        for (float vSub = 0.0f; vSub < vCount; vSub += 1.0f) {
                            float vAdjusted = v + (vSub - vOffset) * vCountRcp;
                            for (float uSub = 0.0f; uSub < uCount; uSub += 1.0f) {
                                float uAdjusted = u + (uSub - uOffset) * uCountRcp;
                                color += textureLookupTransformed(uAdjusted, vAdjusted, mipLevel,
                                                                  TEXTURE_FILTER_NEAREST);
                            }
                        }

                        // Take luminance of mean color
                        color *= uCountRcp * vCountRcp;
                        float lum = luminance(color);
                        lum = max(lum, sEpsilon);
                        mDistribution[mipLevel]->setWeight(x, y, lum);
                    }
                }
            });
        } else {
            // Non-transformed case is simpler (1 sample per texel)
            tbb::parallel_for(tbb::blocked_range<size_type>(0, mipHeight, mipHeight / sRangeDivider),
                              [&](const tbb::blocked_range<size_type> range) {
                for (size_type y = range.begin(); y < range.end(); ++y) {
                    float v = (static_cast<float>(y) + 0.5f) / mipHeight;
                    for (size_type x = 0; x < mipWidth; ++x) {
                        float u = (static_cast<float>(x) + 0.5f) / mipWidth;
                        Color color = textureLookupDirect(u, v, mipLevel, TEXTURE_FILTER_NEAREST);
                        float lum = luminance(color);
                        lum = max(lum, sEpsilon);
                        mDistribution[mipLevel]->setWeight(x, y, lum);
                    }
                }
            });
        }

        // Get ready to sample
        mDistribution[mipLevel]->tabulateCdf(mapping);
    }
}


ImageDistribution::~ImageDistribution()
{
    for (int mipLevel = 0; mipLevel < mNumMipLevels; ++mipLevel) {
        scene_rdl2::util::alignedFreeDtor<float>(mPixelBuffer[mipLevel]);
        scene_rdl2::util::alignedFreeDtor<Distribution2D>(mDistribution[mipLevel]);
    }
    scene_rdl2::util::alignedFreeArray<float *>(mPixelBuffer);
    mPixelBuffer = nullptr;
    scene_rdl2::util::alignedFreeDtor<Distribution2D *>(mDistribution);
    mDistribution = nullptr;
}


//----------------------------------------------------------------------------

// Lookup pixel by a pair of indices and a mip level.
// This function is called by both textureLookupDirect() and textureLookupTransformed(),
// and assumes that the work of treating u,v values outside [0,1] has already been done.
// This ensures that converting (u,v) to a texel index will generate a good result. The
// only exceptions are the edge cases u=1 and v=1, hence the testing against mipWidth and
// mipHeight.

Color
ImageDistribution::lookup(const int xi, const int yi, const int mipLevel) const
{
    int mipWidth  = mWidth  >> mipLevel;
    int mipHeight = mHeight >> mipLevel;

    int xi_ = (xi < 0) ? 0 : (xi >= mipWidth ) ? mipWidth  - 1 : xi;
    int yi_ = (yi < 0) ? 0 : (yi >= mipHeight) ? mipHeight - 1 : yi;

    int index = (yi_ * mipWidth + xi_) * 3;

    // TODO: make mPixelBuffer be of type Color[] and then we won't need this clunky 3-float copy operaiton
    Color color;
    color.r = mPixelBuffer[mipLevel][index];
    color.g = mPixelBuffer[mipLevel][index + 1];
    color.b = mPixelBuffer[mipLevel][index + 2];

    return color;
}

Color
ImageDistribution::filterNearest(const float u, const float v) const
{
    MNRY_ASSERT(u >= 0.0f  &&  u <= 1.0f);
    MNRY_ASSERT(v >= 0.0f  &&  v <= 1.0f);

    float x = u * mWidth;
    float y = v * mHeight;
    int xi = static_cast<int>(scene_rdl2::math::floor(x));
    int yi = static_cast<int>(scene_rdl2::math::floor(y));

    return lookup(xi, yi, 0);
}

Color
ImageDistribution::filterBilinear(const float u, const float v) const
{
    MNRY_ASSERT(u >= 0.0f  &&  u <= 1.0f);
    MNRY_ASSERT(v >= 0.0f  &&  v <= 1.0f);

    int xi, yi;
    float xf, yf;
    intAndFrac(u * mWidth  - 0.5f, &xi, &xf);
    intAndFrac(v * mHeight - 0.5f, &yi, &yf);

    Color color00 = lookup(xi,   yi,   0);
    Color color10 = lookup(xi+1, yi,   0);
    Color color01 = lookup(xi,   yi+1, 0);
    Color color11 = lookup(xi+1, yi+1, 0);

    return bilerp(color00, color10, color01, color11, xf, yf);
}

Color
ImageDistribution::filterNearestMipNearest(const float u, const float v, const float mipLevel) const
{
    MNRY_ASSERT(u >= 0.0f  &&  u <= 1.0f);
    MNRY_ASSERT(v >= 0.0f  &&  v <= 1.0f);

    float m = clamp(mipLevel, 0.0f, static_cast<float>(mNumMipLevels-1));
    int mi = static_cast<int>(round(m));

    int mipWidth  = mWidth  >> mi;
    int mipHeight = mHeight >> mi;

    float x = u * mipWidth;
    float y = v * mipHeight;
    int xi = static_cast<int>(scene_rdl2::math::floor(x));
    int yi = static_cast<int>(scene_rdl2::math::floor(y));

    return lookup(xi, yi, mi);
}

Color
ImageDistribution::filterBilinearMipNearest(const float u, const float v, const float mipLevel) const
{
    MNRY_ASSERT(u >= 0.0f  &&  u <= 1.0f);
    MNRY_ASSERT(v >= 0.0f  &&  v <= 1.0f);

    float m = clamp(mipLevel, 0.0f, static_cast<float>(mNumMipLevels-1));
    int mi = static_cast<int>(round(m));

    int mipWidth  = mWidth  >> mi;
    int mipHeight = mHeight >> mi;

    int xi, yi;
    float xf, yf;
    intAndFrac(u * mipWidth  - 0.5f, &xi, &xf);
    intAndFrac(v * mipHeight - 0.5f, &yi, &yf);

    Color color00 = lookup(xi,   yi,   mi);
    Color color10 = lookup(xi+1, yi,   mi);
    Color color01 = lookup(xi,   yi+1, mi);
    Color color11 = lookup(xi+1, yi+1, mi);

    return bilerp(color00, color10, color01, color11, xf, yf);
}

Color
ImageDistribution::applyTexFilter(const float u, const float v, const float mipLevel,
                                  const TextureFilterType texFilter) const
{
    switch (texFilter) {
    case TEXTURE_FILTER_NEAREST:
        return filterNearest(u, v);
    case TEXTURE_FILTER_BILINEAR:
        return filterBilinear(u, v);
    case TEXTURE_FILTER_NEAREST_MIP_NEAREST:
        return filterNearestMipNearest(u, v, mipLevel);
    case TEXTURE_FILTER_BILINEAR_MIP_NEAREST:
        return filterBilinearMipNearest(u, v, mipLevel);
    default:
        return filterNearest(u, v);
    }
}

// Lookup from untransformed texture
Color
ImageDistribution::textureLookupDirect(const float u, const float v, const float mipLevel,
                                       const TextureFilterType texFilter) const
{
    // Apply border color outside the texture
    if (u < 0.0f || u > 1.0f || v < 0.0f || v > 1.0f) {
        return mBorderColor;

    }

    // Ensure there's a texture
    if (mPixelBuffer == nullptr) {
        return sWhite;
    }

    return applyTexFilter(u, v, mipLevel, texFilter);
}


// Lookup from transformed texture
Color
ImageDistribution::textureLookupTransformed(const float uUntransformed, const float vUntransformed,
                                            const float mipLevel, const TextureFilterType texFilter) const
{
    // Apply transformation
    Vec3f uv = Vec3f(uUntransformed, vUntransformed, 1.0f) * mTransformation;
    float u = uv.x;
    float v = uv.y;

    // Apply border color outside the texture
    if (u < 0.0f || u > mRepsU || v < 0.0f || v > mRepsV) {
        return mBorderColor;
    }

    // Ensure there's a texture
    if (mPixelBuffer == nullptr) {
        return sWhite;
    }

    // Perform wrapping and/or mirroring
    float uFloor = scene_rdl2::math::floor(u);
    float vFloor = scene_rdl2::math::floor(v);
    u -= uFloor;
    v -= vFloor;
    if (mMirrorU && (static_cast<int>(uFloor) & 1)) {
        u = 1.0f - u;
    }
    if (mMirrorV && (static_cast<int>(vFloor) & 1)) {
        v = 1.0f - v;
    }

    return applyTexFilter(u, v, mipLevel, texFilter);
}


// eval just calls the appropriate texture lookup func
Color
ImageDistribution::eval(const float u, const float v, const float mipLevel, const TextureFilterType texFilter) const
{
    if (mIsTransformed) {
        return textureLookupTransformed(u, v, mipLevel, texFilter);
    } else {
        return textureLookupDirect(u, v, mipLevel, texFilter);
    }
}


//----------------------------------------------------------------------------


float
ImageDistribution::pdfNearest(const float u, const float v) const
{
    // We look up from the finest lod, which is mip level 0
    return mDistribution[0]->pdfNearest(u, v);
}

float
ImageDistribution::pdfBilinear(const float u, const float v) const
{
    // We look up from the finest lod, which is mip level 0
    return mDistribution[0]->pdfBilinear(u, v);
}

float
ImageDistribution::pdfNearestMipNearest(const float u, const float v, const float mipLevel) const
{
    float m = clamp(mipLevel, 0.0f, static_cast<float>(mNumMipLevels-1));
    int mi = static_cast<int>(round(m));
    return mDistribution[mi]->pdfNearest(u, v);
}

float
ImageDistribution::pdfBilinearMipNearest(const float u, const float v, const float mipLevel) const
{
    float m = clamp(mipLevel, 0.0f, static_cast<float>(mNumMipLevels-1));
    int mi = static_cast<int>(round(m));
    return mDistribution[mi]->pdfBilinear(u, v);
}

float
ImageDistribution::pdf(const float u, const float v, const float mipLevel, const TextureFilterType texFilter) const
{
    MNRY_ASSERT(isValid());
    switch (texFilter) {
    case TEXTURE_FILTER_NEAREST:
        return pdfNearest(u, v);
    case TEXTURE_FILTER_BILINEAR:
        return pdfBilinear(u, v);
    case TEXTURE_FILTER_NEAREST_MIP_NEAREST:
        return pdfNearestMipNearest(u, v, mipLevel);
    case TEXTURE_FILTER_BILINEAR_MIP_NEAREST:
        return pdfBilinearMipNearest(u, v, mipLevel);
    default:
        return pdfNearest(u, v);
    }
}


//----------------------------------------------------------------------------


void
ImageDistribution::sampleNearest(const float ru, const float rv, scene_rdl2::math::Vec2f *uv, float *pdf) const
{
    // We sample the finest lod, which is mip level 0
    mDistribution[0]->sampleNearest(ru, rv, uv, pdf);
}

void
ImageDistribution::sampleBilinear(const float ru, const float rv, scene_rdl2::math::Vec2f *uv, float *pdf) const
{
    // We sample the finest lod, which is mip level 0
    mDistribution[0]->sampleBilinear(ru, rv, uv, pdf);
}

void
ImageDistribution::sampleNearestMipNearest(const float ru, const float rv, float mipLevel, scene_rdl2::math::Vec2f *uv,
                                           float *pdf) const
{
    float m = clamp(mipLevel, 0.0f, static_cast<float>(mNumMipLevels-1));
    int mi = static_cast<int>(round(m));
    mDistribution[mi]->sampleNearest(ru, rv, uv, pdf);
}

void
ImageDistribution::sampleBilinearMipNearest(const float ru, const float rv, float mipLevel, scene_rdl2::math::Vec2f *uv,
                                           float *pdf) const
{
    float m = clamp(mipLevel, 0.0f, static_cast<float>(mNumMipLevels-1));
    int mi = static_cast<int>(round(m));
    mDistribution[mi]->sampleBilinear(ru, rv, uv, pdf);
}

void
ImageDistribution::sample(const float ru, const float rv, float mipLevel, scene_rdl2::math::Vec2f *uv, float *pdf,
                          const TextureFilterType texFilter) const
{
    MNRY_ASSERT(isValid());
    switch (texFilter) {
    case TEXTURE_FILTER_NEAREST:
        sampleNearest(ru, rv, uv, pdf);
        break;
    case TEXTURE_FILTER_BILINEAR:
        sampleBilinear(ru, rv, uv, pdf);
        break;
    case TEXTURE_FILTER_NEAREST_MIP_NEAREST:
        sampleNearestMipNearest(ru, rv, mipLevel, uv, pdf);
        break;
    case TEXTURE_FILTER_BILINEAR_MIP_NEAREST:
        sampleBilinearMipNearest(ru, rv, mipLevel, uv, pdf);
        break;
    default:
        sampleNearest(ru, rv, uv, pdf);
        break;
    }
}


//----------------------------------------------------------------------------

} // namespace pbr
} // namespace moonray

