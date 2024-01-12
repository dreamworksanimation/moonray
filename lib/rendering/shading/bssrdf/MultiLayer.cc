// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
///

#include "MultiLayer.h"
#include <moonray/rendering/shading/bsdf/Fresnel.h>

#include <cmath>
#include <iomanip>

//#define BSSRDF_SAMPLE_UNIFORM 1
#define BSSRDF_SAMPLE_UNIFORM 0

namespace moonray {
namespace shading {

using namespace scene_rdl2::math;

MultiLayerBssrdf::MultiLayerBssrdf(const Vec3f& N)
    : Bssrdf(N, PROPERTY_COLOR|PROPERTY_RADIUS, nullptr, nullptr), mBssrdfs(), mWeights(),
      mCdfs(), mLobeCount(0)
{
    memset(mBssrdfs, 0, sizeof(mBssrdfs));
    memset(mWeights, 0, sizeof(mWeights));
    memset(mCdfs, 0, sizeof(mCdfs));
    mLobeCount = 0;
}

MultiLayerBssrdf::~MultiLayerBssrdf()
{
}

//----------------------------------------------------------------------------

Color
MultiLayerBssrdf::eval(float r, bool /* global */) const
{
    Color rd(0);
    for (unsigned int t = 0; t < mLobeCount; ++t) {
        Color trd = mBssrdfs[t]->eval(r) * mWeights[t];
        rd += trd;
    }
    return rd;
}

void
MultiLayerBssrdf::addBssrdf(Bssrdf* b, float w)
{
    //max layers allowed is 8
    if (w > 0 && b != NULL && (mLobeCount < BSSRDF_MAX_LOBES - 1)) {
        mBssrdfs[mLobeCount] = b;
        mWeights[mLobeCount] = w;
        mLobeCount++;
    }
}

void
MultiLayerBssrdf::finalize()
{
    float sum = 0;
    mMaxRadius = 0;
    for (unsigned int i = 0; i < mLobeCount; i++) {
        sum += mWeights[i];
        if (mBssrdfs[i]->getMaxRadius() > mMaxRadius) mMaxRadius = mBssrdfs[i]->getMaxRadius();
    }
    if (sum > 0) {
        float norm = 1.0f/sum;
        float cdf = 0;
        //mCdfs = new float[mWeights.size()];
        for (unsigned int i = 0; i < mLobeCount; i++) {
            mWeights[i] *= norm;
            cdf += mWeights[i];
            mCdfs[i] = cdf;
        } 
    }
}

unsigned int
MultiLayerBssrdf::selectIndex(float r1) const
{
    switch (mLobeCount) {
    case 1:
        return 0;
    case 2:
        return r1 < mCdfs[0] ? 0 : 1;
    case 3:
        return (r1 < mCdfs[0]  ?  0  :
               (r1 < mCdfs[1]  ?  1  :  2));
    }     
    const float *ptr = std::upper_bound(mCdfs, mCdfs + mLobeCount, r1);
    unsigned int layerIndex = ptr - mCdfs;
    return  max(layerIndex, mLobeCount - 1);
}


#if !BSSRDF_SAMPLE_UNIFORM
float
MultiLayerBssrdf::sampleLocal(float r1, float r2, Vec3f &dPi, float &r) const
{
    if (mLobeCount == 0) return 1;

    unsigned int layerIndex = selectIndex(r1);
    float curCdf = mWeights[layerIndex];
    if (layerIndex == 0) {
        r1 = r1/curCdf;
    } else {
        r1 = (r1-mCdfs[layerIndex-1])/curCdf;
    }
    float childPdf =  mBssrdfs[layerIndex]->sampleLocal(r1, r2, dPi, r);
    float totalPdf =  childPdf * curCdf;
    for (unsigned int i = 0; i < mLobeCount; i++) {
        if (i != layerIndex) {
            totalPdf += mWeights[i] * mBssrdfs[i]->pdfLocal(r);
        }
    }
    return totalPdf;
}

float
MultiLayerBssrdf::pdfLocal(float r) const
{
    // We compute the estimator with MIS, under Veach's one-sample model.
    // See paper: "Optimally combining sampling techniques for MC rendering"
    // Here we use estimator (16) and therefore: pdf = ci * pi / wi
    // With the balance heuristic it amounts to the weighted average of the
    // pdf for each channel: pdf = sum(ci * pi)
    float pdf = 0.0f;
    for (unsigned int i = 0; i < mLobeCount; ++i) {
        float ci = mWeights[i];
        pdf += ci * mBssrdfs[i]->pdfLocal(r);
    }
    return pdf;
}

/// For Diffusion-Based BSSRDFs, we use the analytically computed
/// diffuse reflectance to compute the area-compensation factor
Color
MultiLayerBssrdf::diffuseReflectance() const
{
    Color dR = scene_rdl2::math::sBlack;
    for (unsigned int i = 0; i < mLobeCount; ++i) {
        float ci = mWeights[i];
        dR += ci * mBssrdfs[i]->diffuseReflectance();
    }
    return dR;
}


#else

// Uniform sampling
float
MultiLayerBssrdf::sampleLocal(float r1, float r2, Vec3f &dPi, float &r) const
{
    Vec2f p;
    p[0] = (r1 - 0.5f) * 2.0f * mMaxRadius;
    p[1] = (r2 - 0.5f) * 2.0f * mMaxRadius;
    r = p.length();

    // Compute position
    dPi = p[0] * mFrame.getX() + p[1] * mFrame.getY();

    float pdf = 1.0f / (4.0f * mMaxRadius * mMaxRadius);
    return pdf;
}


float
MultiLayerBssrdf::pdfLocal(float r) const
{
    float pdf = 1.0f / (4.0f * mMaxRadius * mMaxRadius);
    return pdf;
}
#endif

bool
MultiLayerBssrdf::getProperty(Property property, float *dest) const
{
    Color values(0.0f);
    Color totalValues(0.0f);
    bool handled = true;

    switch (property)
    {
    case PROPERTY_COLOR:
    case PROPERTY_RADIUS:
        for (unsigned int t = 0; t < mLobeCount; ++t) {
            mBssrdfs[t]->getProperty(property, (float*) &values); 
            totalValues += mWeights[t] * values; 
        }
        dest[0] = totalValues[0];
        dest[1] = totalValues[1];
        dest[2] = totalValues[2];
        break;
    default:
        handled = Bssrdf::getProperty(property, dest);
    }

    return handled;
}

void
MultiLayerBssrdf::show(std::ostream& os, const std::string& indent) const
{
    const Color& scale = getScale();
    os << indent << "[MultiLayerBssrdf]\n";
    os << indent << "    " << "scale: "
        << scale.r << " " << scale.g << " " << scale.b << "\n";
    if (mFresnel) {
        mFresnel->show(os, indent + "    ");
    }
}



//----------------------------------------------------------------------------

} // namespace shading
} // namespace moonray

