// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file BsdfLobe.cc
/// $Id$
///

#include "Bsdf.h"
#include <moonray/rendering/shading/bssrdf/Bssrdf.h>
#include "BsdfSlice.h"

namespace moonray {
namespace shading {


using namespace scene_rdl2::math;


//----------------------------------------------------------------------------

BsdfLobe::BsdfLobe(Type type, DifferentialFlags diffFlags, bool isSpherical, int32_t propertyFlags) :
    mType(type),
    mDifferentialFlags(diffFlags),
    mIsSpherical(isSpherical),
    mIsHair(false),
    mScale(1.0f, 1.0f, 1.0f),
    mFresnel(nullptr),
    mLabel(0),
    mPropertyFlags(propertyFlags | PROPERTY_COLOR)
{
}




BsdfLobe::~BsdfLobe()
{
    // Nothing to do here, all lobes and fresnel were allocated with a
    // memory arena
}


void
BsdfLobe::setFresnel(Fresnel *fresnel)
{
    if (fresnel == mFresnel) {
        return;
    }
    mFresnel = fresnel;
}

bool
BsdfLobe::getProperty(Property property, float *dest) const
{
    bool handled = true;

    switch (property)
    {
    case PROPERTY_COLOR:
        {
            Color result = computeScaleAndFresnel(1.f);
            *dest       = result.r;
            *(dest + 1) = result.g;
            *(dest + 2) =result.b;
        }
        break;
    default:
        handled = false;
    }

    return handled;
}

//----------------------------------------------------------------------------

Bsdf::~Bsdf()
{
}


int
Bsdf::getLobeCount(BsdfLobe::Type flags) const
{
    int count = 0;

    int size = mLobeArray.size();
    for (int l=0; l < size; l++) {
        count += (mLobeArray[l]->matchesFlags(flags)  ?  1  : 0);
    }

    return count;
}


Color
Bsdf::eval(const BsdfSlice &slice, const Vec3f &wi, float &pdf) const
{
    BsdfLobe::Type flags = slice.getSurfaceFlags(*this, wi);

    // Add up all matching lobes' eval()
    Color f(zero);
    pdf = 0.0f;
    int size = mLobeArray.size();
    for (int l=0; l < size; l++) {
        BsdfLobe* const lobe = mLobeArray[l];

        // We need to account for lobe pdf, even if the surface flag doesn't match
        float tmpPdf = 0.0f;
        Color color = lobe->eval(slice, wi, &tmpPdf);
        if (lobe->matchesFlags(flags)) {
            f += color;
        }
        pdf += tmpPdf;
    }

    // Assumes all lobes are equi-probable
    if (size > 1) {
        pdf /= size;
    }

    return f;
}


//----------------------------------------------------------------------------

Color
Bsdf::albedo(const BsdfSlice &slice) const
{
    Color result(zero);

    // Add up all matching lobes' albedo()
    int size = mLobeArray.size();
    for (int l=0; l < size; l++) {
        if (mLobeArray[l]->matchesFlags(slice.getFlags())) {
            result += mLobeArray[l]->albedo(slice);
        }
    }

    return result;
}

void
Bsdf::show(const std::string &sceneClass, const std::string &name, std::ostream& os) const
{
    const size_t numLobes = mLobeArray.size();

    os << "\n";
    os << "==========================================================\n";
    os << "BSDF @ " << this << " for " << sceneClass << ": '" << name << "'\n";
    os << "==========================================================\n";
    os << "\n";

    for (size_t i = 0; i < numLobes; ++i) {
        const BsdfLobe* const lobe = mLobeArray[i];
        lobe->show(os, "");
        os << "\n";
    }

    if (mBssrdf) {
        mBssrdf->show(os, "");
        os << "\n";
    }

    os << "[emission] = "
        << mSelfEmission.r << " "
        << mSelfEmission.g << " "
        << mSelfEmission.b << "\n";

    os << "\n";
    os << "==========================================================\n";
}

void
Bsdf::setPostScatterExtraAovs(int numExtraAovs, const int *labelIds, const scene_rdl2::math::Color *colors)
{
    mPostScatterExtraAovs = { numExtraAovs, labelIds, colors };
}

} // namespace shading
} // namespace moonray

