// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file BsdfTableAniso.cc
/// $Id$
///


#include "BsdfTableAniso.h"
#include <moonray/rendering/shading/Util.h>

#include <scene_rdl2/common/except/exceptions.h>
#include <scene_rdl2/common/math/ReferenceFrame.h>

#include <cstring>
#include <stdio.h>

using namespace scene_rdl2::math;

namespace moonray {
namespace shading {

//----------------------------------------------------------------------------

// rotate vector along one axis
static Vec3f rotate(const Vec3f &vector, const Vec3f &axis, float angle)
{
    Mat4f mat = Mat4f::rotate(Vec4f(axis.x, axis.y, axis.z, 0.0f), angle);
    Vec3f result = transformVector(mat, vector);

    return result;
}


//----------------------------------------------------------------------------

AnisotropicBsdfTable::AnisotropicBsdfTable(bool reciprocal) :
    mSizeThetaH(90),
    mSizePhiH(360),
    mSizeThetaD(90),
    mSizePhiD(reciprocal  ?  180  :  360),
    mReciprocal(reciprocal),
    mData(nullptr)
{
    init();
}


AnisotropicBsdfTable::AnisotropicBsdfTable(int sizeThetaH, int sizePhiH,
        int sizeThetaD, int sizePhiD, bool reciprocal) :
    mSizeThetaH(sizeThetaH),
    mSizePhiH(sizePhiH),
    mSizeThetaD(sizeThetaD),
    mSizePhiD(sizePhiD),
    mReciprocal(reciprocal),
    mData(nullptr)
{
    init();
}


void
AnisotropicBsdfTable::init()
{
    int floatCount = getFloatCount();
    mData = new float[floatCount];
    for (int i = 0; i < floatCount; i++) {
        mData[i] = 0.0f;
    }
}


AnisotropicBsdfTable::~AnisotropicBsdfTable()
{
    delete [] mData;
}


//----------------------------------------------------------------------------

// cppcheck-suppress uninitMemberVar // note these is an embree file so we are ignoring these
AnisotropicBsdfTable::AnisotropicBsdfTable(const std::string &filename) :
    mData(nullptr)
{
    // Read file
    FILE *file = fopen(filename.c_str(), "rb");
    if (!file) {
        throw scene_rdl2::except::IoError("Cannot open file \"" + filename + "\" ("
                + strerror(errno) + ")");
    }

    // TODO: read header

    int reciprocal = 0;
    if (fread(&mSizeThetaH, sizeof(int), 1, file) != 1  ||
        fread(&mSizePhiH, sizeof(int), 1, file) != 1  ||
        fread(&mSizeThetaD, sizeof(int), 1, file) != 1  ||
        fread(&mSizePhiD, sizeof(int), 1, file) != 1  ||
        fread(&reciprocal, sizeof(int), 1, file) != 1) {
        fclose(file);
        throw scene_rdl2::except::IoError("Cannot read table size in file \"" + filename + "\" ("
                + strerror(errno) + ")");
    }
    mReciprocal = reciprocal;

    init();

    int floatCount = getFloatCount();
    if (fread(mData, sizeof(float), floatCount, file) != floatCount) {
        fclose(file);
        throw scene_rdl2::except::IoError("Cannot read table in file \"" + filename + "\" ("+ strerror(errno) + ")");
    }
    fclose(file);
}


void
AnisotropicBsdfTable::saveAs(const std::string &filename) const
{
    // Open file
    FILE *file = fopen(filename.c_str(), "wb");
    if (!file) {
        throw scene_rdl2::except::IoError("Cannot open file \"" + filename + "\" ("
                + strerror(errno) + ")");
    }

    // TODO: write header

    int reciprocal = mReciprocal;
    if (fwrite(&mSizeThetaH, sizeof(int), 1, file) != 1  ||
        fwrite(&mSizePhiH, sizeof(int), 1, file) != 1  ||
        fwrite(&mSizeThetaD, sizeof(int), 1, file) != 1  ||
        fwrite(&mSizePhiD, sizeof(int), 1, file) != 1  ||
        fwrite(&reciprocal, sizeof(int), 1, file) != 1) {
        fclose(file);
        throw scene_rdl2::except::IoError("Cannot write table size in file \"" + filename + "\" ("
                + strerror(errno) + ")");
    }

    int floatCount = getFloatCount();
    if (fwrite(mData, sizeof(float), floatCount, file) != floatCount) {
        fclose(file);
        throw scene_rdl2::except::IoError("Cannot write table in file \"" + filename + "\" ("+ strerror(errno) + ")");
    }
    fclose(file);
}


//----------------------------------------------------------------------------

static const float sFloatIndexEpsilon = 1e-3f;

float
AnisotropicBsdfTable::getIndexThetaH(float thetaH) const
{
    // Only do first half od the test to account for directions slightly
    // below the surface due to interpolated normals
    MNRY_ASSERT(thetaH >= 0.0f  /* &&  thetaH <= sHalfPi */);

    return sqrt(thetaH / sHalfPi) * (mSizeThetaH - sFloatIndexEpsilon);
}


float
AnisotropicBsdfTable::getThetaH(int index) const
{
    MNRY_ASSERT(index >= 0  &&  index < mSizeThetaH);
    float tmp = float(index) / mSizeThetaH;
    return  tmp * tmp * sHalfPi;
}


float
AnisotropicBsdfTable::getIndexPhiH(float phiH) const
{
    MNRY_ASSERT(phiH >= -sPi  &&  phiH <= sPi);

    if (phiH < 0.0f) {
        phiH += sTwoPi;
    }

    MNRY_ASSERT(phiH >= 0.0f);
    return phiH / sTwoPi * (mSizePhiH - sFloatIndexEpsilon);
}


float
AnisotropicBsdfTable::getPhiH(int index) const
{
    MNRY_ASSERT(index >= 0  &&  index < mSizePhiH);
    return float(index) / mSizePhiH * sTwoPi;
}


float
AnisotropicBsdfTable::getIndexThetaD(float thetaD) const
{
    MNRY_ASSERT(thetaD >= 0.0f  &&  thetaD <= sHalfPi);

    return thetaD / sHalfPi * (mSizeThetaD - sFloatIndexEpsilon);
}


float
AnisotropicBsdfTable::getThetaD(int index) const
{
    MNRY_ASSERT(index >= 0  &&  index < mSizeThetaD);
    return float(index) / mSizeThetaD * sHalfPi;
}


float
AnisotropicBsdfTable::getIndexPhiD(float phiD) const
{
    MNRY_ASSERT(phiD >= -sPi  &&  phiD <= sPi);

    // Because of reciprocity, the BRDF is unchanged under phiD -> phiD + M_PI
    const float phiOffset = mReciprocal ? sPi : sTwoPi;
    if (phiD < 0.0f) {
        phiD += phiOffset;
    }

    MNRY_ASSERT(phiD >= 0.0f);
    return phiD / phiOffset * (mSizePhiD - sFloatIndexEpsilon);
}


float
AnisotropicBsdfTable::getPhiD(int index) const
{
    MNRY_ASSERT(index >= 0  &&  index < mSizePhiD);
    const float phiOffset = mReciprocal ? sPi : sTwoPi;
    return float(index) / mSizePhiD * phiOffset;
}


//----------------------------------------------------------------------------

// Convert standard coordinates to half vector/difference vector coordinates
// Returns true if ok and false if the conversion failed
// TODO: paper reference
bool
AnisotropicBsdfTable::localWoWi2IndexHD(const scene_rdl2::math::Vec3f &localWo, const scene_rdl2::math::Vec3f &localWi,
        float &indexThetaH, float &indexPhiH, float &indexThetaD, float &indexPhiD) const
{
    Vec3f H;
    if (!computeNormalizedHalfVector(localWo, localWi, H)) {
        return false;
    }

    // Compute thetaH, phiH
    if (H.z < sEpsilon) {
        return false;
    }
    float thetaH = acos(clamp(H.z, -1.0f, 1.0f));
    float phiH = atan2(H.y, H.x);

    // Compute difference vector
    static const ReferenceFrame frame;
    Vec3f D = rotate(localWi, frame.getN(), -phiH);
    D = rotate(D, frame.getY(), -thetaH);

    // Compute thetaD, phiD
    float thetaD = acos(clamp(D.z, -1.0f, 1.0f));
    float phiD = atan2(D.y, D.x);

    // Convert to indices
    indexThetaH = getIndexThetaH(thetaH);
    indexPhiH = getIndexPhiH(phiH);
    indexThetaD = getIndexThetaD(thetaD);
    indexPhiD = getIndexPhiD(phiD);

    /*
    std::cout << "th/ph/td/pd = " << thetaH << ", " << phiH << ", " << thetaD << ", " << phiD;
    std::cout << ", ith/iph/itd/ipd = " << indexThetaH << ", " << indexPhiH << ", " << indexThetaD << ", " << indexPhiD;
    std::cout << std::endl;
     */

    return true;
}


bool
AnisotropicBsdfTable::localWoWi2IndexHD(const scene_rdl2::math::Vec3f &localWo, const scene_rdl2::math::Vec3f &localWi,
        int &indexThetaH, int &indexPhiH, int &indexThetaD, int &indexPhiD) const
{
    float findexThetaH, findexPhiH, findexThetaD, findexPhiD;
    bool result = localWoWi2IndexHD(localWo, localWi,
            findexThetaH, findexPhiH, findexThetaD, findexPhiD);

    findexThetaH = int(findexThetaH);
    findexPhiH = int(findexPhiH);
    findexThetaD = int(findexThetaD);
    findexPhiD = int(findexPhiD);

    return result;
}


void
AnisotropicBsdfTable::indexHD2localWoWi(int indexThetaH, int indexPhiH,
        int indexThetaD, int indexPhiD, scene_rdl2::math::Vec3f &localWo,
        scene_rdl2::math::Vec3f &localWi) const
{
    float thetaH = getThetaH(indexThetaH);
    float phiH = getPhiH(indexPhiH);
    float thetaD = getThetaD(indexThetaD);
    float phiD = getPhiD(indexPhiD);

    // Compute D
    float s, c;
    sincos(thetaD, &s, &c);
    Vec3f D = computeLocalSphericalDirection(c, s, phiD);

    // Compute localWi
    static const ReferenceFrame frame;
    localWi = rotate(D, frame.getY(), thetaH);
    localWi = rotate(localWi, frame.getN(), phiH);

    // Compute H
    sincos(thetaH, &s, &c);
    Vec3f H = computeLocalSphericalDirection(c, s, phiH);

    // Compute localWo
    computeReflectionDirection(H, localWi, localWo);
}


//----------------------------------------------------------------------------

Color
AnisotropicBsdfTable::getBsdf(const scene_rdl2::math::Vec3f &localWo, const scene_rdl2::math::Vec3f &localWi,
        Interpolation interp) const
{
    // Convert (wi, wo) to thetaH, thetaD, phiD floating-point indices
    float findexThetaH, findexPhiH, findexThetaD, findexPhiD;
    if (!localWoWi2IndexHD(localWo, localWi,
            findexThetaH, findexPhiH, findexThetaD, findexPhiD)) {
        return sBlack;
    }
    int indexThetaH = int(findexThetaH);
    int indexPhiH = int(findexPhiH);
    int indexThetaD = int(findexThetaD);
    int indexPhiD = int(findexPhiD);

    if (interp == CONSTANT) {

        // Use piece-wise constant lookup
        return getColor(indexThetaH, indexPhiH, indexThetaD, indexPhiD);

    } else if (interp == LINEAR_PHIH) {

        // Only use linear interpolation over phiH
        Color c;
        float t;

        int indexPhiHNext = indexPhiH + 1;
        indexPhiHNext = indexPhiHNext == mSizePhiH  ?  0  :  indexPhiHNext;

        t = findexPhiH - indexPhiH;
        c = lerp(getColor(indexThetaH, indexPhiH,     indexThetaD, indexPhiD),
                 getColor(indexThetaH, indexPhiHNext, indexThetaD, indexPhiD), t);

        return c;

    } else if (interp == QUADRI_LINEAR) {

        // Use quadri-linear interpolation
        Color c[8];
        float t;

        int indexThetaHNext = min(indexThetaH + 1, mSizeThetaH - 1);
        int indexPhiHNext = indexPhiH + 1;
        indexPhiHNext = indexPhiHNext == mSizePhiH  ?  0  :  indexPhiHNext;
        int indexThetaDNext = min(indexThetaD + 1, mSizeThetaD - 1);
        int indexPhiDNext = indexPhiD + 1;
        indexPhiDNext = indexPhiDNext == mSizePhiD  ?  0  :  indexPhiDNext;

        // Interpolate over thetaH
        t = findexThetaH - indexThetaH;
        c[0] = lerp(getColor(indexThetaH,     indexPhiH,     indexThetaD,     indexPhiD),
                    getColor(indexThetaHNext, indexPhiH,     indexThetaD,     indexPhiD), t);
        c[1] = lerp(getColor(indexThetaH,     indexPhiH,     indexThetaDNext, indexPhiD),
                    getColor(indexThetaHNext, indexPhiH,     indexThetaDNext, indexPhiD), t);
        c[2] = lerp(getColor(indexThetaH,     indexPhiH,     indexThetaD,     indexPhiDNext),
                    getColor(indexThetaHNext, indexPhiH,     indexThetaD,     indexPhiDNext), t);
        c[3] = lerp(getColor(indexThetaH,     indexPhiH,     indexThetaDNext, indexPhiDNext),
                    getColor(indexThetaHNext, indexPhiH,     indexThetaDNext, indexPhiDNext), t);
        c[4] = lerp(getColor(indexThetaH,     indexPhiHNext, indexThetaD,     indexPhiD),
                    getColor(indexThetaHNext, indexPhiHNext, indexThetaD,     indexPhiD), t);
        c[5] = lerp(getColor(indexThetaH,     indexPhiHNext, indexThetaDNext, indexPhiD),
                    getColor(indexThetaHNext, indexPhiHNext, indexThetaDNext, indexPhiD), t);
        c[6] = lerp(getColor(indexThetaH,     indexPhiHNext, indexThetaD,     indexPhiDNext),
                    getColor(indexThetaHNext, indexPhiHNext, indexThetaD,     indexPhiDNext), t);
        c[7] = lerp(getColor(indexThetaH,     indexPhiHNext, indexThetaDNext, indexPhiDNext),
                    getColor(indexThetaHNext, indexPhiHNext, indexThetaDNext, indexPhiDNext), t);

        // Use linear interpolation over phiH
        t = findexPhiH - indexPhiH;
        c[0] = lerp(c[0], c[4], t);
        c[1] = lerp(c[1], c[5], t);
        c[2] = lerp(c[2], c[6], t);
        c[3] = lerp(c[3], c[7], t);

        // Interpolate over thetaD
        t = findexThetaD - indexThetaD;
        c[0] = lerp(c[0], c[1], t);
        c[2] = lerp(c[2], c[3], t);

        // Interpolate over phiD
        t = findexPhiD - indexPhiD;
        c[0] = lerp(c[0], c[2], t);

        return c[0];

    } else {

        MNRY_ASSERT(0  &&  "Bad interpolation type");
        return sBlack;
    }
}


void
AnisotropicBsdfTable::setBsdf(int indexThetaH, int indexPhiH,
        int indexThetaD, int indexPhiD, const scene_rdl2::math::Color &color) const
{
    // Compute index into table
    int index = getIndex(indexThetaH, indexPhiH, indexThetaD, indexPhiD);
    mData[index] = color.r;
    mData[index + 1] = color.g;
    mData[index + 2] = color.b;
}


//----------------------------------------------------------------------------

} // namespace shading
} // namespace moonray

