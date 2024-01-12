// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file BsdfTable.cc
/// $Id$
///


#include "BsdfTable.h"
#include <moonray/rendering/shading/Util.h>

#include <scene_rdl2/common/except/exceptions.h>
#include <scene_rdl2/common/math/ReferenceFrame.h>

#include <cstring>
#include <stdio.h>


namespace moonray {
namespace shading {


using namespace scene_rdl2::math;

static const float sRedScale = 1.0 / 1500.0;
static const float sGreenScale = 1.15 / 1500.0;
static const float sBlueScale = 1.66 / 1500.0;


//----------------------------------------------------------------------------

// rotate vector along one axis
static Vec3f rotate(const Vec3f &vector, const Vec3f &axis, float angle)
{
    Mat4f mat = Mat4f::rotate(Vec4f(axis.x, axis.y, axis.z, 0.0f), angle);
    Vec3f result = transformVector(mat, vector);

    return result;
}


//----------------------------------------------------------------------------

IsotropicBsdfTable::IsotropicBsdfTable() :
    mSizeThetaH(90),
    mSizeThetaD(90),
    mSizePhiD(180),
    mData(nullptr)
{
    init();
}


IsotropicBsdfTable::IsotropicBsdfTable(int sizeThetaH, int sizeThetaD, int sizePhiD) :
    mSizeThetaH(sizeThetaH),
    mSizeThetaD(sizeThetaD),
    mSizePhiD(sizePhiD),
    mData(nullptr)
{
    init();
}


void
IsotropicBsdfTable::init()
{
    int floatCount = 3 * getColorCount();
    mData = new float[floatCount];
    for (int i = 0; i < floatCount; i++) {
        mData[i] = 0.0f;
    }
}


IsotropicBsdfTable::~IsotropicBsdfTable()
{
    delete [] mData;
}


//----------------------------------------------------------------------------
// cppcheck-suppress uninitMemberVar // note these is an embree file so we are ignoring these
IsotropicBsdfTable::IsotropicBsdfTable(const std::string &filename) :
    mData(nullptr)
{
    // Read file
    FILE *file = fopen(filename.c_str(), "rb");
    if (!file) {
        throw scene_rdl2::except::IoError("Cannot open file \"" + filename + "\" ("
                + strerror(errno) + ")");
    }

    if (fread(&mSizeThetaH, sizeof(int), 1, file) != 1  ||
        fread(&mSizeThetaD, sizeof(int), 1, file) != 1  ||
        fread(&mSizePhiD, sizeof(int), 1, file) != 1) {
        fclose(file);
        throw scene_rdl2::except::IoError("Cannot read table size in file \"" + filename + "\" ("
                + strerror(errno) + ")");
    }

    // Sizes are bin sizes over thetaH, thetaD, phiD/2
    int n = getColorCount();
    double *data = new double[3 * n];
    if (fread(data, sizeof(double), 3 * n, file) != 3 * n) {
        delete [] data;
        fclose(file);
        throw scene_rdl2::except::IoError("Cannot read table in file \"" + filename + "\" ("+ strerror(errno) + ")");
    }
    fclose(file);

    init();

    // Copy to floats, re-interleaving data and applying scale
    for (int i=0; i < n; i++) {
        int j = 3 * i;
        mData[j] = data[i] * sRedScale;
        mData[j + 1] = data[i + n] * sGreenScale;
        mData[j + 2] = data[i + n * 2] * sBlueScale;

        //Color c(mData[j], mData[j + 1], mData[j + 2]);
        //std::cout << c << std::endl;
    }

    delete [] data;
}


void
IsotropicBsdfTable::saveAs(const std::string &filename) const
{
    // Open file
    FILE *file = fopen(filename.c_str(), "wb");
    if (!file) {
        throw scene_rdl2::except::IoError("Cannot open file \"" + filename + "\" ("
                + strerror(errno) + ")");
    }

    if (fwrite(&mSizeThetaH, sizeof(int), 1, file) != 1  ||
        fwrite(&mSizeThetaD, sizeof(int), 1, file) != 1  ||
        fwrite(&mSizePhiD, sizeof(int), 1, file) != 1) {
        fclose(file);
        throw scene_rdl2::except::IoError("Cannot write table size in file \"" + filename + "\" ("
                + strerror(errno) + ")");
    }

    int n = getColorCount();
    double *data = new double[3 * n];

    // Copy to doubles, de-interleaving data and inverting scale
    for (int i=0; i < n; i++) {
        int j = 3 * i;
        data[i] = mData[j] / sRedScale;
        data[i + n] = mData[j + 1] / sGreenScale;
        data[i + n * 2] = mData[j + 2] / sBlueScale;
    }

    if (fwrite(data, sizeof(double), 3 * n, file) != 3 * n) {
        fclose(file);
        delete [] data;
        throw scene_rdl2::except::IoError("Cannot write table in file \"" + filename + "\" ("+ strerror(errno) + ")");
    }
    fclose(file);

    delete [] data;
}


//----------------------------------------------------------------------------

static const float sFloatIndexEpsilon = 1e-3f;

float
IsotropicBsdfTable::getIndexThetaH(float thetaH) const
{
    // Only do first half od the test to account for directions slightly
    // below the surface due to interpolated normals
    MNRY_ASSERT(thetaH >= 0.0f  /* &&  thetaH <= sHalfPi */);

    return sqrt(thetaH / sHalfPi) * (mSizeThetaH - sFloatIndexEpsilon);
}


float
IsotropicBsdfTable::getThetaH(int index) const
{
    MNRY_ASSERT(index >= 0  &&  index < mSizeThetaH);
    float tmp = float(index) / mSizeThetaH;
    return  tmp * tmp * sHalfPi;
}


float
IsotropicBsdfTable::getIndexThetaD(float thetaD) const
{
    MNRY_ASSERT(thetaD >= 0.0f  &&  thetaD <= sHalfPi);

    return thetaD / sHalfPi * (mSizeThetaD - sFloatIndexEpsilon);
}


float
IsotropicBsdfTable::getThetaD(int index) const
{
    MNRY_ASSERT(index >= 0  &&  index < mSizeThetaD);
    return float(index) / mSizeThetaD * sHalfPi;
}


float
IsotropicBsdfTable::getIndexPhiD(float phiD) const
{
    MNRY_ASSERT(phiD >= -sPi  &&  phiD <= sPi);

    // Because of reciprocity, the BRDF is unchanged under phiD -> phiD + sPi
    if (phiD < 0.0f) {
        phiD += sPi;
    }

    MNRY_ASSERT(phiD >= 0.0f);
    return phiD / sPi * (mSizePhiD - sFloatIndexEpsilon);
}


float
IsotropicBsdfTable::getPhiD(int index) const
{
    MNRY_ASSERT(index >= 0  &&  index < mSizePhiD);
    return float(index) / mSizePhiD * sPi;
}


//----------------------------------------------------------------------------

// Convert standard coordinates to half vector/difference vector coordinates
// Returns true if ok and false if the conversion failed
// TODO: paper reference
bool
IsotropicBsdfTable::localWoWi2IndexHD(const Vec3f &localWo, const Vec3f &localWi,
        float &indexThetaH, float &indexThetaD, float &indexPhiD) const
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
    indexThetaD = getIndexThetaD(thetaD);
    indexPhiD = getIndexPhiD(phiD);

    //std::cout << ", th/td/pd = " << thetaH << ", " << thetaD << ", " << phiD;
    //std::cout << ", ith/itd/ipd = " << indexThetaH << ", " << indexThetaD << ", " << indexPhiD;

    return true;
}


bool
IsotropicBsdfTable::localWoWi2IndexHD(const Vec3f &localWo, const Vec3f &localWi,
        int &indexThetaH, int &indexThetaD, int &indexPhiD) const
{
    float findexThetaH, findexThetaD, findexPhiD;
    bool result = localWoWi2IndexHD(localWo, localWi,
            findexThetaH, findexThetaD, findexPhiD);

    findexThetaH = int(findexThetaH);
    findexThetaD = int(findexThetaD);
    findexPhiD = int(findexPhiD);

    return result;
}


void
IsotropicBsdfTable::indexHD2localWoWi(int indexThetaH, int indexThetaD, int indexPhiD,
        Vec3f &localWo, Vec3f &localWi) const
{
    // TODO: Compute phiH such that phiWo = 0 ?
    float thetaH = getThetaH(indexThetaH);
    float phiH = 0.0f;
    float thetaD = getThetaD(indexThetaD);
    float phiD = getPhiD(indexPhiD);

    // Compute D
    float s, c;
    sincos(thetaD, &s, &c);
    Vec3f D = computeLocalSphericalDirection(c, s, phiD);

    // Compute localWi
    static const ReferenceFrame frame;
    localWi = rotate(D, frame.getY(), thetaH);
    // Since phiH is 0 we can ignore:
    // localWi = rotate(localWi, frame.getN(), phiH);

    // Compute H
    sincos(thetaH, &s, &c);
    Vec3f H = computeLocalSphericalDirection(c, s, phiH);

    // Compute localWo
    computeReflectionDirection(H, localWi, localWo);
}


//----------------------------------------------------------------------------

Color
IsotropicBsdfTable::getBsdf(const Vec3f &localWo, const Vec3f &localWi, bool useLerp) const
{
    // Convert (wi, wo) to thetaH, thetaD, phiD floating-point indices
    float findexThetaH, findexThetaD, findexPhiD;
    if (!localWoWi2IndexHD(localWo, localWi, findexThetaH, findexThetaD, findexPhiD)) {
        return sBlack;
    }
    int indexThetaH = int(findexThetaH);
    int indexThetaD = int(findexThetaD);
    int indexPhiD = int(findexPhiD);

    if (useLerp) {

        // Use tri-linear interpolation
        Color c[4];
        float t;

        int indexThetaHNext = min(indexThetaH + 1, mSizeThetaH - 1);
        int indexThetaDNext = min(indexThetaD + 1, mSizeThetaD - 1);
        int indexPhiDNext = indexPhiD + 1;
        indexPhiDNext = indexPhiDNext == mSizePhiD  ?  0  :  indexPhiDNext;

        // Interpolate over thetaH
        t = findexThetaH - indexThetaH;
        c[0] = lerp(getColor(indexThetaH,     indexThetaD,     indexPhiD),
                    getColor(indexThetaHNext, indexThetaD,     indexPhiD), t);
        c[1] = lerp(getColor(indexThetaH,     indexThetaDNext, indexPhiD),
                    getColor(indexThetaHNext, indexThetaDNext, indexPhiD), t);
        c[2] = lerp(getColor(indexThetaH,     indexThetaD,     indexPhiDNext),
                    getColor(indexThetaHNext, indexThetaD,     indexPhiDNext), t);
        c[3] = lerp(getColor(indexThetaH,     indexThetaDNext, indexPhiDNext),
                    getColor(indexThetaHNext, indexThetaDNext, indexPhiDNext), t);

        // Interpolate over thetaD
        t = findexThetaD - indexThetaD;
        c[0] = lerp(c[0], c[1], t);
        c[2] = lerp(c[2], c[3], t);

        // Interpolate over phiD
        t = findexPhiD - indexPhiD;
        c[0] = lerp(c[0], c[2], t);

        return c[0];

    } else {

        // Use piece-wise constant lookup
        return getColor(indexThetaH, indexThetaD, indexPhiD);
    }
}


void
IsotropicBsdfTable::setBsdf(int indexThetaH, int indexThetaD, int indexPhiD,
        const Color &color) const
{
    // Compute index into table
    int index = getIndex(indexThetaH, indexThetaD, indexPhiD);
    mData[index] = color.r;
    mData[index + 1] = color.g;
    mData[index + 2] = color.b;
}


//----------------------------------------------------------------------------

} // namespace shading
} // namespace moonray

