// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file BsdfTableAniso.h
/// $Id$
///

#pragma once

#include <scene_rdl2/common/math/Color.h>
#include <scene_rdl2/common/math/Vec3.h>

#include <string>


namespace moonray {
namespace shading {


//----------------------------------------------------------------------------

///
/// @class AnisotropicBsdfTable BsdfTableAniso.h <pbr/BsdfTableAniso.h>
/// @brief An 4D anisotropic bsdf lookup table that uses the (H, D)
/// parameterization of the hemisphere. See the following paper for details:
///
/// A New Change of Variables for Efficient BRDF Representation
/// Szymon Rusinkiewicz, Eurographics Workshop on Rendering 1998
/// http://www.cs.princeton.edu/~smr/papers/brdf_change_of_variables/
///
class AnisotropicBsdfTable
{
public:

    enum Interpolation {
        CONSTANT = 0,       /// Piecewise constant (no interpolation)
        LINEAR_PHIH = 1,    /// Linear interpolation across phiH slices
        QUADRI_LINEAR = 2   /// Quadri-linear interpolation across all 4D dimensions
    };


    explicit AnisotropicBsdfTable(bool reciprocal /* = true */);
    AnisotropicBsdfTable(int sizeThetaH, int sizePhiH, int sizeThetaD, int sizePhiD,
            bool reciprocal /* = true */);
    ~AnisotropicBsdfTable();

    // Load a tabulated brdf file
    // This function is not thread-safe and may throw in case of error
    AnisotropicBsdfTable(const std::string &filename);
    void saveAs(const std::string &filename) const;

    // Accessors
    int getSizeThetaH() const   {  return mSizeThetaH;  }
    int getSizePhiH() const   {  return mSizePhiH;  }
    int getSizeThetaD() const   {  return mSizeThetaD;  }
    int getSizePhiD() const   {  return mSizePhiD;  }

    // Convert between parameterizations
    void indexHD2localWoWi(int indexThetaH, int indexPhiH, int indexThetaD, int indexPhiD,
            scene_rdl2::math::Vec3f &localWo, scene_rdl2::math::Vec3f &localWi) const;
    bool localWoWi2IndexHD(const scene_rdl2::math::Vec3f &localWo, const scene_rdl2::math::Vec3f &localWi,
            int &indexThetaH, int &indexPhiH, int &indexThetaD, int &indexPhiD) const;

    // Return the Bsdf value for a pair of direction in the local reference frame
    scene_rdl2::math::Color getBsdf(const scene_rdl2::math::Vec3f &localWo, const scene_rdl2::math::Vec3f &localWi, Interpolation interp = CONSTANT) const;

    // Used to set measured Bsdf
    void setBsdf(int indexThetaH, int indexPhiH, int indexThetaD, int indexPhiD,
            const scene_rdl2::math::Color &color) const;


private:
    /// Copy is disabled
    AnisotropicBsdfTable(const AnisotropicBsdfTable &other);
    const AnisotropicBsdfTable &operator=(const AnisotropicBsdfTable &other);


    void init();

    // Returns mData size
    finline int getFloatCount() const   {
        return 3 * mSizeThetaH * mSizePhiH * mSizeThetaD * mSizePhiD;
    }

    // [0 .. pi/2]  <-->  [0 .. sizeThetaH[
    float getIndexThetaH(float thetaH) const;
    float getThetaH(int index) const;
    // [-pi .. pi]  <-->  [0 .. sizePhiH[
    float getIndexPhiH(float phiH) const;
    float getPhiH(int index) const;
    // [0 .. pi/2]  <-->  [0 .. sizeThetaD[
    float getIndexThetaD(float thetaD) const;
    float getThetaD(int index) const;
    // [-pi .. pi]  <-->  [0 .. sizePhiD[
    float getIndexPhiD(float phiD) const;
    float getPhiD(int index) const;

    finline int getIndex(int indexThetaH, int indexPhiH, int indexThetaD, int indexPhiD) const {
        MNRY_ASSERT(indexThetaH >= 0  &&  indexThetaH < mSizeThetaH);
        MNRY_ASSERT(indexPhiH >= 0  &&  indexPhiH < mSizePhiH);
        MNRY_ASSERT(indexThetaD >= 0  &&  indexThetaD < mSizeThetaD);
        MNRY_ASSERT(indexPhiD >= 0  &&  indexPhiD < mSizePhiD);
        return 3 * (indexPhiD +
                    indexThetaD * mSizePhiD +
                    indexPhiH   * mSizePhiD * mSizeThetaD +
                    indexThetaH * mSizePhiD * mSizeThetaD * mSizePhiH);
    }

    finline scene_rdl2::math::Color getColor(int indexThetaH, int indexPhiH, int indexThetaD, int indexPhiD) const {
        int index = getIndex(indexThetaH, indexPhiH, indexThetaD, indexPhiD);
        return scene_rdl2::math::Color(mData[index], mData[index + 1], mData[index + 2]);
    }

    // Used for quadri-linear interpolation
    bool localWoWi2IndexHD(const scene_rdl2::math::Vec3f &localWo, const scene_rdl2::math::Vec3f &localWi,
            float &indexThetaH, float &indexPhiH,
            float &indexThetaD, float &indexPhiD) const;


    // Members
    int mSizeThetaH;
    int mSizePhiH;
    int mSizeThetaD;
    int mSizePhiD;
    bool mReciprocal;
    float *mData;
};


//----------------------------------------------------------------------------

} // namespace shading
} // namespace moonray

