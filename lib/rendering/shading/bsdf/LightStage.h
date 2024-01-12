// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <scene_rdl2/common/math/Color.h>
#include <scene_rdl2/common/math/Mat4.h>
#include <scene_rdl2/common/math/Vec2.h>
#include <scene_rdl2/common/math/Vec3.h>

namespace moonray {
namespace shading {


//---------------------------------------------------------------------------

// Light Stage X class, which provides light order for One Light At a Time (OLAT)
// data. It also provides light coordinates in "light-stage" space: (o, x, y, z):
//   o: physical center of the light-stage
//   x: points right,  y: points up,  z: points towards the camera
class LightStage {
public:
    // Singleton
    finline static const LightStage &singleton() {
        static const LightStage lightStage;
        return lightStage;
    }

    // Return a light index from the light order index from the spiraling OLAT
    // pattern for separated cross / parallel polarized sequence. The given
    // lightOrderIndex should be in the range [0..sSeparatedLightCount-1]
    static const int sSeparatedLightCount = 85;
    int getSeparatedLightIndex(int lightOrderIndex) const;

    // Return a light index from the light order index from the spiraling OLAT
    // pattern for combined (non-polarized) sequence. The given
    // lightOrderIndex should be in the range [0..sCombinedLightCount-1]
    static const int sCombinedLightCount = 201;
    int getCombinedLightIndex(int lightOrderIndex) const;

    // Get the position of the light
    scene_rdl2::math::Vec3f getLightPosition(int index) const;

private:
    // Singleton
    LightStage();
    ~LightStage()   {}

    // Copy is disabled
    LightStage(const LightStage &other);
    const LightStage &operator=(const LightStage &other);

    void printLightStageLightsInOrder();
};


//---------------------------------------------------------------------------

// Tabulated Bsdf data re-constructed from measured data on cylinder images
// scanned in the light stage. This stores one phiO slice, over the the light
// directions of the light-stage (using the separated OLAT pattern).
class LightStageCylinderBsdfSlice
{
public:
    LightStageCylinderBsdfSlice(bool top, int sizeThetaO, int lightCount,
            float cylRadius, float cylZCenter, float cylAlpha);
    ~LightStageCylinderBsdfSlice();

    LightStageCylinderBsdfSlice(const std::string &filename);
    void saveAs(const std::string &filename) const;

    // Accessors
    int getSizeThetaO() const   {  return mSizeThetaO;  }
    int getLightCount() const   {  return mLightCount;  }
    float getCylRadius() const    {  return mCylRadius;  }
    float getCylZCenter() const   {  return mCylZCenter;  }
    float getCylAlpha() const     {  return mCylAlpha;  }
    const scene_rdl2::math::Mat4f &getCyl2W() const  {  return mCyl2LS;  };
    const scene_rdl2::math::Mat4f &getW2Cyl() const  {  return mLS2Cyl;  };

    // Compute position P, normal N and tangent T on the cylinder in light-stage space
    void computeCylLightStagePNT(float thetaO, scene_rdl2::math::Vec3f &P, scene_rdl2::math::Vec3f &N, scene_rdl2::math::Vec3f &T) const;

    // See cmd/brdf_cmd/brdf_cylinder_extract
    void setBsdf(int lightOrderIndex, int indexThetaO, const scene_rdl2::math::Color &color);

    // Return the Bsdf value for a pair of direction in the local ReferenceFrame
    // When localWo is close to the normal, we smooth the bsdf around the pole
    // See implementation for details
    scene_rdl2::math::Color getBsdf(const scene_rdl2::math::Vec3f &localWo, const scene_rdl2::math::Vec3f &localWi,
            const scene_rdl2::math::Vec2f &smoothThetaWoRange, bool useLerp = true) const;


private:

    void init();

    // Returns mData size
    finline int getFloatCount() const   {  return 3 * mLightCount * mSizeThetaO;  }

    finline int getIndex(int lightIndex, int indexThetaO) const {
        MNRY_ASSERT(lightIndex >= 0  &&  lightIndex < mLightCount);
        MNRY_ASSERT(indexThetaO >= 0  &&  indexThetaO < mSizeThetaO);
        return 3 * (lightIndex * mSizeThetaO + indexThetaO);
    }

    scene_rdl2::math::Color getBsdfSample(const scene_rdl2::math::Vec3f &localWo, const scene_rdl2::math::Vec3f &localWi, bool useLerp = true) const;

    // Persistent data
    bool mTop;
    int mSizeThetaO;
    int mLightCount;
    float mCylRadius;
    float mCylZCenter;
    float mCylAlpha;
    float *mData;

    // Cylinder <--> Light-Stage space
    scene_rdl2::math::Mat4f mCyl2LS;
    scene_rdl2::math::Mat4f mLS2Cyl;
};


//---------------------------------------------------------------------------

} // namespace shading
} // namespace moonray

