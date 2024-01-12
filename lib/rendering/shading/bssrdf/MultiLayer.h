// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
///

#pragma once

#include "Bssrdf.h"

#include <scene_rdl2/common/math/Color.h>
#include <scene_rdl2/common/math/Vec3.h>

#define BSSRDF_MAX_LOBES 4

namespace moonray {
namespace shading {

//----------------------------------------------------------------------------

///
/// @class MultiLayerBssrdf MultiLayer.h <pbr/bssrdf/MultiLayer.h>
/// @brief Derived class to implement MultiLayer Diffusion Bssrdf with weighted importance sampling.
///
///
class MultiLayerBssrdf : public Bssrdf
{
public:
    // Artist-friendly constructor
    MultiLayerBssrdf(const scene_rdl2::math::Vec3f& N);
    ~MultiLayerBssrdf();

    void addBssrdf(Bssrdf* nBssrdf, float nweight);
    void finalize();

    // This function returns R(r), equation (3) in [1]
    scene_rdl2::math::Color eval(float r, bool global = false) const override;
    float sampleLocal(float r1, float r2, scene_rdl2::math::Vec3f &dPi, float &r) const override;

    // This is only needed for unit-testing and verify that the pdf integrates to 1.
    float pdfLocal(float r) const override;

    /// For Diffusion-Based BSSRDFs, we use the analytically computed
    /// diffuse reflectance to compute the area-compensation factor
    virtual scene_rdl2::math::Color diffuseReflectance() const override;

    bool getProperty(Property property, float *dest) const override;

    void show(std::ostream& os, const std::string& indent) const override;

private:
    // Copy is disabled
    MultiLayerBssrdf(const MultiLayerBssrdf &other) =delete;
    const MultiLayerBssrdf &operator=(const MultiLayerBssrdf &other) =delete;
    unsigned int selectIndex(float r) const;

    Bssrdf* mBssrdfs[BSSRDF_MAX_LOBES];
    float mWeights[BSSRDF_MAX_LOBES];
    float mCdfs[BSSRDF_MAX_LOBES];
    unsigned int mLobeCount;
};


} // namespace shading
} // namespace moonray

