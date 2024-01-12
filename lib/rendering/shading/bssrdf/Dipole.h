// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
///

#pragma once

#include "Bssrdf.h"
#include <moonray/rendering/shading/ispc/bssrdf/Dipole_ispc_stubs.h>

#include <scene_rdl2/common/math/Color.h>
#include <scene_rdl2/common/math/Vec3.h>
#include <scene_rdl2/common/math/ReferenceFrame.h>

namespace moonray {
namespace shading {

ISPC_UTIL_TYPEDEF_STRUCT(DipoleBssrdf, DipoleBssrdfv);

//----------------------------------------------------------------------------

///
/// @class DipoleBssrdf Dipole.h <shading/bssrdf/Dipole.h>
/// @brief Derived class to implement the dipole Bssrdf with importance sampling.
///
/// This Bssrdf implements the dipole approximation as in (multiple scattering
/// only, single scattering is ignored):
/// [1] "A Practical Model for Subsurface Light Transport",
///     Jensen et al., Siggraph 2001
///
/// It implements an exact dipole importance sampling scheme for local multiple
/// scattering is described in:
/// [2] "Efficient Rendering of Local Subsurface Scattering", Mertens et al.,
///     Pacific Conference on Computer Graphics and Applications, 2003
///
/// It uses optionally the Bssrdf re-parameterization described in:
/// [3] "A Rapid Hierarchical Rendering Technique for Translucent Materials",
///     Jensen & Buhler, Siggraph 2002
///
class DipoleBssrdf : public Bssrdf
{
public:
    /// Initialize with standard physical parameters (cross sections in mm-1)
    /// The cross sections are converted to world-units-1 (in render space)
    /// according to the world scale:
    ///      world space unit * worldScale = meter
    DipoleBssrdf(const scene_rdl2::math::Vec3f &N, float eta, const scene_rdl2::math::Color &sigmaA,
           const scene_rdl2::math::Color &sigmaSP, float sceneScale /* = 0.01f */,
           const scene_rdl2::rdl2::Material* material,
           const scene_rdl2::rdl2::EvalNormalFunc evalNormalFn);

    /// Initialize with artist-friendly parameters (see [3] for conversion),
    /// where the color-dependent radius is specified in render space. Note that
    /// the translucentColor cannot be grater than 1 due to the applied conversion.
    DipoleBssrdf(const scene_rdl2::math::Vec3f &N, float eta, const scene_rdl2::math::Color &translucentColor,
           const scene_rdl2::math::Color &radius, const scene_rdl2::rdl2::Material* material,
           const scene_rdl2::rdl2::EvalNormalFunc evalNormalFn);

    /// Initialize using the corresponding vectorized version of this class.
    DipoleBssrdf(scene_rdl2::alloc::Arena *arena, const DipoleBssrdfv &bssrdfv, int lane);

    ~DipoleBssrdf() {}

    // TODO: re-introduce overall translucencyFactor to enable "better dipole"
    // paper normalization constant correction to Sd ?


    /// Sampling API:

    /// This function returns Rd(r), equation (4) in [1]... with errors from
    /// the paper fixed.
    // TODO: This should return the full Bssrdf (defined in [1] in chap. 2.3)
    // instead of returning Rd(r). Perhaps we can split single and multiple
    // scattering, but it should include one or two of the fresnel transmission
    // terms.
    scene_rdl2::math::Color eval(float r, bool global = false) const override;

    /// We use the exact importance sampling technique described in [2]. We sample
    /// from the three wavelengths with MIS using veach's one sample model.
    float sampleLocal(float r1, float r2, scene_rdl2::math::Vec3f &dPi, float &r) const override;

    /// This is only needed for unit-testing and verify that the pdf integrates to 1.
    float pdfLocal(float r) const override;

    scene_rdl2::math::Color diffuseReflectance() const override { return mDiffuseReflectance; }

    bool getProperty(Property property, float *dest) const override;

    void show(std::ostream& os, const std::string& indent) const override;

private:
    // Copy is disabled
    DipoleBssrdf(const DipoleBssrdf &other);
    const DipoleBssrdf &operator=(const DipoleBssrdf &other);


    /// This corresponds to the radius passed in the second constructor, which
    /// is also 1 / sigmaTr
    const scene_rdl2::math::Color getDiffuseMeanFreePath() const
    {
        return scene_rdl2::math::Color(1.0f / mSigmaTr.r, 1.0f / mSigmaTr.g, 1.0f / mSigmaTr.b);
    }

    /// Finish initialization of this class
    void finishInit();


    /// All scattering coefficients and constants, as defined in [1]
    scene_rdl2::math::Color mSigmaA;
    scene_rdl2::math::Color mSigmaSP;

    float mA;
    scene_rdl2::math::Color mSigmaTP;
    scene_rdl2::math::Color mSigmaTr;
    scene_rdl2::math::Color mAlphaP;
    scene_rdl2::math::Color mZr;
    scene_rdl2::math::Color mZv;

    /// Used for local importance sampling.
    scene_rdl2::math::Color mChannelCdf;
    scene_rdl2::math::Color mRatio;
    scene_rdl2::math::Color mPdfNormFactor;

    scene_rdl2::math::Color mDiffuseReflectance;

    scene_rdl2::math::Color mAlbedo; // used only to for PBR validity
};


//----------------------------------------------------------------------------

} // namespace shading
} // namespace moonray

