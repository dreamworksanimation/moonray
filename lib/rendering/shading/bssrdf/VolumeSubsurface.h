// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file VolumeSubsurface.h
///

#include <moonray/rendering/shading/bsdf/Fresnel.h>
#include <moonray/rendering/shading/ispc/bssrdf/VolumeSubsurface_ispc_stubs.h>

#include <scene_rdl2/common/math/Math.h>
#include <scene_rdl2/render/util/Arena.h>
#include <scene_rdl2/scene/rdl2/Material.h>
#include <scene_rdl2/scene/rdl2/TraceSet.h>
#include <scene_rdl2/scene/rdl2/Types.h>

#pragma once

namespace moonray {
namespace shading {

ISPC_UTIL_TYPEDEF_STRUCT(VolumeSubsurface, VolumeSubsurfacev);


/// VolumeSubsurface is a bssrdf/bsdf like shading time allocated object
/// storing volume coefficients for path trace subsurface scattering
class VolumeSubsurface
{
public:
    enum Property {
        PROPERTY_NONE         = 0,
        PROPERTY_COLOR        = 1 << 0,
        PROPERTY_RADIUS       = 1 << 1,
        PROPERTY_PBR_VALIDITY = 1 << 2
    };

    explicit VolumeSubsurface(const scene_rdl2::math::Color& trlColor,
                              const scene_rdl2::math::Color& trlRadius,
                              const scene_rdl2::rdl2::Material* material,
                              scene_rdl2::rdl2::EvalNormalFunc evalNormalFn,
                              const scene_rdl2::math::Vec3f& N,
                              bool resolveSelfIntersections
                              );

    // Constructor for ISPC Hybrid Code
    VolumeSubsurface(scene_rdl2::alloc::Arena *arena,
                     const VolumeSubsurfacev &volumeSubsurfacev,
                     int lane);

    scene_rdl2::math::Color getScatteringColor() const {
        return mScatteringColor;
    }

    scene_rdl2::math::Color getScatteringRadius() const {
        return mScatteringRadius;
    }

    scene_rdl2::math::Color getAlbedo() const {
        return mAlbedo;
    }

    scene_rdl2::math::Color getSigmaT() const {
        return mSigmaT;
    }

    void setScale(const scene_rdl2::math::Color& scale) {
        mScale = scale;
    }

    scene_rdl2::math::Color getScale() const {
        return mScale;
    }

    void setTraceSet(const scene_rdl2::rdl2::TraceSet* traceSet) {
        mTraceSet = traceSet;
    }

    const scene_rdl2::rdl2::TraceSet* getTraceSet() const {
        return mTraceSet;
    }

    const scene_rdl2::rdl2::Material* getMaterial() const {
        return mMaterial;
    }

    scene_rdl2::rdl2::EvalNormalFunc getEvalNormalFunc() const {
        return mEvalNormalFunc;
    }

    void setTransmissionFresnel(Fresnel* fresnel) {
        mFresnel = fresnel;
    }

    const Fresnel* getTransmissionFresnel() const {
        return mFresnel;
    }

    Fresnel* getTransmissionFresnel() {
        return mFresnel;
    }

    void setLabel(int label) {
        mLabel = label;
    }

    int getLabel() const {
        return mLabel;
    }

    bool hasProperty(Property property) const {
        return mPropertyFlags & property;
    }

    int32_t getPropertyFlags() const {
        return mPropertyFlags;
    }

    bool getProperty(Property property, float *dest) const;

    scene_rdl2::math::Vec3f getN() const {
        return mN;
    }

    scene_rdl2::math::Color getZeroScatterSigmaT() const {
        return mZeroScatterSigmaT;
    }

    bool resolveSelfIntersections() const {
        return mResolveSelfIntersections;
    }

    scene_rdl2::math::Color getDwivediV0() const {
        return mDwivediV0;
    }

    scene_rdl2::math::Color getDwivediNormPDF() const {
        return mDwivediNormPDF;
    }

private:
    // user specified albedo color
    // (the multiple scattering integration result on semi-infinite slab case)
    scene_rdl2::math::Color mScatteringColor;
    // user specified volume mean free path
    scene_rdl2::math::Color mScatteringRadius;
    // volume extinction coefficient (sigmaT = sigmaA + sigmaS)
    scene_rdl2::math::Color mSigmaT;
    // extinction coeff for exiting the surface without scattering inside
    scene_rdl2::math::Color mZeroScatterSigmaT;
    // volume scattering coefficient albedo (albedo = sigmaS / sigmaT)
    scene_rdl2::math::Color mAlbedo;

    // a scaling factor similar to what bsdf/bssrdf have
    scene_rdl2::math::Color mScale;
    // Fresnel closure to eval when entering subsurface
    Fresnel* mFresnel;
    // Traceset this voulme subsurface is grouped to
    const scene_rdl2::rdl2::TraceSet* mTraceSet;
    // pointer to the subsurface material
    const scene_rdl2::rdl2::Material* mMaterial;
    // normal map evaluation function pointer
    scene_rdl2::rdl2::EvalNormalFunc mEvalNormalFunc;
    // aov label
    int mLabel;
    // bit flag to query whether certain properties are available (e.g. Color)
    int32_t mPropertyFlags;
    // intersection entry shading normal vector
    scene_rdl2::math::Vec3f mN;
    // toggle to control ignoring self intersecting geometry
    bool mResolveSelfIntersections;

    // MOONRAY-3105 - optimized dwivedi sampling
    // References:
    // (1) https://cgg.mff.cuni.cz/~jaroslav/papers/2014-zerovar/2014-zerovar-abstract.pdf
    // (2) https://jo.dreggn.org/home/2016_dwivedi.pdf
    // (3) https://jo.dreggn.org/home/2016_dwivedi_additional.pdf
    // Desmos Graph : https://www.desmos.com/calculator/fjaxaxu9sp
    // Dwivedi Sampling Related Params
    scene_rdl2::math::Color mDwivediV0;
    scene_rdl2::math::Color mDwivediNormPDF;
};

// Factory functions to create VolumeSubsurface.

VolumeSubsurface*
createVolumeSubsurface(scene_rdl2::alloc::Arena* arena,
                       const scene_rdl2::math::Color& trlColor,
                       const scene_rdl2::math::Color& trlRadius,
                       const scene_rdl2::rdl2::Material* material,
                       scene_rdl2::rdl2::EvalNormalFunc evalNormalFn,
                       const scene_rdl2::math::Vec3f& N,
                       bool resolveSelfIntersections);

VolumeSubsurface*
createVolumeSubsurface(scene_rdl2::alloc::Arena* arena,
                       const VolumeSubsurfacev *volumeSubsurfacev,
                       int lane);

//----------------------------------------------------------------------------

} // namespace shading
} // namespace moonray

