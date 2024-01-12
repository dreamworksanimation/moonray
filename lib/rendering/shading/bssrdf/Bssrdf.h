// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file Bssrdf.h
/// $Id$
///

#pragma once

#include <moonray/rendering/shading/ispc/bssrdf/Bssrdf_ispc_stubs.h>
#include <moonray/rendering/shading/ispc/BsdfComponent_ispc_stubs.h>
#include <scene_rdl2/common/math/Color.h>
#include <scene_rdl2/common/math/ReferenceFrame.h>
#include <scene_rdl2/common/math/Vec3.h>
#include <scene_rdl2/common/platform/IspcUtil.h>
#include <scene_rdl2/scene/rdl2/TraceSet.h>
#include <scene_rdl2/scene/rdl2/Material.h>

namespace scene_rdl2 {
namespace alloc { class Arena; }
}
namespace moonray {
namespace shading {

ISPC_UTIL_TYPEDEF_STRUCT(Bssrdf, Bssrdfv);

class Fresnel;

//----------------------------------------------------------------------------

///
/// @class Bssrdf Bssrdf.h <shading/bssrdf/Bssrdf.h>
/// @brief Base class to implement a Bssrdf with importance sampling.
///
/// The integrator uses a local multiple scattering sampling as the
/// method described in:
/// [2] "Efficient Rendering of Local Subsurface Scattering", Mertens et al.,
///     Pacific Conference on Computer Graphics and Applications, 2003
/// The integrator uses a global multiple scattering sampling as the
/// approximation described in:
/// [4] "Efficient rendering of human skin", D'Eon et al., EGSR 2007
///

class Bssrdf
{
public:
    enum Property {
        PROPERTY_NONE         = 0,
        PROPERTY_COLOR        = 1 << 0,
        PROPERTY_RADIUS       = 1 << 1,
        PROPERTY_PBR_VALIDITY = 1 << 2
    };

    Bssrdf(const scene_rdl2::math::Vec3f &N, int32_t propertyFlags,
           const scene_rdl2::rdl2::Material* material,
           const scene_rdl2::rdl2::EvalNormalFunc evalNormalFn);
    Bssrdf(scene_rdl2::alloc::Arena *arena, const Bssrdfv &bssrdfv, int lane);

    virtual ~Bssrdf() {}

    /// A bssrdf can be scaled by a color
    finline void setScale(const scene_rdl2::math::Color &scale) {  mScale = scale;  }
    finline scene_rdl2::math::Color getScale() const            {  return mScale;  }

    // A bssrdf can have a trace set
    finline void setTraceSet(const scene_rdl2::rdl2::TraceSet *traceSet) { mBssrdfTraceSet = traceSet; }
    finline const scene_rdl2::rdl2::TraceSet *getTraceSet() const { return mBssrdfTraceSet; }

    // A pointer to the subsurface material
    const scene_rdl2::rdl2::Material* getMaterial() const { return mMaterial; }

    // The normal map evaluation function pointer
    scene_rdl2::rdl2::EvalNormalFunc getEvalNormalFunc() const { return mEvalNormalFunc; }

    /// The bssrdf takes ownership of the Fresnel object.
    /// This fresnel object is a transmission fresnel, so make sure to wrap your
    /// actual fresnel closure in a OneMinus*Fresnel adapter.
    void setTransmissionFresnel(Fresnel *fresnel)           {  mFresnel = fresnel;  }
    finline Fresnel *getTransmissionFresnel() const   {  return mFresnel;  }


    /// Sampling API:

    /// This function returns Rd(r), the diffuse reflectance
    /// The radius r should be in render space
    /// The global flag tells the bssrdf whether this is for local scattering
    /// or global scattering.
    /// Important note: it is the responsibility of the integrator to query
    /// getScale() and getTransmissionFresnel() and to apply it accordingly.
    /// Unlike for Bsdf, the result of eval() does NOT contain any scale nor
    /// fresnel term.
    virtual scene_rdl2::math::Color eval(float r, bool global = false) const = 0;

    /// Sample a position around the origin, in the tangent plane as defined by N.
    /// We also return the distance r from the origin for the sampled position.
    /// It's up to the integrator to re-project the sampled points onto the surface.
    /// We return the pdf of sampling dPi with respect to surface area
    /// measure (this is sTwoPi * r away from the radial pdf of sampling r).
    virtual float sampleLocal(float r1, float r2, scene_rdl2::math::Vec3f &dPi, float &r) const = 0;

    /// This is only needed for unit-testing and verify that the pdf integrates to 1.
    virtual float pdfLocal(float r) const = 0;

    /// For Diffusion-Based BSSRDFs, we use the analytically computed
    /// diffuse reflectance to compute the area-compensation factor
    virtual scene_rdl2::math::Color diffuseReflectance() const = 0;

    /// Sample a direction going into the sub-surface, to go search for global
    /// scattering, through a thin layer. It's up to the caller to trace through
    /// the volume to find the back-side of the thin layer (Pi, r) and apply the
    /// Bssrdf accordingly.
    /// We return the pdf of sampling the direction ws with respect to the
    /// solid angle measure.
    // TODO: Move cosThetaMax to ctor ?
    float sampleGlobal(float r1, float r2, float cosThetaMax, scene_rdl2::math::Vec3f &ws) const;


    /// This is the maximum radius the integrator should use to integrate
    /// subsurface scattering. Beyond this radius, the function Rd(d) is
    /// negligibly small.
    float getMaxRadius() const          {  return mMaxRadius;  }

    const scene_rdl2::math::ReferenceFrame &getFrame() const { return mFrame; }

    /// A label can be set on a bssrdf
    void setLabel(int label) { mLabel = label; }
    int getLabel() const { return mLabel; }

    /// bssrdfs can have certain properties (e.g. Color)
    /// this API is used to query if a bssrdf has a particular property and to
    /// evaluate that property value
    bool hasProperty(Property property) const { return mPropertyFlags & property; }
    int32_t getPropertyFlags() const { return mPropertyFlags; }
    virtual bool getProperty(Property property, float *dest) const;

    // prints out a description of this bssrdf with the provided indentation
    // prepended.
    virtual void show(std::ostream& os, const std::string& indent) const = 0;

private:
    // Copy is disabled
    Bssrdf(const Bssrdf &other);
    const Bssrdf &operator=(const Bssrdf &other);


protected:

    Fresnel *mFresnel;
    scene_rdl2::math::ReferenceFrame mFrame;
    scene_rdl2::math::Color mScale;

    float mMaxRadius;
    int mLabel;
    int32_t mPropertyFlags;

    const scene_rdl2::rdl2::TraceSet* mBssrdfTraceSet;
    const scene_rdl2::rdl2::Material* mMaterial;
    scene_rdl2::rdl2::EvalNormalFunc mEvalNormalFunc;
};


//----------------------------------------------------------------------------

// Factory functions to create BSSRDFs.

Bssrdf *createBSSRDF(ispc::SubsurfaceType type,
                     scene_rdl2::alloc::Arena* arena,
                     const scene_rdl2::math::Vec3f &N,
                     const scene_rdl2::math::Color& trlColor,
                     const scene_rdl2::math::Color& trlRadius,
                     const scene_rdl2::rdl2::Material* material,
                     scene_rdl2::rdl2::EvalNormalFunc evalNormalFn);

Bssrdf *createBSSRDF(scene_rdl2::alloc::Arena* arena,
                     const Bssrdfv *bssrdfv,
                     int lane);

//----------------------------------------------------------------------------


} // namespace shading
} // namespace moonray

