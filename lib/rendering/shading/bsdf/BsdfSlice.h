// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file BsdfSlice.h
/// $Id$
///

#pragma once

#include "Bsdf.h"

#include <moonray/rendering/shading/ispc/bsdf/BsdfSlice_ispc_stubs.h>

#include <scene_rdl2/common/math/Vec3.h>

namespace moonray {
namespace shading {

//----------------------------------------------------------------------------

/// A bsdf slice can request that bsdf lobe evaluation perform
/// shadow terminator smoothing
typedef ispc::ShadowTerminatorFix ShadowTerminatorFix;

///
/// @class BsdfSlice Bsdf.h <pbr/Bsdf.h>
/// @brief A BsdfSlice object is used to both represent a lower-dimensional
/// slice of a multi-lobe bsdf and to evaluate multi-lobe bsdfs.
///
class BsdfSlice
{
public:
    /// A bsdf slice is defined given a surface geometric normal Ng and
    /// observer direction wo. We are also narrowing down on only the lobes that
    /// match the given flags. It also controls when a Bsdf is evaluated or
    /// sampled, whether or not its result includes the cosine term,
    /// and what if any shadow terminator smoothing should be applied.
    BsdfSlice(const scene_rdl2::math::Vec3f Ng, const scene_rdl2::math::Vec3f &wo, bool includeCosineTerm,
            bool entering, ShadowTerminatorFix shadowTerminatorFix, BsdfLobe::Type flags = BsdfLobe::ALL) :
        mNg(Ng),
        mWo(wo),
        mFlags(flags),
        mShadowTerminatorFix(shadowTerminatorFix),
        mIncludeCosineTerm(includeCosineTerm),
        mEntering(entering) {}

    virtual ~BsdfSlice() {}


    // Accessors
    const scene_rdl2::math::Vec3f &getNg() const          {  return mNg;  }
    const scene_rdl2::math::Vec3f &getWo() const          {  return mWo;  }
    BsdfLobe::Type getFlags() const                       {  return mFlags;  }
    bool getIncludeCosineTerm() const                     {  return mIncludeCosineTerm;  }
    bool getEntering() const                              {  return mEntering;  }
    ShadowTerminatorFix getShadowTerminatorFix() const    {  return mShadowTerminatorFix; }


    /// This method returns true if a given sample is compatible with the given
    /// lobe and according to the geometric surface. It tests if
    /// wo and wi are in the same / opposite hemispheres wrt. Ng and checks
    /// whether this is compatible with the lobe type REFLECTION vs. TRANSMISSION.
    /// This method always returns false for spherical lobes.
    finline bool isSampleGeometricallyInvalid(const BsdfLobe &lobe, const scene_rdl2::math::Vec3f &wi) const
    {
        return (lobe.getIsSpherical()  ?  false  :
                !(lobe.getType() & ((dot(mNg, mWo) * dot(mNg, wi) > 0.0f)  ?
                        BsdfLobe::REFLECTION : BsdfLobe::TRANSMISSION)));
    }

    /// Computes a lobe type flag that lets integrators ignore btdfs or brdfs
    /// based on whether wo and wi are respectively on the same or opposite side
    /// of the geometric surface (wrt. Ng).
    /// For spherical Bsdfs, this always returns the unmodified slice flags.
    finline BsdfLobe::Type getSurfaceFlags(const Bsdf &bsdf, const scene_rdl2::math::Vec3f &wi) const
    {
        return (bsdf.getIsSpherical()  ?  mFlags  :
                BsdfLobe::Type(mFlags & ((dot(mNg, mWo) * dot(mNg, wi) > 0.0f)  ?
                            ~BsdfLobe::TRANSMISSION : ~BsdfLobe::REFLECTION)));
    }
    finline BsdfLobe::Type getSurfaceFlags(const BsdfLobe &lobe, const scene_rdl2::math::Vec3f &wi) const
    {
        return (lobe.getIsSpherical()  ?  mFlags  :
                BsdfLobe::Type(mFlags & ((dot(mNg, mWo) * dot(mNg, wi) > 0.0f)  ?
                            ~BsdfLobe::TRANSMISSION : ~BsdfLobe::REFLECTION)));
    }

    /// Compute a visibility multipilier based on shadow terminator fix mode
    /// N should be the lobe's shading normal and wi is the incoming light direction
    float computeShadowTerminatorFix(const scene_rdl2::math::Vec3f &N, const scene_rdl2::math::Vec3f &wi) const;
    

protected:
    scene_rdl2::math::Vec3f mNg;
    scene_rdl2::math::Vec3f mWo;
    BsdfLobe::Type mFlags;
    ShadowTerminatorFix mShadowTerminatorFix;
    bool mIncludeCosineTerm;
    bool mEntering;
};


//----------------------------------------------------------------------------

// ispc vector types
ISPC_UTIL_TYPEDEF_STRUCT(BsdfSlice, BsdfSlicev)

} // namespace shading
} // namespace moonray

