// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file Ray.cc

#include "Ray.h"
#include <scene_rdl2/common/platform/HybridVaryingData.h>

/// Defining this avoid rays differential computations. Mainly useful for
/// profiling. TODO: put this on a runtime flag?
//#define FORCE_SKIP_RAY_DIFFERENTIALS

using scene_rdl2::math::isfinite;

namespace ispc {
extern "C" uint32_t RayExtension_hvdValidation(bool);
extern "C" uint32_t RayDifferential_hvdValidation(bool);
}

namespace moonray {
namespace mcrt_common {

HVD_VALIDATOR(RayExtension);
HVD_VALIDATOR(RayDifferential);

bool
Ray::isValid() const
{
    MNRY_ASSERT(isFinite(getOrigin()));
    MNRY_ASSERT(isNormalized(getDirection()));
    MNRY_ASSERT(isfinite(getStart()));
    MNRY_ASSERT(isfinite(getEnd()) && getStart() >= 0.f);
    MNRY_ASSERT(getStart() <= getEnd());
    MNRY_ASSERT(isfinite(getTime()));
    return true;
}

std::ostream&
Ray::print(std::ostream& cout) const
{
    cout << "{ "
        << "org = " << org
        << ", dir = " << dir
        << ", near = " << tnear
        << ", far = " << tfar
        << ", time = " << time
        << ", mask = " << mask
        << ", Ng = " << Ng
        << ", u = " << u
        << ", v = " << v
        << ", geomID = " << geomID
        << ", primID = " << primID
        << ", instID = " << instID
        << ", depth = " << ext.depth
        << " }" << std::endl;
    return cout;
}

std::ostream&
RayDifferential::print(std::ostream& cout) const
{
    Ray::print(cout);

    cout << "Aux origin x = " << mOriginX << std::endl;
    cout << "Aux origin y = " << mOriginY << std::endl;
    cout << "Aux dir x    = " << mDirX << std::endl;
    cout << "Aux dir y    = " << mDirY << std::endl;
    return cout;
}

bool
RayDifferential::isValid() const
{
    MNRY_ASSERT(Ray::isValid());

    MNRY_ASSERT(isFinite(mOriginX));
    MNRY_ASSERT(isFinite(mOriginY));

    if (hasDifferentials()) {
        MNRY_ASSERT(isNormalized(mDirX));
        MNRY_ASSERT(isNormalized(mDirY));
    }

    return true;
}

} // mcrt_common
} // moonray


