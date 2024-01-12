// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file Intersection.cc
/// $Id$
///

#include "Intersection.h"

#include <moonray/rendering/mcrt_common/Ray.h>
#include <moonray/rendering/mcrt_common/ThreadLocalState.h>
#include <scene_rdl2/common/platform/HybridVaryingData.h>

namespace ispc {
extern "C" uint32_t Intersection_hvdValidation(bool);
}

namespace moonray {
namespace shading {

using namespace scene_rdl2::math;

constexpr float sIntersectionEpsilon = 1e-5f;

const uint32_t Intersection::sPathTypeMask;
const uint32_t Intersection::sPathTypeOffset;

//----------------------------------------------------------------------------

void
Intersection::transferAndComputeDerivatives(mcrt_common::ThreadLocalState *tls,
                                            mcrt_common::RayDifferential *ray,
                                            float textureDiffScale)
{
    EXCL_ACCUMULATOR_PROFILE(tls, EXCL_ACCUM_COMPUTE_RAY_DERIVATIVES);

    MNRY_ASSERT(mFlags.get(GeomInitialized));

    // Transfer ray differentials to hit point
    ray->transfer(getP(), getNg());

    // Initialize texture derivatives to 0 so we sample at the finest mip level.
    mdSdx = mdTdx = mdSdy = mdTdy = 0.0f;

    if (!ray->hasDifferentials()) {
        // If we fail to compute the aux ray positions, it's probably
        // because one of the aux rays is at a super glancing angle wrt the
        // hit plane. If that's the case, we should be sampling at our coarsest
        // mip level.
        mdSdx = 1024.0f;
        mdTdy = 1024.0f;

    } else if (hasGeometryDerivatives()) {
        // The intersection has valid data to compute texture derivatives.

        // transfer is assumed to be already called on the ray
        const scene_rdl2::math::Vec3f dpdx = ray->getdPdx();
        const scene_rdl2::math::Vec3f dpdy = ray->getdPdy();

        // Take a short cut if dpdu and dpdv are very close to forming
        // an orthogonal basis. When that's the case, simple projection will
        // suffice to find our derivatives.
        if (isZero(dot(mdPds, mdPdt), sIntersectionEpsilon)) {

            float dotDpdu = dot(mdPds, mdPds);
            float dotDpdv = dot(mdPdt, mdPdt);
            if (dotDpdu > sIntersectionEpsilon && dotDpdv > sIntersectionEpsilon) {
                float invDotDpdu = 1.0f / dot(mdPds, mdPds);
                float invDotDpdv = 1.0f / dot(mdPdt, mdPdt);
                mdSdx = dot(dpdx, mdPds) * invDotDpdu;
                mdTdx = dot(dpdx, mdPdt) * invDotDpdv;
                mdSdy = dot(dpdy, mdPds) * invDotDpdu;
                mdTdy = dot(dpdy, mdPdt) * invDotDpdv;
            }
        } else {
            // We have an non-orthgonal basis, plus the system is over constrained.
            // Find largest component of normal (since it's orthogonal to both dpdu
            // and dpdv) and discard that axis. See pbrt 2nd ed. pg 508 for details.

            scene_rdl2::math::Vec4f A;
            scene_rdl2::math::Vec2f bx, by;

            // component wise multiply to get rid of negative signs
            const scene_rdl2::math::Vec3f n2 = mNg * mNg;
            if (n2.x > n2.y && n2.x > n2.z) {
                // discard x axis, use y and z axes
                A  = scene_rdl2::math::Vec4f(mdPds.y, mdPdt.y, mdPds.z, mdPdt.z);
                bx = scene_rdl2::math::Vec2f(dpdx.y, dpdx.z);
                by = scene_rdl2::math::Vec2f(dpdy.y, dpdy.z);
            } else if (n2.y > n2.z) {
                // discard y axis, use x and z axes
                A  = scene_rdl2::math::Vec4f(mdPds.x, mdPdt.x, mdPds.z, mdPdt.z);
                bx = scene_rdl2::math::Vec2f(dpdx.x, dpdx.z);
                by = scene_rdl2::math::Vec2f(dpdy.x, dpdy.z);
            } else {
                // discard z axis, use x and y axes
                A  = scene_rdl2::math::Vec4f(mdPds.x, mdPdt.x, mdPds.y, mdPdt.y);
                bx = scene_rdl2::math::Vec2f(dpdx.x, dpdx.y);
                by = scene_rdl2::math::Vec2f(dpdy.x, dpdy.y);
            }

            float detA = A.x * A.w - A.y * A.z;
            if (!isZero(detA, sIntersectionEpsilon)) {
                MNRY_VERIFY(solve2x2LinearSystem(&A.x, &mdSdx, &bx.x));
                MNRY_VERIFY(solve2x2LinearSystem(&A.x, &mdSdy, &by.x));
            }
        }
    }

    // Texture derivatives are always assumed to be valid by this point,
    // regardless if we had valid ray differentials or valid geometry derivatives.
    MNRY_ASSERT(finite(mdSdx) && finite(mdTdx) && finite(mdSdy) && finite(mdTdy));

    // Scale the texture derivatives
    mdSdx *= textureDiffScale;
    mdSdy *= textureDiffScale;
    mdTdx *= textureDiffScale;
    mdTdy *= textureDiffScale;
}

//----------------------------------------------------------------------------

std::ostream&
Intersection::print(std::ostream& cout) const
{
    cout << "Entering = " << isEntering() << std::endl;
    cout << "HasGeometryDerivatives = " << hasGeometryDerivatives() << std::endl;
    return cout;
}


//----------------------------------------------------------------------------

scene_rdl2::math::Vec3f
Intersection::adaptToonNormal(const scene_rdl2::math::Vec3f &Ns) const
{
    // This flag is set by the integrator if the ToonMaterial has
    // indicated that no light culling should be performed. If so,
    // the normal is not adapted. This is useful for certain NPR effects
    // that use an arbitrary normal.
    if (!mFlags.get(UseAdaptNormal)) {
        return Ns;
    }

    return adaptNormal(Ns);
}

scene_rdl2::math::Vec3f
Intersection::adaptNormal(const scene_rdl2::math::Vec3f &Ns) const
{
    // Ns is the result of a normal mapping operation.
    // The result may or may not be physically plausible from
    // our particular view point.  This function detects this
    // case and minimally bends the shading normal back towards
    // the physically plausible geometric normal.  This unfortunately
    // breaks bi-directional algorithms.  See "Microfacet-based
    // Normal Mapping for Robust Monte Carlo Path Tracing" for
    // an apporach to normal mapping that can work bi-directionally.
    //
    // The solution employed here is descrbied in
    // "The Iray Light Transport System" Section A.3 "Local Shading
    // Normal Adaption".
    //
    const scene_rdl2::math::Vec3f &wo = getWo();
    const scene_rdl2::math::Vec3f &Ng = getNg();

    scene_rdl2::math::Vec3f result = Ns;

    // Compute reflection vector r, which is wo reflected about Ns
    const scene_rdl2::math::Vec3f r = 2.f * dot(Ns, wo) * Ns - wo;

    // If r is above the horizon of Ng, we're in good shape
    const float rDotNg = dot(r, Ng);

    // compute a bent reflection vector that is just
    // slightly (based on eps) above the Ng horizon
    const float eps = 0.01f;
    if (rDotNg < eps) {
        // this appears to work, but has the wrong normalization
        // if this becomes a problem, the correct normalization can be
        // computed via:
        //   const float s = scene_rdl2::math::sqrt(scene_rdl2::math::max(0.f, (1 - rDotNg * rDotNg)) / (1 - eps * eps));
        //   const float alpha = eps * s - rDotNg;
        //   scene_rdl2::math::Vec3f rBent = r + alpha * Ng;
        // 
        scene_rdl2::math::Vec3f rBent = r + (eps - rDotNg) * Ng;

        // Normalize it.
        // Worth noting: since we are going to normalize(wo + rBent)
        // we could alternatively scale wo via wo = wo * |rBent| and obtain
        // the same result for bentNs.
        rBent = scene_rdl2::math::normalize(rBent);

        // Compute bent Ns as halfway vector between wo and rBent
        scene_rdl2::math::Vec3f bentNs = scene_rdl2::math::normalize(wo + rBent);
        result = bentNs;
    }

    return result;
}

//----------------------------------------------------------------------------

HVD_VALIDATOR(Intersection);

} // namespace shading
} // namespace moonray

