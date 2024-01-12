// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#include "IntersectionInit.h"

#include <moonray/rendering/geom/Primitive.h>

#include <moonray/rendering/bvh/shading/InstanceAttributes.h>
#include <moonray/rendering/bvh/shading/Intersection.h>
#include <moonray/rendering/geom/prim/BVHUserData.h>
#include <moonray/rendering/geom/prim/Instance.h>
#include <moonray/rendering/geom/prim/Util.h>
#include <moonray/rendering/mcrt_common/ThreadLocalState.h>
#include <moonray/rendering/mcrt_common/ProfileAccumulatorHandles.h>
#include <moonray/rendering/mcrt_common/Types.h>

namespace moonray{
namespace geom {

void initIntersectionPhase1(shading::Intersection &isect,
                            mcrt_common::ThreadLocalState *tls,
                            const mcrt_common::Ray         &ray,
                            const scene_rdl2::rdl2::Layer *pRdlLayer)
{
    EXCL_ACCUMULATOR_PROFILE(tls, EXCL_ACCUM_INIT_INTERSECTION);

    isect.init();
    isect.setP(ray.getOrigin() + ray.getDirection() * ray.getEnd());

    // Invoke the primitive's post-intersection
    geom::internal::BVHUserData* userData =
        static_cast<geom::internal::BVHUserData*>(ray.ext.userData);
    const geom::internal::Primitive* primPtr = userData->mPrimitive;

    primPtr->postIntersect(*tls, pRdlLayer, ray, isect);

    // shadow ray epsilon is the same for all geometry primitive types.
    isect.setShadowEpsilonHint(primPtr->getRdlGeometry()->getShadowRayEpsilon());

    scene_rdl2::math::Vec3f dNds, dNdt;
    const bool curvatureComputed =
        primPtr->computeIntersectCurvature(ray, isect, dNds, dNdt);

    if (curvatureComputed) {
        isect.setdNds(dNds);
        isect.setdNdt(dNdt);
    } else {
        isect.setdNds(scene_rdl2::math::zero);
        isect.setdNdt(scene_rdl2::math::zero);
    }

    // We are done setting/overriding instance attributes, we, can now verify if
    // all required attributes are provided.
    isect.validateRequiredAttributes();

    // if the ray hits an instance, we need to apply local to render space
    // transform attached on ray to intersection
    if (ray.isInstanceHit()) {
        // TODO for primitive attributes that need to stay in
        // render space, we need to also apply correspoinding transform
        // on them

        // Override shadow ray epsilon from instance if it exists
        if (internal::hasInstanceAttr(shading::StandardAttributes::sShadowRayEpsilon, ray)) {
            const float shadowRayEpsilon =
                internal::getInstanceAttr(shading::StandardAttributes::sShadowRayEpsilon,
                                          ray,
                                          primPtr->getRdlGeometry()->getShadowRayEpsilon());
            isect.setShadowEpsilonHint(shadowRayEpsilon);
        }

        // Only transform dPds, dPdt, and N if they aren't explicitly
        // set on the instance's primitive attribute table
        if (!internal::hasInstanceAttr(shading::StandardAttributes::sdPds, ray)) {
            isect.setdPds(transformVector(ray.ext.l2r, isect.getdPds()));
        }
        if (!internal::hasInstanceAttr(shading::StandardAttributes::sdPdt, ray)) {
            isect.setdPdt(transformVector(ray.ext.l2r, isect.getdPdt()));
        }
        const scene_rdl2::math::Xform3f r2l = ray.ext.l2r.inverse();
        if (!internal::hasInstanceAttr(shading::StandardAttributes::sNormal, ray)) {
            isect.setNg(normalize(transformNormal(r2l, isect.getNg())));
            isect.setN(normalize(transformNormal(r2l, isect.getN())));
        }
        if (curvatureComputed) {
            isect.setdNds(transformNormal(r2l, isect.getdNds()));
            isect.setdNdt(transformNormal(r2l, isect.getdNdt()));
        }
    }

    // Are we entering or leaving the volume enclosed by the surface hit ?
    // TODO: does this affect hair shading in any way ?
    if (isect.isFlatPoint() || dot(isect.getNg(), ray.getDirection()) < 0.0f) {
        isect.setIsEntering(true);
    } else {
        // Flip both geometric and shading normal towards the "viewer"
        isect.setN(-isect.getN());
        isect.setNg(-isect.getNg());
        isect.setIsEntering(false);
    }
}

void initIntersectionPhase2(shading::Intersection &isect,
                            mcrt_common::ThreadLocalState *tls,
                            int mirrorDepth,
                            int glossyDepth,
                            int diffuseDepth,
                            bool isSubsurfaceAllowed,
                            const scene_rdl2::math::Vec2f &minRoughness,
                            const scene_rdl2::math::Vec3f &wo)
{
    EXCL_ACCUMULATOR_PROFILE(tls, EXCL_ACCUM_INIT_INTERSECTION);

    shading::Intersection::PathType type;
    if (diffuseDepth > 0) {
        type = shading::Intersection::IndirectDiffuse;
    } else if (glossyDepth > 0) {
        type = shading::Intersection::IndirectGlossy;
    } else if (mirrorDepth > 0) {
        type = shading::Intersection::IndirectMirror;
    } else {
        // TODO: Add light path case
        type = shading::Intersection::Primary;
    }

    isect.setPathType(type);

    const bool isCausticPath = diffuseDepth > 0;
    isect.setIsCausticPath(isCausticPath);

    isect.setIsSubsurfaceAllowed(isSubsurfaceAllowed);
    isect.setMinRoughness(minRoughness);
    isect.setWo(wo);
    isect.setGeomInitialized();
    isect.setUseAdaptNormal(true);
}

void initIntersectionFull(shading::Intersection &isect,
                          mcrt_common::ThreadLocalState *tls,
                          const mcrt_common::Ray         &ray,
                          const scene_rdl2::rdl2::Layer *pRdlLayer,
                          int mirrorDepth,
                          int glossyDepth,
                          int diffuseDepth,
                          bool isSubsurfaceAllowed,
                          const scene_rdl2::math::Vec2f &minRoughness,
                          const scene_rdl2::math::Vec3f &wo)
{
    initIntersectionPhase1(isect, tls, ray, pRdlLayer);
    initIntersectionPhase2(isect,
                           tls,
                           mirrorDepth,
                           glossyDepth,
                           diffuseDepth,
                           isSubsurfaceAllowed,
                           minRoughness, wo);

    MNRY_ASSERT(isect.getGeomInitialized());
}
} // end namespace geom
} // end namespace moonray

