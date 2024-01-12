// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#include <moonray/rendering/pbr/core/Scene.h>

#include "Picking.h"
#include <moonray/rendering/pbr/camera/Camera.h>
#include <moonray/rendering/pbr/core/Constants.h>
#include <moonray/rendering/pbr/light/Light.h>

#include <moonray/rendering/geom/IntersectionInit.h>
#include <moonray/rendering/mcrt_common/ThreadLocalState.h>
#include <moonray/rendering/bvh/shading/Intersection.h>
#include <moonray/rendering/bvh/shading/ShadingTLState.h>
#include <moonray/rendering/shading/bsdf/BsdfSlice.h>
#include <moonray/rendering/shading/BsdfBuilder.h>
#include <moonray/rendering/shading/EvalShader.h>
#include <moonray/rendering/shading/Shading.h>

// using namespace scene_rdl2::math; // can't use this as it breaks openvdb in clang.

namespace moonray {
namespace pbr {


float
computeOpenGLDepth(mcrt_common::ThreadLocalState *tls, const Scene* scene,
                   int pixelX, int pixelY)
{
    // Initialize intersection
    shading::Intersection isect;

    // Initialize ray
    mcrt_common::RayDifferential ray;
    scene->getCamera()->createRay(&ray, pixelX + 0.5f, pixelY + 0.5f,
                          0.5f, 0.5f, 0.5f, false);

    // Initialize depth to the far plane of the camera
    float depth = scene->getCamera()->getRdlCamera()->get(scene_rdl2::rdl2::Camera::sFarKey);
    int lobeType = 0;
    // Intersect the ray with the scene. This does the heavy lifting.
    bool hitGeom = scene->intersectRay(tls, ray, isect, lobeType);
    // Transform intersection point into camera space
    scene_rdl2::math::Vec3f p = transformPoint(scene->getCamera()->getRender2Camera(), isect.getP());

    // Only executed for primary rays
    if (hitGeom) {
        depth = std::max(0.f, -p.z);
    }

    return depth;
}

const scene_rdl2::rdl2::Material*
computeMaterial(mcrt_common::ThreadLocalState *tls, const Scene* scene,
                const int x, const int y)
{
    shading::Intersection isect;

    mcrt_common::RayDifferential ray;
    scene->getCamera()->createRay(&ray, x + 0.5f, y + 0.5f,
                        0.5f, 0.5f, 0.5f, false);

    int lobeType = 0;
    bool hitGeom = scene->intersectRay(tls, ray, isect, lobeType);
    if (hitGeom) {
        return isect.getMaterial()->asA<scene_rdl2::rdl2::Material>();
    }

    return NULL;
}

void
computePrimitive(mcrt_common::ThreadLocalState *tls, const Scene* scene,
                                     const int x, const int y,
                                     int& assignmentId)
{
    // Initialize intersection
    shading::Intersection isect;

    mcrt_common::RayDifferential ray;
    scene->getCamera()->createRay(&ray, x + 0.5f, y + 0.5f,
                        0.5f, 0.5f, 0.5f, false);

    int lobeType = 0;
    bool hitGeom = scene->intersectRay(tls, ray, isect, lobeType);
    if (hitGeom) {
        assignmentId = isect.getLayerAssignmentId();
    }
}

void
computeLightContributions(mcrt_common::ThreadLocalState *tls, const Scene* scene,
                          const int x, const int y,
                          shading::LightContribArray& lightContributions,
                          int numSamples, float textureFilterScale)
{
    // Initialize intersection
    shading::Intersection isect;

    mcrt_common::RayDifferential ray;
    scene->getCamera()->createRay(&ray, x + 0.5f, y + 0.5f,
                        0.5f, 0.5f, 0.5f, false);

    // Test whether we hit geometry
    int lobeType = 0;
    bool hitGeom = scene->intersectRay(tls, ray, isect, lobeType);

    // Test whether we hit visible lights in camera, if closer than geometry
    std::vector<const Light*> hitLights;
    std::vector<pbr::LightIntersection> hitLightIsects;
    scene->pickVisibleLights(ray, hitGeom ? ray.getEnd() : pbr::sInfiniteLightDistance, hitLights,
                             hitLightIsects);

    // Evaluate the radiance of all visible lights in camera
    for (size_t i = 0; i < hitLights.size(); ++i) {
        const Light *hitLight = hitLights[i];
        
        LightFilterRandomValues lightFilterR = {
            scene_rdl2::math::Vec2f(0.f, 0.f), 
            scene_rdl2::math::Vec3f(0.f, 0.f, 0.f)}; // light filters don't apply to camera rays        
        scene_rdl2::math::Color contribution = hitLight->eval(tls, ray.getDirection(), ray.getOrigin(), 
            lightFilterR, ray.getTime(), hitLightIsects[i], true, nullptr, ray.getDirFootprint());
        shading::LightContrib lightContrib(hitLight->getRdlLight(), scene_rdl2::math::luminance(contribution));
        lightContributions.push_back(lightContrib);
    }
    if (!hitLights.empty() || !hitGeom) {
        return;
    }

    // Early return if isect doesn't provide all the
    // required attributes shader request
    if (!isect.hasAllRequiredAttributes()) {
        return;
    }

    // At this point we have hit a geometry that is not blocked by any visible lights.
    // Now we get the light contribution on that geometry.

    geom::initIntersectionPhase2(isect, tls, 0, 0, 0, true, scene_rdl2::math::Vec2f(0.0f), -ray.getDirection());

    float rayEpsilon = isect.getEpsilonHint();
    if (rayEpsilon <= 0.0f) {
        // Compute automatic ray-tracing bias
        rayEpsilon = sHitEpsilonStart * scene_rdl2::math::max(ray.getEnd(), 1.0f);
    }

    // Transfer the ray to its intersection before we run shaders. This is
    // needed for texture filtering based on ray differentials.
    // Also scale the final differentials by a user factor. This is left until
    // the very end and not baked into the ray differentials since the factor
    // will typically be > 1, and would cause the ray differentials to be larger
    // than necessary.
    isect.transferAndComputeDerivatives(tls, &ray, textureFilterScale);

    isect.setMediumIor(1.f);

    const scene_rdl2::rdl2::Material* material = isect.getMaterial()->asA<scene_rdl2::rdl2::Material>();
    MNRY_ASSERT(material != NULL);
    shading::Bsdf bsdf;
    shading::State state(&isect);
    auto shadingTls = tls->mShadingTls.get();
    shading::BsdfBuilder builder(bsdf, shadingTls, state);
    shading::shade(material, shadingTls, state, builder);

    // For bssrdf or bsdfs which contain both reflection and transmission lobes
    // or is spherical, a single normal can't be used for culling so skip normal
    // culling.
    scene_rdl2::math::Vec3f normal(scene_rdl2::math::zero);
    scene_rdl2::math::Vec3f *normalPtr = nullptr;
    if (!bsdf.hasSubsurface()  &&  !bsdf.getIsSpherical() &&
        ((bsdf.getType() & shading::BsdfLobe::ALL_SURFACE_SIDES) != shading::BsdfLobe::ALL_SURFACE_SIDES)) {
        normal = (bsdf.getType() & shading::BsdfLobe::REFLECTION) ? isect.getNg() : -isect.getNg();
        normalPtr = &normal;
    }

    // Get the lights contributing to this pixel
    LightSet lightSet;
    bool hasRayTerminatorLights;
    computeActiveLights(&tls->mArena, scene, isect, normalPtr, bsdf, /* rayTime = */ 0.f,
        lightSet, hasRayTerminatorLights);

    // Populate the array with the light and contribution data to be returned.
    int count = lightSet.getLightCount();

    shading::BsdfSlice slice(isect.getNg(), -ray.getDirection(), true,
            isect.isEntering(), ispc::SHADOW_TERMINATOR_FIX_OFF, shading::BsdfLobe::ALL);
    for (int i = 0; i < count; ++i) {
        const Light* light = lightSet.getLight(i);
        const LightFilterList* lightFilterList = lightSet.getLightFilterList(i);
        scene_rdl2::math::Color contribution(scene_rdl2::math::sBlack);
        for (int j = 0; j < numSamples; ++j) {

            scene_rdl2::math::Vec3f r;
            r[0] = float(rand())/float(RAND_MAX);
            r[1] = float(rand())/float(RAND_MAX);
            r[2] = float(rand())/float(RAND_MAX);

            scene_rdl2::math::Vec3f wi;
            LightIntersection sampleIsect;
            if (!light->sample(isect.getP(), normalPtr, ray.getTime(), r, wi, sampleIsect, ray.getDirFootprint())) {
                continue;
            }

            // Is this light sample blocked by geometry ?
            mcrt_common::Ray shadowRay(isect.getP(), wi, rayEpsilon,
                    sampleIsect.distance * sHitEpsilonEnd,
                    ray.getTime(), ray.getDepth() + 1);
            if (scene->isRayOccluded(tls, shadowRay)) {
                continue;
            }

            // Evaluate and sum up contribution
            float pdfLight;
            LightFilterRandomValues lightFilterR = {
                scene_rdl2::math::Vec2f(0.f, 0.f), 
                scene_rdl2::math::Vec3f(0.f, 0.f, 0.f)}; // light filters don't apply to camera rays
            scene_rdl2::math::Color Li = light->eval(tls, wi, isect.getP(), lightFilterR, ray.getTime(), sampleIsect,
                false, lightFilterList, ray.getDirFootprint(), &pdfLight);
            if (pdfLight == 0.f) {
                continue;
            }
            float pdfBsdf;
            scene_rdl2::math::Color f = bsdf.eval(slice, wi, pdfBsdf);

            scene_rdl2::math::Color color = f * Li / pdfLight;
            contribution += color;
        }

        // Average the colors and keep track of light's contribution luminance
        contribution /= numSamples;
        shading::LightContrib lightContrib(light->getRdlLight(), luminance(contribution));
        lightContributions.push_back(lightContrib);
    }
}

} // namespace pbr
} // namespace moonray

