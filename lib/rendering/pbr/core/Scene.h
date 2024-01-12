// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file Scene.h
/// $Id$
///

#pragma once

#include "Aov.h"
#include <moonray/rendering/mcrt_common/ExecutionMode.h>
#include <moonray/rendering/mcrt_common/Ray.h>
#include <moonray/rendering/pbr/light/LightAccelerator.h>
#include <moonray/rendering/pbr/light/LightSet.h>

#include <moonray/rendering/bvh/shading/Intersection.h>
#include <moonray/rendering/geom/prim/EmissiveRegion.h>

#include <scene_rdl2/common/math/Mat4.h>

#include <iostream>
#include <memory>
#include <set>
#include <vector>

namespace scene_rdl2 {
namespace rdl2 {
class Camera;
class Geometry;
class Layer;
class Light;
class Material;
class SceneContext;
class SceneObject;
}
}

namespace moonray {

namespace geom {
namespace internal {
class VolumeAssignmentTable;
}
}

namespace rt {
class GeometryManager;
}

namespace pbr {


// Forward Decl.
class Camera;
class Light;
class TLState;


//----------------------------------------------------------------------------

/// @class Scene Scene.h <pbr/Scene.h>
/// @brief The Scene class handles an abstraction of the scene:
/// - info on the camera
/// - the geometry and how to intersect it and get visibility or shadow intersections
/// - the lights in the scene (for now we only support a single layer lightmask)
/// - the camera
///
/// COORDINATE SYSTEMS IN USE:
/// --------------------------
///
/// - Render space (R) is the coordinate system in which we generate geometry,
/// build our intersection acceleration structure (BVH), trace rays, execute
/// integrators, Bsdf and Bssrdf closures, lights, shading system, shaders.
/// Basically everything that happens during rendering happens in render
/// space. It's usually a space that's more convenient or less likely to have
/// floating point precision issues than world space. Currently, render space
/// is defined to be the same as camera space when we first begin rendering
/// (at shutter-open / t=0 in case the camera transform is animated).
/// However, the camera may move between frames in an interactive application,
/// so we must divorce camera space from render space to avoid rebuilding the
/// BVH (among other things) whenever the camera moves.
///
/// - Camera space (C) is the coordinate system where the origin is at the focal
/// point of the camera and the camera looks down the -Z axis (X points to the
/// right and Y points up). This coordinate system is also unlikely to cause
/// floating point precision issues, and is arguably convenient when debugging.
/// It is used only to initialize render space when the first render starts.
///
/// - World space (W) is world-space. Not much to say about that, except we
/// don't ever use it explicitly, and especially not in single floating point
/// precision, otherwise that's when trouble happens when the scene is far from
/// the origin. We may use it in some transforms to build matrices that go from
/// one space to another. You may need to use it explicitly to convert input
/// data that is in world space, but be careful to use double precision in that
/// case (convert to render space, then down cast to float).
///
/// In this library most interfaces deal with rays and intersection
/// data-structures in render space.

class Scene
{
public:
    /// Constructor / Destructor
    Scene(const scene_rdl2::rdl2::SceneContext *rdlSceneContext,
          const scene_rdl2::rdl2::Layer *rdlLayer, int hardcodedLobeCount);
    ~Scene();

    void updateActiveCamera(const scene_rdl2::rdl2::Camera *rdlCamera);

    /// Called in pre-frame / before a shading process is done / post-frame
    /// Update light list, camera and other scene properties
    void preFrame(const LightAovs &lightAovs, mcrt_common::ExecutionMode executionMode,
            rt::GeometryManager& geometryManager, bool forceMeshLightGeneration);

    void postFrame();

    /// The intersections return hits with geometry and area lights.
    /// TODO: for now the area light's geometry is handled implicitly and not part
    /// of the geometry BVH. This will change in the future so we can handle
    /// arbitrary geometry lights.

    /// Visibility intersections returns first hit
    bool intersectRay(mcrt_common::ThreadLocalState *tls, mcrt_common::Ray &ray,
            shading::Intersection &isect, const int lobeType) const;

    /// Checks whether the given ray intersects the geometry we've marked "is_camera_medium_geometry".
    /// Used to check whether we should add the "medium_material" to the primary ray's material priority list.
    bool intersectCameraMedium(const mcrt_common::Ray &ray) const;

    /// Returns first hit but is a shadow ray, respects visible shadow setting
    /// on geometry and ignores visible in camera.  Used for presence shadows.
    bool intersectPresenceRay(mcrt_common::ThreadLocalState *tls, mcrt_common::Ray &ray,
            shading::Intersection &isect) const;

    /// Shadow intersections returns whether or not there is any hit
    bool isRayOccluded(mcrt_common::ThreadLocalState *tls, mcrt_common::Ray &ray) const;

    /// Query whether the light with index value lightIndex is active
    /// in the light set associated with layerAssignmentId
    bool isLightActive(int layerAssignmentId, int lightIndex) const
    {
        MNRY_ASSERT(layerAssignmentId != -1);
        bool* lightSetActiveList = mIdToLightSetActiveList[layerAssignmentId];
        if (lightSetActiveList) {
            return lightSetActiveList[lightIndex];
        } else {
            return false;
        }
    }

    /// Query whether scene contains volume
    bool hasVolume() const;

    // Query whether any light filter needs samples
    bool lightFilterNeedsSamples() const { return mLightFilterNeedsSamples; }

    /// Find all volume intersections along the ray
    // estimatedInScatter=true  : light transmittance computation for volume
    //                    false : all other situations
    void intersectVolumes(mcrt_common::ThreadLocalState* tls,
                          const mcrt_common::Ray &ray, int rayMask, const void* light,
                          bool estimateInScatter) const;

    // Returns a random light intersected along they ray, or nullptr
    // if the ray didn't intersect any light within the maxDistance constraint.
    const Light *intersectVisibleLight(const mcrt_common::Ray &ray, float maxDistance, IntegratorSample1D &samples,
            LightIntersection &lightIsect, int &numHits) const;

    // Returns all lights intersected along they ray. This has O(N) runtime and is only
    // used for picking.
    void pickVisibleLights(const mcrt_common::Ray &ray, float maxDistance,
            std::vector<const Light*>& lights,
            std::vector<pbr::LightIntersection>& lightIsects) const;

    // Return the list of lights assigned given the layer assignment id.
    const LightPtrList *getLightPtrList(int layerAssignmentId) const
    {
        MNRY_ASSERT(layerAssignmentId != -1);
        MNRY_ASSERT(size_t(layerAssignmentId) < mIdToLightSetMap.size());
        return mIdToLightSetMap[layerAssignmentId];
    }

    // Return the light accelerator given the layer assignment id.
    const LightAccelerator *getLightAccelerator(int layerAssignmentId) const
    {
        MNRY_ASSERT(layerAssignmentId != -1);
        MNRY_ASSERT(size_t(layerAssignmentId) < mIdToLightAcceleratorMap.size());
        return mIdToLightAcceleratorMap[layerAssignmentId];
    }

    // Return the light filter lists given the layer assignment id.
    const LightFilterLists *getLightFilterLists(int layerAssignmentId) const
    {
        MNRY_ASSERT(layerAssignmentId != -1);
        MNRY_ASSERT(size_t(layerAssignmentId) < mIdToLightFilterMap.size());
        return mIdToLightFilterMap[layerAssignmentId];
    }


    /// Debugging: print the whole context used during shading
    static std::ostream& print(std::ostream& cout,
                               const mcrt_common::RayDifferential &ray,
                               const shading::Intersection &isect,
                               int diffuseDepth,
                               int glossyDepth);

    /// Get to some globals
    finline const scene_rdl2::rdl2::SceneContext *getRdlSceneContext() const
    {
        return mRdlSceneContext;
    }

    finline const Camera *getCamera() const
    {
        return mCamera.get();
    }

    /// Get the world <--> render space transforms. These are initialized
    /// in the constructor to be camera space at initialization time, and are
    /// afterwards immutable.
    finline const scene_rdl2::math::Mat4d &getRender2World() const
    {
        return mRender2World;
    }

    finline const scene_rdl2::math::Mat4d &getWorld2Render() const
    {
        return mWorld2Render;
    }

    /// Iterate on all active lights in the scene. These are lights that belong
    /// to at least one material LightSet.
    finline int getLightCount() const
    {
        return mLightList.size();
    }

    finline const Light *getLight(int index) const
    {
        return mLightList[index];
    }

    finline Light *getLight(int index)
    {
        return mLightList[index];
    }

    finline const LightFilterList *getLightFilterList(int assignmentId, const Light* light) const
    {
        return mLightFilterListMap.at(assignmentId).at(light);
    }

    finline int getVolumeLabelId(int volumeId) const
    {
        MNRY_ASSERT(volumeId < (int)mVolumeLabelIds.size());
        return mVolumeLabelIds[volumeId];
    }

    finline const std::vector<geom::internal::EmissiveRegion>&
    getEmissiveRegions() const
    {
        return mEmissiveRegions;
    }

    /// Iterate on all active lights that are visible in camera.
    /// These are lights which should be rendered as geometry whilst rendering
    /// the scene. They are not the same thing as active lights, which can
    /// still illuminate the scene even though they aren't directly visible.
    finline int getVisibleLightCount() const
    {
        return mVisibleLightList.size();
    }

    finline const Light *getVisibleLight(int index) const
    {
        return mVisibleLightList[index];
    }

    finline const LightSet &getVisibleLightSet() const
    {
        return mVisibleLightSet;
    }

    finline const rt::EmbreeAccelerator *getEmbreeAccelerator() const
    {
        return mEmbreeAccel;
    }

    finline void setEmbreeAccelerator(const rt::EmbreeAccelerator* accel)
    {
        mEmbreeAccel = accel;
    }

    const scene_rdl2::rdl2::Layer* getLayer() const
    {
        return mRdlLayer;
    }

private:
    /// Copy is disabled
    Scene(const Scene &other);
    const Scene &operator=(const Scene &other);

    /// Rebuild LightFilters if any light filters have changed (called during preFrame())
    void updateLightFilters();
    void clearLightFilters();

    // Utility function to convert from an rdl2 Light to a moonray Light
    static Light* createLightFromRdlLight(const scene_rdl2::rdl2::Light* rdlLIght);

    /// Rebuild Light list if any Lightsets have changed (called during preFrame())
    void updateLightList();
    void clearLightList();

    // Generate the geometry for the mesh light if the mesh light needs to be updated (called during preFrame())
    void generateMeshLightGeometry(Light* light);

    /// Create or update the SDG dag and SHADER we keep per part.
    void updatePartUserData();

    // Members
    const scene_rdl2::rdl2::SceneContext *mRdlSceneContext;
    const scene_rdl2::rdl2::Layer *mRdlLayer;
    const rt::EmbreeAccelerator *mEmbreeAccel;

    // Owned members
    std::unique_ptr<Camera> mCamera;

    // A map from rdl light filters to pbr light filters
    LightFilterMap mLightFilters;

    // The union of all lights in all lightsets.
    LightPtrList mLightList;

    // The union of all light filter lists in all lightfilter sets.
    // A single light filter list is the list of light filters assigned to a light
    // for a particular layer assignment.
    LightFilterListsUniquePtrs mLightFilterLists;

    // Visible lights in the scene for this frame.
    LightPtrList mVisibleLightList;
    // Light filter lists assigned to visible lights. Each light filter should be null,
    // because lights visible in camera do not have light filters
    LightFilterLists mVisibleLightFilterLists;
    // LightSet that contains the lights visible in camera and their corresponding light filter lists
    LightSet mVisibleLightSet;
    // Map that converts the accelerator's light index with the light set's light index. For the visible
    // lights, the indices are in sync.
    std::unique_ptr<int> mVisibleLightAcceleratorIndexMap;

    // Each element of mLightSets corresponds to an rdl2 LightSet, but instead
    // of referencing rdl2 Lights, they reference the corresponding runtime light.
    // TODO: Support light sets which change contents frame to frame.
    std::vector<LightPtrList> mLightSets;

    // Each element in this vector represents LightFilterLists for 1 or more layer assignments (there can be copies
    // of the same set of light filter lists in a Layer). Each LightFilterLists contains one LightFilterList
    // per light in that layer assignment.
    // [unique_layer_assignment] x [light_index] x [light_filter_index]
    std::vector<LightFilterLists> mLightFilterTable;

    // Maps a LightFilterList to a Light and that Light / LightFilterList map to a layer assignment
    std::unordered_map<int, std::unordered_map<const Light*, const LightFilterList *>> mLightFilterListMap;

    // The number of entry is lightCount * lightSetCount
    // each entry in the mLightSetActiveList marks whether a particular light
    // is active for a particular light set
    // For example, a scene with 2 light set, 3 lights,
    // the first light set has light 0, 1 on
    // the second light set has light 2 on, then the array be
    // [true, true, false, false, false, true]
    std::unique_ptr<bool[]> mLightSetActiveList;

    // Maps an rdl2 light object to a runtime light index.
    std::map<const scene_rdl2::rdl2::Light *, int> mRdlLightToLightMap;

    // Maps a layer assignment id to a runtime light set in constant time.
    std::vector<const LightPtrList *> mIdToLightSetMap;
    std::vector<const LightAccelerator *> mIdToLightAcceleratorMap;

    // Per part light filter lists.
    // Each LightFilterLists has the LightFilterList for all the lights in the light set
    // Each LightFilterList has the LightFilters for that light.
    std::vector<const LightFilterLists*> mIdToLightFilterMap;

    // Maps a layer assignment id to a list of bool which mark the on/off status
    // of all lights in the corresponding light set
    std::vector<bool*> mIdToLightSetActiveList;

    // mapping from volume id to LPE label id for AOV state machine use
    std::vector<int> mVolumeLabelIds;
    // Primitives with emissive volume assignment, their emission will be
    // sampled explicitly like light sources
    std::vector<geom::internal::EmissiveRegion> mEmissiveRegions;

    // Hard-coded lobe count
    int mHardcodedBsdfLobeCount;

    // Transforms from world space to render space and back.
    scene_rdl2::math::Mat4d mRender2World;
    scene_rdl2::math::Mat4d mWorld2Render;

    // Cache the scene variable value on whether ray visibility mask should be propagated
    bool mPropagateVisibilityBounceType;

    // Embree-based light acceleration
    LightAccList mLightAccList;

    bool mHasVolume;

    // Whether any light filter needs samples (e.g. for blur)
    bool mLightFilterNeedsSamples;
};


//----------------------------------------------------------------------------

} // namespace pbr
} // namespace moonray


