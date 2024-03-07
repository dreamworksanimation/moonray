// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file Scene.cc
/// $Id$
///

#include "PbrTLState.h"
#include "Scene.h"
#include "Util.h"
#include <moonray/rendering/pbr/camera/BakeCamera.h>
#include <moonray/rendering/pbr/camera/Camera.h>
#include <moonray/rendering/pbr/camera/DomeMaster3DCamera.h>
#include <moonray/rendering/pbr/camera/OrthographicCamera.h>
#include <moonray/rendering/pbr/camera/PerspectiveCamera.h>
#include <moonray/rendering/pbr/camera/SphericalCamera.h>
#include <moonray/rendering/pbr/integrator/PathIntegratorUtil.h>
#include <moonray/rendering/pbr/light/CylinderLight.h>
#include <moonray/rendering/pbr/light/DiskLight.h>
#include <moonray/rendering/pbr/light/EnvLight.h>
#include <moonray/rendering/pbr/light/MeshLight.h>
#include <moonray/rendering/pbr/light/RectLight.h>
#include <moonray/rendering/pbr/light/SphereLight.h>
#include <moonray/rendering/pbr/light/SpotLight.h>
#include <moonray/rendering/pbr/light/DistantLight.h>
#include <moonray/rendering/pbr/lightfilter/ColorRampLightFilter.h>
#include <moonray/rendering/pbr/lightfilter/CombineLightFilter.h>
#include <moonray/rendering/pbr/lightfilter/CookieLightFilter.h>
#include <moonray/rendering/pbr/lightfilter/CookieLightFilter_v2.h>
#include <moonray/rendering/pbr/lightfilter/BarnDoorLightFilter.h>
#include <moonray/rendering/pbr/lightfilter/DecayLightFilter.h>
#include <moonray/rendering/pbr/lightfilter/IntensityLightFilter.h>
#include <moonray/rendering/pbr/lightfilter/RodLightFilter.h>
#include <moonray/rendering/pbr/lightfilter/VdbLightFilter.h>
#include <moonray/rendering/pbr/core/Util.h>

#include <moonray/rendering/geom/Instance.h>
#include <moonray/rendering/geom/IntersectionInit.h>
#include <moonray/rendering/geom/PrimitiveGroup.h>
#include <moonray/rendering/geom/PrimitiveVisitor.h>
#include <moonray/rendering/geom/prim/GeomTLState.h>
#include <moonray/rendering/geom/prim/PrimitivePrivateAccess.h>
#include <moonray/rendering/geom/prim/VolumeAssignmentTable.h>
#include <moonray/rendering/geom/prim/VolumeRegions.h>
#include <moonray/rendering/geom/Types.h>
#include <moonray/rendering/mcrt_common/Ray.h>
#include <moonray/rendering/rt/GeometryManager.h>
#include <moonray/rendering/shading/bsdf/Bsdf.h>
#include <moonray/rendering/shading/bssrdf/Bssrdf.h>
#include <scene_rdl2/render/logging/logging.h>
#include <scene_rdl2/scene/rdl2/VisibilityFlags.h>

#include <iomanip>

using scene_rdl2::logging::Logger;
using namespace scene_rdl2;

namespace moonray {
namespace pbr {

using mcrt_common::ThreadLocalState;

//----------------------------------------------------------------------------

namespace {
std::unique_ptr<Camera> cameraFactory(const rdl2::Camera* rdlCamera)
{
    const rdl2::SceneClass& cameraClass = rdlCamera->getSceneClass();
    const std::string& className = cameraClass.getName();
    if (className == "PerspectiveCamera") {
        return std::unique_ptr<Camera>(new PerspectiveCamera(rdlCamera));
    } else if (className == "OrthographicCamera") {
        return std::unique_ptr<Camera>(new OrthographicCamera(rdlCamera));
    } else if (className == "SphericalCamera") {
        return std::unique_ptr<Camera>(new SphericalCamera(rdlCamera));
    } else if (className == "DomeMaster3DCamera") {
        return std::unique_ptr<Camera>(new DomeMaster3DCamera(rdlCamera));
    } else if (className == "BakeCamera") {
        return std::unique_ptr<Camera>(new BakeCamera(rdlCamera));
    } else {
        MNRY_ASSERT(!"Should not get here");
        throw std::runtime_error("No valid camera type specified");
    }
}
} // namespace

Scene::Scene(
        const rdl2::SceneContext *rdlSceneContext,
        const rdl2::Layer *rdlLayer, int hardcodedLobeCount) :
    mRdlSceneContext(rdlSceneContext),
    mRdlLayer(rdlLayer),
    mEmbreeAccel(nullptr),
    mHardcodedBsdfLobeCount(hardcodedLobeCount),
    mPropagateVisibilityBounceType(false),
    mHasVolume(false),
    mLightFilterNeedsSamples(false)
{
    std::vector<const rdl2::Camera *> cameras = rdlSceneContext->getActiveCameras();
    updateActiveCamera(cameras[0]);
}

void
Scene::updateActiveCamera(const scene_rdl2::rdl2::Camera *rdlCamera)
{
    // Render space == camera space at the first specified motion step.
    mRender2World = rdlCamera->get(rdl2::Node::sNodeXformKey);
    mWorld2Render = rtInverse(mRender2World);

    std::unique_ptr<Camera> camera = cameraFactory(rdlCamera);
    camera->update(getWorld2Render());
    mCamera = std::move(camera);
}


Scene::~Scene()
{
    clearLightList();
    clearLightFilters();
}

//----------------------------------------------------------------------------
// For MeshLight, iterate through the geometries and assign them to
// the correct light

class MeshLightSetter : public geom::PrimitiveVisitor
{
public:
    MeshLightSetter(MeshLight* meshLight):
        mMeshLight(meshLight)
    {}

    virtual void visitPrimitive(geom::Primitive& p) override
    {
        geom::internal::Primitive* pImpl =
            geom::internal::PrimitivePrivateAccess::getPrimitiveImpl(&p);

        // If the primitive is a mesh, assign it to the mesh light
        if (pImpl->getType() == geom::internal::Primitive::POLYMESH) {
            mMeshLight->setMesh(pImpl);
        }
    }

    virtual void visitPrimitiveGroup(geom::PrimitiveGroup& pg) override
    {
        pg.forEachPrimitive(*this);
    }

    virtual void visitTransformedPrimitive(geom::TransformedPrimitive& t) override
    {
        t.getPrimitive()->accept(*this);
    }

    virtual void visitInstance(geom::Instance& i) override
    {
        const auto& ref = i.getReference();
        // visit the referenced Primitive if it's not visited yet
        if (mSharedPrimitives.insert(ref).second) {
            ref->getPrimitive()->accept(*this);
        }
    }

private:
    MeshLight* mMeshLight;
    geom::SharedPrimitiveSet mSharedPrimitives;
};

void setAttributeTableOnMeshLight(MeshLight* meshLight, const std::string& refGeomName,
    const rdl2::SceneContext* sceneContext)
{
    const rdl2::Material* material = sceneContext->
        getSceneObject(refGeomName + "_MeshLightMaterial")->asA<rdl2::Material>();
    const shading::AttributeTable* table = nullptr;
    if (material->hasExtension()) {
        table = material->get<shading::RootShader>().getAttributeTable();
    }
    meshLight->setAttributeTable(table);
}

void
Scene::generateMeshLightGeometry(Light* light)
{
    MeshLight* meshLight = (MeshLight*) light;
    rdl2::Geometry* refGeom = meshLight->getReferenceGeometry();

    // Check that the mesh light has a reference geometry AND
    // that geometry does not exist in the scene Layer
    if (refGeom && !mRdlLayer->contains(refGeom)) {
        // set layer for mesh light
        meshLight->setLayer(mRdlSceneContext->
            getSceneObject("MeshLightLayer")->asA<rdl2::Layer>());
        // set attribute table for mesh light
        setAttributeTableOnMeshLight(meshLight, refGeom->getName(), mRdlSceneContext);

        geom::Procedural* procedural = refGeom->getProcedural();
        if (procedural) {
            // Set geometry on mesh light.
            MeshLightSetter meshLightSetter(meshLight);
            procedural->forEachPrimitive(meshLightSetter, false /*parallel*/);
        }
    }
    // This builds the sampling BVH and does any other computation that occurs
    // after all meshes have been submitted. If no meshes were submitted
    // then this turns the light off.
    meshLight->finalize();

    // Get hold of the geometry accelerator's embree device
    const RTCDevice& rtcDevice = mEmbreeAccel->getRTCDevice();
    // This builds the embree BVH which is used for light intersection
    meshLight->setEmbreeAccelerator(rtcDevice);

    // Warn user if light has turned off during the mesh light generation process
    if (!light->isOn()) {
        meshLight->getRdlLight()->warn("MeshLight did not load. Check if reference geometry is correct.");
    }
}


//----------------------------------------------------------------------------

void
Scene::preFrame(const LightAovs &lightAovs, mcrt_common::ExecutionMode executionMode,
        rt::GeometryManager& geometryManager, bool forceMeshLightGeneration)
{
    const rdl2::SceneVariables &vars = mRdlSceneContext->getSceneVariables();

    mPropagateVisibilityBounceType = vars.get(
            rdl2::SceneVariables::sPropagateVisibilityBounceType);

    // Update camera info
    mCamera->update(getWorld2Render());

    // Update light-list
    // Updating lights needs to happen after the mCamera->update() above
    // All lights are updated as well
    // Lights needs to be updated if there is a frame change or motion blur is toggled.
    bool updateLights = mRdlLayer->lightSetsChanged() || vars.hasChanged(rdl2::SceneVariables::sFrameKey) ||
        vars.hasChanged(rdl2::SceneVariables::sEnableMotionBlur);
    if (updateLights) {
        updateLightList();
    } else {
        for (int i = 0; i < getLightCount(); ++i) {
            Light *light = getLight(i);
            if (light->isMesh()) {
                if (forceMeshLightGeneration) {
                    // If all the geometries in the scene have changed, either because of a frame change or other
                    // global setting, then the mesh light's geometry or map shader could have changed without any
                    // rdl attributes changing. We must completely update the mesh light.
                    light->update(getWorld2Render());
                    generateMeshLightGeometry(light);
                } else {
                    // The attribute table is always updated in the render context. Therefore
                    // we must always set the attribute table to the mesh light even if the
                    // light itself does not need to be updated.
                    MeshLight* meshLight = (MeshLight*)light;
                    rdl2::Geometry* refGeom = meshLight->getReferenceGeometry();
                    setAttributeTableOnMeshLight(meshLight, refGeom->getName(), mRdlSceneContext);
                }
            }
        }
    }

    // Update light filters. This must be done after we update the lights. Since light filters are mapped
    // to lights, if we do update the lights, we must also update light filters.
    if (mRdlLayer->lightFilterSetsChanged() || updateLights) {
        updateLightFilters();
    }

    mVisibleLightList.clear();
    mVisibleLightFilterLists.clear();
    // Make the visible light list.
    // Do 3 passes, adding the bounded lights (sphere, rect, disk, spot, cylinder) in the first pass
    // distant lights in the second pass, env lights in the third pass. This is done
    // (a) to separate out the unbounded lights, which cannot be put in the light acceleration structure, and
    // (b) so that all distant lights precede all env lights, since distant lights are closer than env
    // lights and can be tested for intersection first.
    for (int pass=0; pass<3; pass++) {
        for (auto iter = mLightList.begin(); iter != mLightList.end(); ++iter) {
            Light* light = *iter;

            // Add the bounded lights in pass 0, distant lights in pass 1, env lights in pass 2
            if (( (pass == 0) && light->isBounded() ) ||
                ( (pass == 1) && light->isDistant() ) ||
                ( (pass == 2) && light->isEnv()     ) ) {

                if (light->getIsVisibleInCamera()) {
                    mVisibleLightList.push_back(light);
                    // lights visible in camera do not have light filters (for now)
                    mVisibleLightFilterLists.push_back(nullptr);
                }
            }
        }
    }

    mVisibleLightSet.init(mVisibleLightList.data(), mVisibleLightList.size(), mVisibleLightFilterLists.data());

    // Setup aov labels on pbr lights
    for (int i = 0; i < getLightCount(); ++i) {
        Light *light = getLight(i);
        const rdl2::Light *rdlLight = light->getRdlLight();
        const auto &label = rdlLight->get(rdl2::Light::sLabel);
        int32_t labelId = label.empty()? -1: lightAovs.getLabelId(label);
        light->setLabelId(labelId);
    }

    // Get hold of the geometry accelerator's embree device
    const RTCDevice& rtcDevice = mEmbreeAccel->getRTCDevice();

    // Create an acceleration structure for each light set.
    // This can only be done after update() has been called for each light,
    // since that's where the lights' bounding boxes are set.

    // Get parameters for adaptive light sampling
    const float sceneDiameter = scene_rdl2::math::length(mEmbreeAccel->getBounds().size());
    const pbr::LightSamplingMode lightSamplingMode = 
                    static_cast<pbr::LightSamplingMode>(vars.get(scene_rdl2::rdl2::SceneVariables::sLightSamplingMode));
    const float lightSamplingQuality = vars.get(scene_rdl2::rdl2::SceneVariables::sLightSamplingQuality);

    // First the per-layer light sets
    LightAccelerator* acc = mLightAccList.data();
    MNRY_ASSERT(acc);
    LightPtrList* lightPtrList = mLightSets.data();
    for (unsigned int i=0; i<mLightSets.size(); i++, acc++, lightPtrList++) {
        acc->init(lightPtrList->data(), lightPtrList->size(), rtcDevice, sceneDiameter, lightSamplingQuality);
        if (lightSamplingMode == pbr::LightSamplingMode::ADAPTIVE) {
            acc->buildSamplingTree();
        }
    }

    // Finally the visible light set
    size_t visibleLightCount = mVisibleLightSet.getLightCount();
    acc->init(mVisibleLightSet.getLights(), visibleLightCount, rtcDevice, sceneDiameter, lightSamplingQuality);
    if (lightSamplingMode == pbr::LightSamplingMode::ADAPTIVE) {
        acc->buildSamplingTree();
    }
    int * visibleLightAcceleratorIndexMap = new int[visibleLightCount];
    for (size_t i = 0; i < visibleLightCount; ++i) {
        visibleLightAcceleratorIndexMap[i] = i;
    }
    mVisibleLightAcceleratorIndexMap.reset(visibleLightAcceleratorIndexMap);
    mVisibleLightSet.setAccelerator(acc, mVisibleLightAcceleratorIndexMap.get());

    // If we need any uv bake maps, now is the time.
    mCamera->bakeUvMaps();

    // Build a volumeId to labelId mapping for LPE aov support
    const geom::internal::VolumeAssignmentTable * volumeAssignmentTable =
        geometryManager.getVolumeAssignmentTable();
    int volumeCount = volumeAssignmentTable->getVolumeCount();
    mVolumeLabelIds.resize(volumeCount, -1);
    for (int volumeId = 0; volumeId < volumeCount; ++volumeId) {
        const rdl2::VolumeShader* volumeShader =
            volumeAssignmentTable->lookupWithVolumeId(volumeId);
        const auto& label = volumeShader->get(rdl2::VolumeShader::sLabel);
        int labelId = label.empty()? -1: lightAovs.getLabelId(label);
        mVolumeLabelIds[volumeId] = labelId;
    }

    // does this scene have volumes?
    mHasVolume = volumeAssignmentTable->getVolumeCount() > 0;

    // Store the list of emissive volume primitives
    geometryManager.getEmissiveRegions(mRdlLayer, mEmissiveRegions);

    // If the ShadowSets or the geometries changed, we must change the shadow linkings.
    if (mRdlLayer->shadowSetsChanged() || !mRdlLayer->getChangedOrDeformedGeometries().empty()) {
        geometryManager.updateShadowLinkings(mRdlLayer);
    }
}



void
Scene::postFrame()
{
}

void
Scene::updateLightFilters()
{
    clearLightFilters();

    // Gather unique rdl light filters
    std::unordered_set<const rdl2::LightFilter *> allEnabledRdlLightFilters;           // What we are gathering into
    uint32_t assignmentCount = mRdlLayer->getAssignmentCount();
    for (uint32_t i = 0; i < assignmentCount; ++i) {                                   // Loop over rdl layers
        const rdl2::LightSet *lightSet = mRdlLayer->lookupLightSet(i);
        if (lightSet) {
            // Gather lightfilters in all the lights.
            const rdl2::SceneObjectVector &lights = lightSet->getLights();
            for (auto it = lights.begin(); it != lights.end(); ++it) {                 // Loop over layer's lights
                const rdl2::SceneObjectVector & rdlLightFilterObjs =
                    static_cast<const rdl2::Light *>(*it)->get(rdl2::Light::sLightFiltersKey);
                for (auto it : rdlLightFilterObjs) {                                   // Loop over light's filters
                    const rdl2::LightFilter *rdlLightFilter = it->asA<rdl2::LightFilter>();
                    MNRY_ASSERT(rdlLightFilter);
                    if (rdlLightFilter->isOn()) {
                        allEnabledRdlLightFilters.insert(rdlLightFilter);              // Gather filter into set
                        rdlLightFilter->getReferencedLightFilters(allEnabledRdlLightFilters);
                    }
                }
            }
        }
    }

    // Create list of unique pbr::LightFilters
    for (const rdl2::LightFilter* rdlLightFilter : allEnabledRdlLightFilters) {        // Loop over gathered rdl filters

        const rdl2::SceneClass &lightFilterClass = rdlLightFilter->getSceneClass();
        const std::string &className = lightFilterClass.getName();

        LightFilter *lightFilter = nullptr;

        if (className == "ColorRampLightFilter") {                                     // Construct filter
            lightFilter = new ColorRampLightFilter(rdlLightFilter);
        } else if (className == "CombineLightFilter") {
            lightFilter = new CombineLightFilter(rdlLightFilter);
        } else if (className == "CookieLightFilter") {
            lightFilter = new CookieLightFilter(rdlLightFilter);
        } else if (className == "CookieLightFilter_v2") {
            lightFilter = new CookieLightFilter_v2(rdlLightFilter);
        } else if (className == "BarnDoorLightFilter") {
            lightFilter = new BarnDoorLightFilter(rdlLightFilter);
        } else if (className == "DecayLightFilter") {
            lightFilter = new DecayLightFilter(rdlLightFilter);
        } else if (className == "IntensityLightFilter") {
            lightFilter = new IntensityLightFilter(rdlLightFilter);
        } else if (className == "RodLightFilter") {
            lightFilter = new RodLightFilter(rdlLightFilter);
        } else if (className == "VdbLightFilter") {
            lightFilter = new VdbLightFilter(rdlLightFilter);
        } else {
            MNRY_ASSERT(!"Unknown light filter type.");
        }

        if (lightFilter) {
            mLightFilters[rdlLightFilter] = std::unique_ptr<LightFilter>(lightFilter); // Build map of rdl -> pbr filter
        }
    }                                                                                  // End loop over gathered filters

    // Initialize (or update) all filters
    mLightFilterNeedsSamples = false;
    for (auto& lfEntry : mLightFilters) {
        auto &pbrFilter = *lfEntry.second;
        pbrFilter.update(mLightFilters, getWorld2Render());                            // Init/update pbr filter

        // Record if any light filter needs samples
        mLightFilterNeedsSamples |= pbrFilter.needsSamples();
    }

    // There are several light filter related data structures:
    //
    //   mLightFilters       - Map from rdl filter to pbr filter
    //   mIdToLightFilterMap - Per-part LightFilterLists
    //   mLightFilterTable   - All LightFilterLists
    //   mLightFilterListMap - Maps a layer to a map of a Light to a LightFilterList

    mIdToLightFilterMap.reserve(assignmentCount);
    mLightFilterTable.resize(assignmentCount);
    // Number of unique pbr light filter sets
    int currentLightFilterSets = 0;

    // unique light filter lists
    std::vector<std::unordered_set<const rdl2::LightFilter *>> uniqueRdlLightFilterLists;

    for (uint32_t i = 0; i < assignmentCount; ++i) {                                   // Loop over layers

        // get rdl light filters for this layer assignment
        const rdl2::LightFilterSet *lightFilterSet = mRdlLayer->lookupLightFilterSet(i);

        // get pbr lightset for this assignment
        const LightPtrList* lightPtrList = mIdToLightSetMap[i];
        size_t lightCount = 0;
        if (lightPtrList) {
            lightCount = lightPtrList->size();
        }

        // create LightFilterLists for this assignment.
        // Grab the current light filter set to fill in.
        LightFilterLists& lightFilterLists = mLightFilterTable[currentLightFilterSets];
        lightFilterLists.clear();

        for (size_t j = 0; j < lightCount; ++j) {                                      // Loop over lights
            // get light
            const Light* light = lightPtrList->at(j);
            const rdl2::Light* rdlLight = light->getRdlLight();

            // get list of light filters for this light
            const rdl2::SceneObjectVector &rdlLightFilters = rdlLight->get(
                rdl2::Light::sLightFiltersKey);
            size_t lightFilterCount = rdlLightFilters.size();
            std::unordered_set<const rdl2::LightFilter *> rdlLightFilterList;
            // If there is a light filter set on this layer assignment, only include light filters in that set
            for (size_t k = 0; k < lightFilterCount; ++k) {                            // Loop over light's filters
                const rdl2::LightFilter* rdlLightFilter = rdlLightFilters[k]->asA<
                    rdl2::LightFilter>();
                MNRY_ASSERT(rdlLightFilter);
                if (!lightFilterSet || (lightFilterSet->contains(rdlLightFilter) &&    // Layer has no filter list OR
                    mLightFilters.find(rdlLightFilter) != mLightFilters.end())) {      // pbr filter is in set & exists
                    if (rdlLightFilter->isOn()) {
                        rdlLightFilterList.insert(rdlLightFilter);                     // Place rdl light filter in set
                    }
                }
            }

            // Check if filter list already exists
            int existingIndex = -1;
            for (int k = 0; k < uniqueRdlLightFilterLists.size(); ++k) {               // Loop over unique filter sets
                if (uniqueRdlLightFilterLists[k] == rdlLightFilterList) {
                    existingIndex = k;                                                 // Note matching set
                    break;
                }
            }

            // a null light filter list represents an empty light filter list
            LightFilterList* lightFilterList = nullptr;
            if (existingIndex < 0) {                                                   // Filter list (set) is new
                uniqueRdlLightFilterLists.push_back(rdlLightFilterList);
                // create light filter list from rdl list
                size_t activeLightFilterCount = rdlLightFilterList.size();
                if (activeLightFilterCount > 0) {                                      // List is not empty
                    std::unique_ptr<const LightFilter* []> lightFilters(               // Create pbr list from rdl list
                        new const LightFilter*[activeLightFilterCount]);
                    size_t k = 0;
                    for (const rdl2::LightFilter* rdlLightFilter : rdlLightFilterList) {   // Loop over filters in set
                        lightFilters[k++] = mLightFilters[rdlLightFilter].get();       // Save pointers to pbr filters
                    }
                    lightFilterList = new LightFilterList();
                    lightFilterList->init(std::move(lightFilters), activeLightFilterCount);     // Init pbr filter list
                }

                // add this light filter list to the list of unique pbr light filter lists
                mLightFilterLists.push_back(std::unique_ptr<LightFilterList>(lightFilterList)); // Save pbr filter list
            } else {
                // grab existing light filter list
                lightFilterList = mLightFilterLists[existingIndex].get();
            }

            // Add this light filter list to the light filter list map, which is used for linking to volumes.
            // Grab list of lists for this layer assignment
            std::unordered_map<const Light*, const LightFilterList *>& lightToLflMap = mLightFilterListMap[i];
            // set list for this light
            lightToLflMap[light] = lightFilterList;

            // add light filter list (for this light) to set of all light filter lists for this layer assignment
            lightFilterLists.push_back(lightFilterList);
        }                                                                              // End loop over lights

        // add LightFilterLists for this layer assignment
        int existingIndex = -1;
        // loop through all existing light filter sets except this one
        // The == operator compares each element of the std::vector in order
        // The lightFilterLists contains raw pointers to light filter list, a pointer can be null
        // There is a light filter list per light in this layer assignment
        for (int j = 0; j < currentLightFilterSets; ++j) {                     // Loop over filter lists
            if (mLightFilterTable[j] == lightFilterLists) {                    // Does filter list already exist?
                existingIndex = j;
                break;
            }
        }

        // Get light filter set for this layer assignment
        const LightFilterLists* lfl = nullptr;
        if (existingIndex < 0) {
            // if we have not generated this light filter set before, it is unique and we can add it to
            // mLightFilterTable
            lfl = &mLightFilterTable[currentLightFilterSets++];                // If list is new, add to table
        } else {
            // If we have generated it before, grab the original instance of it.
            lfl = &mLightFilterTable[existingIndex];
        }

        mIdToLightFilterMap.push_back(lfl);                                    // Add list to map
    }

    // resize mLightFilterTable to the number of unique light filter sets.
    mLightFilterTable.resize(currentLightFilterSets);
}

void
Scene::clearLightFilters()
{
    mLightFilters.clear();
    mLightFilterLists.clear();
    mVisibleLightFilterLists.clear();
    mLightFilterTable.clear();
    mLightFilterListMap.clear();
    mIdToLightFilterMap.clear();
}

Light*
Scene::createLightFromRdlLight(const rdl2::Light* rdlLight)
{
    const rdl2::SceneClass &lightClass = rdlLight->getSceneClass();
    const std::string &className = lightClass.getName();
    Light *light = nullptr;
    if (className == "CylinderLight") {
        light = new CylinderLight(rdlLight);
    } else if (className == "DiskLight") {
        light = new DiskLight(rdlLight);
    } else if (className == "EnvLight") {
        light = new EnvLight(rdlLight);
    } else if (className == "RectLight") {
        light = new RectLight(rdlLight);
    } else if (className == "SphereLight") {
        light = new SphereLight(rdlLight);
    } else if (className == "SpotLight") {
        light = new SpotLight(rdlLight);
    } else if (className == "DistantLight") {
        light = new DistantLight(rdlLight);
    } else if (className == "MeshLight") {
        light = new MeshLight(rdlLight);
    } else {
        MNRY_ASSERT(!"Unknown light type.");
    }
    return light;
}

//----------------------------------------------------------------------------
// Scene::updateLightList()
// The current implementation is a heavy hammer that dumps all pbr lights,
// and fully re-creates them. This causes light textures to be re-loaded and
// their ImageDistribution to be re-computed which is not cheap.
// We are only actually doing this work when a LightSet changes or a Layer
// LightSet assignment changes.
// TODO: optimize to handle incremental updates instead

void
Scene::updateLightList()
{
    clearLightList();

    // Gather unique light sets in layer, and unique lights within those light
    // sets.
    std::set<const rdl2::LightSet *> rdlLightSets;
    std::set<const rdl2::Light *> rdlLights;
    uint32_t assignmentCount = mRdlLayer->getAssignmentCount();
    for (uint32_t i = 0; i < assignmentCount; ++i) {
        const rdl2::LightSet *lightSet = mRdlLayer->lookupLightSet(i);
        if (lightSet) {
            if (rdlLightSets.insert(lightSet).second) {
                // Gather lights in this lightset.
                const rdl2::SceneObjectVector &lights = lightSet->getLights();
                for (auto it = lights.begin(); it != lights.end(); ++it) {
                    rdlLights.insert(static_cast<const rdl2::Light *>(*it));
                }
            }
        }
    }

    // Allocate all runtime light sets and create mapping from corresponding
    // rdl2 light sets.
    size_t lightSetCount = rdlLightSets.size();
    mLightSets.resize(lightSetCount);
    std::map<const rdl2::LightSet *, int> lightSetMap;
    int lightSetIndex = 0;
    for (auto it = rdlLightSets.begin(); it != rdlLightSets.end(); ++it) {
        lightSetMap[*it] = lightSetIndex++;
    }

    // Create all runtime lights and create mapping from corresponding rdl2
    // lights.
    size_t lightCount = rdlLights.size();
    mLightList.reserve(lightCount);

    mLightSetActiveList.reset(new bool[lightCount * lightSetCount]);
    memset(mLightSetActiveList.get(), 0, sizeof(bool) * lightCount * lightSetCount);

    for (const rdl2::Light* rdlLight : rdlLights) {
        std::unique_ptr<Light> light(createLightFromRdlLight(rdlLight));
        if (light) {
            // Update the light (transformation, attributes, texture)
            // update() can return false for a number of reasons, including the
            // rdlLight being switched off, and the light's isBlack(radiance)
            // evaluating to true. We should not include these lights in the
            // scene's light list.

            if (light->update(getWorld2Render())) {

                // Generate the mesh for the mesh light
                const std::string &className = rdlLight->getSceneClass().getName();
                if (className == "MeshLight") {
                    generateMeshLightGeometry(light.get());
                }
                if (light->isOn()) {
                    std::hash<std::string> name_hash;
                    light->setHash(name_hash(rdlLight->getName()));
                    mLightList.push_back(light.release());
                    mRdlLightToLightMap[rdlLight] = mLightList.size() - 1;
                }
            }
        }
    }

    // Hookup newly created lights to the relevant light sets.
    for (const rdl2::LightSet* const rdlLightSet : rdlLightSets) {
        // null light sets have been culled out above by this stage
        MNRY_ASSERT(rdlLightSet);

        const auto lightSetIdx = lightSetMap.find(rdlLightSet);
        MNRY_ASSERT(lightSetIdx != lightSetMap.end());

        LightPtrList &lightPtrList = mLightSets[lightSetIdx->second];
        const rdl2::SceneObjectVector& lights = rdlLightSet->getLights();
        lightPtrList.reserve(lights.size());

        // Make 3 passes, adding the bounded lights (sphere, rect, disk, spot, cylinder) in the first pass
        // distant lights in the second pass, env lights in the third pass. This is done in order to
        // (a) separate out the unbounded lights, which cannot be put in the light acceleration structure, and
        // (b) so that all distant lights precede all env lights, since distant lights are closer than env
        // lights and can be tested for intersection first.
        for (int pass=0; pass<3; pass++) {
            for (const auto& lightobj : lights) {
                const rdl2::Light* const rdlLight = static_cast<const rdl2::Light*>(lightobj);
                const auto lightIdx = mRdlLightToLightMap.find(rdlLight);
                if (lightIdx != mRdlLightToLightMap.end()) {
                    Light* light = mLightList[lightIdx->second];

                    // Add the bounded lights in pass 0, distant lights in pass 1, env lights in pass 2
                    if (( (pass == 0) && light->isBounded() ) ||
                        ( (pass == 1) && light->isDistant() ) ||
                        ( (pass == 2) && light->isEnv()     ) ) {

                        lightPtrList.push_back(light);
                        size_t index = lightCount * lightSetIdx->second + lightIdx->second;
                        mLightSetActiveList[index] = true;
                    }
                }
            }
        }
    }

    // Initialize the light accelerators (there will always be lightSetCount + 1 members,
    // where the extra 1 is the visible light set)
    mLightAccList.resize(lightSetCount + 1);

    // Create mapping from rdl2 assignment ids to runtime light sets and light accelerators
    mIdToLightSetMap.reserve(assignmentCount);
    mIdToLightAcceleratorMap.reserve(assignmentCount);
    mIdToLightSetActiveList.reserve(assignmentCount);
    for (uint32_t i = 0; i < assignmentCount; ++i) {
        const rdl2::LightSet *rdlLightSet = mRdlLayer->lookupLightSet(i);
        const LightPtrList *lightPtrList = nullptr;
        const LightAccelerator *lightAccelerator = nullptr;

        bool* lightSetActiveList = nullptr;
        int remappedIdx = 0;
        if (rdlLightSet) {
            auto lightSetIdx = lightSetMap.find(rdlLightSet);
            MNRY_ASSERT(lightSetIdx != lightSetMap.end());
            remappedIdx = lightSetIdx->second;
            lightPtrList = &mLightSets[remappedIdx];
            lightAccelerator = &mLightAccList[remappedIdx];
            lightSetActiveList = mLightSetActiveList.get() +
                remappedIdx * lightCount;
        }

        // It's ok to add null pointers to this list.
        mIdToLightSetMap.push_back(lightPtrList);
        mIdToLightAcceleratorMap.push_back(lightAccelerator);
        mIdToLightSetActiveList.push_back(lightSetActiveList);
    }
}


void
Scene::clearLightList()
{
    int count = mLightList.size();
    for (int i=0; i < count; i++) {
        delete mLightList[i];
    }
    mLightList.clear();
    mLightSets.clear();
    mRdlLightToLightMap.clear();
    mIdToLightSetMap.clear();
    mIdToLightAcceleratorMap.clear();
    mIdToLightSetActiveList.clear();

    // Clear light acceleration data
    mVisibleLightSet.setAccelerator(nullptr, nullptr);
    mLightAccList.clear();

}

//----------------------------------------------------------------------------

bool
Scene::intersectRay(mcrt_common::ThreadLocalState *tls, mcrt_common::Ray &ray, shading::Intersection &isect,
        const int lobeType) const
{
    if (ray.getStart() >= ray.getEnd()) {
        return false;
    }

    pbr::TLState *pbrTls = tls->mPbrTls.get();

    pbrTls->mStatistics.incCounter(STATS_INTERSECTION_RAYS);

    // Set the proper ray mask. If lobeType is 0, it's a primary ray from camera.
    if (lobeType == 0) {
        ray.mask = rdl2::CAMERA;
    } else {
        // Otherwise, we figure out the ray mask based on the lobe type.
        // for ray mask propagation, we simply merge with existing ray mask.
        // we should exclude the camera ray mask when propagating.
        int lobeMask = lobeTypeToRayMask(lobeType);
        ray.mask = mPropagateVisibilityBounceType
                ?  ((ray.mask | lobeMask) & ~rdl2::CAMERA)
                : lobeMask;
    }

    // Intersect with manifold geometry.
    {
        EXCL_ACCUMULATOR_PROFILE(pbrTls, EXCL_ACCUM_EMBREE_INTERSECTION);
        mEmbreeAccel->intersect(ray);
    }

    if (ray.geomID != -1) {
        geom::initIntersectionPhase1(isect, tls, ray, mRdlLayer);
        return true;
    } else {
        return false;
    }
}

bool
Scene::intersectPresenceRay(mcrt_common::ThreadLocalState *tls,
                            mcrt_common::Ray &ray,
                            shading::Intersection &isect) const
{
    if (ray.getStart() >= ray.getEnd()) {
        return false;
    }

    pbr::TLState *pbrTls = tls->mPbrTls.get();

    pbrTls->mStatistics.incCounter(STATS_PRESENCE_SHADOW_RAYS);

    // Set the proper ray mask
    ray.mask = rdl2::SHADOW;

    // Intersect with manifold geometry.
    {
        EXCL_ACCUMULATOR_PROFILE(pbrTls, EXCL_ACCUM_EMBREE_PRESENCE);
        mEmbreeAccel->intersect(ray);
    }

    if (ray.geomID != -1) {
        geom::initIntersectionPhase1(isect, tls, ray, mRdlLayer);
        return true;
    } else {
        return false;
    }
}

bool Scene::intersectCameraMedium(const mcrt_common::Ray &inRay) const 
{
    // we only need the ray origin and direction -- don't alter original ray
    mcrt_common::Ray ray(inRay.org, inRay.dir);
    // set the ray mask to look only for the camera medium
    ray.mask = rdl2::CONTAINS_CAMERA;
    mEmbreeAccel->intersect(ray);
    if (ray.geomID != -1) {
        return true;
    }
    return false;
}

bool
Scene::isRayOccluded(mcrt_common::ThreadLocalState *tls, mcrt_common::Ray &ray) const
{
    if (ray.getStart() >= ray.getEnd()) {
        return false;
    }

    pbr::TLState *pbrTls = tls->mPbrTls.get();

    pbrTls->mStatistics.incCounter(STATS_OCCLUSION_RAYS);

    ray.mask = rdl2::SHADOW;

    // Test for occlusion.
    EXCL_ACCUMULATOR_PROFILE(pbrTls, EXCL_ACCUM_EMBREE_OCCLUSION);
    return mEmbreeAccel->occluded(ray);
}

bool
Scene::hasVolume() const
{
    return mHasVolume;
}

void
Scene::intersectVolumes(mcrt_common::ThreadLocalState* tls, const mcrt_common::Ray& ray,
                        int rayMask, const void* light, bool estimateInScatter) const
{
    pbr::TLState *pbrTls = tls->mPbrTls.get();

    // need to reset tfar to inf so this probe ray can figure out whether
    // we are inside a volume enclosed by manifold primitive
    mcrt_common::Ray volumeRay(ray.org, ray.dir, ray.tnear, math::sMaxValue, ray.time);

    pbrTls->mStatistics.incCounter(STATS_VOLUME_RAYS);
    if (rayMask != rdl2::CAMERA &&
        rayMask != rdl2::SHADOW &&
        mPropagateVisibilityBounceType) {
        rayMask = ((ray.mask | rayMask) & ~rdl2::CAMERA);
    }
    // Shift bits to "see" volume objects
    volumeRay.mask = (rayMask << rdl2::sNumVisibilityTypes);
    // Feed in tls for BVH to fill out all volume intersection along the ray
    auto geomTls = tls->mGeomTls.get();
    // volume intersections out of tfar range won't be added into
    // collected intervals during the following intersection test
    geomTls->resetVolumeRayState(ray.tfar, estimateInScatter);
    volumeRay.ext.geomTls = (void*)geomTls;
    volumeRay.ext.instance0OrLight = light; // for ShadowSet evaluation
    // Intersect with volume geometry.
    {
        EXCL_ACCUMULATOR_PROFILE(pbrTls, EXCL_ACCUM_EMBREE_VOLUME);
        mEmbreeAccel->intersect(volumeRay);
    }
}

const Light *
Scene::intersectVisibleLight(const mcrt_common::Ray &ray, float maxDistance,
        IntegratorSample1D &samples, LightIntersection &lightIsect, int &numHits) const
{
    int lightIdx = mVisibleLightSet.intersect(ray.getOrigin(), nullptr, ray.getDirection(), ray.getTime(),
        maxDistance, false, samples, ray.getDepth(), rdl2::ALL_VISIBLE, lightIsect, numHits);
    if (lightIdx == -1) {
        return nullptr;
    }
    return mVisibleLightSet.getLight(lightIdx);
}

void
Scene::pickVisibleLights(const mcrt_common::Ray &ray, float maxDistance, std::vector<const Light*>& lights,
        std::vector<pbr::LightIntersection>& lightIsects) const
{
    int lightCount = mVisibleLightSet.getLightCount();
    for (int l = 0; l < lightCount; l++) {

        LightIntersection currentIsect;
        const Light* light = mVisibleLightSet.getLight(l);

        if (!(rdl2::ALL_VISIBLE & light->getVisibilityMask())) {
            // skip light if it is masked
            continue;
        }

        // test light for intersection
        if (light->intersect(ray.getOrigin(), nullptr, ray.getDirection(), ray.getTime(), maxDistance, currentIsect)) {
            // add to list if intersected
            lights.push_back(light);
            lightIsects.push_back(currentIsect);
        }
    }
}

std::ostream& Scene::print(std::ostream& cout,
                           const mcrt_common::RayDifferential &ray,
                           const shading::Intersection &isect,
                           int diffuseDepth,
                           int glossyDepth)
{
    return cout << "Ray: " << ray << '\n'
                << "Intersection: " << isect << '\n'
                << "Diffuse depth: " << diffuseDepth << '\n'
                << "Glossy depth: " << glossyDepth << '\n';

}



//----------------------------------------------------------------------------

} // namespace pbr
} // namespace moonray

// Functions exposed to ISPC:
extern "C"
{

bool CPP_LightFilterNeedsSamples(const moonray::pbr::Scene *scene)
{
    return MNRY_VERIFY(scene)->lightFilterNeedsSamples();
}

}


