// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#include <scene_rdl2/render/util/AtomicFloat.h> // Needs to be included before any OpenImageIO file
#include <moonray/rendering/pbr/core/Scene.h>

#include "RenderContext.h"

#include "RenderOutputHelper.h"
#include "AttributeOverrides.h"
#include "CheckpointSigIntHandler.h"
#include "FrameState.h"
#include "ImageWriteDriver.h"
#include "PixelBufferUtils.h"
#include "ProcKeeper.h"
#include "RenderContextConsoleDriver.h"
#include "RenderDriver.h"
#include "RenderOptions.h"
#include "RenderOutputDriver.h"
#include "RenderOutputHelper.h"
#include "RenderStatistics.h"
#include "ResumeHistoryMetaData.h"
#include "TileScheduler.h"
#include "Types.h"

#include <moonray/common/time/Timer.h>
#include <moonray/common/geometry/GeometryObjects.h>
#include <moonray/rendering/geom/MotionBlurParams.h>
#include <moonray/rendering/geom/Procedural.h>
#include <moonray/rendering/geom/ProceduralLeaf.h>
#include <moonray/rendering/geom/prim/GeomTLState.h>
#include <moonray/rendering/bvh/shading/AttributeKey.h>
#include <moonray/rendering/bvh/shading/Intersection.h>
#include <moonray/rendering/bvh/shading/ThreadLocalObjectState.h>
#include <moonray/rendering/mcrt_common/ThreadLocalState.h>
#include <moonray/rendering/pbr/camera/Camera.h>
#include <moonray/rendering/pbr/core/Aov.h>
#include <moonray/rendering/pbr/core/DebugRay.h>
#include <moonray/rendering/pbr/core/Statistics.h>
#include <moonray/rendering/pbr/handlers/ShadeBundleHandler.h>
#include <moonray/rendering/pbr/integrator/PathIntegrator.h>
#include <moonray/rendering/pbr/integrator/Picking.h>
#include <moonray/rendering/pbr/sampler/SamplingAlgorithms.h>
#include <moonray/rendering/rt/GeometryManager.h>
#include <moonray/rendering/rt/rt.h>
#include <moonray/rendering/shading/BsdfBuilder.h>
#include <moonray/rendering/shading/Geometry.h>
#include <moonray/rendering/shading/Material.h>


#include <scene_rdl2/common/except/exceptions.h>
#include <scene_rdl2/scene/rdl2/rdl2.h>
#include <scene_rdl2/scene/rdl2/RenderOutput.h>
#include <scene_rdl2/scene/rdl2/ValueContainerEnq.h>
#include <scene_rdl2/render/util/Files.h>
#include <scene_rdl2/render/logging/logging.h>
#include <scene_rdl2/render/util/Strings.h>

#include <openvdb/openvdb.h>

#include <tbb/tick_count.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_for_each.h>
#include <tbb/task_arena.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <malloc.h>
#include <memory>
#include <string>
#include <vector>

namespace moonray {
namespace rndr {

using namespace scene_rdl2::math;
using namespace moonray::mcrt_common;
using scene_rdl2::logging::Logger;

using RenderTimer = time::RAIITimerAverageDouble;
using ManualRenderTimer = time::TimerAverageDouble;

namespace {

const std::string FILE_TOKEN = "_FILE_";

static unsigned
getInitialSeed(std::uint32_t frame,
               StereoView stereoView,
               bool lockFrameNoise)
{
    if (lockFrameNoise) {
        return 0u;
    }

    auto stereoSeed = static_cast<uint32_t>(stereoView);
    // We seed the sequenceID based on the frame number and stereo view, so that
    // we propagate different random seeds throughout the render. We can't use
    // the frame number directly, because this value is incremented throughout
    // the pipeline. This means that frame zero would use sequence [0, 1, 2, ...
    // ], and frame 1 would use [1, 2, 3, ... ]. We would still be correlated,
    // just maybe off a bounce.
    //
    // We multiply by a relatively large number that is co-prime to the number
    // of total possibilities (2^32). It's okay if we overflow: being co-prime
    // to the implicit modulus value means that we would eventually hit every
    // value we can store in the type without repeats (until we've hit every
    // value).

    // If we're running at 30 fps, and we don't want to be near previous seeds
    // for two minutes, that's 120 * 30 frames. So, we need a number near
    // 2^32/(120 * 30) =~ 1,193,046. It doesn't matter much, though, since this
    // value ends up getting hashed anyway.
    constexpr std::uint32_t prime = 1193047u;

    // We have to add at least one (the stereoSeed ultimately comes from an enum
    // with 0 as the start), but I want to make this general enough that we can
    // add further values in the future (e.g. a seed value). They should each
    // add enough that they don't interfere with one another.
    stereoSeed += 6037u;

    // Production shots are not usually frame 0, but some of our test shots are.
    frame += 8429u;

    // It's not so important that these magic values be prime. What is important
    // is that this ultimate value is relatively prime to the number of possible
    // values, which should be a power of two. Being a power of two, the prime
    // decomposition is 2^n, so any odd number is relatively prime to the number
    // of possible values.
    const uint32_t multValue = 2u * (prime + stereoSeed) + 1u;
    return frame * multValue;
}


//----------------------------------------------------------------------------

std::unique_ptr<pbr::PixelFilter>
getPixelFilter(scene_rdl2::rdl2::PixelFilterType type, float width)
{
    std::unique_ptr<pbr::PixelFilter> pixelFilter;

    switch (type) {
    case scene_rdl2::rdl2::PixelFilterType::box:
        pixelFilter.reset(new pbr::BoxPixelFilter(width));
        break;
    case scene_rdl2::rdl2::PixelFilterType::cubicBSpline:
        pixelFilter.reset(new pbr::CubicBSplinePixelFilter(width));
        break;
    case scene_rdl2::rdl2::PixelFilterType::quadraticBSpline:
        pixelFilter.reset(new pbr::QuadraticBSplinePixelFilter(width));
        break;
    default:
        Logger::warn("Unknown pixel filter type specified");
        pixelFilter.reset(new pbr::BoxPixelFilter(width));
        break;
    }

    return pixelFilter;
}

}   // End of anon namespace.

//////// Init and Cleanup ///////
void
initGlobalDriver(const RenderOptions &initOptions)
{
    // Setup ProcKeeper
    ProcKeeper::init();

    // Ref counted block pool. Should get picked up deep inside the
    // initRenderDriver call before this method returns.
    scene_rdl2::util::Ref<scene_rdl2::alloc::ArenaBlockPool> arenaBlockPool =
            scene_rdl2::util::alignedMallocCtorArgs<scene_rdl2::alloc::ArenaBlockPool>(CACHE_LINE_SIZE);
    // Create init params and give it a raw ref to the block pool
    mcrt_common::TLSInitParams initParams;
    initParams.mArenaBlockPool = arenaBlockPool.get();

    // have RenderOptions set up the rest of the params
    bool realtimeRender = (initOptions.getRenderMode() == rndr::RenderMode::REALTIME)? true: false;
    initOptions.setupTLSInitParams(&initParams, realtimeRender);

    // Call internal driver init
    initRenderDriver(initParams);

    // Setup ImageWriteDriver for checkpoint file write
    ImageWriteDriver::init();
}

void
cleanUpGlobalDriver()
{
    // just a pass through to keep API clean
    cleanUpRenderDriver();
}

//////////////////////// RenderContext //////////////////////////

RenderContext::RenderContext(RenderOptions& options, std::stringstream* initMessages) :
    mOptions(options),
    mDriver(getRenderDriver()),
    mGeometryManager(nullptr),
    mSceneContext(nullptr),
    mPbrScene(nullptr),
    mCachedPixelFilterType(scene_rdl2::rdl2::PixelFilterType::box),
    mCachedPixelFilterWidth(-1.f),
    mIntegrator(nullptr),
    mLayer(nullptr),
    mCachedSamplingMode(SamplingMode::UNIFORM),
    mCachedMinSamplesPerPixel(0),
    mCachedMaxSamplesPerPixel(0),
    mCachedTargetAdaptiveError(0.0f),
    mRenderPrepRun(false),
    mRendering(false),
    mFirstFrame(true),
    mSceneUpdated(false),
    mHasBeenInit(false),
    mSceneLoaded(false),
    mLogTime(true),
    mInfoLoggingEnabled(false),
    mDebugLoggingEnabled(false),
    mGeometryManagerOptions(new rt::GeometryManagerOptions()),
    mRenderStats(nullptr),
    mRenderPrepTimingStats(nullptr),
    mPbrStatistics(new pbr::Statistics()),
    mGeomStatistics(new geom::internal::Statistics()),
    mExecutionMode(ExecutionMode::AUTO) // for debugConsole command
{
    MNRY_ASSERT(mDriver.get());

    // for resume history meta data, we have to current time as proc start time
    mResumeHistoryMetaData.reset(new ResumeHistoryMetaData);
    mResumeHistoryMetaData->setProcStartTime();

    //------------------------------

    // Initialize set of standard attributes
    shading::StandardAttributes::init();

    // Initialize openvdb
    openvdb::initialize();

    // Set up the RDL SceneContext.
    mSceneContext.reset(new scene_rdl2::rdl2::SceneContext);

    // Set up fatal shade/sample funcs for use when shaders have a fatal error
    mSceneContext->setFatalShadeFunc(fatalShade);
    mSceneContext->setFatalSampleFunc(fatalSample);
    mSceneContext->setFatalSampleNormalFunc(fatalSampleNormal);

    mRenderStats.reset(new RenderStats);
    mRenderPrepTimingStats.reset(new RenderPrepTimingStats);

    // place object loading messages in the init messages
    // only if running at the -debug level
    std::stringstream * const logMessages =
        scene_rdl2::logging::Logger::isDebugEnabled(__FILE__) ? initMessages : nullptr;

    // Set up callbacks to install ThreadLocalObjectState on each Shader SceneObject
    // Over allocate the ThreadLocalObjectState arrays in case the main thread or GUI
    // thread ever call any shader related functions.
    const int numTLS = getMaxNumTLS();
    mSceneContext->addCreateSceneObjectCallback([numTLS, logMessages, this] (scene_rdl2::rdl2::SceneObject *obj) {
            if (obj->isA<scene_rdl2::rdl2::Shader>()) {
                scene_rdl2::rdl2::Shader *shader = obj->asA<scene_rdl2::rdl2::Shader>();
                mRenderStats->logLoadingSceneUpdates(logMessages,
                                                     shader->getSceneClass().getName(),
                                                     shader->getName());
                auto tlos = shader->getThreadLocalObjectState();
                if (tlos == nullptr) {
                    tlos = moonray::shading::ThreadLocalObjectState::alignedAlloc(numTLS);
                    shader->setThreadLocalObjectState(tlos);
                }
            }
        });
    mSceneContext->addDeleteSceneObjectCallback([numTLS, this] (scene_rdl2::rdl2::SceneObject *obj) {
            moonray::shading::ThreadLocalObjectState state;
            if (obj->isA<scene_rdl2::rdl2::Shader>()) {
                auto tlos = obj->asA<scene_rdl2::rdl2::Shader>()->getThreadLocalObjectState();
                moonray::shading::ThreadLocalObjectState::deallocate(numTLS, tlos);
                obj->asA<scene_rdl2::rdl2::Shader>()->setThreadLocalObjectState(nullptr);
            }
        });


    if (mSceneContext->getDsoPath().empty()) {
        mSceneContext->setDsoPath(mOptions.getDsoPath());
    } else if (!mOptions.getDsoPath().empty()) {
        mSceneContext->setDsoPath(mOptions.getDsoPath() + ":" + mSceneContext->getDsoPath());
    }

    parserConfigure();
}

void
RenderContext::createMeshLightLayer() {
    // Dummy LightSet
    scene_rdl2::rdl2::LightSet* dummyLightSet = mSceneContext->
        createSceneObject("LightSet", "MeshLightLightSet")->asA<scene_rdl2::rdl2::LightSet>();

    mMeshLightLayer->beginUpdate();

    // get all light sets from main layer
    scene_rdl2::rdl2::Layer::LightSetSet lightSets;
    mLayer->getAllLightSets(lightSets);

    for (const scene_rdl2::rdl2::LightSet* lightSet : lightSets) {
        const auto& lights = lightSet->getLights();
        for (scene_rdl2::rdl2::SceneObject* so : lights) {
            // get mesh light
            if (so->getSceneClass().getName() != "MeshLight") {
                continue;
            }
            const scene_rdl2::rdl2::Light* rdlLight = so->asA<scene_rdl2::rdl2::Light>();

            // get geometry
            scene_rdl2::rdl2::SceneObject* geomSo = rdlLight->get<scene_rdl2::rdl2::SceneObject*>("geometry");
            if (!geomSo) {
                continue;
            }
            scene_rdl2::rdl2::Geometry* geom = geomSo->asA<scene_rdl2::rdl2::Geometry>();

            // We cannot load in a geometry that already exists in the main scene layer
            // The reasons include:
            //  1) A risk that there is a circular dependency between the mesh light and
            //     the reference geometry using the shadow link
            //  2) Loading and tessellating the same primitive, but a primitive whose
            //     face set assignment ids are different. This would create a conflict
            //     over which face sets to tessellate.
            //  3) We would need a better way to specify which primitives go into the
            //     EmbreeAccelerator and which do not when they come from the same procedural
            if (mLayer->contains(geom)) {
                rdlLight->warn("\"" + geom->getSceneClass().getName() + "(" +
                    geom->getName() + ")\" cannot be referenced in a MeshLight when "
                    "it is in \"" + mLayer->getSceneClass().getName() + "(" +
                    mLayer->getName() + ")\". Please use a different geometry.");
                continue;
            }

            if (geom && geom->updateRequired()) {
                // Material contains "map shader" so that it can grab requested primitive attributes
                scene_rdl2::rdl2::Material *material = mSceneContext->
                    createSceneObject("DwaBaseMaterial",
                            geom->getName() + "_MeshLightMaterial")->asA<scene_rdl2::rdl2::Material>();
                scene_rdl2::rdl2::SceneObject* mapShader= rdlLight->get<scene_rdl2::rdl2::SceneObject*>("map_shader");
                material->beginUpdate();
                material->setBinding("albedo", mapShader);
                material->endUpdate();

                // get part names
                const std::vector<std::string>& partList = rdlLight->get<scene_rdl2::rdl2::StringVector>("parts");

                // assign to layer
                scene_rdl2::rdl2::LayerAssignment layerAssignment;
                layerAssignment.mMaterial = material;
                layerAssignment.mLightSet = dummyLightSet;
                for (auto part : partList) {
                    mMeshLightLayer->assign(geom, part, layerAssignment);
                }
                if (partList.empty()) {
                    mMeshLightLayer->assign(geom, "", layerAssignment);
                }
            }
        }
    }

    mMeshLightLayer->endUpdate();
}

void
RenderContext::initialize(std::stringstream &initMessages, LoggingConfiguration loggingConfig)
{
    // Load the scene into the SceneContext.
    loadScene(initMessages);

    scene_rdl2::rdl2::SceneVariables& sceneVars = mSceneContext->getSceneVariables();

    mInfoLoggingEnabled = sceneVars.get(scene_rdl2::rdl2::SceneVariables::sInfoKey);
    mDebugLoggingEnabled = sceneVars.get(scene_rdl2::rdl2::SceneVariables::sDebugKey);
    if (mDebugLoggingEnabled) {
        // debug logging includes info logging
        scene_rdl2::logging::Logger::setDebugLevel();
    } else if (mInfoLoggingEnabled) {
        // This is subtle: scene_rdl2::logging::Logger::init is called at load time
        // before we ever hit main(). It captures the arguments passed to main.
        // Intentionally or not, we use "-info" on the command line for
        // moonray/moonray_gui to turn on human-readable verbose output. This
        // is the same flag for which scene_rdl2::logging looks to turn on logging for
        // the information log level, which is what we use for our
        // human-readable verbose output. So, by passing "-info" as a
        // command-line argument, we implicitly turn on the informational log
        // level. If, however, we turn on the human-readable verbose output
        // through other means (e.g. the rdl2 file), we have to switch the
        // logging level ourselves.
        scene_rdl2::logging::Logger::setInfoLevel();
    }

    mRenderStats->openInfoStream(mInfoLoggingEnabled || mDebugLoggingEnabled);
    mRenderStats->openStatsStream(sceneVars.get(scene_rdl2::rdl2::SceneVariables::sStatsFile));

#ifndef DEBUG
    if (loggingConfig == LoggingConfiguration::ATHENA_ENABLED) {
        const auto athenaDebug = sceneVars.get(scene_rdl2::rdl2::SceneVariables::sAthenaDebug);
        mRenderStats->openAthenaStream(mOptions.getGUID(), athenaDebug);
    }
#endif

    mRenderStats->logInfoPrependStringHeader();
    mRenderStats->logHardwareConfiguration(mOptions, sceneVars);
    mRenderStats->logInitializationConfiguration(initMessages);

    mRenderStats->logInfoEmptyLine();
    mRenderStats->logRenderOptions(mOptions);

    // Get the camera and layer objects.
    try {
        std::vector<const scene_rdl2::rdl2::Camera*> cameras = mSceneContext->getActiveCameras();
        initActiveCamera(cameras[0]);
    } catch (scene_rdl2::except::KeyError& e) {
        std::stringstream errMsg;
        errMsg << e.what();
        throw scene_rdl2::except::RuntimeError(errMsg.str());
    }
    try {
        scene_rdl2::rdl2::SceneObject *layer = sceneVars.getLayer();
        if (!layer) {
            throw scene_rdl2::except::RuntimeError("SceneVariables contains no layer.");
        }
        mLayer = layer->asA<scene_rdl2::rdl2::Layer>();
        if (!mLayer) {
            throw scene_rdl2::except::RuntimeError("Scene object is not a layer.");
        }
        sceneVars.beginUpdate();
        sceneVars.set(scene_rdl2::rdl2::SceneVariables::sLayer, layer);
        sceneVars.endUpdate();

        // create mesh light layer, which is a dummy layer to load geometries
        // that belong to mesh lights
        mMeshLightLayer = mSceneContext->
            createSceneObject("Layer", "MeshLightLayer")->asA<scene_rdl2::rdl2::Layer>();

    } catch (scene_rdl2::except::KeyError& e) {
        std::stringstream errMsg;
        errMsg << e.what();
        throw scene_rdl2::except::RuntimeError(errMsg.str());
    }

    mRenderStats->logSceneVariables(sceneVars);

    // Update texture system limits.
    texture::TextureSampler *sampler = MNRY_VERIFY(texture::getTextureSampler());
    sampler->setOpenFileLimit(sceneVars.get(scene_rdl2::rdl2::SceneVariables::sTextureFileHandleCount));

    // configure GeometryManager options
    mGeometryManagerOptions->accelOptions.maxThreads = getNumTBBThreads();
    mGeometryManagerOptions->accelOptions.verbose = false;

    mGeometryManagerOptions->stats.logString =
        [stats = mRenderStats.get()](const std::string& str)
        {
            stats->logString(str);
        };
    mGeometryManagerOptions->stats.logDebugString =
        [stats = mRenderStats.get()](const std::string& str)
        {
            stats->logDebugString(str);
        };
    mGeometryManagerOptions->stats.logGeneratingProcedurals =
        [stats = mRenderStats.get()](const std::string& sceneClass, const std::string& name)
        {
            stats->logGeneratingProcedurals(sceneClass, name);
        };
    // Initialize GeometryManager with current SceneContext and Layer
    mGeometryManager.reset(new rt::GeometryManager(mSceneContext.get(),
                                                   *mGeometryManagerOptions));

    createPbrScene();
    mHasBeenInit = true;
}

void
RenderContext::initActiveCamera(const scene_rdl2::rdl2::Camera *camera)
{
    mCamera = camera;

    // Initialize the length of the pixel sample map vectors
    mPixelSampleMap.reset();
    mCachedPixelSampleMapName.clear();
    // The max pixel sample value is a multiplier for the FrameState::mNumSamplesPerPixel
    // This is needed to initialize the render passes with the appropriate mEndSampleIdx.
    // By default this multiplier should be 1.
    mMaxPixelSampleValue = 1.0f;
}


RenderContext::~RenderContext()
{
    if (RenderContextConsoleDriver::get()) {
        // We have to reset RenderContext pointer inside renderContextConsoleDriver
        RenderContextConsoleDriver::get()->setRenderContext(nullptr); // MTsafe
    }

    if (mRendering) {
        stopFrame();
    }
    // There is an issue NOVAVP-12 where we get render artifacts
    // Whenever we try to reset the scene. Adding this call here
    // Fixes it. For some reason it looks like a problem in oiio
    // Where the sampling fails. This is a workaround and should
    // Be investigated further.

    // mlee Update: this line seems to be causing slow shutdown,
    // see MOONRAY-2550. Commenting it out for now and waiting
    // to see if there are still outstanding issues.
    //moonray::texture::getTextureSampler()->invalidateAllResources();
}

void
RenderContext::updateGeometry(const std::vector<moonray::ObjectData>& updateData)
{
    // this will time the function until it goes out of scope.
    RenderTimer funcTimer(mRenderStats->mRebuildGeometryTime);

    MNRY_ASSERT(mPbrScene, "Must render the scene at least once before updating geometry.");
    MNRY_ASSERT_REQUIRE(!mRendering, "Cannot update geometry while rendering is in progress.");

    incrementCurfield();

    int currentFrame = getCurrentFrame();
    // Get the motion blur related parameters
    geom::MotionBlurParams motionBlurParams = getMotionBlurParams();

    // Get the transform world --> render.
    const Mat4d& world2render = mPbrScene->getWorld2Render();

    tbb::blocked_range<size_t> range(0, updateData.size());
    tbb::parallel_for(range, [&](const tbb::blocked_range<size_t> &r) {
        // Handle each object in the update data.
        for (size_t i = r.begin(); i < r.end(); ++i ) {
            const moonray::ObjectData& objectData = updateData[i];

            // Get the geometry object and procedural associated with this update.
            scene_rdl2::rdl2::Geometry* geometry = nullptr;
            std::string geometryName;
            if (objectData.mNodeName.empty()) {
                geometryName = "/" + objectData.mAssetName + "/geometry/" +
                               objectData.mSubAssetName;
            } else {
                geometryName = objectData.mNodeName;
            }

            try {
                geometry = mSceneContext->getSceneObject(geometryName)->asA<scene_rdl2::rdl2::Geometry>();
                // TODO: no way to pass -info to hostd so LOG_INFO doesn't print :-(
            } catch (const std::exception& e) {
                Logger::error("Cannot apply geometry update for geometry \""
                              , geometryName , "\" (" , e.what() , ")");
                continue;
            }

            // Aggregate all the mesh names and vertex buffers for this object.
            std::vector<std::string> meshNames;
            std::vector<const std::vector<float>* > vertexBuffers;
            std::for_each(objectData.mObjectMeshes.begin(), objectData.mObjectMeshes.end(),
                          [&meshNames, &vertexBuffers](const moonray::ObjectMesh& objectMesh) {
                meshNames.push_back(objectMesh.mMeshName);
                vertexBuffers.push_back(&objectMesh.mMeshPositions);
            });

            // change the status to update if it was none.
            mGeometryManager->compareAndSwapFlag(rt::ChangeFlag::ALL, rt::ChangeFlag::NONE);
            mGeometryManager->updateGeometryData(mLayer, geometry, meshNames,
                vertexBuffers, world2render, currentFrame, motionBlurParams,
                getNumTBBThreads());
            mGeometryManager->updateGeometryData(mMeshLightLayer, geometry, meshNames,
                vertexBuffers, world2render, currentFrame, motionBlurParams,
                getNumTBBThreads());
        }
    });

    mSceneUpdated = true;
}

bool
RenderContext::bakeGeometry(std::vector<std::unique_ptr<geom::BakedMesh>>& bakedMeshes,
                            std::vector<std::unique_ptr<geom::BakedCurves>>& bakedCurves)
{
    // The RenderOutputDriver is needed to build the primitive attribute table
    // because some of the attrs depend on whether there are certain render outputs.
    try {
        mRenderOutputDriver.reset(new RenderOutputDriver(this));
    }
    catch(const std::exception &e){
        Logger::error(e.what());
        return false;
    }

    resetShaderStatsAndLogs();  // initialize EventLog to allocate memory for messages

    // Call update() on all SceneObjects
    mSceneContext->applyUpdates(mLayer);
    // Add MeshLight geometry to a MeshLightLayer and assign each geometry a material, 
    // which will keep track of the MeshLight's attribute table
    createMeshLightLayer();
    {
        scene_rdl2::rdl2::Layer::RootShaderSet rootShaders;
        mLayer->getAllRootShaders(rootShaders);
        buildPrimitiveAttributeTables(rootShaders);

        scene_rdl2::rdl2::Layer::RootShaderSet meshLightRootShaders;
        mMeshLightLayer->getAllRootShaders(meshLightRootShaders);
        buildPrimitiveAttributeTables(meshLightRootShaders);
    }

    MNRY_ASSERT(mPbrScene, "Must have loaded PBR library before loading procedurals.");

    int currentFrame = getCurrentFrame();
    // Get the motion blur related parameters
    geom::MotionBlurParams motionBlurParams = getMotionBlurParams(true); // bake=true

    // Get the transform world --> render
    const Mat4d& world2render = mPbrScene->getWorld2Render();

    // Cameras may require per geometry primitive attributes (e.g. the BakeCamera)
    shading::PerGeometryAttributeKeySet perGeometryAttributes;
    mPbrScene->getCamera()->getRequiredPrimAttributes(perGeometryAttributes);

    mGeometryManager->loadGeometries(mLayer, rt::ChangeFlag::ALL, world2render, currentFrame, motionBlurParams,
                                     getNumTBBThreads(), perGeometryAttributes);
    mGeometryManager->loadGeometries(mMeshLightLayer, rt::ChangeFlag::ALL, world2render, currentFrame, motionBlurParams,
                                     getNumTBBThreads(), perGeometryAttributes);

    // Get the camera frustum and render to camera matrices for times points 0 and 1
    // TODO: generalize for multi-segment motion blur. Currently we only have 2
    // motion samples and assume the ray time points are 0 and 1. In multi-segment
    // motion blur there will be multiple motion samples with a ray time range
    // of [0,1].
    std::vector<mcrt_common::Frustum> frustums;
    const pbr::Camera *camera = mPbrScene->getCamera();
    if (camera->hasFrustum()) {
        frustums.push_back(mcrt_common::Frustum());
        camera->computeFrustum(&frustums.back(), 0, true);  // frustum at shutter open
        frustums.push_back(mcrt_common::Frustum());
        camera->computeFrustum(&frustums.back(), 1, true);  // frustum at shutter close
    }

    const scene_rdl2::rdl2::Camera* dicingCamera = mSceneContext->getDicingCamera();
    mGeometryManager->bakeGeometry(mLayer,
                                   motionBlurParams,
                                   frustums,
                                   world2render,
                                   bakedMeshes,
                                   bakedCurves,
                                   dicingCamera);
    return true;
}

void
RenderContext::updateScene(const std::string& manifest, const std::string& payload)
{
    MNRY_ASSERT_REQUIRE(!mRendering, "Cannot update scene data while rendering is in progress.");

    if (!mSceneLoaded) {
        // Add the update to the queue that will be processed after we've loaded our scene
        mUpdateQueue.push_back(std::make_pair(manifest, payload));
    } else {
        // Apply the binary update.
        RenderTimer funcTimer(mRenderStats->mUpdateSceneTime);
        scene_rdl2::rdl2::BinaryReader reader(*mSceneContext);
        reader.fromBytes(manifest, payload);
        mSceneUpdated = true;
    }
}

void
RenderContext::updateScene(const std::string& filename)
{
    MNRY_ASSERT_REQUIRE(!mRendering, "Cannot update scene data while rendering is in progress.");

    if (!mSceneLoaded) {
        mUpdateQueue.push_back(std::make_pair(FILE_TOKEN, filename));
    } else {
        RenderTimer funcTimer(mRenderStats->mUpdateSceneTime);
        readSceneFromFile(filename, *mSceneContext);
        mSceneUpdated = true;
    }
}

void
RenderContext::invalidateAllTextureResources()
{
    moonray::texture::getTextureSampler()->invalidateAllResources();
    setSceneUpdated();
}

void
RenderContext::invalidateTextureResources(const std::vector<std::string>& resources)
{
    moonray::texture::getTextureSampler()->invalidateResources(resources);
    setSceneUpdated();
}

RenderContext::RP_RESULT
RenderContext::startFrame()
{
    mRenderPrepRun = true;

    CheckpointSigIntHandler::disable(); // We don't need checkpointSigIntHandler for renderPrep stage

    scene_rdl2::rec_time::RecTime recTimeWhole;
    recTimeWhole.start();

    mResumeHistoryMetaData->setFrameStartTime(); // record frame start timing for resume history 
    mRenderPrepExecTracker.init();

    {
        texture::TextureSampler* sampler = texture::getTextureSampler();

        // Reset texture system stats
        sampler->resetStats();

        // Update texture system limits.
        const scene_rdl2::rdl2::SceneVariables& sceneVars = mSceneContext->getSceneVariables();        
        int textureCacheSizeMb = sceneVars.get(scene_rdl2::rdl2::SceneVariables::sTextureCacheSizeMb);
        if (mOptions.getTextureCacheSizeMb() != 0) {
            textureCacheSizeMb = mOptions.getTextureCacheSizeMb();
        }
        if (static_cast<int>(sampler->getMemoryUsage()) != textureCacheSizeMb) {
            sampler->setMemoryUsage(static_cast<float>(textureCacheSizeMb));
        }
    }

    MNRY_ASSERT_REQUIRE(!mRendering, "Must stop rendering before starting it again.");

    // The frame officially starts now! This time includes the update portion
    // of the frame also.
    double frameStartTime = scene_rdl2::util::getSeconds();

    const scene_rdl2::rdl2::SceneVariables &vars = mSceneContext->getSceneVariables();

    { // renderContext console setup for debug purpose
        int debugConsolePort = vars.get(scene_rdl2::rdl2::SceneVariables::sDebugConsole);
        if (debugConsolePort >= 0) {
            if (!RenderContextConsoleDriver::get()) {
                RenderContextConsoleDriver::init(debugConsolePort);
            }
        }
        if (RenderContextConsoleDriver::get()) {
            // Very conservative way of updating RenderContext pointer into renderContextConsoleDriver.
            RenderContextConsoleDriver::get()->setRenderContext(this);
        }
    }

    // Determine if to render this frame in scalar, vectorized, or xpu mode.
    ExecutionMode executionMode = ExecutionMode::XPU;
    ExecutionMode desiredExecutionMode = mOptions.getDesiredExecutionMode();
    std::string executionModeString;
    std::string missingVecFeatures;
    const bool vecSupport = canRunVectorized(missingVecFeatures);
    bool allowUnsupportedXPUFeatures = true;

    switch (desiredExecutionMode) {

    case ExecutionMode::SCALAR:
        executionModeString = "Executing a scalar render since execution mode was set to scalar.";
        executionMode = ExecutionMode::SCALAR;
    break;

    case ExecutionMode::VECTORIZED:
        executionModeString = "Executing a vectorized render since execution mode was set to vector.";
        if (!vecSupport) {
            executionModeString += "  The following features will be missing: ";
            executionModeString += missingVecFeatures;
            executionModeString += ".";
        }
        executionMode = ExecutionMode::VECTORIZED;
    break;

    case ExecutionMode::XPU:
        // XPU mode inherits the same limitations of vector mode
        executionModeString = "Executing an xpu render since execution mode was set to xpu.";
        if (!vecSupport) {
            executionModeString += "  The following features will be missing: ";
            executionModeString += missingVecFeatures;
            executionModeString += ".";
        }
        executionMode = ExecutionMode::XPU;
        // If there is an error setting up the GPU, we will fall back to vector mode
        // in GeometryManager::updateGPUAccelerator().
    break;

    case ExecutionMode::AUTO:
    default:
        if (!vecSupport) {
            executionModeString = "Executing a scalar render since execution mode was set to auto.";
            executionModeString += "  The following features are missing vector mode support: ";
            executionModeString += missingVecFeatures;
            executionModeString += ".";
            executionMode = ExecutionMode::SCALAR;
        } else {
            executionModeString = "Executing an XPU render since execution mode was set to auto.";
            allowUnsupportedXPUFeatures = false; // want to fall back if unsupported
            executionMode = ExecutionMode::XPU;
            // If there is an error setting up the GPU, we will fall back to vector mode
            // in GeometryManager::updateGPUAccelerator().
        }    
    }

    // Log information as to whether we're executing in scalar, vectorized, or xpu mode
    // and the reason why.
    mExecutionMode = executionMode; // for debugConsole command and McrtNodeInfo update
    mExecutionModeString = executionModeString; // for debugConsole command
    mRenderStats->logExecModeConfiguration(executionMode);
    Logger::info(executionModeString);

    // Make sure everything is ready to render.
    scene_rdl2::rec_time::RecTime recTime;
    recTime.start();
    RP_RESULT execResult = renderPrep(executionMode, allowUnsupportedXPUFeatures); // may throw
    mDriver->pushRenderPrepTime(recTime.end()); // statistical info update for debug

#if defined(USE_PARTITIONED_PIXEL) || defined(USE_PARTITIONED_LENS) || defined(USE_PARTITIONED_TIME)
    // Ensure frame to frame noise is locked for the real-time case.
    if (getRenderMode() != RenderMode::REALTIME) {
        const auto temporalKey = scene_rdl2::rdl2::SceneVariables::sLockFrameNoise;
        const auto frameNum = getCurrentFrame();

#if defined(USE_PARTITIONED_PIXEL)
        if (!mPbrScene->getRdlSceneContext()->getSceneVariables().get(temporalKey)) {
            if (mOptions.getApplicationMode() != ApplicationMode::MOTIONCAPTURE) {
                pbr::kPixelPartition.rotate(frameNum);
            }
        }
#endif

#if defined(USE_PARTITIONED_LENS)
        if (!mPbrScene->getRdlSceneContext()->getSceneVariables().get(temporalKey)) {
            if (mOptions.getApplicationMode() != ApplicationMode::MOTIONCAPTURE) {
                pbr::kLensPartition.rotate(frameNum);
            }
        }
#endif

#if defined(USE_PARTITIONED_TIME)
        if (!mPbrScene->getRdlSceneContext()->getSceneVariables().get(temporalKey)) {
            if (mOptions.getApplicationMode() != ApplicationMode::MOTIONCAPTURE) {
                pbr::kTimePartition.rotate(frameNum);
            }
        }
#endif
    }
#endif

    // Update pixel filter if needed.
    scene_rdl2::rdl2::PixelFilterType pixelFilterType = (scene_rdl2::rdl2::PixelFilterType)vars.get(scene_rdl2::rdl2::SceneVariables::sPixelFilterType);
    float pixelFilterWidth = vars.get(scene_rdl2::rdl2::SceneVariables::sPixelFilterWidth);
    if (mCachedPixelFilterType != pixelFilterType ||
        mCachedPixelFilterWidth != pixelFilterWidth) {
        mPixelFilter = getPixelFilter(pixelFilterType, pixelFilterWidth);
        mCachedPixelFilterType = pixelFilterType;
        mCachedPixelFilterWidth = pixelFilterWidth;
    }

    // Get the max pixel sample values.

    const std::string pixelSampleMapName = mCamera->get(scene_rdl2::rdl2::Camera::sPixelSampleMap);

    // update pixel sample map
    if (pixelSampleMapName.empty()) {
        if (mPixelSampleMap) {
            // delete pixel sample map
            mPixelSampleMap->cleanUp();
            mPixelSampleMap.reset();
        }
        mCachedPixelSampleMapName.clear();
        mMaxPixelSampleValue = 1.0;
    } else {
        // create pixel buffer
        if (!mPixelSampleMap) {
            mPixelSampleMap = std::make_unique<scene_rdl2::fb_util::PixelBuffer<float>>();
        }

        // change pixel sample map
        unsigned width = vars.getRezedWidth();
        unsigned height = vars.getRezedHeight();
        if (pixelSampleMapName != mCachedPixelSampleMapName ||
             width != mPixelSampleMap->getWidth() ||
             height != mPixelSampleMap->getHeight()) {
            mPixelSampleMap->init(width, height);
            rndr::readPixelBuffer(*mPixelSampleMap, pixelSampleMapName,
                                   width, height);
            mCachedPixelSampleMapName = pixelSampleMapName;

            // find the max value in map
            unsigned numPixels = mPixelSampleMap->getArea();
            auto data = mPixelSampleMap->getData();
            mMaxPixelSampleValue = *std::max_element(data, data + numPixels);
        }
    }

    // get deep id channel names
    mDeepIDChannelNames.reset(new std::vector<std::string>());
    *mDeepIDChannelNames = vars.get(scene_rdl2::rdl2::SceneVariables::sDeepIDAttributeNames);

    // Check if the user requested to record debug rays for this frame.
    if (!vars.get(scene_rdl2::rdl2::SceneVariables::sDebugRaysFile).empty()) {
        if (mDriver->getDebugRayState() == RenderDriver::READY) {
            const Mat4d& render2world = mPbrScene->getRender2World();
            Viewport vp = scene_rdl2::math::convertToClosedViewport(vars.getRezedRegionWindow());
            BBox2i bboxVp(vp.min(), vp.max());
            // Will accept a double to float precision loss for debug rays
            pbr::DebugRayRecorder::enableRecording(bboxVp, toFloat(render2world));
            mDriver->switchDebugRayState(RenderDriver::READY, RenderDriver::REQUEST_RECORD);
        }
    }

    // Don't spam the logs if in real-time mode.
    // mTotalRenderPrepTime was set inside of the prep function call
    if (getRenderMode() != RenderMode::REALTIME &&
        mOptions.getApplicationMode() != ApplicationMode::MOTIONCAPTURE) {
        if (mLogTime) { // only print this if previous pass printing timing
            mRenderStats->logInfoEmptyLine();
            mRenderStats->updateAndLogRenderPrepStats();
        }

        // Print dso usage
        mRenderStats->logDsoUsage(mSceneContext->getDsoCounts());

        mRenderStats->startRenderStats();
    }

    // Condition scene variables and other state for this frame.
    FrameState frameState;
    buildFrameState(&frameState, frameStartTime, executionMode);

    // Record some info for resume history from frameState
    mResumeHistoryMetaData->setNumOfThreads(frameState.mNumRenderThreads);
    if (frameState.mSamplingMode == SamplingMode::UNIFORM) {
        mResumeHistoryMetaData->setUniformSample(frameState.mMinSamplesPerPixel,
                                                 frameState.mMaxSamplesPerPixel);
    } else if (frameState.mSamplingMode == SamplingMode::ADAPTIVE) {
        mResumeHistoryMetaData->setAdaptiveSample(frameState.mMinSamplesPerPixel,
                                                  frameState.mMaxSamplesPerPixel,
                                                  frameState.mTargetAdaptiveError);
    }

    // Construct pixelId scramble table setup.
    Film::constructPixelFillOrderTable(frameState.mRenderNodeIdx, frameState.mNumRenderNodes);

    // save sampling mode and parameters
    mCachedSamplingMode = frameState.mSamplingMode;
    mCachedMinSamplesPerPixel = frameState.mMinSamplesPerPixel;
    mCachedMaxSamplesPerPixel = frameState.mMaxSamplesPerPixel;
    mCachedTargetAdaptiveError = frameState.mTargetAdaptiveError;

    // We store maxSamplesPerPixel info into renderOutputDriver for write action before start render
    mRenderOutputDriver->setFinalMaxSamplesPerPix(frameState.mMaxSamplesPerPixel);

    // add info to ImageWriteDriver
    {
        const scene_rdl2::rdl2::SceneVariables &vars = mSceneContext->getSceneVariables();
        ImageWriteDriver::get()->setTwoStageOutput(frameState.mTwoStageOutput);
        ImageWriteDriver::get()->setTmpDirectory(vars.getTmpDir());
        ImageWriteDriver::get()->setMaxBgCache(vars.get(scene_rdl2::rdl2::SceneVariables::sCheckpointMaxBgCache));
        ImageWriteDriver::get()->setRenderContext(this);
    }

    // Save some info for resume render related info
    {
        const scene_rdl2::rdl2::SceneVariables &vars = getSceneContext().getSceneVariables();
        mOnResumeScript = vars.get(scene_rdl2::rdl2::SceneVariables::sOnResumeScript);
    }
    getResumeHistoryMetaData()->setBgCheckpointWriteMode(frameState.mCheckpointBgWrite);

    // Update integrator state.
    updatePbrState(frameState);

    // In bundled vectorized and xpu mode we lazily allocate queues for any new
    // Primitives and Materials introduced into the scene since the previous frame.
    // In scalar mode we only set the id for each material.
    if (frameState.mExecutionMode == mcrt_common::ExecutionMode::VECTORIZED ||
        frameState.mExecutionMode == mcrt_common::ExecutionMode::XPU) {
        unsigned shadeQueueSize = mcrt_common::getTLSInitParams().mShadeQueueSize;
        MNRY_ASSERT(shadeQueueSize);
        shading::Material::allocShadeQueues(shadeQueueSize, pbr::shadeBundleHandler);
    } else {
        shading::Material::initMaterialIds();
    }

    // Setup RenderContext pointer into frame state
    frameState.mRenderContext = this;

    // Setup the XPU queues in the RenderDriver if we are XPU accelerated.
    // isGPUEnabled() may return false even in XPU mode if we failed to create it,
    // e.g. unsupported geometry, out of VRAM, etc.  In this case we don't have
    // any XPU queues.
    if (mGeometryManager->isGPUEnabled()) {
        mDriver->createXPUQueues();
    } else {
        mDriver->freeXPUQueues();
    }

    if (execResult == RP_RESULT::CANCELED) {
        mRenderPrepRun = false;
        return RP_RESULT::CANCELED;
    }

    // Invoke the render driver.
    mRendering = true;
    mDriver->startFrame(frameState);

    mRenderPrepRun = false;
    mRenderPrepTimingStats->setWholeStartFrame(recTimeWhole.end());

    return RP_RESULT::FINISHED;
}

void
RenderContext::requestStop()
{
    MNRY_ASSERT_REQUIRE(mRendering, "Must start rendering before it can be stopped.");
    mDriver->requestStop();
}

void
RenderContext::requestStopAsyncSignalSafe() // for call from signal handler
{
    mDriver->requestStopAsyncSignalSafe();
}

void
RenderContext::requestStopAtFrameReadyForDisplay()
{
    mDriver->requestStopAtFrameReadyForDisplay();
}

void
RenderContext::stopFrame()
{
    mRenderPrepTimingStats->recTimeStart();

    MNRY_ASSERT_REQUIRE(mRendering, "Must start rendering before it can be stopped.");

    // Halt the render driver.
    mDriver->stopFrame();

    mRenderPrepTimingStats->recTime(RenderPrepTimingStats::StopFrameTag::MDRIVER_STOPFRAME);

    mPbrScene->postFrame();

    mRenderPrepTimingStats->recTime(RenderPrepTimingStats::StopFrameTag::MPBRSCENE_POSTFRAME);

    // Accumulate pbr stats data.
    mPbrStatistics->mMcrtTime = mDriver->getLastFrameMcrtDuration();
    mPbrStatistics->mMcrtUtilization = mDriver->getLastFrameMcrtUtilization();
    pbr::forEachTLS([this](pbr::TLState const *tls){ (*mPbrStatistics) += tls->mStatistics; });

    // Accumulate geom stats data.
    geom::internal::forEachTLS([this](geom::internal::TLState const *tls) {
        (*mGeomStatistics) += tls->mStatistics;
    });

    mRenderPrepTimingStats->recTime(RenderPrepTimingStats::StopFrameTag::ACCUMULATE_PBR_STATS_DATA);

    // Collect all per-shader stats into mRenderStats
    collectShaderStats();

    mRenderPrepTimingStats->recTime(RenderPrepTimingStats::StopFrameTag::COLLECTSHADERSTATS);

    if (mRenderStats->getLogInfo() ||
        mRenderStats->getLogCsv()  ||
        mRenderStats->getLogAthena()) {

        reportGeometryStatistics();
    }

    // Any errors that occurred during shading will be reported at this time
    reportShadingLogs();

    mRenderPrepTimingStats->recTime(RenderPrepTimingStats::StopFrameTag::REPORTSHADINGLOGS);

    // Don't spam the logs if in real-time mode, or if interactive changes interrupt
    // rendering too quickly (less than 10 seconds). I tried much shorter timeouts
    // but it has to be that long to avoid spamming that makes it impossible to see
    // useful messages with typical usage (at least by me) of moving the camera,
    // looking at the result, and then deciding to move the camera again.
    // The decision for this pass is remembered in mLogTime for next pass to stop
    // it from printing renderprep timing.
    if (getRenderMode() != RenderMode::REALTIME &&
        mOptions.getApplicationMode() != ApplicationMode::MOTIONCAPTURE) {
        if (isFrameComplete() || mPbrStatistics->mMcrtTime >= 10.0) {
            mLogTime = true;
            mRenderStats->logInfoEmptyLine();
            mRenderStats->logSamplingStats(*mPbrStatistics, *mGeomStatistics);

            mRenderStats->logInfoEmptyLine();
            mRenderStats->logTexturingStats(*texture::getTextureSampler(), mDebugLoggingEnabled);

            mRenderStats->logRenderingStats(*mPbrStatistics,
                static_cast<mcrt_common::ExecutionMode>(mDriver->getFrameState().mExecutionMode),
                mSceneContext->getSceneVariables());

            mRenderStats->logRenderOutputs(mSceneContext->getAllRenderOutputs());
        } else {
            mLogTime = false;
        }
    }

    mRenderStats->flush();

    // Do debug ray database processing if we've just been recording rays.
    if (mDriver->getDebugRayState() == RenderDriver::RECORDING_COMPLETE) {

        buildAndSaveRayDatabase();

        pbr::forEachTLS([](pbr::TLState *tls) {
            tls->mRayRecorder->cleanUp();
        });

        pbr::DebugRayRecorder::disableRecording();

        // Only save the file out for one frame.
        scene_rdl2::rdl2::SceneVariables& sceneVars = mSceneContext->getSceneVariables();
        scene_rdl2::rdl2::SceneVariables::UpdateGuard guard(&sceneVars);
        sceneVars.set(scene_rdl2::rdl2::SceneVariables::sDebugRaysFile, std::string(""));

        mDriver->switchDebugRayState(RenderDriver::RECORDING_COMPLETE, RenderDriver::READY);
    }

    mRenderPrepTimingStats->recTime(RenderPrepTimingStats::StopFrameTag::LOG_FLUSH);

    mRendering = false;
    mSceneUpdated = false;
    // TODO: Why is resetUpdates() not called at the end of
    // RenderContext::startFrame ? (at least after the call to mPbrScene->preFrame())
    mSceneContext->resetUpdates(mLayer);
    mSceneContext->resetUpdates(mMeshLightLayer);
    mRenderStats->reset();

    mcrt_common::resetAllAccumulators();

    mRenderPrepTimingStats->recTime(RenderPrepTimingStats::StopFrameTag::RESET);
    mRenderPrepTimingStats->recTimeEnd(RenderPrepTimingStats::StopFrameTag::WHOLE);
}

double
RenderContext::getLastFrameMcrtStartTime() const
{
    return mDriver->getLastFrameMcrtStartTime();
}

void
RenderContext::requestStopRenderAtPassBoundary()
{
    if (mRendering && !isFrameComplete()) {
        mDriver->requestStopAtPassBoundary();
    }
}

int
RenderContext::getCurrentFrame() const
{
    return mSceneContext->getSceneVariables().get(scene_rdl2::rdl2::SceneVariables::sFrameKey);
}

scene_rdl2::math::HalfOpenViewport
RenderContext::getRezedRegionWindow() const
{
    return mSceneContext->getSceneVariables().getRezedRegionWindow();
}

scene_rdl2::math::HalfOpenViewport
RenderContext::getRezedApertureWindow() const
{
    return mSceneContext->getSceneVariables().getRezedApertureWindow();
}


scene_rdl2::math::HalfOpenViewport
RenderContext::getRezedSubViewport() const
{
    return mSceneContext->getSceneVariables().getRezedSubViewport();
}

const scene_rdl2::rdl2::SceneContext&
RenderContext::getSceneContext() const
{
    return *mSceneContext;
}

scene_rdl2::rdl2::SceneContext&
RenderContext::getSceneContext()
{
    return *mSceneContext;
}

const pbr::Scene *
RenderContext::getScene() const
{
    return mPbrScene.get();
}

void
RenderContext::snapshotRenderBuffer(scene_rdl2::fb_util::RenderBuffer *renderBuffer, bool untile, bool parallel) const
{
    // Request a snapshot from the render driver.
    mDriver->snapshotRenderBuffer(MNRY_VERIFY(renderBuffer), untile, parallel);
}

void
RenderContext::snapshotRenderBufferOdd(scene_rdl2::fb_util::RenderBuffer *renderBufferOdd, bool untile, bool parallel) const
{
    // This API is used for snapshot of renderBufferODD related logic.
    // We only do snapshot if renderBufferOdd is setup in the renderOutputDriver.
    if (getRenderOutputDriver()->requiresRenderBufferOdd()) {
        mDriver->snapshotRenderBufferOdd(MNRY_VERIFY(renderBufferOdd), untile, parallel);
    }
}

const pbr::DeepBuffer*
RenderContext::getDeepBuffer() const
{
    return mDriver->getDeepBuffer();
}

pbr::CryptomatteBuffer*
RenderContext::getCryptomatteBuffer()
{
    return mDriver->getCryptomatteBuffer();
}

const pbr::CryptomatteBuffer*
RenderContext::getCryptomatteBuffer() const
{
    return mDriver->getCryptomatteBuffer();
}

void
RenderContext::snapshotDelta(scene_rdl2::fb_util::RenderBuffer *renderBuffer,
                             scene_rdl2::fb_util::FloatBuffer *weightBuffer,
                             scene_rdl2::fb_util::ActivePixels &activePixels,
                             bool parallel) const
//
// Snapshots the contents of the renderBuffer/weightBuffer w/ ActivePixels information
// for ProgressiveFrame message related logic. So renderBuffer is not normalized by weight yet.
// no resize, no extrapolation and no untiling logic is handled internally.
// Just create snapshot data with properly constructed activePixels based on
// difference between current and previous renderBuffer and weightBuffer.
// renderBuffer/weightBuffer is tiled format and renderBuffer is not normalized by weight.
//
{
    mDriver->snapshotDelta(renderBuffer, weightBuffer, activePixels, parallel);
}

void
RenderContext::snapshotDeltaRenderBufferOdd(scene_rdl2::fb_util::RenderBuffer *renderBufferOdd,
                                            scene_rdl2::fb_util::FloatBuffer *weightRenderBufferOdd,
                                            scene_rdl2::fb_util::ActivePixels &activePixelsRenderBufferOdd,
                                            bool parallel) const
//
// Snapshots the contents of the renderBufferOdd/weightRenderBufferOdd w/ ActivePixelsRenderBufferOdd information
// for ProgressiveFrame message related logic. So renderBufferOdd is not normalized by weight yet.
// no resize, no extrapolation and no untiling logic is handled internally.
// Just create snapshot data with properly constructed activePixelsRenderBufferOdd based on
// difference between current and previous renderBufferOdd and weightRenderBufferOdd.
// renderBufferOdd/weightRenderBufferOdd is tiled format and renderBufferOdd is not normalized by weight.
//
{
    mDriver->snapshotDeltaRenderBufferOdd(renderBufferOdd,
                                          weightRenderBufferOdd,
                                          activePixelsRenderBufferOdd,
                                          parallel);
}

void
RenderContext::snapshotDeltaPixelInfo(scene_rdl2::fb_util::PixelInfoBuffer *pixelInfoBuffer,
                                      scene_rdl2::fb_util::FloatBuffer *pixelInfoWeightBuffer,
                                      scene_rdl2::fb_util::ActivePixels &activePixelsPixelInfo,
                                      bool parallel) const
//
// Snapshots the contents of the pixelInfoBuffer/pixelInfoWeightBuffer w/
// ActivePixelsPixelInfo information for ProgressiveFrame message related logic.
// Just create snapshot data with properly constructed activePixels based on
// difference between current and previous pixelInfoBuffer and pixelInfoWeightBuffer.
// pixelInfoBuffer/pixelInfoWeightBuffer are tiled format.
//
{
    mDriver->snapshotDeltaPixelInfo(pixelInfoBuffer, pixelInfoWeightBuffer,
                                    activePixelsPixelInfo,
                                    parallel);
}

void
RenderContext::snapshotDeltaHeatMap(scene_rdl2::fb_util::HeatMapBuffer *heatMapBuffer,
                                    scene_rdl2::fb_util::FloatBuffer *heatMapWeightBuffer,
                                    scene_rdl2::fb_util::ActivePixels &activePixelsHeatMap,
                                    scene_rdl2::fb_util::FloatBuffer *heatMapSecBuffer,
                                    bool parallel) const
//
// Snapshots the contents of the heatMapBuffer/heatMapWeightBuffer w/ ActivePixelsHeatMap information
// for ProgressiveFrame message related logic.
// Just create snapshot data with properly constructed activePixels based on
// difference between current and previous heatMapBuffer/heatMapWeightBuffer.
// Also create heatMapSecBuffer just for active pixels only.
// heatMapBuffer/heatMapWeightBuffer/heatMapSecBuffer are tiled format.
//
{
    mDriver->snapshotDeltaHeatMap(heatMapBuffer,
                                  heatMapWeightBuffer,
                                  activePixelsHeatMap,
                                  heatMapSecBuffer,
                                  parallel);
}

void
RenderContext::snapshotDeltaWeightBuffer(scene_rdl2::fb_util::FloatBuffer *weightBuffer,
                                         scene_rdl2::fb_util::ActivePixels &activePixelsWeightBuffer,
                                         bool parallel) const
//
// Snapshots the contents of the weightBuffer w/ ActivePixelsWeightBuffer information
// for ProgressiveFrame message related logic.
// Just create snapshot data with properly constructed activePixels based on
// difference between current and previous weightBuffer.
// weightBuffer is tiled format.
//
{
    mDriver->snapshotDeltaWeightBuffer(weightBuffer,
                                       activePixelsWeightBuffer,
                                       parallel);
}

void
RenderContext::snapshotDeltaRenderOutput(unsigned int rodIndex,
                                         scene_rdl2::fb_util::VariablePixelBuffer *renderOutputBuffer,
                                         scene_rdl2::fb_util::FloatBuffer *renderOutputWeightBuffer,
                                         scene_rdl2::fb_util::ActivePixels &activePixelsRenderOutput,
                                         bool parallel,
                                         bool& denoiserAlbedoInput,
                                         bool& denoiserNormalInput) const
//
// Snapshots the contents of the renderOutputBuffer(rodIndex)/renderOutputWeightBuffer(rodIndex) w/
// ActivePixelsRenderOutput information for ProgressiveFrame message related logic.
// Just create snapshot data with properly constructed activePixels based on
// difference between current and previous renderOutputBuffer(rodIndex) and renderOutputWeightBuffer(rodIndex).
// renderOutputBuffer(rodIndex) and renderOutputWeightBuffer(rodIndex) are tiled format.
//
{
    // AOV
    switchAovType(*mRenderOutputDriver,
                  rodIndex,
                  [](const scene_rdl2::rdl2::RenderOutput * /*ro*/) {},  // non active AOV
                  [&](const int aovIdx) { // Visibility AOV
                      mDriver->snapshotDeltaAovVisibility(aovIdx,
                                                          renderOutputBuffer,
                                                          renderOutputWeightBuffer,
                                                          activePixelsRenderOutput,
                                                          parallel);
                  },
                  [&](const int aovIdx) { // regular AOV
                      mDriver->snapshotDeltaAov(aovIdx,
                                                renderOutputBuffer,
                                                renderOutputWeightBuffer,
                                                activePixelsRenderOutput,
                                                parallel);
                  });

    // DisplayFilter
    const int dfIdx = mRenderOutputDriver->getDisplayFilterIndex(rodIndex);
    if (dfIdx >= 0) {
        mDriver->snapshotDeltaDisplayFilter(dfIdx,
                                            renderOutputBuffer,
                                            renderOutputWeightBuffer,
                                            activePixelsRenderOutput,
                                            parallel);
    }

    // Denoiser related info
    denoiserAlbedoInput = false;
    denoiserNormalInput = false;
    if (mRenderOutputDriver->getDenoiserAlbedoInput() == rodIndex) {
        denoiserAlbedoInput = true;
    } else if (mRenderOutputDriver->getDenoiserNormalInput() == rodIndex) {
        denoiserNormalInput = true;                          
    }
}

void
RenderContext::snapshotPixelInfoBuffer(scene_rdl2::fb_util::PixelInfoBuffer *pixelInfoBuffer, bool untile, bool parallel) const
{
    // Request a snapshot of the pixel info buffer from the render driver.
    mDriver->snapshotPixelInfoBuffer(MNRY_VERIFY(pixelInfoBuffer), untile, parallel);
}

void
RenderContext::snapshotHeatMapBuffer(scene_rdl2::fb_util::HeatMapBuffer *heatMapBuffer, bool untile, bool parallel) const
{
    // Request a snapshot of the heat map buffer from the render driver.
    mDriver->snapshotHeatMapBuffer(MNRY_VERIFY(heatMapBuffer), untile, parallel);
}

void
RenderContext::snapshotVisibilityBuffer(scene_rdl2::fb_util::VariablePixelBuffer *visibilityBuffer,
                                        unsigned int aov, bool untile, bool parallel) const
{
    // Request a snapshot of the visibility aov buffer from the render driver.
    bool fulldumpVisibility = mSceneContext->getResumableOutput() || mSceneContext->getResumeRender();
    mDriver->snapshotVisibilityBuffer(visibilityBuffer, aov, untile, parallel, fulldumpVisibility);
}

void
RenderContext::snapshotAovBuffer(scene_rdl2::fb_util::VariablePixelBuffer *aovBuffer, unsigned int aov,
                                 bool untile, bool parallel) const
//
// This API only processed about non visibility related AOV (i.e. non visibility and also non variance+visibility).
// See RenderContext::snapshotAovBuffers() as well.
//
{
    unsigned numConsistentSamples = getNumConsistentSamples();
    bool fulldump = mSceneContext->getResumableOutput() || mSceneContext->getResumeRender();
    mDriver->snapshotAovBuffer(aovBuffer, numConsistentSamples, aov, untile, parallel, fulldump);
}

void
RenderContext::snapshotAovBuffer(scene_rdl2::fb_util::RenderBuffer *renderBuffer, unsigned int aov,
                                 bool untile, bool parallel) const
{
    unsigned numConsistentSamples = getNumConsistentSamples();
    mDriver->snapshotAovBuffer(renderBuffer, numConsistentSamples, aov, untile, parallel);
}

void
RenderContext::snapshotAovBuffers(std::vector<scene_rdl2::fb_util::VariablePixelBuffer> &aovBuffers,
                                  bool untile, bool parallel) const
{
    aovBuffers.resize(mDriver->getFilm().getNumAovs());
    crawlAllRenderOutput(*mRenderOutputDriver,
                         [](const scene_rdl2::rdl2::RenderOutput */*ro*/) {}, // non active AOV
                         [&](const int aovIdx) { // Visibility AOV
                             snapshotVisibilityBuffer(&aovBuffers[aovIdx], aovIdx, untile, parallel);
                         },
                         [&](const int aovIdx) { // regular AOV
                             snapshotAovBuffer(&aovBuffers[aovIdx], aovIdx, untile, parallel);
                         });
}

void
RenderContext::runDisplayFiltersBatch() const
{
    const DisplayFilterDriver& displayFilterDriver = mDriver->getDisplayFilterDriver();
    if (!displayFilterDriver.hasDisplayFilters()) {
        return;
    }
    snapshotAovsForDisplayFilters(true, true);
    simpleLoop (/*parallel*/ true, 0u, (unsigned)mDriver->getTiles()->size(), [&](unsigned tileIdx) {
        int threadId = tbb::task_arena::current_thread_index();
        displayFilterDriver.runDisplayFilters(tileIdx, threadId);
    });
}

void
RenderContext::snapshotAovsForDisplayFilters(bool untile, bool parallel) const
{
    crawlAllRenderOutput(*mRenderOutputDriver,
                        [](const scene_rdl2::rdl2::RenderOutput * /*ro*/) {}, // non active AOV
                        [&](const int aovIdx) { // Visibility AOV
                            if (mDriver->getDisplayFilterDriver().isAovRequired(aovIdx)) {
                                snapshotVisibilityBuffer(mDriver->getDisplayFilterDriver().getAovBuffer(aovIdx),
                                                         aovIdx, untile, parallel);
                            }
                        },
                        [&](const int aovIdx) { // regular AOV
                            if (mDriver->getDisplayFilterDriver().isAovRequired(aovIdx)) {
                                snapshotAovBuffer(mDriver->getDisplayFilterDriver().getAovBuffer(aovIdx),
                                                  aovIdx, untile, parallel);
                            }
                        });
}

void
RenderContext::snapshotDisplayFilterBuffer(scene_rdl2::fb_util::VariablePixelBuffer *displayFilterBuffer, unsigned int dfIdx,
                                           bool untile, bool parallel) const
{
    mDriver->snapshotDisplayFilterBuffer(displayFilterBuffer, dfIdx, untile, parallel);
}

void
RenderContext::snapshotDisplayFilterBuffers(std::vector<scene_rdl2::fb_util::VariablePixelBuffer> &displayFilterBuffers,
                                            bool untile, bool parallel) const
{
    // This is called in batch mode and checkpoint mode, so we can run the display filter driver here.
    runDisplayFiltersBatch();
    const unsigned int displayFilterCount = mDriver->getFilm().getDisplayFilterCount();
    displayFilterBuffers.resize(displayFilterCount);
    for (unsigned int dfIdx = 0; dfIdx < displayFilterCount; ++dfIdx) {
        snapshotDisplayFilterBuffer(&displayFilterBuffers[dfIdx], dfIdx, untile, parallel);
    }
}

void
RenderContext::snapshotRenderOutput(scene_rdl2::fb_util::VariablePixelBuffer *buffer, int roIndx,
                                    const scene_rdl2::fb_util::RenderBuffer *renderBuffer,
                                    const scene_rdl2::fb_util::HeatMapBuffer *heatMapBuffer,
                                    const scene_rdl2::fb_util::FloatBuffer *weightBuffer,
                                    const scene_rdl2::fb_util::RenderBuffer *renderBufferOdd,
                                    bool untile, bool parallel) const
{
    // First the odd balls
    // assume that renderBuffer, and heatMapBuffer, if
    // required have already been snapshot (with the proper untile)
    // for the appropriate film.
    if (mRenderOutputDriver->requiresRenderBuffer(roIndx) ||
        mRenderOutputDriver->requiresHeatMap(roIndx) ||
        mRenderOutputDriver->requiresWeightBuffer(roIndx) ||
        mRenderOutputDriver->requiresRenderBufferOdd(roIndx)) {
        mRenderOutputDriver->finishSnapshot(buffer, roIndx,
                                            renderBuffer, heatMapBuffer, weightBuffer, renderBufferOdd,
                                            parallel);
        return;
    }

    if (mRenderOutputDriver->requiresDisplayFilter(roIndx)) {
        const int dfIdx = mRenderOutputDriver->getDisplayFilterIndex(roIndx);
        snapshotDisplayFilterBuffer(buffer, dfIdx, untile, parallel);
        return;
    }

    const pbr::AovSchema& schema = mRenderOutputDriver->getAovSchema();
    const int aovIdx = mRenderOutputDriver->getAovBuffer(roIndx);
    if (aovIdx < 0) {
        return;
    }

    if (mRenderOutputDriver->isVisibilityAov(aovIdx)) {
        snapshotVisibilityBuffer(buffer, aovIdx, untile, parallel);
        return;
    }

    // All the rest come from the aov buffers
    snapshotAovBuffer(buffer, aovIdx, untile, parallel);
}

void
RenderContext::snapshotWeightBuffer(scene_rdl2::fb_util::VariablePixelBuffer *outputBuffer, bool untile, bool parallel) const
{
    // This API is used for debug purpose. See rndr/gui/RenderGui.cc RenderGui::snapshotFrame().
    mDriver->snapshotWeightBuffer(MNRY_VERIFY(outputBuffer), untile, parallel);
}

void
RenderContext::snapshotWeightBuffer(scene_rdl2::fb_util::FloatBuffer *weightBuffer, bool untile, bool parallel) const
{
    // This API is used for snapshot of weightAOV related logic.
    // We only do snapshot if renderOutputDriver has weightAOV
    if (getRenderOutputDriver()->requiresWeightBuffer()) {
        mDriver->snapshotWeightBuffer(MNRY_VERIFY(weightBuffer), untile, parallel);
    }
}

bool
RenderContext::hasPixelInfoBuffer() const
{
    return mDriver->getFilm().hasPixelInfoBuffer();
}

bool
RenderContext::areCoarsePassesComplete() const
{
    return mDriver->areCoarsePassesComplete();
}

bool
RenderContext::isFrameComplete() const
{
    return mDriver->isFrameComplete();
}

bool
RenderContext::isFrameCompleteAtPassBoundary() const
{
    return mDriver->isFrameCompleteAtPassBoundary();
}

bool
RenderContext::isFrameReadyForDisplay() const
{
    return mDriver->isReadyForDisplay();
}

float
RenderContext::getFrameProgressFraction(std::size_t* submitted, std::size_t* total) const
{
    bool activeRendering = isFrameRendering() && !isFrameComplete();

    return mDriver->getOverallProgressFraction(activeRendering, submitted, total);
}

void
RenderContext::setMultiMachineGlobalProgressFraction(float fraction)
{
    mMultiMachineGlobalProgressFraction = fraction;
    if (mDriver) {
        mDriver->setMultiMachineGlobalProgressFraction(fraction);
    }
}

float
RenderContext::getMultiMachineGlobalProgressFraction() const
{
    return mMultiMachineGlobalProgressFraction;
}

RenderProgressEstimation *
RenderContext::getFrameProgressEstimation() const
{
    MNRY_ASSERT(mDriver);
    return &(mDriver->getRenderProgressEstimation());
}

const std::vector<scene_rdl2::fb_util::Tile> *
RenderContext::getTiles() const
{
    return mDriver->getTiles();
}

void
RenderContext::getTilesRenderedTo(scene_rdl2::util::BitArray& tiles) const
{
    // Merge all thread local bit arrays of tiles rendered to into a single
    // master bit array of tiles.
    tiles.clearAll();
    pbr::forEachTLS([&](pbr::TLState *tls) {
        tiles.bitwiseOr(tls->mTilesRenderedTo);
        tls->mTilesRenderedTo.clearAll();
    });
}

void
RenderContext::setForceCallStartFrame()
{
    std::lock_guard<std::mutex> lock(mMutexForceCallStartFrame);
    mForceCallStartFrame = true;
}

void
RenderContext::forceGuiCallStartFrameIfNeed()
{
    auto checkForceCallStartFrame = [&]() -> bool {
        std::lock_guard<std::mutex> lock(mMutexForceCallStartFrame);
        if (mForceCallStartFrame) {
            mForceCallStartFrame = false;
            return true;
        }
        return false;
    };

    if (checkForceCallStartFrame()) {
        startFrame();
    }
}

bool
RenderContext::needToSetExecTrackerMsgHandlerCallBack() const
{
    if (getExecTrackerMsgHandlerCallBack() &&
        (mGeometryManager && mGeometryManager->getGeometryManagerExecTracker().getMsgHandlerCallBack())) {
        return false;           // already setup msgHandlerCallBack
    }
    return true;
}

RenderContext::MsgHandler
RenderContext::getExecTrackerMsgHandlerCallBack() const
{
    return mRenderPrepExecTracker.getMsgHandlerCallBack();
}

void
RenderContext::setExecTrackerMsgHandlerCallBack(const MsgHandler &msgHandler)
{
    mRenderPrepExecTracker.setMsgHandlerCallBack(msgHandler);

    if (mGeometryManager) {
        rt::GeometryManagerExecTracker &geometryManagerExecTracker =
            mGeometryManager->getGeometryManagerExecTracker();
        geometryManagerExecTracker.setMsgHandlerCallBack(msgHandler);
    }
}

std::string
RenderContext::execTrackerCancelInfoEncode() const
//
// This function encodes all cancelInfo parameters to byte memory (as std::string)
// This is used by progmcrt_computation debug console command.
// All encoded information will be decoded and setup by execTrackerCancelInfoDecode()
//    
{
    std::string data;
    scene_rdl2::rdl2::ValueContainerEnq vcEnq(&data);
    vcEnq.enqString(mRenderPrepExecTracker.cancelInfoEncode());
    if (mGeometryManager) {
        rt::GeometryManagerExecTracker &geometryManagerExecTracker =
            mGeometryManager->getGeometryManagerExecTracker();

        vcEnq.enqBool(true);
        vcEnq.enqString(geometryManagerExecTracker.cancelInfoEncode());
    } else {
        vcEnq.enqBool(false);
    }
    vcEnq.finalize();
    return data;
}

void    
RenderContext::execTrackerCancelInfoDecode(const std::string &data)
//
// This is used by progmcrt_computation debug console command.
// Encoded data was created by execTrackerCancelInfoEncode()
//    
{
    rt::GeometryManagerExecTracker &geometryManagerExecTracker =
        mGeometryManager->getGeometryManagerExecTracker();

    scene_rdl2::rdl2::ValueContainerDeq vcDeq(data.data(), data.size());
    mRenderPrepExecTracker.cancelInfoDecode(vcDeq.deqString());
    if (vcDeq.deqBool()) {
        geometryManagerExecTracker.cancelInfoDecode(vcDeq.deqString());
    }
}

std::string
RenderContext::getOiioStats(int level) const
{
    return texture::getTextureSampler()->showStats(level, true);
}

unsigned
RenderContext::getFilmActivity() const
{
    const moonray::rndr::Film &film = rndr::getRenderDriver()->getFilm();
    return film.getFilmActivity();
}

unsigned
RenderContext::getNumConsistentSamples() const
{
    const scene_rdl2::rdl2::SceneVariables &vars = mSceneContext->getSceneVariables();

    if (getSamplingMode() == SamplingMode::ADAPTIVE) {
        return (unsigned)std::max(2, vars.get(scene_rdl2::rdl2::SceneVariables::sMinAdaptiveSamples));
    } else {
        unsigned pixelSamplesSqrt = static_cast<unsigned>(std::max(1, vars.get(scene_rdl2::rdl2::SceneVariables::sPixelSamplesSqrt)));
        return pixelSamplesSqrt * pixelSamplesSqrt;
    }
}

RenderContext::RP_RESULT
RenderContext::renderPrep(ExecutionMode executionMode, bool allowUnsupportedXPUFeatures)
{
    if (mRenderPrepExecTracker.startRenderPrep() == RenderPrepExecTracker::RESULT::CANCELED) {
        return RP_RESULT::CANCELED;
    }

    mRenderPrepTimingStats->recTimeStart();

    startUpdatePhaseOfFrame();

    mRenderPrepTimingStats->recTime(RenderPrepTimingStats::RenderPrepTag::START_UPDATE_PHASE_OF_FRAME);

    mRenderStats->logInfoEmptyLine();
    mRenderStats->startRenderPrep();

    mRenderPrepTimingStats->recTime(RenderPrepTimingStats::RenderPrepTag::START_RENDERPREP);

    mGeometryManager->resetStatistics();

    mRenderPrepTimingStats->recTime(RenderPrepTimingStats::RenderPrepTag::RESET_STATISTICS);

    // Setup the render output driver
    mRenderOutputDriver.reset(new RenderOutputDriver(this));

    rt::ChangeFlag geomChangeFlag =
        rt::ChangeFlag::NONE;

    mRenderPrepTimingStats->recTime(RenderPrepTimingStats::RenderPrepTag::RESET_RENDER_OUTPUT_DRIVER);

    bool loadAllGeometries = false;

    if (mFirstFrame) {
        mRenderStats->logString(
            "---------- Render Prep -----------------------------------");
        // Clear stats and logs on each frame
        if (mRenderPrepExecTracker.startApplyUpdate() == RenderPrepExecTracker::RESULT::CANCELED) {
            return RP_RESULT::CANCELED;
        }
        resetShaderStatsAndLogs();  // initialize EventLog to allocate memory for messages
        
        // Call update() on all SceneObjects
        mSceneContext->applyUpdates(mLayer);
        // Add MeshLight geometry to a MeshLightLayer and assign each geometry a material, 
        // which will keep track of the MeshLight's attribute table
        createMeshLightLayer();
        {
            scene_rdl2::rdl2::Layer::RootShaderSet rootShaders;
            mLayer->getAllRootShaders(rootShaders);
            buildPrimitiveAttributeTables(rootShaders);

            scene_rdl2::rdl2::Layer::RootShaderSet meshLightRootShaders;
            mMeshLightLayer->getAllRootShaders(meshLightRootShaders);
            buildPrimitiveAttributeTables(meshLightRootShaders);
        }
        geomChangeFlag = rt::ChangeFlag::ALL;
        if (mRenderPrepExecTracker.endApplyUpdate() == RenderPrepExecTracker::RESULT::CANCELED) {
            return RP_RESULT::CANCELED;
        }
        mFirstFrame = false;
    } else if (mSceneUpdated) {
        // Call update() on all SceneObjects. Flag the shaders in the layer that have been updated and save to 
        // mChangedRootShaders. We will use these changed shaders to build the attribute tables below. Also flag the 
        // associated geometry so that we can update it in loadGeometries.
        mSceneContext->applyUpdates(mLayer);
        buildPrimitiveAttributeTables(mLayer->getChangedRootShaders());

        // This call does everything that applyUpdates does, EXCEPT it does not update SceneObjects, as that has already
        // been done. We just need to know what MeshLight attribute tables to build and geometries to load. 
        mSceneContext->applyUpdatesToMeshLightLayer(mMeshLightLayer);
        buildPrimitiveAttributeTables(mMeshLightLayer->getChangedRootShaders());

        const scene_rdl2::rdl2::SceneVariables & sv = mSceneContext->getSceneVariables();

        // Check for the need to reload all procedurals for changes in frame
        // or motion blur. If the applicationMode is set to MOTIONCAPTURE
        // the program will not load geometry from disk each time
        // the frame ID changes
        bool frameChanged = (sv.hasChanged(scene_rdl2::rdl2::SceneVariables::sFrameKey) &&
            mOptions.getApplicationMode() != ApplicationMode::MOTIONCAPTURE);

        // Our camera could have changed.
        // A camera change requires a geometry rebuild, our definition of
        // render space depends on this.  So changing the camera doesn't seem
        // like a particularly good work flow.
        const std::vector<const scene_rdl2::rdl2::Camera *> activeCameras = mSceneContext->getActiveCameras();
        const bool cameraChanged = sv.hasChanged(scene_rdl2::rdl2::SceneVariables::sCamera);
        if (cameraChanged) {
            initActiveCamera(activeCameras[0]);
            mPbrScene->updateActiveCamera(mCamera);
        }

        bool motionBlurChanged = sv.hasChanged(scene_rdl2::rdl2::SceneVariables::sEnableMotionBlur) ||
            (sv.get(scene_rdl2::rdl2::SceneVariables::sEnableMotionBlur) &&
            (sv.hasChanged(scene_rdl2::rdl2::SceneVariables::sMotionSteps) ||
            mCamera->hasChanged(scene_rdl2::rdl2::Camera::sMbShutterOpenKey) ||
            mCamera->hasChanged(scene_rdl2::rdl2::Camera::sMbShutterCloseKey)));
        // note: the camera specifies shutter open/close times, so we only
        // need to check its attr

        bool globalToggleChanged = sv.hasChanged(scene_rdl2::rdl2::SceneVariables::sEnableDisplacement) ||
            sv.hasChanged(scene_rdl2::rdl2::SceneVariables::sEnableMaxGeomResolution) ||
            sv.hasChanged(scene_rdl2::rdl2::SceneVariables::sMaxGeomResolution);

        loadAllGeometries = frameChanged || motionBlurChanged || globalToggleChanged || cameraChanged;

        geomChangeFlag = loadAllGeometries ?
            rt::ChangeFlag::ALL :
            rt::ChangeFlag::UPDATE;
    }

    mRenderPrepTimingStats->recTime(RenderPrepTimingStats::RenderPrepTag::FLAG_STATUS_UPDATE);

    // Clear stats and logs on each frame
    resetShaderStatsAndLogs();

    mRenderPrepTimingStats->recTime(RenderPrepTimingStats::RenderPrepTag::RESET_SHADER_STATS_AND_LOGS);
    mRenderPrepTimingStats->recTime(RenderPrepTimingStats::RenderPrepTag::LOAD_GEOMETRIES, 0.0);
    mRenderPrepTimingStats->recTime(RenderPrepTimingStats::RenderPrepTag::REPORT_GEOMETRY_MEMORY, 0.0);

    RP_RESULT execResult = RP_RESULT::FINISHED;
    if (geomChangeFlag != rt::ChangeFlag::NONE) {
        execResult = loadGeometries(geomChangeFlag);

        mRenderPrepTimingStats->recTime(RenderPrepTimingStats::RenderPrepTag::LOAD_GEOMETRIES);

        // Report memory footprint for geometry primitives.
        if (mOptions.getApplicationMode() != ApplicationMode::MOTIONCAPTURE) {
            if (mRenderStats->getLogInfo() ||
                mRenderStats->getLogCsv()  ||
                mRenderStats->getLogAthena()) {
                reportGeometryTessellationTime();
                reportGeometryMemory();
            }
        }

        mRenderPrepTimingStats->recTime(RenderPrepTimingStats::RenderPrepTag::REPORT_GEOMETRY_MEMORY);
    }

    // setup primitive attribute aov flags in the materials
    buildMaterialAovFlags();

    mRenderPrepTimingStats->recTime(RenderPrepTimingStats::RenderPrepTag::BUILD_MATERIAL_AOV_FLAGS);

    // build geometry object extensions
    buildGeometryExtensions();

    mRenderPrepTimingStats->recTime(RenderPrepTimingStats::RenderPrepTag::BUILD_GEOMETRY_EXTENSIONS);

    // Clear stats and logs on each frame
    resetShaderStatsAndLogs();

    mRenderPrepTimingStats->recTime(RenderPrepTimingStats::RenderPrepTag::RESET_SHADER_STATS_AND_LOGS2);

    // Reset statistics for this frame
    mPbrStatistics->reset();
    pbr::forEachTLS([](pbr::TLState *tls){ tls->mStatistics.reset(); });
    mGeomStatistics->reset();
    geom::internal::forEachTLS([](geom::internal::TLState *tls) {
        tls->mStatistics.reset();
    });

    mRenderPrepTimingStats->recTime(RenderPrepTimingStats::RenderPrepTag::PBR_STATISTICS_RESET);

    // Update PBR
    RenderTimer timer(mRenderStats->mLoadPbrTime);
    mPbrScene->preFrame(mRenderOutputDriver->getLightAovs(), executionMode, *mGeometryManager, loadAllGeometries);

    mRenderPrepTimingStats->recTime(RenderPrepTimingStats::RenderPrepTag::UPDATE_PBR);
    mRenderPrepTimingStats->recTimeEnd(RenderPrepTimingStats::RenderPrepTag::WHOLE);

    // Update XPU
    if (executionMode == mcrt_common::ExecutionMode::XPU) {
        if (geomChangeFlag == rt::ChangeFlag::ALL) {
            // XPU doesn't support BVH updates.  Also, moving the camera in moonray_gui
            // is a rt::ChangeFlag::UPDATE and we don't want to rebuild the GPU
            // data for that case.
            mGeometryManager->updateGPUAccelerator(allowUnsupportedXPUFeatures, mLayer);
        }
    }
    mRenderStats->mBuildGPUAcceleratorTime =
        mGeometryManager->getStatistics().mBuildGPUAcceleratorTime;

    if (mRenderPrepExecTracker.endRenderPrep() == RenderPrepExecTracker::RESULT::CANCELED) {
        return RP_RESULT::CANCELED;
    }

    return execResult;
}

void
RenderContext::loadScene(std::stringstream &initMessages)
{
    RenderTimer timer(mRenderStats->mLoadSceneTime);

    // Create an ASCII and binary reader which will both read into the same
    // SceneContext (though only one will be actively reading at a time). We
    // want to reuse these readers so that they maintain any state between
    // reading subsequent files.
    scene_rdl2::rdl2::AsciiReader asciiReader(*mSceneContext);
    scene_rdl2::rdl2::BinaryReader binaryReader(*mSceneContext);

    // Apply Lua globals before loading the scene.
    const auto& rdlaGlobals = mOptions.getRdlaGlobals();
    for (const auto& global : rdlaGlobals) {
        asciiReader.fromString(
                scene_rdl2::util::buildString(global.mVar, " = ", global.mExpression),
                "@rdla_set");
    }

    // Load each scene file in order.
    const auto& sceneFiles = mOptions.getSceneFiles();
    for (const auto& sceneFile : sceneFiles) {
        if (!sceneFile.empty()) {
            mRenderStats->logLoadingScene(initMessages, sceneFile);
            // Grab the file extension and convert it to lower case.
            auto ext = scene_rdl2::util::lowerCaseExtension(sceneFile);
            if (ext.empty()) {
                throw scene_rdl2::except::RuntimeError(scene_rdl2::util::buildString(
                        "File '", sceneFile, "' has no extension."
                        " Cannot determine file type."));
            }

            if (ext == "rdla") {
                asciiReader.fromFile(sceneFile);
            } else if (ext == "rdlb") {
                binaryReader.fromFile(sceneFile);
            } else {
                throw scene_rdl2::except::RuntimeError(scene_rdl2::util::buildString(
                        "File '", sceneFile, "' has an unknown extension."
                        " Cannot determine file type."));
            }
        }
    }

    // Now we've loaded our scene.  Apply any queued updates
    mSceneLoaded = true;

    // Apply any queued updates
    for (const auto& updateSceneData: mUpdateQueue) {
        // Two update paths depending on what was queued
        if (updateSceneData.first == FILE_TOKEN) {
            updateScene(updateSceneData.second);
        } else {
            updateScene(updateSceneData.first, updateSceneData.second);
        }
    }
    mUpdateQueue.clear();

    // Set up the SceneVariables as a fallback for RenderOptions.
    mOptions.setSceneContext(mSceneContext.get());

    // Apply any attribute overrides or scene variable overrides specified in
    // the RenderOptions.
    applyAttributeOverrides(*mSceneContext, *mRenderStats, mOptions, initMessages);

    mRenderStats->logLoadingSceneReadDiskIO(initMessages);
}

void
RenderContext::createPbrScene()
{
    RenderTimer timer(mRenderStats->mLoadPbrTime);

    MNRY_ASSERT(!mPbrScene, "Trying to load PBR library but the library is already loaded.");
    MNRY_ASSERT(mCamera, "Must have a valid camera to load PBR library.");

    const scene_rdl2::rdl2::SceneContext* constSceneContext = mSceneContext.get();
    const scene_rdl2::rdl2::Layer* constLayer = mLayer;
    const int hardcodedLobeCount = 2;

    mPbrScene.reset(new pbr::Scene(constSceneContext, constLayer,
            hardcodedLobeCount));
    mSceneContext->setRender2World(&mPbrScene->getRender2World());

    mIntegrator.reset(new pbr::PathIntegrator);

    mPbrScene->setEmbreeAccelerator(mGeometryManager->getEmbreeAccelerator());
}

void
RenderContext::updatePbrState(const FrameState &fs)
{
    MNRY_ASSERT(fs.mScene);

    const scene_rdl2::rdl2::SceneVariables &vars = mSceneContext->getSceneVariables();

    pbr::PathIntegratorParams integratorParams;
    integratorParams.mIntegratorPixelSamplesSqrt               = vars.get(scene_rdl2::rdl2::SceneVariables::sPixelSamplesSqrt);
    integratorParams.mIntegratorLightSamplesSqrt               = vars.get(scene_rdl2::rdl2::SceneVariables::sLightSamplesSqrt);
    integratorParams.mIntegratorBsdfSamplesSqrt                = vars.get(scene_rdl2::rdl2::SceneVariables::sBsdfSamplesSqrt);
    integratorParams.mIntegratorBssrdfSamplesSqrt              = vars.get(scene_rdl2::rdl2::SceneVariables::sBssrdfSamplesSqrt);
    integratorParams.mIntegratorMaxDepth                       = vars.get(scene_rdl2::rdl2::SceneVariables::sMaxDepth);
    integratorParams.mIntegratorMaxDiffuseDepth                = vars.get(scene_rdl2::rdl2::SceneVariables::sMaxDiffuseDepth);
    integratorParams.mIntegratorMaxGlossyDepth                 = vars.get(scene_rdl2::rdl2::SceneVariables::sMaxGlossyDepth);
    integratorParams.mIntegratorMaxMirrorDepth                 = vars.get(scene_rdl2::rdl2::SceneVariables::sMaxMirrorDepth);
    integratorParams.mIntegratorMaxVolumeDepth                 = vars.get(scene_rdl2::rdl2::SceneVariables::sMaxVolumeDepth);
    integratorParams.mIntegratorMaxPresenceDepth               = vars.get(scene_rdl2::rdl2::SceneVariables::sMaxPresenceDepth);
    integratorParams.mIntegratorMaxHairDepth                   = vars.get(scene_rdl2::rdl2::SceneVariables::sMaxHairDepth);
    integratorParams.mIntegratorMaxSubsurfacePerPath           = vars.get(scene_rdl2::rdl2::SceneVariables::sMaxSubsurfacePerPath);
    integratorParams.mIntegratorTransparencyThreshold          = vars.get(scene_rdl2::rdl2::SceneVariables::sTransparencyThreshold);
    integratorParams.mIntegratorPresenceThreshold              = vars.get(scene_rdl2::rdl2::SceneVariables::sPresenceThreshold);
    integratorParams.mIntegratorRussianRouletteThreshold       = vars.get(scene_rdl2::rdl2::SceneVariables::sRussianRouletteThreshold);
    integratorParams.mSampleClampingValue                      = vars.get(scene_rdl2::rdl2::SceneVariables::sSampleClampingValue);
    integratorParams.mSampleClampingDepth                      = vars.get(scene_rdl2::rdl2::SceneVariables::sSampleClampingDepth);
    integratorParams.mRoughnessClampingFactor                  = vars.get(scene_rdl2::rdl2::SceneVariables::sRoughnessClampingFactor);
    integratorParams.mIntegratorVolumeQuality                  = vars.get(scene_rdl2::rdl2::SceneVariables::sVolumeQuality);
    integratorParams.mIntegratorVolumeShadowQuality            = vars.get(scene_rdl2::rdl2::SceneVariables::sVolumeShadowQuality);
    integratorParams.mIntegratorVolumeIlluminationSamples      = vars.get(scene_rdl2::rdl2::SceneVariables::sVolumeIlluminationSamples);
    // for user it's more intuitive to use opacity as volume tweaking parameter,
    // for renderer transmittance (1 - opacity) is more commonly used during
    // volume calculation
    integratorParams.mIntegratorVolumeTransmittanceThreshold   = clamp(
        1.0f - vars.get(scene_rdl2::rdl2::SceneVariables::sVolumeOpacityThreshold));
    integratorParams.mIntegratorVolumeAttenuationFactor        = clamp(
        vars.get(scene_rdl2::rdl2::SceneVariables::sVolumeAttenuationFactor));
    integratorParams.mIntegratorVolumeContributionFactor       = clamp(
        vars.get(scene_rdl2::rdl2::SceneVariables::sVolumeContributionFactor));
    integratorParams.mIntegratorVolumePhaseAttenuationFactor   = clamp(
        vars.get(scene_rdl2::rdl2::SceneVariables::sVolumePhaseAttenuationFactor));
    MNRY_ASSERT(vars.get(scene_rdl2::rdl2::SceneVariables::sVolumeOverlapMode) >= 0 &&
        vars.get(scene_rdl2::rdl2::SceneVariables::sVolumeOverlapMode) <
        static_cast<int>(pbr::VolumeOverlapMode::NUM_MODES));
    integratorParams.mIntegratorVolumeOverlapMode =
        static_cast<pbr::VolumeOverlapMode>(vars.get(scene_rdl2::rdl2::SceneVariables::sVolumeOverlapMode));

    mIntegrator->update(fs, integratorParams);
}

void
RenderContext::buildPrimitiveAttributeTables(
    const scene_rdl2::rdl2::Layer::RootShaderSet& rootShaders)
{
    // This should run after Shaders have had all their attributes bound
    // or the attribute tables will be incomplete
    // This should run before loading procedurals or the procedurals
    // will not know which primitive attributes to load

    RenderTimer timer(mRenderStats->mBuildPrimAttrTableTime);
    for (scene_rdl2::rdl2::RootShader * const s : rootShaders) {
        moonray::shading::AttributeKeySet requiredKeys;
        moonray::shading::AttributeKeySet optionalKeys;

        // Always add the following attributes for explicit shading via instancing
        optionalKeys.insert(shading::StandardAttributes::sNormal);
        optionalKeys.insert(shading::StandardAttributes::sdPds);
        optionalKeys.insert(shading::StandardAttributes::sdPdt);
        optionalKeys.insert(shading::StandardAttributes::sUv);
        optionalKeys.insert(shading::StandardAttributes::sExplicitShading);

        scene_rdl2::rdl2::ConstSceneObjectSet b;
        s->getBindingTransitiveClosure(b);
        for (const scene_rdl2::rdl2::SceneObject * const o : b) {
            if (o->isA<scene_rdl2::rdl2::Shader>()) {
                const auto& reqKeys =
                    o->asA<scene_rdl2::rdl2::Shader>()->getRequiredAttributes();
                requiredKeys.insert(reqKeys.begin(), reqKeys.end());
                const auto& optKeys =
                    o->asA<scene_rdl2::rdl2::Shader>()->getOptionalAttributes();
                optionalKeys.insert(optKeys.begin(), optKeys.end());
            }
        }

        if (s->isA<scene_rdl2::rdl2::Material>()) {
            // The render output driver might itself require certain attributes.
            // Materials need to know what those attributes are so the
            // intersection has access to them during MCRT time.
            if (mRenderOutputDriver->requiresWireframe()) {
                // see Aov.cc:sampleWireframe()
                requiredKeys.insert(shading::StandardAttributes::sPolyVertexType);
                requiredKeys.insert(shading::StandardAttributes::sNumPolyVertices);
                optionalKeys.insert(shading::StandardAttributes::sPolyVertices,
                    shading::StandardAttributes::sPolyVertices +
                    shading::StandardAttributes::MAX_NUM_POLYVERTICES);
            }
            if (mRenderOutputDriver->requiresMotionVector()) {
                // see Aov.cc:computeMotionVector()
                optionalKeys.insert(shading::StandardAttributes::sMotion);
            }
            const auto &primAttrs = mRenderOutputDriver->getPrimAttrs();
            optionalKeys.insert(primAttrs.begin(), primAttrs.end());

            s->getOrCreate<shading::Material>().setAttributeTable(
                std::unique_ptr<moonray::shading::AttributeTable>(
                new moonray::shading::AttributeTable(requiredKeys, optionalKeys)));
        } else if (s->isA<scene_rdl2::rdl2::RootShader>()) {
            s->getOrCreate<shading::RootShader>().setAttributeTable(
                std::unique_ptr<moonray::shading::AttributeTable>(
                new moonray::shading::AttributeTable(requiredKeys, optionalKeys)));
        }

        // TODO: Only if the new required attribute table is different than the
        // previous one should we trigger geometry re-generation for
        // the geometry which is assigned to this shader.
        // The scene_rdl2::rdl2::Layer knows nothing about required primitive attributes so
        // we are in charge of tracking this here.
        // CORRECTION: Actually, we should check if the union of all these tables
        // for a given geometry is different (subtle yet important nuance,
        // see loop over the GeometryToRootShadersMap in GeometryManager::loadGeometries()).
        /*
        if (new attribute table != old attribute table) {
            // This causes
            mLayer->setGeometryUpdated(find the geometry assigned to this shader);
        }
        */
    }
}

void
RenderContext::buildMaterialAovFlags()
{
    // this should run after primitive attribute table for the shaders
    // has run and the render output driver has created the aov schema.
    const auto &aovSchema = mRenderOutputDriver->getAovSchema();
    if (!aovSchema.size()) return;

    // Gather all materials in shader network
    std::unordered_set<scene_rdl2::rdl2::Material *> materials;
    scene_rdl2::rdl2::Layer::MaterialSet topLevelMaterials;
    mLayer->getAllMaterials(topLevelMaterials);
    for (scene_rdl2::rdl2::RootShader * const s : topLevelMaterials) {
        scene_rdl2::rdl2::SceneObjectSet b;
        s->getBindingTransitiveClosure(b);
        for (scene_rdl2::rdl2::SceneObject * obj : b) {
            if (obj->isA<scene_rdl2::rdl2::Material>()) {
                materials.insert(obj->asA<scene_rdl2::rdl2::Material>());
            }
        }
    }

    for (scene_rdl2::rdl2::Material *m : materials) {
        // We need an extension object on all materials
        shading::Material &ext = m->getOrCreate<shading::Material>();

        // primitive attribute aovs
        // create a bool array, one entry per entry in the aov
        // schema indicating that the material does or does not expect
        // an intersection to have the requested primitive attribute.
        // We only need to do this for root materials.  Only root
        // material extension objects have primitive attribute tables.
        const moonray::shading::AttributeTable *attrTable = ext.getAttributeTable();
        if (attrTable) {
            const std::vector<shading::AttributeKey> &reqAttrs = attrTable->getRequiredAttributes();
            const std::vector<shading::AttributeKey> &optAttrs = attrTable->getOptionalAttributes();
            ext.getAovFlags().resize(aovSchema.size());
            auto af = ext.getAovFlags().begin();
            for (const auto &entry: aovSchema) {
                char flag = 1; // assume valid
                if (entry.type() == pbr::AOV_TYPE_PRIM_ATTR) {
                    shading::AttributeKey key(pbr::aovToGeomIndex(entry.id()));
                    if (find(reqAttrs.begin(), reqAttrs.end(), key) == reqAttrs.end() &&
                        find(optAttrs.begin(), optAttrs.end(), key) == optAttrs.end()) {
                        // intersections for this material will not have this
                        // attribute.  so mark it as deactivated.
                        flag = 0;
                    }
                }
                *af++ = flag;
            }
        }

        // material and light aovs
        {
            // build label ids (used for material aovs)
            // and lpe label ids (used for light aovs)
            const pbr::MaterialAovs &matAovs = mRenderOutputDriver->getMaterialAovs();
            const pbr::LightAovs &lightAovs = mRenderOutputDriver->getLightAovs();
            auto &lobeLabelIds = ext.getLobeLabelIds();
            auto &lpeLobeLabelIds = ext.getLpeLobeLabelIds();
            lobeLabelIds.clear();
            lpeLobeLabelIds.clear();

            // first set the material label ids
            const std::string &matLabel = m->get(scene_rdl2::rdl2::Material::sLabel);
            if (!matLabel.empty()) {
                ext.setMaterialLabelId(matAovs.getMaterialLabelIndex(matLabel));
                ext.setLpeMaterialLabelId(lightAovs.getMaterialLabelId(matLabel));
            } else {
                ext.setMaterialLabelId(-1);
                ext.setLpeMaterialLabelId(-1);
            }

            // the code relies on the aov values (used in the shaders) matching
            // the index location in lobeLabelIds and lpeLobeLabelIds - i.e.
            // we expect the aov values used in the shaders to start at '1' and
            // monotonically increase.  in shaders, a label value of '0'
            // applied to a lobe means "no label".
            lobeLabelIds.push_back(-1);
            lpeLobeLabelIds.push_back(-1);

            // does this material define labels?  if so, match those
            // labels to requested labels in the render output driver
            const scene_rdl2::rdl2::SceneClass &sc = m->getSceneClass();
            const char * const *labels = sc.getDataPtr<const char *>("labels");
            if (labels) {
                // changing this requires a change to the values assigned
                // to the shader aov labels (see ispc_dso.py)
                MNRY_ASSERT(lobeLabelIds.size() == 1);
                MNRY_ASSERT(lpeLobeLabelIds.size() == 1);

                // for light aovs, prepend the material label + '.' to every lobe label
                const std::string prefix = matLabel.empty()? "" : matLabel + '.';

                for (int i = 0; labels[i] != nullptr; ++i) {
                    const int labelId = matAovs.getLabelIndex(labels[i]);
                    lobeLabelIds.push_back(labelId);
                    const int lpeLabelId = lightAovs.getLabelId(prefix + labels[i]);
                    lpeLobeLabelIds.push_back(lpeLabelId);
                }
            }
        }

        // extra aovs
        {
            const pbr::LightAovs &lightAovs = mRenderOutputDriver->getLightAovs();
            std::vector<shading::Material::ExtraAov> extraAovs;
            std::vector<shading::Material::ExtraAov> postScatterExtraAovs;
            const scene_rdl2::rdl2::Map *listMap = m->get(scene_rdl2::rdl2::Material::sExtraAovsKey) ?
                m->get(scene_rdl2::rdl2::Material::sExtraAovsKey)->asA<scene_rdl2::rdl2::Map>() : nullptr;
            std::vector<const scene_rdl2::rdl2::Map *> extraAovMaps;
            if (listMap && listMap->getIsListMap(extraAovMaps)) {
                for (const scene_rdl2::rdl2::Map *map : extraAovMaps) {
                    int labelId = -1;
                    scene_rdl2::rdl2::String label;
                    scene_rdl2::rdl2::Bool postScatter;
                    if (!map->getIsExtraAovMap(label, postScatter)) {
                        // This is most likely a setup error.
                        Logger::warn("Map \"", map->getName(), "\" is used as an extra aov "
                            "by material \"", m->getName(), "\" but contains no extra aov data.");
                    } else {
                        // In light path expressions, the label is "U:<extra_aovs_label>"
                        labelId = lightAovs.getLabelId(std::string("U:") + label);
                    }
                    if (labelId != -1) {
                        if (postScatter) {
                            postScatterExtraAovs.push_back(shading::Material::ExtraAov { labelId, map });
                        } else {
                            extraAovs.push_back(shading::Material::ExtraAov { labelId, map });
                        }
                    }
                }
            } else if (listMap) {
                // This is most likely a setup error.
                Logger::warn("Map \"", listMap->getName(), "\" is used to list extra aovs "
                    "by material \"", m->getName(), "\" but is not a ListMap.");
            }
            ext.setExtraAovs(extraAovs);
            ext.setPostScatterExtraAovs(postScatterExtraAovs);
        }
    }
}

void
RenderContext::buildGeometryExtensions()
{
    const bool hasAovs = !mRenderOutputDriver->getAovSchema().empty();
    const pbr::MaterialAovs &matAovs = mRenderOutputDriver->getMaterialAovs();

    scene_rdl2::rdl2::Layer::GeometryToRootShadersMap g2s;
    mLayer->getAllGeometryToRootShaders(g2s);
    for (const auto &entry: g2s) {
        scene_rdl2::rdl2::Geometry *geometry = entry.first;

        // every geometry object requires an extension object
        auto &ext = geometry->getOrCreate<shading::Geometry>();

        // setup the geometry label id, if processing aovs
        if (hasAovs) {
            const std::string &geomLabel = geometry->get(scene_rdl2::rdl2::Geometry::sLabel);
            if (!geomLabel.empty()) {
                ext.setGeomLabelId(matAovs.getGeomLabelIndex(geomLabel));
            }
        }
    }
}

RenderContext::RP_RESULT
RenderContext::loadGeometries(const rt::ChangeFlag flag)
{
    mRenderStats->logInfoEmptyLine();
    mRenderStats->logStartGeneratingProcedurals();
    ManualRenderTimer timer(mRenderStats->mLoadProceduralsTime);

    MNRY_ASSERT(mPbrScene, "Must have loaded PBR library before loading procedurals.");

    int currentFrame = getCurrentFrame();
    // Get the motion blur related parameters
    geom::MotionBlurParams motionBlurParams = getMotionBlurParams();

    // Get the transform world --> render
    const Mat4d& world2render = mPbrScene->getWorld2Render();

    timer.start();

    const pbr::Scene * scene = getScene();

    // Camera may require per geometry primitive attributes (e.g. the BakeCamera)
    shading::PerGeometryAttributeKeySet perGeometryAttributes;
        scene->getCamera()->getRequiredPrimAttributes(perGeometryAttributes);

    mGeometryManager->
        setStageIdAndCallBackLoadGeometries(0,
                                            mRenderPrepExecTracker.getRenderPrepStatsCallBack(),
                                            mRenderPrepExecTracker.getRenderPrepCancelCallBack());
    if (mRenderPrepExecTracker.startLoadGeom0() == RenderPrepExecTracker::RESULT::CANCELED) {
        return RP_RESULT::CANCELED;
    }
    if (mGeometryManager->loadGeometries(mLayer, flag, world2render, currentFrame, motionBlurParams,
                                         getNumTBBThreads(), perGeometryAttributes) ==
        rt::GeometryManager::GM_RESULT::CANCELED) {
        return RP_RESULT::CANCELED;
    }
    if (mRenderPrepExecTracker.endLoadGeom0() == RenderPrepExecTracker::RESULT::CANCELED) {
        return RP_RESULT::CANCELED;
    }
    mGeometryManager->
        setStageIdAndCallBackLoadGeometries(1,
                                            mRenderPrepExecTracker.getRenderPrepStatsCallBack(),
                                            mRenderPrepExecTracker.getRenderPrepCancelCallBack());
    if (mRenderPrepExecTracker.startLoadGeom1() == RenderPrepExecTracker::RESULT::CANCELED) {
        return RP_RESULT::CANCELED;
    }
    if (mGeometryManager->loadGeometries(mMeshLightLayer, flag, world2render, currentFrame, motionBlurParams,
                                         getNumTBBThreads(), perGeometryAttributes) ==
        rt::GeometryManager::GM_RESULT::CANCELED) {
        return RP_RESULT::CANCELED;
    }
    if (mRenderPrepExecTracker.endLoadGeom1() == RenderPrepExecTracker::RESULT::CANCELED) {
        return RP_RESULT::CANCELED;
    }
    mGeometryManager->setChangeFlag(flag);

    timer.stop();

    // return unused memory from malloc() arena to OS so process memory usage
    // stats are accurate
    malloc_trim(0);

    mRenderStats->logEndGeneratingProcedurals();

    // Get the camera frustum and render to camera matrices for times points 0 and 1
    // TODO: generalize for multi-segment motion blur. Currently we only have 2
    // motion samples and assume the ray time points are 0 and 1. In multi-segment
    // motion blur there will be multiple motion samples with a ray time range
    // of [0,1].
    std::vector<mcrt_common::Frustum> frustums;
    const pbr::Camera *camera = getScene()->getCamera();
    if (camera->hasFrustum()) {
        frustums.push_back(mcrt_common::Frustum());
        camera->computeFrustum(&frustums.back(), 0, true);  // frustum at shutter open
        frustums.push_back(mcrt_common::Frustum());
        camera->computeFrustum(&frustums.back(), 1, true);  // frustum at shutter close
    }

    // configure the way to construct spatial accelerator
    constexpr rt::OptimizationTarget accelMode = rt::OptimizationTarget::HIGH_QUALITY_BVH_BUILD;

    // this will trigger BVH build if needed.
    
    mGeometryManager->
        setStageIdAndCallBackFinalizeChange(0,
                                            mRenderPrepExecTracker.getRenderPrepStatsCallBack(),
                                            mRenderPrepExecTracker.getRenderPrepCancelCallBack());
    if (mRenderPrepExecTracker.startFinalizeChange0() == RenderPrepExecTracker::RESULT::CANCELED) {
        return RP_RESULT::CANCELED;
    }
    const scene_rdl2::rdl2::Camera* dicingCamera = mSceneContext->getDicingCamera();
    if (mGeometryManager->finalizeChanges(mLayer, motionBlurParams, frustums,
                                          world2render, accelMode, dicingCamera, 
                                          true /*update scene bvh*/) == rt::GeometryManager::GM_RESULT::CANCELED) {
        return RP_RESULT::CANCELED;
    }
    if (mRenderPrepExecTracker.endFinalizeChange0() == RenderPrepExecTracker::RESULT::CANCELED) {
        return RP_RESULT::CANCELED;
    }
    mGeometryManager->
        setStageIdAndCallBackFinalizeChange(1,
                                            mRenderPrepExecTracker.getRenderPrepStatsCallBack(),
                                            mRenderPrepExecTracker.getRenderPrepCancelCallBack());
    if (mRenderPrepExecTracker.startFinalizeChange1() == RenderPrepExecTracker::RESULT::CANCELED) {
        return RP_RESULT::CANCELED;
    }
    if (mGeometryManager->finalizeChanges(mMeshLightLayer, motionBlurParams, frustums,
                                          world2render, accelMode, dicingCamera, false /*don't update scene bvh*/) ==
        rt::GeometryManager::GM_RESULT::CANCELED) {
        return RP_RESULT::CANCELED;
    }
    if (mRenderPrepExecTracker.endFinalizeChange1() == RenderPrepExecTracker::RESULT::CANCELED) {
        return RP_RESULT::CANCELED;
    }
    mGeometryManager->setChangeFlag(rt::ChangeFlag::NONE);

    // collect statistics
    mRenderStats->mTessellationTime =
        mGeometryManager->getStatistics().mTessellationTime;
    mRenderStats->mPerPrimitiveTessellationTime =
        mGeometryManager->getStatistics().mPerPrimitiveTessellationTime;
    mRenderStats->mBuildAcceleratorTime =
        mGeometryManager->getStatistics().mBuildAcceleratorTime;
    mRenderStats->mBuildProceduralTime =
        mGeometryManager->getStatistics().mBuildProceduralTime;
    mRenderStats->mRtcCommitTime =
        mGeometryManager->getStatistics().mRtcCommitTime;

    return RP_RESULT::FINISHED;
}

void
RenderContext::resetShaderStatsAndLogs()
{
    const int numTLS = getMaxNumTLS();
    scene_rdl2::rdl2::Shader::getLogEventRegistry().clearCounters();
    std::for_each(mSceneContext->beginSceneObject(),
                  mSceneContext->endSceneObject(),
                  [numTLS](const std::pair<std::string, scene_rdl2::rdl2::SceneObject*>& entry) {
            scene_rdl2::rdl2::SceneObject *obj = entry.second;
            if (obj->isA<scene_rdl2::rdl2::Shader>()) {
                scene_rdl2::rdl2::Shader* const shader = obj->asA<scene_rdl2::rdl2::Shader>();
                auto tlos = shader->getThreadLocalObjectState();
                for (int i = 0; i < numTLS; i++) {
                    tlos[i].clear();
                }
            }
        });
}

void
RenderContext::collectShaderStats()
{
    // Use the number of tbb threads here so we're not counting data from the
    // main thread or GUI thread.
    int numRenderThreads = getNumTBBThreads();

    std::for_each(mSceneContext->beginSceneObject(),
                  mSceneContext->endSceneObject(),
        [this,numRenderThreads](const std::pair<std::string, scene_rdl2::rdl2::SceneObject*>& entry) {
            scene_rdl2::rdl2::SceneObject *obj = entry.second;
            moonray::util::InclusiveExclusiveAverage<int64> shaderCallStat;
            if (obj->isA<scene_rdl2::rdl2::Shader>()) {
                obj->asA<scene_rdl2::rdl2::Shader>()->forEachThreadLocalObjectState([&shaderCallStat] (const shading::ThreadLocalObjectState &s) {
                        shaderCallStat += s.mShaderCallStat;
                    }, numRenderThreads);
                mRenderStats->mShaderCallStats[obj] = shaderCallStat;
            } else if (obj->isA<scene_rdl2::rdl2::Map>()) {
                obj->asA<scene_rdl2::rdl2::Map>()->forEachThreadLocalObjectState([&shaderCallStat] (const shading::ThreadLocalObjectState &s) {
                        shaderCallStat += s.mShaderCallStat;
                    }, numRenderThreads);
                mRenderStats->mShaderCallStats[obj] = shaderCallStat;
            }
        });
}

RealtimeFrameStats &
RenderContext::getCurrentRealtimeFrameStats() const
{
    return getRenderDriver()->getCurrentRealtimeFrameStats();
}

void
RenderContext::commitCurrentRealtimeStats() const
{
    getRenderDriver()->commitCurrentRealtimeStats();
}

pbr::Statistics
RenderContext::accumulatePbrStatistics() const
{
    pbr::Statistics pbrStatistics;
    pbrStatistics.mMcrtTime = mDriver->getLastFrameMcrtDuration();
    pbrStatistics.mMcrtUtilization = mDriver->getLastFrameMcrtUtilization();
    pbr::forEachTLS([&](pbr::TLState const *tls){ (pbrStatistics) += tls->mStatistics; });
    return pbrStatistics;
}

void
RenderContext::reportShadingLogs()
{
    auto formatter = [](const scene_rdl2::rdl2::Shader* p, unsigned count, const std::string& description) {
        std::ostringstream oss;
        oss << p->getSceneClass().getName() << R"((")" << p->getName() << R"("): )" << '(' << count << " times) " << description;
        const std::string result = oss.str();
        return result;
    };
    scene_rdl2::rdl2::Shader::getLogEventRegistry().outputReports(formatter);

    std::for_each(mSceneContext->beginSceneObject(),
                  mSceneContext->endSceneObject(),
        [](const std::pair<std::string, scene_rdl2::rdl2::SceneObject*>& entry) {
            scene_rdl2::rdl2::SceneObject *obj = entry.second;
            MNRY_ASSERT(obj);
            if (obj->isA<scene_rdl2::rdl2::Shader>()) {
                // Reset fatal flag so we can potentially render next frame
                // fatal only applies to the current frame
                auto* const shader = obj->asA<scene_rdl2::rdl2::Shader>();
                shader->setFataled(false);
            }
        });
}

void
RenderContext::reportGeometryTessellationTime()
{
    if (mRenderStats->getLogInfo() || mRenderStats->getLogCsv()) {
        mRenderStats->logTopTessellationStats();
    }
    if (mRenderStats->getLogAthena()) {
        mRenderStats->logAllTessellationStats();
    }
}

void
RenderContext::reportGeometryMemory()
{
    // Report memory footprint for geometry primitives
    size_t totalGeometryBytes = 0;
    std::vector<std::pair<std::string, size_t>> perGeometryBytes;
    std::for_each(mSceneContext->beginGeometrySet(),
        mSceneContext->endGeometrySet(),
    [&](scene_rdl2::rdl2::GeometrySet* geometrySet) {
        const scene_rdl2::rdl2::SceneObjectIndexable& geometries = geometrySet->getGeometries();
        std::for_each(geometries.begin(), geometries.end(),
        [&](scene_rdl2::rdl2::SceneObject* sceneObject) {
            geom::Procedural* procedural =
                sceneObject->asA<scene_rdl2::rdl2::Geometry>()->getProcedural();
            // geometry can be in the GeometrySet but not added to layer
            // when user manually setup SceneContext by hand.
            if (procedural != nullptr) {
                // TODO: We can't handle nested procedurals yet.
                if (!procedural->isLeaf()) {
                    throw scene_rdl2::except::NotImplementedError(
                        "Nested procedurals not supported yet.");
                }
                geom::ProceduralLeaf* leaf =
                    static_cast<geom::ProceduralLeaf*>(procedural);
                std::pair<std::string, size_t> geomMemInfo;
                geomMemInfo.first = sceneObject->asA<scene_rdl2::rdl2::Geometry>()->getName();
                size_t mem = leaf->getMemory();
                geomMemInfo.second = mem;
                perGeometryBytes.push_back(geomMemInfo);
                totalGeometryBytes += mem;
            }
        });
    });

    size_t bvhBytes = mGeometryManager->getEmbreeAccelerator()->getMemory();
    std::sort(perGeometryBytes.begin(),
          perGeometryBytes.end(),
          [](std::pair<std::string, size_t> a,
             std::pair<std::string, size_t> b) {
          if (a.second < b.second) {
              return false;
          } else if (a.second > b.second) {
              return true;
          } else {
              return a.first.compare(b.first) <= 0;
          }
    });

    mRenderStats->logMemoryUsage(totalGeometryBytes, perGeometryBytes, bvhBytes);
}

void
RenderContext::reportGeometryStatistics()
{
    // Report number of polys/cvs/curves/instances for geometry primitives
    geom::GeometryStatistics totalGeomStatistics;
    rndr::GeometryStatsTable perGeomStatistics;
    std::for_each(mSceneContext->beginGeometrySet(),
        mSceneContext->endGeometrySet(),
    [&](scene_rdl2::rdl2::GeometrySet* geometrySet) {
        const scene_rdl2::rdl2::SceneObjectIndexable& geometries = geometrySet->getGeometries();
        std::for_each(geometries.begin(), geometries.end(),
        [&](scene_rdl2::rdl2::SceneObject* sceneObject) {
            geom::Procedural* procedural =
                sceneObject->asA<scene_rdl2::rdl2::Geometry>()->getProcedural();
            // geometry can be in the GeometrySet but not added to layer
            // when user manually setup SceneContext by hand.
            if (procedural) {
                // TODO: We can't handle nested procedurals yet.
                if (!procedural->isLeaf()) {
                    throw scene_rdl2::except::NotImplementedError(
                        "Nested procedurals not supported yet.");
                }
                geom::ProceduralLeaf* leaf =
                    static_cast<geom::ProceduralLeaf*>(procedural);
                geom::GeometryStatistics statistics = leaf->getStatistics();
                perGeomStatistics.emplace_back(
                        sceneObject->asA<scene_rdl2::rdl2::Geometry>()->getName(),
                        statistics);
                totalGeomStatistics.mFaceCount += statistics.mFaceCount;
                totalGeomStatistics.mMeshVertexCount += statistics.mMeshVertexCount;
                totalGeomStatistics.mCurvesCount += statistics.mCurvesCount;
                totalGeomStatistics.mCVCount += statistics.mCVCount;
                totalGeomStatistics.mInstanceCount += statistics.mInstanceCount;
            }
        });
    });
    mRenderStats->logGeometryUsage(totalGeomStatistics, perGeomStatistics);
}

namespace {
   void validateSamplingMode(SamplingMode samplingMode)
   {
       switch (samplingMode) {
           case SamplingMode::UNIFORM: break;
           case SamplingMode::ADAPTIVE: break;
           default:
               // We don't want to continue, because defaulting to another
               // sampling mode may take much longer than the user expects.
               Logger::fatal("Using invalid sampling mode ", static_cast<int>(samplingMode));
               std::exit(EXIT_FAILURE);
               break;
       }
   }
} // anonymous namespace

void
RenderContext::buildFrameState(FrameState *fs, double frameStartTime, ExecutionMode executionMode) const
{
    // cppcheck-suppress memsetClassFloat // floating point memset to 0 is fine
    memset(fs, 0, sizeof(FrameState));

    const scene_rdl2::rdl2::SceneVariables &vars = mSceneContext->getSceneVariables();
    unsigned pixelSamplesSqrt = (unsigned)std::max(1, vars.get(scene_rdl2::rdl2::SceneVariables::sPixelSamplesSqrt));

    unsigned numSamplesPerPixel = pixelSamplesSqrt * pixelSamplesSqrt;

    SamplingMode samplingMode = (SamplingMode)vars.get(scene_rdl2::rdl2::SceneVariables::sSamplingMode);
    validateSamplingMode(samplingMode);

    // Adaptive sampling mode takes priority over pixel samples maps if both are active.
    // This means to use the pixel sample maps functionality, adaptive sampling must be
    // set to off.
    // Adaptive sampling isn't currently supported in vector or xpu mode.
    fs->mSamplingMode = samplingMode;
    fs->mOriginalSamplesPerPixel = numSamplesPerPixel;

    if (samplingMode == SamplingMode::UNIFORM) {

        // The maximum number of samples per pixel over the pixel sample map
        fs->mMaxSamplesPerPixel = unsigned(numSamplesPerPixel * mMaxPixelSampleValue);
        fs->mMinSamplesPerPixel = fs->mMaxSamplesPerPixel;
        fs->mTargetAdaptiveError = 0.f;
        fs->mPixelSampleMap = mPixelSampleMap.get();

    } else {

        // Adaptive sampling is on.
        fs->mMinSamplesPerPixel = (unsigned)std::max(2, vars.get(scene_rdl2::rdl2::SceneVariables::sMinAdaptiveSamples));
        fs->mMaxSamplesPerPixel = (unsigned)std::max((int)fs->mMinSamplesPerPixel, vars.get(scene_rdl2::rdl2::SceneVariables::sMaxAdaptiveSamples));

        // The paper on which our adaptive sampling is based uses an error value of 0.0002 for "convergence". We divide
        // by 10,000 in order to provide more user-friendly values (e.g. 2.0).
        const float targetAdaptiveError = vars.get(scene_rdl2::rdl2::SceneVariables::sTargetAdaptiveError) / 10000.0f;
        fs->mTargetAdaptiveError = std::max(0.000001f, targetAdaptiveError);
        fs->mPixelSampleMap = nullptr;
    }

    fs->mDeepFormat = vars.get(scene_rdl2::rdl2::SceneVariables::sDeepFormat);
    fs->mDeepCurvatureTolerance = vars.get(scene_rdl2::rdl2::SceneVariables::sDeepCurvatureTolerance);
    fs->mDeepZTolerance = vars.get(scene_rdl2::rdl2::SceneVariables::sDeepZTolerance);
    fs->mDeepVolCompressionRes = vars.get(scene_rdl2::rdl2::SceneVariables::sDeepVolCompressionRes);

    fs->mDeepIDChannelNames = MNRY_VERIFY(mDeepIDChannelNames.get());
    if (fs->mDeepIDChannelNames->size() > 6) {
        // 6 is needed to maintain the 64-byte size of DeepData
        throw scene_rdl2::except::RuntimeError(scene_rdl2::util::buildString(
            "More than 6 deep ID channels specified."));
    }

    fs->mDeepMaxLayers = vars.get(scene_rdl2::rdl2::SceneVariables::sDeepMaxLayers);
    fs->mDeepLayerBias = vars.get(scene_rdl2::rdl2::SceneVariables::sDeepLayerBias);

    //
    // From pbr::FrameState (all the data which the pbr library requires should
    // be set here).
    //
    fs->mExecutionMode = executionMode;
    fs->mEmbreeAccel = MNRY_VERIFY(mGeometryManager->getEmbreeAccelerator());
    fs->mGPUAccel = mGeometryManager->getGPUAccelerator(); // may be nullptr if not in xpu mode
    fs->mLayer = MNRY_VERIFY(mLayer);
    fs->mTextureBlur = vars.get(scene_rdl2::rdl2::SceneVariables::sTextureBlur);
    fs->mFatalColor = vars.get(scene_rdl2::rdl2::SceneVariables::sFatalColor);
    fs->mPropagateVisibilityBounceType = vars.get(scene_rdl2::rdl2::SceneVariables::sPropagateVisibilityBounceType);
    fs->mIntegrator = MNRY_VERIFY(mIntegrator.get());
    fs->mScene = MNRY_VERIFY(mPbrScene.get());
    fs->mAovSchema = &mRenderOutputDriver->getAovSchema();
    fs->mMaterialAovs = &mRenderOutputDriver->getMaterialAovs();
    fs->mLightAovs = &mRenderOutputDriver->getLightAovs();
    fs->mRequiresHeatMap = mRenderOutputDriver->requiresHeatMap();
    fs->mShadingWorkloadChunkSize = mOptions.getShadingWorkloadChunkSize();
    fs->mRequiresCryptomatteBuffer = mRenderOutputDriver->requiresCryptomatteBuffer();

    //
    // From rndr::FrameState.
    //
    fs->mNumRenderThreads = MNRY_VERIFY(getNumTBBThreads());

    int machineId = vars.get(scene_rdl2::rdl2::SceneVariables::sMachineId);
    int numMachines = vars.get(scene_rdl2::rdl2::SceneVariables::sNumMachines);
    fs->mNumRenderNodes = std::max(numMachines, 1);
    fs->mRenderNodeIdx = clamp(machineId, 0, (int)(fs->mNumRenderNodes - 1));
    fs->mTaskDistributionType = (unsigned)vars.get(scene_rdl2::rdl2::SceneVariables::sTaskDistributionType);

    fs->mLockFrameNoise = vars.get(scene_rdl2::rdl2::SceneVariables::sLockFrameNoise);

    switch (getRenderMode())
    {
    case RenderMode::BATCH:
        fs->mTileSchedulerType = TileScheduler::MORTON;
        break;
    case RenderMode::PROGRESSIVE:
    case RenderMode::PROGRESSIVE_FAST:
        fs->mTileSchedulerType = (unsigned)vars.get(scene_rdl2::rdl2::SceneVariables::sProgressiveTileOrder);
        break;
    case RenderMode::REALTIME:
        // @@@@ ML: Temp, use spiral square for realtime mode until such a time
        //          we can configure the merge node with MORTON.
#if 0
        fs->mTileSchedulerType = TileScheduler::MORTON;
#else
        fs->mTileSchedulerType = TileScheduler::SPIRAL_SQUARE;
#endif
        break;
    case RenderMode::PROGRESS_CHECKPOINT:
        if (fs->mNumRenderNodes == 1) {
            fs->mTileSchedulerType = (unsigned)vars.get(scene_rdl2::rdl2::SceneVariables::sCheckpointTileOrder);
        } else {
            // This reduces image update visual imbalance artifacts and achieves a better balanced progressive
            // update feeling under multi-machine especially many mcrtTotal condition.
            fs->mTileSchedulerType = TileScheduler::MORTON_SHIFTFLIP;
        }
        break;
    default:
        MNRY_ASSERT(0);
    }

    if (getRenderMode() == RenderMode::BATCH) {
        if (mSceneContext->getCheckpointActive() || mSceneContext->getResumeRender()) {
            // We can use checkpoint mode for regular BATCH rendering.
            // Also use checkpoint mode if resumeRender mode is activated (Currently checkpoint render is
            // only supported to re-construct resume sampling schedule. So we have to use checkpoint
            // render for resumeRender execution).
            fs->mRenderMode = RenderMode::PROGRESS_CHECKPOINT;
        } else if (mSceneContext->getSceneVariables().get(scene_rdl2::rdl2::SceneVariables::sPathGuideEnable)) {
            // we'll use progressive mode to enable pass resets
            fs->mRenderMode = RenderMode::PROGRESSIVE;
        } else if (fs->mSamplingMode == SamplingMode::ADAPTIVE) {
            fs->mRenderMode = RenderMode::PROGRESSIVE;
        }
    } else {
        fs->mRenderMode = getRenderMode();

        /* useful debug code : progressCheckPoint mode test for moonray_gui
        if (getRenderMode() == RenderMode::PROGRESSIVE) {
            if (mSceneContext->getCheckpointActive() || mSceneContext->getResumeRender()) {
                fs->mRenderMode = RenderMode::PROGRESS_CHECKPOINT;
            }
        }
        */
    }
    fs->mFastMode = getFastRenderMode();
    fs->mRequiresDeepBuffer = mRenderOutputDriver->requiresDeepBuffer();

    // Find a better way to drive pixel info...
    fs->mGeneratePixelInfo = mOptions.getGeneratePixelInfo();

    fs->mWidth = vars.getRezedWidth();
    fs->mHeight = vars.getRezedHeight();

    // Clamp viewport to rezed dimensions to ensure it's inside of our
    // allocated buffers.
    fs->mViewport = scene_rdl2::math::convertToClosedViewport(vars.getRezedSubViewport());

    MNRY_ASSERT(fs->mViewport.mMinX >= 0 && fs->mViewport.mMinY >= 0);

    fs->mDofEnabled = mPbrScene->getCamera()->getIsDofEnabled();

    fs->mPixelFilter = MNRY_VERIFY(mPixelFilter.get());

    fs->mFrameNumber = getCurrentFrame();
    fs->mFps = std::max(vars.get(scene_rdl2::rdl2::SceneVariables::sFpsKey), 1.f);
    if (mOptions.getFps() > 0.0f) {
        fs->mFps = mOptions.getFps(); // overwrite fps value by option
    }

    fs->mFrameStartTime = frameStartTime;

    const pbr::Camera* camera = getScene()->getCamera();
    StereoView stereoView = camera->getStereoView();
    fs->mInitialSeed = getInitialSeed(fs->mFrameNumber, stereoView, fs->mLockFrameNoise);

    // the presence shadow handler in vectorized mode need to access this value
    fs->mMaxPresenceDepth = vars.get(scene_rdl2::rdl2::SceneVariables::sMaxPresenceDepth);
    fs->mPresenceThreshold = vars.get(scene_rdl2::rdl2::SceneVariables::sPresenceThreshold);

    fs->mShadowTerminatorFix = static_cast<shading::ShadowTerminatorFix>
        (vars.get(scene_rdl2::rdl2::SceneVariables::sShadowTerminatorFix));

    // checkpoint related
    // Setup checkpoint interval (minute) : 0.5sec or bigger
    fs->mCheckpointInterval = std::max(vars.get(scene_rdl2::rdl2::SceneVariables::sCheckpointInterval), 0.5f/60.0f);
    // Setup checkpoint quality steps
    fs->mCheckpointQualitySteps = std::max(vars.get(scene_rdl2::rdl2::SceneVariables::sCheckpointQualitySteps), 1);
    if (mSceneContext->getCheckpointActive() &&
        vars.get(scene_rdl2::rdl2::SceneVariables::sCheckpointTotalFiles) > 0 ) { // 0 = disable
        // checkpointTotalFiles control mode. We convert checkpointTotalFiles parameter to qualitySteps here
        std::string logMessage;
        {
            int totalCheckpointFiles = vars.get(scene_rdl2::rdl2::SceneVariables::sCheckpointTotalFiles);
            int checkpointStartSPP = vars.get(scene_rdl2::rdl2::SceneVariables::sCheckpointStartSPP);
            fs->mCheckpointQualitySteps =
                RenderDriver::convertTotalCheckpointToQualitySteps(samplingMode,
                                                                   checkpointStartSPP,
                                                                   fs->mMaxSamplesPerPixel,
                                                                   totalCheckpointFiles,
                                                                   logMessage);
        }
        Logger::info(logMessage);
        // if (isatty(STDOUT_FILENO)) std::cout << logMessage << std::endl; // useful for debug run from terminal
    }
    // Setup checkpoint time cap (minute). 0 = disable
    fs->mCheckpointTimeCap = vars.get(scene_rdl2::rdl2::SceneVariables::sCheckpointTimeCap);
    // Setup checkpoint sample cap (count). 0 = disable
    // We need tile sample total number for mCheckpointSampleCap. This is why * 64.
    fs->mCheckpointSampleCap = vars.get(scene_rdl2::rdl2::SceneVariables::sCheckpointSampleCap) * 64;
    switch (vars.get(scene_rdl2::rdl2::SceneVariables::sCheckpointMode)) {
    case 0 : fs->mCheckpointMode = CheckpointMode::TIME_BASED; break;
    case 1 : fs->mCheckpointMode = CheckpointMode::QUALITY_BASED; break;
    }
    // Setup start sample number for checkpoint dump action
    fs->mCheckpointStartSPP = vars.get(scene_rdl2::rdl2::SceneVariables::sCheckpointStartSPP);
    // Setup checkpoint bg write mode
    fs->mCheckpointBgWrite = vars.get(scene_rdl2::rdl2::SceneVariables::sCheckpointBgWrite);

    // two stage output mode condition
    fs->mTwoStageOutput = vars.get(scene_rdl2::rdl2::SceneVariables::sTwoStageOutput);

#if defined(USE_PARTITIONED_1D)
    fs->mSamples1D = pbr::k1DSampleTable.data();
#else
    fs->mSamples1D = nullptr;
#endif

#if defined(USE_PARTITIONED_2D)
    fs->mSamples2D = pbr::k2DSampleTable.data();
#else
    fs->mSamples2D = nullptr;
#endif

    fs->mDisplayFilterCount = mRenderOutputDriver->getDisplayFilterCount();
}

void
RenderContext::buildAndSaveRayDatabase()
{
    const scene_rdl2::rdl2::SceneVariables& sceneVars = mSceneContext->getSceneVariables();
    if (sceneVars.get(scene_rdl2::rdl2::SceneVariables::sDebugRaysFile).empty()) {
        return;
    }

    try {
        tbb::tick_count t0 = tbb::tick_count::now();

        Logger::info("Building debug ray database...");

        std::vector<pbr::DebugRayRecorder *> recorders;
        pbr::forEachTLS([&recorders](pbr::TLState *tls) {
            pbr::DebugRayRecorder *recorder = tls->mRayRecorder;
            MNRY_ASSERT(!recorder->isRecording());
            recorders.push_back(recorder);
        });

        // Condition recorded ray data...
        pbr::DebugRayBuilder builder;
        builder.build(sceneVars.getRezedWidth(), sceneVars.getRezedHeight(), recorders);

        // save conditioned data into debug ray database for serialization
        pbr::DebugRayDatabase db;
        builder.exportDatabase(&db);

        if (!db.empty()) {
            tbb::tick_count t1 = tbb::tick_count::now();
            Logger::info("  build completed, primary rays = " , db.getPrimaryRayIndices().size() ,
                       ", vertices = " , db.getRays().size() ,
                       ", build time = " , (t1-t0).seconds());

            std::string fileName = sceneVars.get(scene_rdl2::rdl2::SceneVariables::sDebugRaysFile);
            db.save(fileName.c_str());
        } else {
            Logger::info("  no rays recorded, skipping.");
        }
    } catch (const std::exception& e) {
        Logger::error("Error generating debug ray database: " , e.what());
    }
}

void
RenderContext::incrementCurfield()
{
    scene_rdl2::rdl2::SceneVariables& sceneVars = mSceneContext->getSceneVariables();
    const auto curfield = sceneVars.get(scene_rdl2::rdl2::SceneVariables::sFrameKey);

    sceneVars.beginUpdate();
    sceneVars.set(scene_rdl2::rdl2::SceneVariables::sFrameKey, curfield + 1);
    sceneVars.endUpdate();
}

geom::MotionBlurParams
RenderContext::getMotionBlurParams(bool bake) const
{
    std::vector<float> motionSteps;
    float shutterOpen, shutterClose;
    auto sceneVarsMotionSteps =
        mSceneContext->getSceneVariables().get(scene_rdl2::rdl2::SceneVariables::sMotionSteps);

    bool isMotionBlurOn = mSceneContext->getSceneVariables().get(scene_rdl2::rdl2::SceneVariables::sEnableMotionBlur);

    // Get the motion steps. We need these both when motion blur is on and off
    // in order to correctly interpolate the xform samples of the scene object.
    int numSteps = sceneVarsMotionSteps.size();
    if (numSteps > 0) {
        if (numSteps > 2) {
            Logger::warn("Only 2 motion steps are supported (" ,
                         numSteps , " defined). The first 2 will be used.");
            numSteps = 2;
        }
        for (int i = 0; i < numSteps; i++) {
            motionSteps.push_back(sceneVarsMotionSteps[i]);
        }
    } else {
        Logger::warn("No motion steps defined. Implicitly using a single"
                " motion step of 0.");
        motionSteps.push_back(0.0f);
    }

    // Get the shutter open and close
    if (!isMotionBlurOn) {
        // when motion blur is off, the shutter open and close is the current time
        shutterOpen  = 0.0f;
        shutterClose = 0.0f;
    } else if (bake) {
        // Shutter open and close set to same values as motion steps for baking
        shutterOpen  = sceneVarsMotionSteps[0];
        shutterClose = sceneVarsMotionSteps[1];
    } else {
        // Shutter open and close are controlled by the camera
        shutterOpen  = mCamera->get(scene_rdl2::rdl2::Camera::sMbShutterOpenKey);
        shutterClose = mCamera->get(scene_rdl2::rdl2::Camera::sMbShutterCloseKey);
    }

    float fps = mSceneContext->getSceneVariables().get(scene_rdl2::rdl2::SceneVariables::sFpsKey);

    return geom::MotionBlurParams(motionSteps, shutterOpen, shutterClose, isMotionBlurOn, fps);
}

void
RenderContext::fatalShade(const scene_rdl2::rdl2::Material* self, shading::TLState *tls,
                          const shading::State &state, shading::BsdfBuilder &bsdfBuilder)
{
    const scene_rdl2::rdl2::SceneVariables &sv = self->getSceneClass().getSceneContext()->getSceneVariables();
    const Color fatalColor = sv.get(scene_rdl2::rdl2::SceneVariables::sFatalColor);
    bsdfBuilder.addEmission(fatalColor);
}

void
RenderContext::fatalSample(const scene_rdl2::rdl2::Map* self, shading::TLState *tls,
                           const shading::State& state, Color* sample)
{
    const scene_rdl2::rdl2::SceneVariables &sv = self->getSceneClass().getSceneContext()->getSceneVariables();
    *sample = sv.get(scene_rdl2::rdl2::SceneVariables::sFatalColor);
}

void
RenderContext::fatalSampleNormal(const scene_rdl2::rdl2::NormalMap* self, shading::TLState *tls,
                                 const shading::State& state, Vec3f* sample)
{
    *sample = state.getN();
}

void
RenderContext::handlePickLightContributions(const int x, const int y,
        moonray::shading::LightContribArray& lightContributions) const
{
    ThreadLocalState *tls = getGuiTLS();
    const scene_rdl2::rdl2::SceneVariables& sceneVars = mSceneContext->getSceneVariables();
    // TODO Don't hardcode this
    int numSamples = sceneVars.get(scene_rdl2::rdl2::SceneVariables::sLightSamplesSqrt)
                   * sceneVars.get(scene_rdl2::rdl2::SceneVariables::sPixelSamplesSqrt);
    numSamples = numSamples * numSamples;

    moonray::pbr::computeLightContributions(tls, mPbrScene.get(), x, y,
            lightContributions, numSamples, 1.0f);
}

const scene_rdl2::rdl2::Material*
RenderContext::handlePickMaterial(const int x, const int y) const
{
    ThreadLocalState *tls = getGuiTLS();
    return moonray::pbr::computeMaterial(tls, mPbrScene.get(), x, y);
}

const scene_rdl2::rdl2::Geometry*
RenderContext::handlePickGeometry(const int x, const int y) const
{
    ThreadLocalState *tls = getGuiTLS();
    int assignmentId = -1;
    moonray::pbr::computePrimitive(tls, mPbrScene.get(), x, y, assignmentId);

    // If we didn't pick anything return NULL
    if (assignmentId == -1) {
        return NULL;
    }

    // Get the geom and part for this assignmentId from the layer
    scene_rdl2::rdl2::Layer::GeometryPartPair geomPartPair = mLayer->lookupGeomAndPart(assignmentId);
    return geomPartPair.first;
}

const scene_rdl2::rdl2::Geometry*
RenderContext::handlePickGeometryPart(const int x, const int y, std::string& part) const
{
    ThreadLocalState *tls = getGuiTLS();
    int assignmentId = -1;
    moonray::pbr::computePrimitive(tls, mPbrScene.get(), x, y, assignmentId);

    // If we didn't pick anything return NULL
    if (assignmentId == -1) {
        return NULL;
    }

    // Get the geom and part for this assignmentId from the layer
    scene_rdl2::rdl2::Layer::GeometryPartPair geomPartPair = mLayer->lookupGeomAndPart(assignmentId);
    part = geomPartPair.second;
    return geomPartPair.first;
}

bool
RenderContext::handlePickLocation(const int x, const int y, scene_rdl2::math::Vec3f *hitPoint) const
{
    scene_rdl2::rdl2::SceneVariables& vars = mSceneContext->getSceneVariables();
    scene_rdl2::math::HalfOpenViewport region = vars.getRezedRegionWindow();
    int width = int(region.width());
    int height = int(region.height());

    if (x < 0 || x >= width ||
        y < 0 || y >= height) {
        return false;
    }

    const pbr::Scene *scene = getScene();
    const pbr::Camera *pbrCamera = scene->getCamera();

    mcrt_common::RayDifferential ray;
    pbrCamera->createRay(&ray, x + 0.5f, height - y + 0.5f, 0.0f, 0.0f, 0.0f, false);

    scene->getEmbreeAccelerator()->intersect(ray);
    bool hitGeom = (ray.geomID != -1);
    if (hitGeom) {
        *hitPoint = ray.org + ray.dir * ray.tfar;
    }

    // get closest light intersection
    std::vector<const pbr::Light*> hitLights;
    std::vector<pbr::LightIntersection> hitLightIsects;
    scene->pickVisibleLights(ray, hitGeom ? ray.getEnd() : pbr::sInfiniteLightDistance, hitLights,
        hitLightIsects);
    const pbr::Light *hitLight = nullptr;
    pbr::LightIntersection hitLightIsect;
    // setting the distance to pbr::sInfiniteLightDistance * 0.5f allows us to ignore distant and env lights for the
    // purpose of picking
    float minDistance = pbr::sInfiniteLightDistance * 0.5f;
    for (size_t i = 0; i < hitLights.size(); ++i) {
        if(hitLightIsects[i].distance < minDistance) {
            hitLight = hitLights[i];
            hitLightIsect = hitLightIsects[i];
            minDistance = hitLightIsect.distance;
        }
    }

    if (hitLight) {
        *hitPoint = ray.getOriginX() + ray.getDirection() * hitLightIsect.distance;
    }

    if (hitGeom || hitLight) {
        *hitPoint = transformPoint(scene->getRender2World(), *hitPoint);
        return true;
    }

    return false;
}

bool
RenderContext::canRunVectorized(std::string &reason) const
{
    // Naturally, the goal is to have this method simply be:
    //   return true;
    bool result = true;
    reason.clear();

    std::function<void(std::string)> fail = [&](const std::string &feature) {
        if (!reason.empty()) reason += ", ";
        reason += feature;
        result = false;
    };

    MNRY_ASSERT(mLayer);

    // These are conservative checks.  This function is run before
    // render prep so it isn't possible to know if a particular object
    // in the scene actually participates in rendering.  In some cases,
    // the mere existence of an unsupported object or setting returns false.

    // The order of checks is alphabetical based on the fail string.  This
    // makes reading the debug message easier when multiple features are
    // missing.

    // Since we have multiple issues with render output objects, we'll loop
    // over all the render outputs once, checking for all the problem types.
    // The issues will still be reported in alphabetical order.
    bool hasDeepOutput = false;
    bool hasRefractCrypto = false;
    const scene_rdl2::rdl2::SceneContext::RenderOutputVector &ros = mSceneContext->getAllRenderOutputs();
    for (auto roItr = ros.cbegin(); roItr != ros.cend(); ++roItr) {
        const scene_rdl2::rdl2::RenderOutput *ro = *roItr;
        if (ro->getActive()) {
            if (ro->getOutputType() == "deep") {
                hasDeepOutput = true;
            }
            if (ro->getResult() == scene_rdl2::rdl2::RenderOutput::RESULT_CRYPTOMATTE &&
                ro->getCryptomatteEnableRefract()) {
                hasRefractCrypto = true;
            }
        }
    }
    const scene_rdl2::rdl2::SceneVariables &vars = mSceneContext->getSceneVariables();

    // Overlapping Dielectrics: MOONRAY-3083
    scene_rdl2::rdl2::Layer::MaterialSet materials;
    mLayer->getAllMaterials(materials);
    for (const scene_rdl2::rdl2::Material *mat : materials) {
        if (mat->priority() > 0) {
            fail("overlapping dielectrics");
            break; // no need to keep looping
        }
    }

    // Path Guiding: MOONRAY-3018
    if (vars.get(scene_rdl2::rdl2::SceneVariables::sPathGuideEnable)) {
        fail("path guiding");
    }

    // Volume Rendering + Deep Output: MOONRAY-3133
    if (hasDeepOutput) {
        const auto &volumeShaders = mLayer->get<scene_rdl2::rdl2::SceneObjectVector>("volume shaders");
        for (size_t i = 0; i < volumeShaders.size(); ++i) {
            if (volumeShaders[i] != nullptr) {
                fail("volume rendering with deep output");
                break; // no reason to keep looping
            }
        }
    }

    // Refractive crypto: MOONRAY-5074
    if (hasRefractCrypto) {
        fail("refractive cryptomattes");
    }

    return result;
}

void
RenderContext::parserConfigure()
{
    mParser.description("RenderContext command");

    mParser.opt("startRender", "", "calls RenderContext::setForceCallStartFrame(true) : for moonray_gui",
                [&](Arg &arg) -> bool {
                    if (!isFrameRendering()) {
                        setForceCallStartFrame();
                        return arg.msg("setForceCallStartFrame(true)\n");
                    } else {
                        return arg.msg("already start render\n");
                    }
                });
    mParser.opt("stopRender", "", "calls RenderContext::stopFrame()",
                [&](Arg &arg) -> bool {
                    if (isFrameRendering()) {
                        stopFrame();
                        return arg.msg("stopFrame() done\n");
                    } else {
                        return arg.msg("already stop render\n");
                    }
                });
    mParser.opt("renderContextAddr", "", "show renderContext address",
                [&](Arg &arg) -> bool {
                    std::ostringstream ostr;
                    ostr << "renderContext:0x" << std::hex << (uintptr_t)this;
                    return arg.msg(ostr.str() + '\n');
                });
    mParser.opt("renderPrepExecTracker", "...command...", "renderPrepExecTracker command",
                [&](Arg &arg) -> bool {
                    return mRenderPrepExecTracker.getParser().main(arg.childArg());
                });
    mParser.opt("geometryManagerExecTracker", "...command...", "geometryManagerExecTracker command",
                [&](Arg &arg) -> bool {
                    return mGeometryManager->getGeometryManagerExecTracker().getParser().main(arg.childArg());
                });
    mParser.opt("renderOutputDriver", "...command...", "renderOutputDriver command",
                [&](Arg& arg) -> bool {
                    if (!mRenderOutputDriver) { return arg.msg("RenderOutputDriver is empty\n"); }
                    return mRenderOutputDriver->getParser().main(arg.childArg());
                });
    mParser.opt("renderDriver", "...command...", "renderDriver command",
                [&](Arg& arg) -> bool {
                    if (!mDriver) { return arg.msg("RenderDriver is empty\n"); }
                    return mDriver->getParser().main(arg.childArg());
                });
    mParser.opt("renderOptions", "...command...", "renderOptions command",
                [&](Arg& arg) -> bool { return mOptions.getParser().main(arg.childArg()); });
    mParser.opt("textureSampler", "...command...", "textureSampler command",
                [&](Arg &arg) -> bool {
                    return moonray::texture::getTextureSampler()->getParser().main(arg.childArg());
                });
    mParser.opt("textureCacheSize", "<MB>", "set texture_cache_size scene_variable. <MB> is unsigned int",
                [&](Arg& arg) -> bool {
                    setSceneVarTextureCacheSize((arg++).as<unsigned int>(0));
                    return arg.msg(getSceneVarTextureCacheSize() + '\n');
                });
    mParser.opt("saveScene", "<filename>", "save scene",
                [&](Arg& arg) -> bool { return saveSceneCommand(arg); });
    mParser.opt("showExecMode", "", "show rendering execMode and the reason why",
                [&](Arg& arg) -> bool { return arg.msg(showExecModeAndReason() + '\n'); });
    mParser.opt("showNumThread", "", "show number of thread info",
                [&](Arg& arg) { return arg.fmtMsg("numTBBThread:%d\n", mcrt_common::getNumTBBThreads()); });
}

void
RenderContext::setSceneVarTextureCacheSize(const unsigned int sizeMB)
{
    scene_rdl2::rdl2::SceneVariables& sceneVars = mSceneContext->getSceneVariables();

    sceneVars.beginUpdate();
    sceneVars.set(scene_rdl2::rdl2::SceneVariables::sTextureCacheSizeMb, static_cast<int>(sizeMB));
    sceneVars.endUpdate();
}

std::string    
RenderContext::getSceneVarTextureCacheSize() const
{
    scene_rdl2::rdl2::SceneVariables& sceneVars = mSceneContext->getSceneVariables();
    int textureCacheSizeMb = sceneVars.get(scene_rdl2::rdl2::SceneVariables::sTextureCacheSizeMb);

    size_t byte = static_cast<size_t>(textureCacheSizeMb) * 1024 * 1024;

    std::ostringstream ostr;
    ostr << "texture_cache_size = " << textureCacheSizeMb << " (" << scene_rdl2::str_util::byteStr(byte) << ")";
    return ostr.str();
}

bool
RenderContext::saveSceneCommand(Arg& arg) const
{
    const std::string filename = (arg++)();

    if (!mSceneContext) {
        return arg.msg("sceneContext is empty. skip saveScene\n");
    }

    if (!arg.fmtMsg("saveScene to filename:%s start ...", filename.c_str())) {
        return false;
    }

    size_t elemsPerLine = 0; // Number of ascii array elements per-line, 0=unlimited

    try {
        scene_rdl2::rdl2::writeSceneToFile(*mSceneContext,
                                           filename, 
                                           false, // deltaEncoding
                                           false, // skipDefaults
                                           elemsPerLine);
    } catch (scene_rdl2::except::RuntimeError& e) {
        std::ostringstream ostr;
        ostr << "writeSceneToFile() failed. RuntimeError:" << e.what();
        arg.msg(ostr.str() + '\n');
        return false;
    } catch (...) {
        arg.msg("writeSceneToFile() failed.\n");
        return false;
    }

    return arg.msg("done\n");
}

std::string
RenderContext::showExecModeAndReason() const
{
    std::ostringstream ostr;
    ostr << "executionMode=";
    switch (mExecutionMode) {
    case ExecutionMode::SCALAR : ostr << "SCALAR"; break;
    case ExecutionMode::VECTORIZED : ostr << "VECTORIZED"; break;
    case ExecutionMode::XPU : ostr << "XPU"; break;
    default : ostr << "unknown"; break;
    }
    if (!mExecutionModeString.empty()) ostr << " (" << mExecutionModeString << ")";
    return ostr.str();
}

} // namespace rndr
} // namespace moonray

