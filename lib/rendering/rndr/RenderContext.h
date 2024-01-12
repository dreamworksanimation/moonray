// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#pragma once
#ifndef RENDERCONTEXT_H
#define RENDERCONTEXT_H

#include "RenderOptions.h"
#include "RenderPrepExecTracker.h"
#include "Types.h"

#include <scene_rdl2/common/fb_util/FbTypes.h>
#include <scene_rdl2/common/grid_util/Arg.h>
#include <scene_rdl2/common/grid_util/Parser.h>
#include <scene_rdl2/common/grid_util/RenderPrepStats.h>
#include <scene_rdl2/scene/rdl2/Layer.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace scene_rdl2 {
namespace rdl2 { class Camera;
                 class SceneContext; }

namespace fb_util {
    class ActivePixels;
    class VariablePixelBuffer;
}
}

namespace moonray {

struct ObjectData;

// Forward declarations.
namespace geom { class MotionBlurParams; class BakedMesh; class BakedCurves;
namespace internal { class Statistics; } }
namespace mcrt_common { class ThreadLocalState; }
namespace pbr { class CryptomatteBuffer; class DeepBuffer;
                struct FrameState;
                class PathIntegrator; class PixelFilter;
                class Scene; class Statistics; }
namespace shading { class State; class BsdfBuilder; }

namespace rt { class GeometryManager;
               struct GeometryManagerOptions;
               enum class ChangeFlag; }

namespace rndr {

struct FrameState;
struct RealtimeFrameStats;
class RenderDriver;
class RenderOutputDriver;
struct RenderPrepTimingStats;
class RenderProgressEstimation;
class RenderStats;
class ResumeHistoryMetaData;

/// This must be called at program startup before we create any RenderContexts.
/// It creates a single global render driver which is shared between all future
/// render contexts. Internally this also sets up all or our TLS objects.
void initGlobalDriver(const RenderOptions &initOptions);

/// This call is optional. If not called, the render driver will be destroyed
/// via global destructor of an internal smart pointer. It can be useful to call
/// this manually however if you want to clean up the render driver before the
/// global dtor stage (make sure nothing else is holding a ref to it).
void cleanUpGlobalDriver();

/**
 * The RenderContext is the top level access point to rendering services
 * provided by the MOONRAY renderer. To do rendering, you create a RenderContext.
 *
 * Right now the RenderContext will only check the RDL scene and DSO path when
 * its created. If you want to change scenes or change the DSO path, you'll
 * need to destroy the context and recreate it.
 *
 * Some options are only configurable until the first render, such as the
 * layer. Once we start rendering, we've built data structures
 * assuming such options won't change.
 *
 * The width and height *should* be configurable between each frame, as long
 * as you are not rendering while changing them.
 *
 * Updates to scene data or geometry can be applied by calling the appropriate
 * update function. These also need to be called while not rendering.
 */
class RenderContext
{
public:
    using Arg = scene_rdl2::grid_util::Arg;
    using Parser = scene_rdl2::grid_util::Parser; 

    enum class LoggingConfiguration
    {
        ATHENA_DISABLED,
        ATHENA_ENABLED
    };

    enum class RP_RESULT // renderPrep execution result
    {
        CANCELED, // canceled middle of the renderPrep phase
        FINISHED  // renderPrep phase has been completed
    };

    explicit RenderContext(RenderOptions& options,
                           std::stringstream* initMessages = nullptr);
    ~RenderContext();

    // This is for a call back function in order to report renderPrep progress information
    // to the downstream computation. This functionality is only used under arras context.
    using RenderPrepStatsCallBack = std::function<void(const scene_rdl2::grid_util::RenderPrepStats &rPrepStats)>;
    using RenderPrepCancelCallBack = std::function<bool()>; // return true if cancelled
    void setRenderPrepCallBack(const RenderPrepStatsCallBack &statsCallBack,
                               const RenderPrepCancelCallBack &cancelCallBack) {
        mRenderPrepExecTracker.setRenderPrepStatsCallBack(statsCallBack);
        mRenderPrepExecTracker.setRenderPrepCancelCallBack(cancelCallBack);
    }

    /**
     * Initializes the render context
     *
     */
    void initialize(std::stringstream &initMessages, LoggingConfiguration loggingConfig = LoggingConfiguration::ATHENA_DISABLED);

    /**
     * Updates the RDL scene with the given binary scene data. Must be called
     * between renders.
     *
     * @param   rdlData Binary RDL data to apply to the scene.
     */
    void updateScene(const std::string& manifest, const std::string& payload);

    /**
     * Updates the RDL scene with the given scene file. Must be called
     * between renders.
     *
     * @param   rdlData RDL file to apply to the scene.
    */
    void updateScene(const std::string& filename);

    /**
     * Sets the scene updated flag to true, which can happen if the
     * SceneContext is modified externally
     */
    void setSceneUpdated() { mSceneUpdated = true; }


    /**
     * Updates the geometry data to animate geometry objects in the scene. Must
     * be called between renders.
     *
     * @param   update  The geometry update data from the geometry message.
     */
    void updateGeometry(const std::vector<moonray::ObjectData>& updateData);

    /**
     * Bakes all meshes in the context.
     */
    bool bakeGeometry(std::vector<std::unique_ptr<geom::BakedMesh>>& bakedMeshes,
                      std::vector<std::unique_ptr<geom::BakedCurves>>& bakedCurves);

    /**
     * Starts rendering a frame based on the current state of the scene data.
     * Once rendering begins, no changes to the scene can be made until it is
     * stopped.
     */
    RP_RESULT startFrame();

    void invalidateAllTextureResources();
    void invalidateTextureResources(const std::vector<std::string>& resources);

    /**
     * Signals that you want to stop rendering a frame. This will trigger
     * cancelation logic in each of the active render threads. This function
     * doesn't block, so it should be followed by a call to stopFrame at a later
     * point in time which will block. Calling this is optional, but by calling
     * it early, the time spent blocking in stopFrame can be lessened.
     */
    void requestStop();
    void requestStopAsyncSignalSafe(); // can be called from signal handler

    /**
     * This API is used under arras context and sets a special mode that
     * progressive/checkpointResume rendering logic stops after initial render pass
     * (= very first time-estimation pass) and avoids useless following render passes.
     * This is important to achieve fast interactive response under arras context.
     */
    void requestStopAtFrameReadyForDisplay();

    /**
     * Stops rendering. Blocks until all the rendering threads are parked. The
     * rendered frame can be retrieved using snapshotRenderBuffer().
     */
    void stopFrame();

    double getLastFrameMcrtStartTime() const;

    /**
     * Request stop rendering condition at pass boundary. This API is only used by
     * multi-mcrt-computation context.
     */
    void requestStopRenderAtPassBoundary();

    /**
     * Snapshots the contents of the render buffer. The passed in buffer is
     * automatically resized if needed and any extrapolation and/or untiling
     * logic is handled internally.
     */
    void snapshotRenderBuffer(scene_rdl2::fb_util::RenderBuffer *renderBuffer, bool untile, bool parallel) const;

    /**
     * Snapshots the contents of the render ODD buffer only if renderOutputDriver has renderBufferOdd.
     * The passed in buffer is automatically resized if needed and any extrapolation and/or untiling
     * logic is handled internally.
     */
    void snapshotRenderBufferOdd(scene_rdl2::fb_util::RenderBuffer *renderBufferOdd, bool untile, bool parallel) const;

    /**
     * Snapshots the contents of the renderBuffer/weightBuffer w/ ActivePixels information
     * for ProgressiveFrame message related logic. So renderBuffer is not normalized by weight yet.
     * no resize, no extrapolation and no untiling logic is handled internally.
     * Just create snapshot data with properly constructed activePixels based on
     * difference between current and previous renderBuffer and weightBuffer.
     * renderBuffer/weightBuffer is tiled format and renderBuffer is not normalized by weight.
     */
    void snapshotDelta(scene_rdl2::fb_util::RenderBuffer *renderBuffer,
                       scene_rdl2::fb_util::FloatBuffer *weightBuffer,
                       scene_rdl2::fb_util::ActivePixels &activePixels,
                       bool parallel) const;

    /**
     * Snapshots the contents of the renderBufferOdd/weightRenderBufferOdd w/ ActivePixelsRenderBufferOdd information
     * for ProgressiveFrame message related logic. So renderBufferOdd is not normalized by weight yet.
     * no resize, no extrapolation and no untiling logic is handled internally.
     * Just create snapshot data with properly constructed activePixelsRenderBufferOdd based on
     * difference between current and previous renderBufferOdd and weightRenderBufferOdd.
     * renderBufferOdd/weightRenderBufferOdd is tiled format and renderBufferOdd is not normalized by weight.
     */
    void snapshotDeltaRenderBufferOdd(scene_rdl2::fb_util::RenderBuffer *renderBufferOdd,
                                      scene_rdl2::fb_util::FloatBuffer *weightBufferOdd,
                                      scene_rdl2::fb_util::ActivePixels &activePixelsOdd,
                                      bool parallel) const;

    /**
     * Snapshots the contents of the pixelInfoBuffer/pixelInfoWeightBuffer w/
     * ActivePixelsPixelInfo information for ProgressiveFrame message related logic.
     * Just create snapshot data with properly constructed activePixels based on
     * difference between current and previous pixelInfoBuffer and pixelInfoWeightBuffer.
     * pixelInfoBuffer/pixelInfoWeightBuffer are tiled format.
     */
    void snapshotDeltaPixelInfo(scene_rdl2::fb_util::PixelInfoBuffer *pixelInfoBuffer,
                                scene_rdl2::fb_util::FloatBuffer *pixelInfoWeightBuffer,
                                scene_rdl2::fb_util::ActivePixels &activePixelsPixelInfo,
                                bool parallel) const;

    /**
     * Snapshots the contents of the heatMapBuffer/heatMapWeightBuffer w/ ActivePixelsHeatMap information
     * for ProgressiveFrame message related logic.
     * Just create snapshot data with properly constructed activePixels based on
     * difference between current and previous heatMapBuffer/heatMapWeightBuffer.
     * Also create heatMapSecBuffer just for active pixels only.
     * heatMapBuffer/heatMapWeightBuffer/heatMapSecBuffer are tiled format.
     */
    void snapshotDeltaHeatMap(scene_rdl2::fb_util::HeatMapBuffer *heatMapBuffer,
                              scene_rdl2::fb_util::FloatBuffer *heatMapWeightBuffer,
                              scene_rdl2::fb_util::ActivePixels &activePixelsHeatMap,
                              scene_rdl2::fb_util::FloatBuffer *heatMapSecBuffer,
                              bool parallel) const;

    /**
     * Snapshots the contents of the weightBuffer w/ ActivePixelsWeightBuffer information
     * for ProgressiveFrame message related logic.
     * Just create snapshot data with properly constructed activePixels based on
     * difference between current and previous weightBuffer.
     * weightBuffer is tiled format.
     */
    void snapshotDeltaWeightBuffer(scene_rdl2::fb_util::FloatBuffer *weightBuffer,
                                   scene_rdl2::fb_util::ActivePixels &activePixelsWeightBuffer,
                                   bool parallel) const;

    /**
     * Snapshots the contents of the renderOutputBuffer(rodIndex)/renderOutputWeightBuffer(rodIndex) w/
     * ActivePixelsRenderOutput information for ProgressiveFrame message related logic.
     * Just create snapshot data with properly constructed activePixels based on
     * difference between current and previous renderOutputBuffer(rodIndex) and renderOutputWeightBuffer(rodIndex).
     * renderOutputBuffer(rodIndex) and renderOutputWeightBuffer(rodIndex) are tiled format.
     */
    void snapshotDeltaRenderOutput(unsigned int rodIndex,
                                   scene_rdl2::fb_util::VariablePixelBuffer *renderOutputBuffer,
                                   scene_rdl2::fb_util::FloatBuffer *renderOutputWeightBuffer,
                                   scene_rdl2::fb_util::ActivePixels &activePixelsRenderOutput,
                                   bool parallel,
                                   bool& denoiserAlbedoInput,
                                   bool& denoiserNormalInput) const;

    // Don't need to snapshot here, yet.
    const pbr::DeepBuffer* getDeepBuffer() const;

    pbr::CryptomatteBuffer* getCryptomatteBuffer();
    const pbr::CryptomatteBuffer* getCryptomatteBuffer() const;

    /**
     * Snapshots the contents of the pixel info buffer. The passed in buffer
     * is automatically resized if needed and any extrapolation and/or untiling
     * logic is handled internally.
     */
    void snapshotPixelInfoBuffer(scene_rdl2::fb_util::PixelInfoBuffer *pixelInfoBuffer, bool untile, bool parallel) const;

    /**
     * Snapshots the contents of the heat map buffer. The passed in buffer
     * is automatically resized if needed and any extrapolation and/or untiling
     * logic is handled internally.
     */
    void snapshotHeatMapBuffer(scene_rdl2::fb_util::HeatMapBuffer *heatMapBuffer, bool untile, bool parallel) const;

    /**
     * Snapshots the contents of the visibility buffer.
     */
    void snapshotVisibilityBuffer(scene_rdl2::fb_util::VariablePixelBuffer *visibilityBuffer,
                                  unsigned int aov, bool untile, bool parallel) const;

    /**
     * Snapshots the contents of a particular channel in the aov buffer
     */
    void snapshotAovBuffer(scene_rdl2::fb_util::VariablePixelBuffer *aovBuffer, unsigned int aov, bool untile, bool parallel) const;

    /**
     * Snapshot the contents of an aov into a 4 channel RenderBuffer.  Any channels unused by the
     * aov are set to zero.
     */
    void snapshotAovBuffer(scene_rdl2::fb_util::RenderBuffer *renderBuffer, unsigned int aov, bool until, bool parallel) const;

    /**
     * Snapshots the contents of the aov buffer
     */
    void snapshotAovBuffers(std::vector<scene_rdl2::fb_util::VariablePixelBuffer> &aovBuffers, bool untile, bool parallel) const;

    /**
     * Snapshots the contents of a particular DisplayFilter buffer
     */
    void snapshotDisplayFilterBuffer(scene_rdl2::fb_util::VariablePixelBuffer *DisplayFilterBuffer,
                                     unsigned int dfIdx, bool untile, bool parallel) const;

    /**
     * Snapshots the contents of all DisplayFilter buffers.
     */
    void snapshotDisplayFilterBuffers(std::vector<scene_rdl2::fb_util::VariablePixelBuffer> &displayFilterBuffers,
                                      bool untile, bool parallel) const;
    /**
     * Run all display filters at once. Should be called during batch or checkpoint mode,
     * after rendering is completed and before writing the output file.
     */
    void runDisplayFiltersBatch() const;

    void snapshotAovsForDisplayFilters(bool untile, bool parallel) const;

    /**
     * Snapshot the contents of a particular render output (roIndx).
     * If your render output makes used of the renderBuffer,
     * or heatMapBuffer, you must snapshot it before calling this function.
     * If not, these parameters can safely be set to nullptr.
     */
    void snapshotRenderOutput(scene_rdl2::fb_util::VariablePixelBuffer *buffer, int roIndx,
                              const scene_rdl2::fb_util::RenderBuffer *renderBuffer,
                              const scene_rdl2::fb_util::HeatMapBuffer *heatMapBuffer,
                              const scene_rdl2::fb_util::FloatBuffer *weightBuffer,
                              const scene_rdl2::fb_util::RenderBuffer *renderBufferOdd,
                              bool untile, bool parallel) const;

    /**
     * Snapshots the weight buffer. This contains the number of samples rendered per
     * pixel so far. Output is a single float per pixel. This API is designed debug purpose.
     */
    void snapshotWeightBuffer(scene_rdl2::fb_util::VariablePixelBuffer *outputBuffer, bool untile, bool parallel) const;

    /**
     * Snapshots the contents of the weight buffer only if renderOutputDriver has weightAOV.
     * The passed in buffer is automatically resized if needed and any extrapolation and/or untiling
     * logic is handled internally.
     */
    void snapshotWeightBuffer(scene_rdl2::fb_util::FloatBuffer *weightBuffer, bool untile, bool parallel) const;

    /**
     * Pixel info buffers are primarily useful for interactive rendering and may not
     * be allocated for batch rendering. This call returns whether one has been
     * allocated or not.
     */
    bool hasPixelInfoBuffer() const;

    /**
     * Returns true if there is something rendered at each pixel so we no longer
     * need to run the extrapolation algorithm.
     */
    bool areCoarsePassesComplete() const;

    /**
     * Returns true if we are executing renderPrep stage.
     */
    bool isRenderPrepRun() const { return mRenderPrepRun.load(); }

    /**
     * Returns true if we are currently within a startFrame/stopFrame pair.
     */
    bool isFrameRendering() const   { return mRendering; }

    /**
     * Returns true if the current frame has completed rendering fully.
     */
    bool isFrameComplete() const;

    /**
     * Returns true if the current frame has completed rendering by requestStopRenderAtPassBoundary()
     */
    bool isFrameCompleteAtPassBoundary() const;

    /**
     * Returns true if frame is ready to present in crude form. For this we assume
     * that at least 1 sample has been rendered in each tile.
     */
    bool isFrameReadyForDisplay() const;

    /**
     * Returns true if the render context has been initialized
     */
    bool isInitialized() const { return mHasBeenInit; }

    /**
     * Returns a percentage value of how much of the frame we've completed. The
     * submitted and total parameters are optional but give extra
     * information on how many rays have been submitted and retired.
     */
    float getFrameProgressFraction(std::size_t* submitted, std::size_t* total) const;
    RenderProgressEstimation *getFrameProgressEstimation() const;

    void setMultiMachineGlobalProgressFraction(float fraction); // for multi-machine configuration
    float getMultiMachineGlobalProgressFraction() const;

    bool isVectorizationDesired() const {
        return mOptions.getDesiredExecutionMode() == mcrt_common::ExecutionMode::VECTORIZED;
    }

    const std::vector<scene_rdl2::fb_util::Tile> *getTiles() const;

    void getTilesRenderedTo(scene_rdl2::util::BitArray& tiles) const;

    int getCurrentFrame() const;

    RenderMode getRenderMode() const    { return mOptions.getRenderMode(); }
    void setRenderMode(RenderMode mode) { mOptions.setRenderMode(mode); }

    FastRenderMode getFastRenderMode() const    { return mOptions.getFastRenderMode(); }
    void setFastRenderMode(FastRenderMode mode) { mOptions.setFastRenderMode(mode); }

    scene_rdl2::math::HalfOpenViewport getRezedRegionWindow() const;
    scene_rdl2::math::HalfOpenViewport getRezedApertureWindow() const;
    scene_rdl2::math::HalfOpenViewport getRezedSubViewport() const;

    const scene_rdl2::rdl2::Camera* getCamera() { return mCamera; }

    // Returns a read-only view into the RDL2 SceneContext.
    const scene_rdl2::rdl2::SceneContext& getSceneContext() const;

    // Returns active layer
    finline const scene_rdl2::rdl2::Layer *getActiveLayer() const;

    // Returns a modifiable RDL2 SceneContext which allows SceneVariable updates.
    // Only call this when not rendering.
    scene_rdl2::rdl2::SceneContext& getSceneContext();

    // Returns a read-only view into the pbr Scene.
    const pbr::Scene *getScene() const;

    // API to return the lights affecting the picked pixel.
    /// The x, y coordinates are assumed to be relative to the region window,
    /// i.e. (0, 0) would map to the left bottom most pixel rendered.
    void handlePickLightContributions(const int x, const int y,
            moonray::shading::LightContribArray& lightContributions) const;

    // API to return the materials affecting the picked pixel.
    /// The x, y coordinates are assumed to be relative to the region window,
    /// i.e. (0, 0) would map to the left bottom most pixel rendered.
    const scene_rdl2::rdl2::Material* handlePickMaterial(const int x, const int y) const;

    // API to return the geometry parts affecting the picked pixel.
    /// The x, y coordinates are assumed to be relative to the region window,
    /// i.e. (0, 0) would map to the left bottom most pixel rendered.
    const scene_rdl2::rdl2::Geometry* handlePickGeometryPart(const int x, const int y,
                                                 std::string& part) const;

    // API to return the geometry affecting the picked pixel.
    const scene_rdl2::rdl2::Geometry* handlePickGeometry(const int x, const int y) const;

    /// API to get the world location of a given pixel.
    /// The x, y coordinates are assumed to be relative to the region window,
    /// i.e. (0, 0) would map to the left bottom most pixel rendered.
    /// Not expected to return anything for distant and env lights.
    /// @param hitPoint [out] A pointer to a vec3 to fill in with the result location,
    /// if anything is found there.
    /// @returns true if something was hit, false if not.
    bool handlePickLocation(const int x, const int y, scene_rdl2::math::Vec3f *hitPoint) const;

    RenderStats& getSceneRenderStats()             { return *mRenderStats; }
    const RenderStats& getSceneRenderStats() const { return *mRenderStats; }

    const RenderPrepTimingStats &getRenderPrepTimingStatus() const { return *mRenderPrepTimingStats; }

    RealtimeFrameStats &getCurrentRealtimeFrameStats() const;
    void                commitCurrentRealtimeStats() const;

    const pbr::Statistics &getPbrStatistics() const { return *mPbrStatistics; }
    pbr::Statistics accumulatePbrStatistics() const;

    RenderOutputDriver* getRenderOutputDriver() { return mRenderOutputDriver.get(); }
    const RenderOutputDriver* getRenderOutputDriver() const { return mRenderOutputDriver.get(); }

    ResumeHistoryMetaData *getResumeHistoryMetaData() { return mResumeHistoryMetaData.get(); }
    const ResumeHistoryMetaData *getResumeHistoryMetaData() const { return mResumeHistoryMetaData.get(); }

    const std::string &getOnResumeScript() const { return mOnResumeScript; }

    /**
     * Returns a number that will change if there has been any Film activity.
     */
    unsigned getFilmActivity() const;

    unsigned getNumConsistentSamples() const;

    SamplingMode getSamplingMode() const { return mCachedSamplingMode; }
    finline void getAdaptiveSamplingParam(unsigned &minSamples, unsigned &maxSamples, float &targetError) const;

    //--------------------

    Parser& getParser() { return mParser; }

    void setForceCallStartFrame();
    void forceGuiCallStartFrameIfNeed(); // for moonray_gui

    using MsgHandler = std::function<void(const std::string &msg)>;

    bool needToSetExecTrackerMsgHandlerCallBack() const;
    MsgHandler getExecTrackerMsgHandlerCallBack() const;
    void setExecTrackerMsgHandlerCallBack(const MsgHandler &msgHandler);
    std::string execTrackerCancelInfoEncode() const; // for debug console
    void execTrackerCancelInfoDecode(const std::string &data);  // for debug console

    mcrt_common::ExecutionMode getCurrentExecutionMode() const { return mExecutionMode; }

    std::string getOiioStats(int level) const; // level=1~5

private:
    // Does any pre-render work, like building the spatial accelerator or
    // initializing any necessary libraries. Called in startFrame()
    RP_RESULT renderPrep(mcrt_common::ExecutionMode executionMode,
                         bool allowUnsupportedXPUFeatures);

    // Helper function which loads the scene into the SceneContext.
    void loadScene(std::stringstream &initMessages);

    // Helper function which creates a PBR scene.
    void createPbrScene();

    // Helper function that sets the active camera
    void initActiveCamera(const scene_rdl2::rdl2::Camera *camera);

    // Goes through all the provided Shaders and builds the primitive-
    // attribute table that describes all the primitive attributes that are
    // required to shade each Shader.
    void buildPrimitiveAttributeTables(const scene_rdl2::rdl2::Layer::RootShaderSet &rootShaders);

    // updates material with information about which of its primitive
    // attributes are also requested aovs in the aov schema
    void buildMaterialAovFlags();

    // adds the shading::Geometry object to geometry objects in the layer
    void buildGeometryExtensions();

    // Reloads procedurals for Geometries and RootShaders when there are changes.
    RP_RESULT loadGeometries(const rt::ChangeFlag);

    // Helper function for conditioning scene variables and other state into a
    // constant, fast to access structure for use within renderer inner loops.
    void buildFrameState(FrameState *fs, double frameStartTime, mcrt_common::ExecutionMode executionMode) const;

    // Called each frame in startFrame to update the internal state of the integrator.
    void updatePbrState(const FrameState &fs);

    // Helper function which builds and saves the debug ray database to disk.
    void buildAndSaveRayDatabase();

    // Resets shader stats and logs between frames
    void resetShaderStatsAndLogs();

    // Collect shader stats from SceneObjects and place them in mRenderStats
    void collectShaderStats();

    // Report any logging that occurred during shading
    void reportShadingLogs();

    // Report tessellation time for geometry primitives
    void reportGeometryTessellationTime();

    // Report memory for geometry primitives
    void reportGeometryMemory();

    // Report polycount/cv/curves count for geometry primitives
    void reportGeometryStatistics();

    // Increment the stored RDL frame number.
    void incrementCurfield();

    // Helper funtion to query motion blur related parameters
    // from the current active camera
    geom::MotionBlurParams getMotionBlurParams(bool bake=false) const;

    // Returns true if the loaded scene can be run in vectorized mode without
    // any loss of functionality. Returns false otherwise.  If the the
    // scene cannot run vectorized, the reason is written into the
    // reason parameter.
    bool canRunVectorized(std::string &reason) const;

    // Adds all of the MeshLight geometry to the layer, since it must be processed separately from the rest of the 
    // geometry in the scene. In order to keep track of the primitive attribute table, we create a dummy shader for
    // each mesh light geometry.
    void createMeshLightLayer();

    void parserConfigure();
    void setSceneVarTextureCacheSize(const unsigned int sizeMB);
    std::string getSceneVarTextureCacheSize() const;
    bool saveSceneCommand(Arg& arg) const;
    std::string showExecModeAndReason() const;

    // Options for rendering, such as the frame size, input/output files, etc.
    RenderOptions& mOptions;

    // The render driver which handles the details of rendering tiles and pixels.
    // Not owned by this render context.
    std::shared_ptr<RenderDriver> mDriver;

    float mMultiMachineGlobalProgressFraction {0.0f};

    /// GeometryManager manages all geometries in the scene for ray tracing.
    /// It handles proper update for changes and provides acceleration data
    /// structures for ray intersection queries.
    /// !!! Make sure this object is deleted before mSceneContext.
    std::unique_ptr<rt::GeometryManager> mGeometryManager;

    // The RDL scene data lives in the SceneContext.
    std::unique_ptr<scene_rdl2::rdl2::SceneContext> mSceneContext;

    // The PBR scene handles integrators, evaluation, and shading.
    std::unique_ptr<pbr::Scene> mPbrScene;

    // The pixel filter used for rendering the scene.
    scene_rdl2::rdl2::PixelFilterType mCachedPixelFilterType;
    float mCachedPixelFilterWidth;
    std::unique_ptr<pbr::PixelFilter> mPixelFilter;

    // The pixel sample map is used to multiply the number of samples per pixel
    std::unique_ptr<scene_rdl2::fb_util::PixelBuffer<float>> mPixelSampleMap;
    std::string mCachedPixelSampleMapName;
    float mMaxPixelSampleValue;

    // The integrator. computeRadiance is called by multiple threads.
    std::unique_ptr<pbr::PathIntegrator> mIntegrator;

    // The active RDL Camera we're looking through. The RenderContext does not
    // own this pointer, and should not delete it.
    const scene_rdl2::rdl2::Camera* mCamera;

    // The active RDL Layer we're rendering from. The RenderContext does not
    // own this pointer, and should not delete it.
    scene_rdl2::rdl2::Layer* mLayer;
    // The layer containing all geometries of the mesh light.
    scene_rdl2::rdl2::Layer* mMeshLightLayer;

    // Deep id channel names. The raw pointer is passed on to the FrameState,
    // but the RenderContext owns it.
    std::unique_ptr<std::vector<std::string>> mDeepIDChannelNames;

    int mCryptomatteNumLayers;

    // Sampling mode and parameters
    SamplingMode mCachedSamplingMode;
    unsigned mCachedMinSamplesPerPixel; // for adaptive sampling
    unsigned mCachedMaxSamplesPerPixel; // for adaptive sampling
    float mCachedTargetAdaptiveError;   // for adaptive sampling

    // Status flags.
    std::atomic_bool mRenderPrepRun;
    bool mRendering; // Are we actively rendering?
    bool mFirstFrame; // Is this the first frame the context has rendered?
    bool mSceneUpdated; // Has the scene been updated since the last render?
    bool mHasBeenInit; // Has the scene been initialized?
    bool mSceneLoaded; // Has the scene been loaded?
    bool mLogTime; // either first frame or previous frame logged timing
    bool mInfoLoggingEnabled;
    bool mDebugLoggingEnabled;

    RenderPrepExecTracker mRenderPrepExecTracker;

    /// Controls options for the GeometryManager.
    std::unique_ptr<rt::GeometryManagerOptions> mGeometryManagerOptions;

    // Stats collection/storage system
    std::unique_ptr<RenderStats> mRenderStats;

    // renderPrep related timing detail info
    std::unique_ptr<RenderPrepTimingStats> mRenderPrepTimingStats;

    // pbr related statistics.
    std::unique_ptr<pbr::Statistics> mPbrStatistics;

    // geom related statistics.
    std::unique_ptr<geom::internal::Statistics> mGeomStatistics;

    // RenderOutput object management
    std::unique_ptr<RenderOutputDriver> mRenderOutputDriver;

    // for Resume render
    std::string mOnResumeScript; // on resume script name
    std::unique_ptr<ResumeHistoryMetaData> mResumeHistoryMetaData; // current info for resume history

    // Functions to be used for shading and sampling in case of fatal errors at update
    static void fatalShade(const scene_rdl2::rdl2::Material* self, shading::TLState *tls,
                           const shading::State& state, shading::BsdfBuilder& bsdfBuilder);
    static void fatalSample(const scene_rdl2::rdl2::Map* self, shading::TLState *tls,
                            const shading::State& state, scene_rdl2::math::Color* sample);
    static void fatalSampleNormal(const scene_rdl2::rdl2::NormalMap* self, shading::TLState *tls,
                                  const shading::State& state, scene_rdl2::math::Vec3f* sample);

    typedef std::pair<std::string, std::string> SceneUpdateData;
    typedef std::vector<SceneUpdateData> UpdateQueue;
    UpdateQueue mUpdateQueue;

    std::mutex mMutexForceCallStartFrame;
    bool mForceCallStartFrame = false;
    Parser mParser;

    // final rendering execution mode and the reason why
    mcrt_common::ExecutionMode mExecutionMode; // for debugConsole command and McrtNodeInfo update
    std::string mExecutionModeString; // for debugConsole command
};

const scene_rdl2::rdl2::Layer *
RenderContext::getActiveLayer() const
{
    return getSceneContext().getSceneVariables().getLayer()->asA<scene_rdl2::rdl2::Layer>();
}

void
RenderContext::getAdaptiveSamplingParam(unsigned &minSamples,
                                        unsigned &maxSamples,
                                        float &targetError) const
{
    minSamples = mCachedMinSamplesPerPixel;
    maxSamples = mCachedMaxSamplesPerPixel;
    targetError = mCachedTargetAdaptiveError;
}

} // namespace rndr
} // namespace moonray

#endif // RENDERCONTEXT_H
