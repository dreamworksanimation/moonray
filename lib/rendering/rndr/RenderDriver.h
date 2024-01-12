// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

//
//
#pragma once
#include "AdaptiveRenderTileInfo.h"
#include "CheckpointController.h"
#include "DisplayFilterDriver.h"
#include "FrameState.h"
#include "Film.h"
#include "RenderProgressEstimation.h"
#include "RenderStatistics.h"
#include "RenderTimingRecord.h"
#include "TileWorkQueue.h"
#include "TileScheduler.h"
#include "Types.h"
#include "Util.h"
#include <moonray/rendering/rndr/adaptive/ActivePixelMask.h>
#include <moonray/rendering/pbr/camera/StereoView.h>
#include <moonray/rendering/pbr/core/XPUOcclusionRayQueue.h>
#include <moonray/rendering/pbr/core/XPURayQueue.h>
#include <moonray/common/mcrt_util/AlignedElementArray.h>

#include <scene_rdl2/common/fb_util/TileExtrapolation.h>
#include <scene_rdl2/common/grid_util/Arg.h>
#include <scene_rdl2/common/grid_util/FloatValueTracker.h>
#include <scene_rdl2/common/grid_util/Parser.h>
#include <scene_rdl2/render/util/AtomicFloat.h>

#include <tbb/task_scheduler_init.h>

//#define SINGLE_THREAD_CRAWLALLPIXELS

#ifndef SINGLE_THREAD_CRAWLALLPIXELS
#include <tbb/parallel_for.h>
#endif // end !SINGLE_THREAD_CRAWLALLPIXELS

#include <atomic>
#include <array>
#include <memory>
#include <mutex>
#include <thread>

// Uncomment to force single threaded rendering.
#ifdef DEBUG
//#define FORCE_SINGLE_THREADED_RENDERING
#endif

// Enable runtime verify adaptive sampling passes construction logic.
// This should be commented out for relese version.
//#define VERIFY_ADAPTIVE_SAMPLING_PASSES_CONVERSION_LOGIC

namespace moonray {

namespace pbr {
class CryptomatteBuffer;
class DeepBuffer;
class TLState;
}
namespace rdl2 {
class RenderOutput;
}

namespace mcrt_common {
class ThreadLocalState;
}

namespace fb_util { class ActivePixels; }

namespace rndr {

struct RenderSamplesParams;
struct TileGroup;
class AdaptiveRenderTileInfo;
class RenderOptions;
class TileSampleSpecialEvent;
class TileScheduler;
class VariablePixelBuffer;

// Used to signify that we should wrap up rendering this frame ASAP.
// This flag used to live in RenderDriver. It doesn't anymore for two reasons:
// * We want atomic behavior. While it's possible to do atomic operations in ISPC, doing them on the same data structure
//   would require a lot of effort (i.e. we would have to hand-roll a lot of the atomic operations and pass around a
//   normal bool). Now we just access the boolean from an extern "C" function and let C++ worry about the atomacity.
// * We want to align this on a cache line. False sharing caused a simple render to go from 25 seconds to more than a
//   minute and a half. Unfortunately, ICC compilation results in an error when aligning this member variable, saying
//   that our dynamic allocation of RenderDriver is mis-aligned. I believe this is an ICC bug, as an aligned member
//   variable should only change the size of RenderDriver, and not the allocation alignment requirements of RenderDriver
//   itself (we're not changing the alignment of RenderDriver, only the size).
struct alignas(64) CancelFlag
{
    CancelFlag() noexcept
    : mCanceled(false)
    {
    }

    bool isCanceled() const noexcept
    {
        return mCanceled.load(std::memory_order_relaxed);
    }

    void set(bool v) noexcept
    {
        mCanceled.store(v, std::memory_order_relaxed);
    }

    std::atomic_bool mCanceled;
};

extern CancelFlag gCancelFlag;

//-----------------------------------------------------------------------------

class RenderDriver
{
public:
    using Arg = scene_rdl2::grid_util::Arg;
    using Parser = scene_rdl2::grid_util::Parser; 

    ~RenderDriver();

    unsigned            getWidth() const                { return mUnalignedW; }
    unsigned            getHeight() const               { return mUnalignedH; }

    Film &              getFilm()           { return *mFilm; }
    const Film &        getFilm() const     { return *mFilm; }

    // This function must be called from the same thread as Start/Stop.
    // Also, the returned vector is only valid until Stop is called.
    const std::vector<scene_rdl2::fb_util::Tile> *getTiles() const;

    // Returns either:
    // - MAX_RENDER_PASSES if all passes are fine passes, or
    // - the index of the last coarse pass.
    unsigned            getLastCoarsePassIdx() const    { return mLastCoarsePassIdx; }

    const FrameState &  getFrameState() const           { return mFs; }

    // We copy frame state internally so it doesn't need to persist on the caller side.
    void                startFrame(const FrameState &fs);
    void                requestStop();
    void                requestStopAsyncSignalSafe(); // for call from signal handler
    void                requestStopAtFrameReadyForDisplay();
    void                stopFrame();

    void requestStopAtPassBoundary() { mRenderStopAtPassBoundary = true; } // called by progmcrt_computation

    //
    // Here are the steps which happen internally to produce a linear render
    // buffer:
    //
    // 1)  If we're still render coarse passes then we need to extrapolate the
    //     buffer to fill in the gaps.
    // 2)  Radiance in rgb and alpha is a is normalized by dividing by the
    //     corresponding weight. (This is combined with the extrapolation step
    //     in the current implementation.)
    // 3)  The buffer is still in tiled format. It needs to be detiled before we
    //     can display it.
    //
    // All of these 3 steps happen inside of these snapshot functions. It's up
    // to the calling code to do any sort of synchronization necessary. This call
    // just returns whatever is in the render buffer currently.
    //
    void snapshotRenderBuffer(scene_rdl2::fb_util::RenderBuffer *outputBuffer,
                              bool untile,
                              bool parallel) const;

    // Snapshot the renderBufferOdd buffer which uses by adaptive sampling logic
    void snapshotRenderBufferOdd(scene_rdl2::fb_util::RenderBuffer *outputBuffer,
                                 bool untile,
                                 bool parallel) const;

    //
    // Snapshots the weight buffer. This contains the number of samples rendered per
    // pixel so far.
    //
    void snapshotWeightBuffer(scene_rdl2::fb_util::VariablePixelBuffer *outputBuffer,
                              bool untile,
                              bool parallel) const;

    void snapshotWeightBuffer(scene_rdl2::fb_util::FloatBuffer *outputBuffer,
                              bool untile,
                              bool parallel) const;

    // Don't need to snapshot here, yet.
    const pbr::DeepBuffer* getDeepBuffer() const;

    pbr::CryptomatteBuffer*       getCryptomatteBuffer();
    const pbr::CryptomatteBuffer* getCryptomatteBuffer() const;

    bool                snapshotPixelInfoBuffer(scene_rdl2::fb_util::PixelInfoBuffer *outputBuffer,
                                                bool untile,
                                                bool parallel) const;

    bool                snapshotHeatMapBuffer(scene_rdl2::fb_util::HeatMapBuffer *outputBuffer,
                                              bool untile,
                                              bool parallel) const;

    void snapshotAovBuffer(scene_rdl2::fb_util::VariablePixelBuffer *outputBuffer,
                           int numConsistentSamples,
                           unsigned int aov,
                           bool untile,
                           bool parallel,
                           bool fulldump) const;

    // Snapshot the contents of an aov into a 4 channel RenderBuffer. (for denoising)
    void snapshotAovBuffer(scene_rdl2::fb_util::RenderBuffer *outputBuffer,
                           int numConsistentSamples,
                           unsigned aov,
                           bool untile,
                           bool parallel) const;

    void snapshotDisplayFilterBuffer(scene_rdl2::fb_util::VariablePixelBuffer *outputBuffer,
                                     unsigned int dfIdx,
                                     bool untile,
                                     bool parallel) const;

    void snapshotVisibilityBuffer(scene_rdl2::fb_util::VariablePixelBuffer *outputBuffer,
                                  unsigned int aov,
                                  bool untile,
                                  bool parallel,
                                  bool fulldumpVisibility) const;

    //
    // Whereas snapshotRenderBuffer and snapshotPixelInfoBuffer are buffer
    // specific, this is a general purpose version which can be used on any type
    // of buffer.
    //
    // scratchBuffer is optional. If it's omitted then we'll attempt to reuse
    // one of the internal mExtrapolationBuffers inside of this class instead.
    //
    // No-op hasData function (meaning pixel has data is true):
    //     [](const SRC_PIXEL_TYPE &p, unsigned ofs) -> uint32_t { return 0xffffffff; }
    // No-op pixelXform function:
    //     [](const SRC_PIXEL_TYPE &p, unsigned ofs) -> const DST_PIXEL_TYPE & { return p; }
    //
    template<typename DST_PIXEL_TYPE, typename SRC_PIXEL_TYPE,
             typename HAS_DATA_FUNC, typename PIXEL_XFORM_FUNC>
    inline void         snapshotBuffer(scene_rdl2::fb_util::PixelBuffer<DST_PIXEL_TYPE> *outputBuffer,
                                       const scene_rdl2::fb_util::PixelBuffer<SRC_PIXEL_TYPE> &srcBuffer,
                                       scene_rdl2::fb_util::PixelBuffer<SRC_PIXEL_TYPE> *scratchBuffer,
                                       bool extrapolate,
                                       bool untile,
                                       bool parallel,
                                       HAS_DATA_FUNC hasData,
                                       PIXEL_XFORM_FUNC pixelXform) const;

    //
    // Creates snapshot renderBuffer/weightBuffer data w/ activePixels information
    // for ProgressiveFrame message related logic.
    // No resize, no extrapolation, no untiling logic is handled internally.
    // Simply just create snapshot data with properly constructed activePixels data based on
    // difference between current and previous renderBuffer and weightBuffer.
    // renderBuffer/weightBuffer is tiled format and renderBuffer is not normalized by weight
    //
    void snapshotDelta(scene_rdl2::fb_util::RenderBuffer *renderBuffer,
                       scene_rdl2::fb_util::FloatBuffer *weightBuffer,
                       scene_rdl2::fb_util::ActivePixels &activePixels,
                       bool parallel) const;

    //
    // Creates snapshot renderBufferOdd/weightRenderBufferOdd data w/ activePixelsRenderBufferOdd information
    // for ProgressiveFrame message related logic.
    // No resize, no extrapolation, no untiling logic is handled internally.
    // Simply just create snapshot data with properly constructed activePixelsRenderBufferOdd data based on
    // difference between current and previous renderBufferOdd and weightRenderBufferOdd.
    // renderBufferOdd/weightRenderBufferOdd is tiled format and renderBufferOdd is not normalized by weight
    //
    void snapshotDeltaRenderBufferOdd(scene_rdl2::fb_util::RenderBuffer *renderBufferOdd,
                                      scene_rdl2::fb_util::FloatBuffer *weightRenderBufferOdd,
                                      scene_rdl2::fb_util::ActivePixels &activePixelsRenderBufferOdd,
                                      bool parallel) const;

    //
    // Creates snapshot pixelInfoBuffer/pixelInfoWeightBuffer data w/ activePixelsPixelInfo information
    // for ProgressiveFrame message related logic.
    // Simply just create snapshot data with properly constructed activePixels data based on
    // difference between current and previous pixelInfoBuffer and pixelInfoWeightBuffer.
    // pixelInfoBuffer/pixelInfoWeightBuffer is tiled format.
    //
    void snapshotDeltaPixelInfo(scene_rdl2::fb_util::PixelInfoBuffer *pixelInfoBuffer,
                                scene_rdl2::fb_util::FloatBuffer *pixelInfoWeightBuffer,
                                scene_rdl2::fb_util::ActivePixels &activePixelsPixelInfo,
                                bool parallel) const;

    //
    // Creates snapshot heatMapBuffer/heatMapWeightBuffer data w/ activePixelsHeatMap information
    // for ProgressiveFrame message related logic.
    // Simply just create snapshot data with properly constructed activePixels data based on
    // difference between current and previous heatMapBuffer/heatMapWeightBuffer.
    // Also create heatMapSecBuffer just for active pixels only.
    // heatMapBuffer/heatMapWeightBuffer/heatMapSecBuffer are tiled format.
    //
    void snapshotDeltaHeatMap(scene_rdl2::fb_util::HeatMapBuffer *heatMapBuffer,
                              scene_rdl2::fb_util::FloatBuffer *heatMapWeightBuffer,
                              scene_rdl2::fb_util::ActivePixels &activePixelsHeatMap,
                              scene_rdl2::fb_util::FloatBuffer *heatMapSecBuffer,
                              bool parallel) const;

    //
    // Creates snapshot weightBuffer data w/ activePixelsWeightBuffer information
    // for ProgressiveFrame message related logic.
    // Simply just create snapshot data with properly constructed activePixels data based on
    // difference between current and previous weightBuffer.
    // weightBuffer is tiled format.
    //
    void snapshotDeltaWeightBuffer(scene_rdl2::fb_util::FloatBuffer *weightBuffer,
                                   scene_rdl2::fb_util::ActivePixels &activePixelsWeightBuffer,
                                   bool parallel) const;

    //
    // Creates snapshot renderOutputBuffer(aovIndex)/renderOutputWeightBuffer(aovIndex) data w/
    // activePixelsRenderOutput information for ProgressiveFrame message related logic.
    // Simply just create snapshot data with properly constructed activePixels data based on
    // difference between current and previous renderOutputBuffer(aovIndex) and renderOutputWeightBuffer(aovIndex).
    // renderOutputBuffer(aovIndex)/renderOutputWeightBuffer(aovIndex) are tiled format.
    //
    void snapshotDeltaAov(unsigned aovIndex,
                          scene_rdl2::fb_util::VariablePixelBuffer *renderOutputBuffer,
                          scene_rdl2::fb_util::FloatBuffer *renderOutputWeightBuffer,
                          scene_rdl2::fb_util::ActivePixels &activePixelsRenderOutput,
                          bool parallel) const;
    // This is a specially designed Visibility AOV buffer version of snapshotDeltaAov()
    void snapshotDeltaAovVisibility(unsigned aovIndex,
                                    scene_rdl2::fb_util::VariablePixelBuffer *renderOutputBuffer,
                                    scene_rdl2::fb_util::FloatBuffer *renderOutputWeightBuffer,
                                    scene_rdl2::fb_util::ActivePixels &activePixelsRenderOutput,
                                    bool parallel) const;
    // This is a specially designed DisplayFilter version of snapshotDelta.
    void snapshotDeltaDisplayFilter(unsigned dfIdx,
                                    scene_rdl2::fb_util::VariablePixelBuffer *renderOutputBuffer,
                                    scene_rdl2::fb_util::FloatBuffer *dstRenderOutputWeightBuffer,
                                    scene_rdl2::fb_util::ActivePixels &activePixelsRenderOutput,
                                    bool parallel) const;

    //
    // Returns true if the last frame has been filled in with the minimal amount
    // of pixels required for display. Usually this means that at least one sample
    // in every 8x8 pixel block has been completed. We need to run extrapolation
    // on sparsely filled buffers before displaying them. When this gets set
    // internally depends on the current render mode.
    //
    inline bool         isReadyForDisplay() const       { return mReadyForDisplay; }

    //
    // Returns true if there is something rendered at each pixel so we no longer
    // need to run the extrapolation algorithm.
    //
    inline bool         areCoarsePassesComplete() const { return mCoarsePassesComplete; }

    //
    // Returns true if the last frame triggered was able to render to completion
    // without interruption.
    //
    inline bool         isFrameComplete() const         { return mFrameComplete; }

    //
    // Returns true if the last frame triggered was completed by requestStopAtPassBoundary()
    //
    bool isFrameCompleteAtPassBoundary() const { return mFrameCompleteAtPassBoundary; }

    //
    // Returns a percentage value of how much of the current pass we've completed.
    // The submitted and total parameters are for retrieving data and are optional.
    //
    float getPassProgressPercentage(unsigned passIdx, size_t *submitted, size_t *total) const;
    float getOverallProgressFraction(bool activeRendering, size_t *submitted, size_t *total) const;

    //
    // globalProgressFraction is the whole progress fraction value under multi-machine configuration.
    // Each mcrt_computation receives a globalProgressFraction update from merge computation with
    // reasonable latency (~200ms range).
    //
    void setMultiMachineGlobalProgressFraction(float fraction);
    float getMultiMachineGlobalProgressFraction() const { return mMultiMachineGlobalProgressFraction; }

    //
    // Various stages of recording debug rays. Each state can only be set by a
    // single thread so no mutex needed.
    //
    enum DebugRayState
    {
        READY,              // set from main thread
        REQUEST_RECORD,     // set from main thread
        RECORDING,          // set from render thread
        RECORDING_COMPLETE, // set from render thread
        BUILDING,           // set from main thread
        NUM_DEBUG_RAY_STATES,
    };

    //
    // The state machine mechanism for recording debug rays is dependent on
    // there only being a single thread which has permission to change to a
    // particular state. The thread permissions are shown in the comments in
    // the DebugRayState enum. As long as we follow this one rule, we don't
    // need a mutex around the state. For added verification this is true, the
    // switchDebugRayState function requires the client to pass in what it
    // expects the current state to be. By doing this we can check that no other
    // threads have changed the value in the meantime.
    //
    DebugRayState getDebugRayState() const  { return mDebugRayState.load(std::memory_order_relaxed); }
    void switchDebugRayState(DebugRayState oldState, DebugRayState newState);

    void   setLastFrameUpdateDurationOffset(double v) { mUpdateDurationOffset = v; }
    double getLastFrameUpdateDurationOffset() const   { return mUpdateDurationOffset; }
    double getLastFrameMcrtStartTime() const          { return mMcrtStartTime; }
    double getLastFrameMcrtDuration() const           { return mMcrtDuration; }
    double getLastFrameMcrtUtilization() const        { return mMcrtUtilization; }

    RealtimeFrameStats &getCurrentRealtimeFrameStats();
    void                commitCurrentRealtimeStats();

    // This will clear all the realtime stats information recorded so far and
    // start recording from scratch again.
    void                resetRealtimeStats();

    // This writes out the existing realtime stats to REALTIME_FRAME_STATS_LOGFILE
    // and clears the recorded data.
    void                saveRealtimeStats();

    // Get RenderFrameTimingRecord which uses Realtime/ProgressiveCheckpoint renderMode
    RenderFrameTimingRecord &getRenderFrameTimingRecord() { return mTimeRec; }
    RenderProgressEstimation &getRenderProgressEstimation() { return mProgressEstimation; }

    bool revertFilmData(RenderOutputDriver *renderOutputDriver, const FrameState &fs, unsigned &resumeTileSamples);

    template <typename F> void crawlAllTiledPixels(F pixelFunc) const;

    // The RenderDriver owns the XPU queues but other objects like TLState may
    // have pointers to them so they can queue up rays.
    void createXPUQueues();
    unsigned flushXPUQueues(mcrt_common::ThreadLocalState *tls, scene_rdl2::alloc::Arena *arena);
    void freeXPUQueues();

    // RenderDriver owns DisplayFilterDriver
    const DisplayFilterDriver& getDisplayFilterDriver() const { return mDisplayFilterDriver; }

    const CheckpointController &getCheckpointController() const { return mCheckpointController; }

    // Convert numCheckpointFiles value to checkpoint qualitySteps
    static int convertTotalCheckpointToQualitySteps(SamplingMode mode,
                                                    int checkpointStartSPP,
                                                    int maxSPP,
                                                    int numCheckpointFiles,
                                                    std::string &logMessage);
    // called from unitTest
    static bool verifyKJSequenceTable(const unsigned maxSampleId, std::string *tblStr = nullptr);
    static bool verifyTotalCheckpointToQualitySteps(SamplingMode mode,
                                                    int checkpointStartSPP,
                                                    int maxSPP,
                                                    int userDefinedTotalCheckpointFiles,
                                                    int verifyQSteps,
                                                    std::string *verifyErrorMessage,
                                                    std::string *logMessage = nullptr);
    static bool verifyTotalCheckpointToQualityStepsExhaust(SamplingMode mode,
                                                           int maxSPP,
                                                           int fileCountEndCap, // 0:disable end cap logic
                                                           int startSPPEndCap, // -1:disable end cap logic
                                                           bool liveMessage,
                                                           bool deepVerifyMessage,
                                                           std::string *verifyErrorMessage,
                                                           bool multiThreadA,
                                                           bool multiThreadB);

    void pushRenderPrepTime(const float sec) { mRenderPrepTime.set(sec); }

    Parser& getParser() { return mParser; }

private:
    friend void initRenderDriver(const mcrt_common::TLSInitParams &initParams);

    using UIntTable = std::vector<unsigned>;

    explicit RenderDriver(const mcrt_common::TLSInitParams &initParams);

    enum RenderThreadState
    {
        UNINITIALIZED,          // set from main thread
        READY_TO_RENDER,        // set from main thread
        REQUEST_RENDER,         // set from main thread
        RENDERING,              // set from render thread
        RENDERING_DONE,         // set from render thread
        KILL_RENDER_THREAD,     // set from main thread
        DEAD,                   // set from render thread
        NUM_RENDER_THREAD_STATES,
    };

    // Progress related helpers:
    void                transferAllProgressFromSingleTLS(pbr::TLState *tls);
    void                setReadyForDisplay();
    void                setCoarsePassesComplete();
    void                setFrameComplete();

    // Special tile scheduler setup function for multi-machine configuration
    void setupMultiMachineTileScheduler(unsigned pixW, unsigned pixH);

    // snapshot renderBuffer or renderBufferOdd
    void snapshotRenderBufferSub(scene_rdl2::fb_util::RenderBuffer *outputBuffer,
                                 bool untile, bool parallel, bool oddBuffer) const;

    // snapshot delta related private functions
    void snapshotDeltaAovFloat(unsigned aovIdx,
                               scene_rdl2::fb_util::VariablePixelBuffer *dstRenderOutputBuffer,
                               scene_rdl2::fb_util::FloatBuffer *dstRenderOutputWeightBuffer,
                               scene_rdl2::fb_util::ActivePixels &activePixelsRenderOutput,
                               bool parallel) const;
    void snapshotDeltaAovFloat2(unsigned aovIdx,
                                scene_rdl2::fb_util::VariablePixelBuffer *dstRenderOutputBuffer,
                                scene_rdl2::fb_util::FloatBuffer *dstRenderOutputWeightBuffer,
                                scene_rdl2::fb_util::ActivePixels &activePixelsRenderOutput,
                                bool parallel) const;
    void snapshotDeltaAovFloat3(unsigned aovIdx,
                                scene_rdl2::fb_util::VariablePixelBuffer *dstRenderOutputBuffer,
                                scene_rdl2::fb_util::FloatBuffer *dstRenderOutputWeightBuffer,
                                scene_rdl2::fb_util::ActivePixels &activePixelsRenderOutput,
                                bool parallel) const;
    void snapshotDeltaAovFloat4(unsigned aovIdx,
                                scene_rdl2::fb_util::VariablePixelBuffer *dstRenderOutputBuffer,
                                scene_rdl2::fb_util::FloatBuffer *dstRenderOutputWeightBuffer,
                                scene_rdl2::fb_util::ActivePixels &activePixelsRenderOutput,
                                bool parallel) const;

    void snapshotAovsForDisplayFilters(bool untile, bool parallel) const;

    // revert film related
    void denormalizeAovBuffer(int numConsistentSamples, unsigned int aov);
    void denormalizeBeautyOdd();
    void denormalizeAlphaOdd();
    void zeroWeightMaskAovBuffer(unsigned int aov);
    void zeroWeightMaskVisibilityBuffer(unsigned int aov);
    bool copyBeautyBuffer();

    //
    // Functions which do the rendering of the actual frame:
    //
    static void         renderThread(RenderDriver *driver,
                                     const mcrt_common::TLSInitParams &initParams);

    static void         renderFrame(RenderDriver *driver, const FrameState &fs);

    // Returns true if we ran to completion or false if we were canceled.
    static bool         batchRenderFrame(RenderDriver *driver, const FrameState &fs);

    // Returns true if we ran to completion or false if we were canceled.
    static bool         progressiveRenderFrame(RenderDriver *driver,
                                               const FrameState &fs);

    // No cancelation control for realtime render
    static void realtimeRenderFrame(RenderDriver *driver, const FrameState &fs);

    //------------------------------

    enum class RenderPassesResult : int {
        ERROR_OR_CANCEL,
        OK_AND_TRY_NEXT,
        STOP_AT_PASS_BOUNDARY,
        COMPLETED_ALL_SAMPLES
    };

    static std::vector<unsigned> createKJSequenceTable(const unsigned maxSampleId);

    static unsigned revertFilmObjectAndResetWorkQueue(RenderDriver *driver, const FrameState &fs);

    // Returns true if we ran to completion or false if we were canceled.
    static bool progressCheckpointRenderFrame(RenderDriver *driver, const FrameState &fs,
                                              const unsigned progressCheckpointStartTileSampleId,
                                              const TileSampleSpecialEvent *tileSampleSpecialEvent = nullptr);
    static RenderPassesResult
        progressCheckpointRenderStint(RenderDriver *driver, const FrameState &fs,
                                      const unsigned processStartTileSampleId,
                                      const TileSampleSpecialEvent *tileSampleSpecialEvent,
                                      unsigned &startSampleId, unsigned &endSampleId);

    static bool isProgressCheckpointComplete(RenderDriver *driver, const FrameState &fs);
    static unsigned calcCheckpointStintStartEndSampleId(RenderDriver *driver,
                                                        const FrameState &fs,
                                                        const double remainingTime, // only used by timeBased
                                                        const UIntTable *adaptItrPixSampleIdTbl,
                                                        unsigned &startSampleId, unsigned &endSampleId);
    static double estimateSamplesPerTile(RenderDriver *driver, const FrameState &fs,
                                         const double remainingTime,
                                         const UIntTable *adaptiveIterationPixSampleIdTable,
                                         const unsigned startTileSampleId);

    static RenderPassesResult 
        checkpointRenderMiniStintLoop(RenderDriver *driver,
                                      const FrameState &fs,
                                      const UIntTable *adaptiveIterationPixSampleIdTable,
                                      const TileSampleSpecialEvent *tileSampleSpecialEvent,
                                      unsigned &startSampleId, unsigned &endSampleId);
    static UIntTable calcCheckpointMiniStintStartEndId(const unsigned startSampleId,
                                                       const unsigned endSampleId,
                                                       const TileSampleSpecialEvent *tileSampleSpecialEvent);
    static bool isRequiredSpecialEvent(const UIntTable &specialTileSampleIdArray,
                                       const unsigned endSampleId);

    static RenderPassesResult
        checkpointRenderMicroStintLoop(RenderDriver *driver,
                                       const FrameState &fs,
                                       const UIntTable *adaptiveIterationPixSampleIdTable,
                                       unsigned &startSampleId, unsigned &endSampleId);
    static RenderPassesResult
        checkpointRenderMicroStint(RenderDriver *driver,
                                   const FrameState &fs,
                                   const UIntTable *adaptiveIterationPixSampleIdTable,
                                   const unsigned &startSampleId, const unsigned &endSampleId);

    static void checkpointFileOutput(RenderDriver *driver, const FrameState &fs, const unsigned endSampleId);

#   ifdef VERIFY_ADAPTIVE_SAMPLING_PASSES_CONVERSION_LOGIC
    static bool verifyPassesLogicForAdaptiveSampling();
#   endif // end VERIFY_ADAPTIVE_SAMPLING_PASSES_CONVERSION_LOGIC

    static std::string checkpointModeStr(const CheckpointMode &mode);

    static RenderPassesResult renderPasses(RenderDriver *driver, const FrameState &fs, bool allowCancelation);

    // This is called per tile.
    static void runDisplayFiltersTile(RenderDriver *driver,
                                      size_t tileIdx,
                                      size_t threadId);
    // This is called at the end of a pass.
    static void runDisplayFiltersEndOfPass(RenderDriver *driver, const FrameState &fs);

    static unsigned     renderTiles(RenderDriver *driver,
                                    mcrt_common::ThreadLocalState *topLevelTls,
                                    const TileGroup &group);

    //
    // Functions which do the rendering of the one tile
    //
    static bool renderTile(RenderDriver *driver, mcrt_common::ThreadLocalState *tls, const TileGroup &group,
                           RenderSamplesParams &params, pbr::DeepBuffer *deepBuffer,
                           pbr::CryptomatteBuffer *cryptomatteBuffer, unsigned &processedSampleTotal);
    static bool renderTileAdaptiveStage(RenderDriver* driver,
                                        mcrt_common::ThreadLocalState* tls,
                                        const TileGroup& group,
                                        RenderSamplesParams& params,
                                        pbr::DeepBuffer* deepBuffer,
                                        pbr::CryptomatteBuffer* cryptomatteBuffer,
                                        unsigned& processedSampleTotal);
    static AdaptiveRenderTileInfo::Stage updateTileCondition(RenderDriver *driver,
                                                             const TileGroup &group,
                                                             RenderSamplesParams &params,
                                                             const unsigned endSampleIdx);
    template <bool adaptive>
    static bool renderTileUniformSamples(RenderDriver *driver, mcrt_common::ThreadLocalState *tls,
                                         const TileGroup &group,
                                         RenderSamplesParams &params, pbr::DeepBuffer *deepBuffer,
                                         pbr::CryptomatteBuffer *cryptomatteBuffer,
                                         const unsigned startSampleIdx, const unsigned endSampleIdx,
                                         unsigned &processedSampleTotal,
                                         const ActivePixelMask& inputRegion = ActivePixelMask::all());
    static bool renderPixelScalarSamples(pbr::TLState *pbrTls,
                                         const unsigned startSampleIdx, const unsigned endSampleIdx,
                                         RenderSamplesParams *params);
    static bool renderPixelScalarSamplesFast(pbr::TLState *pbrTls,
                                             const unsigned startSampleIdx, const unsigned endSampleIdx,
                                             RenderSamplesParams *params);
    static bool renderPixelVectorSamples(pbr::TLState *pbrTls,
                                         const unsigned startSampleIdx, const unsigned endSampleIdx,
                                         RenderSamplesParams *params,
                                         const TileGroup &group,
                                         const pbr::DeepBuffer *deepBuffer,
                                         pbr::CryptomatteBuffer *cryptomatteBuffer);
    static void computePixelInfo(RenderDriver *driver, mcrt_common::ThreadLocalState *tls,
                                 Film &film, const unsigned px, const unsigned py);
    static unsigned computeTotalNumSamples(const rndr::FrameState &fs, const unsigned ifilm, unsigned px, unsigned py);

    void parserConfigure();

    std::string showInitFrameControl() const;
    std::string showMultiMachineCheckpointMainLoopInfo() const;

    //------------------------------

    // Frame specific data for rendering just the current frame.
    FrameState          mFs;

    // Actual width and height of final buffer, pre-alignment.
    unsigned            mUnalignedW;
    unsigned            mUnalignedH;

    // Used to track when we need to recompute the passes and tiles.
    unsigned                   mCachedSamplesPerPixel;
    RenderMode                 mCachedRenderMode;
    bool                       mCachedRequiresDeepBuffer;
    bool                       mCachedRequiresCryptomatteBuffer;
    bool                       mCachedGeneratePixelInfo;
    std::vector<unsigned int>  mCachedAovChannels;
    bool                       mCachedRequiresHeatMap;
    int                        mCachedDeepFormat;
    float                      mCachedDeepCurvatureTolerance;
    float                      mCachedDeepZTolerance;
    float                      mCachedTargetAdaptiveError;
    uint                       mCachedDeepVolCompressionRes;
    std::vector<std::string>   mCachedDeepIDChannelNames;
    int                        mCachedDeepMaxLayers;
    float                      mCachedDeepLayerBias;
    scene_rdl2::math::Viewport mCachedViewport;
    SamplingMode               mCachedSamplingMode;
    unsigned                   mCachedDisplayFilterCount;

    Film *              mFilm;

    // tile extrapolation main logic for vectorized mode
    scene_rdl2::fb_util::TileExtrapolation mTileExtrapolation;

    // Scratch buffer passed to Film for use in the extrapolation process. We
    // don't depend on the contents of this buffer outside of the snapshot calls.
    mutable scene_rdl2::fb_util::RenderBuffer mExtrapolationBuffer;

    // This mutex is used for exclusive execution of snapshot between regular snapshot and checkpoint snapshot
    mutable std::mutex mExtrapolationBufferMutex;

    unsigned            mLastCoarsePassIdx;

    std::unique_ptr<TileScheduler>  mTileScheduler;
    std::unique_ptr<TileScheduler>  mTileSchedulerCheckpointInitEstimation;
    TileWorkQueue       mTileWorkQueue;

    tbb::task_scheduler_init *mTaskScheduler;

    // The is the sample count.
    size_t              mSamplesPerPass[MAX_RENDER_PASSES];

    // The number of *primary* rays submitted so far. Used for progress tracking.
    util::AlignedElementArray<std::atomic_size_t, CACHE_LINE_SIZE> mPrimaryRaysSubmitted;

    // Used to signify that it's safe to display the current frame. When this
    // gets set changes depending on the current render mode.
    std::atomic_bool    mReadyForDisplay;

    // Used to signify that there is something rendered at each pixel and we no
    // longer need to run the extrapolation algorithm.
    std::atomic_bool    mCoarsePassesComplete;

    // Used to signify that the last frame triggered was able to render to
    // completion.
    std::atomic_bool    mFrameComplete;

    // Status that last frame triggered was completed by requestStopAtPassBoundary() or not.
    std::atomic_bool    mFrameCompleteAtPassBoundary;

    // For use with the realtime rendering mode.
    double              mUpdateDurationOffset;

    // This is only time spent in the rendering threads, not anything else which
    // is going on during the frame.
    double              mMcrtStartTime;
    double              mMcrtDuration;
    double              mMcrtUtilization;

    // Estimated time of frame end, used to stop rendering for realtime mode.
    std::atomic<double> mFrameEndTime;

    // If this condition is true, checkpointResume rendering logic stops after
    // initial estimation render pass and avoids useless execution of main checkpoint
    // stint render loop. This is important to achieve fast interactive response under arras context
    bool mStopAtFrameReadyForDisplay;

    class RenderThreadStateManager
    {
    public:
        RenderThreadStateManager() noexcept
        : mRenderThreadState(UNINITIALIZED)
        {
        }

        void set(RenderThreadState oldState, RenderThreadState newState, std::memory_order order = std::memory_order_seq_cst) noexcept
        {
            // If this assert triggers we either have:
            //  (a) a code bug where the caller is passing in the wrong state, or
            //  (b) multiple threads are trying to set the same value for newState
            //      which we disallow
            const auto old [[gnu::unused]] = mRenderThreadState.exchange(newState, order);
            MNRY_ASSERT(old == oldState);
            // TODO: C++20 atomic notify all
        }

        RenderThreadState get(std::memory_order order = std::memory_order_seq_cst) const noexcept
        {
            return mRenderThreadState.load(order);
        }

        void wait(RenderThreadState desired) noexcept
        {
            while (mRenderThreadState.load(std::memory_order_relaxed) != desired) {
                // TODO: C++20 atomic wait
                mcrt_common::threadSleep();
            }
        }

    private:
        std::atomic<RenderThreadState>  mRenderThreadState;
    };

    // Persistent rendering controller thread. We don't want to take a thread
    // out of TBB's thread pool for this task.
    std::thread                     mRenderThread;

    // See comment related to DebugRayState to see how this state machine works.
    RenderThreadStateManager       mRenderThreadState;

    // The current state we're in with regards to recording rays for debugging.
    std::atomic<DebugRayState>  mDebugRayState;

    typedef std::vector<RealtimeFrameStats> RealtimeStats;
    RealtimeStats       mRealtimeStats;

    // renderFrame() and multiple renderPasses() detail timing information about each engine threads
    // This info is used for realtime/progressiveCheckpoint renderMode to estimate proper sample count
    RenderFrameTimingRecord mTimeRec;
    RenderProgressEstimation mProgressEstimation; // progress estimation logic for checkpoint render
    unsigned mAdaptiveTileSampleCap; // tile sample cap for adaptive checkpoint render

    int mLastCheckpointFileEndSampleId; // default is -1

    std::string mCheckpointPostScript; // post checkpoint lua script name

    CheckpointController mCheckpointController;
    bool mMultiMachineCheckpointMainLoop {false};
    float mMultiMachineFrameBudgetSecShort {1.0f}; // sec
    float mMultiMachineQuickPhaseLengthSec {2.0f};
    float mMultiMachineFrameBudgetSecLong {30.0f}; // sec

    float mMultiMachineGlobalProgressFraction {0.0f};

    // Condition rendering should stop at pass boundary. Only used under arras multi mcrt computation context
    bool mRenderStopAtPassBoundary;

    // Queues for XPU processing of rays (RenderDriver is owner)
    pbr::XPUOcclusionRayQueue* mXPUOcclusionRayQueue;
    pbr::XPURayQueue* mXPURayQueue;

    // RenderDriver owns DisplayFilterDriver
    DisplayFilterDriver mDisplayFilterDriver;
    tbb::spin_mutex mDisplayFilterMutex;

    // parallel init frame mode
    // Basically, an initial snapshot does not happened before completing 1 sample per tile.  We call this
    // phase as minimum requirement phase. And we can not stop rendering during the minimum requirement phase.
    // Usually, this cost is pretty small, and no one care. However, it is pretty bad interactivity if this
    // minimum requirement phase cost is pretty big like too complex scene and pretty heavy rendering
    // conditions.
    // In order to make a smaller minimum requirement cost, we can compute this minimum requirement phase by
    // multiple machines in parallel if we can use multiple backends. Each host only computes partial tiles
    // (i.e. not entire tiles). As a result, the minimum requirement phase cost is smaller and we can achieve
    // better interactivity in general. 
    // Each backend computes tile by randomly shuffled order and makes an initial snapshot without waiting
    // entire tiles. Snapshot data would be sent to the client. In this case, initial snapshot data includes
    // lots of black tiles. However this is harmless, the initial snapshot data from each backends merged into
    // single image on the client and finally black tiles are solved because each machines calculate tiles by
    // random order.
    bool mParallelInitFrameUpdate;
    unsigned mParallelInitFrameUpdateMcrtCount;
    scene_rdl2::rec_time::RecTime mParallelInitFrameUpdateTime;
    bool mCheckpointEstimationStage; // condition flag to indicate checkpoint render estimation stage
    scene_rdl2::grid_util::FloatValueTracker mRenderPrepTime; // statistical info for debug
    scene_rdl2::grid_util::FloatValueTracker mCheckpointEstimationTime; // statistical info for debug

    Parser mParser;
    Parser mParserInitFrameControl;
    Parser mParserMultiMachineControl;

    DISALLOW_COPY_OR_ASSIGNMENT(RenderDriver);
};

//-----------------------------------------------------------------------------

template<typename DST_PIXEL_TYPE, typename SRC_PIXEL_TYPE, typename HAS_DATA_FUNC,
         typename PIXEL_XFORM_FUNC>
inline void
RenderDriver::snapshotBuffer(scene_rdl2::fb_util::PixelBuffer<DST_PIXEL_TYPE> *outputBuffer,
                             const scene_rdl2::fb_util::PixelBuffer<SRC_PIXEL_TYPE> &srcBuffer,
                             scene_rdl2::fb_util::PixelBuffer<SRC_PIXEL_TYPE> *scratchBuffer,
                             bool extrapolate,
                             bool untile,
                             bool parallel,
                             HAS_DATA_FUNC hasData,
                             PIXEL_XFORM_FUNC pixelXform) const
{
    scene_rdl2::fb_util::Tiler tiler(mUnalignedW, mUnalignedH);

    if (!scratchBuffer) {
        if (sizeof(SRC_PIXEL_TYPE) <= sizeof(scene_rdl2::fb_util::RenderBuffer::PixelType)) {
            scratchBuffer = (scene_rdl2::fb_util::PixelBuffer<SRC_PIXEL_TYPE> *)&mExtrapolationBuffer;
        } else {
            // If this triggers you either didn't pass in a scratch buffer or
            // the mExtrapolationBuffer can't be used since the size of its pixels
            // aren't large enough to accommodate the request.
            MNRY_ASSERT_REQUIRE(0);
        }
    }

    scratchBuffer->init(tiler.mAlignedW, tiler.mAlignedH);
    if (untile) {
        outputBuffer->init(mUnalignedW, mUnalignedH);
    } else {
        outputBuffer->init(tiler.mAlignedW, tiler.mAlignedH);
    }

    // Check if we still require an extrapolation pass.
    if (extrapolate) {

        const bool viewportActive = (mCachedViewport.mMinX != 0 ||
                                     mCachedViewport.mMaxX != mUnalignedW ||
                                     mCachedViewport.mMinY != 0 ||
                                     mCachedViewport.mMaxY != mUnalignedH);

        const bool distributed = mTileScheduler->isDistributed();

        // If we have a viewport set which is smaller than the render buffer
        // then clear the extrapolation buffer beforehand since not all of it
        // may get written to in the ExtrapolateBuffer call.
        if (viewportActive || distributed) {

            scratchBuffer->clear();

            if (distributed) {
                // Only extrapolate tiles in the list.
                rndr::extrapolateBufferWithTileList(scratchBuffer, srcBuffer,
                                                    tiler,
                                                    mTileScheduler->getTiles(),
                                                    hasData,
                                                    parallel);
            } else {
                MNRY_ASSERT(viewportActive);

                // Synthesize each tile and clip them to the viewport.
                rndr::extrapolateBufferWithViewport(scratchBuffer, srcBuffer, tiler,
                                                    mCachedViewport, hasData,
                                                    parallel);
            }
        } else {
            // Fast path.
            rndr::extrapolateBufferFastPath(scratchBuffer, srcBuffer, tiler,
                                            hasData, parallel);
        }
    }

    // Write final result directly to outputBuffer.
    const auto *srcBuf = extrapolate ? scratchBuffer : &srcBuffer;

    if (untile) {
        scene_rdl2::fb_util::untile(outputBuffer, *srcBuf, tiler, parallel, pixelXform);
    } else {
        unsigned area = mUnalignedW * mUnalignedH;
        auto *__restrict dstRow = outputBuffer->getData();
        const auto *__restrict srcRow = srcBuf->getData();
        //
        // This block crashes in a opt build, see comment in Film.h untile()
        //
#if 0
        simpleLoop(parallel, 0u, area, [&](unsigned i) {
            dstRow[i] = pixelXform(srcRow[i], i);
        });
#else
        for (unsigned i = 0; i < area; ++i) {
            dstRow[i] = pixelXform(srcRow[i], i);
        }
#endif
    }
}

//
// Considered tiled format of buffer memory layout and maximize memory access coherency.
//
#ifdef SINGLE_THREAD_CRAWLALLPIXELS
template <typename F>
void
RenderDriver::crawlAllTiledPixels(F pixelFunc) const
{
    scene_rdl2::fb_util::Tiler tiler(getWidth(), getHeight());

    unsigned totalTileY = tiler.mAlignedH >> 3;
    unsigned totalTileX = tiler.mAlignedW >> 3;

    for (unsigned tileY = 0; tileY < totalTileY; ++tileY) {
        unsigned startY = tileY << 3;
        unsigned endY = (startY + 8 < getHeight())? startY + 8: getHeight();

        for (unsigned tileX = 0; tileX < totalTileX; ++tileX) {
            unsigned startX = tileX << 3;
            unsigned endX = (startX + 8 < getWidth())? startX + 8: getWidth();

            for (unsigned pixY = startY; pixY < endY; ++pixY) {
                unsigned pixOffset = tiler.linearCoordsToTiledOffset(startX, pixY);
                for (unsigned pixX = startX; pixX < endX; ++pixX) {
                    pixelFunc(pixOffset);
                    pixOffset++;
                } // pixX
            } // pixY

        } // tileX
    } // tileY
}
#else // else SINGLE_THREAD_CRAWLALLPIXELS
template <typename F>
void
RenderDriver::crawlAllTiledPixels(F pixelFunc) const
// bufferPixLoop
{
    scene_rdl2::fb_util::Tiler tiler(getWidth(), getHeight());

    unsigned totalTileY = tiler.mAlignedH >> 3;
    unsigned totalTileX = tiler.mAlignedW >> 3;

    tbb::blocked_range<size_t> range(0, totalTileY);
    tbb::parallel_for(range, [&](const tbb::blocked_range<size_t> &r) {
            for (size_t tileY = r.begin(); tileY < r.end(); ++tileY) {
                unsigned startY = tileY << 3;
                unsigned endY = (startY + 8 < getHeight())? startY + 8: getHeight();

                for (unsigned tileX = 0; tileX < totalTileX; ++tileX) {
                    unsigned startX = tileX << 3;
                    unsigned endX = (startX + 8 < getWidth())? startX + 8: getWidth();

                    for (unsigned pixY = startY; pixY < endY; ++pixY) {
                        unsigned pixOffset =tiler.linearCoordsToTiledOffset(startX, pixY);
                        for (unsigned pixX = startX; pixX < endX; ++pixX) {
                            pixelFunc(pixOffset);
                            pixOffset++;
                        } // pixX
                    } // pixY
                } // tileX
            } // tileY
        });
}
#endif // else !SINGLE_THREAD_CRAWLALLPIXELS

//-----------------------------------------------------------------------------

/// Internal version of initRenderDriver that takes a configured TLSInitParams.
/// Generally used by unit tests.
/// This must be called at program startup before we create any RenderContexts.
/// It creates a single global render driver which is shared between all future
/// render contexts. Internally this also sets up all or our TLS objects.
void initRenderDriver(const mcrt_common::TLSInitParams &initParams);

/// This call is optional. If not called, the render driver will be destroyed
/// via global destructor of an internal smart pointer. It can be useful to call
/// this manually however if you want to clean up the render driver before the
/// global dtor stage (make sure nothing else is holding a ref to it).
void cleanUpRenderDriver();

/// Returns the global render driver, or nullptr if initRenderDriver hasn't been
/// called yet.
std::shared_ptr<RenderDriver> getRenderDriver();

} // namespace rndr
} // namespace moonray

extern "C" bool isRenderCanceled();

