// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#pragma once



#include "RenderOptions.h"
#include <moonray/rendering/rndr/statistics/ArrasLogStream.h>
#include <moonray/rendering/rndr/statistics/AthenaCSVStream.h>
#include <moonray/common/mcrt_util/Average.h>
#include <moonray/common/mcrt_util/ProcessStats.h>
#include <moonray/statistics/StatsTable.h>

#include <scene_rdl2/common/rec_time/RecTime.h>
#include <scene_rdl2/scene/rdl2/Geometry.h>
#include <scene_rdl2/scene/rdl2/SceneObject.h>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdint.h>
#include <string>
#include <unordered_map>


namespace moonray {
namespace pbr {
class Statistics;
}
namespace texture {
class TextureSampler;
}

namespace geom {
    struct GeometryStatistics;

namespace internal {
    class NamedPrimitive;
    class Statistics;
}
}

namespace rndr {

typedef std::vector<std::pair<std::string, geom::GeometryStatistics>> GeometryStatsTable;

/*
 * RenderStats is a collection of render specific statistics
 * responsible for providing a way to log all the collected values
 * as well as resetting values.  Also provides a method to convert clock ticks
 * to seconds by sampling the number of ticks per unit of known time.
 */


class RenderStats
{
public:
    using TessStat = std::pair<geom::internal::NamedPrimitive*, double>;
    using ShaderStat = std::pair<scene_rdl2::rdl2::SceneObject*, moonray::util::InclusiveExclusiveAverage<int64>>;
    using TickFunction = int64 (moonray::util::InclusiveExclusiveAverage<int64>::*)() const;

    // xtor with file name to enable file logging
    // if no file is specified no file logging will happen
    RenderStats();

    // dtor
    ~RenderStats() { };


    //  bool value passed in to enable/disable time stamp logging.
    void setTimeLogging( bool value) { mLogTime = value; }

    //  bool value passed in to enable/disable memory stamp logging.
    void setMemoryLogging( bool value) { mLogMemory = value; }

    //  bool value passed in to enable/disable readIO stamp logging.
    void setReadIOLogging( bool value) { mLogReadIO = value; }

    //  bool value passed in to enable/disable utilization stamp logging.
    void setUtilizationLogging( bool value) { mLogUtilization = value; }

    bool getLogInfo() const
    {
        return mInfoStream.is_open() && !mInfoStream.fail();
    }

    bool getLogCsv() const
    {
        return mCSVStream.is_open() && !mCSVStream.fail();
    }

    bool getLogAthena() const
    {
        return mAthenaStream.is_open() && !mAthenaStream.fail();
    }

    // Called when render prep start
    void startRenderPrep();

    //  report the first line in the log
    void logInfoPrependStringHeader() const;

    //  report an empty line to help break down sections in the log
    void logInfoEmptyLine() const;

    //  report out the intialization messages and configuration
    void logInitializationConfiguration(std::stringstream &initMessages);

    //  report out the hardware configuration
    void logHardwareConfiguration(const RenderOptions& options, const scene_rdl2::rdl2::SceneVariables& vars);

    //  report out exec mode configuration
    void logExecModeConfiguration(mcrt_common::ExecutionMode executionMode);

    //  report out the command line options state before
    //  rendering begins
    void logRenderOptions(const RenderOptions& options);

    //   report out all of the render output filenames
    void logRenderOutputs(const std::vector<const scene_rdl2::rdl2::RenderOutput*>& renderOutputs);

    //  report out the results of reading in the various
    //  scene values and applying them to the rendeer.
    void logSceneVariables(const scene_rdl2::rdl2::SceneVariables &vars);

    //  report out the memory footprint in megabyte
    //  includes total memory for all geometry objects,
    //  memory for each geometry object,
    //  total memory for BVH
    void logMemoryUsage(size_t                                       totalGeometryBytes,
                        std::vector<std::pair<std::string, size_t>> &perGeometryBytes,
                        size_t                                       bvhBytes   );

    //  report out polycount/cv/curves count for geometry primitives
    //  per geometry and the total for all geometry objects
    void logGeometryUsage(const geom::GeometryStatistics& totalGeomStatistics,
            const GeometryStatsTable& geomStateInfo);

    // report top tessellation times per geometry primitive
    void logTopTessellationStats();
    // report all tessellation times per geometry primitive
    void logAllTessellationStats();

    //  report out the total plus per stage times for
    //  render prep time
    void updateAndLogRenderPrepStats();

    //  log the beginning of the rendering phase
    void startRenderStats();

    //   render stats while application is rendering.
    void updateAndLogRenderProgress(std::size_t* current, std::size_t* total);

    //  log post frame sampling stats
    void logSamplingStats(const pbr::Statistics& pbrStats,
                          const geom::internal::Statistics& geomStats);

    // this is a wrapper for the text that comes out of OIIO
    // in the future we may want to dive into the OIIO data structs and
    // generate specific report data here but for now this is done for
    // completeness .
    void logTexturingStats(texture::TextureSampler& texturesampler, bool verbose);

    //  log the post frame render stats
    void logRenderingStats(const pbr::Statistics& renderstats,
                           mcrt_common::ExecutionMode executionMode,
                           const scene_rdl2::rdl2::SceneVariables& vars);

    // log the count of each dso used
    void logDsoUsage(const std::unordered_map<std::string, size_t>& dsoCounts) const;

    //  log the loading scene header
    void logStartLoadingScene(std::stringstream &initMessages);

    //  log the scene file names
    void logLoadingScene(std::stringstream &initMessages, const std::string& sceneFile);

    //  log the scene file class names
    void logLoadingSceneUpdates(std::stringstream *initMessages, const std::string& sceneClass, const std::string& name);

    //  log the extra formating for end of loading scene section
    void logLoadingSceneReadDiskIO(std::stringstream &initMessages);

    //  log the generating procedurals header
    void logStartGeneratingProcedurals();

    //  log the generating procedurals class name
    void logGeneratingProcedurals(const std::string& sceneClass, const std::string& name);

    //  log the formatting to end the generating procedurals section
    void logEndGeneratingProcedurals();

    //  log the MCRT phase progress made so far.
    void logMcrtProgress(float elapsedMcrtTime, float pctComplete);

    //utility fuctions....

    // Open the human-readable info stream.
    void openInfoStream(bool toOpen) {
        if (toOpen) {
            mInfoStream.open();
            if (!mInfoStream) {
                scene_rdl2::Logger::warn("Unable to log information.");
            }
            MNRY_ASSERT(getLogInfo());
        }
    }

    // Open the stats file.
    void openStatsStream(const std::string& filename) {
        if (!filename.empty()) {
            mCSVStream.open(filename.c_str());
            if (!mCSVStream) {
                scene_rdl2::Logger::warn("Unable to open stats file ", filename, ".");
            }
        }
    }

    // Open the stream to the Athena server.
    void openAthenaStream(const scene_rdl2::util::GUID& guid, bool debug) {
        if (guid != scene_rdl2::util::GUID::nil()) {
            mAthenaStream.open(guid, debug);
        } else {
            mAthenaStream.open(debug);
        }
        MNRY_ASSERT(getLogAthena());
        if (!mAthenaStream) {
            scene_rdl2::Logger::warn("Unable to open connection to Athena server.");
        }
    }

    // reset all values to zero
    void reset();

    void flush()
    {
        if (mCSVStream) {
            mCSVStream.flush();
        }

        if (mAthenaStream) {
            mAthenaStream.flush();
        }

        if (mInfoStream) {
            mInfoStream.flush();
        }
    }

    // generate full paths if filename is a relative path
    std::string getFullPath(const std::string& filename) const;

    // access the logging prepend string
    std::string getPrependString() const;

    std::string getPrependStringHeader() const;

    std::string getReadDiskIOString(int width) const;

    std::string getReadDiskIOCsvString() const;

    // make Moonray self-aware
    std::string getMoonrayVersion(const std::string &executablePath) const;

    // Log string unconditionally
    void logString(const std::string& str) const;

    // Log string only when we're using -info
    void logInfoString(const std::string& str) const;

    // Log string only when we're using -debug
    void logDebugString(const std::string& str) const;


    // stats are public for ease of access
    moonray::util::AverageDouble mLoadSceneTime;
    moonray::util::AverageDouble mLoadPbrTime;
    moonray::util::AverageDouble mBuildPrimAttrTableTime;
    moonray::util::AverageDouble mLoadProceduralsTime;
    moonray::util::AverageDouble mTessellationTime;
    moonray::util::AverageDouble mBuildAcceleratorTime;
    moonray::util::AverageDouble mBuildGPUAcceleratorTime;

    // on going across frames times.
    moonray::util::AverageDouble mRebuildGeometryTime;
    moonray::util::AverageDouble mUpdateSceneTime;

    double mBuildProceduralTime;
    double mRtcCommitTime;

    // tessellation time stats
    std::vector<std::pair<geom::internal::NamedPrimitive*, double> > mPerPrimitiveTessellationTime;

    // shader call stats
    std::unordered_map<scene_rdl2::rdl2::SceneObject *, moonray::util::InclusiveExclusiveAverage<int64> > mShaderCallStats;

private:
    using ShaderStatsTable = moonray_stats::StatsTable<5>;

    enum class OutputFormat
    {
        athenaCSV,
        fileCSV,
        human
    };

    void logRenderOptions(const RenderOptions& options, std::ostream& outs, OutputFormat format);
    void logSceneVariables(const scene_rdl2::rdl2::SceneVariables &vars, std::ostream& outs, OutputFormat format);
    void updateAndLogRenderPrepStats(std::ostream& outs, OutputFormat format);
    void logRenderingStats(const pbr::Statistics& pbrStats,
                           mcrt_common::ExecutionMode executionMode,
                           const scene_rdl2::rdl2::SceneVariables& vars, double processTime, std::ostream& outs,
                           OutputFormat format);

    moonray_stats::StatsTable<3> buildTessellationStatistics(std::size_t maxEntry, std::size_t callDivisor);

    // This function WILL modify the ShaderStat vector.
    ShaderStatsTable buildShaderStatistics(std::vector<ShaderStat>::iterator first,
                                           std::vector<ShaderStat>::iterator last,
                                           const pbr::Statistics& pbrStats,
                                           const std::string& title,
                                           TickFunction getTicks,
                                           std::size_t maxEntry,
                                           std::size_t countDivisor,
                                           bool useCommonPrefix);

    ShaderStatsTable buildShaderStatistics(std::vector<ShaderStat> shaderStats,
                                           const pbr::Statistics& pbrStats,
                                           const std::string& title,
                                           TickFunction getTicks,
                                           std::size_t maxEntry,
                                           std::size_t countDivisor,
                                           bool useCommonPrefix = true);

    ShaderStatsTable buildShaderStatistics(std::vector<ShaderStat> shaderStats,
                                           const pbr::Statistics& pbrStats,
                                           const std::string& title,
                                           TickFunction getTicks);

    void logCSVShaderStats(std::ostream& outs,
                           OutputFormat format,
                           const pbr::Statistics& pbrStats,
                           const std::vector<ShaderStat>& inclStats,
                           const std::vector<ShaderStat>& exclStats);

    void logInfoShaderStats(std::ostream& outs,
                            const pbr::Statistics& pbrStats,
                            const std::vector<ShaderStat>& inclStats,
                            const std::vector<ShaderStat>& exclStats);

    void updateToMostRecentTicksPerSecond();

    // handy conversion function to convert clock ticks to seconds
    double ticksToSec(const uint64 value, const int numThreads);

    // the machine this process runs on
    std::string mHostName;

    // the current path
    std::string mCurPath;

    // 1 / number of cpu ticks per second
    double mInvTicksPerSecond;

    // TODO: get this updating cleanly
    int mPercentRenderDone;

    //  flag to set if process time is appended to log
    bool mLogTime;

    // flag to set if process memory use is appended to log
    bool mLogMemory;

    // flag to set if disk read i/o is appended to log
    bool mLogReadIO;

    // flag to set if system utilization is appended to log
    bool mLogUtilization;

    // this is the start time in seconds
    double mFrameStartTime;

    // keep track of process utilization at the beginning of the frame
    moonray::util::ProcessUtilization mFrameStartUtilization;

    // gross render prep time measurement
    double mTotalRenderPrepTime;

    // keep track of render prep utilization
    double mPrepSysTime;
    double mPrepUserTime;

    // class used to extract stats about our running process.
    moonray::util::ProcessStats mProcessStats;

    std::ofstream mCSVStream;

    moonray::stats::AthenaCSVStream mAthenaStream;

    moonray::stats::ArrasLogStream mInfoStream;
};

class RenderPrepTimingStats
{
public:

    enum class StopFrameTag : int {
        WHOLE = 0,                 // 0:whole
        MDRIVER_STOPFRAME,         // 1:mDriver->stopFrame()
        MPBRSCENE_POSTFRAME,       // 2:mPbrScene->postFrame()
        ACCUMULATE_PBR_STATS_DATA, // 3:accumulate pbr stats data
        COLLECTSHADERSTATS,        // 4:collectShaderStats()
        REPORTSHADINGLOGS,         // 5:reportShadingLogs()
        LOG_FLUSH,                 // 6:log & flush() staff
        RESET,                     // 7:reset

        SIZE                       // total tag size
    };

    enum class RenderPrepTag : int {
        WHOLE = 0,                    // 0:whole
        START_UPDATE_PHASE_OF_FRAME,  // 1:startUpdatePhaseOfFrame()
        START_RENDERPREP,             // 2:startRenderPrep()
        RESET_STATISTICS,             // 3:resetStatistics()
        RESET_RENDER_OUTPUT_DRIVER,   // 4:reset RenderOutputDriver
        FLAG_STATUS_UPDATE,           // 5:flag status update
        RESET_SHADER_STATS_AND_LOGS,  // 6:resetShaderStatsAndLogs
        LOAD_GEOMETRIES,              // 7:loadGeometries()
        REPORT_GEOMETRY_MEMORY,       // 8:reportGeometryMemory()
        BUILD_MATERIAL_AOV_FLAGS,     // 9:buildMaterialAovFlags()
        BUILD_GEOMETRY_EXTENSIONS,    // A:buildGeometryExtensions()
        RESET_SHADER_STATS_AND_LOGS2, // B:resetShaderStatsAndLogs()
        PBR_STATISTICS_RESET,         // C:pbr statistics reset
        UPDATE_PBR,                   // D:update PBR

        SIZE                          // total tag size
    };

    void recTimeStart() { mRecTimeWhole.start(); mRecTime.start(); }
    void recTime(const StopFrameTag &tag) { mStopFrame[id(tag)] = mRecTime.end(); mRecTime.start(); }
    void recTimeEnd(const StopFrameTag &tag) { mStopFrame[id(tag)] = mRecTimeWhole.end(); }

    void recTime(const RenderPrepTag &tag) { mRenderPrep[id(tag)] = mRecTime.end(); mRecTime.start(); }
    void recTime(const RenderPrepTag &tag, const double v) { mRenderPrep[id(tag)] = v; }
    void recTimeEnd(const RenderPrepTag &tag) { mRenderPrep[id(tag)] = mRecTimeWhole.end(); }

    void setWholeStartFrame(const double t) { mWholeStartFrame = t; }
    double getWholeStartFrame() const { return mWholeStartFrame; }

    int getStopFrameTotal() const { return id(StopFrameTag::SIZE); }
    double getStopFrameVal(const int id) const { return mStopFrame[id]; }

    int getRenderPrepTotal() const { return id(RenderPrepTag::SIZE); }
    double getRenderPrepVal(const int id) const { return mRenderPrep[id]; }

protected:

    scene_rdl2::rec_time::RecTime mRecTimeWhole, mRecTime;

    static int id(const StopFrameTag &tag) { return static_cast<int>(tag); }
    static int id(const RenderPrepTag &tag) { return static_cast<int>(tag); }

    //
    // RenderContext::stopFrame() breakdown
    //
    double mStopFrame[static_cast<int>(StopFrameTag::SIZE)];

    double mWholeStartFrame;

    //
    // RenderContext::renderPrep() breakdown
    //
    double mRenderPrep[static_cast<int>(RenderPrepTag::SIZE)];
};

// Probably want to find a home in common/util for these
std::string timeIntervalFormat (double secs, int secPrecision = 3);
std::string memFormat (long long bytes);


// The RenderDriver keeps track of these.
struct RealtimeFrameStats
{
    // How long did we spend updating the scene earlier in this frame.
    double      mUpdateDuration;

    // Offset time during mUpdateDuration to adjust render timing
    double      mUpdateDurationOffset;

    // How much time is left over for rendering after the time spent on the update,
    // given our current FPS setting.
    double      mRenderBudget;

    // Time which RenderDriver::realtimeRenderFrame() was invoked.
    double      mRenderFrameStartTime;

    // Time that we started rendering the first realtime pass. (estimation phase)
    double      mFirstPassStartTime;

    // Time that we completed the first realtime pass. (completed estimation phase)
    double      mFirstPassEndTime;

    // The measured cost to render one sample for all tiles for the first pass.
    // This includes queue draining time for the first pass so is likely a
    // slight over estimation.
    double      mPredictedSampleCost;

    // The measured cost to render one sample for all tiles for the rest of the
    // frame. This should match up closely with mPredictedSampleCost
    // (or mPredictedSampleCost should be computed more accurately).
    double      mActualSampleCost;

    // The time we'll end rendering based on the sample cost of the first pass.
    double      mPredictedEndTime;

    // The actual end time we've finished rendering (timing of isComplete() returns true inside
    // RenderDriver::realtimeRenderFrame()'s while() loop.
    // This should match up closely with mPredictedEndTime (or mPredictedEndTime should be computed
    // more accurately).
    double      mActualEndTime;

    // The measured average cost of non active duration of each render thread.
    double      mOverheadDuration;

    // The number of passes we ended up rendering (including the "first" pass).
    unsigned    mNumRenderPasses;

    // The total number of samples per tile we ended up rendering. This is
    // constant for all tiles this frame.
    unsigned    mSamplesPerTile;

    // The total number of samples for this frame.
    unsigned    mSamplesAll;
};


} // namespace rndr
} // namespace moonray

