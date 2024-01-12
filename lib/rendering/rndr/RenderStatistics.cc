// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#include <moonray/rendering/geom/prim/NamedPrimitive.h>

#include "RenderStatistics.h"
#include "Error.h"
#include "RenderDriver.h"

#include <moonray/statistics/StatsTable.h>
#include <moonray/statistics/StatsTableOutput.h>

#include <moonray/common/mcrt_util/CPUID.h>
#include <moonray/common/time/Timer.h>
#include <moonray/rendering/geom/Procedural.h>
#include <moonray/rendering/geom/prim/Statistics.h>
#include <moonray/rendering/mcrt_common/ProfileAccumulatorHandles.h>
#include <moonray/rendering/mcrt_common/ThreadLocalState.h>
#include <moonray/rendering/pbr/core/Statistics.h>
#include <moonray/rendering/texturing/sampler/TextureSampler.h>

#include <scene_rdl2/common/math/Viewport.h>
#include <scene_rdl2/common/math/Math.h>
#include <scene_rdl2/render/logging/logging.h>
#include <scene_rdl2/scene/rdl2/RenderOutput.h>
#include <scene_rdl2/scene/rdl2/rdl2.h>

#include <iomanip>
#include <fstream>
#include <sys/param.h>
#include <unistd.h>

namespace moonray {
namespace rndr {

using namespace scene_rdl2::math;
using namespace moonray_stats;
using scene_rdl2::logging::Logger;

//---------------------------------------------------------------------------
// utility functions and types
//---------------------------------------------------------------------------

namespace {

// given an array of sorted shader stats, extract a common prefix from
// the first maxEntry items.
std::string
getCommonPrefix(std::vector<RenderStats::ShaderStat>::const_iterator first,
                std::vector<RenderStats::ShaderStat>::const_iterator last)
{
    std::string result = "";
    if (first == last) {
        return result;
    }
    MNRY_ASSERT(first < last);
    const std::string firstName = first->first->getName();
    int rootPos = firstName.length();
    for (auto it = first; it != last; ++it) {
        auto miss = std::mismatch(firstName.begin(), firstName.end(), it->first->getName().begin());
        if (miss.first - firstName.begin() < rootPos) {
            rootPos = miss.first - firstName.begin();
        }
    }
    if (rootPos > 0) {
        rootPos = firstName.rfind('/', rootPos);
        if (rootPos != std::string::npos) { // found it
            result = std::string(firstName.begin(), firstName.begin() + rootPos);
        }
    }

    return result;
}

std::string
commonPrefixObjectName(const scene_rdl2::rdl2::SceneObject *obj, const std::string &prefix)
{
    MNRY_ASSERT(obj);

    std::string pre = "\"";
    std::string post;
    if (!prefix.empty()) {
        MNRY_ASSERT(obj->getName().length() > prefix.length());
        post = "..." + obj->getName().substr(prefix.length());
    } else {
        post = obj->getName();
    }
    post += "\"";
    return pre + post;
}

#define ADD_LANE_UTILIZATION(table, stats, counter) \
    addLaneUtilization(table, pbrStats, counter, #counter)

void
addLaneUtilization(moonray_stats::StatsTable<2> &table, const pbr::Statistics &pbrStats, int counter, const char *name)
{
    const size_t laneActivity = pbrStats.getCounter(counter);
    const size_t laneMax = pbrStats.getCounter(counter + 1);

    MNRY_ASSERT(laneActivity <= laneMax);

    const double utilization = (laneMax == 0) ? 1.0 :
        static_cast<double>(laneActivity) / static_cast<double>(laneMax);

    table.emplace_back(name, moonray_stats::percentage(utilization));
}

std::string
getExecutablePath()
{
    // readlink does not append '\0'
    char dest[PATH_MAX] = {0};
    if (readlink("/proc/self/exe", dest, PATH_MAX - 1) < 0) {
        throw std::runtime_error(std::string("Unable to get self path: ") + getErrorDescription());
    }
    return dest;
}

}

//---------------------------------------------------------------------------

RenderStats::RenderStats():
    mBuildProceduralTime(0.0),
    mRtcCommitTime(0.0),
    mHostName(""),
    mCurPath(""),
    mInvTicksPerSecond(-1.0),
    mPercentRenderDone(5),
    mLogTime(true),
    mLogMemory(true),
    mLogReadIO(false),
    mLogUtilization(false),
    mTotalRenderPrepTime(0.0),
    mPrepSysTime(0.0),
    mPrepUserTime(0.0),
    mAthenaStream()
{
    // Set the initial time we begin rendering.
    mFrameStartTime = time::getTime();
    mFrameStartUtilization = mProcessStats.getProcessUtilization();

    char hostName[HOST_NAME_MAX];
    if (gethostname(hostName, HOST_NAME_MAX) == 0) {
        mHostName = hostName;
    } else {
        mHostName = "N/A";
    }
    char curPath[MAXPATHLEN];
    getcwd(curPath, MAXPATHLEN);
    mCurPath = curPath;
}


void
RenderStats::reset()
{
    // TODO: Figure out what stats we want to display over many re-renders in
    // the context of moonray_gui and the mcrt computation. Right now we just
    // reset the stats and display stats for the last frame. With very little
    // work we can display that AND ouptut the average over many frames in the
    // stats file.
    mLoadSceneTime.reset();
    mLoadPbrTime.reset();
    mBuildPrimAttrTableTime.reset();
    mLoadProceduralsTime.reset();
    mTessellationTime.reset();
    mBuildAcceleratorTime.reset();
    mBuildGPUAcceleratorTime.reset();
    mRebuildGeometryTime.reset();
    mUpdateSceneTime.reset();

    mBuildProceduralTime = 0.0;
    mRtcCommitTime = 0.0;

    mShaderCallStats.clear();

    mTotalRenderPrepTime = 0.0;
    mFrameStartTime = 0.0;

    mFrameStartUtilization.userTime = 0;
    mFrameStartUtilization.systemTime = 0;
}


void
RenderStats::startRenderPrep()
{
    // Set the intial time we begin re-rendering (only if reset() was called)
    if (mFrameStartTime == 0.0) {
        mFrameStartTime = time::getTime();
        mFrameStartUtilization = mProcessStats.getProcessUtilization();
    }
}


//---------------------------------------------------------------------------

void
RenderStats::updateToMostRecentTicksPerSecond()
{
    mInvTicksPerSecond = 1.0 / mcrt_common::computeTicksPerSecond();
}

double
RenderStats::ticksToSec(const uint64 value, const int numThreads)
{
    MNRY_ASSERT(numThreads);

    // If this asserts then updateToMostRecentTicksPerSecond() needs to be called
    // before calling ticksToSec.
    MNRY_ASSERT(mInvTicksPerSecond > 0.0);

    // we divide by numThreads in an attempt to make the
    // per-thread shader times more visually relatable
    // to the total render time.
    return (value / numThreads) * mInvTicksPerSecond;
}


std::string
memFormat(int64 bytes)
{
    std::ostringstream out;
    IOSFlags flags(out);
    flags.precision(1);
    flags.fixed();
    flags.imbue(out);

    moonray_stats::Bytes memOutput(bytes);
    memOutput.write(out, moonray_stats::FormatterHuman());

    return out.str();
}


std::string
timeIntervalFormat(double secs, int secPrecision)
{
    std::ostringstream out;
    out.precision(secPrecision);

    moonray_stats::Time timeOutput(secs);
    timeOutput.write(out, moonray_stats::FormatterHuman());

    return out.str();
}


std::string
RenderStats::getPrependStringHeader() const
{
    std::ostringstream out;

    if (mLogTime) {
        out << "Time    ";
    }
    if (mLogMemory) {
        out << "  Memory  ";
    }
    if (mLogReadIO) {
        out << " ReadIO  ";
    }
    if (mLogUtilization) {
        out <<  "Utilization";
    }

    // return header in case calling code wants it.
    return out.str();
}


std::string
RenderStats::getPrependString() const
{
    std::ostringstream out;

    // our estimated process time from the construction of the renderStats
    // class
    double processTime  = time::getTime() - mFrameStartTime;

    if (mLogTime) {
        out << timeIntervalFormat(processTime, 0);
        out.put(' ');
    }

    if (mLogMemory) {
        int64 memory = mProcessStats.getProcessMemory();
        out << std::setw(9) << memFormat(memory);
    }

    if (mLogReadIO) {
        out << memFormat(mProcessStats.getBytesRead());
        out.put(' ');
    }

    if (mLogUtilization) {
        util::ProcessUtilization us;
        us = mProcessStats.getProcessUtilization();

        out << std::fixed << std::right << std::setw(5);
        out << (us.systemTime + us.userTime) / processTime << '%';
    }
    out << " | ";
    return out.str();
}


std::string
RenderStats::getFullPath(const std::string& filename) const
{
    std::ostringstream out;

    if (!filename.empty() && filename[0] != '/') {
        out << mCurPath << '/';
    }
    out << filename;
    return out.str();
}


std::string
RenderStats::getReadDiskIOString(int width) const
{
    std::ostringstream out;

    out << std::fixed << std::left << std::setw(width)
        << "Read disk I/O" << " = "
        << memFormat(mProcessStats.getBytesRead());

    return out.str();
}


std::string
RenderStats::getReadDiskIOCsvString() const
{
    std::ostringstream out;

    // Convert the bytes to MB for the Csv log
    out << (static_cast<double>(mProcessStats.getBytesRead()) / (1<<20));

    return out.str();
}


std::string
RenderStats::getMoonrayVersion(const std::string &executablePath) const
{
    std::string version = "unknown";
    size_t moonrayPos = executablePath.find("/moonray/");
    if (moonrayPos != std::string::npos) {
        size_t versionPos = moonrayPos + strlen("/moonray/");
        size_t endPos = executablePath.find("/", versionPos);
        version = executablePath.substr(versionPos, endPos-versionPos);
    }
    return version;
}


void
RenderStats::logString(const std::string& str)  const
{
    Logger::info(getPrependString() + str);
}


void
RenderStats::logInfoString(const std::string& str)  const
{
    if (getLogInfo()) {
        Logger::info(getPrependString() + str);
    }
}

void
RenderStats::logDebugString(const std::string& str) const
{
    Logger::debug(getPrependString() + str);
}


void
RenderStats::logInfoEmptyLine() const
{
    if (getLogInfo()) {
        Logger::info("");
    }
}

void
RenderStats::logInfoPrependStringHeader() const
{
    if (getLogInfo()) {
        Logger::info(getPrependStringHeader());
    }
}


//---------------------------------------------------------------------------

void
RenderStats::logInitializationConfiguration(std::stringstream &initMessages)
{
    initMessages.flush();
    std::string messages = initMessages.str();
    if (messages.empty()) {
        return;
    }

    StatsTable<1> table("Initialization & Configuration");
    std::string line;
    while (std::getline(initMessages, line)) {
        if (!line.empty()) {
            table.emplace_back(line);
        }
    }

    if (getLogAthena()) {
        writeCSVTable(mAthenaStream, table, true);
    }
    if (getLogCsv()) {
        writeCSVTable(mCSVStream, table, false);
    }
    if (getLogInfo()) {
        ConstantFlags flags(mInfoStream);
        flags.set().left();
        writeInfoTable(mInfoStream, getPrependString(), table, flags);
    }
}

void
RenderStats::logHardwareConfiguration(const RenderOptions& options, const scene_rdl2::rdl2::SceneVariables& vars)
{
    StatsTable<2> table("Hardware Configuration");
    table.emplace_back("Host name", mHostName);
    table.emplace_back("Number of machines", vars.getNumMachines());
    table.emplace_back("Cluster machine id", vars.getMachineId());
    table.emplace_back("Thread(s)", mcrt_common::getNumTBBThreads());

    if (getLogAthena()) {
        writeEqualityCSVTable(mAthenaStream, table, true);
    }
    if (getLogCsv()) {
        writeEqualityCSVTable(mCSVStream, table, false);
    }
    if (getLogInfo()) {
        writeEqualityInfoTable(mInfoStream, getPrependString(), table);
    }
}

void
RenderStats::logExecModeConfiguration(mcrt_common::ExecutionMode executionMode)
{
    StatsTable<2> table("Exec Mode Configuration");
    table.emplace_back("Vectorized rendering", executionMode == mcrt_common::ExecutionMode::VECTORIZED);
    table.emplace_back("XPU rendering", executionMode == mcrt_common::ExecutionMode::XPU);

    if (getLogAthena()) {
        writeEqualityCSVTable(mAthenaStream, table, true);
    }
    if (getLogCsv()) {
        writeEqualityCSVTable(mCSVStream, table, false);
    }
    if (getLogInfo()) {
        writeEqualityInfoTable(mInfoStream, getPrependString(), table);
    }
}

void
RenderStats::logRenderOptions(const RenderOptions& options, std::ostream& outs, OutputFormat format)
{
    const bool csvStream = format == OutputFormat::athenaCSV || format == OutputFormat::fileCSV;

    util::CPUID cpuid;

    StatsTable<2> hardwareTable("Hardware Support");
    hardwareTable.emplace_back("CPU vendor tag", cpuid.vendor());
    hardwareTable.emplace_back("Atomic   8 bit support", cpuid.atomic_8());
    hardwareTable.emplace_back("Atomic  16 bit support", cpuid.atomic_16());
    hardwareTable.emplace_back("Atomic  32 bit support", cpuid.atomic_32());
    hardwareTable.emplace_back("Atomic  64 bit support", cpuid.atomic_64());
    hardwareTable.emplace_back("Atomic 128 bit support", cpuid.atomic_128());
    hardwareTable.emplace_back("SSE support", cpuid.sse());
    hardwareTable.emplace_back("SSE2 support", cpuid.sse2());
    hardwareTable.emplace_back("SSE3 support", cpuid.sse3());
    hardwareTable.emplace_back("SSE4.1 support", cpuid.sse41());
    hardwareTable.emplace_back("SSE4.2 support", cpuid.sse42());
    hardwareTable.emplace_back("AVX support", cpuid.avx());
    hardwareTable.emplace_back("AVX2 support", cpuid.avx2());
    hardwareTable.emplace_back("AVX512 support", cpuid.avx512());

    char curPath[MAXPATHLEN];
    getcwd(curPath, MAXPATHLEN);

    const std::string executablePath = [] {
        try {
            return getExecutablePath();
        } catch (const std::exception& e) {
            Logger::warn("Unable to resolve executable path: ", e.what());
            return std::string("Unable to resolve");
        }
    }();

#if defined(__AVX512F__)
    const constexpr char* const sSimdName = "avx512";
#elif defined(__AVX2__)
    const constexpr char* const sSimdName = "avx2";
#elif defined(__AVX__)
    const constexpr char* const sSimdName = "avx";
#elif defined(__SSE4_2__)
    const constexpr char* const sSimdName = "sse4.2";
#elif defined(__SSE4_1__)
    const constexpr char* const sSimdName = "sse4.1";
#elif defined(__SSE3__)
    const constexpr char* const sSimdName = "sse3";
#elif defined(__SSE2__)
    const constexpr char* const sSimdName = "sse2";
#elif defined(__SSE__)
    const constexpr char* const sSimdName = "sse";
#else
#error Unknown SIMD type
#endif

    const char *desiredExecutionMode = nullptr;
    switch (options.getDesiredExecutionMode()) {
    case mcrt_common::ExecutionMode::AUTO:       desiredExecutionMode = "AUTO";          break;
    case mcrt_common::ExecutionMode::VECTORIZED: desiredExecutionMode = "VECTORIZED";    break;
    case mcrt_common::ExecutionMode::SCALAR:     desiredExecutionMode = "SCALAR";        break;
    case mcrt_common::ExecutionMode::XPU:        desiredExecutionMode = "XPU";           break;
    default:
        MNRY_ASSERT(0);
    }

    const std::string& athenaTags = options.getAthenaTags();
    StatsTable<2> table("Rendering Options");
    table.emplace_back("Command line", options.getCommandLine());
    table.emplace_back("Executable path", executablePath);
    table.emplace_back("Moonray version", getMoonrayVersion(executablePath));
    table.emplace_back("DSO path override", options.getDsoPath());
    table.emplace_back("Desired vectorization mode", desiredExecutionMode); // Note: the Athena data team requested it be output under this name
    table.emplace_back("SIMD build support", sSimdName);
    table.emplace_back("Athena Tags", athenaTags );

    const auto& sceneFiles = options.getSceneFiles();
    for (const auto& sceneFile : sceneFiles) {
        table.emplace_back("Scene file", getFullPath(sceneFile));
    }

    for (const auto& deltasFile : options.getDeltasFiles()) {
        table.emplace_back("Deltas file", getFullPath(deltasFile));
    }

    using AttributesTable = StatsTable<4>;
    AttributesTable attrTable("Attribute Overrides",
                              "Object", "Attribute", "Value", "Binding");
    const auto& overrides = options.getAttributeOverrides();
    if (csvStream || !overrides.empty()) {
        for (const auto& override : overrides) {
            if (!override.mIsSceneVar) {
                attrTable.emplace_back(override.mObject,
                                       override.mAttribute,
                                       override.mValue,
                                       override.mBinding);
            }
        }
    }

    StatsTable<2> luaTable("Lua Globals");
    const auto& luaGlobals = options.getRdlaGlobals();
    if (csvStream || !luaGlobals.empty()) {
        for (const auto& luaglobal : luaGlobals) {
            luaTable.emplace_back(luaglobal.mVar, luaglobal.mExpression);
        }
    }

    if (csvStream) {
        outs.precision(2);
        writeEqualityCSVTable(outs, hardwareTable, format == OutputFormat::athenaCSV);
        writeEqualityCSVTable(outs, table, format == OutputFormat::athenaCSV);
        writeCSVTable(outs, attrTable, format == OutputFormat::athenaCSV);
        writeEqualityCSVTable(outs, luaTable, format == OutputFormat::athenaCSV);
    } else {
        outs.precision(2);
        writeEqualityInfoTable(outs, getPrependString(), hardwareTable);
        writeEqualityInfoTable(outs, getPrependString(), table);
        if (!attrTable.empty()) {
            writeInfoTable(outs, getPrependString(), attrTable);
        }
        if (!luaTable.empty()) {
            writeEqualityInfoTable(outs, getPrependString(), luaTable);
        }
    }
}

void
RenderStats::logRenderOutputs(const std::vector<const scene_rdl2::rdl2::RenderOutput*>& renderOutputs)
{
    using namespace scene_rdl2::rdl2;
    logInfoEmptyLine();
    logInfoString(std::string("---------- Render Outputs ---------- \""));
    for (size_t i = 0; i < renderOutputs.size(); ++i) {
        const RenderOutput* ro = renderOutputs[i];
        logInfoEmptyLine();
        logInfoString(std::string("RenderOutput: \"" + ro->getName() + "\""));
        std::string resultString = "\tResult: ";
        switch(ro->getResult()) {
        case RenderOutput::Result::RESULT_BEAUTY:
            resultString += "Beauty";
            break;
        case RenderOutput::Result::RESULT_ALPHA:
            resultString += "Alpha";
            break;
        case RenderOutput::Result::RESULT_DEPTH:
            resultString += "Depth";
            break;
        case RenderOutput::Result::RESULT_STATE_VARIABLE:
            {
                resultString += "State Variable - ";
                switch(ro->getStateVariable()) {
                case RenderOutput::StateVariable::STATE_VARIABLE_P:
                    resultString += "P";
                    break;
                case RenderOutput::StateVariable::STATE_VARIABLE_NG:
                    resultString += "Ng";
                    break;
                case RenderOutput::StateVariable::STATE_VARIABLE_N:
                    resultString += "N";
                    break;
                case RenderOutput::StateVariable::STATE_VARIABLE_ST:
                    resultString += "st";
                    break;
                case RenderOutput::StateVariable::STATE_VARIABLE_DPDS:
                    resultString += "dPds";
                    break;
                case RenderOutput::StateVariable::STATE_VARIABLE_DPDT:
                    resultString += "dPdt";
                    break;
                case RenderOutput::StateVariable::STATE_VARIABLE_DSDX:
                    resultString += "dSdx";
                    break;
                case RenderOutput::StateVariable::STATE_VARIABLE_DSDY:
                    resultString += "dSdy";
                    break;
                case RenderOutput::StateVariable::STATE_VARIABLE_WP:
                    resultString += "world position";
                    break;
                case RenderOutput::StateVariable::STATE_VARIABLE_DEPTH:
                    resultString += "depth";
                    break;
                case RenderOutput::StateVariable::STATE_VARIABLE_MOTION:
                    resultString += "motion";
                    break;
                }
            }
            break;
        case RenderOutput::Result::RESULT_PRIMITIVE_ATTRIBUTE:
            resultString += "Primitive Attribute - ";
            resultString += "\"" + ro->getPrimitiveAttribute() + "\" ";
            switch(ro->getPrimitiveAttributeType()) {
            case RenderOutput::PrimitiveAttributeType::PRIMITIVE_ATTRIBUTE_TYPE_FLOAT:
                resultString += "(float)";
                break;
            case RenderOutput::PrimitiveAttributeType::PRIMITIVE_ATTRIBUTE_TYPE_VEC2F:
                resultString += "(Vec2f)";
                break;
            case RenderOutput::PrimitiveAttributeType::PRIMITIVE_ATTRIBUTE_TYPE_VEC3F:
                resultString += "(Vec3f)";
                break;
            case RenderOutput::PrimitiveAttributeType::PRIMITIVE_ATTRIBUTE_TYPE_RGB:
                resultString += "(rgb)";
                break;
            }
            break;
        case RenderOutput::Result::RESULT_HEAT_MAP:
            resultString += "Heat Map";
            break;
        case RenderOutput::Result::RESULT_WIREFRAME:
            resultString += "Wireframe";
            break;
        case RenderOutput::Result::RESULT_MATERIAL_AOV:
            resultString += "Material AOV - ";
            resultString += "\"" + ro->getMaterialAov() + "\"";
            break;
        case RenderOutput::Result::RESULT_LIGHT_AOV:
            resultString += "Light AOV - ";
            resultString += "\"" + ro->getLpe() + "\"";
            break;
        case RenderOutput::Result::RESULT_VISIBILITY_AOV:
            resultString += "Visibility AOV - ";
            resultString += "\"" + ro->getVisibilityAov() + "\"";
            break;
        case RenderOutput::Result::RESULT_WEIGHT:
            resultString += "Weight";
            break;
        case RenderOutput::Result::RESULT_BEAUTY_AUX:
            resultString += "Beauty aux";
            break;
        case RenderOutput::Result::RESULT_CRYPTOMATTE:
            resultString += "Cryptomatte";
            break;
        case RenderOutput::Result::RESULT_ALPHA_AUX:
            resultString += "Alpha aux";
            break;
        case RenderOutput::Result::RESULT_DISPLAY_FILTER:
            resultString += "Display filter";
            break;
        }
        logInfoString(resultString);
        logInfoString(std::string("\tChannel: '" + ro->getChannelName() + "'"));

        logInfoString(std::string("\tWrote: '" + ro->getFileName() + "'"));
    }
}

void
RenderStats::logRenderOptions(const RenderOptions& options)
{
    if (getLogAthena()) {
        logRenderOptions(options, mAthenaStream, OutputFormat::athenaCSV);
    }
    if (getLogCsv()) {
        logRenderOptions(options, mCSVStream, OutputFormat::fileCSV);
    }
    if (getLogInfo()) {
        logRenderOptions(options, mInfoStream, OutputFormat::human);
    } else {
        const auto& sceneFiles = options.getSceneFiles();
        for (const auto& sceneFile : sceneFiles) {
            std::cout << "Loading Scene File(s): ";
            std::cout << getFullPath(sceneFile) << '\n';
        }
        std::cout << "Starting render prep..." << '\r';
        std::cout.flush();
    }
}

void
RenderStats::logSceneVariables(const scene_rdl2::rdl2::SceneVariables &vars, std::ostream& outs, OutputFormat format)
{
    const bool csvStream = format == OutputFormat::athenaCSV || format == OutputFormat::fileCSV;

    // TODO: do we want to add anymore scene vars to this list?
    HalfOpenViewport aperture = vars.getRezedApertureWindow();
    HalfOpenViewport region = vars.getRezedRegionWindow();

    StatsTable<2> sceneVarTable("Scene Variables");
    sceneVarTable.emplace_back("Width", vars.get(scene_rdl2::rdl2::SceneVariables::sImageWidth));
    sceneVarTable.emplace_back("Height", vars.get(scene_rdl2::rdl2::SceneVariables::sImageHeight));
    sceneVarTable.emplace_back("Resolution", vars.get(scene_rdl2::rdl2::SceneVariables::sResKey));
    sceneVarTable.emplace_back("Final width", vars.getRezedWidth());
    sceneVarTable.emplace_back("Final height", vars.getRezedHeight());
    sceneVarTable.emplace_back("Final aperture window min x", aperture.mMinX);
    sceneVarTable.emplace_back("Final aperture window min y", aperture.mMinY);
    sceneVarTable.emplace_back("Final aperture window max x", aperture.mMaxX);
    sceneVarTable.emplace_back("Final aperture window max y", aperture.mMaxY);
    sceneVarTable.emplace_back("Final region window min x", region.mMinX);
    sceneVarTable.emplace_back("Final region window min y", region.mMinY);
    sceneVarTable.emplace_back("Final region window max x", region.mMaxX);
    sceneVarTable.emplace_back("Final region window max y", region.mMaxY);

    const scene_rdl2::rdl2::PixelFilterType  pfilter = static_cast<scene_rdl2::rdl2::PixelFilterType>(
        vars.get(scene_rdl2::rdl2::SceneVariables::sPixelFilterType));

    switch (pfilter) {
        case scene_rdl2::rdl2::PixelFilterType::box:
            sceneVarTable.emplace_back("Pixel filter type", "box");
            break;
        case scene_rdl2::rdl2::PixelFilterType::cubicBSpline:
            sceneVarTable.emplace_back("Pixel filter type", "cubicBSpline");
            break;
        case scene_rdl2::rdl2::PixelFilterType::quadraticBSpline:
            sceneVarTable.emplace_back("Pixel filter type", "quadraticBSpline");
            break;
        default:
            sceneVarTable.emplace_back("Pixel filter type", "unknown");
            break;
    }

    sceneVarTable.emplace_back("Pixel filter width",
                       static_cast<int>(vars.get(scene_rdl2::rdl2::SceneVariables::sPixelFilterWidth)));
    sceneVarTable.emplace_back("Texture blur",
                      static_cast<int>(vars.get(scene_rdl2::rdl2::SceneVariables::sTextureBlur)));
    sceneVarTable.emplace_back("Output file",
                       getFullPath(vars.get(scene_rdl2::rdl2::SceneVariables::sOutputFile)));
    sceneVarTable.emplace_back("Stats file",
                       getFullPath(vars.get(scene_rdl2::rdl2::SceneVariables::sStatsFile)));
    sceneVarTable.emplace_back("DSO path", vars.getSceneClass().getSceneContext()->getDsoPath());
    sceneVarTable.emplace_back("Camera", vars.getSceneClass().getSceneContext()->getPrimaryCamera()->getName());
    sceneVarTable.emplace_back("Layer", vars.getLayer()->getName());
    sceneVarTable.emplace_back("Debug ray file", vars.get(scene_rdl2::rdl2::SceneVariables::sDebugRaysFile));

    sceneVarTable.emplace_back("texture_cache_size",
                               static_cast<int>(vars.get(scene_rdl2::rdl2::SceneVariables::sTextureCacheSizeMb)));

    HalfOpenViewport subViewport;
    if (vars.getSubViewport(subViewport)) {
        sceneVarTable.emplace_back("Viewport min x", subViewport.mMinX);
        sceneVarTable.emplace_back("Viewport min y", subViewport.mMinY);
        sceneVarTable.emplace_back("Viewport max x", subViewport.mMaxX);
        sceneVarTable.emplace_back("Viewport max y", subViewport.mMaxY);
    } else if (csvStream) {
        sceneVarTable.emplace_back("Viewport min x", "not set");
        sceneVarTable.emplace_back("Viewport min y", "not set");
        sceneVarTable.emplace_back("Viewport max x", "not set");
        sceneVarTable.emplace_back("Viewport max y", "not set");
    }

    Vec2i pixel;
    if (vars.getDebugPixel(pixel)) {
        sceneVarTable.emplace_back("Debug pixel x", pixel.x);
        sceneVarTable.emplace_back("Debug pixel y", pixel.y);
    } else if (csvStream) {
        sceneVarTable.emplace_back("Debug pixel x", "not set");
        sceneVarTable.emplace_back("Debug pixel y", "not set");
    }

    int start, end;
    if (vars.getDebugRaysPrimaryRange(start, end)) {
        sceneVarTable.emplace_back("Debug rays range start", start);
        sceneVarTable.emplace_back("Debug rays range end", end);
    } else if (csvStream) {
        sceneVarTable.emplace_back("Debug rays range start", "not set");
        sceneVarTable.emplace_back("Debug rays range end", "not set");
    }

    if (vars.getDebugRaysDepthRange(start,end)) {
        sceneVarTable.emplace_back("Debug rays depth start", start);
        sceneVarTable.emplace_back("Debug rays depth end", end);
    } else if (csvStream) {
        sceneVarTable.emplace_back("Debug rays depth start", "not set");
        sceneVarTable.emplace_back("Debug rays depth end", "not set");
    }

    StatsTable<2> ssTable("Sampling Settings");
    ssTable.emplace_back("Sampling mode", vars.get(scene_rdl2::rdl2::SceneVariables::sSamplingMode));
    ssTable.emplace_back("Min adaptive samples", vars.get(scene_rdl2::rdl2::SceneVariables::sMinAdaptiveSamples));
    ssTable.emplace_back("Max adaptive samples", vars.get(scene_rdl2::rdl2::SceneVariables::sMaxAdaptiveSamples));
    ssTable.emplace_back("Target adaptive error", vars.get(scene_rdl2::rdl2::SceneVariables::sTargetAdaptiveError));
    ssTable.emplace_back("Pixel samples sqrt", vars.get(scene_rdl2::rdl2::SceneVariables::sPixelSamplesSqrt));
    ssTable.emplace_back("Light samples sqrt", vars.get(scene_rdl2::rdl2::SceneVariables::sLightSamplesSqrt));
    ssTable.emplace_back("Bsdf samples sqrt", vars.get(scene_rdl2::rdl2::SceneVariables::sBsdfSamplesSqrt));
    ssTable.emplace_back("Bssrdf samples sqrt", vars.get(scene_rdl2::rdl2::SceneVariables::sBssrdfSamplesSqrt));
    ssTable.emplace_back("Max depth", vars.get(scene_rdl2::rdl2::SceneVariables::sMaxDepth));
    ssTable.emplace_back("Max diffuse depth", vars.get(scene_rdl2::rdl2::SceneVariables::sMaxDiffuseDepth));
    ssTable.emplace_back("Max glossy depth", vars.get(scene_rdl2::rdl2::SceneVariables::sMaxGlossyDepth));
    ssTable.emplace_back("Max mirror depth", vars.get(scene_rdl2::rdl2::SceneVariables::sMaxMirrorDepth));
    ssTable.emplace_back("Max hair depth", vars.get(scene_rdl2::rdl2::SceneVariables::sMaxHairDepth));
    ssTable.emplace_back("Max volume depth", vars.get(scene_rdl2::rdl2::SceneVariables::sMaxVolumeDepth));
    ssTable.emplace_back("Max presence depth", vars.get(scene_rdl2::rdl2::SceneVariables::sMaxPresenceDepth));
    ssTable.emplace_back("Presence threshold", vars.get(scene_rdl2::rdl2::SceneVariables::sPresenceThreshold));
    ssTable.emplace_back("Transparency threshold", vars.get(scene_rdl2::rdl2::SceneVariables::sTransparencyThreshold));
    ssTable.emplace_back("Max subsurface per path", vars.get(scene_rdl2::rdl2::SceneVariables::sMaxSubsurfacePerPath));
    ssTable.emplace_back("Russian roulette threshold", vars.get(scene_rdl2::rdl2::SceneVariables::sRussianRouletteThreshold));
    ssTable.emplace_back("Sample clamping value", vars.get(scene_rdl2::rdl2::SceneVariables::sSampleClampingValue));
    ssTable.emplace_back("Sample clamping depth", vars.get(scene_rdl2::rdl2::SceneVariables::sSampleClampingDepth));
    ssTable.emplace_back("Roughness clamping factor", vars.get(scene_rdl2::rdl2::SceneVariables::sRoughnessClampingFactor));
    ssTable.emplace_back("Volume quality", vars.get(scene_rdl2::rdl2::SceneVariables::sVolumeQuality));
    ssTable.emplace_back("Volume illumination samples", vars.get(scene_rdl2::rdl2::SceneVariables::sVolumeIlluminationSamples));
    ssTable.emplace_back("Volume opacity threshold", vars.get(scene_rdl2::rdl2::SceneVariables::sVolumeOpacityThreshold));



    if (csvStream) {
        outs.precision(2);
        writeEqualityCSVTable(outs, sceneVarTable, format == OutputFormat::athenaCSV);
        writeEqualityCSVTable(outs, ssTable, format == OutputFormat::athenaCSV);
    } else {
        const auto prepend = getPrependString();
        auto sceneVarFmt = getHumanEqualityColumnFlags(outs, sceneVarTable);
        sceneVarFmt.set(0).left();
        sceneVarFmt.set(1).left();
        sceneVarFmt.set(1).precision(2);
        sceneVarFmt.set(1).fixed();
        writeEqualityInfoTable(outs, prepend, sceneVarTable, sceneVarFmt);

        auto ssFmt = getHumanEqualityColumnFlags(outs, ssTable);
        ssFmt.set(1).precision(5);
        ssFmt.set(1).fixed();
        writeEqualityInfoTable(outs, prepend, ssTable, ssFmt);
    }
}

void
RenderStats::logSceneVariables(const scene_rdl2::rdl2::SceneVariables &vars)
{
    if (getLogAthena()) {
        logSceneVariables(vars, mAthenaStream, OutputFormat::athenaCSV);
    }
    if (getLogCsv()) {
        logSceneVariables(vars, mCSVStream, OutputFormat::fileCSV);
    }
    if (getLogInfo()) {
        logSceneVariables(vars, mInfoStream, OutputFormat::human);
    }
}

void
RenderStats::logMemoryUsage(size_t                                       totalGeometryBytes,
                            std::vector<std::pair<std::string, size_t>> &perGeometryBytes,
                            size_t                                       bvhBytes   )
{
    using GeomTable = StatsTable<2>;
    GeomTable geomTable("Geometry Memory Usage", "Geometry Name", "MB");

    for(std::size_t i = 0; i < perGeometryBytes.size(); ++i) {
        geomTable.emplace_back(perGeometryBytes[i].first,
                               perGeometryBytes[i].second/1024.0f/1024.0f);
    }

    StatsTable<2> summaryTable("Memory Summary");
    summaryTable.emplace_back("Geometry memory", bytes(totalGeometryBytes));
    summaryTable.emplace_back("BVH memory", bytes(bvhBytes));
    summaryTable.emplace_back("Total memory", bytes(totalGeometryBytes + bvhBytes));

    auto writeCSV = [&](std::ostream& outs, bool athenaFormat) {
        outs.precision(2);
        outs.setf(std::ios_base::fixed, std::ios_base::floatfield);
        writeEqualityCSVTable(outs, geomTable, athenaFormat);
        writeEqualityCSVTable(outs, summaryTable, athenaFormat);
    };

    if (getLogAthena()) {
        writeCSV(mAthenaStream, true);
    }
    if (getLogCsv()) {
        writeCSV(mCSVStream, false);
    }
    if (getLogInfo()) {
        auto fmt = getHumanColumnFlags(mInfoStream, geomTable);
        fmt.set(0).left();
        fmt.set(1).left();
        fmt.set(1).width(10);
        fmt.set(1).precision(2);
        fmt.set(1).fixed();
        const std::string pre = getPrependString();
        writeInfoTablePermutation<1, 0>(mInfoStream, pre, geomTable, fmt);

        mInfoStream.precision(2);
        mInfoStream.setf(std::ios_base::fixed, std::ios_base::floatfield);
        writeEqualityInfoTable(mInfoStream, pre, summaryTable);
    }
}


void
RenderStats::logGeometryUsage(const geom::GeometryStatistics& totalGeomStatistics,
        const GeometryStatsTable& geomStateInfo)
{
    using GeomTable = StatsTable<6>;
    GeomTable geomTable("Geometry Statistics", "Geometry Name",
        "Face Count", "Mesh Vertex Count",
        "Curves Count", "Curves CV Count",
        "Instance Count");

    for(std::size_t i = 0; i < geomStateInfo.size(); ++i) {
        geomTable.emplace_back(geomStateInfo[i].first,
           geomStateInfo[i].second.mFaceCount,
           geomStateInfo[i].second.mMeshVertexCount,
           geomStateInfo[i].second.mCurvesCount,
           geomStateInfo[i].second.mCVCount,
           geomStateInfo[i].second.mInstanceCount);
    }

    StatsTable<2> summaryTable("Geometry Statistics Summary");
    summaryTable.emplace_back("Total Face Count",
        totalGeomStatistics.mFaceCount);
    summaryTable.emplace_back("Total Mesh Vertex Count",
        totalGeomStatistics.mMeshVertexCount);
    summaryTable.emplace_back("Total Curves Count",
        totalGeomStatistics.mCurvesCount);
    summaryTable.emplace_back("Total Curves CV Count",
        totalGeomStatistics.mCVCount);
    summaryTable.emplace_back("Total Instance Count",
        totalGeomStatistics.mInstanceCount);

    auto writeCSV = [&](std::ostream& outs, bool athenaFormat) {
        outs.precision(2);
        outs.setf(std::ios_base::fixed, std::ios_base::floatfield);
        writeCSVTable(outs, geomTable, athenaFormat);
        writeEqualityCSVTable(outs, summaryTable, athenaFormat);
    };

    if (getLogAthena()) {
        writeCSV(mAthenaStream, true);
    }
    if (getLogCsv()) {
        writeCSV(mCSVStream, false);
    }
    if (getLogInfo()) {
        mInfoStream.imbue(getLocale());
        mInfoStream.precision(2);
        mInfoStream.setf(std::ios_base::fixed, std::ios_base::floatfield);
        auto fmt = getHumanColumnFlags(mInfoStream, geomTable);
        fmt.set(0).left();
        const std::string pre = getPrependString();
        writeInfoTable(mInfoStream, pre, geomTable, fmt);
        writeEqualityInfoTable(mInfoStream, pre, summaryTable);
    }
}

void
RenderStats::logTopTessellationStats()
{
    const std::size_t maxEntry = 10;
    const std::size_t callDivisor = 1000;
    const auto tsTableInfo = buildTessellationStatistics(maxEntry, callDivisor);

    if (getLogInfo()) {
        auto tsFormat = getHumanColumnFlags(mInfoStream, tsTableInfo);
        tsFormat.set(0).left();
        tsFormat.set(1).left();
        tsFormat.set(2).precision(3);
        tsFormat.set(2).right();
        writeInfoTable(mInfoStream, getPrependString(), tsTableInfo, tsFormat);
    }
    if (getLogCsv()) {
        auto tsFormat = getCSVFlags(mCSVStream, tsTableInfo);
        tsFormat.set().setf(std::ios::fixed, std:: ios::floatfield);
        tsFormat.set().precision(5);
        writeCSVTable(mCSVStream, tsTableInfo, false /* not athena */, tsFormat);
    }
}

void
RenderStats::logAllTessellationStats()
{
    auto first = mPerPrimitiveTessellationTime.begin();
    auto last = mPerPrimitiveTessellationTime.end();

    moonray_stats::StatsTable<3> table("Tessellation time", "Rdl Geometry", "part name", "time (s)");

    for (auto it = first; it != last; ++it) {
        const auto& obj = *it;
        table.emplace_back(obj.first->getRdlGeometry()->getName(), obj.first->getName(), moonray_stats::time(obj.second));
    }

    auto tsFormat = getCSVFlags(mAthenaStream, table);
    tsFormat.set().setf(std::ios::fixed, std:: ios::floatfield);
    tsFormat.set().precision(5);
    writeCSVTable(mAthenaStream, table, true, tsFormat);
}

void
RenderStats::updateAndLogRenderPrepStats(std::ostream& outs, OutputFormat format)
{
    const char* const header = (format == OutputFormat::human) ?
                               "Render Prep Stats ---- (hh:mm:ss.ms)" :
                               "Render Prep Stats";

    StatsTable<2> table(header);
    table.emplace_back("Loading scene", moonray_stats::time(mLoadSceneTime.getSum()));
    table.emplace_back("Initialize renderer", moonray_stats::time(mLoadPbrTime.getSum() + mBuildPrimAttrTableTime.getSum()));
    table.emplace_back("Generating procedurals", moonray_stats::time(mLoadProceduralsTime.getSum()));
    table.emplace_back("Tessellation", moonray_stats::time(mTessellationTime.getSum()));
    table.emplace_back("Building BVH", moonray_stats::time(mBuildAcceleratorTime.getSum()));
    table.emplace_back("Building GPU BVH", moonray_stats::time(mBuildGPUAcceleratorTime.getSum()));
    table.addSeparator();
    table.emplace_back("Total render prep", moonray_stats::time(mTotalRenderPrepTime));
    table.emplace_back("Render prep read disk I/O", bytes(mProcessStats.getBytesRead()));

    const bool csvStream = format == OutputFormat::athenaCSV || format == OutputFormat::fileCSV;
    if (csvStream) {
        outs.precision(5);
        outs.setf(std::ios::fixed, std:: ios::floatfield);
        writeEqualityCSVTable(outs, table, format == OutputFormat::athenaCSV);
    } else {
        const auto prepend = getPrependString();
        outs.precision(3);
        outs.setf(std::ios::fixed, std:: ios::floatfield);
        writeEqualityInfoTable(outs, prepend, table);
    }
}

void
RenderStats::updateAndLogRenderPrepStats()
{
    mTotalRenderPrepTime = time::getTime() - mFrameStartTime;

    // Update prep utilization stats
    util::ProcessUtilization us = util::ProcessStats().getProcessUtilization();
    mPrepSysTime = us.getSystemSeconds(mFrameStartUtilization);
    mPrepUserTime = us.getUserSeconds(mFrameStartUtilization);

    if (getLogAthena()) {
        updateAndLogRenderPrepStats(mAthenaStream, OutputFormat::athenaCSV);
    }
    if (getLogCsv()) {
        updateAndLogRenderPrepStats(mCSVStream, OutputFormat::fileCSV);
    }
    if (getLogInfo()) {
        updateAndLogRenderPrepStats(mInfoStream, OutputFormat::human);
    } else {
        std::cout << "Render prep time = " <<
                  timeIntervalFormat(mTotalRenderPrepTime) << '\n';
    }
}

void
RenderStats::startRenderStats()
{
    if (getLogInfo()) {
        logInfoEmptyLine();
        const auto prepend = getPrependString();
        mInfoStream << prepend
                    << createDashTitle("MCRT Rendering") << '\n';
    }
}

void
RenderStats::updateAndLogRenderProgress(std::size_t* current, std::size_t* total)
{
    if (!getLogInfo()) {
        return;
    }

    std::string prepend = getPrependString();
    std::ostringstream out;
    out.precision(2);

    if (*total > 0) {
        float actualpercent = ((double)*current/(double)*total)*100.f;
        //  try to get as close to 5% print increments as we can.
        // the longer the render time the closer we hit 5% increments.
        // fast renders will be very variable.
        if (actualpercent > mPercentRenderDone) {
            mPercentRenderDone = (actualpercent/5) * 5 + 5;
            out << prepend << std::fixed << std::left << std::setw(6)
                << actualpercent << "% complete";
            Logger::info(out.str());
        }
    }
}


void
RenderStats::logSamplingStats(const pbr::Statistics& pbrStats, const geom::internal::Statistics& geomStats)
{
    const size_t pixelSamples = pbrStats.getCounter(pbr::STATS_PIXEL_SAMPLES);
    const size_t lightSamples = pbrStats.getCounter(pbr::STATS_LIGHT_SAMPLES);
    const size_t bsdfSamples = pbrStats.getCounter(pbr::STATS_BSDF_SAMPLES);
    const size_t bssrdfSamples = pbrStats.getCounter(pbr::STATS_SSS_SAMPLES);
    const size_t totalSamples = pixelSamples + lightSamples + bsdfSamples + bssrdfSamples;

    const size_t isectRays = pbrStats.getCounter(pbr::STATS_INTERSECTION_RAYS);
    const size_t bundledIsectRays = pbrStats.getCounter(pbr::STATS_BUNDLED_INTERSECTION_RAYS);
    const size_t bundledGPUIsectRays = pbrStats.getCounter(pbr::STATS_BUNDLED_GPU_INTERSECTION_RAYS);

    const size_t presenceShadowRays = pbrStats.getCounter(pbr::STATS_PRESENCE_SHADOW_RAYS);

    const size_t occlRays = pbrStats.getCounter(pbr::STATS_OCCLUSION_RAYS);
    const size_t bundledOcclRays = pbrStats.getCounter(pbr::STATS_BUNDLED_OCCLUSION_RAYS);
    const size_t bundledGPUOcclRays = pbrStats.getCounter(pbr::STATS_BUNDLED_GPU_OCCLUSION_RAYS);

    const size_t totalRays = isectRays + occlRays;

    const size_t shaderEvals = pbrStats.getCounter(pbr::STATS_SHADER_EVALS);

    const size_t velocityGridSampls = geomStats.getCounter(geom::internal::STATS_VELOCITY_GRID_SAMPLES);
    const size_t densityGridSampls = geomStats.getCounter(geom::internal::STATS_DENSITY_GRID_SAMPLES);
    const size_t emissionGridSampls = geomStats.getCounter(geom::internal::STATS_EMISSION_GRID_SAMPLES);
    const size_t colorGridSampls = geomStats.getCounter(geom::internal::STATS_COLOR_GRID_SAMPLES);
    const size_t bakedDensityGridSampls = geomStats.getCounter(geom::internal::STATS_BAKED_DENSITY_GRID_SAMPLES);

    StatsTable<2> table("Sampling Statistics");
    table.emplace_back("Pixel samples", pixelSamples);
    table.emplace_back("Light samples", lightSamples);
    table.emplace_back("Bsdf samples", bsdfSamples);
    table.emplace_back("Bssrdf samples", bssrdfSamples);
    table.emplace_back("Total samples", totalSamples);

    table.emplace_back("Intersection rays", isectRays);
    table.emplace_back("Bundled intersection rays", bundledIsectRays);
    const double gpuIntersectionUtilization = (bundledIsectRays > 0) ?
        static_cast<double>(bundledGPUIsectRays) / static_cast<double>(bundledIsectRays) : 0.0;
    table.emplace_back("GPU bundled intersection ray utilization", percentage(gpuIntersectionUtilization));

    table.emplace_back("Presence shadow rays", presenceShadowRays);

    table.emplace_back("Occlusion rays", occlRays);
    table.emplace_back("Bundled occlusion rays", bundledOcclRays);
    const double gpuOcclusionUtilization = (bundledOcclRays > 0) ?
        static_cast<double>(bundledGPUOcclRays) / static_cast<double>(bundledOcclRays) : 0.0;
    table.emplace_back("GPU bundled occlusion ray utilization", percentage(gpuOcclusionUtilization));

    table.emplace_back("Total rays", totalRays);

    table.emplace_back("Shader evals", shaderEvals);

    table.emplace_back("Velocity grid samples", velocityGridSampls);
    table.emplace_back("Density grid samples", densityGridSampls);
    table.emplace_back("Emission grid samples", emissionGridSampls);
    table.emplace_back("Color grid samples", colorGridSampls);
    table.emplace_back("BakedDensity grid samples", bakedDensityGridSampls);

    // We want all of the rows below to be right justified in human readable
    // format.
    const auto numRightJustified = table.getNumRows();

    const size_t lightSampleLaneMax = pbrStats.getCounter(pbr::STATS_LIGHT_SAMPLE_LANE_MAX);
    const double lightSimdUtilization = (lightSampleLaneMax == 0) ? 1.0 :
        static_cast<double>(lightSamples) / static_cast<double>(lightSampleLaneMax);

    const size_t bsdfSampleLaneMax = pbrStats.getCounter(pbr::STATS_BSDF_SAMPLE_LANE_MAX);
    const double bsdfSimdUtilization = (bsdfSampleLaneMax == 0) ? 1.0 :
        static_cast<double>(bsdfSamples) / static_cast<double>(bsdfSampleLaneMax);

    const size_t bssrdfSampleLaneMax = pbrStats.getCounter(pbr::STATS_SSS_SAMPLE_LANE_MAX);
    const double bssrdfSimdUtilization = (bssrdfSampleLaneMax == 0) ? 1.0 :
        static_cast<double>(bssrdfSamples) / static_cast<double>(bssrdfSampleLaneMax);

    const double rayCullingRate = (totalRays >= totalSamples) ? 0.0 :
        static_cast<double>(totalSamples - totalRays) / totalSamples;

    const double millionsPerSecondScale = 1.0 / (pbrStats.mMcrtTime * 1000000.0);

    table.emplace_back("Light SIMD utilization", percentage(lightSimdUtilization));
    table.emplace_back("Bsdf SIMD utilization", percentage(bsdfSimdUtilization));
    table.emplace_back("Bssrdf SIMD utilization", percentage(bssrdfSimdUtilization));

    ADD_LANE_UTILIZATION(table, pbrStats, pbr::STATS_VEC_BSDF_LOBES);
    ADD_LANE_UTILIZATION(table, pbrStats, pbr::STATS_VEC_BSDF_LOBE_SAMPLES_PRE);
    ADD_LANE_UTILIZATION(table, pbrStats, pbr::STATS_VEC_BSDF_LOBE_SAMPLES_POST);

    ADD_LANE_UTILIZATION(table, pbrStats, pbr::STATS_VEC_LIGHT_SAMPLES_PRE);
    ADD_LANE_UTILIZATION(table, pbrStats, pbr::STATS_VEC_LIGHT_SAMPLES_POST);

    ADD_LANE_UTILIZATION(table, pbrStats, pbr::STATS_VEC_COUNTER_A);
    ADD_LANE_UTILIZATION(table, pbrStats, pbr::STATS_VEC_COUNTER_B);

    ADD_LANE_UTILIZATION(table, pbrStats, pbr::STATS_VEC_ADD_DIRECT_VISIBLE_BSDF);
    ADD_LANE_UTILIZATION(table, pbrStats, pbr::STATS_VEC_ADD_DIRECT_VISIBLE_LIGHTING);

    ADD_LANE_UTILIZATION(table, pbrStats, pbr::STATS_VEC_INDIRECT_A);
    ADD_LANE_UTILIZATION(table, pbrStats, pbr::STATS_VEC_INDIRECT_B);
    ADD_LANE_UTILIZATION(table, pbrStats, pbr::STATS_VEC_INDIRECT_C);
    ADD_LANE_UTILIZATION(table, pbrStats, pbr::STATS_VEC_INDIRECT_D);
    ADD_LANE_UTILIZATION(table, pbrStats, pbr::STATS_VEC_INDIRECT_E);
    ADD_LANE_UTILIZATION(table, pbrStats, pbr::STATS_VEC_INDIRECT_F);

    ADD_LANE_UTILIZATION(table, pbrStats, pbr::STATS_VEC_FILL_BUNDLED_RADIANCE);
    ADD_LANE_UTILIZATION(table, pbrStats, pbr::STATS_VEC_FILL_OCCL_RAY);

    table.emplace_back("Ray culling rate", percentage(rayCullingRate));
    table.emplace_back("Million samples/sec",
                       static_cast<double>(totalSamples) * millionsPerSecondScale);
    table.emplace_back("Million rays/sec",
                       static_cast<double>(totalRays) * millionsPerSecondScale);
    table.emplace_back("Million shader-evals/sec",
                       static_cast<double>(shaderEvals) * millionsPerSecondScale);

    auto writeCSV = [&](std::ostream& outs, bool athenaFormat) {
        outs.setf(std::ios::fixed, std:: ios::floatfield);
        outs.precision(2);
        writeEqualityCSVTable(outs, table, athenaFormat);
    };

    if (getLogAthena()) {
        writeCSV(mAthenaStream, true);
    }
    if (getLogCsv()) {
        writeCSV(mCSVStream, false);
    }
    if (getLogInfo()) {
        mInfoStream.setf(std::ios::fixed, std:: ios::floatfield);
        mInfoStream.precision(2);
        mInfoStream.imbue(getLocale());
        auto format = getHumanFullFlags(mInfoStream, table);
        for (std::size_t i = 0; i < numRightJustified; ++i) {
            format.set(i, 0).left();
            format.set(i, 1).right();
        }
        for (std::size_t i = numRightJustified; i < table.getNumRows(); ++i) {
            format.set(i, 0).left();
            format.set(i, 1).left();
        }

        writeEqualityInfoTable(mInfoStream, getPrependString(), table, format);
    }
}


void
RenderStats::logTexturingStats(texture::TextureSampler& texturesampler, bool verbose)
{
    if (getLogAthena()) {
        texturesampler.getStatisticsForCsv(mAthenaStream, true);
    }
    if (getLogCsv()) {
        // currently we are a pass through for the text string
        texturesampler.getStatisticsForCsv(mCSVStream, false);
    }

    if (getLogInfo()) {
        texturesampler.getStatistics(getPrependString(), mInfoStream, verbose);
        texturesampler.getMainCacheInfo(getPrependString(), mInfoStream);
    }
}

namespace {

template <typename Iterator, typename PartitionFunction, typename CompFunction>
std::pair<Iterator, Iterator>
getRelevantStats(Iterator first,
                 Iterator last,
                 PartitionFunction partFunction,
                 CompFunction compFunction,
                 std::size_t maxEntry)
{

    MNRY_ASSERT(first <= last);

    // Trim off the parts with 0 value.
    last = std::partition(first, last, partFunction);
    MNRY_ASSERT(first <= last);

    // Count the number of relevant entries.
    const auto dist = std::distance(first, last);
    MNRY_ASSERT(dist >= 0);
    const std::size_t num = std::min(maxEntry, static_cast<std::size_t>(dist));
    MNRY_ASSERT(first <= last);
    MNRY_ASSERT(first + num <= last);

    // Sort "num" values from the relevant list
    std::partial_sort(first, first + num, last, compFunction);

    MNRY_ASSERT(first <= last);
    MNRY_ASSERT(first + num <= last);
    return std::make_pair(first, std::next(first, num));
}

} // end anonymous namespace

moonray_stats::StatsTable<3>
RenderStats::buildTessellationStatistics(std::size_t maxEntry, std::size_t callDivisor)
{
    auto first = mPerPrimitiveTessellationTime.begin();
    auto last = mPerPrimitiveTessellationTime.end();
    std::tie(first, last) = getRelevantStats(first, last,
            [](const TessStat& ts) { return ts.second > 0; },
            [=](const TessStat& s1, const TessStat& s2)
            {
                return s1.second > s2.second;
            },
            maxEntry);

    moonray_stats::StatsTable<3> table("Tessellation time", "Rdl Geometry", "part name", "time");

    for (auto it = first; it != last; ++it) {
        const auto& obj = *it;
        table.emplace_back(obj.first->getRdlGeometry()->getName(), obj.first->getName(), moonray_stats::time(obj.second));
    }

    return table;
}

// This function WILL modify the ShaderStat vector.
RenderStats::ShaderStatsTable RenderStats::buildShaderStatistics(std::vector<ShaderStat>::iterator first,
                                                                 std::vector<ShaderStat>::iterator last,
                                                                 const pbr::Statistics& pbrStats,
                                                                 const std::string& title,
                                                                 TickFunction getTicks,
                                                                 std::size_t maxEntry,
                                                                 std::size_t countDivisor,
                                                                 bool useCommonPrefix)
{
    auto compFunction = [=](const ShaderStat& s1, const ShaderStat& s2)
    {
        const auto& s1avg = s1.second;
        const auto& s2avg = s2.second;
        return (s1avg.*getTicks)() > (s2avg.*getTicks)();
    };
    std::tie(first, last) = getRelevantStats(first, last, [](const ShaderStat& ss) { return ss.second.getCount() > 0;}, compFunction, maxEntry);
    MNRY_ASSERT(first <= last);

    const std::string commonPrefix = (useCommonPrefix) ? getCommonPrefix(first, last) : std::string();
    ShaderStatsTable table(title, "Name", "Shader Class", "Time (s)", "% of MCRT", "Num Calls");

    if (countDivisor > 1) {
        auto& headers = table.getHeaders();
        headers[0] += ' ' + commonPrefix;
        headers[4] += "(x" + std::to_string(countDivisor) + ")";
    }

    const unsigned numThreads = mcrt_common::getNumTBBThreads();
    for (auto it = first; it != last; ++it) {
        const auto& obj = *it;
        MNRY_ASSERT(obj.second.getCount() > 0);
        const auto& avg = obj.second;
        const double secs = ticksToSec((avg.*getTicks)(), numThreads);
        table.emplace_back(commonPrefixObjectName(obj.first, commonPrefix),
                            obj.first->getSceneClass().getName(),
                            moonray_stats::time(secs),
                            secs / pbrStats.mMcrtTime,
                            obj.second.getCount() / countDivisor);
    }

    return table;
}

// Pass-by-value: let the compiler implicitly copy the ShaderStats vector.
RenderStats::ShaderStatsTable RenderStats::buildShaderStatistics(std::vector<ShaderStat> shaderStats,
                                                                 const pbr::Statistics& pbrStats,
                                                                 const std::string& title,
                                                                 TickFunction getTicks,
                                                                 std::size_t maxEntry,
                                                                 std::size_t countDivisor,
                                                                 bool useCommonPrefix)
{
    return buildShaderStatistics(shaderStats.begin(), shaderStats.end(), pbrStats, title, getTicks, maxEntry, countDivisor, useCommonPrefix);
}

RenderStats::ShaderStatsTable RenderStats::buildShaderStatistics(std::vector<ShaderStat> shaderStats,
                                                                 const pbr::Statistics& pbrStats,
                                                                 const std::string& title,
                                                                 TickFunction getTicks)
{
    return buildShaderStatistics(shaderStats, pbrStats, title, getTicks, std::numeric_limits<std::size_t>::max(), 1, false);
}

namespace {
void getSampleStatistics(const scene_rdl2::rdl2::SceneVariables& vars,
                         double& proportionOfPixelsAtAdaptiveMax,
                         double& avgSamplesPerPixel)
{
    const unsigned adaptiveMaxSamples = vars.get(scene_rdl2::rdl2::SceneVariables::sMaxAdaptiveSamples);
    const moonray::rndr::Film &film = rndr::getRenderDriver()->getFilm();
    unsigned totalSamples = 0;
    unsigned nPixelsAtAdaptiveMax = 0;
    for (unsigned y = 0; y < film.getHeight(); ++y) {
        for (unsigned x = 0; x < film.getWidth(); ++x) {
            const unsigned ps = film.getNumRenderBufferPixelSamples(x, y);
            totalSamples += ps;
            if (ps >= adaptiveMaxSamples) {
                ++nPixelsAtAdaptiveMax;
            }
        }
    }

    const double npixels = film.getWidth() * film.getHeight();
    proportionOfPixelsAtAdaptiveMax = static_cast<double>(nPixelsAtAdaptiveMax) / npixels;
    avgSamplesPerPixel = totalSamples / npixels;
}
} // namespace

void
RenderStats::logRenderingStats(const pbr::Statistics& pbrStats,
                               mcrt_common::ExecutionMode executionMode,
                               const scene_rdl2::rdl2::SceneVariables& vars, double processTime, std::ostream& outs,
                               OutputFormat format)
{
    const bool csvStream = format == OutputFormat::athenaCSV || format == OutputFormat::fileCSV;

    updateToMostRecentTicksPerSecond();
    const unsigned numThreads = mcrt_common::getNumTBBThreads();

    const util::ProcessUtilization us = mProcessStats.getProcessUtilization();
    const int64 userTimeSecs = us.getUserSeconds(mFrameStartUtilization);
    const int64 sysTimeSecs = us.getSystemSeconds(mFrameStartUtilization);

    std::vector<ShaderStat> inclShaderStats;
    for (auto obj : mShaderCallStats) {
        if (obj.first->isA<scene_rdl2::rdl2::RootShader>()) {
            inclShaderStats.push_back(obj);
        }
    }

    std::vector<ShaderStat> exclShaderStats;
    for (auto obj : mShaderCallStats) {
        exclShaderStats.push_back(obj);
    }

    logInfoEmptyLine();
    if (csvStream) {
        logCSVShaderStats(outs, format, pbrStats, inclShaderStats, exclShaderStats);
    } else {
        logInfoShaderStats(outs, pbrStats, inclShaderStats, exclShaderStats);
    }

    int64 totalTicks = 0;
    for (auto obj : mShaderCallStats) {
        if (obj.second.getCount() > 0) {
            const int64 ticks = obj.second.getSum();
            totalTicks += ticks;
        }
    }

    const double secs = ticksToSec(totalTicks, numThreads);
    StatsTable<2> totalTimeTable("Total Shading Time");
    totalTimeTable.emplace_back("Total shading time", moonray_stats::time(secs));
    totalTimeTable.emplace_back("Shading % of MCRT time", percentage(secs / pbrStats.mMcrtTime));

    //
    // Accumulator stats:
    //
    using AccumulatorTable = StatsTable<3>;
    AccumulatorTable accumulatorTable("MCRT Time Breakdown", "Name", "Avg Time per Thread (s)", "Percentage of Total");

    std::vector<mcrt_common::AccumulatorResult> accStats;
    const unsigned numAccumulators = mcrt_common::snapshotAccumulators(&accStats,
                                        mInvTicksPerSecond, csvStream ? 0.f : 0.0001f);
    bool separatorAdded = false;

    for (unsigned i = 0; i < numAccumulators; ++i) {
        const auto &stat = accStats[i];

        if (stat.mFlags & mcrt_common::ACCFLAG_TOTAL) {
            // Add separator before we display the "totals" row.
            accumulatorTable.addSeparator();
        } else if (!separatorAdded && i > 0 && accStats[i - 1].mName[0] != '[' && stat.mName[0] == '[') {

            // MOONRAY-2613, avoid sending extraneous data back to athena.
            if (format == OutputFormat::athenaCSV) {
                break;
            }
            // Add another separator before we display the bracketed entries.
            accumulatorTable.addSeparator();
            separatorAdded = true;
        }

        accumulatorTable.emplace_back(stat.mName,
                                      stat.mTimePerThread,
                                      moonray_stats::percentage(stat.mPercentageOfTotal/100.0));
    }

    const moonray::rndr::Film &film = rndr::getRenderDriver()->getFilm();
    double proportionOfPixelsAtAdaptiveMax;
    double avgSamplesPerPixel;
    getSampleStatistics(vars, proportionOfPixelsAtAdaptiveMax, avgSamplesPerPixel);

    // Compute "normalized" sample cost using Wil Whaley's metric:
    // (based on uniform sampling)
    // cost = (1920 * 1080) * (mcrt_time * threads) / (pixel_samples_sqrt * pixel_samples_sqrt * x_res * y_res)
    double xRes = (double)vars.getRezedWidth();
    double yRes = (double)vars.getRezedHeight();
    double sampleCost = (1920.0 * 1080.0) * (pbrStats.mMcrtTime * (double)mcrt_common::getNumTBBThreads()) /
                        (avgSamplesPerPixel * xRes * yRes);

    const char* const header = (format == OutputFormat::human) ?
                               "Rendering Stats ------ (hh:mm:ss.ms)" :
                               "Rendering Stats";
    StatsTable<2> renderingStatsTable(header);
    renderingStatsTable.emplace_back("Render prep", moonray_stats::time(mTotalRenderPrepTime));
    renderingStatsTable.emplace_back("Mcrt time", moonray_stats::time(pbrStats.mMcrtTime));
    renderingStatsTable.addSeparator();
    renderingStatsTable.emplace_back("Prep user time", moonray_stats::time(mPrepUserTime));
    renderingStatsTable.emplace_back("Prep system time", moonray_stats::time(mPrepSysTime));
    renderingStatsTable.emplace_back("Total time", moonray_stats::time(processTime));
    renderingStatsTable.emplace_back("User time", moonray_stats::time(userTimeSecs));
    renderingStatsTable.emplace_back("System time", moonray_stats::time(sysTimeSecs));
    renderingStatsTable.emplace_back("Prep utilization", percentage((mPrepSysTime + mPrepUserTime) /mTotalRenderPrepTime));
    renderingStatsTable.emplace_back("Mcrt utilization", percentage(pbrStats.mMcrtUtilization/100));
    renderingStatsTable.emplace_back("Total utilization", percentage((sysTimeSecs + userTimeSecs) / processTime));
    // "efficiency" = "utilization" / number of threads
    renderingStatsTable.emplace_back("Prep efficiency", percentage((mPrepSysTime + mPrepUserTime)/(mTotalRenderPrepTime * numThreads)));
    renderingStatsTable.emplace_back("Mcrt efficiency", percentage(pbrStats.mMcrtUtilization/(100 * numThreads)));
    renderingStatsTable.emplace_back("Total efficiency", percentage((sysTimeSecs + userTimeSecs) / (processTime * numThreads)));
    if (film.isAdaptive()) {
        renderingStatsTable.emplace_back("Pixels at max adaptive samples", percentage(proportionOfPixelsAtAdaptiveMax));
    }
    renderingStatsTable.emplace_back("Render stats read disk I/O", bytes(mProcessStats.getBytesRead()));
    renderingStatsTable.emplace_back("Normalized sample cost", sampleCost);

    if (csvStream) {
        outs.setf(std::ios::fixed, std:: ios::floatfield);
        outs.precision(5);
        writeEqualityCSVTable(outs, totalTimeTable, format == OutputFormat::athenaCSV);
        writeCSVTable(outs, accumulatorTable, format == OutputFormat::athenaCSV);
        writeEqualityCSVTable(outs, renderingStatsTable, format == OutputFormat::athenaCSV);
    } else {
        const std::string prepend = getPrependString();
        outs.imbue(getLocale());
        outs.setf(std::ios_base::fixed, std::ios_base::floatfield);
        outs.precision(3);

        auto accumFormat = getHumanColumnFlags(outs, accumulatorTable);
        accumFormat.set(0).left();

        writeEqualityInfoTable(outs, prepend, totalTimeTable);
        logInfoEmptyLine();
        writeInfoTable(outs, prepend, accumulatorTable, accumFormat);
        logInfoEmptyLine();
        writeEqualityInfoTable(outs, prepend, renderingStatsTable);
    }
}

void
RenderStats::logDsoUsage(const std::unordered_map<std::string, size_t>& dsoCounts) const
{
    std::vector<std::string> keys;
    keys.reserve(dsoCounts.size());
    for (const auto& it : dsoCounts) {
        keys.push_back(it.first);
    }
    std::sort(keys.begin(), keys.end());

    logInfoEmptyLine();
    logInfoString(std::string("---------- Dso Usage ----------"));
    for (const std::string& key : keys) {
        std::ostringstream oss;
        oss << std::setw(30) << std::left << key << dsoCounts.at(key);
        logInfoString(oss.str());
    }
    logInfoEmptyLine();
}

void
RenderStats::logRenderingStats(const pbr::Statistics& pbrStats,
                               mcrt_common::ExecutionMode executionMode,
                               const scene_rdl2::rdl2::SceneVariables& vars)
{
    // figure out how long we have been running for in seconds.
    const double processTime = time::getTime() - mFrameStartTime;

    if (getLogAthena()) {
        logRenderingStats(pbrStats, executionMode, vars, processTime, mAthenaStream, OutputFormat::athenaCSV);
    }
    if (getLogCsv()) {
        logRenderingStats(pbrStats, executionMode, vars, processTime, mCSVStream, OutputFormat::fileCSV);
    }
    if (getLogInfo()) {
        logRenderingStats(pbrStats, executionMode, vars, processTime, mInfoStream, OutputFormat::human);
    } else {
        const std::string prepend = getPrependString();
        StatsTable<2> renderTimeTable("Time");
        renderTimeTable.emplace_back("Render time", moonray_stats::time(pbrStats.mMcrtTime));
        renderTimeTable.emplace_back("Total time", moonray_stats::time(processTime));
        writeEqualityInfoTable(std::cout, prepend, renderTimeTable);
    }
}

void
RenderStats::logCSVShaderStats(std::ostream& outs,
                               OutputFormat format,
                               const pbr::Statistics &pbrStats,
                               const std::vector <ShaderStat> &inclStats,
                               const std::vector <ShaderStat> &exclStats)
{
    const auto ssInclTable = buildShaderStatistics(inclStats,
                                                   pbrStats,
                                                   "Inclusive Shader Calls",
                                                   &util::InclusiveExclusiveAverage<int64>::getInclusiveSum);

    const auto ssExclTable = buildShaderStatistics(exclStats,
                                                   pbrStats,
                                                   "Exclusive Shader Calls",
                                                   &util::InclusiveExclusiveAverage<int64>::getSum);

    outs.setf(std::ios::fixed, std:: ios::floatfield);
    outs.precision(5);
    writeCSVTable(outs, ssInclTable, format == OutputFormat::athenaCSV);
    writeCSVTable(outs, ssExclTable, format == OutputFormat::athenaCSV);
}

void
RenderStats::logInfoShaderStats(std::ostream& outs,
                                const pbr::Statistics &pbrStats,
                                const std::vector <ShaderStat> &inclStats,
                                const std::vector <ShaderStat> &exclStats)
{
    const std::size_t maxEntry = 10;
    const std::size_t callDivisor = 1000;
    const auto ssInclTableInfo = buildShaderStatistics(inclStats,
                                                       pbrStats,
                                                       "Inclusive Shader Calls",
                                                       &util::InclusiveExclusiveAverage<int64>::getInclusiveSum,
                                                       maxEntry,
                                                       callDivisor);

    const auto ssExclTableInfo = buildShaderStatistics(exclStats,
                                                       pbrStats,
                                                       "Exclusive Shader Calls",
                                                       &util::InclusiveExclusiveAverage<int64>::getSum,
                                                       maxEntry,
                                                       callDivisor);

    const std::string prepend = getPrependString();
    outs.imbue(getLocale());
    outs.setf(std::ios_base::fixed, std::ios_base::floatfield);
    outs.precision(2);

    auto ssInclFormat = getHumanColumnFlags(outs, ssInclTableInfo);
    ssInclFormat.set(0).left();
    ssInclFormat.set(2).precision(3);
    auto ssExclFormat = getHumanColumnFlags(outs, ssExclTableInfo);
    ssExclFormat.set(0).left();
    ssExclFormat.set(2).precision(3);

    writeInfoTablePermutation<2, 4, 3, 1, 0>(outs, prepend, ssInclTableInfo, ssInclFormat, maxEntry);
    writeInfoTablePermutation<2, 4, 3, 1, 0>(outs, prepend, ssExclTableInfo, ssExclFormat, maxEntry);
}

void
RenderStats::logLoadingScene(std::stringstream &initMessages, const std::string& sceneFile)
{
    initMessages << "Loading Scene File: " << sceneFile << '\n';
}

void
RenderStats::logLoadingSceneUpdates(std::stringstream *initMessages, const std::string& sceneClass, const std::string& name)
{
    if (initMessages) {
        (*initMessages) << "Loading " << sceneClass << "(\"" << name << "\")\n";
    }
}

void
RenderStats::logLoadingSceneReadDiskIO(std::stringstream &initMessages)
{
    initMessages << getReadDiskIOString(0) << '\n';
}

void
RenderStats::logStartGeneratingProcedurals()
{
    Logger::info(getPrependString() + createDashTitle("Generating Procedurals"));

    mCSVStream << createArrowTitle("Generating Procedurals") << '\n';
    mCSVStream << "Name,Scene Class\n";
}

void
RenderStats::logGeneratingProcedurals(const std::string& className, const std::string& name)
{
    mCSVStream << name << ',' << className << '\n';
}

void
RenderStats::logEndGeneratingProcedurals()
{
    mCSVStream << '\n';
}

void
RenderStats::logMcrtProgress(float elapsedMcrtTime, float pctComplete)
{
    if (getLogAthena()) {
        int64 memoryUsed = mProcessStats.getProcessMemory();
        StatsTable<3> progressTable("MCRT Progress", "Elapsed MCRT Time", "Percentage Complete", "Memory Used");
        progressTable.emplace_back(elapsedMcrtTime, pctComplete, memoryUsed);
        writeCSVTable(mAthenaStream, progressTable, true);
    }
}


} // namespace rndr
} // namespace moonray

