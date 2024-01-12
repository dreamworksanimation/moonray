// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#include "ResumeHistoryMetaData.h"

#include <moonray/rendering/pbr/core/Statistics.h>

#include <scene_rdl2/common/platform/Platform.h>
#include <scene_rdl2/render/util/GetEnv.h>
#include <scene_rdl2/render/util/StrUtil.h>
#include <scene_rdl2/render/util/TimeUtil.h>

#include <cstdlib>              // getenv()
#include <sstream>
#include <string>

#include <stdarg.h>             // va_arg
#include <unistd.h>             // gethostname()

namespace {
    
inline
std::string
getHostname()
{
    char buff[4096];
    if (gethostname(buff, 4096) == -1) {
        return std::string("");
    }
    return std::string(buff);
}

inline
std::string
getEnvVal(const std::string& envStr)
{
    return scene_rdl2::util::getenv<std::string>(envStr.c_str());
}

//------------------------------

MAYBE_UNUSED std::size_t size(const std::string& s) { return s.size(); }
MAYBE_UNUSED std::size_t size(char) { return 1; }
MAYBE_UNUSED std::size_t size(const char* s) { return std::strlen(s); }
std::size_t totalSize() { return 0; }

template <typename First, typename... Rest>
std::size_t
totalSize(const First& first, const Rest&... rest)
{
    return size(first) + totalSize(rest...);
}

void stringCatHelper(std::string&) {}

template <typename First, typename... Rest>
void
stringCatHelper(std::string& result, const First& first, const Rest&... rest)
{
    result += first;
    stringCatHelper(result, rest...);
}

template <typename... T>    
std::string
stringCat(const T&... vals)
{
    std::string result;
    result.reserve(totalSize(vals...));
    stringCatHelper(result, vals...);
    return result;
}

//------------------------------

inline
std::string
jsonKey(const std::string& key)
{
    return stringCat("\"", key, "\"");
}

inline
std::string
jsonVal(const std::string& v)
{
    return stringCat("{\n", scene_rdl2::str_util::addIndent(v), "\n}");
}

inline
std::string
jsonPairObj(const std::string& key, const std::string& pairs)
{
    return stringCat(jsonKey(key), ':', jsonVal(pairs));
}

inline
std::string
jsonPairStr(const std::string& key, const std::string& v)
{
    return stringCat(jsonKey(key), ":\"", v, "\"");
}

inline
std::string
jsonPairUnsigned(const std::string& key, const unsigned v)
{
    return stringCat(jsonKey(key), ':', std::to_string(v));
}

inline
std::string    
jsonPairSizeT(const std::string& key, const size_t v)
{
    return stringCat(jsonKey(key), ':', std::to_string(v));
}

inline
std::string
jsonPairFloat(const std::string& key, const float v)
{
    return stringCat(jsonKey(key), ':', std::to_string(v));
}

inline
std::string    
jsonPairBool(const std::string& key, const bool b)
{
    return stringCat(jsonKey(key), ':', (b)? "true": "false");
}

inline
std::string
jsonPairTimeVal(const std::string& key, const struct timeval& tv)
{
    return stringCat(jsonKey(key), ":{",
                     jsonPairStr("date", scene_rdl2::time_util::timeStr(tv)), ',',
                     jsonPairUnsigned("sec", tv.tv_sec), ',',
                     jsonPairUnsigned("usec", tv.tv_usec), "}");
}

} // namespace

//---------------------------------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------------------------------

namespace moonray {
namespace rndr {

ResumeHistoryMetaData::MCRTStint::MCRTStint()
{
    scene_rdl2::time_util::init(mMCRTStartTime);
    scene_rdl2::time_util::init(mMCRTEndTime);
}

ResumeHistoryMetaData::ResumeHistoryMetaData()
{
    scene_rdl2::time_util::init(mProcStartTime);
    scene_rdl2::time_util::init(mFrameStartTime);
    scene_rdl2::time_util::init(mFinalizeSyncStartTime);
    scene_rdl2::time_util::init(mFinalizeSyncEndTime);
}

void
ResumeHistoryMetaData::setUniformSample(const unsigned minSamples,
                                        const unsigned maxSamples)
{
    mSampleType = SAMPLE_TYPE::UNIFORM;
    mMinSamples = minSamples;
    mMaxSamples = maxSamples;
    mAdaptiveTargetError = 0.0f;
}

void    
ResumeHistoryMetaData::setAdaptiveSample(const unsigned minSamples,
                                         const unsigned maxSamples,
                                         const float targetError)
{
    mSampleType = SAMPLE_TYPE::ADAPTIVE;
    mMinSamples = minSamples;
    mMaxSamples = maxSamples;
    mAdaptiveTargetError = targetError;
}

void
ResumeHistoryMetaData::setStartTileSampleId(const unsigned startTileSampleId)
{
    mStartTileSamplesId = startTileSampleId;
}

float
ResumeHistoryMetaData::getTimeSaveSecBySignalCheckpoint() const
{
    if (mMCRT.size() <= 0) return 0.0f;

    const MCRTStint& lastStint = mMCRT.back();
    if (!lastStint.mExtraSnapshotFlag) return 0.0f;
    
    // last stint was created by signal-based checkpoint logic
    return secInterval(lastStint.mMCRTStartTime, lastStint.mMCRTEndTime);
}

std::string
ResumeHistoryMetaData::resumeHistory(const std::string& oldHistory,
                                     const pbr::Statistics& pbrStats) const
{
    std::string str = oldHistory;
    if (oldHistory.empty()) {
        str = "{\n";
        str += jsonKey("history"); str += ":[\n";
    } else {
        int pos = str.find_last_of(']');
        if (pos != std::string::npos) str.resize(pos);
        str += ",\n";
    }
    str += toJson(pbrStats); str += "\n";
    str += "]\n";
    str += "}";
    return str;
}

std::string
ResumeHistoryMetaData::showAllStint() const
{
    return toJsonAllStint();
}

//---------------------------------------------------------------------------------------------------------------

std::string
ResumeHistoryMetaData::toJson(const pbr::Statistics& pbrStats) const
{
    return jsonVal(stringCat(toJsonSampleInfo(pbrStats), ",\n",
                             toJsonExecEnvInfo(), ",\n",
                             toJsonTimingInfo(), ",\n",
                             toJsonTimingSummary()));
}

std::string
ResumeHistoryMetaData::toJsonSampleInfo(const pbr::Statistics& pbrStats) const
{
    std::string str;
    switch (mSampleType) {
    case SAMPLE_TYPE::UNIFORM :
        str = stringCat(jsonPairStr("samplingType", "UNIFORM"), ",\n",
                        jsonPairUnsigned("minSamples", mMinSamples), ",\n",
                        jsonPairUnsigned("maxSamples", mMaxSamples), ",\n");
        break;
    case SAMPLE_TYPE::ADAPTIVE :
        // acoording to the RenderContext.cc RenderContext::buildFrameState(). We are using 10000x bigger number
        // for user control of adaptive sampling error target.
        float targetError = mAdaptiveTargetError * 10000.0f;
        str = stringCat(jsonPairStr("samplingType", "ADAPTIVE"), ",\n",
                        jsonPairUnsigned("minSamples", mMinSamples), ",\n",
                        jsonPairUnsigned("maxSamples", mMaxSamples), ",\n",
                        jsonPairFloat("adaptiveTargetError", targetError), ",\n");
        break;
    }
    str += toJsonSampleResultStat(pbrStats);
    return jsonPairObj("sampling", str);
}

std::string
ResumeHistoryMetaData::toJsonSampleResultStat(const pbr::Statistics& pbrStats) const
{
    const size_t pixelSamples = pbrStats.getCounter(pbr::STATS_PIXEL_SAMPLES);
    const size_t lightSamples = pbrStats.getCounter(pbr::STATS_LIGHT_SAMPLES);
    const size_t bsdfSamples = pbrStats.getCounter(pbr::STATS_BSDF_SAMPLES);
    const size_t bssrdfSamples = pbrStats.getCounter(pbr::STATS_SSS_SAMPLES);
    const size_t totalSamples = pixelSamples + lightSamples + bsdfSamples + bssrdfSamples;

    std::string str;
    str = stringCat(jsonPairSizeT("PixelSamples", pixelSamples), ",\n",
                    jsonPairSizeT("lightSamples", lightSamples), ",\n",
                    jsonPairSizeT("bsdfSamples", bsdfSamples), ",\n",
                    jsonPairSizeT("bssrdfSamples", bssrdfSamples), ",\n",
                    jsonPairSizeT("totalSamples", totalSamples));
    return jsonPairObj("sampleResult", str);
}

std::string    
ResumeHistoryMetaData::toJsonExecEnvInfo() const
{
    std::string str = stringCat(jsonPairStr("hostname", getHostname()), ",\n",
                                jsonPairUnsigned("numberOfThreads", mNumOfThreads), ",\n",
                                jsonPairFloat("UTCOffsetHours", scene_rdl2::time_util::utcOffsetHours()));

    auto addEnvVal = [](std::string &str, const std::string& envParam) {
        std::string envStr = getEnvVal(envParam);
        if (!envStr.empty()) {
            str += ",\n";
            str += jsonPairStr(envParam, envStr);
        }
    };
    addEnvVal(str, "MNRY_HOST_RU");
    addEnvVal(str, "MNRY_FULL_ID");

    return jsonPairObj("execEnv", str);
}

std::string
ResumeHistoryMetaData::toJsonTimingInfo() const
{
    return jsonPairObj("timingDetail",
                       stringCat(jsonPairBool("bgCheckpointWrite", mBgCheckpointWriteMode), ",\n",
                                 jsonPairUnsigned("startTileSamplesId", mStartTileSamplesId), ",\n",
                                 jsonPairTimeVal("procStartTime", mProcStartTime), ",\n",
                                 jsonPairTimeVal("frameStartTime", mFrameStartTime), ",\n",
                                 toJsonAllStint()));
}

std::string
ResumeHistoryMetaData::toJsonAllStint() const
{
    auto isActiveStint = [&](size_t id) -> bool {
        if (id >= mMCRT.size()) return false; // outside MCRT stint range
        if (id == 0) return true; // always active for 1st stint
        if (mMCRT[id - 1].mEndTileSamplesId == mMCRT[id].mEndTileSamplesId) {
            return false; // This is non active empty stint
        }
        return true;
    };

    std::string str = stringCat(jsonKey("MCRT"), ":[\n");
    for (size_t id = 0; id < mMCRT.size(); ++id) {
        if (isActiveStint(id)) { // We skip empty MCRTStint
            str += scene_rdl2::str_util::addIndent(toJsonStint(id));
            if (id < mMCRT.size() - 1) str += ",\n";
            else                       str += '\n';
        }
    }
    str += "]";
    return str;
}

std::string
ResumeHistoryMetaData::toJsonStint(size_t id) const
{
    return jsonVal(stringCat(jsonPairUnsigned("stint", id), ",\n",
                             jsonPairBool("extraSnapshot", mMCRT[id].mExtraSnapshotFlag), ",\n",
                             jsonPairTimeVal("MCRTStartTime", mMCRT[id].mMCRTStartTime), ",\n",
                             jsonPairTimeVal("MCRTEndTime", mMCRT[id].mMCRTEndTime), ",\n",
                             jsonPairUnsigned("endTileSamplesId", mMCRT[id].mEndTileSamplesId)));
}

std::string
ResumeHistoryMetaData::toJsonTimingSummary() const
{
    float renderPrepSec, mcrtSec, checkpointTotalSecExcludeLast, checkpointAveSec;
    float timeSaveSecBySignalCheckpoint;
    computeTimingSummary(renderPrepSec, mcrtSec, checkpointTotalSecExcludeLast, checkpointAveSec,
                         timeSaveSecBySignalCheckpoint);
    
    std::string str =
        stringCat(jsonPairFloat("renderPrepSec", renderPrepSec), ",\n",
                  jsonPairFloat("mcrtSec", mcrtSec), ",\n",
                  jsonPairFloat("checkpointTotalSecExcludeLast", checkpointTotalSecExcludeLast), ",\n",
                  jsonPairUnsigned("checkpointTotal", mMCRT.size()), ",\n",
                  jsonPairFloat("checkpointAverageSec", checkpointAveSec));
    if (timeSaveSecBySignalCheckpoint > 0.0f) {
        str = stringCat(str, ",\n",
                        jsonPairFloat("timeSaveSecBySignalCheckpoint", timeSaveSecBySignalCheckpoint));
    }
    return jsonPairObj("timingSummary", str);
}

void
ResumeHistoryMetaData::computeTimingSummary(float& renderPrepSec,
                                            float& mcrtSec,
                                            float& checkpointTotalSecExcludeLast,
                                            float& checkpointAveSec,
                                            float& timeSaveSecBySignalCheckpoint) const
{
    renderPrepSec = 0.0f;
    mcrtSec = 0.0f;
    checkpointTotalSecExcludeLast = 0.0f; // total time of checkpoint output action exclude last output

    if (!mMCRT.size()) return;

    // renderPrep = frameStartTime ~ 1st MCRT startTime
    renderPrepSec = secInterval(mFrameStartTime, mMCRT[0].mMCRTStartTime);
    for (unsigned i = 0; i < mMCRT.size(); ++i) {
        if (i > 0) {
            checkpointTotalSecExcludeLast += secInterval(mMCRT[i-1].mMCRTEndTime, mMCRT[i].mMCRTStartTime);
        }
        mcrtSec += secInterval(mMCRT[i].mMCRTStartTime, mMCRT[i].mMCRTEndTime);
    }

    if (mMCRT.size() > 1) {
        checkpointAveSec = checkpointTotalSecExcludeLast / (float)(mMCRT.size() - 1);
    } else {
        checkpointAveSec = 0.0f;
    }

    timeSaveSecBySignalCheckpoint = getTimeSaveSecBySignalCheckpoint();
}

float
ResumeHistoryMetaData::secInterval(const struct timeval& start, const struct timeval& end) const
{
    return (float)(end.tv_sec - start.tv_sec) + (float)(end.tv_usec - start.tv_usec) / 1000.0f / 1000.0f;
}

} // namespace rndr
} // namespace moonray
