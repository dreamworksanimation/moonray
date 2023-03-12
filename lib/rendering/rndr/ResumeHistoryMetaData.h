// Copyright 2023 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

//
//
#pragma once

#include <scene_rdl2/render/util/TimeUtil.h>

#include <sstream>
#include <string>
#include <vector>

namespace moonray {

namespace pbr {
    class Statistics;
}

namespace rndr {

class ResumeHistoryMetaData
{
public:
    ResumeHistoryMetaData();

    void setBgCheckpointWriteMode(const bool mode) { mBgCheckpointWriteMode = mode; }

    void setProcStartTime() { mProcStartTime = scene_rdl2::time_util::getCurrentTime(); }
    void setFrameStartTime() { mFrameStartTime = scene_rdl2::time_util::getCurrentTime(); }
    void setNumOfThreads(const unsigned n) { mNumOfThreads = n; }
    void setUniformSample(const unsigned minSamples, const unsigned maxSamples);
    void setAdaptiveSample(const unsigned minSamples, const unsigned maxSamples, const float targetError);
    void setStartTileSampleId(const unsigned startTileSamplesId);

    void setMCRTStintStartTime()
    {
        mMCRT.emplace_back();
        mMCRT.back().mMCRTStartTime = scene_rdl2::time_util::getCurrentTime();
    }
    void setMCRTStintEndTime(const unsigned n)
    {
        mMCRT.back().mMCRTEndTime = scene_rdl2::time_util::getCurrentTime();
        mMCRT.back().mEndTileSamplesId = n;
    }
    void setFinalizeSyncStartTime() { mFinalizeSyncStartTime = scene_rdl2::time_util::getCurrentTime(); }
    void setFinalizeSyncEndTime() { mFinalizeSyncEndTime = scene_rdl2::time_util::getCurrentTime(); }

    std::string resumeHistory(const std::string &oldHistory,
                              const pbr::Statistics &pbrStats) const;

private:
    class MCRTStint
    {
    public:
        MCRTStint();

        struct timeval mMCRTStartTime;
        struct timeval mMCRTEndTime;

        unsigned mEndTileSamplesId;
    };

    enum class SAMPLE_TYPE { UNIFORM, ADAPTIVE };

    bool mBgCheckpointWriteMode;

    struct timeval mProcStartTime;
    struct timeval mFrameStartTime;

    unsigned mNumOfThreads;

    SAMPLE_TYPE mSampleType;
    unsigned mMinSamples;
    unsigned mMaxSamples;
    float mAdaptiveTargetError;

    unsigned mStartTileSamplesId;

    std::vector<MCRTStint> mMCRT;

    struct timeval mFinalizeSyncStartTime;
    struct timeval mFinalizeSyncEndTime;

    //------------------------------

    std::string toJson(const pbr::Statistics &pbrStats) const;
    std::string toJsonSampleInfo(const pbr::Statistics &pbrStats) const;
    std::string toJsonSampleResultStat(const pbr::Statistics &pbrStats) const;
    std::string toJsonExecEnvInfo() const;
    std::string toJsonTimingInfo() const;
    std::string toJsonTimingSummary() const;

    void computeTimingSummary(float &renPrepSec, float &mcrtSec,
                              float &checkpointTotalSecExcludeLast,
                              float &checkpointAveSec) const;
    float secInterval(const struct timeval &start, const struct timeval &end) const;
};

} // namespace rndr
} // namespace moonray

