// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

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
    void setMCRTStintEndTime(const unsigned n, const bool extraSnapshotFlag)
    {
        mMCRT.back().mMCRTEndTime = scene_rdl2::time_util::getCurrentTime();
        mMCRT.back().mEndTileSamplesId = n;
        mMCRT.back().mExtraSnapshotFlag = extraSnapshotFlag;
    }
    void setFinalizeSyncStartTime() { mFinalizeSyncStartTime = scene_rdl2::time_util::getCurrentTime(); }
    void setFinalizeSyncEndTime() { mFinalizeSyncEndTime = scene_rdl2::time_util::getCurrentTime(); }

    float getTimeSaveSecBySignalCheckpoint() const;

    std::string resumeHistory(const std::string& oldHistory, const pbr::Statistics& pbrStats) const;
    std::string showAllStint() const; // for debug

private:
    class MCRTStint
    {
    public:
        MCRTStint();

        struct timeval mMCRTStartTime;
        struct timeval mMCRTEndTime;

        unsigned mEndTileSamplesId {0};

        bool mExtraSnapshotFlag {false};
    };

    enum class SAMPLE_TYPE { UNIFORM, ADAPTIVE };

    bool mBgCheckpointWriteMode {true};

    struct timeval mProcStartTime;
    struct timeval mFrameStartTime;

    unsigned mNumOfThreads {1};

    SAMPLE_TYPE mSampleType {SAMPLE_TYPE::UNIFORM};
    unsigned mMinSamples {0};
    unsigned mMaxSamples {0};
    float mAdaptiveTargetError {0.0f};

    unsigned mStartTileSamplesId {0};

    std::vector<MCRTStint> mMCRT;

    struct timeval mFinalizeSyncStartTime;
    struct timeval mFinalizeSyncEndTime;

    //------------------------------

    std::string toJson(const pbr::Statistics& pbrStats) const;
    std::string toJsonSampleInfo(const pbr::Statistics& pbrStats) const;
    std::string toJsonSampleResultStat(const pbr::Statistics& pbrStats) const;
    std::string toJsonExecEnvInfo() const;
    std::string toJsonTimingInfo() const;
    std::string toJsonTimingSummary() const;
    std::string toJsonAllStint() const;
    std::string toJsonStint(size_t id) const;

    void computeTimingSummary(float& renPrepSec,
                              float& mcrtSec,
                              float& checkpointTotalSecExcludeLast,
                              float& checkpointAveSec,
                              float& timeSaveSecBySignalCheckpoint) const;
    float secInterval(const struct timeval& start, const struct timeval& end) const;
};

} // namespace rndr
} // namespace moonray
