// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

//
//
#pragma once

#include <scene_rdl2/common/platform/Platform.h>

#include <vector>
#include <sstream>
#include <stddef.h>
#include <float.h>
#include <limits>

// This directive makes to keep all passes timing record in memory for debugging purpose.
// If this is on, memory for record timing result will increase on demand. This might cause problem if
// passes total is huge number even each pass timing info is small.
// If this directive is off, we only keep at least last 64 samples per tile passes result.
// Default should be commented out.
//#define KEEP_ALL_PASSES_INFO

//#define PRINT_DEBUG_MESSAGE

namespace moonray {
namespace rndr {

class RenderEngineTimingRecord
//
// Timing information for 1 render thread engine
//
{
public:
    RenderEngineTimingRecord()
        : mId(0)
    {
        mValuesDBL.resize(id(TagDBL::SIZE), 0.0);
        mValuesULL.resize(id(TagULL::SIZE), 0ULL);
    }
    RenderEngineTimingRecord(const RenderEngineTimingRecord &) = default;
    RenderEngineTimingRecord(RenderEngineTimingRecord &&) = default;

    RenderEngineTimingRecord &operator =(const RenderEngineTimingRecord &) = default;
    RenderEngineTimingRecord &operator =(RenderEngineTimingRecord &&) = default;

    //
    //   Measurement timing inside RenderDriver::renderPasses() function
    //
    //           |<------ ACTIVE : active working time ----->|
    //   +-+-----+---+----------------------------+-------+--+-----+  -> time+
    //   ^ ^  :  ^   ^        render tiles        ^ queue ^  ^  :  ^
    //   | |  :  |   |                            | drain |  |  :  |
    //   | |  :  |   |                            |       |  |  :  +-- allEnd : finish thread (ENDGAP = allEnd - timeEnd)
    //   | |  :  |   |                            |       |  |  +... waiting for other all threads are finished
    //   | |  :  |   |                            |       |  |
    //   | |  :  |   |                            |       |  +-- timeEnd : completed rendering
    //   | |  :  |   |                            |       |      (FINISHUP = timeEnd - timeQueueDraining)
    //   | |  :  |   |                            |       |
    //   | |  :  |   |                            |       +-- timeQueueDraining : completed queue draining stage
    //   | |  :  |   |                            |           (QUEUEDRAINING = timeQueueDraining - timeRenderTiles)
    //   | |  :  |   |                            |
    //   | |  :  |   |                            +-- timeRenderTiles : completed tiles
    //   | |  :  |   |                                (RENDERTILES = timeRenderTiles - timeInitTexturing)
    //   | |  :  |   |
    //   | |  :  |   +-- timeInitTexturing : completed texture init (INITTEXTURING = timeInitTextureing - timeReady)
    //   | |  :  |
    //   | |  :  +-- timeReady : thread is read to work (READY = timeReady - start)
    //   | |  +... waiting for other all threads are booted
    //   | |
    //   | +-- timeStart : render engine thread booted and start working (START = timeStart - start)
    //   +-- start
    //    

    // enum for double value
    enum class TagDBL : int {
        START = 0,              // duration of start to thread is start working
        READY,                  // duration of start to ready to working 
        INITTEXTURING,          // duration of initialize texture staff
        RENDERTILES,            // duration of rendering tiles
        QUEUEDRAINING,          // duration of queue draining
        FINISHUP,               // duration of finish up
        ENDGAP,                 // duration of end gap (waiting other threads are finished)
        ACTIVE,                 // actively working time

        SIZE                    // total tagDBL size
    };

    // enum for unsigned long long value
    enum class TagULL : int {
        PROCESSEDTILESTOTAL,    // processed tiles total number
        PROCESSEDSAMPLETOTAL,   // processed sample total number

        SIZE                    // total tagULL size
    };

    void setId(const size_t id) { mId = id; }

    void init(const double d = 0.0, const unsigned long long ull = 0ULL) {
        updateDBL(d, [](double& a, const double b) { a = b; });
        updateULL(ull, [](unsigned long long& a, const unsigned long long b) { a = b; });
    }
    void add(const RenderEngineTimingRecord &rec) {
        update(rec,
               [](double& a, const double b) { a += b; },
               [](unsigned long long& a, const unsigned long long b) { a += b; });
    }
    void scale(const double v) {
        updateDBL(v, [](double& a, const double b) { a *= b; });
        scaleULL(v);
    }
    void min(const RenderEngineTimingRecord &rec) {
        update(rec,
               [](double &a, const double b) { a = std::min(a, b); },
               [](unsigned long long& a, const unsigned long long b) { a = std::min(a, b); });
    }
    void max(const RenderEngineTimingRecord &rec) {
        update(rec,
               [](double& a, const double b) { a = std::max(a, b); },
               [](unsigned long long& a, const unsigned long long b) { a = std::max(a, b); });
    }

    RenderEngineTimingRecord &operator +=(const RenderEngineTimingRecord &rec) { add(rec); return *this; }
    RenderEngineTimingRecord &operator *=(const double v) { scale(v); return *this; }
    RenderEngineTimingRecord &operator !=(const RenderEngineTimingRecord &rec) { min(rec); return *this; }
    RenderEngineTimingRecord &operator ^=(const RenderEngineTimingRecord &rec) { max(rec); return *this; }

    int id(const TagDBL& tag) const { return static_cast<int>(tag); }
    void set(const TagDBL& tag, const double val) { mValuesDBL[id(tag)] = val; }
    double get(const TagDBL& tag) const { return mValuesDBL[id(tag)]; } // return sec
    double getMs(const TagDBL &tag) const { return get(tag) * 1000.0; } // return millisec

    int id(const TagULL& tag) const { return static_cast<int>(tag); }
    void set(const TagULL& tag, const unsigned long long val) { mValuesULL[id(tag)] = val; }
    unsigned long long get(const TagULL& tag) const { return mValuesULL[id(tag)]; }

    std::string showSimple() const;
    std::string showDetail() const;

protected:
    void update(const RenderEngineTimingRecord &rec,
                void (*updateDBLFunc)(double &, const double),
                void (*updateULLFunc)(unsigned long long&, const unsigned long long)) {
        for (size_t i = 0; i < mValuesDBL.size(); ++i) updateDBLFunc(mValuesDBL[i], rec.mValuesDBL[i]);
        for (size_t i = 0; i < mValuesULL.size(); ++i) updateULLFunc(mValuesULL[i], rec.mValuesULL[i]);
    }
    void updateDBL(const double v,
                   void (*updateDBLFunc)(double &, const double)) {
        for (size_t i = 0; i < mValuesDBL.size(); ++i) updateDBLFunc(mValuesDBL[i], v);
    }
    void updateULL(const unsigned long long v,
                   void (*updateULLFunc)(unsigned long long&, const unsigned long long)) {
        for (size_t i = 0; i < mValuesULL.size(); ++i) updateULLFunc(mValuesULL[i], v);
    }
    void scaleULL(const double v) {
        for (size_t i = 0; i < mValuesULL.size(); ++i) {
            mValuesULL[i] = static_cast<unsigned long long>(static_cast<double>(mValuesULL[i]) * v);
        }
    }

    size_t mId;      // tls->mThreadIdx
    std::vector<double> mValuesDBL;
    std::vector<unsigned long long> mValuesULL;
}; // class RenderEngineTimingRecord

class RenderPassesTimingRecord
//
// Timing information for 1 pass which includes multiple render thread engines information.
//
{
public:
    RenderPassesTimingRecord() { init(); }
    RenderPassesTimingRecord(const RenderPassesTimingRecord &) = default;
    RenderPassesTimingRecord(RenderPassesTimingRecord &&) = default;

    RenderPassesTimingRecord &operator = (const RenderPassesTimingRecord &) = default;
    RenderPassesTimingRecord &operator = (RenderPassesTimingRecord &&) = default;

    void init() {
        mTimeBudget = 0.0;
        mStartTime = 0.0;
        mEndTime = 0.0;
        mDuration = 0.0;

        mSamplesPerTile = 0;
        mSamplesAll = 0;

        for (auto &currEng : mRenderEngines) currEng.init();
        mRenderEngineMin.init(std::numeric_limits<double>::max(),
                              std::numeric_limits<unsigned long long>::max());
        mRenderEngineMax.init();
        mRenderEngineAverage.init();

        mSampleCost = 0.0;
        mOverheadCost = 0.0;
    }

    void setTimeBudget(const double sec) { mTimeBudget = sec; }

    // only valid realtime/progressiveCheckpoint
    void setRenderPassesSamplesPerTile(const unsigned v) { mSamplesPerTile = v; }

    void startRenderPasses(const size_t totalEngine) {
        mStartTime = scene_rdl2::util::getSeconds();
        if (mRenderEngines.size() < totalEngine) mRenderEngines.resize(totalEngine);
    }
    void finalizeRenderPasses() {
        double now = scene_rdl2::util::getSeconds();

        mRenderEngineMin.init(std::numeric_limits<double>::max(),
                              std::numeric_limits<unsigned long long>::max());
        mRenderEngineMax.init();
        mRenderEngineAverage.init();
        mSamplesAll = 0ULL;
        for (auto &currEng : mRenderEngines) {
            double currGap = now - currEng.get(RenderEngineTimingRecord::TagDBL::ENDGAP);
            currEng.set(RenderEngineTimingRecord::TagDBL::ENDGAP, currGap); // update by proper gap duration.

            mRenderEngineMin != currEng; // min value update
            mRenderEngineMax ^= currEng; // max value update
            mRenderEngineAverage += currEng; // add
            mSamplesAll += currEng.get(RenderEngineTimingRecord::TagULL::PROCESSEDSAMPLETOTAL);
        }
        mRenderEngineAverage *=
            static_cast<double>(1.0 / static_cast<double>(std::max(mRenderEngines.size(), (size_t)1)));

        mEndTime = scene_rdl2::util::getSeconds();
        mDuration = mEndTime - mStartTime;

        double averageActiveDuration = getActiveDuration();
        mSampleCost = averageActiveDuration / static_cast<double>(std::max(mSamplesPerTile, (unsigned)1));
        mOverheadCost = mDuration - averageActiveDuration;
    }
    
    double getStartTime() const { return mStartTime; }
    double getEndTime() const { return mEndTime; }

    RenderEngineTimingRecord &getEngineTimingRecord(const size_t id) { return mRenderEngines[id]; }

    unsigned getSamplesPerTile() const { return mSamplesPerTile; } // only valid when realtime/progressiveCheckpoint renderMode
    unsigned long long getSamplesAll() const { return mSamplesAll; }
    double getSampleCost() const { return mSampleCost; } // only valid when realtime/progressiveCheckpoint renderMode
    double getActiveDuration() const { return mRenderEngineAverage.get(RenderEngineTimingRecord::TagDBL::ACTIVE); }
    double getOverheadCost() const { return mOverheadCost; }

    std::string show() const;

protected:

    std::string showRenderEngines() const;
    std::string showRecord(const RenderEngineTimingRecord &rec, const std::string &msg) const;

    double mTimeBudget; // timebudget of this renderPasses() (sec)
    double mStartTime;  // RenderDriver::renderPasses() start timing (sec)
    double mEndTime;    // RenderDriver::renderPasses() end timing (sec) : computed by finalizeRenderPasses()
    double mDuration;   // Duration of RenderDriver::renderPasses() (sec) : computed by finalizeRenderPasses()

    unsigned mSamplesPerTile; // Total number of samples per tile for renderPasses()
    unsigned long long mSamplesAll; // computed by finalizeRenderPasses()

    std::vector<RenderEngineTimingRecord> mRenderEngines;
    RenderEngineTimingRecord mRenderEngineMin;     // min info : computed by finalizeRenderPasses()
    RenderEngineTimingRecord mRenderEngineMax;     // max info : computed by finalizeRenderPasses()
    RenderEngineTimingRecord mRenderEngineAverage; // average info : computed by finalizeRenderPasses()

    double mSampleCost;        // sample cost of this renderPasses() : computed by finalizeRenderPasses()
    double mOverheadCost;      // overhead cost of this renderPasses() : computed by finalizeRenderPasses()
}; // class RenderPassesTimingRecord

class RenderFrameTimingRecord
//
// Timing information for multiple passes
//
{
public:
    RenderFrameTimingRecord() :
        mLastFrameUpdate(0.0),
        mRenderFrameStartTime(0.0),
        mRenderFrameEndTime(-1.0),
        mRenderFrameTimeBudget(0.0),
        mNumFramesRendered(0) {
        reset(0);
    }

    void reset(unsigned initialNumSampleAll);
    void newStint();

    double getLastFrameUpdate() const { return mLastFrameUpdate; }
    double setRenderFrameTimeBudget(const double t) { // t = sec
        mRenderFrameTimeBudget = t;
        return mRenderFrameStartTime + mRenderFrameTimeBudget; // return predicted frame end time
    }

    //------------------------------

    void setRenderPassesSamplesPerTile(const unsigned newSamplesPerTile) {
#ifdef KEEP_ALL_PASSES_INFO
        if (mRenderPasses.size() < mNumPassesRendered + 1) mRenderPasses.resize(mNumPassesRendered + 1); // Just in case
        RenderPassesTimingRecord &currPass = mRenderPasses[mNumPassesRendered];
#else  // else KEEP_ALL_PASSES_INFO
        updateRenderPassesTable(newSamplesPerTile, 4096); // maintaine passes for last 4096 samples per tile
        RenderPassesTimingRecord &currPass = mRenderPasses[0];
#endif // end !KEEP_ALL_PASSES_INFO
        currPass.init();
        currPass.setTimeBudget(mTimeBudget);
        currPass.setRenderPassesSamplesPerTile(newSamplesPerTile);
    }
    RenderPassesTimingRecord &startRenderPasses(const size_t totalEngine) {
#ifdef KEEP_ALL_PASSES_INFO
        if (mRenderPasses.size() < mNumPassesRendered + 1) mRenderPasses.resize(mNumPassesRendered + 1); // Just in case
        RenderPassesTimingRecord &currPass = mRenderPasses[mNumPassesRendered];
#else  // else KEEP_ALL_PASSES_INFO
        if (mRenderPasses.empty()) mRenderPasses.resize(1); // This is for Batch/Progress render mode
        RenderPassesTimingRecord &currPass = mRenderPasses[0];
#endif // end !KEEP_ALL_PASSES_INFO
        currPass.startRenderPasses(totalEngine);
        return currPass;
    }
    void finalizeRenderPasses() {
#ifdef KEEP_ALL_PASSES_INFO
        RenderPassesTimingRecord &currPass = mRenderPasses[mNumPassesRendered];
#else  // else KEEP_ALL_PASSES_INFO
        RenderPassesTimingRecord &currPass = mRenderPasses[0];
#endif // end !KEEP_ALL_PASSES_INFO
        currPass.finalizeRenderPasses();

        mNumSamplesPerTile += currPass.getSamplesPerTile();
        mNumSamplesAll += currPass.getSamplesAll();
        mSampleCost = currPass.getSampleCost();
        mOverheadCost = currPass.getOverheadCost();
        mTotalOverheadDuration += currPass.getOverheadCost();
        mTotalActiveDuration += currPass.getActiveDuration();
        mAveragedSampleCost = mTotalActiveDuration / static_cast<double>(std::max(mNumSamplesPerTile, (unsigned)1));
        /* Useful debug message
        std::cerr << ">>   RenderTimingRecord.h addSamples:" << currPass.getSamplesPerTile()
                  << " addActiveDuration:" << currPass.getActiveDuration()
                  << " sampleCost:" << currPass.getSampleCost()
                  << std::endl;
        std::cerr << ">>   RenderTimingRecord.h mNumSamplesPerTile:" << mNumSamplesPerTile
                  << " mTotalActiveDuration:" << mTotalActiveDuration
                  << " mAveragedSampleCost: " << mAveragedSampleCost
                  << std::endl;
        */

        if (mNumPassesRendered == 0) {
            mEstimatedSampleCost = currPass.getSampleCost(); // estimation phase result
        }

        ++mNumPassesRendered;
    }
    const RenderPassesTimingRecord & getPasses(const int passId) const {
#ifdef KEEP_ALL_PASSES_INFO
        return mRenderPasses[passId];
#else  // else KEEP_ALL_PASSES_INFO
        return mRenderPasses[0];
#endif // end !KEEP_ALL_PASSES_INFO
    }

    //------------------------------

    unsigned getNumFramesRendered() const { return mNumFramesRendered; }
    unsigned getNumPassesRendered() const { return mNumPassesRendered; }
    unsigned getNumSamplesPerTile() const { return mNumSamplesPerTile; }
    unsigned long long getNumSamplesAll() const { return mNumSamplesAll; }

    double getEstimatedSampleCost() const { return mEstimatedSampleCost; }
    
    bool isComplete(const double timeBudgetSec, const double now)
    // Do we have time to render more ?
    {
        if (timeBudgetSec < (mOverheadCost + mSampleCost)) {
            mRenderFrameEndTime = now;
            mRenderFrameDuration = mRenderFrameEndTime - mRenderFrameStartTime;
            return true;        // this frame is completed. out of time budget.
        }
        return false;           // we should render more
    }
    double estimateSamples(const double timeBudgetSec)
    // Compute how many new samples we can safely render within the remaining time interval.
    // Estimation is done based on updated sampling cost by at least last 64 samples per tile.
    {
        mTimeBudget = timeBudgetSec;
        double remainingTime = mTimeBudget - mOverheadCost;

        double sampleCost = 0.0;
        {
            // We have to estimate sampleCost by history of last several passes.
            // mRenderPasses already maintained and keeps up to at least 64 samples per tile information.
            double totalActiveDuration = 0.0;
            unsigned totalSamplesPerTile = 0;
            for (size_t i = 0; i < mRenderPasses.size(); ++i) {
                totalActiveDuration += mRenderPasses[i].getActiveDuration();
                totalSamplesPerTile += mRenderPasses[i].getSamplesPerTile();
            }
            sampleCost = totalActiveDuration / static_cast<double>(std::max(totalSamplesPerTile, (unsigned)1));
#           ifdef PRINT_DEBUG_MESSAGE
            std::ostringstream ostr;
            ostr << "estimateSamples mRenderPasses.size():" << mRenderPasses.size() << " {\n";
            for (size_t i = 0; i < mRenderPasses.size(); ++i) {
                ostr << "  i:" << i
                     << " activeDuration:" << mRenderPasses[i].getActiveDuration()
                     << " samplePerTile:" << mRenderPasses[i].getSamplesPerTile() << '\n';
            }
            ostr << "} totalActiveDuration:" << totalActiveDuration
                 << " totalSamplesPerTile:" << totalSamplesPerTile
                 << " sampleCost:" << sampleCost;
            std::cerr << ostr.str() << std::endl;
#           endif // end PRINT_DEBUG_MESSAGE
        }
        return (remainingTime < 0.0)? 0.0: (remainingTime / sampleCost);
    }
    double estimateSamplesAllStint(const double timeBudgetSec)
    // Compute how many new samples we can safely render within the remaining time interval.
    // Estimation is done based on updated sampling cost by all executed passes from reset().
    {
        mTimeBudget = timeBudgetSec;
        double remainingTime = mTimeBudget - mOverheadCost;
        /* Useful debug message
        std::cerr << ">>         RenderTimingRecord.h remainingTime:" << remainingTime
                  << " mAveragedSampleCost:" << mAveragedSampleCost << std::endl;
        */
        return (remainingTime < 0.0)? 0.0: (remainingTime / mAveragedSampleCost);
    }
    double actualSampleCost() const { // compute actual sample cost based on actual active duration of engine thread
        return (mNumSamplesPerTile <= 1)? 0.0: mTotalActiveDuration / static_cast<double>(mNumSamplesPerTile - 1);
    }

    double getRenderFrameStartTime() const { return mRenderFrameStartTime; }
    double getRenderFrameEndTime() const { return mRenderFrameEndTime; }
    double getTotalOverheadDuration() const { return mTotalOverheadDuration; }
    double getTotalActiveDuration() const { return mTotalActiveDuration; }
    double getAveragedSampleCost() const { return mAveragedSampleCost; }

    //------------------------------

    std::string show() const;
    std::string showSimple() const;

protected:

    // maintaine passes for last requestedKeepSamplesPerTile samples per tile
    void updateRenderPassesTable(const unsigned newSamplesPerTile,
                                 const unsigned requestedKeepSamplesPerTile);

    std::string showPasses() const;

    double mLastFrameUpdate;       // last frame's update duration (= lastEnd ~ currentStart)
    double mRenderFrameStartTime;  // time of renderFrame() start 
    double mRenderFrameEndTime;    // time of last isComplete() retuns true moment (not renderFrame() end timing)
    double mRenderFrameTimeBudget; // time bugdget for current frame's renderFrame() (i.e. = frame duration) sec
    double mRenderFrameDuration;   // actual renderFrame() duration. (renderFrame()'s start ~ isComplete():true)

    // Keep all passes or only keep at least last 64 samples per tile passes (depends on KEEP_ALL_PASSES_INFO directive)
    std::vector<RenderPassesTimingRecord> mRenderPasses; // multiple renderPasses timing records

    unsigned mNumFramesRendered; // total frames rendered from process start
    unsigned mNumPassesRendered; // total passes rendered at this renderFrame()
    unsigned mNumSamplesPerTile; // total samples per tile over multiple passes includes estimation phase at this renderFrame()
    unsigned long long mNumSamplesAll; // total samples for this renderFrame()

    double mEstimatedSampleCost; // estimated sample cost by estimation phase (only estimation phase result)

    double mTimeBudget;          // current time budget for renderPasses() (sec)
    double mOverheadCost;        // current overhead cost for single renderPasses() (sec)
    double mSampleCost;          // current best estimation for single sample per tile (sec)

    double mTotalOverheadDuration; // total overhead duration about renderFrame() (sec)
    double mTotalActiveDuration;   // total active renderPasses() duration (sec)
    double mAveragedSampleCost;    // averaged sample cost of one sample per tile for all phases.
}; // class RenderFrameTimingRecord

} // namespace rndr
} // namespace moonray

