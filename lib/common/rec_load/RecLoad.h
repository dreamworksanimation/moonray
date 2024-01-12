// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

//
//
#pragma once

//
// -- Multi core load logging APIs --
//
// Tracking multi core load on some host between start and stop by some interval.
// RecLoad creates own watchdoc thread by std::thread (very light weight thread)
// Create special CSV log file and write/read to/from the disk.
//

#include <scene_rdl2/common/rec_time/RecTime.h>

#include <string>
#include <vector>
#include <thread>

#include <sys/time.h>           // timeval

namespace moonray {
namespace rec_load {

//
// single core specification data
//    
class RecLoadCoreSpec
{
public:
    RecLoadCoreSpec(const std::string &model, const float mhz, const size_t cacheKB, const float bogoMips) :
        mModel(model),
        mMHz(mhz),
        mCacheKB(cacheKB),
        mBogoMips(bogoMips)
    {}

    std::string show(const std::string &hd) const;

protected:
    std::string mModel;         // model name
    float       mMHz;           // CPU frequency
    size_t      mCacheKB;       // KByte
    float       mBogoMips;      // bogo mips
};

//
// single core load data at some timing
//
class RecLoadCoreStat
{
public:
    enum class Tag : int {
        USER = 0,               // user
        NICE,                   // nice
        SYS,                    // system
        IDLE,                   // idle

        SIZE
    };

    RecLoadCoreStat() : mData{ 0.0f, 0.0f, 0.0f, 0.0f} {}

    void set(const float user, const float nice, const float sys, const float idle) {
        mData[id(Tag::USER)] = user;
        mData[id(Tag::NICE)] = nice;
        mData[id(Tag::SYS)] = sys;
        mData[id(Tag::IDLE)] = idle;
    }

    void reset() { for (size_t i = 0; i < id(Tag::SIZE); ++i) mData[i] = 0.0f; }

    RecLoadCoreStat genAdd(const RecLoadCoreStat &a) const {
        return dataGen(a, [](const float A, const float B)->float { return A + B; });
    }
    RecLoadCoreStat genSub(const RecLoadCoreStat &a) const {
        return dataGen(a, [](const float A, const float B)->float { return A - B; });
    }
    void updateAdd(const RecLoadCoreStat &a) { return update(a, [](float &A, const float B) { A += B; }); }
    void updateScale(const float v) { update(v, [](float &A, const float B) { A *= B; }); }

    RecLoadCoreStat  operator  +(const RecLoadCoreStat &a) const { return genAdd(a); }
    RecLoadCoreStat  operator  -(const RecLoadCoreStat &a) const { return genSub(a); }
    RecLoadCoreStat &operator +=(const RecLoadCoreStat &a) { updateAdd(a); return *this; }
    RecLoadCoreStat &operator *=(const float v) { updateScale(v); return *this; }

    float getTotal() const { float total = 0.0f; for (size_t i = 0; i < id(Tag::SIZE); ++i) total += mData[i]; return total; }

    float getUser() const { return mData[id(Tag::USER)]; }
    float getSys() const { return mData[id(Tag::SYS)]; }

    void rangeClip(const float minVal, const float maxVal) {
        for (size_t i = 0; i < id(Tag::SIZE); ++i) { mData[i] = std::max(minVal, std::min(maxVal, mData[i])); }
    }

    std::string show(const std::string &hd) const;
    std::string showCSV(const bool title = false) const; // CSV format

protected:

    static size_t id(const Tag &tag) { return (size_t)tag; }

    RecLoadCoreStat dataGen(const RecLoadCoreStat &a, float (*genFunc)(const float, const float)) const {
        RecLoadCoreStat tmp;
        for (size_t i = 0; i < id(Tag::SIZE); ++i) { tmp.mData[i] = genFunc(mData[i], a.mData[i]); }
        return tmp;
    }
    void update(const RecLoadCoreStat &a, void (*updateFunc)(float &, const float)) {
        for (size_t i = 0; i < id(Tag::SIZE); ++i) updateFunc(mData[i], a.mData[i]);
    }
    void update(const float v, void (*updateFunc)(float &, const float)) {
        for (size_t i = 0; i < id(Tag::SIZE); ++i) updateFunc(mData[i], v);
    }

    float mData[(size_t)Tag::SIZE];
};

//
// All cores load data with core average data at some timing
//
class RecLoadCoresStat
{
public:
    void resize(const size_t totalCores) { mCoresStat.resize(totalCores); }

    RecLoadCoreStat &get(const size_t coreId) { return mCoresStat[coreId]; }
    RecLoadCoreStat &getAverage() { return mCoresStatAverage; }
    
    void setTimestamp(const long tv_sec = 0, const long tv_usec = 0);

    void computeAverage();

    std::string show(const std::string &hd) const;
    std::string showAverage(const std::string &hd) const;
    std::string showCSV(const bool title = false) const;

protected:
    static std::string to_string(const struct timeval &timeval);
    std::string showTimestampDate() const;
    std::string showTimestampTime() const;
    
    struct timeval mTimestamp;
    std::vector<RecLoadCoreStat> mCoresStat;
    RecLoadCoreStat mCoresStatAverage; // average of all cores
};

//
// Multiple timing for all cores load data on specific host
//
class RecLoadHostTimelog
{
public:
    
    void clear() { mLog.clear(); }
    size_t size() const { return mLog.size(); }

    std::vector<RecLoadCoresStat> &getLog() { return mLog; }

    RecLoadCoresStat       &operator[](int id)       { return mLog[id]; }
    const RecLoadCoresStat &operator[](int id) const { return mLog[id]; }

    std::string showLog(const std::string &hd) const;
    std::string showAverageLog(const std::string &hd) const;
    std::string showLastLog(const std::string &hd) const;
    std::string showLogCSV() const;

    bool readCSV(std::ifstream &ifs);

protected:
    std::vector<RecLoadCoresStat> mLog;
};

//
// Create own thread (by std::thread) and recording CPU load by some interval.
// Finaly all information will be dumped as string or dumped to disk.
// Also RecLoad can read previously dumped data from disk.
//
class RecLoad
{
public:
    RecLoad(const float intervalSec, const float logSizeMaxMB, const std::string &outname);
    ~RecLoad();

    void startLog();
    void stopLog();

    std::string showCoresSpec(const std::string &hd) const;
    std::string showStartEndDurationCSV() const;
    std::string showAllLogs(const std::string &hd) const { return mLog.showLog(hd); }
    std::string showAllLogsCSV() const { return mLog.showLogCSV(); }
    std::string showLastLog(const std::string &hd) const { return mLog.showLastLog(hd); }

    static std::string showStartEndDurationCSV(const float duration);
    static bool readStartEndDurationCSV(std::ifstream &ifs, float &duration);

protected:

    bool getCoresSpec(); // called from constructor : setup all cores specification information
    
    void stopWatchDogThread();
    void watchDogMainLoop();
    
    // get core load information between previous and current call of getesLoad() and saved int argument cores.
    bool getCoresLoad(RecLoadCoresStat &cores); // non MTsafe

    static bool getCoresStat(std::vector<RecLoadCoreStat> &coresStat); // get non normalize raw value
    std::string showCoresStat(const std::string &hd, std::vector<RecLoadCoreStat> &cores) const;

    void outputLogs();
    void outputLogsCSV();

    //------------------------------

    std::vector<RecLoadCoreSpec> mCoresSpec;     // core spec info
    std::vector<RecLoadCoreStat> mCoresCurrent;  // non normalize raw value
    std::vector<RecLoadCoreStat> mCoresPrevious; // non normalize raw value

    //------------------------------

    float mIntervalSec;  // sec
    float mLogSizeMaxMB; // MByte
    std::string mOutname;

    bool mLogCondition;
    std::thread mWatchDogThread;
    RecLoadHostTimelog mLog;

    //------------------------------

    scene_rdl2::rec_time::RecTime mRecTime;           // time between startLog ~ stopLog
    float mStartEndDurationSec;
};

} // namespace rec_load
} // namespace moonray

