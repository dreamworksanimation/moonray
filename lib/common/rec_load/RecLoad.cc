// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

//
//
#include "RecLoad.h"

#include <sstream>
#include <iomanip>
#include <fstream>
#include <chrono>
#include <algorithm>

#include <time.h>
#include <sys/time.h>

#include <iostream>
#include <unistd.h>

namespace moonray {
namespace rec_load {

//------------------------------------------------------------------------------

std::string
RecLoadCoreSpec::show(const std::string &hd) const
{
    std::ostringstream ostr;
    ostr << hd << "RecLoadCoreSpec {\n";
    ostr << hd << "     mModel:" << mModel << '\n';
    ostr << hd << "       mMHz:" << mMHz << '\n';
    ostr << hd << "   mCacheKB:" << mCacheKB << '\n';
    ostr << hd << "  mBogoMips:" << mBogoMips << '\n';
    ostr << hd << "}";
    return ostr.str();
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------

std::string
RecLoadCoreStat::show(const std::string &hd) const
{
    std::ostringstream ostr;
    static const char *title[] = {
        "user:",
        " nice:",
        " sys:",
        " idle:"
    };

    ostr << hd;
    for (size_t i = 0; i < id(Tag::SIZE); ++i) {
        const float &v = mData[i];
        ostr << title[i];
        if (v < 0.0005f) {
            ostr << "     ";
        } else {
            ostr << std::setw(5) << std::fixed << std::setprecision(3) << v;
        }
    }
    return ostr.str();
}

std::string
RecLoadCoreStat::showCSV(const bool title) const
{
    std::ostringstream ostr;
    if (title) {
        ostr << "user,nice,sys,idle";
    } else {
        for (size_t i = 0; i < id(Tag::SIZE); ++i) {
            if (i != 0) ostr << ",";
            ostr << mData[i];
        }
    }
    return ostr.str();
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------

void
RecLoadCoresStat::setTimestamp(const long tv_sec, const long tv_usec)
{
    if (tv_sec == 0 && tv_usec == 0) {
        gettimeofday(&mTimestamp, NULL);
    } else {
        mTimestamp.tv_sec = tv_sec;
        mTimestamp.tv_usec = tv_usec;
    }
}

void
RecLoadCoresStat::computeAverage()
{
    mCoresStatAverage.reset();
    for (size_t coreId = 0; coreId < mCoresStat.size(); ++coreId) {
        mCoresStatAverage += mCoresStat[coreId];
    }
    mCoresStatAverage *= ((float)1.0f / (float)mCoresStat.size());
}

std::string
RecLoadCoresStat::show(const std::string &hd) const
{
    std::ostringstream ostr;
    ostr << hd << "Cores (total:" << mCoresStat.size() << ") (time:" << to_string(mTimestamp) << ") {\n";
    for (size_t coreId = 0; coreId < mCoresStat.size(); ++coreId) {
        ostr << hd << "  coreId:" << std::setw(2) << coreId << " " << mCoresStat[coreId].show("") << '\n';
    }
    ostr << hd << "}";
    return ostr.str();
}

std::string
RecLoadCoresStat::showAverage(const std::string &hd) const
{
    std::ostringstream ostr;
    ostr << hd << "Cores Average " << mCoresStatAverage.show("") << " (time:" << to_string(mTimestamp) << ")";
    return ostr.str();
}

std::string
RecLoadCoresStat::showCSV(const bool title) const
{
    std::ostringstream ostr;
    if (title) {
        ostr << "date,time,tv_sec,tv_usec,";
        for (size_t coreId = 0; coreId < mCoresStat.size(); ++coreId) {
            ostr << "coreId," << mCoresStat[coreId].showCSV(title);
            if (coreId < mCoresStat.size() - 1) ostr << ',';
        }
    } else {
        ostr << showTimestampDate() << ',' << showTimestampTime() << ','
             << mTimestamp.tv_sec << ',' << mTimestamp.tv_usec << ',';
        for (size_t coreId = 0; coreId < mCoresStat.size(); ++coreId) {
            ostr << coreId << ',' << mCoresStat[coreId].showCSV(title);
            if (coreId < mCoresStat.size() - 1) ostr << ',';
        }
    }
    return ostr.str();
}

// static function
std::string
RecLoadCoresStat::to_string(const struct timeval &timeval)
{
    static char *mon[] = {"Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"};
    static char *wday[] = {"Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"};

    struct tm *time_st = localtime(&timeval.tv_sec);

    std::ostringstream ostr;
    ostr << time_st->tm_year + 1900 << "/"
         << mon[time_st->tm_mon] << "/"
         << time_st->tm_mday << " "
         << wday[time_st->tm_wday] << " "
         << time_st->tm_hour << ":"
         << time_st->tm_min << ":"
         << time_st->tm_sec << ":"
         << timeval.tv_usec;
    return ostr.str();
}

std::string
RecLoadCoresStat::showTimestampDate() const
{
    static char *mon[] = {"Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"};

    struct tm *time_st = localtime(&mTimestamp.tv_sec);

    std::ostringstream ostr;
    ostr << mon[time_st->tm_mon] << '/'
         << time_st->tm_mday << '/'
         << time_st->tm_year + 1900;
    return ostr.str();
}

std::string
RecLoadCoresStat::showTimestampTime() const
{
    struct tm *time_st = localtime(&mTimestamp.tv_sec);

    std::ostringstream ostr;
    ostr << time_st->tm_hour << ":"
         << time_st->tm_min << ":"
         << time_st->tm_sec;
    return ostr.str();
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------

std::string
RecLoadHostTimelog::showLog(const std::string &hd) const
{
    std::ostringstream ostr;
    ostr << hd << "log (total:" << mLog.size() << ") {\n";
    for (size_t i = 0; i < mLog.size(); ++i) {
        ostr << hd << "  i:" << i << " {\n"; {
            ostr << mLog[i].show(hd + "    ") << '\n';
        }        
        ostr << hd << "  }\n";
    }
    ostr << hd << "}";
    return ostr.str();
}

std::string
RecLoadHostTimelog::showAverageLog(const std::string &hd) const
{
    std::ostringstream ostr;
    ostr << hd << "log (total:" << mLog.size() << ") {\n";
    for (size_t i = 0; i < mLog.size(); ++i) {
        ostr << hd << "  i:" << std::setw(4) << i << " " << mLog[i].showAverage(hd + "    ") << '\n';
    }
    ostr << hd << "}";
    return ostr.str();
}

std::string
RecLoadHostTimelog::showLastLog(const std::string &hd) const
{
    return mLog.back().show(hd);
}

std::string
RecLoadHostTimelog::showLogCSV() const
{
    if (!mLog.size()) return std::string("");

    std::ostringstream ostr;
    ostr << "id," << mLog[0].showCSV(true) << '\n'; // output title
        
    for (size_t i = 0; i < mLog.size(); ++i) {
        ostr << i << ',' << mLog[i].showCSV(false) << '\n';
    }
    return ostr.str();
}

bool
RecLoadHostTimelog::readCSV(std::ifstream &ifs)
{
    std::string oneline;

    //
    // get title info
    //
    if (!std::getline(ifs, oneline)) return false; // could not get title line
    std::vector<std::string> titles;
    size_t coreTotal = 0;
    {
        std::replace(oneline.begin(), oneline.end(), ',', ' ');
        std::istringstream istr(oneline);
        while (!istr.eof()) {
            std::string token;
            istr >> token;
            titles.push_back(token);
            if (token == "coreId") coreTotal++;
        }
    }

    //
    // get value info
    //
    size_t maxCores = coreTotal;

    std::vector<float> userArray;
    std::vector<float> niceArray;
    std::vector<float> sysArray;
    std::vector<float> idleArray;
    userArray.resize(maxCores);
    niceArray.resize(maxCores);
    sysArray.resize(maxCores);
    idleArray.resize(maxCores);

    while (std::getline(ifs, oneline)) {
        std::replace(oneline.begin(), oneline.end(), ',', ' ');
        std::istringstream istr(oneline);
        int itemId = 0;
        long tv_sec, tv_usec;
        int coreId;
        while (!istr.eof()) {
            if (titles[itemId] == "tv_sec") {
                istr >> tv_sec;
            } else if (titles[itemId] == "tv_usec") {
                istr >> tv_usec;
            } else if (titles[itemId] == "coreId") {
                istr >> coreId;
            } else if (titles[itemId] == "user") {
                istr >> userArray[coreId];
            } else if (titles[itemId] == "nice") {
                istr >> niceArray[coreId];
            } else if (titles[itemId] == "sys") {
                istr >> sysArray[coreId];
            } else if (titles[itemId] == "idle") {
                istr >> idleArray[coreId];
            } else {
                std::string token;
                istr >> token;
            }
            itemId++;
        }

        mLog.emplace_back(RecLoadCoresStat()); // create log for this timestamp
        RecLoadCoresStat &log = mLog.back();
        log.setTimestamp(tv_sec, tv_usec); // set timestamp info
        log.resize(maxCores);              // craete all cores information
        for (size_t id = 0; id < maxCores; ++id) {
            RecLoadCoreStat &currCore = log.get(id);
            currCore.set(userArray[id], niceArray[id], sysArray[id], idleArray[id]); // all cores stat
        }
        log.computeAverage();
    }

    return true;
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------

RecLoad::RecLoad(const float intervalSec, const float logSizeMaxMB, const std::string &outname) :
    mIntervalSec(intervalSec),
    mLogSizeMaxMB(logSizeMaxMB),
    mOutname(outname),
    mLogCondition(false),
    mStartEndDurationSec(0.0f)
{
    if (!getCoresSpec()) {
        throw "RecLoad : internal work memory allocation failed";
    }

    mCoresCurrent.resize(mCoresSpec.size());
    mCoresPrevious.resize(mCoresSpec.size());

    if (!getCoresStat(mCoresPrevious)) {
        throw "RecLoad : could not get initial cores status";
    }

    try {
        mWatchDogThread = std::thread([](RecLoad *recLoad) { recLoad->watchDogMainLoop(); }, this); // spawn watchDog thread
        mWatchDogThread.detach();
    }
    catch (...) {
        throw "Could not boot watchDog thread";
    }
}

RecLoad::~RecLoad()
{
    outputLogsCSV();
}

void
RecLoad::startLog()
{
    if (mLogCondition) return;

    mRecTime.start();

    mLog.clear();
    mLogCondition = true;
    std::cerr << "RecLoad::startLog() ..." << std::endl;
}

void
RecLoad::stopLog()
{
    if (!mLogCondition) return;

    std::cerr << "RecLoad::stopLog() ... start" << std::endl;
    mStartEndDurationSec = mRecTime.end();

    mLogCondition = false;

    if (mLog.size() > 1) {
        outputLogsCSV();
    }
    std::cerr << "RecLoad::stopLog() ... end" << std::endl;
}

std::string
RecLoad::showCoresSpec(const std::string &hd) const
{
    std::ostringstream ostr;
    ostr << hd << "RecLoad::mCoresSpec (total:" << mCoresSpec.size() << ") {\n";
    for (size_t coreId = 0; coreId < mCoresSpec.size(); ++coreId) {
        ostr << hd << "  coreId:" << coreId << " ";
        ostr << mCoresSpec[coreId].show(hd + "  ") << '\n';
    }
    ostr << hd << "}";
    return ostr.str();
}

// static function
std::string
RecLoad::showStartEndDurationCSV(const float duration)
{
    std::ostringstream ostr;
    ostr << "start-end(sec)," << duration << '\n';
    return ostr.str();
}

// static function
bool
RecLoad::readStartEndDurationCSV(std::ifstream &ifs, float &duration)
{
    std::string oneline;
    if (!std::getline(ifs, oneline)) return false; // could not get start/end duration info line

    std::replace(oneline.begin(), oneline.end(), ',', ' ');
    std::istringstream istr(oneline);
    std::string token;

    istr >> token;
    if (token != "start-end(sec)") return false; // not start/end duration info line

    istr >> duration;

    return true;
}

//------------------------------------------------------------------------------

bool
RecLoad::getCoresSpec()
//
// get all cores specification from /proc/cpuinfo and stored into mCoresSpec
//
{
    mCoresSpec.clear();

    std::ifstream ifs("/proc/cpuinfo");
    if (ifs.fail()) return false;

    std::string buff;
    for (size_t coreId = 0; ; ++coreId) {
        if (ifs.eof()) break;

        int procId = -1;
        std::string modelName;
        float mHz = 0.0f;
        int cacheSizeKB = 0;
        float bogoMips = 0.0f;
        
        while (std::getline(ifs, buff)) {
            if (!buff.size()) {
                if (mCoresSpec.size() != procId) return false; // unknown format : core order mismatch
                mCoresSpec.emplace_back(RecLoadCoreSpec(modelName, mHz, cacheSizeKB, bogoMips));
                break;
            }

            std::stringstream ss(buff);
            std::string token0, token1, token2;
            ss >> token0 >> token1;
            if (token0 == "processor") {
                ss >> procId;
            } else if (token0 == "model" && token1 == "name") {
                size_t id = buff.find(":") + 1;
                modelName = buff.substr(id, buff.size() - id);
            } else if (token0 == "cpu" && token1 == "MHz") {
                ss >> token2 >> mHz;
            } else if (token0 == "cache" && token1 == "size") {
                ss >> token2 >> cacheSizeKB;
            } else if (token0 == "bogomips") {
                ss >> bogoMips;
            }
        }
    }

    return true;
}

void
RecLoad::watchDogMainLoop()
{
    const size_t oneLogSize = sizeof(RecLoadCoresStat);
    size_t totalLogSize = 0; 
    int maxLogSize = std::max(0, (int)(mLogSizeMaxMB * 1024.0f * 1024.0f) - (int)oneLogSize);
    
    std::cerr << "RecLoad::watchDog thread start ... maxLogSize:" << mLogSizeMaxMB << " MByte" << std::endl;
    while (1) {
        if (!mLogCondition) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10)); // 10 ms
            continue;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds((int)(mIntervalSec * 1000.0f)));

        std::vector<RecLoadCoresStat> &log = mLog.getLog();

        log.emplace_back(RecLoadCoresStat());
        if (!getCoresLoad(log.back())) {
            std::cerr << "RecLoad::Could not get cores load -> stop logging" << std::endl;
            break;
        }

        totalLogSize += oneLogSize;

        if (totalLogSize > (size_t)maxLogSize) {
            std::cerr << "RecLoad::exceed max internal log data size. " << mLogSizeMaxMB << "MB" << std::endl;
            break;
        }
    }
    std::cerr << "RecLoad::watchDog thread done" << std::endl;
}

bool
RecLoad::getCoresLoad(RecLoadCoresStat &cores)
{
    if (!getCoresStat(mCoresCurrent)) {
        return false;
    }

    //
    // loop over each cores
    //
    cores.resize(mCoresCurrent.size());
    for (size_t coreId = 0; coreId < mCoresCurrent.size(); ++coreId) {
        RecLoadCoreStat &currCore = cores.get(coreId);

        //
        // compute delta then normalize
        //
        currCore = mCoresCurrent[coreId] - mCoresPrevious[coreId];
        float total = std::max(currCore.getTotal(), 1.0f); // minimum = 1 tick 

        currCore *= 1.0f / total;
        currCore.rangeClip(0.0f, 1.0f); // 0.0 ~ 1.0

        mCoresPrevious[coreId] = mCoresCurrent[coreId];
    }

    cores.computeAverage();     // compute core average
    cores.setTimestamp();       // set timestamp

    return true;
}

// static function    
bool
RecLoad::getCoresStat(std::vector<RecLoadCoreStat> &coresStat)
//
// get cores current status from /proc/stat
//
{
    std::ifstream ifs("/proc/stat");
    if (ifs.fail()) return false; // error

    std::string buff;
    size_t coreId = 0;
    while (std::getline(ifs, buff)) {
        std::stringstream ss(buff);
        std::string token;
        ss >> token;

        if (token.size() < 4) continue;

        if (token.substr(0, 3) == "cpu") {
            unsigned long user, nice, sys, idle;
            ss >> user >> nice >> sys >> idle;

            coresStat[coreId++].set(static_cast<float>(user),
                                    static_cast<float>(nice),
                                    static_cast<float>(sys),
                                    static_cast<float>(idle));
        }
    }
    return true;
}

std::string
RecLoad::showCoresStat(const std::string &hd, std::vector<RecLoadCoreStat> &cores) const
{
    std::ostringstream ostr;
    ostr << hd << "coresStat (total:" << cores.size() << ") {\n";
    for (size_t coreId = 0; coreId < cores.size(); ++coreId) {
        ostr << hd << "  coreId:" << coreId << " " << cores[coreId].show("") << '\n';
    }
    ostr << hd << "}";
    return ostr.str();
}

void
RecLoad::outputLogs()
{
    std::ofstream ofs;
    ofs.open(mOutname, std::ios::out);
    if (!ofs) return;           // could not open file -> skip

    ofs << showAllLogs("");
    ofs.close();
}

void
RecLoad::outputLogsCSV()
{
    std::cerr << "RecLoad::===>>> outputLogsCSV filename:" << mOutname << " <<<===" << std::endl;

    std::ofstream ofs;
    ofs.open(mOutname, std::ios::out);
    if (!ofs) {
        std::cerr << "RecLoad:: ERROR : Could not create CSV file:" << mOutname << std::endl;
        return;           // coult not open file -> skip
    }
    
    ofs << RecLoad::showStartEndDurationCSV(mStartEndDurationSec);
    ofs << showAllLogsCSV();
    ofs.close();
}

} // namespace rec_load
} // namespace moonray

