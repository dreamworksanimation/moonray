// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

//

#include "ProcessStats.h"

#include <unistd.h>

namespace moonray {

namespace util {

int64
ProcessUtilization::getUserSeconds(const ProcessUtilization &start) const
{
    return (this->userTime - start.userTime) / sysconf(_SC_CLK_TCK);
}

int64
ProcessUtilization::getSystemSeconds(const ProcessUtilization &start) const
{
    return (this->systemTime - start.systemTime) / sysconf(_SC_CLK_TCK);
}

ProcessStats::ProcessStats()
{
    mStatmFile.open("/proc/self/statm");
    mIOStatFile.open("/proc/self/io");
    mProcStatFile.open("/proc/self/stat");
}


int64
ProcessStats::getBytesRead() const
{
    std::string inStr;
    int64  bytesRead = 0;

    mReadIOMutex.lock();
    if(mIOStatFile.good()) {
        mIOStatFile >> inStr >> bytesRead;
        //reset file pointer for later reads
        mIOStatFile.seekg(std::ios_base::beg);
    } else {
        //  sometimes we lose the file handle so
        //  close it and try to reopen for the next query.
        mIOStatFile.close();
        mIOStatFile.open("/proc/self/io");
    }
    mReadIOMutex.unlock();
    return bytesRead;

}

int64
ProcessStats::getProcessMemory() const
{
    int64 currentMemoryUsage = 0;

    mMemoryReadMutex.lock();
    if(mStatmFile.good()) {
        int64 memorySize = 0;
        int64 memoryResident = 0;
        int64 memoryShared = 0;
        //sometimes the OS is writing to the file
        try {
            mStatmFile >> memorySize >> memoryResident >> memoryShared;
        } catch( std::ios_base::failure f ) {
            scene_rdl2::logging::Logger::error(f.what());
        }
        currentMemoryUsage = memoryResident * sysconf(_SC_PAGE_SIZE);
        //reset file pointer for later reads
        mStatmFile.seekg(std::ios_base::beg);
    } else {
        //  sometimes we lose the file handle so
        //  close it and try to reopen for the next query.
        mStatmFile.close();
        mStatmFile.open("/proc/self/statm");
    }
    mMemoryReadMutex.unlock();
    return currentMemoryUsage;
}


ProcessUtilization
ProcessStats::getProcessUtilization() const
{

    ProcessUtilization result;
    std::string inStr;

    mSystemUtilMutex.lock();
    if(mProcStatFile.good()) {
        mProcStatFile >> inStr;  //1
        mProcStatFile >> inStr;  //2
        mProcStatFile >> inStr;  //3
        mProcStatFile >> inStr;  //4
        mProcStatFile >> inStr;  //5
        mProcStatFile >> inStr;  //6
        mProcStatFile >> inStr;  //7
        mProcStatFile >> inStr;  //8
        mProcStatFile >> inStr;  //9
        mProcStatFile >> inStr;  //10
        mProcStatFile >> inStr;  //11
        mProcStatFile >> inStr;  //12
        mProcStatFile >> inStr;  //13
        mProcStatFile >> result.userTime;  //14
        mProcStatFile >> result.systemTime;  //15

        //reset file pointer for later reads
        mProcStatFile.seekg(std::ios_base::beg);
    } else {
        //  sometimes we lose the file handle so
        //  close it and try to reopen for the next query.
        mProcStatFile.close();
        mProcStatFile.open("/proc/self/stat");
    }
    mSystemUtilMutex.unlock();
    return result;

}


} //util
} //moonray


