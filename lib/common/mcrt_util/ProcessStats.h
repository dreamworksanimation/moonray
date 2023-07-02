// Copyright 2023 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

//

#pragma once

#include <scene_rdl2/common/platform/Platform.h>
#include <scene_rdl2/render/logging/logging.h>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <mutex>

namespace moonray {
namespace util {

struct ProcessUtilization {
    // stored in a system dependent (not seconds) unit
    int64 userTime;
    int64 systemTime;

    // get elapsed time (since start) in seconds
    int64 getUserSeconds(const ProcessUtilization &start) const;
    int64 getSystemSeconds(const ProcessUtilization &start) const;
};

class ProcessStats
{
public:
    ProcessStats();

    int64 getBytesRead() const;

    int64 getProcessMemory() const;

    ProcessUtilization getProcessUtilization() const;


private:

    mutable std::ifstream mStatmFile;
    mutable std::ifstream mIOStatFile;
    mutable std::ifstream mProcStatFile;

    // ifstream mutex to prevent corrupt reads
    // when we are getting log messages from
    // threaded sections of code
    mutable std::mutex mMemoryReadMutex;
    mutable std::mutex mReadIOMutex;
    mutable std::mutex mSystemUtilMutex;

};


}  //util;
}  //moonray


