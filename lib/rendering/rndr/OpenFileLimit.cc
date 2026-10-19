// Copyright 2026 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#include "OpenFileLimit.h"

#include <algorithm>
#ifdef __APPLE__
#include <sys/sysctl.h>
#endif

namespace moonray {
namespace rndr {

namespace {

// setrlimit() on macOS accepts values above kern.maxfilesperproc, but the
// kernel won't hand out more descriptors than that.
rlim_t
getKernelOpenFileMax()
{
#ifdef __APPLE__
    int maxFilesPerProc = 0;
    size_t size = sizeof(maxFilesPerProc);
    if (sysctlbyname("kern.maxfilesperproc", &maxFilesPerProc, &size, nullptr, 0) == 0 &&
        maxFilesPerProc > 0) {
        return static_cast<rlim_t>(maxFilesPerProc);
    }
#endif
    return RLIM_INFINITY;
}

} // anonymous namespace

rlim_t
getOpenFileLimit()
{
    struct rlimit limit;
    if (getrlimit(RLIMIT_NOFILE, &limit) != 0) {
        return RLIM_INFINITY;
    }
    return std::min(limit.rlim_cur, getKernelOpenFileMax());
}

std::string
openFileLimitToString(rlim_t openFileLimit)
{
    return openFileLimit == RLIM_INFINITY ? "unlimited" : std::to_string(openFileLimit);
}

bool
isOpenFileLimitLow(rlim_t openFileLimit)
{
    return openFileLimit != RLIM_INFINITY && openFileLimit <= sLowOpenFileLimit;
}

bool
exceedsOpenFileLimit(int fileCount, rlim_t openFileLimit)
{
    return fileCount > 0 &&
           openFileLimit != RLIM_INFINITY &&
           static_cast<rlim_t>(fileCount) > openFileLimit;
}

} // namespace rndr
} // namespace moonray

