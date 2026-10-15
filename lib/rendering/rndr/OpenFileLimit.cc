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

bool
raiseOpenFileLimitToMax()
{
    struct rlimit limit;
    if (getrlimit(RLIMIT_NOFILE, &limit) != 0) {
        return false;
    }
    const rlim_t target = std::min(limit.rlim_max, getKernelOpenFileMax());
    if (limit.rlim_cur != RLIM_INFINITY && limit.rlim_cur < target) {
        limit.rlim_cur = target;
        return setrlimit(RLIMIT_NOFILE, &limit) == 0;
    }
    return true;
}

int
fitTextureFileHandles(int textureFileHandles, rlim_t openFileLimit)
{
    if (textureFileHandles <= 0 || openFileLimit == RLIM_INFINITY) {
        return textureFileHandles;
    }
    const rlim_t needed = static_cast<rlim_t>(textureFileHandles) + sNonTextureFileHandles;
    if (openFileLimit >= needed) {
        return textureFileHandles;
    }
    // openFileLimit < needed, so these all fit in an int.
    const rlim_t reserved = static_cast<rlim_t>(sNonTextureFileHandles);
    const rlim_t fitted = openFileLimit > 2 * reserved ? openFileLimit - reserved
                                                       : openFileLimit / 2;
    return static_cast<int>(fitted);
}

} // namespace rndr
} // namespace moonray

