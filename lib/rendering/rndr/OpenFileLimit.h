// Copyright 2026 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <sys/resource.h>

#include <string>

namespace moonray {
namespace rndr {

// Open file limits at or below this are too low for production scenes. It is
// the default soft limit on most Linux distributions (macOS defaults to 256).
constexpr rlim_t sLowOpenFileLimit = 1024;

// Returns the number of files this process may have open at once: the soft
// RLIMIT_NOFILE limit, further bounded by kern.maxfilesperproc on macOS.
// Returns RLIM_INFINITY if there is no limit. MoonRay doesn't change this
// limit; it must be raised in the environment (e.g. "ulimit -n") before
// launching.
rlim_t getOpenFileLimit();

// Formats an open file limit for display ("unlimited" for RLIM_INFINITY).
std::string openFileLimitToString(rlim_t openFileLimit);

// Returns true if openFileLimit is low enough that geometry and textures may
// fail to load on large production scenes.
bool isOpenFileLimitLow(rlim_t openFileLimit);

// Returns true if fileCount is more than openFileLimit allows.
bool exceedsOpenFileLimit(int fileCount, rlim_t openFileLimit);

} // namespace rndr
} // namespace moonray

