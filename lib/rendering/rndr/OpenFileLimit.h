// Copyright 2026 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <sys/resource.h>

namespace moonray {
namespace rndr {

// Number of file descriptors kept in reserve for everything other than
// texture files (geometry, output images, sockets, shared libraries, ...).
constexpr int sNonTextureFileHandles = 256;

// Returns the number of files this process may have open at once: the soft
// RLIMIT_NOFILE limit, further bounded by kern.maxfilesperproc on macOS.
// Returns RLIM_INFINITY if there is no limit.
rlim_t getOpenFileLimit();

// Raises the soft RLIMIT_NOFILE limit to the hard limit (bounded by
// kern.maxfilesperproc on macOS). This changes process-wide state, so it is
// meant to be called by applications, not by library code running inside
// someone else's process. Returns false if the limit could not be raised.
bool raiseOpenFileLimitToMax();

// Returns the number of texture file handles that fit within openFileLimit
// while keeping sNonTextureFileHandles in reserve. textureFileHandles is
// returned unchanged if it already fits.
int fitTextureFileHandles(int textureFileHandles, rlim_t openFileLimit);

} // namespace rndr
} // namespace moonray

