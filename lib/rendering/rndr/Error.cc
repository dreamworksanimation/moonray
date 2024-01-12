// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

//

#include "Error.h"

#include <array>
#include <cerrno>
#include <cstring>

namespace moonray {
namespace rndr {

// Unfortunately we strerror isn't thread safe, 
// strerror_r is thread safe, however depending on the
// compiler/glibc we get a different version of strerror_r.
// 
// This is a copy/paste from the error handling code from 
// RenderStatistics.cc getExecutablePath()
const std::string
getErrorDescription()
{
    std::array<char, 1024> errbuf;
#if (_POSIX_C_SOURCE >= 200112L || _XOPEN_SOURCE >= 600) && ! _GNU_SOURCE
    std::string e("Unknown error");
    if (strerror_r(errno, errbuf.data(), errbuf.size()) >= 0) {
        e = errbuf.data();
    }

    return e;
#else
    // Oddly enough, this function may not use the errbuf data at all, and
    // if you use it instead of the return value, you may get garbage.
    return std::string(strerror_r(errno, errbuf.data(), errbuf.size()));
#endif
}

} // namespace rndr
} // namespace moonray

