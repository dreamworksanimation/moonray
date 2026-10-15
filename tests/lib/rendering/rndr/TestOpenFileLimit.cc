// Copyright 2026 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#include "TestOpenFileLimit.h"
#include <moonray/rendering/rndr/OpenFileLimit.h>

#include <fcntl.h>
#include <unistd.h>
#include <vector>

namespace moonray {
namespace rndr {
namespace unittest {

void
TestOpenFileLimit::setUp()
{
    CPPUNIT_ASSERT(getrlimit(RLIMIT_NOFILE, &mSavedLimit) == 0);
}

void
TestOpenFileLimit::tearDown()
{
    // Only the soft limit is changed by these tests, so this always succeeds.
    setrlimit(RLIMIT_NOFILE, &mSavedLimit);
}

void
TestOpenFileLimit::testFitTextureFileHandles()
{
    // Fits, including exactly at the limit.
    CPPUNIT_ASSERT_EQUAL(24000, fitTextureFileHandles(24000, 100000));
    CPPUNIT_ASSERT_EQUAL(24000, fitTextureFileHandles(24000, 24000 + sNonTextureFileHandles));

    // No limit.
    CPPUNIT_ASSERT_EQUAL(24000, fitTextureFileHandles(24000, RLIM_INFINITY));

    // Zero and negative counts are passed through untouched.
    CPPUNIT_ASSERT_EQUAL(0, fitTextureFileHandles(0, 256));
    CPPUNIT_ASSERT_EQUAL(-1, fitTextureFileHandles(-1, 256));

    // Too low: keep sNonTextureFileHandles in reserve...
    CPPUNIT_ASSERT_EQUAL(24000 - 1, fitTextureFileHandles(24000, 24000 + sNonTextureFileHandles - 1));
    CPPUNIT_ASSERT_EQUAL(4096 - sNonTextureFileHandles, fitTextureFileHandles(24000, 4096));
    CPPUNIT_ASSERT_EQUAL(245760 - sNonTextureFileHandles, fitTextureFileHandles(500000, 245760));

    // ...or half the limit, whichever is larger.
    CPPUNIT_ASSERT_EQUAL(150, fitTextureFileHandles(24000, 300));
    CPPUNIT_ASSERT_EQUAL(128, fitTextureFileHandles(24000, 256));
    CPPUNIT_ASSERT_EQUAL(0, fitTextureFileHandles(24000, 1));
}

void
TestOpenFileLimit::testRaiseOpenFileLimitToMax()
{
    // Lower the soft limit, as the macOS default does.
    struct rlimit limit = mSavedLimit;
    limit.rlim_cur = 64;
    CPPUNIT_ASSERT(setrlimit(RLIMIT_NOFILE, &limit) == 0);
    CPPUNIT_ASSERT_EQUAL(rlim_t(64), getOpenFileLimit());

    CPPUNIT_ASSERT(raiseOpenFileLimitToMax());

    // The soft limit is now the hard limit, or the kernel's per-process
    // maximum if that is lower.
    CPPUNIT_ASSERT(getrlimit(RLIMIT_NOFILE, &limit) == 0);
    CPPUNIT_ASSERT(limit.rlim_cur > 64);
    CPPUNIT_ASSERT(limit.rlim_cur <= limit.rlim_max);
    const rlim_t openFileLimit = getOpenFileLimit();
    CPPUNIT_ASSERT(openFileLimit > 64);
    CPPUNIT_ASSERT(openFileLimit <= limit.rlim_cur);

    // Calling it again changes nothing.
    CPPUNIT_ASSERT(raiseOpenFileLimitToMax());
    struct rlimit again;
    CPPUNIT_ASSERT(getrlimit(RLIMIT_NOFILE, &again) == 0);
    CPPUNIT_ASSERT_EQUAL(limit.rlim_cur, again.rlim_cur);

    // The raised limit is really usable: open more descriptors than the old
    // soft limit allowed (unless the hard limit is too low to try).
    if (openFileLimit < 512) {
        return;
    }
    std::vector<int> fds;
    for (int i = 0; i < 256; ++i) {
        const int fd = open("/dev/null", O_RDONLY);
        if (fd < 0) {
            break;
        }
        fds.push_back(fd);
    }
    const size_t opened = fds.size();
    for (const int fd : fds) {
        close(fd);
    }
    CPPUNIT_ASSERT_EQUAL(size_t(256), opened);
}

} // namespace unittest
} // namespace rndr
} // namespace moonray

