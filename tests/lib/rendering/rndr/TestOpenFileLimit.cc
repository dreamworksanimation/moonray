// Copyright 2026 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#include "TestOpenFileLimit.h"
#include <moonray/rendering/rndr/OpenFileLimit.h>

#include <string>

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
TestOpenFileLimit::testGetOpenFileLimit()
{
    // Reports the soft limit, as the macOS default would be.
    struct rlimit limit = mSavedLimit;
    limit.rlim_cur = 256;
    CPPUNIT_ASSERT(setrlimit(RLIMIT_NOFILE, &limit) == 0);
    CPPUNIT_ASSERT_EQUAL(rlim_t(256), getOpenFileLimit());

    // Never more than the hard limit.
    limit.rlim_cur = limit.rlim_max;
    if (setrlimit(RLIMIT_NOFILE, &limit) == 0) {
        CPPUNIT_ASSERT(getOpenFileLimit() <= limit.rlim_max);
        CPPUNIT_ASSERT(getOpenFileLimit() > 256);
    }
}

void
TestOpenFileLimit::testOpenFileLimitToString()
{
    CPPUNIT_ASSERT_EQUAL(std::string("256"), openFileLimitToString(256));
    CPPUNIT_ASSERT_EQUAL(std::string("245760"), openFileLimitToString(245760));
    CPPUNIT_ASSERT_EQUAL(std::string("unlimited"), openFileLimitToString(RLIM_INFINITY));
}

void
TestOpenFileLimit::testIsOpenFileLimitLow()
{
    CPPUNIT_ASSERT(isOpenFileLimitLow(256));    // macOS default
    CPPUNIT_ASSERT(isOpenFileLimitLow(1024));   // most Linux defaults
    CPPUNIT_ASSERT(!isOpenFileLimitLow(1025));
    CPPUNIT_ASSERT(!isOpenFileLimitLow(65536));
    CPPUNIT_ASSERT(!isOpenFileLimitLow(RLIM_INFINITY));
}

void
TestOpenFileLimit::testExceedsOpenFileLimit()
{
    CPPUNIT_ASSERT(exceedsOpenFileLimit(24000, 256));
    CPPUNIT_ASSERT(exceedsOpenFileLimit(24001, 24000));
    CPPUNIT_ASSERT(!exceedsOpenFileLimit(24000, 24000));
    CPPUNIT_ASSERT(!exceedsOpenFileLimit(24000, 65536));
    CPPUNIT_ASSERT(!exceedsOpenFileLimit(24000, RLIM_INFINITY));

    // Zero and negative counts never exceed the limit.
    CPPUNIT_ASSERT(!exceedsOpenFileLimit(0, 256));
    CPPUNIT_ASSERT(!exceedsOpenFileLimit(-1, 256));
}

} // namespace unittest
} // namespace rndr
} // namespace moonray

