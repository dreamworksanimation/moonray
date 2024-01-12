// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#include "test_time.h"
#include <moonray/common/time/Ticker.h>


#include <functional>
#include <typeinfo>

CPPUNIT_TEST_SUITE_REGISTRATION(TestCommonTime);

using namespace moonray::time;


void TestCommonTime::testTimer()
{
    int64 time;
    Ticker<int64> ticker(time);

    ticker.start();
    ticker.lap();
    ticker.stop();
}


