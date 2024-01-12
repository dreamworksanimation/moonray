// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#include "TestFormatters.h"

#include <moonray/statistics/Formatters.h>
#include <cppunit/extensions/HelperMacros.h>

#include <cstring>

namespace moonray_stats {
namespace unittest {

void
TestFormatters::setUp()
{
}

void
TestFormatters::tearDown()
{
}

void
TestFormatters::testBytes()
{
    Bytes b0(4220534454LL);
    Bytes b1(2784345248LL);
    Bytes b2(0);
    Bytes b3(1024);
    Bytes b4(1024 * 1024);

    CPPUNIT_ASSERT_DOUBLES_EQUAL(3.930679013953, b0.autoConvert(), 0.0001);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(2.593123584986, b1.autoConvert(), 0.0001);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0, b2.autoConvert(), 0.0001);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1, b3.autoConvert(), 0.0001);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1, b4.autoConvert(), 0.0001);
    CPPUNIT_ASSERT(std::strncmp("GB", b0.getAutoUnit(), 2) == 0);
    CPPUNIT_ASSERT(std::strncmp("GB", b1.getAutoUnit(), 2) == 0);
    CPPUNIT_ASSERT(std::strncmp("B",  b2.getAutoUnit(), 1) == 0);
    CPPUNIT_ASSERT(std::strncmp("KB", b3.getAutoUnit(), 2) == 0);
    CPPUNIT_ASSERT(std::strncmp("MB", b4.getAutoUnit(), 2) == 0);
}

void
TestFormatters::testTime()
{
    Time t0(0.0f);
    Time t1(1.0f);
    Time t2(30.0f);
    Time t3(30.5f);
    Time t4(60.0f);
    Time t5(90.0f);
    Time t6(60.0f * 60.0f);
    Time t7(60.0f * 60.0f + 90.0f);

    std::ostringstream outs;
    outs.precision(2);

    t0.write(outs, FormatterHuman());
    CPPUNIT_ASSERT_EQUAL(std::string("00:00:00.00"), outs.str());
    outs.str("");

    t1.write(outs, FormatterHuman());
    CPPUNIT_ASSERT_EQUAL(std::string("00:00:01.00"), outs.str());
    outs.str("");

    t2.write(outs, FormatterHuman());
    CPPUNIT_ASSERT_EQUAL(std::string("00:00:30.00"), outs.str());
    outs.str("");

    t3.write(outs, FormatterHuman());
    CPPUNIT_ASSERT_EQUAL(std::string("00:00:30.50"), outs.str());
    outs.str("");

    t4.write(outs, FormatterHuman());
    CPPUNIT_ASSERT_EQUAL(std::string("00:01:00.00"), outs.str());
    outs.str("");

    t5.write(outs, FormatterHuman());
    CPPUNIT_ASSERT_EQUAL(std::string("00:01:30.00"), outs.str());
    outs.str("");

    t6.write(outs, FormatterHuman());
    CPPUNIT_ASSERT_EQUAL(std::string("01:00:00.00"), outs.str());
    outs.str("");

    t7.write(outs, FormatterHuman());
    CPPUNIT_ASSERT_EQUAL(std::string("01:01:30.00"), outs.str());
    outs.str("");
}

} // namespace unittest
} // namespace moonray_stats

