// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

//
//
#include "TestCheckpoint.h"

#include <moonray/rendering/rndr/RenderDriver.h>

namespace moonray {
namespace rndr {
namespace unittest {

void
TestCheckpoint::testKJSequenceTable()
{
    constexpr unsigned maxSampleId = 65535;
    CPPUNIT_ASSERT(RenderDriver::verifyKJSequenceTable(maxSampleId,
                                                       nullptr)); // tbl as string
}

void
TestCheckpoint::testTotalCheckpointToQualitySteps()
{
    //
    // Verification for uniform sampling is potentially faster than adaptive.
    // So we try to test up to 65536 SPP for uniform and 16384 SPP for adaptive sampling here.
    // Typical production usage only focused on the small number of checkpoint files like less
    // than 32 and start SPP control is also used as a very limited pattern like 0 or 1 right now.
    // So we are not testing all possible variations at this moment because it's too costly.
    //

    // This test is done by multi-thread and requires around 8sec on pearldiva.
    CPPUNIT_ASSERT(RenderDriver::verifyTotalCheckpointToQualityStepsExhaust
                   (moonray::rndr::SamplingMode::UNIFORM,
                    65536,         // maxSPP
                    64,            // fileCountEndCap
                    64,            // startSPPEndCap
                    false,         // liveMessage
                    false,         // deepVerifyMessage
                    nullptr,       // verify result message
                    false,         // multiThreadA
                    true));        // multiThreadB

    // This test is done by multi-thread and requires around 19sec on pearldiva.
    CPPUNIT_ASSERT(RenderDriver::verifyTotalCheckpointToQualityStepsExhaust
                   (moonray::rndr::SamplingMode::ADAPTIVE,
                    16384,         // maxSPP
                    16,            // fileCountEndCap
                    16,            // startSPPEndCap
                    false,         // liveMessage
                    false,         // deepVerifyMessage
                    nullptr,       // verify result message
                    false,         // multiThreadA
                    true));        // multiThreadB
}

} // namespace unittest
} // namespace rndr
} // namespace moonray

