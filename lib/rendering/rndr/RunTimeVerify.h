// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

//
//
#pragma once

//
// All of the following directives should be disabled for the release version of moonray.
// They are pretty useful for debugging purposes but have huge performance penalties.
//

// Enable runtime verification regarding TileGroup is properly dispatched to the MCRT threads
// by tileWorkQueue
//#define RUNTIME_VERIFY_TILE_WORK_QUEUE

// Enable runtime accumulation of sample count for each pixel for debugging purposes and show result.
//#define RUNTIME_VERIFY_PIX_SAMPLE_COUNT

// Enable runtime sampling span verification for all pixels executed by multiple MCRT threads
//#define RUNTIME_VERIFY_PIX_SAMPLE_SPAN

