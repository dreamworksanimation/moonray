// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file MipSelector.hh

#pragma once

// See the TextureOpt::conservative_filter flag - true means over-blur rather than alias.
#define MIP_FILTER_MIN          1       // Slow since it'll tend to pick higher mips, but most comparable to OIIO EWA scheme.
#define MIP_FILTER_MAX          2       // Closest to OIIO_CONSERVATIVE for testing. Not a great choice though.
#define MIP_FILTER_AVG          3       // Good general purpose choice. Take the average of *square* lengths.
#define MIP_FILTER_AREA         4       // Good general purpose choice. Take the area of the parallelogram.
                                        // More accurate than MIP_FILTER_AVG.

#define START_OIIO_FILTER       6

#define OIIO_CONSERVATIVE       6       // opts->mipmode = OIIO::TextureOpt::MipModeTrilinear, conservative_filter = true
#define OIIO_NON_CONSERVATIVE   7       // opts->mipmode = OIIO::TextureOpt::MipModeTrilinear, conservative_filter = false

#define DEFAULT_MIP_FILTER      MIP_FILTER_AREA


