// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file BsdfvToBsdf.h
#pragma once

#include "Bsdf.h"

namespace moonray {
namespace shading {

/// Convert one lane of a Bsdfv into a scalar Bsdf.
void BsdfvToBsdf(const Bsdfv *bsdfv, const int lane,
                 Bsdf *bsdf, alloc::Arena &arena);

Bsdf *BsdfvToBsdf(unsigned numBlocks, const Bsdfv *bsdfv,
                  unsigned numEntries, alloc::Arena *arena);

} // shading
} // moonray

