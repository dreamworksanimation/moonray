// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#include "Statistics.h"
#include <scene_rdl2/common/platform/HybridUniformData.h>

namespace ispc {
extern "C" uint32_t PbrStatistics_hudValidation(bool);
}

namespace moonray {
namespace pbr {

HUD_VALIDATOR( PbrStatistics );

} // namespace pbr
} // namespace moonray


