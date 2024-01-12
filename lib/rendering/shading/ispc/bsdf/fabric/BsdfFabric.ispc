// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file BsdfFabric.ispc

#include "BsdfFabric.isph"
#include "BsdfFabricVelvet.isph"

#include <scene_rdl2/common/platform/IspcUtil.isph>

#pragma ignore warning(all)
ISPC_UTIL_EXPORT_STRUCT_TO_HEADER(FabricBsdfLobe);
#pragma ignore warning(all)
ISPC_UTIL_EXPORT_STRUCT_TO_HEADER(FabricVelvetBsdfLobe);

