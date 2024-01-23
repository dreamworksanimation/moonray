// Copyright 2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#include "LightTree.isph"
#include <scene_rdl2/common/platform/IspcUtil.isph>

// =====================================================================================================================
// References:
// =====================================================================================================================
// [1] Alejandro Conty Estevez and Christopher Kulla. 2018. 
//     "Importance Sampling of Many Lights with Adaptive Tree Splitting"
// =====================================================================================================================


export uniform uint32_t
LightTree_hudValidation(uniform bool verbose)
{
    LIGHT_TREE_VALIDATION;
}

// --------------------------------- SAMPLING METHODS --------------------------------------- //

void LightTree_sample(const uniform LightTree * const uniform lightTree) {}

// --------------------------------- PRINT FUNCTIONS -------------------------------------------

void LightTree_print(const uniform LightTree * const uniform lightTree) {}

