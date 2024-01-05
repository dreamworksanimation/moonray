// Copyright 2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#include "LightTree.h"
#include <moonray/rendering/pbr/light/LightTree_ispc_stubs.h>

namespace moonray {
namespace pbr {

// ----------------------------- BUILD METHODS ---------------------------------------------------------------------- //

LightTree::LightTree(float sceneDiameter, float samplingThreshold) 
    : mSceneDiameter(sceneDiameter), mSamplingThreshold(samplingThreshold) 
{}

void LightTree::build(const Light* const* boundedLights, unsigned int boundedLightCount,
                      const Light* const* unboundedLights, unsigned int unboundedLightCount) 
{
    mBoundedLights = boundedLights;
    mUnboundedLights = unboundedLights;
    mBoundedLightCount = boundedLightCount;
    mUnboundedLightCount = unboundedLightCount;
}

// --------------------------------- SAMPLING METHODS --------------------------------------------------------------- //

void LightTree::sample() const {}

// --------------------------------- PRINT FUNCTIONS ---------------------------------------------------------------- //

void LightTree::print() const {}

} // end namespace pbr
} // end namespace moonray