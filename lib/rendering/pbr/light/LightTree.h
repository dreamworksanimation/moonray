// Copyright 2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Light.h"
#include "LightTreeUtil.h"
#include "LightTree.hh"

namespace moonray {
namespace pbr {

// --------------------------------------------------- LightTree -------------------------------------------------------
/// A LightTree is an acceleration structure for light sampling. It is composed of Nodes, which are basically clusters 
/// of lights, grouped strategically. To avoid making any changes to the given array of lights, we instead have an array 
/// of light indices, mLightIndices. These indices will be partitioned such that each Node's lights are contiguous.

class LightTree
{
public:
    /// Constructor
    LightTree(float sceneDiameter, float samplingThreshold);

    /// Destructor
    ~LightTree() {};

    // HUD validation and type casting
    static uint32_t hudValidation(bool verbose) {
        LIGHT_TREE_VALIDATION;
    }

    /// Build the acceleration structure. 
    /// @see (Section 4.2) from "Importance Sampling of Many Lights..." (Conty, Kulla)
    void build(const Light* const* boundedLights, unsigned int boundedLightCount,
               const Light* const* unboundedLights, unsigned int unboundedLightCount);

    /// Chooses light(s) using importance sampling and adaptive tree splitting 
    /// (Section 5.4) from "Importance Sampling of Many Lights..." (Conty, Kulla)
    void sample() const;

    /// Sets the scene diameter (size of the scene bvh's bounding box)
    void setSceneDiameter(float sceneDiameter) { mSceneDiameter = sceneDiameter; }

    /// Sets the sampling threshold (which determines the amount of adaptive tree splitting)
    void setSamplingThreshold(float threshold) { mSamplingThreshold = threshold; }

    /// Print the tree
    void print() const;

// ------------------------------------ Member Variables ---------------------------------------------------------------
    LIGHT_TREE_MEMBERS;
    std::vector<Node> mNodes = {};                  // array of nodes 
    std::vector<uint> mLightIndices = {};           // array of light indices -- allows us to change the "order" of 
                                                    // lights in the light tree without mutating the lightset itself
};

} // end namespace pbr
} // end namespace moonray