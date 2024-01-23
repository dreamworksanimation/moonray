// Copyright 2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Light.h"
#include "LightTreeUtil.h"
#include "LightTree.hh"

namespace moonray {
namespace pbr {

// =====================================================================================================================
// References:
// =====================================================================================================================
// [1] Alejandro Conty Estevez and Christopher Kulla. 2018. 
//     "Importance Sampling of Many Lights with Adaptive Tree Splitting"
// =====================================================================================================================


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
    /// @see [1] section 4.2
    void build(const Light* const* boundedLights, unsigned int boundedLightCount,
               const Light* const* unboundedLights, unsigned int unboundedLightCount);

    /// Chooses light(s) using importance sampling and adaptive tree splitting 
    /// @see [1] section 5.4
    void sample() const;

    /// Sets the scene diameter (size of the scene bvh's bounding box)
    void setSceneDiameter(float sceneDiameter) { mSceneDiameter = sceneDiameter; }

    /// Sets the sampling threshold (which determines the amount of adaptive tree splitting)
    void setSamplingThreshold(float threshold) { mSamplingThreshold = threshold; }

    /// Print the tree
    void print() const;

private:

/// ----------------------------- Inline Helpers -----------------------------------------

    // Returns a new list of buckets and splits with empty buckets/splits removed,
    // as well as the number of splits that remain
    inline int purgeEmptyBuckets(const LightTreeBucket* const oldBuckets, const SplitCandidate* const oldSplits,
                                 LightTreeBucket* newBuckets, SplitCandidate* newSplits, int oldBucketCount) const

    {
        int newBucketCount = 0;
        int oldSplitCount = oldBucketCount - 1;

        for (int i = 0; i < oldBucketCount; ++i) {
            if (!oldBuckets[i].mBBox.empty()) {
                newBuckets[newBucketCount] = oldBuckets[i];
                if (i < oldSplitCount) {
                    newSplits[newBucketCount] = oldSplits[i];
                }
                ++newBucketCount;
            }
        }
        return newBucketCount - 1;
    }

    // the following assumes buckets and splits are set up like this:
    // |   b0   |   b1   |   b2   |   ...   |   bn   |
    //          s0       s1       s2       s(n-1)

    // Populates the left side of each split
    inline void populateSplitsLeftSide(SplitCandidate* splits, const LightTreeBucket* buckets, int splitCount) const
    {
        for (int i = 0; i < splitCount; ++i) {
            if (i == 0) {
                // the very first split only contains the bucket to the left
                splits[0].setLeftSide(buckets[0]);
            } else {
                // split contains bucket directly to the left,
                // and everything in the split directly to the left
                splits[i].setLeftSide(splits[i - 1], buckets[i]);
            }
        }
    }
    // Populates the right side of each split
    inline void populateSplitsRightSide(SplitCandidate* splits, const LightTreeBucket* buckets, int splitCount) const
    {
        int lastSplitIndex = splitCount - 1;
        for (int i = lastSplitIndex; i >= 0; --i) {
            if (i == lastSplitIndex) {
                // the last split only contains the bucket to the right
                splits[lastSplitIndex].setRightSide(buckets[lastSplitIndex + 1]);
            } else {
                // split contains bucket directly to the right,
                // and everything in the split directly to the right
                splits[i].setRightSide(splits[i + 1], buckets[i + 1]);
            }
        }
    }

    // Finds the lowest-cost split of the node along the given axis
    inline float getCheapestSplit(int splitCount, const SplitCandidate* const splits, const LightTreeNode& node, 
                                  SplitCandidate& minSplit) const
    {
        float minCost = std::numeric_limits<float>::max();
        for (int i = 0; i < splitCount; ++i) {
            const SplitCandidate& split = splits[i];
            // make sure neither node would be empty
            if (split.leftIsEmpty() || split.rightIsEmpty()) {
                continue;
            }
            float cost = split.cost(node.getBBox(), node.getCone());
            if (cost < minCost) {
                minCost = cost;
                minSplit = split;
            }
        }
        return minCost;
    }

/// ------------------------------------- Function Declarations ---------------------------------------

    /// Recursively build tree
    void buildRecurse(uint nodeIndex);


    /// Create a tree split. This involves initializing SplitCandidate objects, which are possibilities of tree splits. 
    /// Each SplitCandidate contains an axis and an associated cost. We choose (and initialize) the SplitCandidate
    /// with the lowest cost.
    ///
    /// INPUTS:
    ///     @param nodeIndex The node we are trying to split
    ///
    /// OUTPUTS:
    ///     @param leftNode The new left node resulting from the split
    ///     @param rightNode The new right node resulting from the split
    ///
    float split(uint nodeIndex, LightTreeNode& leftNode, LightTreeNode& rightNode);


    /// Splits the axis into the specified number of pieces, called buckets. We create SplitCandidates between each 
    /// bucket, then we choose the SplitCandidate for this axis that has the lowest cost and return it. In the split()
    /// function, we will consider the SplitCandidates chosen for each axis, then choose the cheapest among those.
    ///
    /// INPUTS:
    ///     @param axis The axis we are splitting down (0 = x, 1 = y, 2 = z)
    ///     @param node The node we are trying to split
    /// OUTPUT:
    ///     @param minSplit The SplitCandidate with the lowest cost for this axis
    ///
    float splitAxis(int axis, SplitCandidate& minSplit, const LightTreeNode& node) const;

// ------------------------------------ Member Variables ---------------------------------------------------------------
    LIGHT_TREE_MEMBERS;
    std::vector<LightTreeNode> mNodes;         // array of nodes 
    std::vector<uint> mLightIndices;           // array of light indices -- allows us to change the "order" of 
                                               // lights in the light tree without mutating the lightset itself
};

} // end namespace pbr
} // end namespace moonray