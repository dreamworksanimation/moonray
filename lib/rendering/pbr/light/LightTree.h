// Copyright 2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Light.h"
#include "LightTreeUtil.h"
#include "LightTree.hh"
#include "MeshLight.h"

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

    /// Chooses light(s) using importance sampling and adaptive tree splitting [1] (Section 5.4).
    ///
    /// OUTPUTS:
    ///     @param lightSelectionPdfs A list of light selection probabilities, saved to the associated light's index.
    ///                               Any pdf of -1 indicates that the light was not chosen (default).
    ///
    void sample(float* lightSelectionPdfs,
                const scene_rdl2::math::Vec3f& P, 
                const scene_rdl2::math::Vec3f& N, 
                const scene_rdl2::math::Vec3f* cullingNormal,
                const IntegratorSample1D& lightSelectionSample,
                const int* lightIdMap, int nonMirrorDepth) const;

    /// Sets the scene diameter (size of the scene bvh's bounding box)
    void setSceneDiameter(float sceneDiameter) { mSceneDiameter = sceneDiameter; }

    /// Sets the sampling threshold (which determines the amount of adaptive tree splitting)
    void setSamplingThreshold(float threshold) { mSamplingThreshold = threshold; }

    /// Returns the memory footprint of the tree (in bytes)
    size_t getMemoryFootprint() const;

    /// Print the tree
    void print() const;

private:

/// ----------------------------- Inline Helpers -----------------------------------------

    inline void chooseLight(float* lightSelectionPdfs, int lightIndex, float pdf, const int* lightIdMap) const
    {
        if (lightIndex < 0) {
            return;
        }
        // Convert light index into the light set's index. If it's -1, it means that 
        // light has been culled, and therefore is not in the set
        const int visibleLightIndex = lightIdMap[lightIndex];
        if (visibleLightIndex >= 0) {
            lightSelectionPdfs[visibleLightIndex] = pdf;
        }
    }

    // Returns whether all lights are in the same position
    inline bool lightsAreCoincident(const LightTreeNode& node, const Light* const* lights)
    {
        int startIndex = node.getStartIndex();
        int lightCount = node.getLightCount();

        const scene_rdl2::math::Vec3f firstLightPosition = lights[startIndex]->getPosition(0.f);
        for (int i = startIndex + 1; i < startIndex + lightCount; ++i) {
            const scene_rdl2::math::Vec3f& p = lights[i]->getPosition(0.f);
            if (!scene_rdl2::math::isEqual(p, firstLightPosition)) {
                return false;
            }
        }
        return true;
    }

    // Returns a new list of buckets and splits with empty buckets/splits removed,
    // as well as the number of splits that remain
    inline int purgeEmptyBuckets(LightTreeBucket* newBuckets, SplitCandidate* newSplits,
                                 const LightTreeBucket* const oldBuckets, const SplitCandidate* const oldSplits,
                                 int oldBucketCount) const

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
    inline float getCheapestSplit(SplitCandidate& minSplit, int splitCount, 
                                  const SplitCandidate* const splits, const LightTreeNode& node) const
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
    void buildRecurse(uint32_t nodeIndex);


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
    float split(LightTreeNode& leftNode, LightTreeNode& rightNode, uint32_t nodeIndex);


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
    float splitAxis(SplitCandidate& minSplit, int axis, const LightTreeNode& node) const;

    /// Splits the node into two nodes, where it's just an even split of all the lights in the parent node
    float splitLightsEvenly(LightTreeNode& leftNode, LightTreeNode& rightNode, const LightTreeNode& parent) const;


    /// Choose a light from the hierarchy using importance sampling. We traverse the hierarchy by using a random number, 
    /// r, to determine which subtree to traverse. Each subtree (node) has an associated importance weight which 
    /// determines the probability of choosing one node over another. 
    /// @see [1] eq (5)
    ///
    /// NOTABLE INPUTS:
    ///     @param nodeIndex The current node of the tree
    ///
    /// OUTPUTS:
    ///     @param lightIndex The index of the light we selected
    ///     @param pdf The probability of selecting that light
    ///     @param r The random number used to determine the subtree to traverse, rescaled in each iteration
    ///
    void sampleBranch(int& lightIndex, float& pdf, float& r, uint32_t nodeIndex,
                      const scene_rdl2::math::Vec3f& p, const scene_rdl2::math::Vec3f& n, bool cullLights) const;


    /// Recursive function that chooses light(s) to sample, using adaptive tree splitting and a user-specified quality 
    /// control. This quality control, mSamplingQuality, is a threshold [0, 1] that determines whether we traverse both 
    /// subtrees or stop traversing and choose a light using importance sampling. When mSamplingQuality is closer to 0.0, 
    /// fewer lights will be sampled, and when it is closer to 1.0, more lights will be sampled. 
    ///
    /// @see [1] (Section 5.4)
    ///
    /// NOTABLE INPUTS:
    ///     @param lightSelectionSample Random number sequence we use when selecting a light
    ///     @param nodeIndices The current node(s) we are traversing (we explore both branches)
    ///
    /// OUTPUTS:
    ///     @param lightSelectionPdfs A list of light selection pdfs, where the pdf is stored in the corresponding 
    ///                               light's index. Any lights not chosen will have a pdf of -1.
    ///
    void sampleRecurse(float* lightSelectionPdfs, int nodeIndices[2], 
                       const scene_rdl2::math::Vec3f& p, const scene_rdl2::math::Vec3f& n, bool cullLights, 
                       const IntegratorSample1D& lightSelectionSample,
                       const int* lightIdMap, int nonMirrorDepth) const;

    /// Recursive helper to size()
    size_t getMemoryFootprintRecurse(int nodeIndex) const;

    /// Recursively print the tree
    void printRecurse(uint32_t nodeIndex, int depth) const;

// ------------------------------------ Member Variables ---------------------------------------------------------------
    LIGHT_TREE_MEMBERS;
    std::vector<LightTreeNode> mNodes;         // array of nodes 
    std::vector<uint32_t> mLightIndices;       // array of light indices -- allows us to change the "order" of 
                                               // lights in the light tree without mutating the lightset itself
};

} // end namespace pbr
} // end namespace moonray