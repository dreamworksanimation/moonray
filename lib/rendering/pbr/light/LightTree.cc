// Copyright 2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#include "LightTree.h"
#include <moonray/rendering/pbr/light/LightTree_ispc_stubs.h>

namespace moonray {
namespace pbr {

// =====================================================================================================================
// References:
// =====================================================================================================================
// [1] Alejandro Conty Estevez and Christopher Kulla. 2018. 
//     "Importance Sampling of Many Lights with Adaptive Tree Splitting"
// =====================================================================================================================

LightTree::LightTree(float sceneDiameter, float samplingThreshold) 
    : mSceneDiameter(sceneDiameter), mSamplingThreshold(samplingThreshold) 
{}

// ----------------------------- BUILD METHODS ---------------------------------------------------------------------- //

void LightTree::build(const Light* const* boundedLights, uint boundedLightCount,
                      const Light* const* unboundedLights, uint unboundedLightCount) 
{
    mBoundedLights = boundedLights;
    mUnboundedLights = unboundedLights;
    mBoundedLightCount = boundedLightCount;
    mUnboundedLightCount = unboundedLightCount;

    // pre-allocate since we know the size of the array
    mLightIndices.reserve(boundedLightCount);

    if (boundedLightCount > 0) {
        // create list of indices to order lights in tree
        for (int i = 0; i < mBoundedLightCount; ++i) {
            mLightIndices.push_back(i);
        }

        // create root node
        mNodes.emplace_back();
        LightTreeNode& rootNode = mNodes.back();
        rootNode.init(mBoundedLightCount, /* root index*/ 0, mBoundedLights, mLightIndices);

        // build light tree recursively
        buildRecurse(/* root index */ 0);

        // update HUD data
        mNodesPtr = mNodes.data();
        mLightIndicesPtr = mLightIndices.data();
    }
}

/// Recursively build tree
void LightTree::buildRecurse(uint nodeIndex) 
{
    LightTreeNode* node = &mNodes[nodeIndex];

    // if leaf node, return
    if (node->isLeaf()) {
        node->setLeafLightIndex(mLightIndices);
        return;
    }

    // -------- [1] section 4.4 ---------

    // find the lowest cost axis to split across
    LightTreeNode leftNode, rightNode;
    const float splitCost = split(nodeIndex, leftNode, rightNode);

    /// NOTE: we might investigate gains if we terminate early when cost to split is greater 
    /// than cost of producing a leaf (i.e. the total node energy)
    /// Looked noisy to me, but perhaps it was buggy previously
    /// two options: 1) choose a random light from the resulting cluster, or 
    /// 2) sample all lights in the cluster

    // ----------------------------------

    mNodes.push_back(leftNode);
    buildRecurse(mNodes.size() - 1);

    mNodes.push_back(rightNode);
    mNodes[nodeIndex].setRightNodeIndex(mNodes.size() - 1);
    buildRecurse(mNodes.size() - 1);
}

float LightTree::split(uint nodeIndex, LightTreeNode& leftNode, LightTreeNode& rightNode)
{
    LightTreeNode& node = mNodes[nodeIndex];

    // if there are only two lights in node, just create two nodes
    if (node.getLightCount() == 2) {
        leftNode.init(/* lightCount */ 1, node.getStartIndex(), mBoundedLights, mLightIndices);
        rightNode.init(/* lightCount */ 1, node.getStartIndex() + 1, mBoundedLights, mLightIndices);
        return /* zero split cost */ 0;
    }

    // find the axis with the lowest split cost
    float minCost = -1.f;
    SplitCandidate minSplit;
    for (int axis = 0; axis < 3; ++axis) {
        // calculate cost of splitting down this axis
        SplitCandidate split;

        // split axis into buckets, return splitCandidate for the axis 
        // between buckets that has the minimum cost
        float splitCost = splitAxis(axis, split, node);

        // if this axis's split cost is lower than the current min
        // OR this is the first time through the loop, replace the min cost
        if (axis == 0 || splitCost < minCost) {
            minCost = splitCost;
            minSplit = split;
        }
    }

    // partition into left and right nodes using the chosen SplitCandidate
    minSplit.performSplit(leftNode, rightNode, mBoundedLights, mLightIndices, node);
    return minCost;
}

float LightTree::splitAxis(int axis, SplitCandidate& minSplit, const LightTreeNode& node) const
{
    /// TODO: make this a user parameter
    int NUM_BUCKETS = 12, numSplits = NUM_BUCKETS - 1;

    // find bucket size using the node bbox and num_buckets
    float minBound = node.getBBox().lower[axis];            // lower bound of axis
    float range = node.getBBox().size()[axis];              // length of the axis covered by the node
    float bucketSize = range / NUM_BUCKETS;                 // size of each bucket

    // create buckets
    LightTreeBucket buckets[NUM_BUCKETS];
    SplitCandidate splits[numSplits];

    // initialize split axes
    for (int i = 0; i < numSplits; ++i) {
        float axisOffset = minBound + bucketSize * (i + 1);
        splits[i].mAxis = {axis, axisOffset};
    }

    // populate buckets
    for (int i = node.getStartIndex(); i < node.getStartIndex() + node.getLightCount(); ++i) {
        const Light* const light = mBoundedLights[mLightIndices[i]];
        float p = light->getPosition(0)[axis];

        // remap position to bucket range, then bin it in a bucket
        int bucketIndex = scene_rdl2::math::floor(((p - minBound) / range) * NUM_BUCKETS);

        // add light to bucket
        LightTreeBucket& bucket = buckets[bucketIndex];
        bucket.addLight(light);
    }

    // Purge empty buckets and splits
    LightTreeBucket finalBuckets[NUM_BUCKETS];
    SplitCandidate finalSplits[numSplits];
    int finalSplitCount = purgeEmptyBuckets(buckets, splits, finalBuckets, finalSplits, NUM_BUCKETS);

    // Populate the splits, left and right sides separately
    populateSplitsLeftSide(finalSplits, finalBuckets, finalSplitCount);
    populateSplitsRightSide(finalSplits, finalBuckets, finalSplitCount);

    // find lowest cost split candidate and return
    return getCheapestSplit(finalSplitCount, finalSplits, node, minSplit);
}

// --------------------------------- SAMPLING METHODS --------------------------------------------------------------- //

void LightTree::sample() const {}

// --------------------------------- PRINT FUNCTIONS ---------------------------------------------------------------- //

void LightTree::print() const {}

} // end namespace pbr
} // end namespace moonray