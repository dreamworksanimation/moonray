// Copyright 2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#include "LightTree.h"
#include <moonray/rendering/pbr/light/LightTree_ispc_stubs.h>

namespace moonray {
namespace pbr {


LightTree::LightTree(float samplingThreshold) 
    : mSamplingThreshold(samplingThreshold) 
{}

// ----------------------------- BUILD METHODS ---------------------------------------------------------------------- //

void LightTree::build(const Light* const* boundedLights, uint32_t boundedLightCount,
                      const Light* const* unboundedLights, uint32_t unboundedLightCount) 
{
    mBoundedLights = boundedLights;
    mUnboundedLights = unboundedLights;
    mBoundedLightCount = boundedLightCount;
    mUnboundedLightCount = unboundedLightCount;

    // pre-allocate since we know the size of the array
    mLightIndices.reserve(boundedLightCount);

    // create list of indices to order lights in tree
    for (int i = 0; i < boundedLightCount; ++i) {
        // don't use mesh lights in light bvh
        if (dynamic_cast<const MeshLight*>(boundedLights[i])) {
            mBoundedLightCount--;
        } else {
            mLightIndices.push_back(i);
        }
    }

    if (mBoundedLightCount > 0) {

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
void LightTree::buildRecurse(uint32_t nodeIndex) 
{
    LightTreeNode* node = &mNodes[nodeIndex];
    MNRY_ASSERT(node->getLightCount() != 0);

    // if leaf node, return
    if (node->isLeaf()) {
        node->setLeafLightIndex(mLightIndices);
        return;
    }

    // -------- [1] section 4.4 ---------

    // find the lowest cost axis to split across
    LightTreeNode leftNode, rightNode;
    const float splitCost = split(leftNode, rightNode, nodeIndex);

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

float LightTree::split(LightTreeNode& leftNode, LightTreeNode& rightNode, uint32_t nodeIndex)
{
    LightTreeNode& node = mNodes[nodeIndex];

    // if there are only two lights in node, just create two nodes
    if (node.getLightCount() == 2) {
        leftNode.init(/* lightCount */ 1, node.getStartIndex(), mBoundedLights, mLightIndices);
        rightNode.init(/* lightCount */ 1, node.getStartIndex() + 1, mBoundedLights, mLightIndices);
        return /* zero split cost */ 0;
    }

    if (lightsAreCoincident(node, mBoundedLights)) {
        // if lights are all stacked on top of each other,
        // just split the number of lights evenly
        return splitLightsEvenly(leftNode, rightNode, node);
    } 
    
    // find the axis with the lowest split cost
    float minCost = -1.f;
    SplitCandidate minSplit;
    for (int axis = 0; axis < 3; ++axis) {
        // calculate cost of splitting down this axis
        SplitCandidate split;

        // split axis into buckets, return splitCandidate for the axis 
        // between buckets that has the minimum cost
        float splitCost = splitAxis(split, axis, node);

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

float LightTree::splitLightsEvenly(LightTreeNode& leftNode, LightTreeNode& rightNode, const LightTreeNode& parent) const
{
    // just evenly split the lights into both nodes
    const auto startIt = mLightIndices.begin() + parent.getStartIndex();
    // start the right node lights halfway through
    const auto rightStartIt = startIt + (parent.getLightCount() / 2);

    // create left and right child nodes
    const uint32_t lightCountLeft  = rightStartIt - startIt;
    const uint32_t lightCountRight = parent.getLightCount() - lightCountLeft;
    const uint32_t startIndexLeft  = parent.getStartIndex();
    const uint32_t startIndexRight = parent.getStartIndex() + lightCountLeft;

    // initialize nodes
    leftNode.init(lightCountLeft, startIndexLeft, mBoundedLights, mLightIndices);
    rightNode.init(lightCountRight, startIndexRight, mBoundedLights, mLightIndices);

    // we currently aren't using this cost anywhere, so it can be zero
    return 0.f;
}

float LightTree::splitAxis(SplitCandidate& minSplit, int axis, const LightTreeNode& node) const
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
        // TODO: investigate the bug where bucketIndex > 11
        bucketIndex = scene_rdl2::math::min(bucketIndex, 11);

        // add light to bucket
        LightTreeBucket& bucket = buckets[bucketIndex];
        bucket.addLight(light);
    }

    // Purge empty buckets and splits
    LightTreeBucket finalBuckets[NUM_BUCKETS];
    SplitCandidate finalSplits[numSplits];
    int finalSplitCount = purgeEmptyBuckets(finalBuckets, finalSplits, buckets, splits, NUM_BUCKETS);

    // Populate the splits, left and right sides separately
    populateSplitsLeftSide(finalSplits, finalBuckets, finalSplitCount);
    populateSplitsRightSide(finalSplits, finalBuckets, finalSplitCount);

    // find lowest cost split candidate and return
    return getCheapestSplit(minSplit, finalSplitCount, finalSplits, node);
}



// ---------------------------------------------- SAMPLING METHODS -------------------------------------------------- //

void LightTree::sample(float* lightSelectionPdfs, const scene_rdl2::math::Vec3f& P, const scene_rdl2::math::Vec3f& N, 
                       const scene_rdl2::math::Vec3f* cullingNormal, const IntegratorSample1D& lightSelectionSample,
                       const int* lightIdMap, int nonMirrorDepth) const
{
    // For bounded lights, importance sample the BVH with adaptive tree splitting
    if (mBoundedLightCount > 0) {
        const bool cullLights = cullingNormal != nullptr;
        int startNodes[2] = {0, -1}; // {root index, null index}
        sampleRecurse(lightSelectionPdfs, startNodes, P, N, cullLights, lightSelectionSample, lightIdMap, nonMirrorDepth);
    }
}

void LightTree::sampleBranch(int& lightIndex, float& pdf, float& r, uint32_t nodeIndex,
                             const scene_rdl2::math::Vec3f& p, const scene_rdl2::math::Vec3f& n, 
                             bool cullLights) const
{
    const LightTreeNode& node = mNodes[nodeIndex];

    MNRY_ASSERT(node.getLightCount() > 0);

    // if node is a leaf, return the light
    if (node.isLeaf()) {
        lightIndex = node.getLightIndex();
        return;
    }

    // otherwise, get child nodes and traverse based on importance
    const uint32_t iL = nodeIndex + 1;
    const uint32_t iR = node.getRightNodeIndex();
    const float wL = mNodes[iL].importance(p, n, mNodes[iR], cullLights);
    const float wR = mNodes[iR].importance(p, n, mNodes[iL], cullLights);

    /// detect dead branch
    /// NOTE: there are three options: 1) just return invalid, as we're doing here, 2) choose a random
    /// light and return (technically more correct, but costly and doesn't improve convergence), 3) backtrack
    /// and choose the other branch. Worth exploring the best option in the future
    if (wL + wR == 0.f) {
        lightIndex = -1;
        return;
    }

    const float pdfL = wL / (wL + wR);

    // Choose which branch to traverse
    if (r < pdfL || pdfL == 1.f) {
        r = r / pdfL;
        pdf *= pdfL;
        sampleBranch(lightIndex, pdf, r, iL, p, n, cullLights);
    } else {
        const float pdfR = 1.f - pdfL;
        r = (r - pdfL) / pdfR;
        pdf *= pdfR;
        sampleBranch(lightIndex, pdf, r, iR, p, n, cullLights);
    }
}

/// Computes the spread of the lights in the node, biased by the energy variance. A higher value indicates that the 
/// lights are fairly compact, and/or that the energy doesn't vary much between them. A lower value indicates that the 
/// lights are spread apart, and/or that the energy varies much more. This measure is used to determine whether we 
/// split the node in question during sampling traversal, or whether we use importance sampling to choose a
/// representative light to sample. We could think of this as a confidence measure. How confident are we that our
/// importance sampling algorithm will choose a good representative light for this node? 
///
/// @see Alejandro Conty Estevez and Christopher Kulla. 2018. 
///      "Importance Sampling of Many Lights with Adaptive Tree Splitting" 
///       eqs (8), (9), (10)
///
float splittingHeuristic(const LightTreeNode& node, const scene_rdl2::math::Vec3f& p)
{
    // TODO: also base this on the orientation cone?

    const scene_rdl2::math::BBox3f bbox = node.getBBox();
    const float radius = scene_rdl2::math::length(bbox.size()) * 0.5f;
    const float distance = scene_rdl2::math::distance(p, center(bbox));

    // if inside the bounding box, always split
    if (distance <= radius) {
        return 0.f;
    }

    // Find the size of the bbox (i.e. angle made with the bounding sphere) from the perspective of the point
    // theta should be between 0 and 90 deg (this is the half angle)
    const float lightSpreadTheta = scene_rdl2::math::asin(radius / distance); 
    // map to the [0,1] range by dividing by pi/2
    const float lightSpread = scene_rdl2::math::min(lightSpreadTheta * scene_rdl2::math::sTwoOverPi, 1.f);
    // take the sqrt to boost the lower values (i.e. raise the chance of splitting).
    // when splitting is low, it results in scenes with high amounts of noise -- this is 
    // why we generally bias more splitting over less.
    const float lightSpreadSqrt = sqrt(lightSpread);
    // map the energy variance to [0, 1] range, then take power of 4 to boost splitting chances
    // then map to [0.5, 1] so that we only ever boost (not lower) splitting chances
    // this is the simplified version of the calculation (1 - (1 / (1 + sqrt(x)))^4 ) + 1) / 2
    // (this calculation is primarily based on experimentation; finding what works best)
    float arg = 1.f + sqrt(node.getEnergyVariance());
    float arg2 = arg * arg;
    float energyVarianceMapped = 1.f - (0.5f / (arg2*arg2));

    // energy variance is often 0, and in those cases we don't want to completely ignore the distance 
    // variance. So, let's instead bias the distance variance using the energy variance
    return 1 - scene_rdl2::math::bias_Schlick(lightSpreadSqrt, energyVarianceMapped);
}

void LightTree::sampleRecurse(float* lightSelectionPdfs, int nodeIndices[2], const scene_rdl2::math::Vec3f& p, 
                              const scene_rdl2::math::Vec3f& n, bool cullLights, 
                              const IntegratorSample1D& lightSelectionSample, 
                              const int* lightIdMap, int nonMirrorDepth) const
{
    // For each node in list, decide whether to traverse both subtrees or to use a stochastic approach
    for (int i = 0; i < 2; ++i) {
        int nodeIndex = nodeIndices[i];
        if (nodeIndex == -1) continue; // -1 means index doesn't exist

        const LightTreeNode& node = mNodes[nodeIndex];
        float lightPdf = 1.f;
        int lightIndex = -1;

        if (node.getLightCount() == 0) { continue; }

        // There's only 1 light in node -- no splitting left to be done
        if (node.isLeaf()) {
            // The pdf is 1 since splitting is deterministic
            lightIndex = node.getLightIndex();
            chooseLight(lightSelectionPdfs, lightIndex, /*lightPdf*/ 1.f, lightIdMap);
            continue;
        }

        // Decide whether to traverse both subtrees (if the splitting heuristic is below the threshold/sampling quality)
        // OR to stop traversing and choose a light using importance sampling. 
        if (mSamplingThreshold == 0.f || splittingHeuristic(node, p) > mSamplingThreshold) {
            // must generate new random number for every subtree traversal
            float r;
            lightSelectionSample.getSample(&r, nonMirrorDepth);
            sampleBranch(lightIndex, lightPdf, r, nodeIndex, p, n, cullLights);
            chooseLight(lightSelectionPdfs, lightIndex, lightPdf, lightIdMap);
            continue;
        } else {
            int iL = nodeIndex + 1;
            int iR = node.getRightNodeIndex();
            int children[2] = {iL, iR};
            sampleRecurse(lightSelectionPdfs, children, p, n, cullLights, 
                          lightSelectionSample, lightIdMap, nonMirrorDepth); 
        } 
    }
}

// --------------------------------- UTILITY FUNCTIONS -------------------------------------------------------------- //

size_t LightTree::getMemoryFootprintRecurse(int nodeIndex) const
{
    const LightTreeNode& node = mNodes[nodeIndex];

    // if node is a leaf, return size of node
    if (node.isLeaf()) {
        return sizeof(LightTreeNode);
    }

    int iL = nodeIndex + 1;
    int iR = node.getRightNodeIndex();

    return getMemoryFootprintRecurse(iL) + getMemoryFootprintRecurse(iR);
}

size_t LightTree::getMemoryFootprint() const
{
    if (mBoundedLightCount > 0 && mNodes.size() > 0) {
        size_t lightIndicesSize = mBoundedLightCount * sizeof(uint32_t);
        return lightIndicesSize + getMemoryFootprintRecurse(0);
    }
    return 0;
}

void LightTree::print() const
{
    std::cout << "nodeCount: " << mNodes.size() << std::endl;
    if (!mNodes.empty()) {
        printRecurse(0, 0);
    }   
}

void LightTree::printRecurse(uint32_t nodeIndex, int depth) const
{
    const LightTreeNode& node = mNodes[nodeIndex];
    const Light* const light = mBoundedLights[node.getLightIndex()];

    for (int i = 0; i < depth; ++i) {
        std::cout << " ";
    }
    std::cout << nodeIndex;

    // if node is a leaf, return
    if (node.isLeaf()) {
        std::cout << ", leaf: " << light->getRdlLight()->get(scene_rdl2::rdl2::Light::sLabel) << std::endl;
        return;
    }

    uint iL = nodeIndex + 1;
    uint iR = node.getRightNodeIndex();
    std::cout << "\n";

    printRecurse(iL, depth+1);
    printRecurse(iR, depth+1);
}

} // end namespace pbr
} // end namespace moonray