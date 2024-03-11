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

void LightTree::build(const Light* const* boundedLights, uint32_t boundedLightCount,
                      const Light* const* unboundedLights, uint32_t unboundedLightCount) 
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
    // Sample ALL unbounded lights
    /// TODO: the paper has some probability specifying whether to sample unbounded lights vs. the BVH
    /// Then, if the unbounded light set is chosen, it randomly picks an unbounded light to sample from the set. 
    /// Since we use so few unbounded lights, I figure it would be better to just sample all unbounded lights for now
    for (int unboundedLightIdx = 0; unboundedLightIdx < mUnboundedLightCount; ++unboundedLightIdx) {
        int lightIndex = mBoundedLightCount + unboundedLightIdx;
        chooseLight(lightSelectionPdfs, lightIndex, /*lightPdf*/ 1.f, lightIdMap);
    }

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

/// Computes the combined variance of the distance and energy terms. A higher variance indicates fewer lights should be 
/// sampled, since there is a clearer winner for importance sampling. A lower variance indicates more lights should be 
/// sampled, since each subtree has the potential for equally important contributions. 
///
/// @see [1] eqs (8), (9), (10)
///
float splittingHeuristic(const LightTreeNode& node, const scene_rdl2::math::Vec3f& p, 
                         const LightTreeNode& root, float sceneDiameter)
{
    // "(a, b) is the range where the distance to an emitter in the cluster varies"
    // "This can be simply obtained from the cluster center and the radius of the bounding sphere"
    const scene_rdl2::math::BBox3f bbox = node.getBBox();
    const float radius = scene_rdl2::math::length(bbox.size()) * 0.5f;
    const float dist_to_bbox_center = scene_rdl2::math::distance(p, center(bbox));

    /// TODO: find tighter bounds 
    // Normalize a and b by dividing by the scene scale
    const float b = (dist_to_bbox_center + radius) / sceneDiameter;
    const float a = (dist_to_bbox_center - radius) / sceneDiameter;

    if (a <= 0.f || b == 0.f) {
        return 0.f;
    }
    const float b_minus_a = b - a;

    // [1] eq (8) E[g] = 1 / ab
    const float distanceMean = 1.f / (a * b);
    const float distanceMean2 = distanceMean*distanceMean;

    // [1] eq (9) V[g] =  [(b^3 - a^3) / (3(b - a) * a^3 * b^3)] - [1 / a^2b^2]
    const float a2 = a*a;
    const float b2 = b*b;
    const float a3 = a2*a;
    const float b3 = b2*b;
    const float distanceVariance = ((b3 - a3) / (3 * b_minus_a * a3 * b3)) - (1.f / (a2*b2));

    // [1] eq (10) (total light variance) omega^2 = [V[e] * V[g] + V[e] * E[g]^2 + E[e]^2 * V[g]]
    // where V[e] is the precomputed variance of the energy stored in the cluster
    float lightVariance = node.getEnergyVariance()*distanceVariance 
                        + node.getEnergyVariance()*distanceMean2 
                        + node.getEnergyMean()*node.getEnergyMean()*distanceVariance;

    // remap light variance to [0, 1] range using (1 / 1 + omega)^(1/4)
    float powArg1 = 1.f / (1.f + sqrt(lightVariance)); 
    lightVariance = lightVariance == 0.f ? 0.f : pow(powArg1, 0.25);
    return lightVariance;
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
        float continueCost = splittingHeuristic(node, p, mNodes[0], mSceneDiameter);
        if (continueCost > mSamplingThreshold) {
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

size_t LightTree::getMemoryFootprintRecurse(int nodeIndex) const
{
    const LightTreeNode& node = mNodes[nodeIndex];

    // if node is a leaf, return size of node
    if (node.isLeaf()) {
        return node.getMemoryFootprint();
    }

    int iL = nodeIndex + 1;
    int iR = node.getRightNodeIndex();

    return getMemoryFootprintRecurse(iL) + getMemoryFootprintRecurse(iR);
}

size_t LightTree::getMemoryFootprint() const
{
    if (mBoundedLightCount > 0 && mNodes.size() > 0) {
        size_t lightIndicesSize = mBoundedLightCount * sizeof(float);
        return lightIndicesSize + getMemoryFootprintRecurse(0);
    }
    return 0;
}

// --------------------------------- PRINT FUNCTIONS ---------------------------------------------------------------- //

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