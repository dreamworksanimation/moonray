// Copyright 2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#include "LightTree.isph"
#include <scene_rdl2/common/platform/IspcUtil.isph>


export uniform uint32_t
LightTree_hudValidation(uniform bool verbose)
{
    LIGHT_TREE_VALIDATION;
}

// --------------------------------- SAMPLING METHODS --------------------------------------- //

void LightTree_sample(const uniform LightTree * const uniform me,
                      varying float * uniform lightSelectionPdfs,
                      const varying Vec3f& P,
                      const varying Vec3f& N,
                      const varying Vec3f* cullingNormal,
                      const varying IntegratorSample1D& lightSelectionSample,
                      const uniform int * uniform lightIdMap)
{
    // For bounded lights, importance sample the BVH with adaptive tree splitting
    const varying bool cullLights = cullingNormal != nullptr;
    uniform int startNodes[2] = {0, -1}; // {root index, null index}
    LightTree_sampleRecurse(me, lightSelectionPdfs, startNodes, P, N, cullLights, lightSelectionSample, lightIdMap);
}

void LightTree_sampleBranch(const uniform LightTree * const uniform me,
                            varying int& lightIndex, 
                            varying float& pdf, 
                            varying float& r, 
                            uniform uint32_t nodeIndex,
                            const varying Vec3f& p, 
                            const varying Vec3f& n,
                            varying bool cullLights)
{
    const uniform LightTreeNode& node = me->mNodesPtr[nodeIndex];

    MNRY_ASSERT(LightTreeNode_getLightCount(node) > 0);

    // if node is a leaf, return the light
    if (LightTreeNode_isLeaf(node)) {
        lightIndex = LightTreeNode_getLightIndex(node);
        return;
    }

    // otherwise, get child nodes and traverse based on importance
    const uniform uint32_t iL = nodeIndex + 1;
    const uniform uint32_t iR = LightTreeNode_getRightNodeIndex(node);
    const varying float wL = LightTreeNode_importance(me->mNodesPtr[iL], p, n, me->mNodesPtr[iR], cullLights);
    const varying float wR = LightTreeNode_importance(me->mNodesPtr[iR], p, n, me->mNodesPtr[iL], cullLights);

    /// detect dead branch
    /// NOTE: there are three options: 1) just return invalid, as we're doing here, 2) choose a random
    /// light and return (technically more correct, but costly and doesn't improve convergence), 3) backtrack
    /// and choose the other branch. Worth exploring the best option in the future
    if (wL + wR == 0.f) {
        lightIndex = -1;
        return;
    }

    const varying float pdfL = wL / (wL + wR);

    // Choose which branch to traverse
    if (r < pdfL || pdfL == 1.f) {
        r = r / pdfL;
        pdf *= pdfL;
        LightTree_sampleBranch(me, lightIndex, pdf, r, iL, p, n, cullLights);
    } else {
        const varying float pdfR = 1.f - pdfL;
        r = (r - pdfL) / pdfR;
        pdf *= pdfR;
        LightTree_sampleBranch(me, lightIndex, pdf, r, iR, p, n, cullLights);
    }
}

/// Computes the spread of the lights in the node, biased by the energy variance. A higher value indicates that the 
/// lights are fairly compact, and that the energy doesn't vary much between them. A lower value indicates that the 
/// lights are spread apart, and that the energy varies much more. This measure is used to determine whether we 
/// split the node in question during sampling traversal, or whether we use importance sampling to choose a
/// representative light to sample. We could think of this as a confidence measure. How confident are we that our
/// importance sampling algorithm will choose a good representative light for this node? 
///
/// @see Alejandro Conty Estevez and Christopher Kulla. 2018. 
///      "Importance Sampling of Many Lights with Adaptive Tree Splitting" 
///       eqs (8), (9), (10)
///
varying float splittingHeuristic(const uniform LightTreeNode& node, const varying Vec3f& p)
{
    // TODO: also base this on the orientation cone?

    const uniform Vec3f center = LightTreeNode_getCenter(node);
    const uniform BBox3f bbox = LightTreeNode_getBBox(node);
    const uniform float radius = length(center - bbox.lower);
    const float distance = length(p - center);

    // if inside the bounding box, always split
    if (distance <= radius) {
        return 0.f;
    }

    // Find the size of the bbox (i.e. angle made with the bounding sphere) from the perspective of the point
    // theta should be between 0 and 90 deg (this is the half angle)
    const float lightSpreadTheta = asin(radius / distance); 
    // map to the [0,1] range by dividing by pi/2
    const float lightSpread = min(lightSpreadTheta * sTwoOverPi, 1.f);
    // take the sqrt to boost the lower values (i.e. raise the chance of splitting).
    // when splitting is low, it results in scenes with high amounts of noise -- this is 
    // why we generally bias more splitting over less.
    const float lightSpreadSqrt = sqrt(lightSpread);
    // map the energy variance to [0, 1] range, then take power of 4 to boost splitting chances
    // then map to [0.5, 1] so that we only ever boost (not lower) splitting chances
    // this is the simplified version of the calculation (1 - (1 / (1 + sqrt(x)))^4 ) + 1) / 2
    // (this calculation is primarily based on experimentation; finding what works best)
    const float arg = 1.f + sqrt(LightTreeNode_getEnergyVariance(node));
    const float energyVarianceMapped = 1.f - (1.f / (2.f * arg*arg*arg*arg));

    // energy variance is often 0, and in those cases we don't want to completely ignore the distance 
    // variance. So, let's instead bias the distance variance using the energy variance
    return 1.f - bias_Schlick(lightSpreadSqrt, energyVarianceMapped);
}

void LightTree_sampleRecurse(const uniform LightTree * const uniform me,
                             varying float * uniform lightSelectionPdfs,
                             uniform int nodeIndices[2],
                             const varying Vec3f& p, 
                             const varying Vec3f& n, 
                             varying bool cullLights, 
                             const varying IntegratorSample1D& lightSelectionSample,
                             const uniform int * uniform lightIdMap)
{
    // For each node in list, decide whether to traverse both subtrees or to use a stochastic approach
    for (uniform int i = 0; i < 2; ++i) {
        uniform int nodeIndex = nodeIndices[i];
        if (nodeIndex == -1) continue; // -1 means index doesn't exist

        const uniform LightTreeNode& node = me->mNodesPtr[nodeIndex];
        varying float lightPdf = 1.f;
        varying int lightIndex = -1;

        if (LightTreeNode_getLightCount(node) == 0) { continue; }

        // There's only 1 light in node -- no splitting left to be done
        if (LightTreeNode_isLeaf(node)) {
            // The pdf is 1 since splitting is deterministic
            lightIndex = LightTreeNode_getLightIndex(node);
            LightTree_chooseLight(lightSelectionPdfs, lightIndex, /*lightPdf*/ 1.f, lightIdMap);
            continue;
        }

        // Decide whether to traverse both subtrees (if the splitting heuristic is below the threshold/sampling quality)
        // OR to stop traversing and choose a light using importance sampling. 
        if (me->mSamplingThreshold == 0.0f || splittingHeuristic(node, p) > me->mSamplingThreshold) {
            // must generate new random number for every subtree traversal
            float r;
            getPseudoRandomSample(lightSelectionSample, r);
            LightTree_sampleBranch(me, lightIndex, lightPdf, r, nodeIndex, p, n, cullLights);
            LightTree_chooseLight(lightSelectionPdfs, lightIndex, lightPdf, lightIdMap);
            continue;
        } else {
            uniform int iL = nodeIndex + 1;
            uniform int iR = LightTreeNode_getRightNodeIndex(node);
            uniform int children[2] = {iL, iR};
            LightTree_sampleRecurse(me, lightSelectionPdfs, children, p, n, cullLights, 
                                    lightSelectionSample, lightIdMap); 
        } 
    }
}

// --------------------------------- PRINT FUNCTIONS -------------------------------------------

void LightTree_print(const uniform LightTree * const uniform lightTree)
{
    if (LightTreeNode_getLightCount(lightTree->mNodesPtr[0]) > 0) {
        LightTree_printRecurse(lightTree, 0, 0);
    } 
}

void LightTree_printRecurse(const uniform LightTree * const uniform lightTree, 
                            uniform uint32_t nodeIndex, uniform int depth)
{
    const uniform LightTreeNode& node = lightTree->mNodesPtr[nodeIndex];

    for (uniform int i = 0; i < depth; ++i) {
        print(" ");
    }
    print("%i", nodeIndex);

    // if node is a leaf, return
    if (LightTreeNode_isLeaf(node)) {
        print(", leaf (light index): %i\n", LightTreeNode_getLightIndex(node));
        return;
    }

    uniform uint32_t iL = nodeIndex + 1;
    uniform uint32_t iR = LightTreeNode_getRightNodeIndex(node);
    print("\n");

    LightTree_printRecurse(lightTree, iL, depth+1);
    LightTree_printRecurse(lightTree, iR, depth+1);
}

