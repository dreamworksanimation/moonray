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

/// Computes the combined variance of the distance and energy terms. A higher variance indicates fewer lights should be 
/// sampled, since there is a clearer winner for importance sampling. A lower variance indicates more lights should be 
/// sampled, since each subtree has the potential for equally important contributions. 
///
/// @see [1] eqs (8), (9), (10)
///
varying float splittingHeuristic(const uniform LightTreeNode& node, 
                                 const varying Vec3f& p, 
                                 const uniform LightTreeNode& root,
                                 uniform float sceneSize)
{
    // "(a, b) is the range where the distance to an emitter in the cluster varies"
    // "This can be simply obtained from the cluster center and the radius of the bounding sphere"
    const uniform Vec3f center = LightTreeNode_getCenter(node);
    const uniform BBox3f bbox = LightTreeNode_getBBox(node);
    const uniform float radius = length(center - bbox.lower);
    const varying float dist_to_bbox_center = length(p - center);

    /// TODO: find tighter bounds 
    // Normalize a and b by dividing by the scene scale
    const varying float b = (dist_to_bbox_center + radius) / sceneSize;
    const varying float a = (dist_to_bbox_center - radius) / sceneSize;

    if (a <= 0.f || b == 0.f) {
        return 0.f;
    }
    const varying float b_minus_a = b - a;

    // (8) E[g] = 1 / ab
    const varying float distanceMean = 1.f / (a * b);
    const varying float distanceMean2 = distanceMean*distanceMean;

    // (9) V[g] =  [(b^3 - a^3) / (3(b - a) * a^3 * b^3)] - [1 / a^2b^2]
    const varying float a2 = a*a;
    const varying float b2 = b*b;
    const varying float a3 = a2*a;
    const varying float b3 = b2*b;
    const varying float distanceVariance = ((b3 - a3) / (3 * b_minus_a * a3 * b3)) - (1.f / (a2*b2));

    // (10) (total light variance) omega^2 = [V[e] * V[g] + V[e] * E[g]^2 + E[e]^2 * V[g]]
    // where V[e] is the precomputed variance of the energy stored in the cluster
    varying float lightVariance = LightTreeNode_getEnergyVariance(node)*distanceVariance 
                                 + LightTreeNode_getEnergyVariance(node)*distanceMean2 
                                 + LightTreeNode_getEnergyMean(node)*LightTreeNode_getEnergyMean(node)*distanceVariance;

    // remap light variance to [0, 1] range using (1 / 1 + omega)^(1/4)
    varying float powArg1 = 1.f / (1.f + sqrt(lightVariance)); 
    lightVariance = lightVariance == 0.f ? 0.f : pow(powArg1, (varying float)0.25f);
    return lightVariance;
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
        varying float continueCost = splittingHeuristic(node, p, me->mNodesPtr[0], me->mSceneDiameter);
        if (continueCost > me->mSamplingThreshold) {
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

