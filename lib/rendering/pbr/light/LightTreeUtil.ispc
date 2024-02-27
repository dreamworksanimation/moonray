#include "LightTreeUtil.isph"
#include <scene_rdl2/common/platform/IspcUtil.isph>

// =====================================================================================================================
// References:
// =====================================================================================================================
// [1] Alejandro Conty Estevez and Christopher Kulla. 2018. 
//     "Importance Sampling of Many Lights with Adaptive Tree Splitting"
// [2] Cem Yuksel. 2019. 
//     "Stochastic Lightcuts"
// =====================================================================================================================


// Calculate the importance weight for the node
varying float LightTreeNode_importance(const uniform LightTreeNode& node,
                              const varying Vec3f& p, 
                              const varying Vec3f& n,
                              const uniform LightTreeNode& sibling,
                              varying bool cullLights)
{
    if (node.mLightCount == 0) return 0.f;

    // calculations used throughout
    const uniform Vec3f center = LightTreeNode_getCenter(node);
    const varying Vec3f bboxToPt = p - center;
    const varying Vec3f dirToPt = normalize(bboxToPt);
    const varying float dSqr = lengthSqr(bboxToPt);                   // squared distance to point
    const uniform float rSqr = lengthSqr(node.mBBox.upper - center);  // squared radius of sphere circumscribing the bbox

    // Find uncertainty angle
    varying float sinThetaU, cosThetaU;
    calcSinCosThetaU(dSqr, rSqr, (varying float * uniform)&sinThetaU, (varying float * uniform)&cosThetaU);

    const varying float distanceTerm = calcDistanceTerm(p, sibling, dSqr, rSqr);
    const varying float geometricTerm = calcGeometricTerm(node, p, cosThetaU, sinThetaU, dirToPt);
    const varying float materialTerm = calcMaterialTerm(p, n, cullLights, cosThetaU, sinThetaU, dirToPt);

    return node.mEnergy * geometricTerm * materialTerm * distanceTerm;
}

// Calculate the uncertainty angle
static void calcSinCosThetaU(const varying float dSqr, const uniform float rSqr, 
                             varying float * uniform sinTheta, varying float * uniform cosTheta)
{
    MNRY_ASSERT(rSqr > 0);

    // if p is inside bounding sphere, return full sphere of directions
    if (dSqr < rSqr) {
        *cosTheta = -1.f;
        *sinTheta =  0.f;
        return;
    }

    // calc sin(theta)^2
    const varying float sinSqrTheta = rSqr / dSqr;
    *sinTheta = sqrt(sinSqrTheta);

    // use the cos(theta)^2 + sin(theta)^2 = 1 identity to find cos(theta)
    *cosTheta = sqrt(1.f - sinSqrTheta); 
}

static varying float calcDistanceTerm(const varying Vec3f& p,
                                      const uniform LightTreeNode& sibling,
                                      const varying float dSqr,
                                      const uniform float rSqr)
{
    const uniform float diagLenSqr = 4.f * rSqr; // (2r)^2 = 4r^2
    const uniform BBox3f siblingBBox = LightTreeNode_getBBox(sibling);
    const uniform Vec3f diagSibling = siblingBBox.upper - siblingBBox.lower;
    const uniform float diagLenSqrSibling = lengthSqr(diagSibling);

    // find distance to bbox
    /// NOTE: I've found that getting the min dist to the bbox (instead of the center) produces 
    /// about the same results, but is more problematic
    const varying float dSqrSibling = lengthSqr(p - LightTreeNode_getCenter(sibling));

    // We ensure the point is more than a threshold distance outside both bounding boxes because we want to ignore 
    // the distance term for nodes that are higher up in the BVH. The reason for this is that the distance term 
    // doesn't give us very meaningful information when the bounding boxes contain a large number of spread out lights, 
    // and it overpowers more useful info, like the material and geometric terms. So, we just include it at lower 
    // levels of the BVH, where 1/d^2 is more indicative of the lights' positions. [2] eq (3)
    if (dSqr > diagLenSqr && dSqrSibling > diagLenSqrSibling) {
        // we normalize this term with diagLen^2 so that it's continuous from 1.f
        // (then we make it the reciprocal so we can multiply it into the importance term)
        return diagLenSqr / dSqr;
    } 
    // if closer than some threshold distance for either node, just ignore distance term
    return 1.f;
}

varying float calcGeometricTerm(const uniform LightTreeNode& node,
                                const varying Vec3f& p, 
                                varying float cosThetaU, 
                                varying float sinThetaU,
                                const varying Vec3f& dirToPt)
{
    varying float cosTheta = clamp(dot(node.mCone.mAxis, dirToPt), -1.f, 1.f);
    if (node.mCone.mTwoSided) {
        cosTheta = abs(cosTheta);
    }
    const varying float sinTheta = sqrt(1.f - cosTheta*cosTheta);

    const varying float cosThetaO = node.mCone.mCosThetaO;
    const varying float sinThetaO = node.mCone.mSinThetaO;

    // minimum angle any emitter in the cluster will form with the direction toward the shading pt
    // max(theta - thetaO - thetaU, 0)
    // ---------------------------------------------------------------------------------------------
    // max(cos(theta - thetaO), 1)
    const varying float cosThetaSubtract = cosSubClamped(sinTheta, cosTheta, sinThetaO, cosThetaO);
    const varying float sinThetaSubtract = sqrt(1.f - cosThetaSubtract*cosThetaSubtract);
    // max(cos(thetaSubtract - thetaU), 1)
    const varying float cosThetaPrime = cosSubClamped(sinThetaSubtract, cosThetaSubtract, sinThetaU, cosThetaU);
    // ---------------------------------------------------------------------------------------------

    // if min angle of emitter is larger than the cone's emission angle, can't contribute to point
    return cosThetaPrime <= node.mCone.mCosThetaE ? 0.f : cosThetaPrime;
}

varying float calcMaterialTerm(const varying Vec3f& p, 
                               const varying Vec3f& n, 
                               varying bool cullLights,
                               varying float cosThetaU, 
                               varying float sinThetaU, 
                               const varying Vec3f& dirToPt)
{
    // calc angle between normal and direction to the center of the bbox (cull lights below the horizon)
    const varying Vec3f dirToBbox = neg(dirToPt); 
    const varying float cosThetaI = clamp(dot(n, dirToBbox), -1.f, 1.f);
    const varying float sinThetaI = sqrt(1.f - cosThetaI*cosThetaI);

    // get the MINIMUM angle to the light, which is theta_i minus theta_u
    const varying float cosThetaIPrime = cosSubClamped(sinThetaI, cosThetaI, sinThetaU, cosThetaU);

    /// TODO: the algorithm in the paper says absdot, but we cull if dot is less than zero, 
    /// so the light sample will be invalid anyway.... look at this again 
    if (cullLights && cosThetaIPrime < 0.f) {
        return 0.f;
    }
    return abs(cosThetaIPrime);
}
