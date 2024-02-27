#include "LightTreeUtil.h"

using namespace scene_rdl2::math;

namespace moonray {
namespace pbr {

// =====================================================================================================================
// References:
// =====================================================================================================================
// [1] Alejandro Conty Estevez and Christopher Kulla. 2018. 
//     "Importance Sampling of Many Lights with Adaptive Tree Splitting"
// [2] Cem Yuksel. 2019. 
//     "Stochastic Lightcuts"
// =====================================================================================================================


/// --------------------------------------------- LightTreeCone --------------------------------------------------------

LightTreeCone combineCones(const LightTreeCone& a, const LightTreeCone& b)
{
    if (a.isEmpty()) { return b; }
    if (b.isEmpty()) { return a; }

    // find cone with the largest orientation angle
    const auto sortedCones = std::minmax(a, b, [](const LightTreeCone& x, const LightTreeCone& y) { 
        return x.mCosThetaO > y.mCosThetaO; 
    });
    const LightTreeCone& smallerCone = sortedCones.first;
    const LightTreeCone& largerCone = sortedCones.second;

    const float largerConeThetaO = largerCone.getThetaO();
    const float smallerConeThetaO = smallerCone.getThetaO();
    const bool twoSided = a.mTwoSided || b.mTwoSided;

    // get angle between two axes
    const float theta_d = scene_rdl2::math::dw_acos(dot(largerCone.mAxis, smallerCone.mAxis));
    // get the max emission angle (min cosine)
    const float theta_e = max(largerCone.getThetaE(), smallerCone.getThetaE());

    // check if bounds for "a" already cover "b"
    if (min(theta_d + smallerConeThetaO, sPi) <= largerConeThetaO) {
        return LightTreeCone(largerCone.mAxis, largerCone.mCosThetaO, scene_rdl2::math::cos(theta_e), twoSided);
    } else {
        // generate a new cone that covers a and b
        const float theta_o = (largerConeThetaO + theta_d + smallerConeThetaO) * 0.5f;
        // if theta_o > pi, this is a sphere
        if (theta_o >= sPi) {
            return LightTreeCone(largerCone.mAxis, /* cos of Pi*/ -1, scene_rdl2::math::cos(theta_e), twoSided);
        }

        // rotate a's axis towards b's axis to get the new central axis for the cone
        const float theta_r = theta_o - largerConeThetaO;
        // axis to rotate about (axis orthogonal to a axis and b axis)
        Vec3f rotAxis = cross(largerCone.mAxis, smallerCone.mAxis);

        // if facing the same way, keep the same axis
        if (length(rotAxis) < sEpsilon) {
            return LightTreeCone(largerCone.mAxis, scene_rdl2::math::cos(theta_o), scene_rdl2::math::cos(theta_e), twoSided);
        }
        rotAxis = normalize(rotAxis);

        // rotation a's axis around rotAxis by theta_r
        const Vec3f axis = scene_rdl2::math::cos(theta_r) * largerCone.mAxis + 
                           scene_rdl2::math::sin(theta_r) * cross(rotAxis, largerCone.mAxis);
        return LightTreeCone(normalize(axis), scene_rdl2::math::cos(theta_o), scene_rdl2::math::cos(theta_e), twoSided);
    }
}


/// ----------------------------------------- LightTreeBucket ----------------------------------------------------------

void LightTreeBucket::addLight(const Light* const light)
{
    mEnergy += scene_rdl2::math::luminance(light->getRadiance());
    mBBox.extend(light->getBounds());
    LightTreeCone cone(light->getDirection(0.f), scene_rdl2::math::cos(light->getThetaO()), 
              scene_rdl2::math::cos(light->getThetaE()), light->isTwoSided());
    mCone = combineCones(cone, mCone);
}


/// ----------------------------------------- SplitCandidate -----------------------------------------------------------

void SplitCandidate::setLeftSide(const LightTreeBucket& leftBucket)
{
    mLeftEnergy = leftBucket.mEnergy;
    mLeftBBox = leftBucket.mBBox;
    mLeftCone = leftBucket.mCone;
}
void SplitCandidate::setLeftSide(const SplitCandidate& leftSplit, const LightTreeBucket& leftBucket)
{
    mLeftEnergy = leftSplit.mLeftEnergy + leftBucket.mEnergy;
    mLeftBBox = leftSplit.mLeftBBox;
    mLeftBBox.extend(leftBucket.mBBox);
    mLeftCone = combineCones(leftSplit.mLeftCone, leftBucket.mCone);
}
void SplitCandidate::setRightSide(const LightTreeBucket& rightBucket)
{
    mRightEnergy = rightBucket.mEnergy;
    mRightBBox = rightBucket.mBBox;
    mRightCone = rightBucket.mCone;
}
void SplitCandidate::setRightSide(const SplitCandidate& rightSplit, const LightTreeBucket& rightBucket)
{
    mRightEnergy = rightSplit.mRightEnergy + rightBucket.mEnergy;
    mRightBBox = rightSplit.mRightBBox;
    mRightBBox.extend(rightBucket.mBBox);
    mRightCone = combineCones(rightSplit.mRightCone, rightBucket.mCone);
}

float SplitCandidate::calcOrientationTerm(const LightTreeCone& cone) const
{
    const float theta_o = cone.getThetaO();
    const float theta_w = scene_rdl2::math::min(theta_o + cone.getThetaE(), sPi);
    const float sinThetaO = scene_rdl2::math::sqrt(1 - cone.mCosThetaO*cone.mCosThetaO);

    const float orientationTermLeft = 2.f * sPi * (1.f - cone.mCosThetaO);
    const float orientationTermRight = 0.5f * sPi * (
                                       (2.f * theta_w * sinThetaO) - 
                                       scene_rdl2::math::cos(theta_o - 2.f * theta_w) -
                                       (2.f * theta_o * sinThetaO) + cone.mCosThetaO
                                    );
    return orientationTermLeft + orientationTermRight;
}

float SplitCandidate::cost(const BBox3f& parentBBox, const LightTreeCone& parentCone) const
{
    // regularization factor Kr = length_max / length_axis
    const float length_max = parentBBox.size()[maxDim(parentBBox.size())];
    const float length_axis = parentBBox.size()[mAxis.first];
    const float kr = length_max / length_axis;

    const float numeratorL = mLeftEnergy  * bboxArea(mLeftBBox)  * calcOrientationTerm(mLeftCone);
    const float numeratorR = mRightEnergy * bboxArea(mRightBBox) * calcOrientationTerm(mRightCone);
    const float denominator = bboxArea(parentBBox) * calcOrientationTerm(parentCone);

    return kr * ( (numeratorL + numeratorR) / denominator );
}

void SplitCandidate::performSplit(LightTreeNode& leftNode, LightTreeNode& rightNode, const Light* const* lights, 
                                  std::vector<uint>& lightIndices, const LightTreeNode& parent)
{
    // sort lights on either side of axis
    const auto startIt = lightIndices.begin() + parent.getStartIndex();
    const auto endIt   = lightIndices.begin() + parent.getStartIndex() + parent.getLightCount();
    const float splitPlane = mAxis.second;
    const auto splitFunc = [&](uint lightIndex) {
        // Using < instead of <= here because lights that land on the split end 
        // up in the bucket above it (see splitAxis() where we populate buckets)
        return lights[lightIndex]->getPosition(0)[mAxis.first] < splitPlane;
    };
    const auto rightStartIt = std::partition(startIt, endIt, splitFunc);

    // create left and right child nodes
    const uint lightCountLeft  = rightStartIt - startIt;
    const uint lightCountRight = parent.getLightCount() - lightCountLeft;
    const uint startIndexLeft  = parent.getStartIndex();
    const uint startIndexRight = parent.getStartIndex() + lightCountLeft;

    // initialize nodes
    leftNode.init(startIndexLeft, mLeftEnergy, mLeftCone, mLeftBBox, lights, lightIndices, lightCountLeft);
    rightNode.init(startIndexRight, mRightEnergy, mRightCone, mRightBBox, lights, lightIndices, lightCountRight);
}



/// ------------------------------------------- LightTreeNode ----------------------------------------------------------

void LightTreeNode::calcEnergyVariance(uint lightCount, uint startIndex, const Light* const* lights, 
                                       const std::vector<uint>& lightIndices)
{
    mEnergyMean = mEnergy / lightCount;
    for (uint i = 0; i < lightCount; ++i) {
        const Light* light = lights[lightIndices[startIndex + i]];

        float diff = luminance(light->getRadiance()) - mEnergyMean;
        diff *= diff;
        diff /= lightCount;
        mEnergyVariance += diff;
    }
}


void LightTreeNode::init(uint lightCount, uint startIndex, const Light* const* lights, 
                         const std::vector<uint>& lightIndices)
{
    MNRY_ASSERT(lightCount > 0);

    mStartIndex = startIndex;
    mLightCount = lightCount;

    for (uint i = 0; i < lightCount; ++i) {
        const Light* light = lights[lightIndices[startIndex + i]];

        // add to the total energy
        // "maximum radiance emitted in some direction integrated over the emitting area"
        mEnergy += luminance(light->getRadiance());

        // extend the bounding box
        mBBox.extend(light->getBounds());

        // extend the orientation cone
        const LightTreeCone cone(light);
        mCone = combineCones(mCone, cone);
    }

    calcEnergyVariance(lightCount, startIndex, lights, lightIndices);
}

void LightTreeNode::init(uint startIndex, float energy, const LightTreeCone& cone, const BBox3f& bbox,
                const Light* const* lights, const std::vector<uint>& lightIndices, uint lightCount)
{
    MNRY_ASSERT(lightCount > 0);
    
    mStartIndex = startIndex;
    mLightCount = lightCount;

    // We already have the energy, bbox, and cone from the split candidate
    mEnergy = energy;
    mBBox = bbox;
    mCone = cone;

    calcEnergyVariance(lightCount, startIndex, lights, lightIndices);
}

float LightTreeNode::importance(const Vec3f& p, const Vec3f& n, const LightTreeNode& sibling, bool cullLights) const
{
    if (mLightCount == 0) return 0.f;

    // calculations used throughout
    const Vec3f bboxToPt = p - center(mBBox);
    const Vec3f dirToPt = normalize(bboxToPt);
    const float dSqr = lengthSqr(bboxToPt);           // squared distance to point
    const float rSqr = lengthSqr(mBBox.size() / 2.f); // squared radius of sphere circumscribing the bounding box

    // Find uncertainty angle
    float sinThetaU, cosThetaU;
    calcSinCosThetaU(dSqr, rSqr, &sinThetaU, &cosThetaU);

    const float distanceTerm = calcDistanceTerm(p, sibling, dSqr, rSqr);
    const float geometricTerm = calcGeometricTerm(p, cosThetaU, sinThetaU, dirToPt);
    const float materialTerm = calcMaterialTerm(p, n, cullLights, cosThetaU, sinThetaU, dirToPt);

    return mEnergy * geometricTerm * materialTerm * distanceTerm;
}

void LightTreeNode::calcSinCosThetaU(const float dSqr, const float rSqr, float* sinTheta, float* cosTheta) const
{
    MNRY_ASSERT(rSqr > 0.f);

    // if p is inside bounding sphere, return full sphere of directions
    if (dSqr < rSqr) {
        *cosTheta = -1.f;
        *sinTheta =  0.f;
        return;
    }

    // calc sin(theta)^2
    const float sinSqrTheta = rSqr / dSqr;
    *sinTheta = scene_rdl2::math::sqrt(sinSqrTheta);

    // use the cos(theta)^2 + sin(theta)^2 = 1 identity to find cos(theta)
    *cosTheta = scene_rdl2::math::sqrt(1.f - sinSqrTheta); 
}

float LightTreeNode::calcDistanceTerm(const Vec3f& p, const LightTreeNode& sibling, 
                                      const float rSqr, const float dSqr) const
{
    const float diagLenSqr = 4.f * rSqr; // (2r)^2 = 4r^2
    Vec3f diagSibling = sibling.getBBox().size();
    const float diagLenSqrSibling = lengthSqr(diagSibling);

    // find distance to bbox
    /// NOTE: I've found that getting the min dist to the bbox (instead of the center) produces 
    /// about the same results, but is more problematic
    const float dSqrSibling = lengthSqr(p - center(sibling.getBBox()));

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

float LightTreeNode::calcGeometricTerm(const Vec3f& p, float cosThetaU, float sinThetaU, const Vec3f& dirToPt) const
{
    float cosTheta = clamp(dot(mCone.mAxis, dirToPt), -1.f, 1.f);
    if (mCone.mTwoSided) {
        cosTheta = scene_rdl2::math::abs(cosTheta);
    }
    const float sinTheta = scene_rdl2::math::sqrt(1.f - cosTheta*cosTheta);

    const float cosThetaO = mCone.mCosThetaO;
    const float sinThetaO = mCone.mSinThetaO;

    // minimum angle any emitter in the cluster will form with the direction toward the shading pt
    // max(theta - thetaO - thetaU, 0)
    // ---------------------------------------------------------------------------------------------
    // max(cos(theta - thetaO), 1)
    const float cosThetaSubtract = cosSubClamped(sinTheta, cosTheta, sinThetaO, cosThetaO);
    const float sinThetaSubtract = scene_rdl2::math::sqrt(1.f - cosThetaSubtract*cosThetaSubtract);
    // max(cos(thetaSubtract - thetaU), 1)
    const float cosThetaPrime = cosSubClamped(sinThetaSubtract, cosThetaSubtract, sinThetaU, cosThetaU);
    // ---------------------------------------------------------------------------------------------

    // if min angle of emitter is larger than the cone's emission angle, can't contribute to point
    return cosThetaPrime <= mCone.mCosThetaE ? 0.f : cosThetaPrime;
}

float LightTreeNode::calcMaterialTerm(const Vec3f& p, const Vec3f& n, bool cullLights,
                                      float cosThetaU, float sinThetaU, const Vec3f& dirToPt) const
{
    // calc angle between normal and direction to the center of the bbox (cull lights below the horizon)
    const Vec3f dirToBbox = -dirToPt;
    const float cosThetaI = clamp(dot(n, dirToBbox), -1.f, 1.f);
    const float sinThetaI = scene_rdl2::math::sqrt(1.f - cosThetaI*cosThetaI);

    // get the MINIMUM angle to the light, which is theta_i minus theta_u
    // if theta_u > theta_i, then the normal is pointing toward the bounding box (inside uncertainty angle)
    const float cosThetaIPrime = cosSubClamped(sinThetaI, cosThetaI, sinThetaU, cosThetaU);

    /// TODO: the algorithm in the paper says absdot, but we cull if dot is less than zero, 
    /// so the light sample will be invalid anyway.... look at this again 
    if (cullLights && cosThetaIPrime < 0.f) {
        return 0.f;
    }
    return scene_rdl2::math::abs(cosThetaIPrime);
}

void LightTreeNode::printLights(const std::vector<uint>& lightIndices)
{
    std::cout << "{ ";
    for (uint i = 0; i < mLightCount; ++i) {
        std::cout << lightIndices[mStartIndex + i] << ", ";
    }
    std::cout << "}\n";
}

void LightTreeNode::print()
{
    std::cout << "{\n\tmBBox sizes: " << mBBox.size()
                << "\n\ttype: " << (mLightIndex == -1 ? "cluster" : "leaf") 
                << "\n\tmStartIndex: " << mStartIndex
                << "\n\tmLightCount: " << mLightCount
                << "\n\tmRightNodeIndex: " << mRightNodeIndex 
                << "\n}" << std::endl;
}

} // end namespace pbr
} // end namespace moonray
