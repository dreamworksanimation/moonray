#include "LightTreeUtil.h"

using namespace scene_rdl2::math;

namespace moonray {
namespace pbr {

/// --------------------------------------------- Cone -----------------------------------------------------------------

Cone combineCones(const Cone& a, const Cone& b)
{
    if (a.isEmpty()) { return b; }
    if (b.isEmpty()) { return a; }

    // find cone with the largest orientation angle
    const auto sortedCones = std::minmax(a, b, [](const Cone& x, const Cone& y) { 
        return x.mCosThetaO > y.mCosThetaO; 
    });
    const Cone& smallerCone = sortedCones.first;
    const Cone& largerCone = sortedCones.second;

    const float largerConeThetaO = largerCone.getThetaO();
    const float smallerConeThetaO = smallerCone.getThetaO();
    const bool twoSided = a.mTwoSided || b.mTwoSided;

    // get angle between two axes
    const float theta_d = scene_rdl2::math::dw_acos(dot(largerCone.mAxis, smallerCone.mAxis));
    // get the max emission angle (min cosine)
    const float theta_e = max(largerCone.getThetaE(), smallerCone.getThetaE());

    // check if bounds for "a" already cover "b"
    if (min(theta_d + smallerConeThetaO, sPi) <= largerConeThetaO) {
        return Cone(largerCone.mAxis, largerCone.mCosThetaO, scene_rdl2::math::cos(theta_e), twoSided);
    } else {
        // generate a new cone that covers a and b
        const float theta_o = (largerConeThetaO + theta_d + smallerConeThetaO) * 0.5f;
        // if theta_o > pi, this is a sphere
        if (theta_o >= sPi) {
            return Cone(largerCone.mAxis, /* cos of Pi*/ -1, scene_rdl2::math::cos(theta_e), twoSided);
        }

        // rotate a's axis towards b's axis to get the new central axis for the cone
        const float theta_r = theta_o - largerConeThetaO;
        // axis to rotate about (axis orthogonal to a axis and b axis)
        Vec3f rotAxis = cross(largerCone.mAxis, smallerCone.mAxis);

        // if facing the same way, keep the same axis
        if (length(rotAxis) < sEpsilon) {
            return Cone(largerCone.mAxis, scene_rdl2::math::cos(theta_o), scene_rdl2::math::cos(theta_e), twoSided);
        }
        rotAxis = normalize(rotAxis);

        // rotation a's axis around rotAxis by theta_r
        const Vec3f axis = scene_rdl2::math::cos(theta_r) * largerCone.mAxis + 
                           scene_rdl2::math::sin(theta_r) * cross(rotAxis, largerCone.mAxis);
        return Cone(normalize(axis), scene_rdl2::math::cos(theta_o), scene_rdl2::math::cos(theta_e), twoSided);
    }
}


/// ------------------------------------------- Node -------------------------------------------------------------------

void Node::calcEnergyVariance(uint lightCount, uint startIndex, const Light* const* lights, 
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


void Node::init(uint lightCount, uint startIndex, const Light* const* lights, const std::vector<uint>& lightIndices)
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
        const Cone cone(light);
        mCone = combineCones(mCone, cone);
    }

    calcEnergyVariance(lightCount, startIndex, lights, lightIndices);
}

void Node::init(uint startIndex, float energy, const Cone& cone, const BBox3f& bbox,
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

} // end namespace pbr
} // end namespace moonray
