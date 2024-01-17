#pragma once

#include "Light.h"
#include "LightTree.hh"

namespace moonray {
namespace pbr {

/// -------------------------------------------------- Cone -----------------------------------------------------------
/// This struct represents the orientation cone that bounds the normals and emission falloff for a cluster of lights. 
/// We use this Cone structure to 1) decide how to cluster lights, 2) calculate the material and geometric terms for 
/// the importance heuristic.
///
/// @see (Section 4.1)

struct Cone
{
    /// Default Constructor
    Cone() 
        : mAxis(scene_rdl2::math::Vec3f(0.f)),  // central orientation axis
          mCosThetaO(0.f),                      // cosine of the angle bounding the spread of normals around the axis
          mCosThetaE(0.f),                      // cosine of the angle representing the bound on the emission falloff
          mTwoSided(false) {}                   // does this cone contain a two-sided light?

    /// Full Constructor
    Cone(const scene_rdl2::math::Vec3f& axis, float cos_theta_o, float cos_theta_e, bool isTwoSided) 
        : mAxis(axis), 
          mCosThetaO(cos_theta_o), 
          mCosThetaE(cos_theta_e),
          mTwoSided(isTwoSided) {}

    /// Copy Constructor
    Cone(const Cone& coneToCopy)
        : mAxis(coneToCopy.mAxis),
          mCosThetaO(coneToCopy.mCosThetaO),
          mCosThetaE(coneToCopy.mCosThetaE),
          mTwoSided(coneToCopy.mTwoSided) {}

    /// Constructor using light properties
    Cone(const Light* const light)
        : mAxis(light->getDirection(0.f)),
          mCosThetaO(scene_rdl2::math::cos(light->getThetaO())),
          mCosThetaE(scene_rdl2::math::cos(light->getThetaE())),
          mTwoSided(light->isTwoSided()) {}

    /// Is this Cone empty?
    bool isEmpty() const { return isZero(mAxis); }

    /// Get orientation angle in radians
    float getThetaO() const { return scene_rdl2::math::dw_acos(mCosThetaO); }

    /// Get emission angle in radians
    float getThetaE() const { return scene_rdl2::math::dw_acos(mCosThetaE); }

    void print() const
    {
        std::cout << "Cone:\n\tAxis: " << mAxis << "\n\tCosThetaO: " << mCosThetaO << "\n\tCosThetaE: " 
                  << mCosThetaE << "\n\tTwoSided? " << mTwoSided << std::endl;
    }

    CONE_MEMBERS;
};

/// Combine orientation cones a and b.
/// @see [Algorithm 1] from "Importance Sampling of Many Lights..." (Conty, Kulla)
Cone combineCones(const Cone& a, const Cone& b);


} // end namespace pbr
} // end namespace moonray
