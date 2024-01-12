// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file EvalAttribute.cc
/// $Id$
///

#include "EvalAttribute.h"

#include <moonray/rendering/bvh/shading/ThreadLocalObjectState.h>
#include <moonray/rendering/mcrt_common/Util.h>

using namespace scene_rdl2;

namespace moonray {
namespace shading {

namespace {

math::Vec3f
evalNormalImpl(const rdl2::Shader* const obj,
               shading::TLState *tls,
               const State& state,
               const math::Vec3f& normalMapValue,
               const float normalDial,
               const int normalMapSpace,
               bool& needsAdapting,
               float *lengthN = nullptr)
{
    const math::Vec3f &N = state.getN();

    // Check for numerical issues
    if (// Check for nan and infinity
        (std::isnan(normalMapValue.x) ||
         std::isnan(normalMapValue.y) ||
         std::isnan(normalMapValue.z)) ||
        (!std::isfinite(normalMapValue.x) ||
         !std::isfinite(normalMapValue.y) ||
         !std::isfinite(normalMapValue.z)) ||
        // We require a non-zero length vector
        math::isZero(normalMapValue)) {
        scene_rdl2::rdl2::Shader::getLogEventRegistry().log(obj, obj->getInvalidNormalMapLogEvent());
        needsAdapting = false;
        return N;
    }

    math::Vec3f result = normalMapValue;

    if (normalMapSpace == 0) { // TBN space
        // Re-center sampled normal from [0, 1] --> [-1, 1]
        // z is usually encoded from [0.5, 1.0] so that this re-centering
        // keeps it positive (in [0, 1])
        result = 2.0f * result - math::Vec3f(1.0f);
    }

    // Calculating the normal's length here is necessary for the Toksvig
    // normal AA strategy used in the Dwa materials since it must then
    // be normalized to render properly.
    if (lengthN != nullptr) {
        *lengthN = result.length();
    }

    if (normalMapSpace == 0) { // TBN space
        // We check that the vector be in the upper half plane of tangent space.
        if (result.z < 0.f) {
            scene_rdl2::rdl2::Shader::getLogEventRegistry().log(obj, obj->getInvalidNormalMapLogEvent());
            needsAdapting = false;
            return N;
        }

        // Transform from tangent space to shade space
        const math::ReferenceFrame frame(N, math::normalize(state.getdPds()));
        result = frame.localToGlobal(result);
    }

    result.normalize();

    // Linearly interpolate between the surface normal and the mapped
    // normal if our blend factor is anything other than 1.
    if (normalDial != 1.0f) {
        result = math::normalize(N + (result - N) * math::saturate(normalDial));
    }

    needsAdapting = true;
    return result;
}


} // end anonymous namespace

math::Vec3f
evalNormal(const rdl2::Shader* const obj,
           shading::TLState *tls,
           const State& state,
           const math::Vec3f& normalMapValue,
           const float normalDial,
           const int normalMapSpace,
           float *lengthN)
{
    bool needsAdapting;
    math::Vec3f result = evalNormalImpl(obj, tls, state,
                                        normalMapValue, normalDial, normalMapSpace,
                                        needsAdapting,
                                        lengthN);
    if (needsAdapting) {
        // hack the result if it is not physically plausible
        result = state.adaptNormal(result);
    }
    return result;
}

math::Vec3f
evalToonNormal(const rdl2::Shader* const obj,
               shading::TLState *tls,
               const State& state,
               const math::Vec3f& normalMapValue,
               const float normalDial,
               const int normalMapSpace,
               float *lengthN)
{
    // Intended for use by diffuse lobes from NPR materials only.
    bool needsAdapting;
    math::Vec3f result = evalNormalImpl(obj, tls, state,
                                        normalMapValue, normalDial, normalMapSpace,
                                        needsAdapting);
    if (needsAdapting) {
        // hack the result if it is not physically plausible,
        // unless the material is requesting physically implausible normals.

        // Some NPR materials that want to allow for completely arbitrary shading normals
        // can request that the integrator does not perform any light culling based on the
        // normal. In those cases, adaptToonNormal() will check the flag set on the Intersection
        // and not adapt the normal. This allows the explicit input normals to be unbent
        // and the black artifacts that would otherwise occur are mitigated because
        // there is no light culling.
        result =  state.adaptToonNormal(result);
    }
    return result;
}

/// Same as evalColor but for floats, including an additive term before
/// the multiplicative term = (mapEvalValue + preAddValue) * attributeValue
float
evalFloatWithPreAdd(const rdl2::Shader* const obj,
                    const rdl2::AttributeKey<rdl2::Float>& valueKey,
                    shading::TLState *tls,
                    const State& state,
                    float preAddValue)
{
    float result = obj->get(valueKey);
    if (!math::isZero(result)) {
        const rdl2::SceneObject* mapObj = obj->getBinding(valueKey);
        if (mapObj != nullptr) {
            const rdl2::Map* map = mapObj->asA<rdl2::Map>();
            math::Color sampleColor;
            sample(obj, map, tls, state, &sampleColor);

            float avg = (sampleColor.r + sampleColor.g + sampleColor.b) / 3.0f;
            result *= (avg + preAddValue);
        }
    }
    return result;
}


/// Same as evalColor but for Vec2f
math::Vec2f
evalVec2f(const rdl2::Shader* const obj,
          const rdl2::AttributeKey<rdl2::Vec2f>& valueKey,
          shading::TLState *tls,
          const State& state)
{
    math::Vec2f result = obj->get(valueKey);
    if (!math::isEqual(result, math::Vec2f(math::zero))) {
        const rdl2::SceneObject* mapObj = obj->getBinding(valueKey);
        if (mapObj != nullptr) {
            const rdl2::Map* map = mapObj->asA<rdl2::Map>();
            math::Color sampleColor;
            sample(obj, map, tls, state, &sampleColor);

            result[0] *= sampleColor.r;
            result[1] *= sampleColor.g;
        }
    }
    return result;
}


/// Similar to evalColor for Vec3f. Note that if the attribute is bound
/// to a map, the attribute value is ignored and is  NOT multiplied to the
/// result of the map. If the attribute is not bound to a map, the attribute
/// value is returned.
// TODO: Fix this inconsistency compared to other eval functions
math::Vec3f
evalVec3f(const rdl2::Shader* const obj,
          const rdl2::AttributeKey<rdl2::Vec3f>& valueKey,
          shading::TLState *tls,
          const State& state)
{
    math::Vec3f result = obj->get(valueKey);
    const rdl2::SceneObject* mapObj = obj->getBinding(valueKey);
    if (mapObj != nullptr) {
        const rdl2::Map* map = mapObj->asA<rdl2::Map>();
        math::Color input_vec3;
        sample(obj, map, tls, state, &input_vec3);

        result[0] = input_vec3.r;
        result[1] = input_vec3.g;
        result[2] = input_vec3.b;
    }
    return result;
}


/// Convenience function that evaluates an attribute triplet: color, factor and
/// boolean toggle. This is a very common case in material shaders (ex:
/// "diffuse color", "diffuse factor", "show diffuse").
/// This function will only try to evaluate the color attribute if the boolean
/// toggle is on and if the factor attribute is non-zero.
/// This function returns the product of the three.
math::Color
evalColorComponent(const rdl2::Shader* const obj,
                   const rdl2::AttributeKey<rdl2::Bool>& showKey,
                   const rdl2::AttributeKey<rdl2::Float>& factorKey,
                   const rdl2::AttributeKey<rdl2::Rgb>& valueKey,
                   shading::TLState *tls,
                   const State& state)
{
    math::Color result(0.0f);

    if (obj->get(showKey)) {
        float factor = obj->get(factorKey);
        if (!math::isZero(factor)) {
            result = factor * evalColor(obj, valueKey, tls, state);
        }
    }
    return result;
}

} // namespace shading 
} // namespace moonray

