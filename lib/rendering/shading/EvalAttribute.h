// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file EvalAttribute.h
/// $Id$
///

#pragma once

#include <moonray/rendering/shading/EvalShader.h>

#include <moonray/rendering/bvh/shading/State.h>
#include <scene_rdl2/common/math/ReferenceFrame.h>
#include <scene_rdl2/common/math/MathUtil.h>
#include <scene_rdl2/scene/rdl2/rdl2.h>

namespace moonray {
namespace shading {

///
/// This header contains functions that shaders can use to evaluate their
/// attributes. These are high-level functions that trigger bound map shader
/// evaluations via low-level API calls (see: EvalShader.h).
///

namespace {

/// Helper function to validate normal inputs
finline const scene_rdl2::rdl2::NormalMap*
getNormalMap(const scene_rdl2::rdl2::Shader* const obj,
             const scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::SceneObject*>& valueKey,
             const float dialValue)
{
    const scene_rdl2::rdl2::SceneObject* sceneObject = obj->get(valueKey);
    if (sceneObject == nullptr || scene_rdl2::math::isZero(dialValue)) {
        return nullptr;
    }

    return sceneObject->asA<scene_rdl2::rdl2::NormalMap>();
}
} // end anonymous namespace


//---------------------------------------------------------------------------

/// Evaluate an object color attribute given its key. If the attribute is bound
/// to a map, the map shader is sampled and the result multiplied to the attribute
/// value. Otherwise only the attribute value is returned.
finline scene_rdl2::math::Color
evalColor(const scene_rdl2::rdl2::Shader* const obj,
          const scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Rgb>& valueKey,
          shading::TLState *tls,
          const State& state)
{
    scene_rdl2::math::Color result = obj->get(valueKey);
    if (!scene_rdl2::math::isBlack(result)) {
        const scene_rdl2::rdl2::SceneObject* mapObj = obj->getBinding(valueKey);
        if (mapObj != nullptr) {
            const scene_rdl2::rdl2::Map* map = mapObj->asA<scene_rdl2::rdl2::Map>();
            scene_rdl2::math::Color sampleColor;
            sample(obj, map, tls, state, &sampleColor);

            result *= sampleColor;
        }
    }
    return result;
}

scene_rdl2::math::Vec3f
evalNormal(const scene_rdl2::rdl2::Shader* const obj,
           shading::TLState *tls,
           const State& state,
           const scene_rdl2::math::Vec3f& normalMapValue,
           const float normalDial,
           const int normalMapSpace,
           float *lengthN = nullptr);

scene_rdl2::math::Vec3f
evalToonNormal(const scene_rdl2::rdl2::Shader* const obj,
               shading::TLState *tls,
               const State& state,
               const scene_rdl2::math::Vec3f& normalMapValue,
               const float normalDial,
               const int normalMapSpace,
               float *lengthN = nullptr);

/// Evaluate normal mapping, given an object input normal attribute key and the
/// corresponding input normal dial attribute key. The normal map is assumed to
/// be expressed in tangent space (frame around the shading normal). Colors in
/// R and G are re-centered from [0..1] to [-0.5..0.5].
/// The input normal dial blends the resulting normal between the shading normal
/// (0.0) and the normal-mapped normal (1.0). If the attribute is not bound to a
/// map, the shading normal is returned.
/// NOTE: This function can be safely removed after support for moonshine 5 and 6
/// has ended
finline scene_rdl2::math::Vec3f
evalNormal(const scene_rdl2::rdl2::Shader* const obj,
           const scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Vec3f>& valueKey,
           const scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float>& dialKey,
           const scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Int>& spaceKey,
           shading::TLState *tls,
           const State& state,
           float& lengthN)
{
    const scene_rdl2::math::Vec3f &N = state.getN();

    const scene_rdl2::rdl2::SceneObject* sceneObject = obj->getBinding(valueKey);
    if (sceneObject == nullptr) {
        return N;
    }

    float normalMapDial = obj->get(dialKey);
    if (scene_rdl2::math::isZero(normalMapDial)) {
        return N;
    }

    const scene_rdl2::rdl2::Map* map = sceneObject->asA<scene_rdl2::rdl2::Map>();
    scene_rdl2::math::Color normalMapColor;
    sample(obj, map, tls, state, &normalMapColor);

    scene_rdl2::math::Vec3f normalMapValue(normalMapColor.r,
                               normalMapColor.g,
                               normalMapColor.b);

    int normalMapSpace = obj->get(spaceKey);
    return evalNormal(obj,
                      tls,
                      state,
                      normalMapValue,
                      normalMapDial,
                      normalMapSpace,
                      &lengthN);
}

/// NOTE: This function can be safely removed after support for moonshine 5 and 6
/// has ended
finline scene_rdl2::math::Vec3f
evalNormal(const scene_rdl2::rdl2::Shader* const obj,
           const scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Vec3f>& valueKey,
           const scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float>& dialKey,
           const scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Int>& spaceKey,
           shading::TLState *tls,
           const State& state)
{
    float lengthN;
    return evalNormal(obj, valueKey, dialKey, spaceKey, tls, state, lengthN);
}

/// Evaluate normal mapping, given a NormalMap attribute key and the
/// corresponding input normal dial value. The normal map is assumed to
/// be expressed in render space.
/// The input normal dial blends the resulting normal between the shading normal
/// (0.0) and the normal-mapped normal (1.0). If the attribute does not point to
/// a NormalMap, the shading normal is returned.
finline scene_rdl2::math::Vec3f
evalNormal(const scene_rdl2::rdl2::Shader* const obj,
           const scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::SceneObject*>& valueKey,
           const float dialValue,
           shading::TLState *tls,
           const State& state,
           float *lengthN = nullptr)
{
    const scene_rdl2::rdl2::NormalMap* normalMap = getNormalMap(obj, valueKey, dialValue);
    if (normalMap == nullptr)
    {
        return state.getN();
    }

    scene_rdl2::math::Vec3f normalMapValue;
    sampleNormal(obj, normalMap, tls, state, &normalMapValue);

    return evalNormal(obj, tls, state,
                      normalMapValue, dialValue, 1 /* render space */,
                      lengthN);
}

finline scene_rdl2::math::Vec3f
evalToonNormal(const scene_rdl2::rdl2::Shader* const obj,
               const scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::SceneObject*>& valueKey,
               const float dialValue,
               shading::TLState *tls,
               const State& state,
               float *lengthN = nullptr)
{
    const scene_rdl2::rdl2::NormalMap* normalMap = getNormalMap(obj, valueKey, dialValue);
    if (normalMap == nullptr)
    {
        return state.getN();
    }

    scene_rdl2::math::Vec3f normalMapValue;
    sampleNormal(obj, normalMap, tls, state, &normalMapValue);

    return evalToonNormal(obj, tls, state,
                          normalMapValue, dialValue, 1 /* render space */,
                          lengthN);
}

/// Same as evalColor but for floats
finline float
evalFloat(const scene_rdl2::rdl2::Shader* const obj,
          const scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float>& valueKey,
          shading::TLState *tls,
          const State& state)
{
    float result = obj->get(valueKey);
    if (!scene_rdl2::math::isZero(result)) {
        const scene_rdl2::rdl2::SceneObject* mapObj = obj->getBinding(valueKey);
        if (mapObj != nullptr) {
            const scene_rdl2::rdl2::Map* map = mapObj->asA<scene_rdl2::rdl2::Map>();
            scene_rdl2::math::Color sampleColor;
            sample(obj, map, tls, state, &sampleColor);

            float avg = (sampleColor.r + sampleColor.g + sampleColor.b) / 3.0f;
            result *= avg;
        }
    }
    return result;
}



/// Same as evalColor but for floats, including an additive term before
/// the multiplicative term = (mapEvalValue + preAddValue) * attributeValue
float
evalFloatWithPreAdd(const scene_rdl2::rdl2::Shader* const obj,
                    const scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float>& valueKey,
                    shading::TLState *tls,
                    const State& state,
                    float preAddValue);

/// Same as evalColor but for Vec2f
scene_rdl2::math::Vec2f
evalVec2f(const scene_rdl2::rdl2::Shader* const obj,
          const scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Vec2f>& valueKey,
          shading::TLState *tls,
          const State& state);

/// Similar to evalColor for Vec3f. Note that if the attribute is bound
/// to a map, the attribute value is ignored and is  NOT multiplied to the
/// result of the map. If the attribute is not bound to a map, the attribute
/// value is returned.
// TODO: Fix this inconsistency compared to other eval functions
scene_rdl2::math::Vec3f
evalVec3f(const scene_rdl2::rdl2::Shader* const obj,
          const scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Vec3f>& valueKey,
          shading::TLState *tls,
          const State& state);

/// Convenience function that evaluates an attribute triplet: color, factor and
/// boolean toggle. This is a very common case in material shaders (ex:
/// "diffuse color", "diffuse factor", "show diffuse").
/// This function will only try to evaluate the color attribute if the boolean
/// toggle is on and if the factor attribute is non-zero.
/// This function returns the product of the three.
scene_rdl2::math::Color
evalColorComponent(const scene_rdl2::rdl2::Shader* const obj,
                   const scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Bool>& showKey,
                   const scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float>& factorKey,
                   const scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Rgb>& valueKey,
                   shading::TLState *tls,
                   const State& state);

} // namespace shading 
} // namespace moonray

