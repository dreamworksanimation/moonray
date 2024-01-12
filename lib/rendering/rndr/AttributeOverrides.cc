// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#include "AttributeOverrides.h"

#include "RenderOptions.h"
#include "RenderStatistics.h"

#include <scene_rdl2/render/logging/logging.h>
#include <scene_rdl2/common/except/exceptions.h>
#include <scene_rdl2/scene/rdl2/rdl2.h>

#include <exception>
#include <string>
#include <vector>

namespace scene_rdl2 {
using logging::Logger;
}

using namespace scene_rdl2;

namespace moonray {

namespace rndr {

namespace {

template <typename T>
void
setOverrideHelper(
    scene_rdl2::rdl2::SceneContext& context,
    scene_rdl2::rdl2::SceneObject* object,
    const RenderOptions::AttributeOverride& attrOverride)
{
    scene_rdl2::rdl2::AttributeKey<T> key =
        object->getSceneClass().getAttributeKey<T>(attrOverride.mAttribute);

    scene_rdl2::rdl2::SceneObject::UpdateGuard guard(object);
    if (!attrOverride.mValue.empty()) {
        object->set(key, scene_rdl2::rdl2::convertFromString<T>(attrOverride.mValue));
    }
    if (!attrOverride.mBinding.empty() && key.isBindable()) {
        object->setBinding(key, context.getSceneObject(attrOverride.mBinding));
    }
}

template <>
void
setOverrideHelper<scene_rdl2::rdl2::SceneObject*>(
    scene_rdl2::rdl2::SceneContext& context,
    scene_rdl2::rdl2::SceneObject* object,
    const RenderOptions::AttributeOverride& attrOverride)
{
    scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::SceneObject*> key =
        object->getSceneClass().getAttributeKey<scene_rdl2::rdl2::SceneObject*>(attrOverride.mAttribute);

    scene_rdl2::rdl2::SceneObject::UpdateGuard guard(object);
    if (!attrOverride.mValue.empty()) {
        object->set(key, context.getSceneObject(attrOverride.mValue));
    }
    if (!attrOverride.mBinding.empty() && key.isBindable()) {
        object->setBinding(key, context.getSceneObject(attrOverride.mBinding));
    }
}

template <>
void
setOverrideHelper<scene_rdl2::rdl2::SceneObjectVector>(
    scene_rdl2::rdl2::SceneContext& context,
    scene_rdl2::rdl2::SceneObject* object,
    const RenderOptions::AttributeOverride& attrOverride)
{
    scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::SceneObjectVector> key =
        object->getSceneClass().getAttributeKey<scene_rdl2::rdl2::SceneObjectVector>(attrOverride.mAttribute);
    scene_rdl2::rdl2::StringVector objNames =
        scene_rdl2::rdl2::convertFromString<scene_rdl2::rdl2::StringVector>(attrOverride.mValue);
    scene_rdl2::rdl2::SceneObjectVector objPointers;

    for (auto iter = objNames.begin(); iter != objNames.end(); ++iter) {
        objPointers.push_back(context.getSceneObject(*iter));
    }

    scene_rdl2::rdl2::SceneObject::UpdateGuard guard(object);
    if (!attrOverride.mValue.empty()) {
        object->set(key, objPointers);
    }
    if (!attrOverride.mBinding.empty() && key.isBindable()) {
        object->setBinding(key, context.getSceneObject(attrOverride.mBinding));
    }
}

void
setOverride(scene_rdl2::rdl2::SceneContext& context,
            const RenderOptions::AttributeOverride& attrOverride)
{
    scene_rdl2::rdl2::SceneObject* object = context.getSceneObject(attrOverride.mObject);

    switch (object->getSceneClass().getAttribute(attrOverride.mAttribute)->getType()) {
    case scene_rdl2::rdl2::TYPE_BOOL:
        return setOverrideHelper<scene_rdl2::rdl2::Bool>(context, object, attrOverride);

    case scene_rdl2::rdl2::TYPE_INT:
        return setOverrideHelper<scene_rdl2::rdl2::Int>(context, object, attrOverride);

    case scene_rdl2::rdl2::TYPE_LONG:
        return setOverrideHelper<scene_rdl2::rdl2::Long>(context, object, attrOverride);

    case scene_rdl2::rdl2::TYPE_FLOAT:
        return setOverrideHelper<scene_rdl2::rdl2::Float>(context, object, attrOverride);

    case scene_rdl2::rdl2::TYPE_DOUBLE:
        return setOverrideHelper<scene_rdl2::rdl2::Double>(context, object, attrOverride);

    case scene_rdl2::rdl2::TYPE_STRING:
        return setOverrideHelper<scene_rdl2::rdl2::String>(context, object, attrOverride);

    case scene_rdl2::rdl2::TYPE_RGB:
        return setOverrideHelper<scene_rdl2::rdl2::Rgb>(context, object, attrOverride);

    case scene_rdl2::rdl2::TYPE_RGBA:
        return setOverrideHelper<scene_rdl2::rdl2::Rgba>(context, object, attrOverride);

    case scene_rdl2::rdl2::TYPE_VEC2F:
        return setOverrideHelper<scene_rdl2::rdl2::Vec2f>(context, object, attrOverride);

    case scene_rdl2::rdl2::TYPE_VEC2D:
        return setOverrideHelper<scene_rdl2::rdl2::Vec2d>(context, object, attrOverride);

    case scene_rdl2::rdl2::TYPE_VEC3F:
        return setOverrideHelper<scene_rdl2::rdl2::Vec3f>(context, object, attrOverride);

    case scene_rdl2::rdl2::TYPE_VEC3D:
        return setOverrideHelper<scene_rdl2::rdl2::Vec3d>(context, object, attrOverride);

    case scene_rdl2::rdl2::TYPE_MAT4F:
        return setOverrideHelper<scene_rdl2::rdl2::Mat4f>(context, object, attrOverride);

    case scene_rdl2::rdl2::TYPE_MAT4D:
        return setOverrideHelper<scene_rdl2::rdl2::Mat4d>(context, object, attrOverride);

    case scene_rdl2::rdl2::TYPE_SCENE_OBJECT:
        return setOverrideHelper<scene_rdl2::rdl2::SceneObject*>(context, object, attrOverride);

    case scene_rdl2::rdl2::TYPE_BOOL_VECTOR:
        return setOverrideHelper<scene_rdl2::rdl2::BoolVector>(context, object, attrOverride);

    case scene_rdl2::rdl2::TYPE_INT_VECTOR:
        return setOverrideHelper<scene_rdl2::rdl2::IntVector>(context, object, attrOverride);

    case scene_rdl2::rdl2::TYPE_LONG_VECTOR:
        return setOverrideHelper<scene_rdl2::rdl2::LongVector>(context, object, attrOverride);

    case scene_rdl2::rdl2::TYPE_FLOAT_VECTOR:
        return setOverrideHelper<scene_rdl2::rdl2::FloatVector>(context, object, attrOverride);

    case scene_rdl2::rdl2::TYPE_DOUBLE_VECTOR:
        return setOverrideHelper<scene_rdl2::rdl2::DoubleVector>(context, object, attrOverride);

    case scene_rdl2::rdl2::TYPE_STRING_VECTOR:
        return setOverrideHelper<scene_rdl2::rdl2::StringVector>(context, object, attrOverride);

    case scene_rdl2::rdl2::TYPE_RGB_VECTOR:
        return setOverrideHelper<scene_rdl2::rdl2::RgbVector>(context, object, attrOverride);

    case scene_rdl2::rdl2::TYPE_RGBA_VECTOR:
        return setOverrideHelper<scene_rdl2::rdl2::RgbaVector>(context, object, attrOverride);

    case scene_rdl2::rdl2::TYPE_VEC2F_VECTOR:
        return setOverrideHelper<scene_rdl2::rdl2::Vec2fVector>(context, object, attrOverride);

    case scene_rdl2::rdl2::TYPE_VEC2D_VECTOR:
        return setOverrideHelper<scene_rdl2::rdl2::Vec2dVector>(context, object, attrOverride);

    case scene_rdl2::rdl2::TYPE_VEC3F_VECTOR:
        return setOverrideHelper<scene_rdl2::rdl2::Vec3fVector>(context, object, attrOverride);

    case scene_rdl2::rdl2::TYPE_VEC3D_VECTOR:
        return setOverrideHelper<scene_rdl2::rdl2::Vec3dVector>(context, object, attrOverride);

    case scene_rdl2::rdl2::TYPE_MAT4F_VECTOR:
        return setOverrideHelper<scene_rdl2::rdl2::Mat4fVector>(context, object, attrOverride);

    case scene_rdl2::rdl2::TYPE_MAT4D_VECTOR:
        return setOverrideHelper<scene_rdl2::rdl2::Mat4dVector>(context, object, attrOverride);

    case scene_rdl2::rdl2::TYPE_SCENE_OBJECT_VECTOR:
        return setOverrideHelper<scene_rdl2::rdl2::SceneObjectVector>(context, object, attrOverride);

    default:
        throw except::TypeError("Unknown type.");
    }
}

} // namespace

void
applyAttributeOverrides(scene_rdl2::rdl2::SceneContext& context,
                        const RenderStats& renderStats,
                        const RenderOptions& options,
                        std::stringstream& initMessages)
{
    std::vector<RenderOptions::AttributeOverride> overrides =
        options.getAttributeOverrides();
    std::string prepend = renderStats.getPrependString();
    
    for (auto iter = overrides.begin(); iter != overrides.end(); ++iter) {
        try {
            setOverride(context, *iter);
            if (!iter->mValue.empty()) {
                initMessages << "Overriding '" << iter->mAttribute <<
                             "' value with '" << iter->mValue << "'." << '\n';
            }
            if (!iter->mBinding.empty()) {
                initMessages << "Overriding '" << iter->mAttribute <<
                             "' binding with '" << iter->mBinding << "'." << '\n';
            }
        } catch (std::exception& e) {
            scene_rdl2::logging::Logger::warn("Skipping override: " , e.what());
        }
    }
}

} // namespace rndr
} // namespace moonray

