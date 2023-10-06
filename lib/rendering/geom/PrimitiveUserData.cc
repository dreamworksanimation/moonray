// Copyright 2023 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file PrimitiveUserData.cc

#include "PrimitiveUserData.h"

#include <scene_rdl2/scene/rdl2/UserData.h>
#include <scene_rdl2/render/logging/logging.h>

using scene_rdl2::logging::Logger;

namespace moonray {
namespace geom {

AttributeRate
pickRate(
    const scene_rdl2::rdl2::SceneObject* object,
    const std::string& keyName,
    size_t size0,
    size_t size1,
    const geom::RateCounts& rates)
{
    if (size0 != size1) {
        Logger::warn(object->getName(), '.', keyName, ": time samples not equal size, ", size0, " != ", size1);
        if (size1 < size0) size0 = size1;
    }
    return pickRate(object, keyName, size0, rates);
}

AttributeRate
pickRate(
    const scene_rdl2::rdl2::SceneObject* object,
    const std::string& keyName,
    size_t size,
    const geom::RateCounts& rates)
{
    if (size == 0) {
        Logger::warn(object->getName(), '.', keyName, ": invalid size 0");
        return AttributeRate::RATE_UNKNOWN;
    } else if (size == 1) {
        return AttributeRate::RATE_CONSTANT;
    } else if (size == rates.faceVaryingCount) {
        return AttributeRate::RATE_FACE_VARYING;
    } else if (size == rates.partCount) {
        return AttributeRate::RATE_PART;
    } else if (size == rates.vertexCount) {
        return AttributeRate::RATE_VERTEX;
    } else if (size == rates.varyingCount) {
        return AttributeRate::RATE_VARYING;
    } else if (size == rates.uniformCount) {
        return AttributeRate::RATE_UNIFORM;
    }
    // Pick one that fits. Tried in assumed largest->smallest order. Some geometry
    // may produce a different order but it is probably ok that the interpolation
    // guess is not the closest one. Also 1 always turns into constant even if others
    // have counts of 1.
    size_t best;
    AttributeRate rate;
    if (rates.faceVaryingCount > 1 && size > rates.faceVaryingCount) {
        best = rates.faceVaryingCount;
        rate = AttributeRate::RATE_FACE_VARYING;
    } else if (rates.vertexCount > 1 && size > rates.vertexCount) {
        best = rates.vertexCount;
        rate = AttributeRate::RATE_VERTEX;
    } else if (rates.varyingCount > 1 && size > rates.varyingCount) {
        best = rates.varyingCount;
        rate = AttributeRate::RATE_VARYING;
    } else if (rates.uniformCount > 1 && size > rates.uniformCount) {
        best = rates.uniformCount;
        rate = AttributeRate::RATE_UNIFORM;
    } else if (rates.partCount > 1 && size > rates.partCount) {
        best = rates.partCount;
        rate = AttributeRate::RATE_PART;
    } else {
        best = 1;
        rate = AttributeRate::RATE_CONSTANT;
    }
    Logger::warn(object->getName(), '.', keyName, ": invalid size ", size, " truncated to ", best);
    return rate;
}

bool sizeCheck(
    const scene_rdl2::rdl2::SceneObject* object,
    const std::string& keyName,
    size_t size,
    size_t correctSize)
{
    if (size == correctSize) return true;
    if (size) {
        Logger::warn(object->getName(), '.', keyName, ": invalid size ", size, " should be ", correctSize);
    }
    return false;
}


void
processArbitraryData(
    const scene_rdl2::rdl2::SceneObject* geometry,
    const scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::SceneObjectVector> attributeKey,
    shading::PrimitiveAttributeTable& primitiveAttributeTable,
    const geom::RateCounts& rates,
    bool useFirstFrame,
    bool useSecondFrame)
{
    const scene_rdl2::rdl2::SceneObjectVector& arbitraryData = geometry->get(attributeKey);
    for (auto sceneObject : arbitraryData) {
        const scene_rdl2::rdl2::UserData* userData = sceneObject->asA<scene_rdl2::rdl2::UserData>();
        if (!userData) {
            continue;
        }

        if (userData->hasBoolData()) {
            shading::TypedAttributeKey<scene_rdl2::rdl2::Bool> key(userData->getBoolKey());
            // bool vector is a std::deque
            const scene_rdl2::rdl2::BoolVector& constData = userData->getBoolValues();
            std::vector<scene_rdl2::rdl2::Bool> data(constData.begin(), constData.end());
            primitiveAttributeTable.addAttribute(
                key,
                pickRate(geometry, key.getName(), data.size(), rates),
                std::move(data));
        }

        if (userData->hasIntData()) {
            shading::TypedAttributeKey<scene_rdl2::rdl2::Int> key(userData->getIntKey());
            scene_rdl2::rdl2::IntVector data = userData->getIntValues();
            primitiveAttributeTable.addAttribute(
                key,
                pickRate(geometry, key.getName(), data.size(), rates),
                std::move(data));
        }

        {
            std::vector<scene_rdl2::rdl2::FloatVector> samples;
            if (useFirstFrame && userData->hasFloatData0()) {
                samples.push_back(userData->getFloatValues0());
            }
            if (useSecondFrame && userData->hasFloatData1()) {
                samples.push_back(userData->getFloatValues1());
            }
            if (!samples.empty()) {
                shading::TypedAttributeKey<float> key(userData->getFloatKey());
                size_t size0 = samples[0].size();
                size_t size1 = samples.size() > 1 ? samples[1].size() : size0;
                primitiveAttributeTable.addAttribute(
                    key,
                    pickRate(geometry, key.getName(), size0, size1, rates),
                    std::move(samples));
            }
        }

        if (userData->hasStringData()) {
            shading::TypedAttributeKey<std::string> key(userData->getStringKey());
            scene_rdl2::rdl2::StringVector data = userData->getStringValues();
            primitiveAttributeTable.addAttribute(
                key,
                pickRate(geometry, key.getName(), data.size(), rates),
                std::move(data));
        }

        {
            std::vector<scene_rdl2::rdl2::RgbVector> samples;
            if (useFirstFrame && userData->hasColorData0()) {
                samples.push_back(userData->getColorValues0());
            }
            if (useSecondFrame && userData->hasColorData1()) {
                samples.push_back(userData->getColorValues1());
            }
            if (!samples.empty()) {
                shading::TypedAttributeKey<scene_rdl2::rdl2::Rgb> key(userData->getColorKey());
                size_t size0 = samples[0].size();
                size_t size1 = samples.size() > 1 ? samples[1].size() : size0;
                primitiveAttributeTable.addAttribute(
                    key,
                    pickRate(geometry, key.getName(), size0, size1, rates),
                    std::move(samples));
            }
        }

        {
            std::vector<scene_rdl2::rdl2::Vec2fVector> samples;
            if (useFirstFrame && userData->hasVec2fData0()) {
                samples.push_back(userData->getVec2fValues0());
            }
            if (useSecondFrame && userData->hasVec2fData1()) {
                samples.push_back(userData->getVec2fValues1());
            }
            if (!samples.empty()) {
                shading::TypedAttributeKey<scene_rdl2::rdl2::Vec2f> key(userData->getVec2fKey());
                size_t size0 = samples[0].size();
                size_t size1 = samples.size() > 1 ? samples[1].size() : size0;
                primitiveAttributeTable.addAttribute(
                    key,
                    pickRate(geometry, key.getName(), size0, size1, rates),
                    std::move(samples));
            }
        }

        {
            std::vector<scene_rdl2::rdl2::Vec3fVector> samples;
            if (useFirstFrame && userData->hasVec3fData0()) {
                samples.push_back(userData->getVec3fValues0());
            }
            if (useSecondFrame && userData->hasVec3fData1()) {
                samples.push_back(userData->getVec3fValues1());
            }
            if (!samples.empty()) {
                shading::TypedAttributeKey<scene_rdl2::rdl2::Vec3f> key(userData->getVec3fKey());
                size_t size0 = samples[0].size();
                size_t size1 = samples.size() > 1 ? samples[1].size() : size0;
                primitiveAttributeTable.addAttribute(
                    key,
                    pickRate(geometry, key.getName(), size0, size1, rates),
                    std::move(samples));
            }
        }

        {
            std::vector<scene_rdl2::rdl2::Mat4fVector> samples;
            if (useFirstFrame && userData->hasMat4fData0()) {
                samples.push_back(userData->getMat4fValues0());
            }
            if (useSecondFrame && userData->hasMat4fData1()) {
                samples.push_back(userData->getMat4fValues1());
            }
            if (!samples.empty()) {
                shading::TypedAttributeKey<scene_rdl2::rdl2::Mat4f> key(userData->getMat4fKey());
                size_t size0 = samples[0].size();
                size_t size1 = samples.size() > 1 ? samples[1].size() : size0;
                primitiveAttributeTable.addAttribute(
                    key,
                    pickRate(geometry, key.getName(), size0, size1, rates),
                    std::move(samples));
            }
        }
    }
}

} // namespace geom
} // namespace moonray

