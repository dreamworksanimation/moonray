// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file InstanceAttributes.h
/// $Id$
///

#pragma once

#include <moonray/rendering/bvh/shading/AttributeKey.h>
#include <moonray/rendering/bvh/shading/PrimitiveAttribute.h>
#include <moonray/rendering/bvh/shading/Xform.h>

namespace moonray {
namespace shading {

class InstanceAttributes
{
public:
    finline InstanceAttributes(
            PrimitiveAttributeTable&& primitiveAttributeTable);

    template <typename T>
    finline void getAttribute(TypedAttributeKey<T> key, char* data) const;

    template <typename T>
    finline T &getAttribute(TypedAttributeKey<T> key);

    template <typename T>
    finline const T &getAttribute(TypedAttributeKey<T> key) const;

    finline bool isSupported(AttributeKey key) const;

    size_t getMemory() const {
        return sizeof(InstanceAttributes) +
            scene_rdl2::util::getVectorElementsMemory(mKeyOffset) +
            scene_rdl2::util::getVectorElementsMemory(mConstants);
    }

    void transformAttributes(const XformSamples& xforms,
                             float shutterOpenDelta,
                             float shutterCloseDelta,
                             const std::vector<Vec3KeyType>& keyTypePairs);

private:
    finline int keyOffset(int key) const;

private:
    std::vector<int> mKeyOffset;
    Vector<char> mConstants;
};

InstanceAttributes::InstanceAttributes(
        PrimitiveAttributeTable&& primitiveAttributeTable)
{
    // make sure data in allocated buffer aligned
    // we sort the key by attribute size, and build key->offset table
    // with size descending order (to avoid wasted padding space)
    int maxKey = -1;
    std::map<size_t, std::vector<AttributeKey>> attrSizeMap;
    for (const auto& kv : primitiveAttributeTable) {
        AttributeKey key = kv.first;
        if (primitiveAttributeTable.getRate(key) == RATE_CONSTANT &&
            primitiveAttributeTable.getTimeSampleCount(key) == 1) {
            maxKey = scene_rdl2::math::max(maxKey, (int)key);
            attrSizeMap[key.getSize()].push_back(key);
        } else {
            MNRY_ASSERT(false, "InstanceAttributes only support "
                "constant rate static primitive attributes now");
        }
    }
    mKeyOffset.resize(maxKey + 1, -1);
    size_t totalSize = 0;
    for (auto rit = attrSizeMap.rbegin(); rit != attrSizeMap.rend(); ++rit) {
        size_t attrSize = rit->first;
        for (auto k: rit->second) {
            mKeyOffset[k] = totalSize;
            totalSize += attrSize;
        }
    }
    mConstants.resize(totalSize);
    // Fill in data
    for (const auto& kv : primitiveAttributeTable) {
        AttributeKey key = kv.first;
        int offset = mKeyOffset[key];
        if (mKeyOffset[key] != -1) {
            const auto& primitiveAttribute = kv.second[0];
            primitiveAttribute->fetchData(0, &mConstants[offset]);
        }
    }
}

template <typename T>
void
InstanceAttributes::getAttribute(TypedAttributeKey<T> key, char* data) const
{
    MNRY_ASSERT(isSupported(key));
    int offset = keyOffset(key);
    char* constData = const_cast<char*>(&mConstants[offset]);
    *(reinterpret_cast<T *>(data)) = *(reinterpret_cast<T *>(constData));
}

template <>
inline void
InstanceAttributes::getAttribute(TypedAttributeKey<std::string> key,
        char* data) const
{
    MNRY_ASSERT(isSupported(key));
    int offset = keyOffset(key);
    char* constData = const_cast<char*>(&mConstants[offset]);
    *(reinterpret_cast<std::string **>(data)) =
        *(reinterpret_cast<std::string **>(constData));
}

template <typename T>
T&
InstanceAttributes::getAttribute(TypedAttributeKey<T> key)
{
    MNRY_ASSERT(isSupported(key));
    int offset = keyOffset(key);
    char* constData = const_cast<char*>(&mConstants[offset]);
    return *(reinterpret_cast<T *>(constData));
}

template <typename T>
const T&
InstanceAttributes::getAttribute(TypedAttributeKey<T> key) const
{
    MNRY_ASSERT(isSupported(key));
    int offset = keyOffset(key);
    char* constData = const_cast<char*>(&mConstants[offset]);
    return *(reinterpret_cast<T *>(constData));
}

template <>
inline std::string&
InstanceAttributes::getAttribute(TypedAttributeKey<std::string> key)
{
    MNRY_ASSERT(isSupported(key));
    int offset = keyOffset(key);
    char* constData = const_cast<char*>(&mConstants[offset]);
    return *(*(reinterpret_cast<std::string **>(constData)));
}

template <>
inline const std::string&
InstanceAttributes::getAttribute(TypedAttributeKey<std::string> key) const
{
    MNRY_ASSERT(isSupported(key));
    int offset = keyOffset(key);
    char* constData = const_cast<char*>(&mConstants[offset]);
    return *(*(reinterpret_cast<std::string **>(constData)));
}

bool
InstanceAttributes::isSupported(AttributeKey key) const
{
    return 0 <= key && key < static_cast<int>(mKeyOffset.size()) &&
        mKeyOffset[key] != -1;
}

int
InstanceAttributes::keyOffset(int key) const
{
    return mKeyOffset[key];
}

} // namespace shading
} // namespace moonray

