// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file PrimitiveAttribute.cc
/// $Id$
///

#include "PrimitiveAttribute.h"

namespace moonray {
namespace shading {

bool
PrimitiveAttributeTable::hasAttribute(AttributeKey key) const
{
    return mMap.find(key) != mMap.end();
}

AttributeRate
PrimitiveAttributeTable::getRate(AttributeKey key) const
{
    auto it = mMap.find(key);
    if (it != mMap.end()) {
        MNRY_ASSERT(it->second.size() > 0);
        return it->second[0]->getRate();
    } else {
        return RATE_UNKNOWN;
    }
}

size_t
PrimitiveAttributeTable::getTimeSampleCount(AttributeKey key) const
{
    auto it = mMap.find(key);
    if (it != mMap.end()) {
        return it->second.size();
    } else {
        return 0;
    }
}

PrimitiveAttributeTable::iterator
PrimitiveAttributeTable::begin()
{
    return mMap.begin();
}

PrimitiveAttributeTable::const_iterator
PrimitiveAttributeTable::begin() const
{
    return mMap.begin();
}

PrimitiveAttributeTable::iterator
PrimitiveAttributeTable::end()
{
    return mMap.end();
}

PrimitiveAttributeTable::const_iterator
PrimitiveAttributeTable::end() const
{
    return mMap.end();
}

bool
PrimitiveAttributeTable::empty() const
{
    return mMap.empty();
}

PrimitiveAttributeTable::iterator
PrimitiveAttributeTable::find(const key_type& k)
{
    return mMap.find(k);
}

PrimitiveAttributeTable::const_iterator
PrimitiveAttributeTable::find(const key_type& k) const
{
    return mMap.find(k);
}

size_t
PrimitiveAttributeTable::erase(const key_type& key)
{
    return mMap.erase(key);
}

void PrimitiveAttributeTable::copy(PrimitiveAttributeTable& result ) const
{
    for (const auto& kv : mMap) {
        AttributeKey key = kv.first;
        switch (key.getType()) {
        case scene_rdl2::rdl2::TYPE_BOOL:
            result.mMap.emplace(key, copyAttribute(
                TypedAttributeKey<bool>(key)));
            break;
        case scene_rdl2::rdl2::TYPE_INT:
            result.mMap.emplace(key, copyAttribute(
                TypedAttributeKey<int>(key)));
            break;
        case scene_rdl2::rdl2::TYPE_LONG:
            result.mMap.emplace(key, copyAttribute(
                TypedAttributeKey<long>(key)));
            break;
        case scene_rdl2::rdl2::TYPE_FLOAT:
            result.mMap.emplace(key, copyAttribute(
                TypedAttributeKey<float>(key)));
            break;
        case scene_rdl2::rdl2::TYPE_STRING:
            result.mMap.emplace(key, copyAttribute(
                TypedAttributeKey<std::string>(key)));
            break;
        case scene_rdl2::rdl2::TYPE_RGB:
            result.mMap.emplace(key, copyAttribute(
                TypedAttributeKey<scene_rdl2::math::Color>(key)));
            break;
        case scene_rdl2::rdl2::TYPE_RGBA:
            result.mMap.emplace(key, copyAttribute(
                TypedAttributeKey<scene_rdl2::math::Color4>(key)));
            break;
        case scene_rdl2::rdl2::TYPE_VEC2F:
            result.mMap.emplace(key, copyAttribute(
                TypedAttributeKey<scene_rdl2::math::Vec2f>(key)));
            break;
        case scene_rdl2::rdl2::TYPE_VEC3F:
            result.mMap.emplace(key, copyAttribute(
                TypedAttributeKey<scene_rdl2::math::Vec3f>(key)));
            break;
        case scene_rdl2::rdl2::TYPE_MAT4F:
            result.mMap.emplace(key, copyAttribute(
                TypedAttributeKey<scene_rdl2::math::Mat4f>(key)));
            break;
        default:
            MNRY_ASSERT(false, (std::string("unsupported attribute type ") +
                std::string(attributeTypeName(key.getType())) +
                std::string(" for atttribute ") +
                std::string(key.getName())).c_str());
            break;
        }
    }
}

} // namespace shading
} // namespace moonray


