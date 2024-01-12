// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file AttributeTable
/// $Id$
///

#pragma once

#include <moonray/rendering/bvh/shading/AttributeKey.h>
#include <moonray/rendering/bvh/shading/ispc/AttributeTable.hh>

namespace moonray {
namespace shading {

//----------------------------------------------------------------------------

// The AttributeTable is a list of required and optional attributes needed
// by a Material/Volume/Displacement.  It lives in the RootShader, which is
// the base class of all Materials/Volumes/Displacements.
// Not to be confused with PrimitiveAttributeTable which contains the actual
// attribute values.

class AttributeTable
{
public:
    AttributeTable(const AttributeKeySet& requiredKeys,
                   const AttributeKeySet& optionalKeys);

    finline ~AttributeTable() {
        delete [] mKeyOffset;
    }

    finline bool requests(AttributeKey key) const {
        return (key >= 0) && (key < mNumKeys) && mKeyOffset[key] != -1;
    }

    finline int keyOffset(AttributeKey key) const {
        MNRY_ASSERT(key >= 0 && key < mNumKeys);
        int k = mKeyOffset[key];
        MNRY_ASSERT(k != -1);
        return k;
    }

    finline const int *getKeyOffsets() const {
        return mKeyOffset;
    }

    finline int getAttributesSize() const {
        return mTotalSize;
    }

    finline int getNumKeys() const {
        return mNumKeys;
    }

    finline const std::vector<AttributeKey>& getRequiredAttributes() const {
        return mRequiredAttributes;
    }

    finline const std::vector<AttributeKey>& getOptionalAttributes() const {
        return mOptionalAttributes;
    }

    finline const std::vector<AttributeKey>& getDifferentialAttributes() const {
        return mDifferentialAttributes;
    }

    finline bool requestDerivatives() const {
        return !mDifferentialAttributes.empty();
    }

    // HVD validation
    static uint32_t hvdValidation(bool verbose) { SHADING_ATTRIBUTE_TABLE_VALIDATION(VLEN); }

private:
    SHADING_ATTRIBUTE_TABLE_MEMBERS;
};

//----------------------------------------------------------------------------

} // namespace shading
} // namespace rendering



