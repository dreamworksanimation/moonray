// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#include "AttributeTable.h"

#include <scene_rdl2/render/util/BitUtils.h>
#include <scene_rdl2/render/util/Memory.h>
#include <scene_rdl2/common/platform/HybridVaryingData.h>

namespace ispc {
extern "C" uint32_t AttributeTable_hvdValidation(bool);
}

using namespace scene_rdl2;

namespace moonray {
namespace shading {

AttributeTable::AttributeTable(const AttributeKeySet& requiredKeys,
        const AttributeKeySet& optionalKeys)
{
    int maxKey = -1;
    for (auto k : requiredKeys) {
        maxKey = math::max(maxKey, (int)k);
    }
    for (auto k : optionalKeys) {
        maxKey = math::max(maxKey, (int)k);
    }
    mNumKeys = maxKey+1;
    mKeyOffset = new int[mNumKeys];
    for (int i = 0; i < mNumKeys; i++) {
        mKeyOffset[i] = -1;
    }
    // make sure data in allocated buffer aligned
    // we sort the key by attribute size, and build key->offset table
    // with size descending order (to avoid wasted padding space)
    std::map<size_t, std::vector<AttributeKey> > attrSizeMap;
    for (auto k : requiredKeys) {
        attrSizeMap[k.getSize()].push_back(k);
    }
    for (auto k : optionalKeys) {
        attrSizeMap[k.getSize()].push_back(k);
    }
    size_t totalSize = 0;
    for (auto rit = attrSizeMap.rbegin(); rit != attrSizeMap.rend(); ++rit) {
        size_t attrSize = rit->first;
        for (auto k: rit->second) {
            mKeyOffset[k] = totalSize;
            if (k.hasDerivatives()) {
                // for an attribute f we need to store f, dfds, dfdt
                totalSize += 3 * attrSize;
                mDifferentialAttributes.push_back(k);
            } else {
                totalSize += attrSize;
            }
        }
    }
    totalSize = util::alignUp(totalSize, sizeof(float));

    mTotalSize = totalSize;
    mRequiredAttributes.insert(mRequiredAttributes.begin(),
        requiredKeys.begin(), requiredKeys.end());
    mOptionalAttributes.insert(mOptionalAttributes.begin(),
        optionalKeys.begin(), optionalKeys.end());
}

HVD_VALIDATOR(AttributeTable);

} // namespace shading
} // namespace rendering


