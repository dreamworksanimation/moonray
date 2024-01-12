// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#include "StringPool.h"

namespace moonray {
namespace util {

StringPool& getStringPool() 
{
    static StringPool sInstance;
    return sInstance;
}

StringPool::~StringPool() 
{
    for (auto& s: mStringMap) {
        delete s.second;
    }
    mStringMap.clear();
}

} // namespace util
} // namespace moonray

