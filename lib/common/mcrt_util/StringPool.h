// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file StringPool.h
/// $Id$
///


#pragma once

#include <scene_rdl2/common/platform/Platform.h>
#include <tbb/concurrent_unordered_map.h>
#include <string>

namespace moonray {
namespace util {

class StringPool
{
public:
    StringPool() = default;
    ~StringPool();

    StringPool(const StringPool&) = delete;
    StringPool& operator=(const StringPool&) = delete;

    const std::string* get(const std::string& s);

private:
    typedef tbb::concurrent_unordered_map<std::string, std::string*> StringMap;

private:
    StringMap mStringMap;
};

finline const std::string* StringPool::get(const std::string& s)
{
    auto it = mStringMap.find(s);
    if (it != mStringMap.end()) {
        return it->second;
    } else {
        std::string* newString = new std::string(s);  
        std::pair<std::string, std::string*> item(s, newString);
        std::pair<StringMap::iterator, bool> result = mStringMap.insert(item);
        if (result.second) {
            return newString;
        } else {
            delete newString;
            return mStringMap[s];
        }
    }
}

// TODO remove this temporary singleton-like interface when we have
// primitive manager to hold a instance of StringPool
StringPool& getStringPool();

//---------------------------------------------------------------------------

} // namespace util
} // namespace moonray

