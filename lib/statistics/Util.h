// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
///

#pragma once

#include <memory>
#include <string>
#include <utility>

namespace moonray_stats {

/// Split a line at a space (literally ' ') nearest the center of the string.
std::pair<std::string, std::string> splitMiddle(const std::string& s);
int computeWindowWidth();
std::string createDashTitle(std::string s);
std::string createArrowTitle(std::string s);

template <typename T, typename... Args>
inline std::unique_ptr<T> make_unique(Args&& ... args)
{
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

} // namespace moonray_stats


