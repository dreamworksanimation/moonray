// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

//
//
#include "AdaptiveRenderTileInfo.h"

#include <sstream>

namespace moonray {
namespace rndr {

std::string    
AdaptiveRenderTileInfo::show(const std::string &hd) const
{
    static constexpr char const *conditionStr[] = {"UNIFORM_STAGE", "ADAPTIVE_STAGE", "COMPLETED" };

    std::ostringstream ostr;
    ostr << hd << "AdaptiveRenderTileInfo {\n";
    ostr << hd << "  mStage:" << conditionStr[(int)mStage] << '\n';
    ostr << hd << "  mCompletedSamples:" << mCompletedSamples << '\n';
    ostr << hd << "}";
    return ostr.str();
}

} // namespace rndr
} // namespace moonray

