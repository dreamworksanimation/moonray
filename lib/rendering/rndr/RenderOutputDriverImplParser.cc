// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

//
//
#include "RenderOutputDriverImpl.h"

#include <scene_rdl2/common/grid_util/Arg.h>

namespace moonray {
namespace rndr {

void
RenderOutputDriver::Impl::parserConfigure()
{
    using Arg = scene_rdl2::grid_util::Arg;

    mParser.description("RenderOutputDriver command");
    mParser.opt("denoiseInfo", "", "show denoise related info",
                [&](Arg& arg) -> bool {
                    return arg.msg(showDenoiseInfo() + '\n');
                });
}
    
std::string
RenderOutputDriver::Impl::showDenoiseInfo() const
{
    auto getName = [&](int id) -> std::string {
        if (id < 0 || static_cast<int>(mEntries.size()) <= id) return std::string();
        return mEntries[id]->mRenderOutput->getName();
    };

    std::ostringstream ostr;
    ostr << "denoiseInfo {\n"
         << "  mDenoiserAlbedoInput:" << mDenoiserAlbedoInput
         << " name:" << getName(mDenoiserAlbedoInput) << '\n'
         << "  mDenoiserNormalInput:" << mDenoiserNormalInput
         << " name:" << getName(mDenoiserNormalInput) << '\n'
         << "}";
    return ostr.str();
}

} // namespace rndr
} // namespace moonray

