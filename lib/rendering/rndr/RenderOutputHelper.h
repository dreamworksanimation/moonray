// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file RenderOutputHelper.h

#pragma once

#include "RenderOutputDriver.h"
#include <moonray/rendering/pbr/core/Aov.h>

namespace moonray {

namespace pbr {
class AovSchema;
} // namespace pbr

namespace rndr {

/// Crawl all internal renderOutput AOV buffers and calls one of the user defined function based on the AOV type
/// for every AOV buffer. This template is useful if you would like to do something for every AOVs inside
/// renderOutputDriver.
/// @param funcNonActive execute function if AOV buffer is non active
/// @param funcVisibility execute function if AOV buffer is Visibility type
/// @param funcAOV execute function if AOV buffer is regular AOV type (and non of above)
template <typename FUNC_NONACTIVE,
          typename FUNC_VISIBILITY,
          typename FUNC_AOV>
void
crawlAllRenderOutput(const RenderOutputDriver& outputDriver,
                     FUNC_NONACTIVE funcNonActive,
                     FUNC_VISIBILITY funcVisibility,
                     FUNC_AOV funcAOV)
{
    for (unsigned int roIdx = 0; roIdx < outputDriver.getNumberOfRenderOutputs(); ++roIdx) {
        switchAovType(outputDriver, roIdx, funcNonActive, funcVisibility, funcAOV);
    }
}

/// Useful template if you need to process something for AOV based on AOV's type.
/// roIdx is used to specify one of AOV buffer inside RenderOutputDriver and you need to specify
/// all possible functions to execute based on AOV type.
/// @param roIdx return of getRenderOutputIndx() this defines current AOV buffer which need to process
/// @param funcNonActive execute function if this AOV is non active
/// @param funcVisibility execute function if this AOV is Visibility type
/// @param funcAOV execute function if this AOV is regular AOV type (and non of above)
template <typename FUNC_NONACTIVE,
          typename FUNC_VISIBILITY,
          typename FUNC_AOV>
void
switchAovType(const RenderOutputDriver& outputDriver,
              const int roIdx,
              FUNC_NONACTIVE funcNonActive,
              FUNC_VISIBILITY funcVisibility,
              FUNC_AOV funcAOV)
//
// roIdx is return of getRenderOutputIndx().
// 
{
    const pbr::AovSchema &schema = outputDriver.getAovSchema();

    const int aovIdx = outputDriver.getAovBuffer(roIdx);
    if (aovIdx < 0) {
        // non active AOV
        funcNonActive(outputDriver.getRenderOutput(roIdx));
    } else if (outputDriver.isVisibilityAov(roIdx)) {
        // Visibility AOV
        funcVisibility(aovIdx);
    } else {
        // regular AOV
        funcAOV(aovIdx);
    }
}

} // namespace rndr
} // namespace moonray
