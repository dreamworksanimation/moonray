// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

//
//

#pragma once

#include <scene_rdl2/common/fb_util/VariablePixelBuffer.h>
#include <scene_rdl2/scene/rdl2/DisplayFilter.h>

namespace scene_rdl2 {

namespace math {
class Viewport;
}
}

namespace moonray {
namespace displayfilter {
class InputBuffer;
}

namespace rndr {

class Film;
class RenderOutputDriver;

// There are a few optimizations that can be considered for future work.
// 1. Internally there are local pixel buffers. There are potential areas of optimization
//    when copying / storing those pixel buffers.
// 2. DisplayFilters do not have a coarse pass, however all aovs do. Currently we extrapolate the
//    aovs then run the DisplayFilter on a full res image. That's unecessary extra computation.
class DisplayFilterDriver {

public:
    DisplayFilterDriver();
    ~DisplayFilterDriver();

    void init(rndr::Film* film,
              const rndr::RenderOutputDriver *roDriver,
              const std::vector<scene_rdl2::fb_util::Tile>* tiles,
              const uint32_t *tileIndices,
              const scene_rdl2::math::Viewport& viewport,
              unsigned int threadCount);

    bool hasDisplayFilters() const;

    // For snapshotting aovs.
    bool isAovRequired(unsigned int aovIdx) const;
    scene_rdl2::fb_util::VariablePixelBuffer * getAovBuffer(unsigned int aovIdx) const;

    // Executes the "filter" function for all DisplayFilters that need to be updated within
    // this tile.
    void runDisplayFilters(unsigned int tileIdx,
                           unsigned int threadId) const;

    // Set tile update mask to indicate that this tile needs to be updated.
    // A tile needs to be updated when new pixel samples are rendered in
    // that tile. This is not thread safe, but that is not an issue because
    // each tile is operated on by one thread at a time.
    void requestTileUpdate(unsigned int tileIdx) const;

private:
    class Impl;
    std::unique_ptr<Impl> mImpl;
};

} // namespace rndr
} // namespace moonray

