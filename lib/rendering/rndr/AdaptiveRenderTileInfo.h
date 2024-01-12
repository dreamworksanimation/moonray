// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

//
//
#pragma once

#include <scene_rdl2/common/fb_util/FbTypes.h> // fb_util::Tile
#include <scene_rdl2/common/math/Viewport.h> // math::Viewport
#include <scene_rdl2/common/platform/Platform.h> // finline
#include <cstdint> // uint64_t

namespace moonray {
namespace rndr {

class AdaptiveRenderTileInfo
{ 
public:
    enum class Stage : int {
        UNIFORM_STAGE, // add same # of samples for all pixels inside tile. tile base operation
        ADAPTIVE_STAGE,
        COMPLETED
    };

    AdaptiveRenderTileInfo() :
        mStage(Stage::UNIFORM_STAGE),
        mCompletedSamples(0)
    {}

    finline void reset();

    Stage getCondition() const { return mStage; }
    void setCondition(Stage c) { mStage = c; }

    finline void update(const unsigned addedSamples) { mCompletedSamples += addedSamples; }

    // return delta completed samples of this tile from previous call.
    finline unsigned complete(const unsigned maxSamplesPerPixel);

    finline bool isCompleted() const { return (mStage == Stage::COMPLETED); }

    std::string show(const std::string &hd) const;

private:
    Stage mStage;
    unsigned mCompletedSamples; // total completed samples for this tile (= sum of all completed pix samples).
};

finline void
AdaptiveRenderTileInfo::reset()
{
    mStage = Stage::UNIFORM_STAGE;
    mCompletedSamples = 0;
}

finline unsigned
AdaptiveRenderTileInfo::complete(const unsigned maxSamplesPerPixel)
// return delta completed samples of this tile from previous call.
{
    mStage = Stage::COMPLETED;

    unsigned oldCompletedSamples = mCompletedSamples;
    mCompletedSamples = maxSamplesPerPixel * 64;
    return mCompletedSamples - oldCompletedSamples;
}

} // namespace rndr
} // namespace moonray

