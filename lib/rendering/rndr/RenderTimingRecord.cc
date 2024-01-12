// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

//
//
#include "RenderTimingRecord.h"

#include <scene_rdl2/render/util/StrUtil.h>

#include <iomanip>

namespace moonray {
namespace rndr {

#define _D2 std::setw(2)
#define _D7 std::setw(7)
#define _F7_3 std::setw(7) << std::fixed << std::setprecision(3)

//#define DEBUG_MSG

std::string
RenderEngineTimingRecord::showSimple() const
{
    std::ostringstream ostr;
    ostr << _D2 << mId
         << " s:" << _F7_3 << getMs(TagDBL::START)
         << " r:" << _F7_3 << getMs(TagDBL::READY)
         << " i:" << _F7_3 << getMs(TagDBL::INITTEXTURING)
         << " T:" << _F7_3 << getMs(TagDBL::RENDERTILES)
         << " Q:" << _F7_3 << getMs(TagDBL::QUEUEDRAINING)
         << " f:" << _F7_3 << getMs(TagDBL::FINISHUP)
         << " g:" << _F7_3 << getMs(TagDBL::ENDGAP)
         << " A:" << _F7_3 << getMs(TagDBL::ACTIVE)
         << " t:" << _D7 << get(TagULL::PROCESSEDTILESTOTAL)
         << " p:" << _D7 << get(TagULL::PROCESSEDSAMPLETOTAL);
    return ostr.str();
}

std::string
RenderEngineTimingRecord::showDetail() const
{
    std::ostringstream ostr;
    ostr << "s:" << _F7_3 << getMs(TagDBL::START) << " ms : duration of thread spawn ~ thread start\n"
         << "r:" << _F7_3 << getMs(TagDBL::READY) << " ms : duration of thread start ~ all threads ready timing\n"
         << "i:" << _F7_3 << getMs(TagDBL::INITTEXTURING) << " ms : duration of init texturing and caancellation setup\n"
         << "T:" << _F7_3 << getMs(TagDBL::RENDERTILES) << " ms : duration of renderTiles()\n"
         << "Q:" << _F7_3 << getMs(TagDBL::QUEUEDRAINING) << " ms : duration of queue draining phase\n"
         << "f:" << _F7_3 << getMs(TagDBL::FINISHUP) << " ms : duration of finish up thread\n"
         << "g:" << _F7_3 << getMs(TagDBL::ENDGAP) << " ms : end gap duration of thread\n"
         << "A:" << _F7_3 << getMs(TagDBL::ACTIVE) << " ms : engine active duration for rendering\n"
         << "t:" << _D7 << get(TagULL::PROCESSEDTILESTOTAL) << " tiles : processed tiles total\n"
         << "p:" << _D7 << get(TagULL::PROCESSEDSAMPLETOTAL) << " samples : processed samples total";
    return ostr.str();
}

//------------------------------------------------------------------------------

std::string
RenderPassesTimingRecord::show() const
{
    std::ostringstream ostr;
    ostr << "renderPasses {\n";
    {
        ostr << scene_rdl2::str_util::addIndent(showRenderEngines()) << '\n'
             << scene_rdl2::str_util::addIndent(showRecord(mRenderEngineMax, "max")) << '\n'
             << scene_rdl2::str_util::addIndent(showRecord(mRenderEngineMin, "min")) << '\n'
             << scene_rdl2::str_util::addIndent(showRecord(mRenderEngineAverage, "average")) << '\n'
             << "      mTimeBudget:" << _F7_3 << mTimeBudget * 1000.0 << " ms : time budget of renderPasses\n"
             << "        mDuration:" << _F7_3 << mDuration * 1000.0 << " ms : duration of renderPasses\n"
             << "  mSamplesPerTile:" << mSamplesPerTile << " samples : samples per tile of renderPasses()\n"
             << "      mSamplesAll:" << mSamplesAll << " samples : total samples of renderPasses()\n"
             << "      mSampleCost:" << _F7_3 << mSampleCost * 1000.0 << " ms : sample cost of this renderPasses\n"
             << "    mOverheadCost:" << _F7_3 << mOverheadCost * 1000.0 << " ms : overhead cost of this renderPasses\n";
    }
    ostr << "}";
    return ostr.str();
}

std::string
RenderPassesTimingRecord::showRenderEngines() const
{
    std::ostringstream ostr;
    ostr << "engine timing detail (total:" << mRenderEngines.size() << ") {\n";
    for (auto &currEng : mRenderEngines) {
        ostr << scene_rdl2::str_util::addIndent(currEng.showSimple()) << '\n';
    }
    ostr << "}";
    return ostr.str();
}

std::string
RenderPassesTimingRecord::showRecord(const RenderEngineTimingRecord &rec, const std::string &msg) const
{
    std::ostringstream ostr;
    ostr << msg << " {\n";
    {
        ostr << scene_rdl2::str_util::addIndent(rec.showDetail()) << '\n';
    }
    ostr << "}";
    return ostr.str();
}

//------------------------------------------------------------------------------

void
RenderFrameTimingRecord::reset(unsigned initialNumSamplesAll)
{
    mNumFramesRendered++;

    mNumPassesRendered = 0;
    mNumSamplesPerTile = 0;
    mNumSamplesAll = static_cast<unsigned long long>(initialNumSamplesAll);

    mTotalOverheadDuration = 0.0;
    mTotalActiveDuration = 0.0;
    mAveragedSampleCost = 0.0;

    newStint();
}

void
RenderFrameTimingRecord::newStint()
{
    mRenderFrameStartTime = scene_rdl2::util::getSeconds();
    mLastFrameUpdate = (mRenderFrameEndTime < 0.0)? 0.0: mRenderFrameStartTime - mRenderFrameEndTime;

    mRenderFrameTimeBudget = 0.0;
    mRenderFrameDuration = 0.0;
        
    // We should not initialize mRenderPasses because we have to keep past mRenderPasses info
    // up to at least 64 samples per tile for next checkpoint stint

    mEstimatedSampleCost = 0.0; // estimated sample cost by estimation phase
        
    mTimeBudget = 0.0;      // current timebudget for renderPasses() (sec)
    mOverheadCost = 0.0;    // current overhead cost for single renderPasses() (sec)
    mSampleCost = 0.0;      // curreng best estimation for single sample per tile (sec)
}

std::string
RenderFrameTimingRecord::show() const
{
    std::ostringstream ostr;
    ostr << "RenderPassesTimingRecord {\n";
    {
        ostr << scene_rdl2::str_util::addIndent(showPasses()) << '\n';

        ostr << "        mLastFrameUpdate:" << _F7_3 << mLastFrameUpdate * 1000.0 << " ms\n"
             << "  mRenderFrameTimeBudget:" << _F7_3 << mRenderFrameTimeBudget * 1000.0 << " ms\n"
             << "    mRenderFrameDuration:" << _F7_3 << mRenderFrameDuration * 1000.0 << " ms\n";
            
        ostr << "      mNumFramesRendered:" << mNumFramesRendered << '\n'
             << "      mNumPassesRendered:" << mNumPassesRendered << '\n'
             << "      mNumSamplesPerTile:" << mNumSamplesPerTile << '\n'
             << "          mNumSamplesAll:" << mNumSamplesAll << '\n'
             << "    mEstimatedSampleCost:" << _F7_3 << mEstimatedSampleCost * 1000.0 << " ms\n"
             << "             mTimeBudget:" << _F7_3 << mTimeBudget * 1000.0 << " ms\n"
             << "           mOverheadCost:" << _F7_3 << mOverheadCost * 1000.0 << " ms\n"
             << "             mSampleCost:" << _F7_3 << mSampleCost * 1000.0 << " ms\n"
             << "  mTotalOverheadDuration:" << _F7_3 << mTotalOverheadDuration * 1000.0 << " ms\n"
             << "    mTotalActiveDuration:" << _F7_3 << mTotalActiveDuration * 1000.0 << " ms\n";
    }
    ostr << "}";
    return ostr.str();
}

std::string
RenderFrameTimingRecord::showSimple() const
{
    double overrun = mRenderFrameDuration - mRenderFrameTimeBudget;

    std::ostringstream ostr;
    ostr << "f:" << std::setw(3) << mNumFramesRendered
         << " last:" << _F7_3 << mLastFrameUpdate * 1000.0 << " ms"
         << " budget:" << _F7_3 << mRenderFrameTimeBudget * 1000.0 << " ms"
         << " over:" << _F7_3 << overrun * 1000.0 << " ms"
         << " sample:" << mNumSamplesPerTile
         << " all:" << mNumSamplesAll;
    return ostr.str();
}

std::string
RenderFrameTimingRecord::showPasses() const
{
    std::ostringstream ostr;
#ifdef KEEP_ALL_PASSES_INFO
    ostr << "renderPasses (keep all passes) total=" << mNumPassesRendered << " {\n";
    for (size_t i = 0; i < mNumPassesRendered; ++i) {
        std::ostringstream ostr2;
        ostr2 << ">>> Pass:" << i << ' ' << mRenderPasses[i].show();
        ostr << scene_rdl2::str_util::addIndent(ostr2.str()) << '\n';
    }
    ostr << "}";
#else  // else KEEP_ALL_PASSES_INFO
    ostr << "renderPasses (keep current pass only) {\n"; {
        ostr << scene_rdl2::str_util::addIndent(mRenderPasses[0].show()) << '\n';
    }
    ostr << "}";
#endif // end !KEEP_ALL_PASSES_INFO
    return ostr.str();
}

void    
RenderFrameTimingRecord::updateRenderPassesTable(const unsigned newSamplesPerTile,
                                                 const unsigned requestedKeepSamplesPerTile)
//
// Keep passes at least last requestedKeepSamplesPerTile samples per tile.
//
{
    if (newSamplesPerTile > requestedKeepSamplesPerTile || !mRenderPasses.size()) {
        // We don't need old passes information or mRenderPasses is empty
        mRenderPasses.resize(1);
        mRenderPasses.shrink_to_fit();

    } else {
        // find last passes we need to keep
        size_t keepTotal = 0;
        size_t totalSamplesPerTile = newSamplesPerTile;
        for (size_t i = 0; i < mRenderPasses.size(); ++i) {
            keepTotal = i + 1;
            totalSamplesPerTile += mRenderPasses[i].getSamplesPerTile();
            if (totalSamplesPerTile >= requestedKeepSamplesPerTile) {
                break;
            }
        }

        // remove unused items (i.e. more than 64 samples)
        mRenderPasses.resize(keepTotal + 1);
        mRenderPasses.shrink_to_fit();

        // shift all items in the tables and make room for current pass
        for (size_t i = mRenderPasses.size() - 1; i > 0; --i) {
            mRenderPasses[i] = mRenderPasses[i - 1];
        }
    }

    mRenderPasses[0].init();

#ifdef DEBUG_MSG
    std::cerr << ">> RenderTimingRecord.cc updateRenderPassesTable() v:" << newSamplesPerTile << " (";
    size_t total = 0;
    for (size_t i = 0; i < mRenderPasses.size(); ++i) {
        if (i == 0) {
            std::cerr << newSamplesPerTile;
            total += newSamplesPerTile;
        } else {
            std::cerr << ' ' << mRenderPasses[i].getSamplesPerTile();
            total += mRenderPasses[i].getSamplesPerTile();
        }
    }
    std::cerr << ") total:" << total << std::endl;
#endif // end DEBUG_MSG
}

} // namespace rndr
} // namespace moonray

