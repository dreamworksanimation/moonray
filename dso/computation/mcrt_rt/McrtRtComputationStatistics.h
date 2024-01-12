// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

//
//
#pragma once

#include <scene_rdl2/common/rec_time/RecTimeLap.h>

#include <iomanip>
#include <sstream>

namespace moonray {
namespace mcrt_rt_computation {

class McrtRtComputationStatistics
{
public:
    McrtRtComputationStatistics() {
        mLap.setName("==>> McrtRt onIdle() <<==");
        mLap.setMessageInterval(MSG_INTERVAL_SEC);

                        mLap.sectionRegistration("  breakdown {");
        mId_whole     = mLap.sectionRegistration("    whole   ");
        mId_endStart  = mLap.sectionRegistration("    endStart");
        mId_stop      = mLap.sectionRegistration("    stop    ");
        mId_snapshot  = mLap.sectionRegistration("    snapshot");
        mId_start     = mLap.sectionRegistration("    restart ");
        mId_startA    = mLap.sectionRegistration("      sectionA");
        mId_startB[0] = mLap.sectionRegistration("      sectionB");
        mId_startB[1] = mLap.sectionRegistration("        1:stopFrame() ");
        mId_startB[2] = mLap.sectionRegistration("        2:updateGeo   ");
        mId_startB[3] = mLap.sectionRegistration("        3:comiitStat  ");
        mId_startB[4] = mLap.sectionRegistration("        4:startFrame()");
        mId_startB[5] = mLap.sectionRegistration("        5:setupInfo   ");
        mId_startEnd  = mLap.sectionRegistration("    startEnd");
                        mLap.sectionRegistration("  }");

                                mLap.auxSectionRegistration("  stopFrame() breakdown {");
        mIdAux_stopFrame[0]   = mLap.auxSectionRegistration("    whole");
        mIdAux_stopFrame[1]   = mLap.auxSectionRegistration("      1:mDriver->stopFrame()  ");
        mIdAux_stopFrame[2]   = mLap.auxSectionRegistration("      2:mPbrScene->postFrame()");
        mIdAux_stopFrame[3]   = mLap.auxSectionRegistration("      3:accumulate pbr status ");
        mIdAux_stopFrame[4]   = mLap.auxSectionRegistration("      4:collectShaderStats    ");
        mIdAux_stopFrame[5]   = mLap.auxSectionRegistration("      5:reportShadingLogs()   ");
        mIdAux_stopFrame[6]   = mLap.auxSectionRegistration("      6:log & flush() staff   ");
        mIdAux_stopFrame[7]   = mLap.auxSectionRegistration("      7:reset                 ");
                                mLap.auxSectionRegistration("  }");

                                mLap.auxSectionRegistration("  startFrame() breakDown {");
        mIdAux_sfWhole        = mLap.auxSectionRegistration("    whole");
                                mLap.auxSectionRegistration("    renderPrep() breakDown {");
        mIdAux_renderPrep[ 0] = mLap.auxSectionRegistration("      whole");
        mIdAux_renderPrep[ 1] = mLap.auxSectionRegistration("        1:startUpdatePhaseOfFrame()");
        mIdAux_renderPrep[ 2] = mLap.auxSectionRegistration("        2:startRenderPrep()        ");
        mIdAux_renderPrep[ 3] = mLap.auxSectionRegistration("        3:resetStatistics()        ");
        mIdAux_renderPrep[ 4] = mLap.auxSectionRegistration("        4:reset RenderOutputDriver ");
        mIdAux_renderPrep[ 5] = mLap.auxSectionRegistration("        5:flag status update       ");
        mIdAux_renderPrep[ 6] = mLap.auxSectionRegistration("        6:resetShaderStatsAndLogs  ");
        mIdAux_renderPrep[ 7] = mLap.auxSectionRegistration("          7:loadGeometries()       ");
        mIdAux_renderPrep[ 8] = mLap.auxSectionRegistration("          8:reportGeometryMemory() ");
        mIdAux_renderPrep[ 9] = mLap.auxSectionRegistration("        9:buildMaterialAovFlags() ");
        mIdAux_renderPrep[10] = mLap.auxSectionRegistration("        A:buildGeometryExtensions()");
        mIdAux_renderPrep[11] = mLap.auxSectionRegistration("        B:resetShaderStatsAndLogs()");
        mIdAux_renderPrep[12] = mLap.auxSectionRegistration("        C:pbr statistics reset     ");
        mIdAux_renderPrep[13] = mLap.auxSectionRegistration("        D:update PBR               ");
                                mLap.auxSectionRegistration("    }");

                                mLap.auxSectionRegistration("    perticular logic {");
        mIdAux_primAttrTbl    = mLap.auxSectionRegistration("       primAttrTbl");
        mIdAux_loadProc       = mLap.auxSectionRegistration("          loadProc");
        mIdAux_tessellation   = mLap.auxSectionRegistration("      tessellation");
        mIdAux_buildBVH       = mLap.auxSectionRegistration("          buildBVH");
        mIdAux_buildProc      = mLap.auxSectionRegistration("           buildProc");
        mIdAux_rtcCommit      = mLap.auxSectionRegistration("           rtcCommit");
        mIdAux_rebuildGeo     = mLap.auxSectionRegistration("        rebuildGeo");
                                mLap.auxSectionRegistration("    }");
                                mLap.auxSectionRegistration("  }");


                                mLap.auxSectionRegistration("  frame adajust section {");
        mIdAux_frameInterval  = mLap.auxSectionRegistration("    frameInterval");
        mIdAux_overhead       = mLap.auxSectionRegistration("         overhead");
        mIdAux_active         = mLap.auxSectionRegistration("           active");
        mIdAux_pureGap        = mLap.auxSectionRegistration("          pureGap");
        mIdAux_overrun        = mLap.auxSectionRegistration("          overrun");
        mIdAux_adjust         = mLap.auxSectionRegistration("           adjust");
                                mLap.auxSectionRegistration("  }");

                               mLap.auxUInt64SectionRegistration("  value section {");
        mIdAuxL_primRayTotal = mLap.auxUInt64SectionRegistration("    primaryRay(ray/ms)");
        mIdAuxL_passesTotal  = mLap.auxUInt64SectionRegistration("           passesTotal");
        mIdAuxL_extrapTotal  = mLap.auxUInt64SectionRegistration("           extrapTotal");
                               mLap.auxUInt64SectionRegistration("  }");
    }

    rec_time::RecTimeAutoInterval geoRecvInterval; // incoming geometry message
    
    //    static const float MSG_INTERVAL_SEC = 10.0f; // sec : message display interval
    static const float MSG_INTERVAL_SEC = 200.0f; // sec : message display interval

    rec_time::RecTimeLap mLap;

    size_t mId_whole;
    size_t mId_endStart;
    size_t mId_stop;
    size_t mId_snapshot;
    size_t mId_start;
    size_t mId_startA;
    size_t mId_startB[6];
    size_t mId_startEnd;

    // stopFrame
    size_t mIdAux_stopFrame[8];

    // inside renderPrep()
    size_t mIdAux_sfWhole;
    size_t mIdAux_renderPrep[14];
    size_t mIdAux_primAttrTbl;
    size_t mIdAux_loadProc;
    size_t mIdAux_tessellation;
    size_t mIdAux_buildBVH;
    size_t mIdAux_buildProc;
    size_t mIdAux_rtcCommit;
    size_t mIdAux_rebuildGeo;

    // frame
    size_t mIdAux_frameInterval;
    size_t mIdAux_overhead;
    size_t mIdAux_active;
    size_t mIdAux_pureGap;
    size_t mIdAux_overrun;
    size_t mIdAux_adjust;

    // primary ray count
    size_t mIdAuxL_primRayTotal;
    size_t mIdAuxL_passesTotal;
    size_t mIdAuxL_extrapTotal;
};

class StatisticsPrimaryRayTotal
{
public:
    StatisticsPrimaryRayTotal() { reset(); }

    void reset() {
        mMsgTotal = 0;
        mUpdateTotal = 0;
        mTotalPrimarySample = 0;
        mSampleGroundTotal = 0.0;
    }

    inline void
    update(const size_t primarySample, const size_t msgInterval, const size_t msgTotal) {
        if (mMsgTotal >= msgTotal) return;

        mTotalPrimarySample += primarySample;
        mUpdateTotal++;
        if ((mUpdateTotal % msgInterval) == 0) {
            double sample = (double)mTotalPrimarySample / (double)msgInterval;

            mSampleGroundTotal += sample;
            double sampleAverage = mSampleGroundTotal / (double)(mMsgTotal + 1);
            
#           define _F15_5 std::setw(15) << std::fixed << std::setprecision(5)
            
            std::ostringstream ostr;
            ostr << mMsgTotal << " " << _F15_5 << sample << " ave:" << _F15_5 << sampleAverage << " primayRayTotal";
            std::cout << ostr.str().c_str() << std::endl;
            MOONRAY_LOG_INFO(ostr.str().c_str());

            mUpdateTotal = 0;
            mTotalPrimarySample = 0;
            mMsgTotal++;
        }
    }

protected:
    size_t mMsgTotal;
    size_t mUpdateTotal;
    size_t mTotalPrimarySample;
    double mSampleGroundTotal;
};

class StatisticsBusyMessage
{
public:
    StatisticsBusyMessage() : mOnMessageCounter(0), mOnIdleCounter(0) {}

    void onMessageUpdate() { mOnMessageCounter++; };
    void onIdleUpdate() {
        mOnIdleCounter++;
        if (mOnIdleCounter < mOnMessageCounter) {
            int c = mOnMessageCounter - mOnIdleCounter;

            std::ostringstream ostr;
            ostr << ">>> busy onMessage() execution:" << c;
            MOONRAY_LOG_INFO(ostr.str().c_str());
        }
        mOnIdleCounter = mOnMessageCounter; // reset onIdle counter as onMessage counter here
    }

protected:
    size_t mOnMessageCounter;
    size_t mOnIdleCounter;
};

} // namespace mcrt_rt_computation
} // namespace moonray

