// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#pragma once

//
// -- Message construction APIs for send Fb (frame buffer) @ MCRT computation --
//
// This class provides APIs to construct messages for sending various different image data to
// downstream by RenderedFrame, PartialFrame and ProgressiveFrame message.
// RenderedFrame is used for single MCRT computation situation and message is directoly sent to
// frontend client.
// PartialFrame is used for multiple MCRT computation situation and message is send to merger
// computation.
// ProgressiveFrame is used for both of single MCRT (directory send to frontend client) or
// multiple MCRT (send to merger).
//

#include "ImgEncodingType.h"

#include <moonray/rendering/rndr/RenderOutputDriver.h>

#include <scene_rdl2/common/fb_util/ActivePixels.h>
#include <scene_rdl2/common/fb_util/FbTypes.h>
#include <scene_rdl2/common/fb_util/VariablePixelBuffer.h>
#include <scene_rdl2/common/grid_util/ActivePixelsArray.h>
#include <scene_rdl2/common/grid_util/LatencyLog.h>
#include <scene_rdl2/common/grid_util/PackTiles.h>
#include <scene_rdl2/common/grid_util/PackTilesPassPrecision.h>
#include <scene_rdl2/common/grid_util/Parser.h>
#include <scene_rdl2/common/math/Viewport.h>
#include <scene_rdl2/common/platform/Platform.h> // finline

namespace moonray {
    namespace rndr { class RenderContext; }
}

namespace moonray {
namespace engine_tool {

class McrtFbSender
{
public:
    using ActivePixels = scene_rdl2::fb_util::ActivePixels;
    using RenderBuffer = scene_rdl2::fb_util::RenderBuffer;
    using RenderColor = scene_rdl2::fb_util::RenderColor;
    using FloatBuffer = scene_rdl2::fb_util::FloatBuffer;
    using PixelInfoBuffer = scene_rdl2::fb_util::PixelInfoBuffer;
    using HeatMapBuffer = scene_rdl2::fb_util::HeatMapBuffer;
    using VariablePixelBuffer = scene_rdl2::fb_util::VariablePixelBuffer;

    using DataPtr = std::shared_ptr<uint8_t>;
    using MessageAddBuffFunc = std::function<void(const DataPtr &data, const size_t dataSize,
                                                  const char *aovName,
                                                  const ImgEncodingType encodingType)>;

    using Parser = scene_rdl2::grid_util::Parser;

    //
    // We have 4 different types of precision control options for PackTile encoding.
    // Basically, AUTO16 is the best option and we can achieve minimum data transfer size always.
    // Other options are mostly designed for comparison and debugging purposes.
    // The only drawback of AUTO16 is the runtime computation cost. Some AOV has
    // scene_rdl2::grid_util::CoarsePassPrecision::RUNTIME_DECISION setting and this requires
    // HDRI pixel test every time inside the encoding phase. Obviously, this test needs extra
    // overhead costs. At this moment, this overhead is in acceptable range.
    // It would be better if we add a new mode that skip RUNTIME_DECISION setting here when we can
    // use drastically increased network bandwidth in the future.
    //                                                                                                             
    enum class PrecisionControl : char {
        FULL32, // Always uses F32 for both of Coarse and Fine pass

        FULL16, // Always uses H16 if possible for both of Coarse and Fine pass.
                // However, uses F32 if minimum precision is F32

        AUTO32, // CoarsePass : Choose proper precision automatically based on the AOV data
                // FinePass   : Always uses F32

        AUTO16  // CoarsePass : Choose proper precision automatically based on the AOV data
                // FinePass   : Basically use H16. Only uses F32 if minimum precision is F32
    };

    //------------------------------

    McrtFbSender() :
        mPrecisionControl(PrecisionControl::AUTO16),
        mRoiViewportStatus(false),
        mSnapshotDeltaCoarsePass(true),
        mBeautyHDRITest(HdriTestCondition::INIT),
        mRenderBufferCoarsePassPrecision(COARSE_PASS_PRECISION_BEAUTY),
        mRenderBufferFinePassPrecision(FinePassPrecision::F32),
        mPixelInfoStatus(false),
        mPixelInfoCoarsePassPrecision(COARSE_PASS_PRECISION_PIXEL_INFO),
        mPixelInfoFinePassPrecision(FinePassPrecision::F32),
        mHeatMapStatus(false),
        mHeatMapSkipCondition(false),
        mHeatMapId(-1),
        mWeightBufferStatus(false),
        mWeightBufferSkipCondition(false),
        mWeightBufferId(-1),
        mWeightBufferCoarsePassPrecision(COARSE_PASS_PRECISION_WEIGHT),
        mWeightBufferFinePassPrecision(FinePassPrecision::F32),
        mRenderBufferOddStatus(false),
        mRenderBufferOddSkipCondition(false),
        mBeautyAuxId(-1),
        mAlphaAuxId(-1),
        mDenoiserAlbedoInputNamePtr(nullptr),
        mDenoiserNormalInputNamePtr(nullptr),
        mMin(0),
        mMax(0)
    {
        parserConfigure();
    }

    // Non-copyable
    McrtFbSender &operator = (const McrtFbSender) = delete;
    McrtFbSender(const McrtFbSender &) = delete;

    //------------------------------

    void setPrecisionControl(PrecisionControl &precisionControl) { mPrecisionControl = precisionControl; }

    // w, h are original size and not need to be as tile size aligned
    void init(const unsigned w, const unsigned h);

    void setRoiViewport(scene_rdl2::math::HalfOpenViewport &roiViewport)
    {
        mRoiViewportStatus = true;
        mRoiViewport = roiViewport;
    }
    void resetRoiViewport() { mRoiViewportStatus = false; }
    bool getRoiViewportStatus() const { return mRoiViewportStatus; }
    const scene_rdl2::math::HalfOpenViewport &getRoiViewport() const { return mRoiViewport; }

    void initPixelInfo(const bool sw); // should be called after init()
    void initRenderOutput(const rndr::RenderOutputDriver *rod); // should be called after init()

    void setMachineId(const int machineId) { mLatencyLog.setMachineId(machineId); }

    void fbReset();

    //------------------------------

    unsigned getWidth() const { return mActivePixels.getWidth(); }
    unsigned getHeight() const { return mActivePixels.getHeight(); }

    bool getPixelInfoStatus() const { return mPixelInfoStatus; }
    bool getHeatMapStatus() const { return mHeatMapStatus; }
    bool getHeatMapSkipCondition() const { return mHeatMapSkipCondition; }
    bool getWeightBufferStatus() const { return mWeightBufferStatus; }
    bool getWeightBufferSkipCondition() const { return mWeightBufferSkipCondition; }
    bool getRenderBufferOddStatus() const { return mRenderBufferOddStatus; }
    bool getRenderBufferOddSkipCondition() const { return mRenderBufferOddSkipCondition; }
    size_t getRenderOutputTotal() const { return mActivePixelsRenderOutput.size(); }

    const std::string* getDenoiserAlbedoInputNamePtr() const { return mDenoiserAlbedoInputNamePtr; }
    const std::string* getDenoiserNormalInputNamePtr() const { return mDenoiserNormalInputNamePtr; }

    //------------------------------

    // for ProgressiveFrame message
    // const bool coarsePass : only used by activePixels rec logic. You don't need to set
    // if you don't rec.
    void snapshotDelta(const rndr::RenderContext &renderContext,
                       const bool doPixelInfo, const bool doParallel, const uint32_t snapshotId,
                       std::function<bool(const std::string &bufferName)> checkOutputIntervalFunc,
                       const bool coarsePass = false);

    // for debug : record all snapshotDelta activePixels internally and dump to file
    //             This logic required huge internal memory and need to use with attention.
    //             If not using snapshotDeltaRecStart(), there is no impact to the performance.
    //             Default (and after construction) condition is rec off
    void snapshotDeltaRecStart();
    void snapshotDeltaRecStop();
    void snapshotDeltaRecReset(); // stop and reset
    // return true : created file and free internal data
    //        false : error and still keep internal data
    bool snapshotDeltaRecDump(const std::string &fileName);

    //------------------------------

    // Send to merger (multiple MCRT) or frontend client (single MCRT) by ProgressiveFrame
    void addBeautyToProgressiveFrame(const bool directToClient,
                                     MessageAddBuffFunc func);
    void addPixelInfoToProgressiveFrame(MessageAddBuffFunc func);
    void addHeatMapToProgressiveFrame(const bool directToClient,
                                      MessageAddBuffFunc func);
    void addWeightBufferToProgressiveFrame(MessageAddBuffFunc func);
    void addRenderBufferOddToProgressiveFrame(const bool directToClient,
                                              MessageAddBuffFunc func);
    void addRenderOutputToProgressiveFrame(const bool directToClient,
                                           MessageAddBuffFunc func);
    void addAuxInfoToProgressiveFrame(const std::vector<std::string> &infoDataArray,
                                      MessageAddBuffFunc func);

    void addLatencyLog(MessageAddBuffFunc func);

    //------------------------------

    uint64_t getSnapshotStartTime() const { return mLatencyLog.getTimeBase(); }

    std::string jsonPrecisionInfo() const; // return all precision info by JSON ascii
    static std::string precisionControlStr(const PrecisionControl &precisionControl);

    //------------------------------

    Parser& getParser() { return mParser; }

protected:
    using PackTilePrecision = scene_rdl2::grid_util::PackTiles::PrecisionMode;
    using PackTilePrecisionCalcFunc = std::function<PackTilePrecision()>;
    using CoarsePassPrecision = scene_rdl2::grid_util::CoarsePassPrecision;
    using FinePassPrecision = scene_rdl2::grid_util::FinePassPrecision;

    PrecisionControl mPrecisionControl;

    bool mRoiViewportStatus;
    scene_rdl2::math::HalfOpenViewport mRoiViewport;

    static CoarsePassPrecision constexpr COARSE_PASS_PRECISION_BEAUTY = CoarsePassPrecision::RUNTIME_DECISION;
    static CoarsePassPrecision constexpr COARSE_PASS_PRECISION_ALPHA = CoarsePassPrecision::UC8;
    static CoarsePassPrecision constexpr COARSE_PASS_PRECISION_PIXEL_INFO = CoarsePassPrecision::H16;
    static CoarsePassPrecision constexpr COARSE_PASS_PRECISION_WEIGHT = CoarsePassPrecision::UC8;

    bool mSnapshotDeltaCoarsePass; // coarse pass condition of last snapshotDelta()

    enum class HdriTestCondition : char {
        INIT,    // Initial condition. The test has not been completed yet
        HDRI,    // The test has been completed and the result included HDRI pixel
        NON_HDRI // The test has been completed and the result did not include HDRI pixel
    };
    HdriTestCondition mBeautyHDRITest; // result of HDRI test for beauty buffer after done snapshotDelta

    //------------------------------
    //
    // Snapshot information
    //

    // RenderBuffer (beauty/alpha) frame buffer
    ActivePixels mActivePixels;                  // active pixel mask info for beauty
    RenderBuffer mRenderBufferTiled;             // non normalized : tile aligned resolution
    FloatBuffer  mRenderBufferWeightBufferTiled; // pixel weight data : tile size aligned resolution
    CoarsePassPrecision mRenderBufferCoarsePassPrecision; // minimum packTile precision
    FinePassPrecision mRenderBufferFinePassPrecision;     // minimum packTile precision

    // PixelInfo buffer
    bool mPixelInfoStatus;
    ActivePixels mActivePixelsPixelInfo;     // active pixel mask info for pixelInfo data
    PixelInfoBuffer mPixelInfoBufferTiled;   // pixelInfo data : tile size aligned resolution
    FloatBuffer mPixelInfoWeightBufferTiled; // pixel weight data for pixelInfo : tile aligned
    CoarsePassPrecision mPixelInfoCoarsePassPrecision; // minimum packTile precision
    FinePassPrecision mPixelInfoFinePassPrecision;     // minimum packTile precision

    // HeatMap buffer
    bool mHeatMapStatus;
    bool mHeatMapSkipCondition;
    int mHeatMapId;                        // index of mRenderOutputName[]
    ActivePixels mActivePixelsHeatMap;     // active pixel mask info for heat map data
    HeatMapBuffer mHeatMapBufferTiled;     // heatMap data : tile size aligned resolution (uint64_t)
    FloatBuffer mHeatMapWeightBufferTiled; // pixel weight data for heatMap : tile aligned resolution
    FloatBuffer mHeatMapSecBufferTiled;    // convert sec only activePixels (work buffer)

    // Weight buffer
    bool mWeightBufferStatus;
    bool mWeightBufferSkipCondition;
    int mWeightBufferId;                    // index of mRenderOutputName[]
    ActivePixels mActivePixelsWeightBuffer; // active pixel mask information for weight buffer
    FloatBuffer mWeightBufferTiled;         // pixel weight data : tile size aligned resolution
    CoarsePassPrecision mWeightBufferCoarsePassPrecision; // minimum packTile precision
    FinePassPrecision mWeightBufferFinePassPrecision;     // minimum packTile precision

    // RenderBufferOdd (beautyAux/alphaAux)
    bool mRenderBufferOddStatus;
    bool mRenderBufferOddSkipCondition;
    int mBeautyAuxId;                              // index of mRenderOutputName[]
    int mAlphaAuxId;                               // index of mRenderOutputName[]
    ActivePixels mActivePixelsRenderBufferOdd;     // active pixel mask information for renderBufferOdd
    RenderBuffer mRenderBufferOddTiled;            // renderBufferOdd data : tile aligned resolution
    FloatBuffer mRenderBufferOddWeightBufferTiled; // pixel weight data : tile size aligned resolution

    // RenderOutput buffer
    std::vector<std::string> mRenderOutputName;                      // AOV buffer name
    // mRenderOutputSkipCondition[] bit mask : this is char(8bit) now but you can expand to ushort
    // if you need.
    //        |
    // 7 6 5 4 3 2 1 0
    //   ^ ^ ^ ^ ^ ^ ^
    //   | | | | | | +--- skip on/off by checkOutputInterval() test : update every snapshot call
    //   | | | | | +----- buffer is Beauty RGB and skip regular AOV operation
    //   | | | | +------- buffer is Alpha and skip regular AOV operation
    //   | | | +--------- buffer is HeatMap and skip regular AOV operation
    //   | | +----------- buffer is Weight and skip regular AOV operation
    //   | +------------- buffer is BeautyAUX and skip regular AOV operation
    //   +--------------- buffer is AlphaAUX and skip regular AOV operation
    //
    static char constexpr SKIP_CONDITION_MANUALLY_SKIP = 0x1;
    static char constexpr SKIP_CONDITION_BEAUTY_AOV = 0x2;
    static char constexpr SKIP_CONDITION_ALPHA_AOV = 0x4;
    static char constexpr SKIP_CONDITION_HEATMAP_AOV = 0x8;
    static char constexpr SKIP_CONDITION_WEIGHT_AOV = 0x10;
    static char constexpr SKIP_CONDITION_BEAUTYAUX_AOV = 0x20;
    static char constexpr SKIP_CONDITION_ALPHAAUX_AOV = 0x40;
    std::vector<char> mRenderOutputSkipCondition;              // buffer skip condition
    std::vector<ActivePixels> mActivePixelsRenderOutput;       // active pixel mask for renderOutput 
    std::vector<VariablePixelBuffer> mRenderOutputBufferTiled; // renderOutput buff : tile aligned
    std::vector<float> mRenderOutputBufferDefaultValue;        // renderOutput buff default value
    std::vector<FloatBuffer> mRenderOutputWeightBufferTiled;   // pixWeight for renderOutput tile aligned
    std::vector<char> mRenderOutputBufferScaledByWeight;       // requires scaled by weight condition.

    std::vector<int> mRenderOutputBufferOrigNumChan; // original renderOutputBuffer numChan
                                                     // regardless of using closestFilter or not.
    std::vector<char> mRenderOutputBufferClosestFilterStatus; // using closestFilter condition (bool)

    std::vector<CoarsePassPrecision> mRenderOutputBufferCoarsePassPrecision; // minimum packTile precision
    std::vector<FinePassPrecision> mRenderOutputBufferFinePassPrecision;     // minimum packTile precision

    const std::string* mDenoiserAlbedoInputNamePtr;
    const std::string* mDenoiserNormalInputNamePtr;

    //------------------------------
    //
    // ProgressiveFrame work buffer
    //
    size_t mMin, mMax; // for performance analyze. packet size min/max info
    std::string mWork; // work memory for encoding

    //------------------------------

    scene_rdl2::grid_util::LatencyLog mLatencyLog; // latency log information for performance analyze

    //------------------------------

    // This is an array of activePixels which records snapshotDelta action in particular period
    std::unique_ptr<scene_rdl2::grid_util::ActivePixelsArray> mActivePixelsArray;

    //------------------------------

    Parser mParser;

    //------------------------------

    void initHeatMap(const int heatMapId); // should call after init()
    void initWeightBuffer(const rndr::RenderOutputDriver *rod,
                          const int weightBufferId); // this function should be called after init()
    void initRenderBufferOdd(const int beautyAuxId, const int alphaAuxId); // should call after init()
    void initRenderOutputVisibilityAOV(const rndr::RenderOutputDriver *rod, const unsigned int roIdx);
    void initRenderOutputRegularAOV(const rndr::RenderOutputDriver *rod, const unsigned int roIdx,
                                    int &beautyId, int &alphaId,
                                    int &heatMapId, int &weightBufferId,
                                    int &beautyAuxId, int &alphaAuxId);
    void adjustRenderBufferFinePassPrecision(const rndr::RenderOutputDriver *rod,
                                             const int beautyId, const int alphaId,
                                             const int beautyAuxId, const int alphaAuxId);
    FinePassPrecision calcRenderOutputBufferFinePassPrecision(const rndr::RenderOutputDriver *rod,
                                                              const int roIdx) const;
    CoarsePassPrecision calcRenderOutputBufferCoarsePassPrecision(const rndr::RenderOutputDriver *rod,
                                                                  const int roIdx) const;
    PackTilePrecision calcPackTilePrecision(const CoarsePassPrecision coarsePassPrecision,
                                            const FinePassPrecision finePassPrecision,
                                            PackTilePrecisionCalcFunc runtimeDecisionFunc = nullptr) const;
    PackTilePrecision getBeautyHDRITestResult();
    bool beautyHDRITest() const;
    bool renderOutputHDRITest(const ActivePixels &activePixels,
                              const VariablePixelBuffer &buff,
                              const FloatBuffer &weightBuff) const;

    void computeSecBuffer();

    void setRenderOutputSkipCondition(const int index, bool skip);
    bool getRenderOutputSkipCondition(const int index) const {
        return (mRenderOutputSkipCondition[index] != 0x0)? true: false;
    }

    bool isRenderOutputDisable(const int index) const {
        return (mRenderOutputSkipCondition[index] & SKIP_CONDITION_MANUALLY_SKIP)? true: false;
    }
    bool isRenderOutputBeauty(const int index) const {
        return (mRenderOutputSkipCondition[index] & SKIP_CONDITION_BEAUTY_AOV)? true: false;
    }
    bool isRenderOutputAlpha(const int index) const {
        return (mRenderOutputSkipCondition[index] & SKIP_CONDITION_ALPHA_AOV)? true: false;
    }
    bool isRenderOutputHeatMap(const int index) const {
        return (mRenderOutputSkipCondition[index] & SKIP_CONDITION_HEATMAP_AOV)? true: false;
    }
    bool isRenderOutputWeightBuffer(const int index) const {
        return (mRenderOutputSkipCondition[index] & SKIP_CONDITION_WEIGHT_AOV)? true: false;
    }
    bool isRenderOutputBeautyAux(const int index) const {
        return (mRenderOutputSkipCondition[index] & SKIP_CONDITION_BEAUTYAUX_AOV)? true: false;
    }
    bool isRenderOutputAlphaAux(const int index) const {
        return (mRenderOutputSkipCondition[index] & SKIP_CONDITION_ALPHAAUX_AOV)? true: false;
    }

    finline uint8_t *duplicateWorkData();

    void timeLogStart(const uint32_t snapshotId)
    {
        mLatencyLog.start();
        mLatencyLog.setSnapshotId(snapshotId);
    }

    DataPtr makeSharedPtr(uint8_t *data) { return DataPtr(data, std::default_delete<uint8_t[]>()); }

    //------------------------------

    unsigned getNonBlackBeautyPixelTotal() const; // for debug
    unsigned getNonZeroWeightPixelTotal() const; // for debug

    bool saveRenderBufferTiledByPPM(const std::string &filename,
                                    const ActivePixels &activePixels,
                                    const scene_rdl2::fb_util::RenderBuffer &renderBuf) const; // for debug

    //------------------------------

    void parserConfigure();
    std::string showDenoiseInfo() const;
    std::string showRenderBufferPix(const unsigned sx, const unsigned sy) const;

    RenderColor getRenderBufferPix(const int x, const int y) const;

}; // McrtFbSender

finline uint8_t *
McrtFbSender::duplicateWorkData()
{
    uint8_t *data = new uint8_t[mWork.size()];
    std::memcpy(data, mWork.data(), mWork.size());
    return data;
}

} // namespace engine_tool
} // namespace moonray
