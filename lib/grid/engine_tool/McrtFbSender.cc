// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#include "McrtFbSender.h"

#include <moonray/rendering/mcrt_common/Clock.h>
#include <moonray/rendering/rndr/RenderContext.h>
#include <moonray/rendering/pbr/core/Aov.h>

#include <scene_rdl2/common/fb_util/PixelBufferUtilsGamma8bit.h>
#include <scene_rdl2/common/fb_util/SparseTiledPixelBuffer.h>
#include <scene_rdl2/common/grid_util/PackTiles.h>
#include <scene_rdl2/common/grid_util/ProgressiveFrameBufferName.h>
#include <scene_rdl2/render/util/StrUtil.h>
#include <scene_rdl2/scene/rdl2/RenderOutput.h>

#include <fstream>

namespace moonray {
namespace engine_tool {

void
McrtFbSender::init(const unsigned w, const unsigned h)
//
// w and h are original size and not need to be as tile size aligned
//
{
    mActivePixels.init(w, h);
    {
        unsigned tileAlignedWidth = mActivePixels.getAlignedWidth();
        unsigned tileAlignedHeight = mActivePixels.getAlignedHeight();
        mRenderBufferTiled.init(tileAlignedWidth, tileAlignedHeight);
        mRenderBufferWeightBufferTiled.init(tileAlignedWidth, tileAlignedHeight);
        mRenderBufferCoarsePassPrecision = COARSE_PASS_PRECISION_BEAUTY;
        mRenderBufferFinePassPrecision = FinePassPrecision::F32;
    }

    mMin = 0;
    mMax = 0;
}

void
McrtFbSender::initPixelInfo(const bool sw)
{
    mPixelInfoStatus = sw;

    if (!sw) {
        //
        // disable pixelInfo data
        //
        mActivePixelsPixelInfo.cleanUp();
        mPixelInfoBufferTiled.cleanUp();
        mPixelInfoWeightBufferTiled.cleanUp();
        mPixelInfoCoarsePassPrecision = COARSE_PASS_PRECISION_PIXEL_INFO;
        mPixelInfoFinePassPrecision = FinePassPrecision::F32;
        return;
    }

    //
    // enable pixelInfo data (You should call init() first)
    //
    unsigned width = mActivePixels.getWidth();
    unsigned height = mActivePixels.getHeight();
    mActivePixelsPixelInfo.init(width, height);
    {
        unsigned tileAlignedWidth = mActivePixelsPixelInfo.getAlignedWidth();
        unsigned tileAlignedHeight = mActivePixelsPixelInfo.getAlignedHeight();
        mPixelInfoBufferTiled.init(tileAlignedWidth, tileAlignedHeight);
        mPixelInfoBufferTiled.clear(scene_rdl2::fb_util::PixelInfo(FLT_MAX));
        mPixelInfoWeightBufferTiled.init(tileAlignedWidth, tileAlignedHeight);
    }
    mPixelInfoCoarsePassPrecision = COARSE_PASS_PRECISION_PIXEL_INFO;
    mPixelInfoFinePassPrecision = FinePassPrecision::F32;
}

void
McrtFbSender::initRenderOutput(const rndr::RenderOutputDriver *rod)
{
    if (!rod) {
        //
        // disable renderOutput
        //
        mRenderOutputName.clear();
        mRenderOutputName.shrink_to_fit();
        mRenderOutputSkipCondition.clear();
        mRenderOutputSkipCondition.shrink_to_fit();
        mActivePixelsRenderOutput.clear();
        mActivePixelsRenderOutput.shrink_to_fit();
        mRenderOutputBufferTiled.clear();
        mRenderOutputBufferTiled.shrink_to_fit();
        mRenderOutputBufferDefaultValue.clear();
        mRenderOutputBufferDefaultValue.shrink_to_fit();
        mRenderOutputWeightBufferTiled.clear();
        mRenderOutputWeightBufferTiled.shrink_to_fit();
        mRenderOutputBufferScaledByWeight.clear();
        mRenderOutputBufferScaledByWeight.shrink_to_fit();
        mRenderOutputBufferOrigNumChan.clear();
        mRenderOutputBufferOrigNumChan.shrink_to_fit();
        mRenderOutputBufferClosestFilterStatus.clear();
        mRenderOutputBufferClosestFilterStatus.shrink_to_fit();
        mRenderOutputBufferFinePassPrecision.clear();
        mRenderOutputBufferFinePassPrecision.shrink_to_fit();
        mRenderOutputBufferCoarsePassPrecision.clear();
        mRenderOutputBufferCoarsePassPrecision.shrink_to_fit();

        initHeatMap(-1);               // no heatMap buffer
        initWeightBuffer(nullptr, -1); // no weight buffer
        initRenderBufferOdd(-1, -1);   // no renderBufferOdd buffer
        return;
    }

    const unsigned int total = rod->getNumberOfRenderOutputs();
    mRenderOutputName.resize(total);
    mRenderOutputSkipCondition.resize(total);
    mActivePixelsRenderOutput.resize(total);
    mRenderOutputBufferTiled.resize(total);
    mRenderOutputBufferDefaultValue.resize(total);
    mRenderOutputWeightBufferTiled.resize(total);
    mRenderOutputBufferScaledByWeight.resize(total);
    mRenderOutputBufferOrigNumChan.resize(total);
    mRenderOutputBufferClosestFilterStatus.resize(total);
    mRenderOutputBufferCoarsePassPrecision.resize(total);
    mRenderOutputBufferFinePassPrecision.resize(total);

    const pbr::AovSchema &schema = rod->getAovSchema();

    int beautyId = -1;
    int alphaId = -1;
    int heatMapId = -1;
    int weightBufferId = -1;
    int beautyAuxId = -1;
    int alphaAuxId = -1;
    for (unsigned int roIdx = 0; roIdx < total; ++roIdx) {
        mRenderOutputName[roIdx] = rod->getRenderOutput(roIdx)->getName(); // save buffer name

        if (rod->isVisibilityAov(roIdx)) {
            //
            // Visibility AOV
            //
            initRenderOutputVisibilityAOV(rod, roIdx);
            
        } else {
            //
            // Regular AOV
            //
            initRenderOutputRegularAOV(rod, roIdx,
                                       beautyId, alphaId, heatMapId, weightBufferId, beautyAuxId, alphaAuxId);
        }
    }

    adjustRenderBufferFinePassPrecision(rod, beautyId, alphaId, beautyAuxId, alphaAuxId);

    initHeatMap(heatMapId);                       // initialize heatMap memory information
    initWeightBuffer(rod, weightBufferId);        // initialize weight buffer memory information
    initRenderBufferOdd(beautyAuxId, alphaAuxId); // initialize renderBufferOdd memory information
}

void
McrtFbSender::fbReset()
//
// only reset fb related information
//
{
    mRenderBufferTiled.clear();
    mRenderBufferWeightBufferTiled.clear();

    if (mPixelInfoStatus) {
        mPixelInfoBufferTiled.clear(scene_rdl2::fb_util::PixelInfo(FLT_MAX));
        mPixelInfoWeightBufferTiled.clear();
    }

    if (mHeatMapStatus) {
        mHeatMapBufferTiled.clear();
        mHeatMapWeightBufferTiled.clear();
        mHeatMapSecBufferTiled.clear();
    }

    if (mWeightBufferStatus) {
        mWeightBufferTiled.clear();
    }

    if (mRenderBufferOddStatus) {
        mRenderBufferOddTiled.clear();
        mRenderBufferOddWeightBufferTiled.clear();
    }

    for (size_t rodId = 0; rodId < mActivePixelsRenderOutput.size(); ++rodId) {
        if (mActivePixelsRenderOutput[rodId].isActive()) {
            mRenderOutputBufferTiled[rodId].clear(mRenderOutputBufferDefaultValue[rodId]);
            mRenderOutputWeightBufferTiled[rodId].clear();
        }
    }
}

//------------------------------------------------------------------------------

void
McrtFbSender::snapshotDelta(const rndr::RenderContext &renderContext,
                            const bool doPixelInfo, const bool doParallel, const uint32_t snapshotId,
                            std::function<bool(const std::string &bufferName)> checkOutputIntervalFunc,
                            const bool coarsePass)
//
// for ProgressiveFrame message
//
{
    mSnapshotDeltaCoarsePass = coarsePass; // condition of coarse pass or not
    mBeautyHDRITest = HdriTestCondition::INIT; // condition of HDRI test for beauty buffer

    //
    // Beauty
    //
    timeLogStart(snapshotId); // for performance analyze
    renderContext.snapshotDelta(&mRenderBufferTiled, &mRenderBufferWeightBufferTiled, mActivePixels, doParallel);
    mLatencyLog.enq(scene_rdl2::grid_util::LatencyItem::Key::SNAPSHOT_END_BEAUTY); // for performance analyze : finish snapshot

    if (mActivePixelsArray) {
        // record all activePixels info for analyzing purpose
        mActivePixelsArray->set(mActivePixels, coarsePass); // record Beauty's activePixels info
    }

    //
    // PixelInfo
    //
    if (doPixelInfo && mPixelInfoStatus) {
        mLatencyLog.enq(scene_rdl2::grid_util::LatencyItem::Key::SNAPSHOT_START_PIXELINFO); // for performance analyze
        renderContext.snapshotDeltaPixelInfo(&mPixelInfoBufferTiled,
                                             &mPixelInfoWeightBufferTiled,
                                             mActivePixelsPixelInfo,
                                             doParallel);
        mLatencyLog.enq(scene_rdl2::grid_util::LatencyItem::Key::SNAPSHOT_END_PIXELINFO); // for performance analyze
    }

    //
    // HeatMap
    //
    if (mHeatMapStatus) {
        if (checkOutputIntervalFunc(mRenderOutputName[mHeatMapId])) {
            mLatencyLog.enq(scene_rdl2::grid_util::LatencyItem::Key::SNAPSHOT_START_HEATMAP); // for performance analyze
            renderContext.snapshotDeltaHeatMap(&mHeatMapBufferTiled,
                                               &mHeatMapWeightBufferTiled,
                                               mActivePixelsHeatMap,
                                               &mHeatMapSecBufferTiled,
                                               doParallel);
            mLatencyLog.enq(scene_rdl2::grid_util::LatencyItem::Key::SNAPSHOT_END_HEATMAP); // for performance analyze
            mHeatMapSkipCondition = false;
        } else {
            mHeatMapSkipCondition = true;
        }
    }

    //
    // Weight buffer
    //
    if (mWeightBufferStatus) {
        if (checkOutputIntervalFunc(mRenderOutputName[mWeightBufferId])) {
            mLatencyLog.enq(scene_rdl2::grid_util::LatencyItem::Key::SNAPSHOT_START_WEIGHTBUFFER); // for performance analyze
            renderContext.snapshotDeltaWeightBuffer(&mWeightBufferTiled, mActivePixelsWeightBuffer, doParallel);
            mLatencyLog.enq(scene_rdl2::grid_util::LatencyItem::Key::SNAPSHOT_END_WEIGHTBUFFER); // for performance analyze
            mWeightBufferSkipCondition = false;
        } else {
            mWeightBufferSkipCondition = true;
        }
    }

    //
    // BeautyOdd buffer
    //
    if (mRenderBufferOddStatus) {
        if ((mBeautyAuxId > 0 && checkOutputIntervalFunc(mRenderOutputName[mBeautyAuxId])) ||
            (mAlphaAuxId > 0 && checkOutputIntervalFunc(mRenderOutputName[mAlphaAuxId]))) {
            mLatencyLog.enq(scene_rdl2::grid_util::LatencyItem::Key::SNAPSHOT_START_BEAUTYODD); // for performance analyze
            renderContext.snapshotDeltaRenderBufferOdd(&mRenderBufferOddTiled,
                                                       &mRenderBufferOddWeightBufferTiled,
                                                       mActivePixelsRenderBufferOdd,
                                                       doParallel);
            mLatencyLog.enq(scene_rdl2::grid_util::LatencyItem::Key::SNAPSHOT_END_BEAUTYODD); // for performance analyze
            mRenderBufferOddSkipCondition = false;            
        } else {
            mRenderBufferOddSkipCondition = true;
        }
    }

    //
    // RenderOutput
    //
    mDenoiserAlbedoInputNamePtr = nullptr;
    mDenoiserNormalInputNamePtr = nullptr;
    for (unsigned id = 0; id < mActivePixelsRenderOutput.size(); ++id) {
        // We have to update buffer skip condition at every snapshotBuffer call.
        const std::string &bufferName = mRenderOutputName[id];
        bool skipCondition = ((checkOutputIntervalFunc(bufferName))? false: true);

        setRenderOutputSkipCondition(id, skipCondition);
        if (getRenderOutputSkipCondition(id)) {
            // We don't need snapshot for
            // 1) manually disabled buffer by rate control,
            // 2) beautyRGB, beautyAlpha heatMap or weight buffer
            continue; // skip this buffer
        }

        mLatencyLog.enq(scene_rdl2::grid_util::LatencyItem::Key::SNAPSHOT_START_RENDEROUTPUT);
        bool denoiserAlbedoInput, denoiserNormalInput;
        renderContext.snapshotDeltaRenderOutput(id,
                                                &mRenderOutputBufferTiled[id],
                                                &mRenderOutputWeightBufferTiled[id],
                                                mActivePixelsRenderOutput[id],
                                                doParallel,
                                                denoiserAlbedoInput,
                                                denoiserNormalInput);
        if (denoiserAlbedoInput) mDenoiserAlbedoInputNamePtr = &bufferName;
        if (denoiserNormalInput) mDenoiserNormalInputNamePtr = &bufferName;
        mLatencyLog.enq(scene_rdl2::grid_util::LatencyItem::Key::SNAPSHOT_END_RENDEROUTPUT);
    }
}

void
McrtFbSender::snapshotDeltaRecStart()
{
    if (!mActivePixelsArray) {
        mActivePixelsArray.reset(new scene_rdl2::grid_util::ActivePixelsArray);
    }
    mActivePixelsArray->start();
}

void
McrtFbSender::snapshotDeltaRecStop()
{
    if (mActivePixelsArray) {
        mActivePixelsArray->stop();
    }
}

void
McrtFbSender::snapshotDeltaRecReset()
{
    if (mActivePixelsArray) {
        mActivePixelsArray->stop();
        mActivePixelsArray->reset();
    }
}

bool
McrtFbSender::snapshotDeltaRecDump(const std::string &fileName)
//
// only used for debug / performance analyze purpose
//
{
    if (!mActivePixelsArray) {
        return false;           // not snapshotDeltaRecStart() yet
    }
    if (mActivePixelsArray->isStart()) {
        return false;           // not stopped yet
    }
    if (!mActivePixelsArray->size()) {
        return false;           // data is empty
    }

    std::string data;
    mActivePixelsArray->encode(data);

    uint32_t machineId = mLatencyLog.getMachineId();
    std::string outName = fileName + '.' + std::to_string(machineId);

    std::ofstream fout(outName, std::ios::out | std::ios::binary | std::ios::trunc);
    if (!fout) {
        std::cerr << ">> McrtFbSender.cc snapshotDeltaRecDump() Can't open file:" << outName << std::endl;
        return false;
    }

    fout.write((const char *)data.data(), data.size());
    if (!fout) {
        std::cerr << ">> McrtFbSender.cc snapshotDeltaRecDump() Can't write data."
                  << " file:" << outName << std::endl;
        return false;
    }

    fout.close();

    mActivePixelsArray = nullptr;

    // Using std::cerr intentionally. This API is used only for debug/performance analyze purpose.
    // And this API is called inside mcrt computations and using std::cerr always guarantees
    // to bypass output logging system of arras_framework and minimise the delay to see the message.
    std::cerr << ">> McrtFbSender.cc snapshotDeltaRecDump() done" << std::endl;

    return true;
}

//------------------------------------------------------------------------------

void
McrtFbSender::addBeautyToProgressiveFrame(const bool directToClient,
                                          MessageAddBuffFunc func)
{
    static const bool sha1HashSw = false;
    size_t dataSize = 0;

    mLatencyLog.enq(scene_rdl2::grid_util::LatencyItem::Key::ENCODE_START_BEAUTY);
    {
        PackTilePrecision packTilePrecision =
            calcPackTilePrecision(mRenderBufferCoarsePassPrecision,
                                  mRenderBufferFinePassPrecision,
                                  [&]() -> PackTilePrecision { // runtimeDecisionFunc for coarse pass
                                      return getBeautyHDRITestResult();
                                  });
        mWork.clear();
        dataSize =
            scene_rdl2::grid_util::PackTiles::encode(false,
                                                     mActivePixels,
                                                     mRenderBufferTiled,
                                                     mRenderBufferWeightBufferTiled,
                                                     mWork,
                                                     packTilePrecision,
                                                     mRenderBufferCoarsePassPrecision,
                                                     mRenderBufferFinePassPrecision,
                                                     directToClient,
                                                     sha1HashSw);
    }
    mLatencyLog.enq(scene_rdl2::grid_util::LatencyItem::Key::ENCODE_END_BEAUTY);

    /* runtime verify
    if (!directToClient) {
        // verify multi-machine mode only so far
        if (!scene_rdl2::grid_util::PackTiles::
            verifyEncodeResultMultiMcrt(mWork.data(),
                                        mWork.size(),
                                        mActivePixels,
                                        mRenderBufferTiled,
                                        mRenderBufferWeightBufferTiled)) {
            std::cerr << "verify NG" << std::endl;
        } else {
            std::cerr << "verify OK" << std::endl;
        }
    }
    */

    /* construct SHA1 hash for debug
    if (sha1HashSw) {
        if (!scene_rdl2::grid_util::PackTiles::verifyDecodeHash(mWork.data(), mWork.size())) {
            std::cerr << ">> McrtFbSender.cc hashVerify NG" << std::endl;
        } else {
            std::cerr << ">> McrtFbSender.cc hashVerify OK" << std::endl;
        }
    }
    */
    /* useful debug dump
    std::cerr << scene_rdl2::grid_util::PackTiles::
                 showHash(">> progmcrt McrtFbSender.cc ", (const unsigned char *)mWork.data())
              << std::endl;
    */

    {
        func(makeSharedPtr(duplicateWorkData()),
             dataSize,
             scene_rdl2::grid_util::ProgressiveFrameBufferName::Beauty,
             ImgEncodingType::ENCODING_UNKNOWN);
    }
    mLatencyLog.enq(scene_rdl2::grid_util::LatencyItem::Key::ADDBUFFER_END_BEAUTY);
    mLatencyLog.addDataSize(dataSize);
}

void
McrtFbSender::addPixelInfoToProgressiveFrame(MessageAddBuffFunc func)
{
    static const bool sha1HashSw = false;

    if (!mPixelInfoStatus) return;

    size_t dataSize = 0;

    mLatencyLog.enq(scene_rdl2::grid_util::LatencyItem::Key::ENCODE_START_PIXELINFO);
    {
        PackTilePrecision packTilePrecision =
            calcPackTilePrecision(mPixelInfoCoarsePassPrecision, mPixelInfoFinePassPrecision);
        mWork.clear();
        dataSize =
            scene_rdl2::grid_util::PackTiles::encodePixelInfo(mActivePixelsPixelInfo,
                                                              mPixelInfoBufferTiled,
                                                              mWork,
                                                              packTilePrecision,
                                                              mPixelInfoCoarsePassPrecision,
                                                              mPixelInfoFinePassPrecision,
                                                              sha1HashSw);
    }
    mLatencyLog.enq(scene_rdl2::grid_util::LatencyItem::Key::ENCODE_END_PIXELINFO);
    {
        func(makeSharedPtr(duplicateWorkData()),
             dataSize,
             scene_rdl2::grid_util::ProgressiveFrameBufferName::PixelInfo,
             ImgEncodingType::ENCODING_UNKNOWN);
    }
    mLatencyLog.enq(scene_rdl2::grid_util::LatencyItem::Key::ADDBUFFER_END_PIXELINFO);
    mLatencyLog.addDataSize(dataSize);
}

void
McrtFbSender::addHeatMapToProgressiveFrame(const bool directToClient, MessageAddBuffFunc func)
{
    static const bool sha1HashSw = false;

    if (!mHeatMapStatus || mHeatMapSkipCondition) return;

    size_t dataSize = 0;

    mLatencyLog.enq(scene_rdl2::grid_util::LatencyItem::Key::ENCODE_START_HEATMAP);
    {
        mWork.clear();
        // no precision control for HeatMap (always uses H16 internally)
        dataSize = scene_rdl2::grid_util::PackTiles::encodeHeatMap(mActivePixelsHeatMap,
                                                       mHeatMapSecBufferTiled,
                                                       mHeatMapWeightBufferTiled,
                                                       mWork,
                                                       directToClient,
                                                       sha1HashSw);
    }
    mLatencyLog.enq(scene_rdl2::grid_util::LatencyItem::Key::ENCODE_END_HEATMAP);
    {
        const char *buffName = 0x0;
        if (mHeatMapId >= 0) {
            buffName = mRenderOutputName[mHeatMapId].c_str();
        } else {
            buffName = scene_rdl2::grid_util::ProgressiveFrameBufferName::HeatMapDefault;
        }
        func(makeSharedPtr(duplicateWorkData()), dataSize, buffName, ImgEncodingType::ENCODING_UNKNOWN);
    }
    mLatencyLog.enq(scene_rdl2::grid_util::LatencyItem::Key::ADDBUFFER_END_HEATMAP);
    mLatencyLog.addDataSize(dataSize);
}

void
McrtFbSender::addWeightBufferToProgressiveFrame(MessageAddBuffFunc func)
{
    static const bool sha1HashSw = false;
    size_t dataSize = 0;

    mLatencyLog.enq(scene_rdl2::grid_util::LatencyItem::Key::ENCODE_START_WEIGHTBUFFER);
    {
        PackTilePrecision packTilePrecision =
            calcPackTilePrecision(mWeightBufferCoarsePassPrecision, mWeightBufferFinePassPrecision);
        mWork.clear();
        dataSize =
            scene_rdl2::grid_util::PackTiles::encodeWeightBuffer(mActivePixels,
                                                                 mWeightBufferTiled,
                                                                 mWork,
                                                                 packTilePrecision,
                                                                 mWeightBufferCoarsePassPrecision,
                                                                 mWeightBufferFinePassPrecision,
                                                                 sha1HashSw);
    }
    mLatencyLog.enq(scene_rdl2::grid_util::LatencyItem::Key::ENCODE_END_WEIGHTBUFFER);
    {
        const char *buffName = 0x0;
        if (mWeightBufferId >= 0) {
            buffName = mRenderOutputName[mWeightBufferId].c_str();
        } else {
            buffName = scene_rdl2::grid_util::ProgressiveFrameBufferName::WeightDefault;
        }
        func(makeSharedPtr(duplicateWorkData()), dataSize, buffName, ImgEncodingType::ENCODING_UNKNOWN);
    }
    mLatencyLog.enq(scene_rdl2::grid_util::LatencyItem::Key::ADDBUFFER_END_WEIGHTBUFFER);
    mLatencyLog.addDataSize(dataSize);
}

void
McrtFbSender::addRenderBufferOddToProgressiveFrame(const bool directToClient,
                                                   MessageAddBuffFunc func)
{
    static const bool sha1HashSw = false;
    size_t dataSize = 0;

    mLatencyLog.enq(scene_rdl2::grid_util::LatencyItem::Key::ENCODE_START_BEAUTYODD);
    {
        // Actually we don't have {coarse,fine}PassPrecision info for renderBufferOdd.
        // And {coarse,fine}PassPrecision of renderBuffer is used instead.
        PackTilePrecision packTilePrecision =
            calcPackTilePrecision(mRenderBufferCoarsePassPrecision,
                                  mRenderBufferFinePassPrecision,
                                  [&]() -> PackTilePrecision { // runtimeDecisionFunc for coarse pass
                                      return getBeautyHDRITestResult(); // access shared beauty HDRI test result
                                  });
        mWork.clear();
        // encode RGBA(normalized) 4 channels
        dataSize =
            scene_rdl2::grid_util::PackTiles::encode(true,
                                                     mActivePixelsRenderBufferOdd,
                                                     mRenderBufferOddTiled,
                                                     mRenderBufferOddWeightBufferTiled,
                                                     mWork,
                                                     packTilePrecision,
                                                     mRenderBufferCoarsePassPrecision, // dummy value
                                                     mRenderBufferFinePassPrecision, // dummy value
                                                     directToClient,
                                                     sha1HashSw);
    }
    mLatencyLog.enq(scene_rdl2::grid_util::LatencyItem::Key::ENCODE_END_BEAUTYODD);
    {
        func(makeSharedPtr(duplicateWorkData()),
             dataSize,
             scene_rdl2::grid_util::ProgressiveFrameBufferName::RenderBufferOdd,
             ImgEncodingType::ENCODING_UNKNOWN);
    }
    mLatencyLog.enq(scene_rdl2::grid_util::LatencyItem::Key::ADDBUFFER_END_BEAUTYODD);
    mLatencyLog.addDataSize(dataSize);
}

void
McrtFbSender::addRenderOutputToProgressiveFrame(const bool directToClient,
                                                MessageAddBuffFunc func)
{
    static const bool sha1HashSw = false;

    for (size_t id = 0; id < mActivePixelsRenderOutput.size(); ++id) {
        if (getRenderOutputSkipCondition(id)) {
            //
            // Need to skip standard output operation but still need to check bit more detail
            // to support "reference" type buffer (i.e. "BeautyRGB", "BeautyAlpha" and "HeatMap")
            //
            if (!isRenderOutputDisable(id)) {
                //
                // Now this buffer is NOT manually disabled by rate control
                //
                scene_rdl2::grid_util::FbReferenceType referenceType = scene_rdl2::grid_util::FbReferenceType::UNDEF;
                if (isRenderOutputBeauty(id)) {
                    referenceType = scene_rdl2::grid_util::FbReferenceType::BEAUTY;
                } else if (isRenderOutputAlpha(id)) {
                    referenceType = scene_rdl2::grid_util::FbReferenceType::ALPHA;
                } else if (isRenderOutputHeatMap(id)) {
                    referenceType = scene_rdl2::grid_util::FbReferenceType::HEAT_MAP;
                } else if (isRenderOutputWeightBuffer(id)) {
                    referenceType = scene_rdl2::grid_util::FbReferenceType::WEIGHT;
                } else if (isRenderOutputBeautyAux(id)) {
                    referenceType = scene_rdl2::grid_util::FbReferenceType::BEAUTY_AUX;
                } else if (isRenderOutputAlphaAux(id)) {
                    referenceType = scene_rdl2::grid_util::FbReferenceType::ALPHA_AUX;
                }

                if (referenceType != scene_rdl2::grid_util::FbReferenceType::UNDEF) {
                    //
                    // AOV is reference type and need to output
                    //
                    size_t dataSize = 0;
                    mLatencyLog.enq(scene_rdl2::grid_util::LatencyItem::Key::ENCODE_START_RENDEROUTPUT);
                    {
                        mWork.clear();
                        dataSize = scene_rdl2::grid_util::PackTiles::encodeRenderOutputReference(referenceType, mWork, sha1HashSw);
                    }
                    mLatencyLog.enq(scene_rdl2::grid_util::LatencyItem::Key::ENCODE_END_RENDEROUTPUT);
                    {
                        func(makeSharedPtr(duplicateWorkData()), dataSize, mRenderOutputName[id].c_str(),
                             ImgEncodingType::ENCODING_UNKNOWN);
                    }
                    mLatencyLog.enq(scene_rdl2::grid_util::LatencyItem::Key::ADDBUFFER_END_RENDEROUTPUT);
                    mLatencyLog.addDataSize(dataSize);
                }
            }
        } else {
            //
            // AOV actual buffer data need to output
            //
            size_t dataSize = 0;

            mLatencyLog.enq(scene_rdl2::grid_util::LatencyItem::Key::ENCODE_START_RENDEROUTPUT);
            {
                PackTilePrecision packTilePrecision =
                    calcPackTilePrecision(mRenderOutputBufferCoarsePassPrecision[id],
                                          mRenderOutputBufferFinePassPrecision[id],
                                          [&]() -> PackTilePrecision { // runtimeDecisionFunc for coarse pass
                                              if (renderOutputHDRITest(mActivePixelsRenderOutput[id],
                                                                       mRenderOutputBufferTiled[id],
                                                                       mRenderOutputWeightBufferTiled[id])) {
                                                  return PackTilePrecision::H16;
                                              } else {
                                                  return PackTilePrecision::UC8;
                                              }
                                          });
                mWork.clear();
                dataSize =
                    scene_rdl2::grid_util::PackTiles::encodeRenderOutput
                    (mActivePixelsRenderOutput[id],
                     mRenderOutputBufferTiled[id],
                     mRenderOutputBufferDefaultValue[id],
                     mRenderOutputWeightBufferTiled[id],
                     mWork,
                     packTilePrecision,
                     directToClient, // noNumSampleMode
                     static_cast<bool>(mRenderOutputBufferScaledByWeight[id]), // doNormalizeMode
                     mRenderOutputBufferClosestFilterStatus[id],
                     mRenderOutputBufferOrigNumChan[id],
                     mRenderOutputBufferCoarsePassPrecision[id],
                     mRenderOutputBufferFinePassPrecision[id],
                     sha1HashSw);
            }
            mLatencyLog.enq(scene_rdl2::grid_util::LatencyItem::Key::ENCODE_END_RENDEROUTPUT);
            {
                func(makeSharedPtr(duplicateWorkData()), dataSize, mRenderOutputName[id].c_str(),
                     ImgEncodingType::ENCODING_UNKNOWN);
            }
            mLatencyLog.enq(scene_rdl2::grid_util::LatencyItem::Key::ADDBUFFER_END_RENDEROUTPUT);
            mLatencyLog.addDataSize(dataSize);
        }
    }
}

void
McrtFbSender::addAuxInfoToProgressiveFrame(const std::vector<std::string> &infoDataArray,
                                           MessageAddBuffFunc func)
//
// This function converts infoDataArray (array of strings) into simple string by ValueContainerEnq.
// Then execute the callback function with proper channel name.
//    
{
    mWork.clear();
    scene_rdl2::rdl2::ValueContainerEnq cEnq(&mWork);

    cEnq.enqStringVector(infoDataArray);
    size_t dataSize = cEnq.finalize();

    func(makeSharedPtr(duplicateWorkData()),
         dataSize,
         scene_rdl2::grid_util::ProgressiveFrameBufferName::AuxInfo,
         ImgEncodingType::ENCODING_UNKNOWN);
}

//------------------------------------------------------------------------------

void
McrtFbSender::addLatencyLog(MessageAddBuffFunc func)
{
    mLatencyLog.setName("mcrt");
    mLatencyLog.enq(scene_rdl2::grid_util::LatencyItem::Key::SEND_MSG);

    mWork.clear();              // We have to clear work
    scene_rdl2::rdl2::ValueContainerEnq vContainerEnq(&mWork);

    mLatencyLog.encode(vContainerEnq);

    size_t dataSize = vContainerEnq.finalize();

    /* useful dump info for debug.
    std::cerr << ">> progmcrt McrtFbSender.cc dataSize:" << dataSize << std::endl;
    {
        scene_rdl2::grid_util::LatencyLog tmpLog;

        scene_rdl2::rdl2::ValueContainerDeq vContainerDeq(mWork.data(), dataSize);
        tmpLog.decode(vContainerDeq);
        std::cerr << ">> progmcrt McrtFbSender.cc dataSize:" << dataSize << " {\n"
                  << tmpLog.show("  ") << '\n'
                  << "}" << std::endl;
    }
    */

    func(makeSharedPtr(duplicateWorkData()),
         dataSize,
         scene_rdl2::grid_util::ProgressiveFrameBufferName::LatencyLog,
         ImgEncodingType::ENCODING_UNKNOWN);
}

std::string    
McrtFbSender::jsonPrecisionInfo() const
//
// For debugging purposes in order to dump precision info by JSON ascii
//
{
    auto showC = [](const CoarsePassPrecision &p) -> std::string {
        return scene_rdl2::grid_util::showCoarsePassPrecision(p);
    };
    auto showF = [](const FinePassPrecision &p) -> std::string {
        return scene_rdl2::grid_util::showFinePassPrecision(p);
    };
    auto showCF = [&](const std::string &name,
                      const CoarsePassPrecision &cp, const FinePassPrecision &fp) -> std::string {
        return "\"" + name + "\":{\"Coarse\":\"" + showC(cp) + "\", \"Fine\":\"" + showF(fp) + "\"}";
    };

    std::ostringstream ostr;
    ostr << "\"McrtFbSender precision info\" : {\n"
         << "  \"PrecisionControl\":\"" << precisionControlStr(mPrecisionControl) << "\",\n"
         << "  " << showCF("RenderBuffer",
                           mRenderBufferCoarsePassPrecision, mRenderBufferFinePassPrecision) << ",\n"
         << "  " << showCF("PixelInfo",
                           mPixelInfoCoarsePassPrecision, mPixelInfoFinePassPrecision) << ",\n"
         << "  " << showCF("WeightBuffer",
                           mWeightBufferCoarsePassPrecision, mWeightBufferFinePassPrecision) << ",\n"
         << "  \"renderOutputBuffer\":[\n";
    for (size_t i = 0; i < mRenderOutputName.size(); ++i) {
        ostr << "    {"
             << showCF(mRenderOutputName[i],
                       mRenderOutputBufferCoarsePassPrecision[i], mRenderOutputBufferFinePassPrecision[i])
             << "}" << ((i < mRenderOutputName.size() - 1) ? ",\n" : "\n");
    }
    ostr << "  ]\n"
         << "}";
    return ostr.str();
}

// static function
std::string
McrtFbSender::precisionControlStr(const PrecisionControl &precisionControl)
{
    switch (precisionControl) {
    case PrecisionControl::FULL32 : return "FULL32";
    case PrecisionControl::FULL16 : return "FULL16";
    case PrecisionControl::AUTO32 : return "AUTO32";
    case PrecisionControl::AUTO16 : return "AUTO16";
    default : break;
    }
    return "?";
}

//------------------------------------------------------------------------------

void
McrtFbSender::initHeatMap(const int heatMapId)
//
// heatMapId : negative value is no heatMap
//
{
    if (heatMapId < 0) {
        //
        // disable heatMap data
        //
        mHeatMapStatus = false;
        mHeatMapId = -1;        // no heatMap

        mActivePixelsHeatMap.cleanUp();
        mHeatMapBufferTiled.cleanUp();
        mHeatMapWeightBufferTiled.cleanUp();
        mHeatMapSecBufferTiled.cleanUp();
        return;
    }

    //
    // enable heatMap data (You should call init() first)
    //
    mHeatMapStatus = true;
    mHeatMapId = heatMapId;

    unsigned width = mActivePixels.getWidth();
    unsigned height = mActivePixels.getHeight();
    mActivePixelsHeatMap.init(width, height);
    {
        unsigned tileAlignedWidth = mActivePixelsHeatMap.getAlignedWidth();
        unsigned tileAlignedHeight = mActivePixelsHeatMap.getAlignedHeight();
        mHeatMapBufferTiled.init(tileAlignedWidth, tileAlignedHeight);
        mHeatMapWeightBufferTiled.init(tileAlignedWidth, tileAlignedHeight);
        mHeatMapSecBufferTiled.init(tileAlignedWidth, tileAlignedHeight);
    }
}

void
McrtFbSender::initWeightBuffer(const rndr::RenderOutputDriver *rod,
                               const int weightBufferId)
//
// weightBufferId : negative value is no weightBuffer
//
{
    if (weightBufferId < 0) {
        //
        // disable weight buffer
        //
        mWeightBufferStatus = false;
        mWeightBufferId = -1;   // no weight buffer

        mActivePixelsWeightBuffer.cleanUp();
        mWeightBufferTiled.cleanUp();

        // We already know the coarse pass's weight value is 0.0 or 1.0. UC8 is enough.
        mWeightBufferCoarsePassPrecision = COARSE_PASS_PRECISION_WEIGHT;
        mWeightBufferFinePassPrecision = calcRenderOutputBufferFinePassPrecision(rod, weightBufferId);
        return;
    }

    //
    // enable weight buffer data (You should call init() first)
    //
    mWeightBufferStatus = true;
    mWeightBufferId = weightBufferId;

    unsigned width = mActivePixels.getWidth();
    unsigned height = mActivePixels.getHeight();
    mActivePixelsWeightBuffer.init(width, height);
    {
        unsigned tileAlignedWidth = mActivePixelsWeightBuffer.getAlignedWidth();
        unsigned tileAlignedHeight = mActivePixelsWeightBuffer.getAlignedHeight();
        mWeightBufferTiled.init(tileAlignedWidth, tileAlignedHeight);
    }

    // We already know the coarse pass's weight value is 0.0 or 1.0. UC8 is enough.
    mWeightBufferCoarsePassPrecision = COARSE_PASS_PRECISION_WEIGHT;
    mWeightBufferFinePassPrecision = calcRenderOutputBufferFinePassPrecision(rod, weightBufferId);
}

void
McrtFbSender::initRenderBufferOdd(const int beautyAuxId, const int alphaAuxId)
//
// beautyAuxId, alphaAuxId : negative value is no access to renderBufferOdd buffer
//
{
    if (beautyAuxId < 0 && alphaAuxId < 0) {
        //
        // disable renderBufferOdd buffer
        //
        mRenderBufferOddStatus = false;
        mBeautyAuxId = -1;      // no beautyAux access
        mAlphaAuxId = -1;       // no alphaAux access
        mActivePixelsRenderBufferOdd.cleanUp();
        mRenderBufferOddTiled.cleanUp();
        mRenderBufferOddWeightBufferTiled.cleanUp();

        // RenderBufferOdd's coarse/fine pass precision control should be the same as
        // mRenderBuffer{Coarse,Fine}PassPrecision.
        return;
    }

    //
    // enable renderBufferOdd buffer data (You should call init() first because this function
    // refers to mActivePixels which setup by init())
    //
    mRenderBufferOddStatus = true;
    mBeautyAuxId = beautyAuxId;
    mAlphaAuxId = alphaAuxId;

    unsigned width = mActivePixels.getWidth();
    unsigned height = mActivePixels.getHeight();
    mActivePixelsRenderBufferOdd.init(width, height);
    {
        unsigned tileAlignedWidth = mActivePixelsRenderBufferOdd.getAlignedWidth();
        unsigned tileAlignedHeight = mActivePixelsRenderBufferOdd.getAlignedHeight();
        mRenderBufferOddTiled.init(tileAlignedWidth, tileAlignedHeight);
        mRenderBufferOddWeightBufferTiled.init(tileAlignedWidth, tileAlignedHeight);
    }

    // RenderBufferOdd's coarse/fine pass precision control should be the same as
    // mRenderBuffer{Coarse,Fine}PassPrecision.
}

void
McrtFbSender::initRenderOutputVisibilityAOV(const rndr::RenderOutputDriver *rod,
                                            const unsigned int roIdx)
{
    //
    // Visibility (non Variance) AOV : size = FLOAT
    //
    mRenderOutputSkipCondition[roIdx] = 0x0; // We know this is a visibility AOV. So skip condition should be 0x0

    mRenderOutputBufferDefaultValue[roIdx] = 0.0f; // visibility AOV's default is 0.0
    mRenderOutputBufferScaledByWeight[roIdx] = static_cast<char>(false);

    unsigned width = mActivePixels.getWidth();
    unsigned height = mActivePixels.getHeight();
    unsigned tileAlignedWidth = mActivePixels.getAlignedWidth();
    unsigned tileAlignedHeight = mActivePixels.getAlignedHeight();

    mActivePixelsRenderOutput[roIdx].init(width, height);
    mRenderOutputBufferTiled[roIdx].init(VariablePixelBuffer::FLOAT, tileAlignedWidth, tileAlignedHeight);
    mRenderOutputWeightBufferTiled[roIdx].init(tileAlignedWidth, tileAlignedHeight);

    mRenderOutputBufferCoarsePassPrecision[roIdx] = CoarsePassPrecision::H16;
    mRenderOutputBufferFinePassPrecision[roIdx] =
        calcRenderOutputBufferFinePassPrecision(rod, static_cast<int>(roIdx));
}

void
McrtFbSender::initRenderOutputRegularAOV(const rndr::RenderOutputDriver *rod,
                                         const unsigned int roIdx,
                                         int &beautyId,
                                         int &alphaId,
                                         int &heatMapId,
                                         int &weightBufferId,
                                         int &beautyAuxId,
                                         int &alphaAuxId)
{
    static constexpr VariablePixelBuffer::Format fmtTbl[] = {
        VariablePixelBuffer::UNINITIALIZED,
        VariablePixelBuffer::FLOAT,  // 1 channel
        VariablePixelBuffer::FLOAT2, // 2 channels
        VariablePixelBuffer::FLOAT3, // 3 channels
        VariablePixelBuffer::FLOAT4, // 4 channels
    };

    // mRenderOutputSkipCondition[] bit mask : this is char(8bit) now but you can expand to ushort if you need.
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
    char skipST = 0x0;
    if (rod->requiresRenderBuffer(roIdx)) {
        if (rod->getRenderOutput(roIdx)->getResult() == scene_rdl2::rdl2::RenderOutput::RESULT_BEAUTY) {
            skipST = SKIP_CONDITION_BEAUTY_AOV;
            beautyId = roIdx;
        }
        if (rod->getRenderOutput(roIdx)->getResult() == scene_rdl2::rdl2::RenderOutput::RESULT_ALPHA) {
            skipST = SKIP_CONDITION_ALPHA_AOV;
            alphaId = roIdx;
        }
    } else if (rod->requiresHeatMap(roIdx)) {
        skipST = SKIP_CONDITION_HEATMAP_AOV;
        heatMapId = roIdx;    // We save heatMap id for init mem and checkOutputInterval
    } else if (rod->requiresWeightBuffer(roIdx)) {
        skipST = SKIP_CONDITION_WEIGHT_AOV;
        weightBufferId = roIdx; // We save weight buffer id for init mem and checkOutputInterval
    } else if (rod->requiresRenderBufferOdd(roIdx)) {
        // This is a case of AOV about beauty_AUX/alpha_AUX data (moonray::rndr::Film::mRenderBufOdd)
        if (rod->getRenderOutput(roIdx)->getResult() == scene_rdl2::rdl2::RenderOutput::RESULT_BEAUTY_AUX) {
            skipST = SKIP_CONDITION_BEAUTYAUX_AOV;
            beautyAuxId = roIdx; // We save beautyAux id for init mem and checkOutputInterval
        }
        if (rod->getRenderOutput(roIdx)->getResult() == scene_rdl2::rdl2::RenderOutput::RESULT_ALPHA_AUX) {
            skipST = SKIP_CONDITION_ALPHAAUX_AOV;
            alphaAuxId = roIdx; // We save alphaAux id for init mem and checkOutputInterval
        }
    }
    mRenderOutputSkipCondition[roIdx] = skipST;

    mRenderOutputBufferDefaultValue[roIdx] = rod->getAovDefaultValue(roIdx);
    if (skipST == SKIP_CONDITION_BEAUTYAUX_AOV || skipST == SKIP_CONDITION_ALPHAAUX_AOV) {
        mRenderOutputBufferScaledByWeight[roIdx] = true; // special case. We can not get this info from rod
    } else {
        mRenderOutputBufferScaledByWeight[roIdx] = static_cast<char>(rod->requiresScaledByWeight(roIdx));
    }

    unsigned int numChans = rod->getNumberOfChannels(roIdx);
    mRenderOutputBufferOrigNumChan[roIdx] = numChans; // save original numChan here
    mRenderOutputBufferClosestFilterStatus[roIdx] = false;
    {
        auto isClosestFilter = [rod, roIdx]() {
            const pbr::AovSchema &schema = rod->getAovSchema();
            const int aovIdx = rod->getAovBuffer(roIdx);
            const bool result = aovIdx >= 0 && schema[aovIdx].filter() == pbr::AOV_FILTER_CLOSEST;
            return result;
        };
        if (isClosestFilter()) {
            MNRY_ASSERT(numChans <= 3);
            numChans = 4;       // this AOV uses closestFilter, and uses FLOAT4 buffer
            mRenderOutputBufferClosestFilterStatus[roIdx] = true;
        } else {
            if (numChans == 0 || 3 < numChans) {
                skipST |= SKIP_CONDITION_MANUALLY_SKIP; // Wrong numChans, set manually skip condition
            }
        }
    }

    if (skipST == 0x0) {
        unsigned width = mActivePixels.getWidth();
        unsigned height = mActivePixels.getHeight();
        unsigned tileAlignedWidth = mActivePixels.getAlignedWidth();
        unsigned tileAlignedHeight = mActivePixels.getAlignedHeight();

        mActivePixelsRenderOutput[roIdx].init(width, height);
        mRenderOutputBufferTiled[roIdx].init(fmtTbl[numChans], tileAlignedWidth, tileAlignedHeight);
        mRenderOutputWeightBufferTiled[roIdx].init(tileAlignedWidth, tileAlignedHeight);
    } else {
        // skip this buffer
        mActivePixelsRenderOutput[roIdx].cleanUp();
        mRenderOutputBufferTiled[roIdx].cleanUp();
        mRenderOutputWeightBufferTiled[roIdx].cleanUp();
    }

    mRenderOutputBufferCoarsePassPrecision[roIdx] =
        calcRenderOutputBufferCoarsePassPrecision(rod, static_cast<int>(roIdx));
    mRenderOutputBufferFinePassPrecision[roIdx] =
        calcRenderOutputBufferFinePassPrecision(rod, static_cast<int>(roIdx));
}

void
McrtFbSender::adjustRenderBufferFinePassPrecision(const rndr::RenderOutputDriver *rod,
                                                  const int beautyId, const int alphaId,
                                                  const int beautyAuxId, const int alphaAuxId)
//
// adjust renderBuffer's fine pass precision based on the channel-format (float/half) definition
//
{
    auto calcPrecision = [&](const rndr::RenderOutputDriver *rod,
                             const int beautyId, const int alphaId,
                             const int beautyAuxId, const int alphaAuxId) -> FinePassPrecision {
        //
        // This is a heuristic solution. We want to decide renderBuffer (RGB + A) fine pass precision
        // based on the renderOutput definition here.
        // We have 4 input beautyId (RGB), alphaId (A), beautyAuxId (RGB) and alphaAuxId (A).
        // We don't know all the information is available. (In most of the case we can get beautyId
        // and alphaId but it is not mandatory.)
        // Alpha buffer is unlikely to keep more than 1.0 value mostly and we should pick beauty buffer
        // condition to define renderBuffer fine pass precision if available.
        // This is why beautyId and beautyAuxId are checked before alphaId and alphaAuxId.
        // If we don't have all the input, precision is not changed by renderOutput definition.
        //        
        FinePassPrecision precision = mRenderBufferFinePassPrecision;
        if (beautyId != -1) {
            precision = calcRenderOutputBufferFinePassPrecision(rod, beautyId);
        } else if (beautyAuxId != -1) {
            precision = calcRenderOutputBufferFinePassPrecision(rod, beautyAuxId);
        } else if (alphaId != -1) {
            precision = calcRenderOutputBufferFinePassPrecision(rod, alphaId);
        } else if (alphaAuxId != -1) {
            precision = calcRenderOutputBufferFinePassPrecision(rod, alphaAuxId);
        }
        return precision;
    };
                            
    mRenderBufferFinePassPrecision = calcPrecision(rod, beautyId, alphaId, beautyAuxId, alphaAuxId);
}

McrtFbSender::FinePassPrecision
McrtFbSender::calcRenderOutputBufferFinePassPrecision(const rndr::RenderOutputDriver *rod,
                                                      const int roIdx) const
//
// Define fine pass precision by channel-format definition (float/half) inside RenderOutput block.
//
{
    FinePassPrecision precision = FinePassPrecision::F32;

    if (roIdx >= 0) {
        const scene_rdl2::rdl2::RenderOutput *ro = rod->getRenderOutput(roIdx);
        switch (ro->getChannelFormat()) {
        case scene_rdl2::rdl2::RenderOutput::ChannelFormat::CHANNEL_FORMAT_FLOAT : precision = FinePassPrecision::F32; break;
        case scene_rdl2::rdl2::RenderOutput::ChannelFormat::CHANNEL_FORMAT_HALF : precision = FinePassPrecision::H16; break;
        default : break;
        }
    }
    return precision;
}

McrtFbSender::CoarsePassPrecision
McrtFbSender::calcRenderOutputBufferCoarsePassPrecision(const rndr::RenderOutputDriver *rod,
                                                        const int roIdx) const
//
// This function compute best packTile precision mode for coarse pass encoding about each AOV data
//    
{
    CoarsePassPrecision precision = CoarsePassPrecision::F32;

    const scene_rdl2::rdl2::RenderOutput *ro = rod->getRenderOutput(roIdx);
    switch (ro->getResult()) {
    case scene_rdl2::rdl2::RenderOutput::RESULT_BEAUTY : precision = COARSE_PASS_PRECISION_BEAUTY; break;
    case scene_rdl2::rdl2::RenderOutput::RESULT_ALPHA : precision = COARSE_PASS_PRECISION_ALPHA; break;
    case scene_rdl2::rdl2::RenderOutput::RESULT_DEPTH : precision = CoarsePassPrecision::H16; break;
    case scene_rdl2::rdl2::RenderOutput::RESULT_STATE_VARIABLE : precision = CoarsePassPrecision::H16; break;
    case scene_rdl2::rdl2::RenderOutput::RESULT_PRIMITIVE_ATTRIBUTE :
        if (ro->getMathFilter() == scene_rdl2::rdl2::RenderOutput::MATH_FILTER_AVG ||
            ro->getMathFilter() == scene_rdl2::rdl2::RenderOutput::MATH_FILTER_SUM) {
            precision = CoarsePassPrecision::H16;
        } else {
            // This primitive attribute might be an integer base value.
            // If so we can not do any compression.
            precision = CoarsePassPrecision::F32;
        }
        break;
    case scene_rdl2::rdl2::RenderOutput::RESULT_HEAT_MAP :
        // Actually, we don't have precision control for Heat Map.
        // Heat Map is always encoded as H16 regardless of coarse pass or non coarse pass
        precision = CoarsePassPrecision::H16;
        break;
    case scene_rdl2::rdl2::RenderOutput::RESULT_WIREFRAME : precision = CoarsePassPrecision::UC8; break;
    case scene_rdl2::rdl2::RenderOutput::RESULT_MATERIAL_AOV : precision = CoarsePassPrecision::H16; break;
    case scene_rdl2::rdl2::RenderOutput::RESULT_LIGHT_AOV : precision = CoarsePassPrecision::H16; break;
    case scene_rdl2::rdl2::RenderOutput::RESULT_VISIBILITY_AOV : precision = CoarsePassPrecision::UC8; break;
    case scene_rdl2::rdl2::RenderOutput::RESULT_WEIGHT : precision = COARSE_PASS_PRECISION_WEIGHT; break;
    case scene_rdl2::rdl2::RenderOutput::RESULT_BEAUTY_AUX : precision = COARSE_PASS_PRECISION_BEAUTY; break;
    case scene_rdl2::rdl2::RenderOutput::RESULT_CRYPTOMATTE : precision = CoarsePassPrecision::F32; break;
    case scene_rdl2::rdl2::RenderOutput::RESULT_ALPHA_AUX : precision = COARSE_PASS_PRECISION_ALPHA; break;
    case scene_rdl2::rdl2::RenderOutput::RESULT_DISPLAY_FILTER :
        // We might be OK just using H16 instead of RUNTIME_DECISION.
        // RUNTIME_DECISION has the possibility to create a small packet for coarse pass but needs more cost.
        precision = CoarsePassPrecision::RUNTIME_DECISION;
        break;
    default : break;
    }

    return precision;
}

McrtFbSender::PackTilePrecision
McrtFbSender::calcPackTilePrecision(const CoarsePassPrecision coarsePassPrecision,
                                    const FinePassPrecision finePassPrecision,
                                    PackTilePrecisionCalcFunc runtimeDecisionFunc) const
//
// This function decides runtime encoding precision
//
{
    auto calcCoarsePassPrecision = [&]() -> PackTilePrecision {
        PackTilePrecision precision = PackTilePrecision::F32;
        switch (coarsePassPrecision) {
        case CoarsePassPrecision::F32 : precision = PackTilePrecision::F32; break;
        case CoarsePassPrecision::H16 : precision = PackTilePrecision::H16; break;
        case CoarsePassPrecision::UC8 : precision = PackTilePrecision::UC8; break;
        case CoarsePassPrecision::RUNTIME_DECISION :
            if (runtimeDecisionFunc) precision = runtimeDecisionFunc();
            break;
        default : break;
        }
        return precision;
    };
    auto calcFinePassPrecision = [&]() -> PackTilePrecision {
        PackTilePrecision precision = PackTilePrecision::F32;
        switch (finePassPrecision) {
        case FinePassPrecision::F32 : precision = PackTilePrecision::F32; break;
        case FinePassPrecision::H16 : precision = PackTilePrecision::H16; break;
        default : break;
        }
        return precision;
    };

    PackTilePrecision precision = PackTilePrecision::F32;

    switch (mPrecisionControl) {
    case PrecisionControl::FULL32 :
        // Always uses F32 for both of Coarse and Fine pass
        precision = PackTilePrecision::F32; // Always uses F32
        break;

    case PrecisionControl::FULL16 :
        // Always uses H16 if possible for both of Coarse and Fine pass.
        // However, uses F32 if minimum precision is F32
        if (mSnapshotDeltaCoarsePass) {
            if (coarsePassPrecision == CoarsePassPrecision::F32) {
                precision = PackTilePrecision::F32; // This data is not able to use H16
            } else {
                precision = PackTilePrecision::H16; // Use H16 if possible
            }
        } else {
            if (finePassPrecision == FinePassPrecision::F32) {
                precision = PackTilePrecision::F32; // This data is not able to use H16                
            } else {
                precision = PackTilePrecision::H16; // Use H16 if possible
            }
        }
        break;

    case PrecisionControl::AUTO32 :
        // CoarsePass : Choose proper precision automatically based on the AOV data
        // FinePass   : Always uses F32
        if (mSnapshotDeltaCoarsePass) {
            precision = calcCoarsePassPrecision(); // respect coarse pass precision decision
        } else {
            precision = PackTilePrecision::F32;    // Always uses F32
        }
        break;

    case PrecisionControl::AUTO16 :
        // CoarsePass : Choose proper precision automatically based on the AOV data
        // FinePass   : Basically use H16. Only uses F32 if minimum precision is F32
        if (mSnapshotDeltaCoarsePass) {
            precision = calcCoarsePassPrecision(); // respect coarse pass precision decision
        } else {
            precision = calcFinePassPrecision(); // respect fine pass precision decision
        }
        break;

    default : break;
    }

    return precision;
}

scene_rdl2::grid_util::PackTiles::PrecisionMode
McrtFbSender::getBeautyHDRITestResult()
//
// Return beautyHDRI test result with test result cache logic in order to avoid duplicate
// execution of testing itself.
//    
{
    auto getResult = [&]() -> scene_rdl2::grid_util::PackTiles::PrecisionMode {
        return ((mBeautyHDRITest == HdriTestCondition::HDRI) ?
                scene_rdl2::grid_util::PackTiles::PrecisionMode::H16 :
                scene_rdl2::grid_util::PackTiles::PrecisionMode::UC8);
    };

    if (mBeautyHDRITest != HdriTestCondition::INIT) return getResult();

    mBeautyHDRITest = (beautyHDRITest()) ? HdriTestCondition::HDRI : HdriTestCondition::NON_HDRI;
    return getResult();
}

bool
McrtFbSender::beautyHDRITest() const
//
// HDR pixel existence test for beauty buffer
//
{
    size_t area = mActivePixels.getAlignedWidth() * mActivePixels.getAlignedHeight();
    // We want to use minLimit in order to ignore small number of HDRI pixels like firefly
    // Use experimental number here. This cost was around 0.5ms~2ms range in my HD reso test scene.
    // Future enhancement related ticket is MOONRAY-3588
    size_t minLimit = (size_t)((float)area * 0.005f); // 0.5% of whole pixels
    const scene_rdl2::math::Vec4f *c = mRenderBufferTiled.getData();
    const float *w = mRenderBufferWeightBufferTiled.getData();

    size_t totalHDRI = 0;
    for (size_t i = 0; i < area; ++i, ++c, ++w) {
        if (*w != 0.0f) {
            float scale = 1.0f / *w;
            if (c->x * scale > 1.0f || c->y * scale > 1.0f ||
                c->z * scale > 1.0f || c->w * scale > 1.0f) {
                if (totalHDRI > minLimit) {
                    return true;
                }
                totalHDRI++;
            }
        }
    }

    return false;
}

bool
McrtFbSender::renderOutputHDRITest(const ActivePixels &activePixels,
                                   const VariablePixelBuffer &buff,
                                   const FloatBuffer &weightBuff) const
//
// HDR pixel existence test for renderOutput AOV buffer.
// The idea is the same as beautyHDRITest() and we are not testing whole pixels.
//    
{
    if (buff.getFormat() == scene_rdl2::fb_util::VariablePixelBuffer::Format::RGB888 ||
        buff.getFormat() == scene_rdl2::fb_util::VariablePixelBuffer::Format::RGBA8888) {
        return false; // uc8 based frame buffer
    }
    if (buff.getFormat() != scene_rdl2::fb_util::VariablePixelBuffer::Format::FLOAT &&
        buff.getFormat() != scene_rdl2::fb_util::VariablePixelBuffer::Format::FLOAT2 &&
        buff.getFormat() != scene_rdl2::fb_util::VariablePixelBuffer::Format::FLOAT3 &&
        buff.getFormat() != scene_rdl2::fb_util::VariablePixelBuffer::Format::FLOAT4) {
        return true; // We can not apply HDRI test here. return true as non 8bit range data
    }

    size_t area = activePixels.getAlignedWidth() * activePixels.getAlignedHeight();
    size_t minLimit = static_cast<size_t>((float)area * 0.005f); // 0.5% of whole pixels
    unsigned pixByte = buff.getSizeOfPixel();
    size_t pixFloatCount = pixByte / sizeof(float);
    const float *p = reinterpret_cast<const float *>(buff.getData());
    const float *w = weightBuff.getData();

    size_t totalHDRI = 0;
    for (size_t i = 0; i < area; ++i, ++w) {
        if (*w > 0.0f) {
            int currTotalHDRIchan = 0;
            for (size_t j = 0; j < pixFloatCount ; ++j) {
                if (p[j] > *w) currTotalHDRIchan++;
            }
            p += pixFloatCount;
            if (currTotalHDRIchan > 0) {
                if (totalHDRI > minLimit) {
                    return true; // HDRI fb
                }
                totalHDRI++;
            }
        }
    }
    return false; // non HDRI fb
}

void
McrtFbSender::computeSecBuffer()
{
    for (unsigned offset = 0; offset < mHeatMapBufferTiled.getArea(); ++offset) {
        const int64_t *src = mHeatMapBufferTiled.getData() + offset;
        float *dst = mHeatMapSecBufferTiled.getData() + offset;

        *dst = mcrt_common::Clock::seconds(*src);
    }
}

void
McrtFbSender::setRenderOutputSkipCondition(const int index, const bool skip)
{
    mRenderOutputSkipCondition[index] &= ~SKIP_CONDITION_MANUALLY_SKIP;
    mRenderOutputSkipCondition[index] |= ((skip)? SKIP_CONDITION_MANUALLY_SKIP: 0x0);
}

unsigned
McrtFbSender::getNonBlackBeautyPixelTotal() const // for debug
{
    unsigned total = 0;
    ActivePixels::crawlAllActivePixels(mActivePixels,
                                       [&](unsigned currPixOffset) {
                                           const scene_rdl2::fb_util::RenderColor *v = mRenderBufferTiled.getData() + currPixOffset;
                                           if ((*v)[0] != 0.0f && (*v)[1] != 0.0f && (*v)[2] != 0.0f) total++;
                                       });
    return total;
}

unsigned
McrtFbSender::getNonZeroWeightPixelTotal() const // for debug
{
    unsigned total = 0;
    ActivePixels::crawlAllActivePixels(mActivePixelsWeightBuffer,
                                       [&](unsigned currPixOffset) {
                                           const float *v = mWeightBufferTiled.getData() + currPixOffset;
                                           if ((*v) != 0.0f) total++;
                                       });
    return total;
}

bool
McrtFbSender::saveRenderBufferTiledByPPM(const std::string &filename,
                                         const ActivePixels &activePixels,
                                         const scene_rdl2::fb_util::RenderBuffer &renderBuf) const
//
// For debugging purpose
// Save renderBuf's RGB (clipped 0.0~1.0) by PPM format w/ value resolution as 0~4095.
//
{
    constexpr int valReso = 4096;
    auto toIntFCol = [&](const float v) -> int {
        auto clamp = [](const float v) -> float { return ((v < 0.0f)? 0.0f: ((v > 1.0f)? 1.0f: v)); };
        return (int)(clamp(v) * (float)(valReso - 1) + 0.5f);
    };

    std::ostringstream ostr;
    size_t n = filename.rfind(".ppm");
    if (n != std::string::npos) {
        ostr << filename.substr(0, n);
    } else {
        ostr << filename;
    }
    ostr << "_mId" << mLatencyLog.getMachineId() << ".ppm";
    std::string currFilename = ostr.str();

    std::ofstream ofs(currFilename);
    if (!ofs) {
        std::cerr << ">> ERROR McrtFbSender.cc saveRenderBufferTiledByPPM() Can not create file "
                  << currFilename << std::endl;
        return false;
    }

    int xMin = 0;
    int yMin = 0;
    int xMax = activePixels.getWidth() - 1;
    int yMax = activePixels.getHeight() - 1;
    int xSize = xMax - xMin + 1;
    int ySize = yMax - yMin + 1;

    scene_rdl2::fb_util::Tiler tiler(activePixels.getWidth(), activePixels.getHeight());

    ofs << "P3\n" << xSize << ' ' << ySize << '\n' << (valReso - 1) << '\n';
    for (int y = yMax; y >= yMin; --y) {
        for (int x = xMin; x <= xMax; ++x) {
            unsigned px, py;
            tiler.linearToTiledCoords((unsigned)x, (unsigned)y, &px, &py);
            const scene_rdl2::fb_util::RenderColor &c = renderBuf.getPixel(px, py);
            ofs << toIntFCol(c[0]) << ' ' << toIntFCol(c[1]) << ' ' << toIntFCol(c[2]) << ' ';
        }
    }

    ofs.close();
    return true;
}

void
McrtFbSender::parserConfigure()
{
    using Arg = scene_rdl2::grid_util::Arg;
    
    mParser.description("McrtFbSender command");
    mParser.opt("denoiseInfo", "", "show denoise related info",
                [&](Arg& arg) -> bool {
                    return arg.msg(showDenoiseInfo() + '\n');
                });
    mParser.opt("showPix", "<sx> <sy>", "show pixel value",
                [&](Arg& arg) {
                    unsigned sx = (arg++).as<unsigned>(0);
                    unsigned sy = (arg++).as<unsigned>(0);
                    return arg.msg(showRenderBufferPix(sx, sy) + '\n');
                });
}

std::string
McrtFbSender::showDenoiseInfo() const
{
    auto showName = [&](const std::string* name) -> std::string {
        std::ostringstream ostr;
        ostr << "0x" << std::hex << reinterpret_cast<uintptr_t>(name) << std::dec;
        if (!name) ostr << " empty";
        else ostr << ' ' << *name;
        return ostr.str();
    };

    std::ostringstream ostr;
    ostr << "denoiseInfo {\n"
         << "  mDenoiserAlbedoInputNamePtr:" << showName(mDenoiserAlbedoInputNamePtr) << '\n'
         << "  mDenoiserNormalInputNamePtr:" << showName(mDenoiserNormalInputNamePtr) << '\n'
         << "}";
    return ostr.str();
}

std::string
McrtFbSender::showRenderBufferPix(const unsigned sx, const unsigned sy) const
{
    if (sx >= getWidth() || sy >= getHeight()) {
        std::ostringstream ostr;
        ostr << "pix(sx:" << sx << ", sy:" << sy << ") is out of framebuffer size";
        return ostr.str();
    }            

    bool activePixFlag = mActivePixels.isActivePixel(sx, sy);

    scene_rdl2::fb_util::Tiler tiler(getWidth(), getHeight());
    unsigned px, py;
    tiler.linearToTiledCoords(sx, sy, &px, &py);
    RenderColor c = mRenderBufferTiled.getPixel(px, py);
    float w = mRenderBufferWeightBufferTiled.getPixel(px, py);

    std::ostringstream ostr;
    ostr << "RenderBuffer (sx:" << sx << ", sy:" << sy << ") {\n"
         << "  activePixFlag:" << scene_rdl2::str_util::boolStr(activePixFlag) << '\n'
         << "  color:" << c[0] << ' ' << c[1] << ' ' << c[2] << ' ' << c[3] << '\n'
         << "  w:" << w << '\n'
         << "}";
    return ostr.str();
}

} // namespace engine_tool
} // namespace moonray
