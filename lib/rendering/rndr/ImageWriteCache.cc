// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#include <scene_rdl2/render/util/AtomicFloat.h> // Needs to be included before any OpenImageIO file
#include "ImageWriteCache.h"
#include "ProcKeeper.h"
#include "RenderDriver.h"
#include "RenderOutputDriver.h"

#include <scene_rdl2/common/math/Math.h>
#include <scene_rdl2/common/rec_time/RecTime.h>
#include <scene_rdl2/render/util/StrUtil.h>
#include <scene_rdl2/render/util/Strings.h>
#include <scene_rdl2/render/util/LuaScriptRunner.h>
#include <fstream>

#include <algorithm>            // clamp
#include <errno.h>              // errno
#include <fcntl.h>              // open
#include <malloc.h>             // malloc_trim
#include <string.h>             // strerror
#include <sys/sendfile.h>       // sendfile
#include <sys/stat.h>           // fstat
#include <sys/types.h>          // fstat
#include <unistd.h>             // unlink, close, isatty

// for runtime verify of block data size calculation
//#define RUNTIME_VERIFY_BLOCKDATASIZE

// debug message for write action progress update
//#define DEBUG_MSG_PROGRESS

namespace moonray {
namespace rndr {

size_t
ImageWriteCacheSpecs::memSizeByte() const
{
    // This is a rough estimate based on the OIIO::ImageSpec size and this is not include dynamically
    // allocate memory size inside OIIO::ImageSpec.
    return sizeof(*this) + sizeof(OIIO::ImageSpec) * mSpecs.size();
}

//-----------------------------------------------------------------------------------------

void
ImageWriteCacheImageSpec::setSizeInfo(int width,
                                      int height,
                                      int x,
                                      int y,
                                      int fullX,
                                      int fullY,
                                      int fullWidth,
                                      int fullHeight)
{
    mWidth = width;
    mHeight = height;
    
    mSpecX = x;
    mSpecY = y;
    mSpecFullX = fullX;
    mSpecFullY = fullY;
    mSpecFullWidth = fullWidth;
    mSpecFullHeight = fullHeight;
}

void
ImageWriteCacheImageSpec::pushBackChanFormatN(int total, const ChannelFormat &chanFormat)
{
    mChanFormat.insert(mChanFormat.end(), total, chanFormat);
}

void
ImageWriteCacheImageSpec::pushBackChanNames(const std::vector<std::string> &chanNames)
{
    mChanNames.insert(mChanNames.end(), chanNames.begin(), chanNames.end());
}

void
ImageWriteCacheImageSpec::setExrHeaderInfo(const std::vector<std::string> &attrNames,
                                           const std::vector<std::string> &attrTypes,
                                           const std::vector<std::string> &attrValues,
                                           const std::string &metaDataName)
{
    mAttrNames = attrNames;
    mAttrTypes = attrTypes;
    mAttrValues = attrValues;
    mMetaDataName = metaDataName;
}

void
ImageWriteCacheImageSpec::setupOIIOSpecs(const CallBackSetup &callBack) const
{
    callBack(mName,
             mWidth,
             mHeight,
             mSpecX,
             mSpecY,
             mSpecFullX,
             mSpecFullY,
             mSpecFullWidth,
             mSpecFullHeight,
             mTotalNumChans,
             mCompression,
             mLevel,
             mChanFormat,
             mChanNames,
             mAttrNames,
             mAttrTypes,
             mAttrValues,
             mMetaDataName,
             mResumeAttr);
}

void
ImageWriteCacheImageSpec::updateHash(scene_rdl2::grid_util::Sha1Gen &sha1Gen) const
{
    sha1Gen.updateInt2(mWidth, mHeight);
    sha1Gen.update<int>(mTotalNumChans);
    sha1Gen.updateInt2(mSpecX, mSpecY);
    sha1Gen.updateInt2(mSpecFullX, mSpecFullY);
    sha1Gen.updateInt2(mSpecFullWidth, mSpecFullHeight);

    for (size_t i = 0; i < mChanFormat.size(); ++i) {
        sha1Gen.update<ChannelFormat>(mChanFormat[i]);
    }
    sha1Gen.updateStrVec(mChanNames);
    if (!mName.empty()) {
        sha1Gen.updateStr(mName);
    }
    sha1Gen.updateStr(mCompression);
    if (isCompressionDwa()) {
        sha1Gen.update<float>(mLevel);
    }

    if (!mAttrNames.empty()) {
        sha1Gen.updateStrVec(mAttrNames);
        sha1Gen.updateStrVec(mAttrTypes);
        sha1Gen.updateStrVec(mAttrValues);
        sha1Gen.updateStr(mMetaDataName);
    }
    if (!mResumeAttr.empty()) {
        for (size_t i = 0; i < mResumeAttr.size(); i += 3) {
            sha1Gen.updateStr3(mResumeAttr[i],
                               mResumeAttr[i+1],
                               mResumeAttr[i+2]);
        }
    }
}

size_t
ImageWriteCacheImageSpec::memSizeByte() const
{
    auto strVecMemSize = [](const std::vector<std::string> &strVec) -> size_t {
        size_t total = 0;
        for (const auto &itr: strVec) {
            total += itr.size();
        }
        return total;
    };

    return (sizeof(*this) +
            mName.size() +
            mCompression.size() +
            sizeof(ChannelFormat) * mChanFormat.size() +
            strVecMemSize(mChanNames) +
            strVecMemSize(mAttrNames) +
            strVecMemSize(mAttrTypes) +
            strVecMemSize(mAttrValues) +
            mMetaDataName.size() +
            strVecMemSize(mResumeAttr));
}

std::string
ImageWriteCacheImageSpec::show(bool detail) const
{
    auto showFormat = [](const ChannelFormat &fmt) -> std::string {
        switch (fmt) {
        case ChannelFormat::CHANNEL_FORMAT_FLOAT : return "CHANNEL_FORMAT_FLOAT";
        case ChannelFormat::CHANNEL_FORMAT_HALF : return "CHANNEL_FORAMT_HALF";
        default : return "?";
        }
    };
    auto showChanFormat = [&]() -> std::string {
        std::ostringstream ostr;
        ostr << "mChanFormat:";
        if (mChanFormat.size() > 0) {
            ostr << "(size:" << mChanFormat.size() << ")";
            if (detail) {
                ostr << " {\n";
                int w = scene_rdl2::str_util::getNumberOfDigits(mChanFormat.size());
                for (size_t i = 0; i < mChanFormat.size(); ++i) {
                    ostr << "  i:" << std::setw(w) << i << ' ' << showFormat(mChanFormat[i]) << '\n';
                }
                ostr << "}";
            }
        } else {
            ostr << "empty";
        }
        return ostr.str();
    };
    auto showStrVec = [&](const std::vector<std::string> &strVec) -> std::string {
        std::ostringstream ostr;
        if (strVec.empty()) {
            ostr << "empty";
        } else {
            ostr << "(size:" << strVec.size() << ")";
            if (detail) {
                ostr << " {\n";
                int w = scene_rdl2::str_util::getNumberOfDigits(strVec.size());
                for (size_t i = 0; i < strVec.size(); ++i) {
                    std::ostringstream ostr2;
                    ostr2 << "i:" << std::setw(w) << i << ' ' << strVec[i];
                    ostr << scene_rdl2::str_util::addIndent(ostr2.str()) << '\n';
                }
                ostr << "}";
            }
        }
        return ostr.str();
    };

    std::ostringstream ostr;
    ostr << "ImageWriteCacheImageSpec {\n"
         << "  mName:" << mName << '\n'
         << "  mWidth:" << mWidth << '\n'
         << "  mHeight:" << mHeight << '\n'
         << "  mSpecX:" << mSpecX << '\n'
         << "  mSpecY:" << mSpecY << '\n'
         << "  mSpecFullX:" << mSpecFullX << '\n'
         << "  mSpecFullY:" << mSpecFullY << '\n'
         << "  mSpecFullWidth:" << mSpecFullWidth << '\n'
         << "  mSpecFullHeight:" << mSpecFullHeight << '\n'
         << "  mTotalNumChans:" << mTotalNumChans << '\n'
         << "  mCompression:" << mCompression << '\n'
         << "  mLevel:" << mLevel << '\n'
         << scene_rdl2::str_util::addIndent(showChanFormat()) << '\n'
         << scene_rdl2::str_util::addIndent(std::string("mChanNames:") + showStrVec(mChanNames)) << '\n'
         << scene_rdl2::str_util::addIndent(std::string("mAttrNames:") + showStrVec(mAttrNames)) << '\n'
         << scene_rdl2::str_util::addIndent(std::string("mAttrTypes:") + showStrVec(mAttrTypes)) << '\n'
         << scene_rdl2::str_util::addIndent(std::string("mAttrValues:") + showStrVec(mAttrValues)) << '\n'
         << "  mMetaDataName:" << mMetaDataName << '\n'
         << scene_rdl2::str_util::addIndent(std::string("mResumeAttr:") + showStrVec(mResumeAttr)) << '\n'
         << '}';
    return ostr.str();
}

//-----------------------------------------------------------------------------------------

void
ImageWriteCacheBufferSpecSubImage::setup(const ImageWriteCacheBufferSpecSubImage *prev,
                                         const std::string &name,
                                         const size_t aovBuffSize,
                                         const size_t displayFilterBuffSize,
                                         const size_t pixCacheFullSize,
                                         const size_t pixCacheHalfSize,
                                         const std::vector<ChanFormat> &pixChanFormat,
                                         const std::vector<int> &pixNumChan)
{
    mName = name;

    mAovBuffSize = aovBuffSize;
    mDisplayFilterBuffSize = displayFilterBuffSize;
    mPixCacheFullSize = pixCacheFullSize;
    mPixCacheHalfSize = pixCacheHalfSize;
    mPixChanFormat = pixChanFormat;
    mPixNumChan = pixNumChan;

    if (prev) {
        mImgSpecId = prev->mImgSpecId + 1;
        mAovBuffOffset = prev->mAovBuffOffset + prev->mAovBuffSize;
        mDisplayFilterBuffOffset = prev->mDisplayFilterBuffOffset + prev->mDisplayFilterBuffSize;
        mPixCacheFullOffset = prev->mPixCacheFullOffset + prev->mPixCacheFullSize;
        mPixCacheHalfOffset = prev->mPixCacheHalfOffset + prev->mPixCacheHalfSize;
    }
}

void
ImageWriteCacheBufferSpecSubImage::updateHash(scene_rdl2::grid_util::Sha1Gen *sha1Gen) const
{
    if (!sha1Gen) return;

    sha1Gen->updateStr(mName);

    sha1Gen->update<size_t>(mImgSpecId);

    sha1Gen->update<size_t>(mAovBuffOffset);
    sha1Gen->update<size_t>(mDisplayFilterBuffOffset);
    sha1Gen->update<size_t>(mAovBuffSize);
    sha1Gen->update<size_t>(mDisplayFilterBuffSize);
    
    sha1Gen->update<size_t>(mPixCacheFullOffset);
    sha1Gen->update<size_t>(mPixCacheHalfOffset);
    sha1Gen->update<size_t>(mPixCacheFullSize);
    sha1Gen->update<size_t>(mPixCacheHalfSize);

    for (const auto &itr: mPixChanFormat) {
        sha1Gen->update<ChanFormat>(itr);
    }
    for (const auto &itr: mPixNumChan) {
        sha1Gen->update<int>(itr);
    }
}

size_t
ImageWriteCacheBufferSpecSubImage::memSizeByte() const
{
    return (sizeof(*this) +
            mName.size() +
            sizeof(ChanFormat) * mPixChanFormat.size() +
            sizeof(int) * mPixNumChan.size());
}

std::string
ImageWriteCacheBufferSpecSubImage::show() const
{
    auto showPixInfo = [&]() -> std::string {
        std::ostringstream ostr;
        ostr << "pixInfo (size:" << mPixChanFormat.size() << ") {\n";
        int w = scene_rdl2::str_util::getNumberOfDigits(mPixChanFormat.size());
        for (size_t i = 0; i < mPixChanFormat.size(); ++i) {
            ostr << "  i:" << std::setw(w) << i
                 << " chanFormat:" << showChanFormat(mPixChanFormat[i])
                 << " numChan:" << mPixNumChan[i] << '\n';
        }
        ostr << "}";
        return ostr.str();
    };

    std::ostringstream ostr;
    ostr << "ImageWriteCacheBufferSpecSubImage {\n"
         << "  mName:" << mName << '\n'
         << "  mImgSpecId:" << mImgSpecId << '\n'
         << "  mAovBuffOffset:" << mAovBuffOffset << '\n'
         << "  mDisplayFilterBuffOffset:" << mDisplayFilterBuffOffset << '\n'
         << "  mAovBuffSize:" << mAovBuffSize << '\n'
         << "  mDisplayFilterBuffSize:" << mDisplayFilterBuffSize << '\n'
         << "  mPixCacheFullOffset:" << mPixCacheFullOffset << '\n'
         << "  mPixCacheHalfOffset:" << mPixCacheHalfOffset << '\n'
         << "  mPixCacheFullSize:" << mPixCacheFullSize << '\n'
         << "  mPixCacheHalfSize:" << mPixCacheHalfSize << '\n'
         << scene_rdl2::str_util::addIndent(showPixInfo()) << '\n'
         << "}";
    return  ostr.str();
}

// static function
std::string
ImageWriteCacheBufferSpecSubImage::showChanFormat(const ChanFormat &format)
{
    switch (format) {
    case ChanFormat::UNKNOWN : return "UNKNOWN";
    case ChanFormat::HALF : return "HALF";
    case ChanFormat::FULL : return "FULL";
    default : return "?";
    }
}

void
ImageWriteCacheBufferSpecFile::updateHash(scene_rdl2::grid_util::Sha1Gen *sha1Gen) const
{
    if (!sha1Gen) return;

    for (const auto &itr: mSubImgTbl) {
        itr.updateHash(sha1Gen);
    }
}

size_t
ImageWriteCacheBufferSpecFile::memSizeByte() const
{
    size_t total = sizeof(*this);
    for (const auto &itr: mSubImgTbl) {
        total += itr.memSizeByte();
    }
    return total;
}

std::string
ImageWriteCacheBufferSpecFile::show() const
{
    std::ostringstream ostr;
    ostr << "ImageWriteCacheBufferSpecFile {\n"
         << "  mSubImgTbl (size:" << mSubImgTbl.size() << ") {\n";
    for (size_t i = 0; i < mSubImgTbl.size(); ++i) {
        ostr << scene_rdl2::str_util::addIndent("i:" + std::to_string(i) + ' ' + mSubImgTbl[i].show(), 2)
             << '\n';
    }
    ostr << "  }\n"
         << "}";
    return ostr.str();
}

void
ImageWriteCacheBufferSpec::updateHash(scene_rdl2::grid_util::Sha1Gen *sha1Gen) const
{
    if (!sha1Gen) return;

    for (const auto &itr: mFileTbl) {
        itr.updateHash(sha1Gen);
    }
}

size_t
ImageWriteCacheBufferSpec::memSizeByte() const
{
    size_t total = sizeof(*this);
    for (const auto &itr: mFileTbl) {
        total += itr.memSizeByte();
    }
    return total;
}

std::string
ImageWriteCacheBufferSpec::show() const
{
    std::ostringstream ostr;
    ostr << "ImageWriteCacheBufferSpec {\n"
         << "  mFileTbl (size:" << mFileTbl.size() << ") {\n";
    for (size_t i = 0; i < mFileTbl.size(); ++i) {
        ostr << scene_rdl2::str_util::addIndent("i:" + std::to_string(i) + ' ' + mFileTbl[i].show(), 2)
             << '\n';
    }
    ostr << "  }\n"
         << "}";
    return ostr.str();
}

//-----------------------------------------------------------------------------------------
    
size_t
ImageWriteCacheTimingLog::memSizeByte() const
{
    return sizeof(*this) + sizeof(float) * (mEnqTimes.size() + mDeqTimes.size());
}

//-----------------------------------------------------------------------------------------
    
bool
ImageWriteCacheTmpFileItem::openTmpFileAndUnlink()
// In order to guarantee removal of the temporary file in the event of abnormal program termination
{
    if (mTmpFilename.empty()) return false;

    mTmpFileFd = open(mTmpFilename.c_str(), O_RDONLY, 0);
    if (mTmpFileFd == -1) {
        return false;
    }
    if (unlink(mTmpFilename.c_str()) == -1) {
        return false;
    }
    return true;
}

bool
ImageWriteCacheTmpFileItem::closeTmpFile()
{
    if (mTmpFileFd == -1) return false;
    if (close(mTmpFileFd) == -1) return false;
    mTmpFileFd = -1;
    return true;
}

size_t
ImageWriteCacheTmpFileItem::memSizeByte() const
{
    return (sizeof(*this) +
            mTmpFilename.size() +
            mCheckpointFilename.size() +
            mCheckpointMultiVersionFilename.size() +
            mFinalFilename.size());
}

bool
ImageWriteCacheTmpFileItem::copyFile(const std::string &dstName, std::string &errMsg) const
//
// Copy tmpFile data to destination location but name is not a final dstName.
// Destination filename is created based on dstName by genCopyDestName() into the same directory
// of dstName. After this copyFile() is properly completed, we are going to do rename() and
// create dstName at later stage.
//
{
    MNRY_ASSERT(mTmpFileFd >= 0);

    std::string copyDestName = genCopyDestName(dstName);

    int dstFd = open(copyDestName.c_str(), O_WRONLY | O_CREAT, 0666);
    if (dstFd < 0) {
        errMsg = scene_rdl2::util::buildString("Could not create copy destination file '",
                                   copyDestName.c_str(), "' ", strerror(errno));
        return false;
    }

    struct stat stat_src;
    if (fstat(mTmpFileFd, &stat_src) == -1) {
        close(dstFd);
        errMsg = scene_rdl2::util::buildString("Could not get tmpFile size for '",
                                   copyDestName.c_str(), "' ", strerror(errno));
        return false;
    }

    size_t sendSize = stat_src.st_size;
    size_t sentSize = 0;
    while (sendSize > 0) {
        off_t pos = sentSize;
        ssize_t size = sendfile(dstFd, mTmpFileFd, &pos, sendSize);
        if (size == -1) {
            close(dstFd);
            errMsg = scene_rdl2::util::buildString("Failed tmpFile copy to destination location '",
                                       copyDestName.c_str(), "' ", strerror(errno));
            return false;
        }
        if (size == sendSize) break;  // sendfile completed
        sentSize += size;
        sendSize -= size;
    }

    if (close(dstFd) == -1) {
        errMsg = scene_rdl2::util::buildString("Failed close file of '", copyDestName.c_str(), "' ",
                                   strerror(errno));
        return false;
    }

#   ifdef IMAGE_WRITE_DETAIL_MESSAGE
    std::string msg = scene_rdl2::util::buildString("Copied: tmpFile to '", copyDestName.c_str(), "'");
    scene_rdl2::logging::Logger::info(msg);
    if (isatty(STDOUT_FILENO)) std::cout << msg << std::endl;
#   endif // end IMAGE_WRITE_DETAIL_MESSAGE

    return true;
}

bool
ImageWriteCacheTmpFileItem::renameFile(const std::string &dstName, std::string &errMsg) const
//
// Source filename is created based on dstName by genCopyDestName().
// Then rename source filename to dstName.
//    
{
    MNRY_ASSERT(!dstName.empty() && errMsg.empty());

    std::string srcName = genCopyDestName(dstName);
    if (rename(srcName.c_str(), dstName.c_str()) == -1) {
        errMsg = scene_rdl2::util::buildString("Failed to rename from '", srcName.c_str(), "' to '",
                                   dstName.c_str(), "' ", strerror(errno));
        return false;
    }

#   ifdef IMAGE_WRITE_DETAIL_MESSAGE
    std::string msg = scene_rdl2::util::buildString("Renamed: to '", dstName.c_str(), "'");
#   else // else IMAGE_WRITE_DETAIL_MESSAGE
    std::string msg = scene_rdl2::util::buildString("Wrote: '", dstName.c_str(), "'");
#   endif // end !IMAGE_WRITE_DETAIL_MESSAGE
    scene_rdl2::logging::Logger::info(msg);
    // if (isatty(STDOUT_FILENO)) std::cout << msg << std::endl; // useful for debug run from terminal
    
    return true;
}

//------------------------------------------------------------------------------------------

void
ImageWriteCache::setupEnqCacheImageMemory(const int width, const int height)
{
    mWidth = width;
    mHeight = height;

    int tileAlignedHeight = (mHeight + 7) & ~7;

    // I tested several different Y block size but same size of tile Y looks best
    // If you want to change YblockSize, pick 8 (= tile size) * N would be best.
    // This maximizes framebuffer access coherency because our framebuffer memory layout is tiled format.
    mYBlockSize = 8 * 1;
    mYBlockTotal = (tileAlignedHeight + mYBlockSize - 1) / mYBlockSize;

    mDataFullArray.resize(mYBlockTotal);
    mDataHalfArray.resize(mYBlockTotal);

    mCacheQueueFullBuffArray.resize(mYBlockTotal);
    mCacheQueueHalfBuffArray.resize(mYBlockTotal);

    //------------------------------

    for (int yBlockId = 0; yBlockId < mYBlockTotal; ++yBlockId) {
        new(&mCacheQueueFullBuffArray[yBlockId].mCEnq) std::unique_ptr<scene_rdl2::cache::CacheEnqueue>;
        mCacheQueueFullBuffArray[yBlockId].mCEnq.reset(new scene_rdl2::cache::CacheEnqueue(&mDataFullArray[yBlockId]));

        new(&mCacheQueueHalfBuffArray[yBlockId].mCEnq) std::unique_ptr<scene_rdl2::cache::CacheEnqueue>;
        mCacheQueueHalfBuffArray[yBlockId].mCEnq.reset(new scene_rdl2::cache::CacheEnqueue(&mDataHalfArray[yBlockId]));

        // If we have blockData size info (i.e. this ImageWriteCache is not a 1st data and
        // it is 2nd or later), we do pre allocate memory here in order to avoid unnecessary
        // reallocation. However this preallocation should be executed after CacheEnqueue
        // construction has been completed (See CacheEnqueue constructor for more detail).
        if (mBlockDataSizeFull) {
            mDataFullArray[yBlockId].resize(mBlockDataSizeFull);
        }
        if (mBlockDataSizeHalf) {
            mDataHalfArray[yBlockId].resize(mBlockDataSizeHalf);
        }
    }
}

void
ImageWriteCache::expandInternalDataIfNeeded(int yBlockId,
                                            size_t addByteFull, // byte
                                            size_t addByteHalf) // byte
{
    size_t currCapaFull, currCapaHalf; // byte
    getCapacityInternalData(yBlockId, currCapaFull, currCapaHalf);

    size_t currDataSizeFull = enqFullBuff(yBlockId)->currentSize(); // byte
    size_t currDataSizeHalf = enqHalfBuff(yBlockId)->currentSize(); // byte

    size_t remainFull = currCapaFull - currDataSizeFull; // byte
    size_t remainHalf = currCapaHalf - currDataSizeHalf; // byte

    // We only resize if needed.
    if (remainFull < addByteFull) {
        mDataFullArray[yBlockId].resize(currCapaFull + addByteFull - remainFull);
    }
    if (remainHalf < addByteHalf) {
        mDataHalfArray[yBlockId].resize(currCapaHalf + addByteHalf - remainHalf);
    }
}

void
ImageWriteCache::getCapacityInternalData(int yBlockId,
                                         size_t &sizeFull, // byte
                                         size_t &sizeHalf) // byte
{
    sizeFull = mDataFullArray[yBlockId].size();
    sizeHalf = mDataHalfArray[yBlockId].size();
}
                                    
void
ImageWriteCache::freeInternalData()
{
    // Save cache buffer size for future preallocation of imageWriteCache
    calcFinalBlockInternalDataSize();

    // We don't need cache data itself and need to free here
    // We should call shrink_to_fit() in order to exactly free internal memory of vectors.
    mData.clear();
    mData.shrink_to_fit();
    
    mDataFullArray.clear();
    mDataFullArray.shrink_to_fit();
    mDataHalfArray.clear();
    mDataHalfArray.shrink_to_fit();

    mCacheQueueFullBuffArray.clear();
    mCacheQueueFullBuffArray.shrink_to_fit();
    mCacheQueueHalfBuffArray.clear();
    mCacheQueueHalfBuffArray.shrink_to_fit();

    mWidth = 0;
    mHeight = 0;
    mYBlockSize = 0;
    mYBlockTotal = 0;

    mImageWriteCacheSpecs.clear();
    mImageWriteCacheSpecs.shrink_to_fit();

    malloc_trim(0); // Return unused memory from malloc() arena to OS 
}

void
ImageWriteCache::calcFinalBlockInternalDataSize()
{
    if (mDataFullArray.size()) {
        mBlockDataSizeFull = mDataFullArray[0].size();
    }
    if (mDataHalfArray.size()) {
        mBlockDataSizeHalf = mDataHalfArray[0].size();
    }

#   ifdef RUNTIME_VERIFY_BLOCKDATASIZE
    // Image resolution Y is not always divisible by mYBlockSize and the last data might have less
    // data than other yBlockId. So we simply verify that the very first yBlockId data is always
    // equal or bigger than other yBlockId here. This is enough logic to verify current ImageWriteData
    // cache memory allocation logic because all cache data size is the same and can cover last
    // yBlock (might be smaller) as well.
    for (size_t i = 1; i < (size_t)mYBlockTotal; ++i) {
        if (mBlockDataSizeFull < mDataFullArray[i].size() ||
            mBlockDataSizeHalf < mDataHalfArray[i].size()) {
            std::cerr << ">> ImageWriteCache.cc RUNTIME_VERIFY FAILED : getBlockInternalDataSize()"
                      << std::endl;
        }
    }
#   endif // end RUNTIME_VERIFY_BLOCKDATASIZE
}

ImageWriteCacheImageSpec *
ImageWriteCache::newImgSpec()
{
    mImgSpecTbl.emplace_back();
    return &mImgSpecTbl.back();
}

ImageWriteCacheImageSpec *
ImageWriteCache::getImgSpec(int id)
{
    if (id < 0 || mImgSpecTbl.size() <= (size_t)id) return nullptr;
    return &mImgSpecTbl[id];
}

void
ImageWriteCache::setupDeqBuff(const size_t fileId, const size_t subImgId)
{
    const ImageWriteCacheBufferSpecSubImage &buffSpecSubImg =
        mBufferSpec.getBufferSpecFile(fileId).getBufferSpecSubImage(subImgId);

    // Seeking to the data beginning address
    for (int yBlockId = 0; yBlockId < mYBlockTotal; ++yBlockId) {
        int yMin = yBlockId * mYBlockSize;
        int yMax = yMin + mYBlockSize;
        if (yMax > mHeight) yMax = mHeight;
        int currBucketHeight = yMax - yMin;
        int currBucketTotalPix = mWidth * currBucketHeight;

        size_t yBlockFullOffset = currBucketTotalPix * buffSpecSubImg.getPixCacheFullOffset();
        size_t yBlockHalfOffset = currBucketTotalPix * buffSpecSubImg.getPixCacheHalfOffset();

        mCacheQueueFullBuffArray[yBlockId].mCDeq->seekSet(yBlockFullOffset);
        mCacheQueueHalfBuffArray[yBlockId].mCDeq->seekSet(yBlockHalfOffset);
    }
}

ImageWriteCache::ImageWriteCacheTmpFileItemShPtr
ImageWriteCache::setTmpFileItem(const int fileSequenceId,
                                const std::string &tmpFilename,
                                const std::string &checkpointFilename,
                                const std::string &checkpointMultiVersionFilename,
                                const std::string &finalFilename)
{
    mTmpFileItemArray.resize(fileSequenceId + 1);

    mTmpFileItemArray[fileSequenceId] =
        std::make_shared<ImageWriteCacheTmpFileItem>(tmpFilename,
                                                     checkpointFilename,
                                                     checkpointMultiVersionFilename,
                                                     finalFilename);
    return mTmpFileItemArray[fileSequenceId];
}

ImageWriteCache::ImageWriteCacheTmpFileItemShPtr    
ImageWriteCache::getTmpFileItem(const int fileSequenceId)
{
    return mTmpFileItemArray[fileSequenceId];
}

bool
ImageWriteCache::allFinalizeCheckpointFile() const
{
    return allFinalizeFile
        (mResumableOutput,
         [&](const ImageWriteCacheTmpFileItemShPtr item, std::string &errMsg) { // copyCallBack
            return item->copyCheckpointFile(errMsg);
         },
         [&](const ImageWriteCacheTmpFileItemShPtr item, std::string &errMsg) { // renameCallBack
             return item->renameCheckpointFile(errMsg);
         });
}

bool
ImageWriteCache::allFinalizeCheckpointMultiVersionFile() const
{
    return allFinalizeFile
        (mResumableOutput,
         [&](const ImageWriteCacheTmpFileItemShPtr item, std::string &errMsg) { // copyCallBack
            return item->copyCheckpointMultiVersionFile(errMsg);
         },
         [&](const ImageWriteCacheTmpFileItemShPtr item, std::string &errMsg) { // renameCallBack
             return item->renameCheckpointMultiVersionFile(errMsg);
         });
}

bool
ImageWriteCache::allFinalizeFinalFile() const
{
    return allFinalizeFile
        (mResumableOutput,
         [&](const ImageWriteCacheTmpFileItemShPtr item, std::string &errMsg) { // copyCallBack
            return item->copyFinalFile(errMsg);
         },
         [&](const ImageWriteCacheTmpFileItemShPtr item, std::string &errMsg) { // renameCallBack
             return item->renameFinalFile(errMsg);
         });
}

void
ImageWriteCache::outputDataFinalize()
//
// for ImageWriteDriver
//
{
    if (getRenderDriver()->getCheckpointController().isMemorySnapshotActive()) {        
        // signal-based checkpoint action enabled
        ProcKeeper::ProcKeeperShPtr keeper = ProcKeeper::get();
        if (keeper) {
            keeper->startImageWrite(this); // Let keeper know image writing action started
        }
    }

    //
    // We have to consider two stage outputs and overwrite conditions.
    // See comment of CheckpointController::fileOutputMain() for more detail.
    //
    mRenderOutputDriver->writeCheckpointDeq(this, false);
    mRenderOutputDriver->loggingErrorAndInfo(this);

    if (!mTwoStageOutput) {
        if (!mCheckpointOverwrite) {
            // non two stage mode multi version file output
            mRenderOutputDriver->writeCheckpointDeq(this, true);
            mRenderOutputDriver->loggingErrorAndInfo(this);
        }
    } else {
        // TwoStage output is on and data was written out to tmpFile.
        allFinalizeCheckpointFile(); // copy and rename for checkpoint file
        if (!mCheckpointOverwrite) {
            // copy and rename for checkpoint multi version file
            allFinalizeCheckpointMultiVersionFile();
        }
    }
    if (!mPostCheckpointScript.empty()) {
        runPostCheckpointScript();
    }

    if (getRenderDriver()->getCheckpointController().isMemorySnapshotActive()) {        
        // signal-based checkpoint action enabled
        ProcKeeper::ProcKeeperShPtr keeper = ProcKeeper::get();
        if (keeper) {
            keeper->finishImageWrite(); // Let keeper know image writing action finished
#           ifdef DEBUG_MSG_PROGRESS
            std::cerr << ">> ImageWriteCache.cc outputDataFinalize() after finishImageWrite()\n";
#           endif // end DEBUG_MSG_PROGRESS
        }
    }
}
    
void
ImageWriteCache::runPostCheckpointScript()
{
    if (mPostCheckpointScript.empty()) {
        return; // skip post checkpoint script execution
    }

    scene_rdl2::util::LuaScriptRunner luaRun;
    try {
        setupLuaGlobalVariables(luaRun);
        luaRun.runFile(mPostCheckpointScript);
    }
    catch (std::runtime_error &e) {
        std::ostringstream ostr;
        ostr << "LuaScriptRunner runtime ERROR {\n"
             << scene_rdl2::str_util::addIndent(e.what()) << '\n'
             << "}";
        scene_rdl2::logging::Logger::error(ostr.str());
    }
}

void
ImageWriteCache::timeStart()
{
    initAllTimeArray();
    mRecTimeAll.start();
    mRecTime.start();

    updateProgress(ProgressStage::WRITE);
}

void
ImageWriteCache::timeRec(int id)
{
    mTime.rec(mRecTime, (mMode == Mode::DEQ), id, false);
    updateProgress(ProgressStage::WRITE);
}

void
ImageWriteCache::timeEnd()
{
    mTimeAll[(mMode == Mode::DEQ) ? 1 : 0] = mRecTimeAll.end();
    updateProgress(ProgressStage::WRITE);
}

void
ImageWriteCache::timeStartFile()
{
    mRecTimeFile.start();
    updateProgress(ProgressStage::FILE);
}

void
ImageWriteCache::timeRecFile(int id)
{
    mTimeFile.rec(mRecTimeFile, (mMode == Mode::DEQ), id, true);
    updateProgress(ProgressStage::FILE);
}

void
ImageWriteCache::timeStartImage()
{
    mRecTimeImage.start();
    updateProgress(ProgressStage::IMAGE);
}

void
ImageWriteCache::timeRecImage(int id)
{
    mTimeImage.rec(mRecTimeImage, (mMode == Mode::DEQ), id, true);
    updateProgress(ProgressStage::IMAGE);
}

void
ImageWriteCache::timeStartBuffWrite()
{
    mRecTimeBuffWrite.start();
    mCurrBuffWriteProgressCounter.store(0, std::memory_order_relaxed);
    updateProgress(ProgressStage::BUFF0);

#   ifdef DEBUG_MSG_PROGRESS
    std::cerr << ">> ImageWriteCache.cc start buffWrite\n";
#   endif // end DEBUG_MSG_PROGRESS
}

void
ImageWriteCache::timeUpdateBuffWrite(float fraction)
{
    ProgressStage stage;
    if (fraction <= 0.0f) {
        // We categorize very first fraction = 0.0 for buff stage as BUFF0.
        // And categorize second or more fraction value 0.0 as BUFF1.
        // We don't know exactly inside openimageio but BUFF0 is a compression stage and
        // BUFF1 is a phase of beginning stage of write buffer action I guess.
        stage = ((mCurrBuffWriteProgressCounter.load(std::memory_order_relaxed) == 0)
                 ? ProgressStage::BUFF0 : ProgressStage::BUFF1);
    } else {
        stage = ProgressStage::BUFF2;
    }
    mCurrBuffWriteProgressFraction = scene_rdl2::math::clamp(static_cast<int>(fraction * 10.0f), 0, 10);
    mCurrBuffWriteProgressCounter.fetch_add(1, std::memory_order_relaxed);

    updateProgress(stage);

#   ifdef DEBUG_MSG_PROGRESS
    std::cerr << ">> ImageWriteCache.cc update:" << mRecTimeBuffWrite.end()
              << " fraction:" << fraction << '\n';
#   endif // end DEBUG_MSG_PROGRESS
}

void
ImageWriteCache::timeEndBuffWrite()
{
    updateProgress(ProgressStage::BUFF1);

#   ifdef DEBUG_MSG_PROGRESS    
    std::cerr << ">> ImageWriteCache.cc end:" << mRecTimeBuffWrite.end() << '\n';
#   endif // end DEBUG_MSG_PROGRESS
}

std::string
ImageWriteCache::timeShow() const
{
    const bool deq = (mMode == Mode::DEQ);
    float timeAll = mTimeAll[(deq)? 1: 0];
    auto showTimeWithPct = [&](float time) -> std::string {
        float pct = time / timeAll * 100.0f;
        if (pct < 0.0005f) return " -";
        std::ostringstream ostr;
        ostr
        << std::setw(6) << std::fixed << std::setprecision(3) << time << " sec "
        << std::setw(6) << std::fixed << std::setprecision(3) << (time / timeAll * 100.0f) << " %";
        return ostr.str();
    };

    int w = scene_rdl2::str_util::getNumberOfDigits(mTime.size(deq));
    int wF = scene_rdl2::str_util::getNumberOfDigits(mTimeFile.size(deq));
    int wI = scene_rdl2::str_util::getNumberOfDigits(mTimeImage.size(deq));

    std::ostringstream ostr;
    ostr << "timeLog " << ((deq)? "DEQ ": "STD/ENQ ") << "(total:" << mTime.size(deq) << ") {\n";
    for (size_t i = 0; i < mTime.size(deq); ++i) {
        ostr << "  i:" << std::setw(w) << i << ' ' << showTimeWithPct(mTime.get(deq, i)) << '\n';
        if (i == 2) {
            ostr << "  timeFile (total:" << mTimeFile.size(deq) << ") {\n";
            for (size_t j = 0; j < mTimeFile.size(deq); ++j) {
                ostr << "    j:" << std::setw(wF) << j << ' '
                     << showTimeWithPct(mTimeFile.get(deq, j)) << '\n';
                if (j == 3) {
                    ostr << "    timeImage (total:" << mTimeImage.size(deq) << ") {\n";
                    for (size_t k = 0; k < mTimeImage.size(deq); ++k) {
                        ostr << "      k:" << std::setw(wI) << k << ' '
                             << showTimeWithPct(mTimeImage.get(deq, k)) << '\n';
                    }
                    ostr << "    }\n";
                }
            }
            ostr << "  }\n";
        }
    }
    ostr << "} all:" << timeAll << " sec";
    return ostr.str();
}

size_t
ImageWriteCache::memSizeByte() const
{
    // This solution to count memory size is very fragile when updating the code from time to time.
    // Currently this memSizeByte() function is only used for debugging and investigation of memory usage
    // purposes. However don't forget to update this function when you add a new member in this class.

    size_t total = sizeof(*this);

    for (size_t i = 0; i < mErrors.size(); ++i) total += mErrors[i].size();
    for (size_t i = 0; i < mInfos.size(); ++i) total += mInfos[i].size();

    total += mPostCheckpointScript.size();
    for (const auto &itr: mCheckpointFilename) {
        total += itr.size();
    }

    total += mBufferSpec.memSizeByte();

    total += memSizeDataByte();

    for (const auto &itr: mImgSpecTbl) {
        total += itr.memSizeByte();
    }

    total += sizeof(CacheQueue) * mCacheQueueFullBuffArray.size();
    total += sizeof(CacheQueue) * mCacheQueueHalfBuffArray.size();

    for (size_t i = 0; i < mImageWriteCacheSpecs.size(); ++i) {
        total += mImageWriteCacheSpecs[i].memSizeByte();
    }

    for (const auto &itr: mTmpFileItemArray) {
        total += itr->memSizeByte();
    }

    total += mTime.memSizeByte() - sizeof(mTime);
    total += mTimeFile.memSizeByte() - sizeof(mTimeFile);
    total += mTimeImage.memSizeByte() - sizeof(mTimeImage);

    return total;
}

size_t
ImageWriteCache::memSizeDataByte() const
{
    size_t total = mData.size();
    for (int yBlockId = 0; yBlockId < mYBlockTotal; ++yBlockId) {
        total += mDataFullArray[yBlockId].size() + mDataHalfArray[yBlockId].size();
    }
    return total;
}

std::string
ImageWriteCache::showMemUsage() const
{
    size_t totalSize = memSizeByte();
    size_t dataSize = mData.size();
    size_t fullDataSize = 0;
    size_t halfDataSize = 0;
    for (int yBlockId = 0; yBlockId < mYBlockTotal; ++yBlockId) {
        fullDataSize += mDataFullArray[yBlockId].size();
        halfDataSize += mDataHalfArray[yBlockId].size();
    }
    size_t dataSizeTotal = dataSize + fullDataSize + halfDataSize;
    size_t otherSize = totalSize - dataSize;

    std::ostringstream ostr;
    ostr << "ImageWriteCache memory usage {\n"
         << "  cache:" << dataSizeTotal << " byte (" << scene_rdl2::str_util::byteStr(dataSizeTotal) << ") {\n"
         << "    ctrl:" << dataSize << " byte (" << scene_rdl2::str_util::byteStr(dataSize) << ")\n"
         << "    full:" << fullDataSize << " byte (" << scene_rdl2::str_util::byteStr(fullDataSize) << ")\n"
         << "    half:" << halfDataSize << " byte (" << scene_rdl2::str_util::byteStr(halfDataSize) << ")\n"
         << "  }\n"
         << "  other:" << otherSize << " byte (" << scene_rdl2::str_util::byteStr(otherSize) << ")\n"
         << "  total:" << totalSize << " byte (" << scene_rdl2::str_util::byteStr(totalSize) << ")\n"
         << "}";
    return ostr.str();
}

// static function
std::string
ImageWriteCache::modeStr(const Mode &mode)
{
    switch (mode) {
    case Mode::STD: return "STD";
    case Mode::ENQ: return "ENQ";
    case Mode::DEQ: return "DEQ";
    default : return "?";
    }
}

// static function
std::string
ImageWriteCache::typeStr(const Type &type)
{
    switch (type) {
    case Type::OUTPUT: return "OUTPUT";
    case Type::SNAPSHOT_ONLY: return "SNAPSHOT_ONLY";
    default : return "?";
    }
}

//------------------------------------------------------------------------------------------

void
ImageWriteCache::setupLuaGlobalVariables(scene_rdl2::util::LuaScriptRunner &luaRun)
{
    //
    // Setup some checkpoint related parameters to the LUA global variables
    //
    // RenderOptions has option "-rdla_set" and can set Lua global variable before any RDLA run
    // however this -rdla_set does not effect to the post checkpoint script execution.
    // -rdla_set and post checkpoint script are independent.
    //
    luaRun.beginDictionary("checkpoint");
    {
        luaRun.setVarInt("tileSampleTotal", (int)mCheckpointTileSampleTotals);
        luaRun.setArrayString("filename", mCheckpointFilename);
        luaRun.setVarBool("signalInteruption", (mType == Type::SNAPSHOT_ONLY));
    }
    luaRun.endDictionary();
}

bool
ImageWriteCache::allFinalizeFile(bool resumable,
                                 const TmpFileActionCallBack &copyCallBack,
                                 const TmpFileActionCallBack &renameCallBack) const
{
    if (resumable) {
        // This is a resumable output situation. Output files will be used for
        // input of resume rendering.
        // In order to reduce the risk of resume render failure, we do finalize by
        // copy-all and rename-all solution.

        // First of all we are going to copy all tmp files to the destination temp files.
        if (!crawlAllTmpFileItems(copyCallBack)) {
            return false;
        }

        // Then 2nd step is that rename all destination temp files to final name.
        if (!crawlAllTmpFileItems(renameCallBack)) {
            return false;
        }

    } else {
        // This is a non resumable output situation.
        // We do minimize the risk of left over the part file (copied file) in the unexpected event.
        // We are using copy and rename for individual file and loop over all temporary files.
        bool err = false;
        crawlAllTmpFileItems([&](const ImageWriteCacheTmpFileItemShPtr item, std::string &errMsg) {
                // Even some of the tmpFile finalize failed (copy failed or rename failed),
                // we would like to continue other file's finalize.
                if (!copyCallBack(item, errMsg)) { err = true; return true; }
                if (!renameCallBack(item, errMsg)) { err = true; return true; }
                return true;
            });
        if (err) return false;
    }

    return true;
}

bool
ImageWriteCache::crawlAllTmpFileItems(const TmpFileActionCallBack &callBack) const
{
    if (!mTwoStageOutput) return true;

    bool result = true;
    for (size_t id = 0; id < mTmpFileItemArray.size(); ++id) {
        std::string errMsg;
        if (!callBack(mTmpFileItemArray[id], errMsg)) {
            scene_rdl2::logging::Logger::error(errMsg);
            result = false;
        }
    }
    return result;
}

void
ImageWriteCache::updateProgress(const ProgressStage &stage)
{
    mCurrProgressStage.store(stage, std::memory_order_relaxed);
    mProgressCounter.fetch_add(1, std::memory_order_seq_cst);
}

} // namespace rndr
} // namespace moonray
