// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

//
//
#include "SampleIdBuff.h"

#include <iomanip>
#include <sstream>

// Enable naive initial data construction runtime verfiy
//#define RUNTIME_VERIFY

namespace moonray {
namespace rndr {

void
SampleIdBuff::init(const FloatBuffer &srcWeightBuff, unsigned maxSampleId)
{
    initSub(srcWeightBuff.getWidth(), srcWeightBuff.getHeight(), maxSampleId);
    setupSrcWeight(maxSampleId,
                   [&](int sx, int sy) -> unsigned {
                       return getSampleId(srcWeightBuff, sx, sy);
                   });
}

bool
SampleIdBuff::verify(const FloatBuffer &weightBuff) const
//
// Run verification as compared with final weight buffer value
//
{
    unsigned totalError = 0;
    crawlAllPix([&](unsigned sx, unsigned sy) -> bool {
            if (getSampleId(sx, sy) != getSampleId(weightBuff, sx, sy)) {
                if (totalError < 100) { // We only dump the first 100 errors
                    std::cerr << "sx:" << sx << " sy:" << sy
                              << " " << getSampleId(sx, sy)
                              << " != " << getSampleId(weightBuff, sx, sy) << '\n';
                }
                totalError++;
            }
            return true;
        });
    if (totalError > 0) {
        std::cerr << ">> SampleIdBuffer.cc verify failed totalError:" << totalError << " pixels\n";
        return false;
    }
    return true;
}

std::string    
SampleIdBuff::show() const
{
    std::ostringstream ostr;
    ostr << "SampleIdBuff {\n"
         << "  mWidth:" << mWidth << '\n'
         << "  mHeight:" << mHeight << '\n'
         << "  mPixDepth:" << mPixDepth << '\n'
         << "  mArray:0x" << std::hex << reinterpret_cast<uintptr_t>(mArray.get()) << '\n'
         << "}";
    return ostr.str();
}

//------------------------------------------------------------------------------------------

void
SampleIdBuff::initSub(unsigned width, unsigned height, unsigned maxSampleId)
{
    mWidth = width;
    mHeight = height;
    mTiler.~Tiler();
    new(&mTiler) Tiler(mWidth, mHeight);
    mPixDepth = (maxSampleId < 256) ? 1 : (maxSampleId < 65536) ? 2 : 4; // byte

    unsigned alignedWidth = (mWidth + 7) & ~7;
    unsigned alignedHeight = (mHeight + 7) & ~7;
    unsigned memSizeByte = alignedWidth * alignedHeight * mPixDepth;
    mArray.reset(new char[memSizeByte]);

    switch (mPixDepth) {
    case 1 :
        mGetFunc = [&](int sx, int sy) -> unsigned { return getPix<unsigned char>(sx, sy); };
        mSetFunc = [&](int sx, int sy, unsigned v) { setPix<unsigned char>(sx, sy, v); };
        mAtomicReserveFunc = [&](int sx, int sy, unsigned sampleDelta) -> unsigned {
            return atomicReservePix<unsigned char>(sx, sy, sampleDelta);
        };
        break;
    case 2 :
        mGetFunc = [&](int sx, int sy) -> unsigned { return getPix<unsigned short>(sx, sy); };
        mSetFunc = [&](int sx, int sy, unsigned v) { setPix<unsigned short>(sx, sy, v); };
        mAtomicReserveFunc = [&](int sx, int sy, unsigned sampleDelta) -> unsigned {
            return atomicReservePix<unsigned short>(sx, sy, sampleDelta);
        };
        break;
    default :
        mGetFunc = [&](int sx, int sy) -> unsigned { return getPix<unsigned int>(sx, sy); };
        mSetFunc = [&](int sx, int sy, unsigned v) { setPix<unsigned int>(sx, sy, v); };
        mAtomicReserveFunc = [&](int sx, int sy, unsigned sampleDelta) -> unsigned {
            return atomicReservePix<unsigned int>(sx, sy, sampleDelta);
        };
        break;
    }
}

void
SampleIdBuff::setupSrcWeight(unsigned maxSampleId, const GetSrcWeightFunc &getSrcWeightFunc)
{
#ifdef RUNTIME_VERIFY
    unsigned long long srcTotal = 0;
    unsigned long long dstTotal = 0;
#endif // end RUNTIME_VERIFY

    crawlAllPix([&](unsigned sx, unsigned sy) -> bool {
            setSampleId(sx, sy, std::min(getSrcWeightFunc(sx, sy), maxSampleId));

#           ifdef RUNTIME_VERIFY
            srcTotal += std::min(getSrcWeightFunc(sx, sy), maxSampleId);
            dstTotal += getSampleId(sx, sy);
#           endif // end RUNTIME_VERIFY

            return true;
        });

#   ifdef RUNTIME_VERIFY
    std::cerr << ">> SampleIdBuff.cc runtime verify for SampleIdBuff::setupSrcWeight() "
              << ((srcTotal == dstTotal) ? "OK" : "NG") << '\n';
#   endif // end RUNTIME_VERIFY
}

unsigned
SampleIdBuff::getSampleId(const FloatBuffer &weightBuff, unsigned sx, unsigned sy) const
{
    unsigned px, py;
    mTiler.linearToTiledCoords(sx, sy, &px, &py);
    return static_cast<unsigned>(weightBuff.getPixel(px, py));
}

} // namespace rndr
} // namespace moonray

