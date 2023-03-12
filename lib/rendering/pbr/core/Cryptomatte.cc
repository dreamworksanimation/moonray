// Copyright 2023 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#include "Cryptomatte.h"

#include <cstring> // for size_t

namespace moonray {
namespace pbr {

CryptomatteBuffer::CryptomatteBuffer() :
    mWidth(0),
    mHeight(0),
    mFinalized(false)
{
    mPixelMutexes = scene_rdl2::util::alignedMallocArrayCtor<AlignedMutex>(mMutexTileSize * mMutexTileSize);
}

CryptomatteBuffer::~CryptomatteBuffer()
{
    clear();
    scene_rdl2::util::alignedFreeArrayDtor(mPixelMutexes, mMutexTileSize * mMutexTileSize);
}

void
CryptomatteBuffer::init(unsigned width, unsigned height, unsigned numIdChannels)
{
    MNRY_ASSERT_REQUIRE(numIdChannels == 1);     // Production only wants simple 32-bit ids at present

    mWidth = width;
    mHeight = height;
    mPixelEntries.reserve(width * height);
    for (size_t iPixel = 0; iPixel < width * height; iPixel++) {
        mPixelEntries.push_back(PixelEntry());
    }
    mFinalized = false;
}

void
CryptomatteBuffer::clear()
{
    for (size_t iPixel = 0; iPixel < mWidth * mHeight; iPixel++) {
        PixelEntry &pixelEntry = mPixelEntries[iPixel];
        pixelEntry.mFragments.clear();
    }
    mFinalized = false;
}

void
CryptomatteBuffer::addSampleScalar(unsigned x, unsigned y, float sampleId, float weight)
{
    PixelEntry &pixelEntry = mPixelEntries[y * mWidth + x];

    // Iterate over fragments stored at current pixel and see if we can merge the sample in to any of them
    for (Fragment &fragment : pixelEntry.mFragments) {
        if (fragment.mId == sampleId) {
            fragment.mCoverage += weight;
            return;
        }
    }

    // No match, so add a new fragment.
    pixelEntry.mFragments.push_back(Fragment(sampleId, weight));
}

void
CryptomatteBuffer::addSampleVector(unsigned x, unsigned y, float sampleId, float weight)
{
    // Lock in case multiple threads want to add samples to this pixel
    tbb::mutex::scoped_lock lock(mPixelMutexes[getMutexIdx(x, y)]);

    PixelEntry &pixelEntry = mPixelEntries[y * mWidth + x];

    // Iterate over fragments stored at current pixel and see if we can merge the sample in to any of them
    for (Fragment &fragment : pixelEntry.mFragments) {
        if (fragment.mId == sampleId) {
            fragment.mCoverage += weight;
            return;
        }
    }

    // No match, so add a new fragment.
    pixelEntry.mFragments.push_back(Fragment(sampleId, weight));
}

void
CryptomatteBuffer::addFragments(unsigned x, unsigned y, const float *data, int numFragments)
{
    PixelEntry &pixelEntry = mPixelEntries[y * mWidth + x];
    for (int i = 0; i < numFragments; i++) {
        const float id = data[2*i];
        const float coverage = data[2*i + 1];
        pixelEntry.mFragments.push_back(Fragment(id, coverage));
    }
}

void
CryptomatteBuffer::finalize(const scene_rdl2::fb_util::PixelBuffer<unsigned>& samplesCount)
{
    if (mFinalized) {
        return;
    }

    // Sort fragments in each pixel and compute final coverage values
    for (size_t py = 0; py < mHeight; py++) {
        for (size_t px = 0; px < mWidth; px++) {
            // Get number of samples in this pixel
            const unsigned numSamples = samplesCount.getPixel(px, py);
            if (numSamples > 0) {
                // Sort fragments by decreasing coverage
                PixelEntry &pixelEntry = mPixelEntries[py * mWidth + px];
                // We want ties to be broken deterministically so that we don't get false negatives when running Rats
                // test when the fragments are added to the pixel entry in different orders.
                pixelEntry.mFragments.sort([](const Fragment &f0, const Fragment &f1) {
                    if (f0.mCoverage != f1.mCoverage) {
                        return f0.mCoverage > f1.mCoverage;
                    }
                    return f0.mId > f1.mId;
                });

                // Normalize coverages so that they sum to 1 over the pixel
                float recipNumSamples = 1.0f / static_cast<float>(numSamples);
                for (Fragment &fragment : pixelEntry.mFragments) {
                    fragment.mCoverage *= recipNumSamples;
                }
            }
        }
    }

    mFinalized = true;
}

// This function simply undoes the effect of finalize(), so the coverage values will be ready for further
// accumulation following a checkpoint
void
CryptomatteBuffer::unfinalize(const scene_rdl2::fb_util::PixelBuffer<unsigned>& samplesCount)
{
    if (!mFinalized) {
        return;
    }

    for (size_t py = 0; py < mHeight; py++) {
        for (size_t px = 0; px < mWidth; px++) {
            const unsigned numSamples = samplesCount.getPixel(px, py);
            if (numSamples > 0) {
                PixelEntry &pixelEntry = mPixelEntries[py * mWidth + px];
                float numSamplesFloat = static_cast<float>(numSamples);
                for (Fragment &fragment : pixelEntry.mFragments) {
                    // Convert back from a fraction to a count. The round() function is needed here
                    // because the product might not be exactly an integer due to truncation error.
                    fragment.mCoverage = std::round(fragment.mCoverage * numSamplesFloat);
                }
            }
        }
    }

    mFinalized = false;
}

void
CryptomatteBuffer::outputFragments(unsigned x, unsigned y, int numLayers, float *dest) const
{
    // Clamp numLayers so the output string will contain 2 digits 00-99 (per Cryptomatte spec)
    if (numLayers < 1)   numLayers = 1;
    if (numLayers > 100) numLayers = 100;

    // Output 2 fragments per layer: one pair goes to the R,G channels, the other to the B,A channels.
    const PixelEntry &pixelEntry = mPixelEntries[y * mWidth + x];
    int rank = 0;
    for (const Fragment &fragment : pixelEntry.mFragments) {
        *dest++ = fragment.mId;
        *dest++ = fragment.mCoverage;
        if (++rank >= 2 * numLayers) {
            break;
        }
    }

    // Supplement requested layers with zeros
    for ( ; rank < 2 * numLayers; rank++)
    {
        *dest++ = 0.f;
        *dest++ = 0.f;
    }
}

// Useful for debugging
void
CryptomatteBuffer::printAllPixelEntries() const
{
    for (size_t py = 0; py < mHeight; py++) {
        for (size_t px = 0; px < mWidth; px++) {
            printFragments(px, py);
        }
    }
}

void
CryptomatteBuffer::printFragments(unsigned x, unsigned y) const
{
    const PixelEntry &pixelEntry = mPixelEntries[y * mWidth + x];
    unsigned numFragments = pixelEntry.mFragments.size();

    printf("(%u, %u): %u fragments; ", x, y, numFragments);
    int iFragment = 0;
    for (const Fragment &fragment : pixelEntry.mFragments) {
        printf("Fragment %d: ", iFragment++);
        printf("Coverage = %g, ", fragment.mCoverage);
        printf("Id = %g, ", fragment.mId);
    }
    printf("\n");
}

}
}

