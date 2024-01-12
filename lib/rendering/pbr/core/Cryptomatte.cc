// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#include "Cryptomatte.h"

#include <cstring> // for size_t

namespace moonray {
namespace pbr {

CryptomatteBuffer::CryptomatteBuffer() :
    mWidth(0),
    mHeight(0),
    mFinalized(false),
    mMultiPresenceOn(false)
{
    mPixelMutexes = scene_rdl2::util::alignedMallocArrayCtor<AlignedMutex>(mMutexTileSize * mMutexTileSize);
}

CryptomatteBuffer::~CryptomatteBuffer()
{
    clear();
    scene_rdl2::util::alignedFreeArrayDtor(mPixelMutexes, mMutexTileSize * mMutexTileSize);
}

void CryptomatteBuffer::init(unsigned width, unsigned height, unsigned numIdChannels, bool multiPresenceOn)
{
    MNRY_ASSERT_REQUIRE(numIdChannels == 1);     // Production only wants simple 32-bit ids at present

    mWidth = width;
    mHeight = height;
    for (int iType = 0; iType < NUM_CRYPTOMATTE_TYPES; iType++) {
        mPixelEntries[iType].reserve(width * height);
        for (size_t iPixel = 0; iPixel < width * height; iPixel++) {
            mPixelEntries[iType].push_back(PixelEntry());
        }
    }
    mFinalized = false;
    mMultiPresenceOn = multiPresenceOn;
}

void CryptomatteBuffer::clear()
{
    for (int iType = 0; iType < NUM_CRYPTOMATTE_TYPES; iType++) {
        for (size_t iPixel = 0; iPixel < mWidth * mHeight; iPixel++) {
            PixelEntry &pixelEntry = mPixelEntries[iType][iPixel];
            pixelEntry.mFragments.clear();
        }
    }
    mFinalized = false;
}

void CryptomatteBuffer::addSampleScalar(unsigned x, unsigned y, float sampleId, float weight, 
                                        const scene_rdl2::math::Vec3f& position, 
                                        const scene_rdl2::math::Vec3f& normal,
                                        const scene_rdl2::math::Color4& beauty,
                                        const scene_rdl2::math::Vec3f refP,
                                        const scene_rdl2::math::Vec3f refN,
                                        const scene_rdl2::math::Vec2f uv,
                                        unsigned presenceDepth,
                                        int cryptoType)
{
    PixelEntry &pixelEntry = mPixelEntries[cryptoType][y * mWidth + x];

    // Iterate over fragments stored at current pixel and see if we can merge the sample in to any of them
    for (Fragment &fragment : pixelEntry.mFragments) {
        // if multi presence is on, we treat each presence bounce as a separate cryptomatte fragment
        bool fragMatches = mMultiPresenceOn ? fragment.mId == sampleId && fragment.mPresenceDepth == presenceDepth
                                            : fragment.mId == sampleId;
        if (fragMatches) {
            fragment.mCoverage += weight;
            fragment.mPosition += position;
            fragment.mNormal += normal;
            fragment.mBeauty += beauty;
            fragment.mRefP += refP;
            fragment.mRefN += refN;
            fragment.mUV += uv;
            fragment.mNumSamples++;
            return;
        }
    }

    // No match, so add a new fragment.
    pixelEntry.mFragments.push_back(Fragment(sampleId, weight, position, normal, beauty, 
                                             refP, refN, uv, presenceDepth));
}

void CryptomatteBuffer::addSampleVector(unsigned x, unsigned y, float sampleId, float weight, 
                                        const scene_rdl2::math::Vec3f& position,
                                        const scene_rdl2::math::Vec3f& normal,
                                        const scene_rdl2::math::Color4& beauty,
                                        const scene_rdl2::math::Vec3f refP,
                                        const scene_rdl2::math::Vec3f refN,
                                        const scene_rdl2::math::Vec2f uv,
                                        unsigned presenceDepth,
                                        bool incrementSamples)
{
    // Lock in case multiple threads want to add samples to this pixel
    tbb::mutex::scoped_lock lock(mPixelMutexes[getMutexIdx(x, y)]);

    PixelEntry &pixelEntry = mPixelEntries[CRYPTOMATTE_TYPE_REGULAR][y * mWidth + x];

    // Iterate over fragments stored at current pixel and see if we can merge the sample in to any of them
    for (Fragment &fragment : pixelEntry.mFragments) {
        // if multi presence is on, we treat each presence bounce as a separate cryptomatte fragment
        bool fragMatches = mMultiPresenceOn ? fragment.mId == sampleId && fragment.mPresenceDepth == presenceDepth
                                            : fragment.mId == sampleId;
        if (fragMatches) {
            fragment.mCoverage += weight;
            fragment.mPosition += position;
            fragment.mNormal += normal;
            fragment.mBeauty += beauty;
            fragment.mRefP += refP;
            fragment.mRefN += refN;
            fragment.mUV += uv;
            if (incrementSamples) fragment.mNumSamples++;
            return;
        }
    }

    // No match, so add a new fragment.
    pixelEntry.mFragments.push_back(Fragment(sampleId, weight, position, normal, beauty, 
                                             refP, refN, uv, presenceDepth));
}

void CryptomatteBuffer::addBeautySampleVector(unsigned x, unsigned y, 
                                              float id, const scene_rdl2::math::Color4& beauty, 
                                              unsigned depth) 
{
    // Only adds beauty, with all other data zeroed out
    // We only call this function when dealing with presence paths in vector mode. When a presence path is encountered, 
    // we must add the fragment data in shadeBundleHandler. However, we do not have the beauty data at that stage, so 
    // we need to use this function to add it in Film::addSampleBundleHandlerHelper. We don't want to increment the 
    // number of samples (which we use to average position/normal data) because we already added this fragment in 
    // shadeBundleHandler, and this is basically an addendum, where we add no new position/normal data. We pass in false
    // to the incrementSamples parameter in order to suppress this incrementation 
    addSampleVector(x, y, id, 0.f, scene_rdl2::math::Vec3f(0.f), scene_rdl2::math::Vec3f(0.f), beauty,
                    scene_rdl2::math::Vec3f(0.f), scene_rdl2::math::Vec3f(0.f), scene_rdl2::math::Vec2f(0.f),
                    depth, false);
}

void CryptomatteBuffer::finalize(const scene_rdl2::fb_util::PixelBuffer<unsigned>& samplesCount) 
{
    if (mFinalized) {
        return;
    }

    // Sort fragments in each pixel and compute final coverage values
    for (int cryptoType = 0; cryptoType < NUM_CRYPTOMATTE_TYPES; cryptoType++) {
        for (size_t py = 0; py < mHeight; py++) {
            for (size_t px = 0; px < mWidth; px++) {
                // Get number of samples in this pixel
                const unsigned numSamples = samplesCount.getPixel(px, py);

                if (numSamples > 0) {
                    // Sort fragments by decreasing coverage
                    PixelEntry &pixelEntry = mPixelEntries[cryptoType][py * mWidth + px];
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
                        if (fragment.mNumSamples > 0) {
                            float recipFragNumSamples = 1.f / static_cast<float>(fragment.mNumSamples);
                            // take average of position/normal/beauty data over the number of samples taken for the fragment
                            fragment.mPosition *= recipFragNumSamples;
                            fragment.mNormal   *= recipFragNumSamples;
                            fragment.mBeauty   *= recipFragNumSamples;
                            fragment.mRefP     *= recipFragNumSamples;
                            fragment.mRefN     *= recipFragNumSamples;
                            fragment.mUV       *= recipFragNumSamples;
                        }
                    }
                }
            }
        }
    }

    mFinalized = true;
}

void CryptomatteBuffer::outputFragments(unsigned x, unsigned y, 
                                        int numLayers, float *dest, 
                                        const scene_rdl2::rdl2::RenderOutput& ro) const
{
    // Clamp numLayers so the output string will contain 2 digits 00-99 (per Cryptomatte spec)
    if (numLayers < 1)   numLayers = 1;
    if (numLayers > 100) numLayers = 100;

    for (int cryptoType = 0; cryptoType < NUM_CRYPTOMATTE_TYPES; cryptoType++) {

        if (cryptoType == CRYPTOMATTE_TYPE_REFRACTED && !ro.getCryptomatteEnableRefract()) {
            // Don't output the refracted cryptomatte channels to the render output if it doesn't
            // want them.
            continue;
        }

        // Output 2 fragments per layer: one pair goes to the R,G channels, the other to the B,A channels.
        const PixelEntry &pixelEntry = mPixelEntries[cryptoType][y * mWidth + x];

        // Sum up the total coverage for all fragments
        float totalCoverage = 0.f;
        for (const Fragment &fragment : pixelEntry.mFragments) {
            totalCoverage += fragment.mCoverage;
        }

        // Sum up the total coverage for the fragments we intend to output
        int numOutputFragments = 0;
        float totalOutputCoverage = 0.f;
        for (const Fragment &fragment : pixelEntry.mFragments) {
            totalOutputCoverage += fragment.mCoverage;
            if (++numOutputFragments >= 2 * numLayers) {
                break;
            }
        }

        // Compute scaling factor to compensate for coverage we are throwing away
        float coverageScale = totalCoverage / totalOutputCoverage;

        int numFragments = 0;
        for (const Fragment &fragment : pixelEntry.mFragments) {
            *dest++ = fragment.mId;
            *dest++ = fragment.mCoverage * coverageScale;
            // ensure numFragments added to memory isn't larger than max number 
            // of fragments supported
            if (++numFragments >= 2 * numLayers) {
                break;
            }
        }

        // Supplement requested layers with zeros
        for ( ; numFragments < 2 * numLayers; numFragments++)
        {
            *dest++ = 0.f;
            *dest++ = 0.f;
        }

        // ------------------ Output extra data (positions, normals, beauty) -------------------------- //
        if (!ro.cryptomatteHasExtraOutput()) continue;

        numFragments = 0;
        // Output positions, normals, and beauty
        for (const Fragment &fragment : pixelEntry.mFragments) {

            if (ro.getCryptomatteOutputPositions()) {
                *dest++ = fragment.mPosition.x;
                *dest++ = fragment.mPosition.y;
                *dest++ = fragment.mPosition.z;
                *dest++ = 0.0f;
            }
            if (ro.getCryptomatteOutputNormals()) {
                *dest++ = fragment.mNormal.x;
                *dest++ = fragment.mNormal.y;
                *dest++ = fragment.mNormal.z;
                *dest++ = 0.0f;
            }
            if (ro.getCryptomatteOutputBeauty()) {
                *dest++ = fragment.mBeauty.r;
                *dest++ = fragment.mBeauty.g;
                *dest++ = fragment.mBeauty.b;
                *dest++ = fragment.mBeauty.a;
            }
            if (ro.getCryptomatteOutputRefP()) {
                *dest++ = fragment.mRefP.x;
                *dest++ = fragment.mRefP.y;
                *dest++ = fragment.mRefP.z;
                *dest++ = 0.0f;
            }
            if (ro.getCryptomatteOutputRefN()) {
                *dest++ = fragment.mRefN.x;
                *dest++ = fragment.mRefN.y;
                *dest++ = fragment.mRefN.z;
                *dest++ = 0.0f;
            }
            if (ro.getCryptomatteOutputUV()) {
                *dest++ = fragment.mUV.x;
                *dest++ = fragment.mUV.y;
            }
            if (ro.getCryptomatteSupportResumeRender()) {
                // need the following to reconstruct the data during resume/checkpoint rendering
                *dest++ = fragment.mPresenceDepth;
                *dest++ = fragment.mNumSamples;
            }

            // ensure numFragments added to memory isn't larger than max number 
            // of fragments supported
            if (++numFragments >= 2 * numLayers) break;
        }

        unsigned numExtraChannels = ro.getCryptomatteNumExtraChannels();
        // Supplement the extra data layers with zeros
        for (int i = numFragments * numExtraChannels ; i < 2 * numLayers * numExtraChannels; ++i) {
            *dest++ = 0.0f;
        }
    }
}

// This function simply undoes the effect of finalize(), so the coverage values will be ready for further
// accumulation following a checkpoint
void CryptomatteBuffer::unfinalize(const scene_rdl2::fb_util::PixelBuffer<unsigned>& samplesCount)
{
    if (!mFinalized) {
        return;
    }

    for (int cryptoType = 0; cryptoType < NUM_CRYPTOMATTE_TYPES; cryptoType++) {
        for (size_t py = 0; py < mHeight; py++) {
            for (size_t px = 0; px < mWidth; px++) {
                const unsigned numSamples = samplesCount.getPixel(px, py);
                if (numSamples > 0) {
                    PixelEntry &pixelEntry = mPixelEntries[cryptoType][py * mWidth + px];
                    float numSamplesFloat = static_cast<float>(numSamples);
                    for (Fragment &fragment : pixelEntry.mFragments) {
                        fragment.mCoverage = fragment.mCoverage * numSamplesFloat;
                        // multiply by the number of fragment samples to get the accumulated (not averaged) data
                        fragment.mPosition = fragment.mPosition * fragment.mNumSamples;
                        fragment.mNormal   = fragment.mNormal   * fragment.mNumSamples;
                        fragment.mBeauty   = fragment.mBeauty   * fragment.mNumSamples;
                        fragment.mRefP     = fragment.mRefP     * fragment.mNumSamples;
                        fragment.mRefN     = fragment.mRefN     * fragment.mNumSamples;
                        fragment.mUV       = fragment.mUV       * static_cast<float>(fragment.mNumSamples);
                    }
                }
            }
        }
    }

    mFinalized = false;
}

void CryptomatteBuffer::addFragments(unsigned x, unsigned y, 
                                     const scene_rdl2::rdl2::RenderOutput& ro,
                                     const float *idAndCoverageData,
                                     const float *positionData,
                                     const float *normalData,
                                     const float *beautyData,
                                     const float *refPData,
                                     const float *refNData,
                                     const float *uvData,
                                     const float *resumeRenderSupportData)
{
    // The number of fragments will always equal the cryptomatte depth when we reconstruct the data. The cryptomatte 
    // depth represents the max number of fragments. Though there will not necessarily be that many fragments, the 
    // channels will still exist (we pad the un-filled channels with zeroes)
    int numFragments = ro.getCryptomatteDepth(); 

    for (int cryptoType = 0; cryptoType < NUM_CRYPTOMATTE_TYPES; cryptoType++) {

        if (cryptoType == CRYPTOMATTE_TYPE_REFRACTED && !ro.getCryptomatteEnableRefract()) {
            // Don't input the refracted cryptomatte channels to the render output if it doesn't
            // want them.
            continue;
        }

        PixelEntry &pixelEntry = mPixelEntries[cryptoType][y * mWidth + x];
        for (int i = 0; i < numFragments; i++) {
            const float id       = idAndCoverageData[0];
            const float coverage = idAndCoverageData[1];
            idAndCoverageData += 2;

            // -------------- Reconstruct extra data (positions, normals, beauty) ------------------ //
            scene_rdl2::math::Vec3f position(0.f);
            scene_rdl2::math::Vec3f normal(0.f);
            scene_rdl2::math::Color4 beauty(0.f);
            scene_rdl2::math::Vec3f refP(0.f);
            scene_rdl2::math::Vec3f refN(0.f);
            scene_rdl2::math::Vec2f uv(0.f);
            unsigned presenceDepth = 0;
            unsigned numFragSamples = 1;

            if (ro.getCryptomatteOutputPositions()) {
                position = scene_rdl2::math::Vec3f(positionData);
                positionData += 4;
            }
            if (ro.getCryptomatteOutputNormals()) {
                normal = scene_rdl2::math::Vec3f(normalData);
                normalData += 4;
            }
            if (ro.getCryptomatteOutputBeauty()) {
                beauty = scene_rdl2::math::Color4(beautyData[0], beautyData[1], beautyData[2], beautyData[3]);
                beautyData += 4;
            }
            if (ro.getCryptomatteOutputRefP()) {
                refP = scene_rdl2::math::Vec3f(refPData);
                refPData += 4;
            }
            if (ro.getCryptomatteOutputRefN()) {
                refN = scene_rdl2::math::Vec3f(refNData);
                refNData += 4;
            }
            if (ro.getCryptomatteOutputUV()) {
                uv = scene_rdl2::math::Vec2f(uvData);
                uvData += 2;
            }
            if (ro.getCryptomatteSupportResumeRender()) {
                presenceDepth  = static_cast<unsigned>(resumeRenderSupportData[0]);
                numFragSamples = static_cast<unsigned>(resumeRenderSupportData[1]);
                resumeRenderSupportData += 2;
            }

            if (coverage > 0.f) {
                pixelEntry.mFragments.push_back(Fragment(id, coverage, position, normal, beauty, 
                                                         refP, refN, uv,
                                                         presenceDepth, numFragSamples));
            }
        }
    }
}

// Useful for debugging
void CryptomatteBuffer::printAllPixelEntries() const
{
    for (size_t py = 0; py < mHeight; py++) {
        for (size_t px = 0; px < mWidth; px++) {
            printFragments(px, py, CRYPTOMATTE_TYPE_REGULAR);
            printFragments(px, py, CRYPTOMATTE_TYPE_REFRACTED);
        }
    }
}

void CryptomatteBuffer::printFragments(unsigned x, unsigned y, int cryptoType) const
{
    const PixelEntry &pixelEntry = mPixelEntries[cryptoType][y * mWidth + x];
    unsigned numFragments = pixelEntry.mFragments.size();

    printf("(%u, %u): %u fragments; ", x, y, numFragments);
    int iFragment = 0;
    for (const Fragment &fragment : pixelEntry.mFragments) {
        printf("Fragment %d: ", iFragment++);
        printf("Coverage = %g, ", fragment.mCoverage);
        printf("Id = %g, ", fragment.mId);
        printf("Position = (%g, %g, %g), ", fragment.mPosition.x, fragment.mPosition.y, fragment.mPosition.z);
        printf("Normal = (%g, %g, %g), ", fragment.mNormal.x, fragment.mNormal.y, fragment.mNormal.z);
        printf("Beauty = (%g, %g, %g, %g), ", fragment.mBeauty.r, fragment.mBeauty.g, fragment.mBeauty.b, fragment.mBeauty.a);
        printf("RefP = (%g, %g, %g), ", fragment.mRefP.x, fragment.mRefP.y, fragment.mRefP.z);
        printf("RefN = (%g, %g, %g), ", fragment.mRefN.x, fragment.mRefN.y, fragment.mRefN.z);
        printf("UV = (%g, %g), ", fragment.mUV.x, fragment.mUV.y);
        printf("PresenceDepth = %u, ", fragment.mPresenceDepth);
        printf("# Frag Samples = %u", fragment.mNumSamples);
        printf("\n");
    }
    printf("\n");
}

}
}

