// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#include "DeepBuffer.h"

#include <moonray/rendering/pbr/integrator/VolumeProperties.h>
#include <scene_rdl2/common/except/exceptions.h>

#include <OpenEXR/ImfBoxAttribute.h>
#include <OpenEXR/ImfChromaticitiesAttribute.h>
#include <OpenEXR/ImfDeepImageIO.h>
#include <OpenEXR/ImfDoubleAttribute.h>
#include <OpenEXR/ImfFloatAttribute.h>
#include <OpenEXR/ImfIntAttribute.h>
#include <OpenEXR/ImfMatrixAttribute.h>
#include <OpenEXR/ImfStringAttribute.h>
#include <OpenEXR/ImfVecAttribute.h>

#include <fstream>


namespace moonray {
namespace pbr {

const unsigned DeepBuffer::VolumePixelBuffer::mMaxPixelSamples;

DeepBuffer::DeepBuffer() :
    mWidth(0),
    mHeight(0),
    mAovSchema(nullptr),
    mNumChannels(0),
    mCurvatureTolerance(1.f),
    mZTolerance(2.f),
    mVolCompressionRes(10),
    mSubpixelRes(0),
    mSamplesPerPixel(0),
    mNumRenderThreads(0),
    mFormat(DeepFormat::OpenDCX2_0),
    mMaxLayers(1)
{
    mPixelMutex = scene_rdl2::util::alignedMallocArrayCtor<AlignedMutex>(mMutexTileSize * mMutexTileSize);
}

DeepBuffer::~DeepBuffer()
{
    clear();
    scene_rdl2::util::alignedFreeArrayDtor(mPixelMutex, mMutexTileSize * mMutexTileSize);
}

void
DeepBuffer::clear()
{
    // iterate over the hard surface segment linked lists and free the segments
    for (size_t i = 0; i < mWidth * mHeight; i++) {
        for (size_t layer = 0; layer < mMaxLayers; layer++) {
            HardSurfaceSegment *current = mHardSurfaceSegments[layer][i];
            while (current) {
                HardSurfaceSegment *next = current->mNext;
                free(current);
                current = next;
            }
            mHardSurfaceSegments[layer][i] = nullptr;
        }
    }

    // iterate over the volume output segment linked lists and free the segments
    for (size_t i = 0; i < mWidth * mHeight; i++) {
        VolumeOutputSegment *current = mVolumeOutputSegments[i];
        while (current) {
            VolumeOutputSegment *next = current->mNext;
            free(current);
            current = next;
        }
        mVolumeOutputSegments[i] = nullptr;
    }

    for (unsigned i = 0; i < mNumRenderThreads; i++) {
        mVolumePixelBuffers[i].clear();
    }
}

void
DeepBuffer::initDeep(unsigned width, unsigned height,
                     DeepFormat format,
                     float curvatureTolerance,
                     float zTolerance,
                     unsigned volCompressionRes,
                     unsigned numRenderThreads,
                     const pbr::AovSchema &aovSchema,
                     const std::vector<std::string>& deepIDChannels,
                     int maxLayers)
{
    mWidth = width;
    mHeight = height;
    mFormat = format;
    mSubpixelRes = 8;
    mSamplesPerPixel = 64;
    mDeepIDChannels = deepIDChannels;
    mNumRenderThreads = numRenderThreads;
    mAovSchema = &aovSchema;
    mMaxLayers = maxLayers;

    mNumChannels = 3;  // Beauty RGB is always channels 0-2

    // examine the aovs and add them to the total number of channels
    for (size_t i = 0; i < aovSchema.size(); i++) {
        unsigned numFloats = aovSchema[i].numChannels();
        mNumChannels += numFloats;
        mChannelsPerAOV.push_back(numFloats);
    }

    // curvatureTolerance is the max angle in degrees, mCurvatureTolerance is the
    // dot product of two vectors differing by that angle: cos(curvatureTolerance)
    mCurvatureTolerance = cos(curvatureTolerance * M_PI / 180.f);

    mZTolerance = zTolerance;

    mVolCompressionRes = volCompressionRes;
    if (mVolCompressionRes == 0) {
        // zero is invalid... lowest res is 1
        mVolCompressionRes = 1;
    }

    mHardSurfaceSegments.resize(mMaxLayers);
    for (size_t layer = 0; layer < mMaxLayers; layer++) {
        mHardSurfaceSegments[layer].resize(mWidth * mHeight, nullptr);
    }

    mVolumeOutputSegments.resize(mWidth * mHeight, nullptr);

    mVolumePixelBuffers.resize(mNumRenderThreads);
    for (unsigned i = 0; i < mNumRenderThreads; i++) {
        mVolumePixelBuffers[i].mSampleList = nullptr;
        mVolumePixelBuffers[i].clear();
    }
}

void
DeepBuffer::addSample(pbr::TLState *pbrTls,
                      unsigned x, unsigned y, float sx, float sy,
                      int layer,
                      const float *deepIDs, float t, float rayZ,
                      const scene_rdl2::math::Vec3f& normal, float alpha,
                      const scene_rdl2::math::Color& rgb, const float *aovs,
                      float scale, float weight)
{
    // transform the rgb + aov params into channel data that the regular addSample()
    //  function can accept

    int channels[mNumChannels];
    for (int i = 0; i < mNumChannels; i++) {
        // we're writing all the channels in this case so the channel list is
        //  just 0..mNumChannels-1
        channels[i] = i;
    }

    // copy the beauty and aov data into the channel values array
    float values[mNumChannels];
    values[0] = rgb[0];
    values[1] = rgb[1];
    values[2] = rgb[2];
    for (int i = 3; i < mNumChannels; i++) {
        values[i] = aovs[i - 3];
    }

    addSample(pbrTls, x, y, sx, sy, layer, deepIDs, t, rayZ, normal, alpha,
              channels, mNumChannels, values, scale, weight);
}

void
DeepBuffer::addSample(pbr::TLState *pbrTls,
                      unsigned x, unsigned y, float sx, float sy,
                      int layer,
                      const float *deepIDs, float t, float rayZ,
                      const scene_rdl2::math::Vec3f& normal, float alpha,
                      const int *channels, int numChannels,
                      const float *values,
                      float scale, float weight)
{
    // Duplicate the sample data if the subpixel resolution is < 8.

    unsigned subpixelX = (uint8_t)(sx * 8.f);
    unsigned subpixelY = (uint8_t)(sy * 8.f);

    // Make sure we're normalized.
    scene_rdl2::math::Vec3f nnormal = normal;
    nnormal.safeNormalize();

    // Don't need to lock in scalar mode
    auto addSample8x8Func = pbrTls->mFs->mExecutionMode == mcrt_common::ExecutionMode::SCALAR ?
        &DeepBuffer::addSample8x8 :
        &DeepBuffer::addSample8x8Safe;

    if (mFormat == DeepFormat::OpenEXR2_0) {
        // No sample duplication is needed.  Although we are still calling the 8x8
        // function, and an 8x8 mask is being constructed, we ignore it when we
        // output the deep file.
        (this->*addSample8x8Func)(x, y, subpixelX, subpixelY, layer, deepIDs, t, rayZ, nnormal,
                                  alpha, channels, numChannels, values, scale, weight);
    } else {
        // If the subpixel res is less than 8, we need to duplicate samples to
        //  fill the subpixel mask.
        switch (mSubpixelRes) {
        case 8:  // 8x8, no sample duplication needed
            (this->*addSample8x8Func)(x, y, subpixelX, subpixelY, layer, deepIDs, t, rayZ, nnormal,
                                      alpha, channels, numChannels, values, scale, weight);
        break;
        case 4:  // 4x4, duplicate samples
        {
            unsigned startX = subpixelX & 0x06;
            unsigned startY = subpixelY & 0x06;
            for (int ssy = startY; ssy < startY + 2; ssy++) {
                for (int ssx = startX; ssx < startX + 2; ssx++) {
                    (this->*addSample8x8Func)(x, y, ssx, ssy, layer, deepIDs, t, rayZ, nnormal,
                                     alpha, channels, numChannels, values, scale, weight);
                }
            }
        }
        break;
        case 2:  // 2x2, duplicate samples
        {
            unsigned startX = subpixelX & 0x04;
            unsigned startY = subpixelY & 0x04;
            for (int ssy = startY; ssy < startY + 4; ssy++) {
                for (int ssx = startX; ssx < startX + 4; ssx++) {
                    (this->*addSample8x8Func)(x, y, ssx, ssy, layer, deepIDs, t, rayZ, nnormal,
                                              alpha, channels, numChannels, values, scale, weight);
                }
            }
        }
        break;
        case 1:  // 1x1, duplicate samples
            for (int ssy = 0; ssy < 8; ssy++) {
                for (int ssx = 0; ssx < 8; ssx++) {
                    (this->*addSample8x8Func)(x, y, ssx, ssy, layer, deepIDs, t, rayZ, nnormal,
                                              alpha, channels, numChannels, values, scale, weight);
                }
            }
        break;
        default:
            MNRY_ASSERT(0);
        }
    }
}

// No internal locking, suitable for scalar mode execution.
void
DeepBuffer::addSample8x8(unsigned x, unsigned y, unsigned subpixelX, unsigned subpixelY,
                         int layer,
                         const float *ids, float t, float rayZ,
                         const scene_rdl2::math::Vec3f& normal, float alpha,
                         const int *channels, int numChannels,
                         const float *values,
                         float scale, float weight)
{
    // This is the core of the deep images implementation.

    // Pixel and 8x8 subpixel indices
    int idx = y * mWidth + x;
    int sidx = subpixelY * 8 + subpixelX;

    // Iterate over existing segments and see if we can merge in.
    HardSurfaceSegment *current = mHardSurfaceSegments[layer][idx];
    HardSurfaceSegment *prev = nullptr;
    while (current) {
        bool idsMatch = true;
        for (size_t i = 0; i < mDeepIDChannels.size(); i++) {
            // Look at all the deep IDs for this sample and see if they match
            //  the IDs of this segment.
            if (ids[i] != getID(current, i)) {
                idsMatch = false;
                break;
            }
        }
        // We can merge this sample into this segment if the following is satisfied:
        // 1) The deep IDs match.
        // 2) The sample's normal isn't too far off the segment's normal.
        // 3) The sample's z value is within mZTolerance/2 of the first sample
        //    that went into the segment.  This limits the spread of z values to
        //    mZTolerance.
        if (idsMatch &&
            scene_rdl2::math::dot(normal, current->mNormal) > mCurvatureTolerance &&
            fabs(current->mRayZ * (current->mTFirst - t)) < mZTolerance * 0.5f) {
            if (t < current->mTFront) current->mTFront = t;
            if (t > current->mTBack) current->mTBack = t;
            current->mMask |= (1ull << sidx);
            current->mAlpha += alpha * scale;
            for (int i = 0; i < numChannels; i++) {
                getChannel(current, channels[i]) += values[i] * scale;
            }
            current->mTotalWeight += weight;
            return;
        }
        prev = current;
        current = current->mNext;
    }

    // At the end of the list without being able to merge.  Append a new segment.
    HardSurfaceSegment *newSegment = (HardSurfaceSegment*)malloc(getHardSurfaceSegmentSize());
    // Intuitively, doing a malloc() here which needs to lock the heap is a bad idea
    //  as it would block other threads adding samples here.  But... this code rarely
    //  gets called because the vast majority of the samples are merged into existing
    //  segments.  The number of segments allocated here is in the same ballpark as
    //  the number of pixels in the image, as some pixels will be empty (no segments),
    //  most pixels will have one segment (as they overlap only one scene object),
    //  and a small percentage of pixels will have multiple segments, such as at
    //  object edges.  Allocating a few million small objects on the heap in this way
    //  doesn't seem to measurably affect rendering time.
    // If we have massive numbers of threads this might start to become a problem,
    //  but we could pre-allocate blocks of segments per thread, and the new segment
    //  would be pulled from the current thread's preallocated segments.  Testing
    //  hasn't shown a need for this yet.
    // Discussing with Mark: We can add a frame arena that persists for the lifetime
    //  of the frame to the ThreadLocalState object.  This arena is cleared at the end
    //  of the frame.
    if (prev) {
        prev->mNext = newSegment;
    } else { // at start of list
        mHardSurfaceSegments[layer][idx] = newSegment;
    }
    newSegment->mTFront = t;
    newSegment->mTBack = t;
    newSegment->mTFirst = t;
    newSegment->mRayZ = rayZ;
    for (size_t i = 0; i < mDeepIDChannels.size(); i++) {
        getID(newSegment, i) = ids[i];
    }
    newSegment->mMask = 1ull << sidx;
    newSegment->mNormal = normal;
    newSegment->mAlpha = alpha * scale;
    for (int i = 0; i < mNumChannels; i++) {
        // clear all channel data
        getChannel(newSegment, i) = 0;
    }
    for (int i = 0; i < numChannels; i++) {
        // set channel data for specified channels (note numChannels vs. mNumChannels!)
        getChannel(newSegment, channels[i]) += values[i] * scale;
    }
    newSegment->mTotalWeight = weight;
    newSegment->mNext = nullptr;
}

// Does internal locking, suitable for vector mode execution.
void
DeepBuffer::addSample8x8Safe(unsigned x, unsigned y, unsigned subpixelX, unsigned subpixelY,
                             int layer,
                             const float *ids, float t, float rayZ,
                             const scene_rdl2::math::Vec3f& normal, float alpha,
                             const int *channels, int numChannels,
                             const float *values,
                             float scale, float weight)
{
    // Lock in case multiple threads want to add samples to this pixel
    tbb::mutex::scoped_lock lock(mPixelMutex[getMutexIdx(x, y)]);

    addSample8x8(x, y, subpixelX, subpixelY, layer, ids, t, rayZ, normal, alpha,
                 channels, numChannels, values, scale, weight);
}

void
DeepBuffer::finishPixel(unsigned threadIdx)
{
    VolumePixelBuffer &vpb = mVolumePixelBuffers[threadIdx];

    if (vpb.mCurrentX != 0xffffffff && vpb.mCurrentY != 0xffffffff) {
        mVolumeOutputSegments[mWidth * vpb.mCurrentY + vpb.mCurrentX] =
            vpb.mergeSegments(*this,
                              std::min(mSamplesPerPixel, VolumePixelBuffer::mMaxPixelSamples),
                              mVolCompressionRes,
                              mHardSurfaceSegments[0][mWidth * vpb.mCurrentY + vpb.mCurrentX]);
        vpb.clear();
    }
}

void
DeepBuffer::addVolumeSegments(pbr::TLState *pbrTls,
                              unsigned x, unsigned y, float sx, float sy,
                              float rayZ,
                              const pbr::VolumeProperties *vp,
                              size_t numVps)
{
    VolumePixelBuffer &vpb = mVolumePixelBuffers[pbrTls->mThreadIdx];
    vpb.mCurrentX = x;
    vpb.mCurrentY = y;

    unsigned subpixelX = (uint8_t)(sx * 8.f);
    unsigned subpixelY = (uint8_t)(sy * 8.f);
    int subpixelIdx = subpixelY * 8 + subpixelX;
    uint64_t maskBit = 1ull << subpixelIdx;

    if (vpb.mNumPixelSamples == VolumePixelBuffer::mMaxPixelSamples) {
        // just throw away the extras as they add little anyway assuming mMaxVolumePixelSamples
        //  is high
        return;
    }

    vpb.mAccumRayZ += rayZ;  // taking the avg of this when converting T to Z
    unsigned sample = vpb.mNumPixelSamples++;
    vpb.mMaskBits[sample] = maskBit;

    // copy in the segment data
    VolumeInputSegmentList& isl = vpb.mInputSegmentLists[sample];
    isl.mNumSegments = numVps;
    for (int i = 0; i < numVps; i++) {
        const pbr::VolumeProperties& props = vp[i];
        VolumeInputSegment *segment = (VolumeInputSegment*)pbrTls->mPixelArena->alloc(sizeof(VolumeInputSegment));
        segment->mNext = nullptr;
        segment->mTFront = props.mTStart;
        segment->mTBack = props.mTStart + props.mDelta;
        segment->mSigmaT = props.mSigmaT;
        if (isl.mFirst == nullptr) {
            isl.mFirst = segment;
            isl.mLast = segment;
        } else {
            isl.mLast->mNext = segment;
            isl.mLast = segment;
        }
    }
}

void
DeepBuffer::addVolumeSample(pbr::TLState *pbrTls,
                            unsigned x, unsigned y,
                            float t,
                            const scene_rdl2::math::Color& transmittance,
                            const scene_rdl2::math::Color& rgb, const float *aovs)
{
    VolumePixelBuffer &vpb = mVolumePixelBuffers[pbrTls->mThreadIdx];
    vpb.mCurrentX = x;
    vpb.mCurrentY = y;

    VolumeSample *sample = (VolumeSample*)pbrTls->mPixelArena->alloc(getVolumeSampleSize());
    sample->mNext = vpb.mSampleList;  // put at the start of the list
    vpb.mSampleList = sample;
    sample->mT = t;
    sample->mTransmittance = transmittance;
    sample->mChannels[0] = rgb[0];
    sample->mChannels[1] = rgb[1];
    sample->mChannels[2] = rgb[2];
    if (aovs) {
        for (int i = 3; i < mNumChannels; i++) {
            sample->mChannels[i] = aovs[i - 3];
        }
    } else {
        // TODO: we should really handle the emission aovs
        for (int i = 3; i < mNumChannels; i++) {
            sample->mChannels[i] = 0.f;
        }
    }
}

// Metadata handling code: Parse text data array to binary.  Same as the
//  code in ExrUtils.cc.
template <typename T, std::size_t N>
void makeArray(std::array<T, N>& array, const std::string& values)
{
    std::istringstream ins(values);
    ins.exceptions(std::ios_base::badbit | std::ios::failbit);

    try {
        std::copy_n(std::istream_iterator<T>(ins), N, array.begin());
    } catch (...) {
        throw scene_rdl2::except::ValueError("error converting to array of values.\n"
            "\tMake sure input string has correct number of elements."
            " Input string format should be \"elem0 elem1 ...\"\n");
    }
}

void
DeepBuffer::fillHeaderMetadata(const scene_rdl2::rdl2::Metadata *metadata, Imf::Header& header) const
{
    // get list of attributes
    const std::vector<std::string>& attrNames = metadata->getAttributeNames();
    const std::vector<std::string>& attrTypes = metadata->getAttributeTypes();
    const std::vector<std::string>& attrValues = metadata->getAttributeValues();

    // convert each attribute to appropriate data type
    for (size_t i = 0; i < attrNames.size(); ++i) {

        try {
            // scalar data types
            if (attrTypes[i] == "float") {
                header.insert(attrNames[i], Imf::FloatAttribute(std::stof(attrValues[i])));

            } else if (attrTypes[i] == "int") {
                header.insert(attrNames[i], Imf::IntAttribute(std::stoi(attrValues[i])));

            } else if (attrTypes[i] == "string") {
                header.insert(attrNames[i], Imf::StringAttribute(attrValues[i]));

            } else if (attrTypes[i] == "double") {
                header.insert(attrNames[i], Imf::DoubleAttribute(std::stod(attrValues[i])));

            } else if (attrTypes[i] == "chromaticities") {
                // array of 8 floats
                std::array<float, 8> elements;
                makeArray(elements, attrValues[i]);
                Imf::Chromaticities chrom(Imath::V2f(elements[0], elements[1]),
                                          Imath::V2f(elements[2], elements[3]),
                                          Imath::V2f(elements[4], elements[5]),
                                          Imath::V2f(elements[6], elements[7]));
                header.insert(attrNames[i], Imf::ChromaticitiesAttribute(chrom));
            }

            // vector data types
            else if (attrTypes[i] == "v2i") {
                std::array<int, 2> elements;
                makeArray(elements, attrValues[i]);
                Imath::V2i v(elements[0], elements[1]);
                header.insert(attrNames[i], Imf::V2iAttribute(v));

            } else if (attrTypes[i] == "v2f") {
                std::array<float, 2> elements;
                makeArray(elements, attrValues[i]);
                Imath::V2f v(elements[0], elements[1]);
                header.insert(attrNames[i], Imf::V2fAttribute(v));

            } else if (attrTypes[i] == "v3i") {
                std::array<int, 3> elements;
                makeArray(elements, attrValues[i]);
                Imath::V3i v(elements[0], elements[1], elements[2]);
                header.insert(attrNames[i], Imf::V3iAttribute(v));

            } else if (attrTypes[i] == "v3f") {
                std::array<float, 3> elements;
                makeArray(elements, attrValues[i]);
                Imath::V3f v(elements[0], elements[1], elements[2]);
                header.insert(attrNames[i], Imf::V3fAttribute(v));
            }

            // matrix data types
            else if (attrTypes[i] == "m33f") {
                std::array<float, 9> elements;
                makeArray(elements, attrValues[i]);
                Imath::M33f m(elements[0], elements[1], elements[2],
                              elements[3], elements[4], elements[5],
                              elements[6], elements[7], elements[8]);
                header.insert(attrNames[i], Imf::M33fAttribute(m));

            } else if (attrTypes[i] == "m44f") {
                std::array<float, 16> elements;
                makeArray(elements, attrValues[i]);
                Imath::M44f m(elements[0], elements[1], elements[2], elements[3],
                              elements[4], elements[5], elements[6], elements[7],
                              elements[8], elements[9], elements[10], elements[11],
                              elements[12], elements[13], elements[14], elements[15]);
                header.insert(attrNames[i], Imf::M44fAttribute(m));
            }

            // box data types
            else if (attrTypes[i] == "box2i") {
                // array of 2 vec2s
                std::array<int, 4> elements;
                makeArray(elements, attrValues[i]);

                // make sure min < max
                if (elements[0] > elements[2]) {
                    std::swap(elements[0], elements[2]);
                }
                if (elements[1] > elements[3]) {
                    std::swap(elements[1], elements[3]);
                }

                Imath::Box2i box(Imath::V2i(elements[0], elements[1]),
                                 Imath::V2i(elements[2], elements[3]));
                header.insert(attrNames[i], Imf::Box2iAttribute(box));

            } else if (attrTypes[i] == "box2f") {
                // array of 2 vec2s
                std::array<float, 4> elements;
                makeArray(elements, attrValues[i]);

                // make sure min < max
                if (elements[0] > elements[2]) {
                    std::swap(elements[0], elements[2]);
                }
                if (elements[1] > elements[3]) {
                    std::swap(elements[1], elements[3]);
                }

                Imath::Box2f box(Imath::V2f(elements[0], elements[1]),
                                 Imath::V2f(elements[2], elements[3]));
                header.insert(attrNames[i], Imf::Box2fAttribute(box));
            }

            // unknown data types
            else {
                throw scene_rdl2::except::TypeError("datatype " + attrTypes[i] + " is not supported");
            }

        } catch (const std::exception& e) {
            scene_rdl2::logging::Logger::error("Metadata(\"" + metadata->getName() + "\") : \"" + attrNames[i] +
                    "\": " + e.what());
        }
    }
}

void DeepBuffer::VolumePixelBuffer::clear()
{
    mCurrentX = 0xffffffff;
    mCurrentY = 0xffffffff;
    mNumPixelSamples = 0;
    mAccumRayZ = 0.f;

    // Clear all the VolumeInputSegments.  These are stored in an arena and
    // are automatically freed when the current pixel is finished rendering.
    for (int ps = 0; ps < mMaxPixelSamples; ps++) {
        mInputSegmentLists[ps].mFirst = nullptr;
        mInputSegmentLists[ps].mLast = nullptr;
        mInputSegmentLists[ps].mNumSegments = 0;
    }

    // Don't need to free the VolumeSamples because they live in an arena
    // and are automatically freed when the current pixel is finished rendering.
    mSampleList = nullptr;
}

DeepBuffer::VolumeOutputSegment*
DeepBuffer::VolumePixelBuffer::mergeSegments(const DeepBuffer& deepBuffer,
                                             unsigned samplesPerPixel,
                                             unsigned volCompressionRes,
                                             const HardSurfaceSegment *hardSegments)
{
    // We need to figure out how many segments the merged output segment list has.
    // Use the average segment length across all the segments in the set of
    // input segment lists and the min/max t extents of all those segments.
    unsigned totalNumSegments = 0;
    float totalSegmentLength = 0.f;
    float minT = FLT_MAX;
    float maxT = -FLT_MAX;
    for (unsigned ps = 0; ps < mNumPixelSamples; ps++) {
        totalNumSegments += mInputSegmentLists[ps].mNumSegments;
        float sampleMinT = mInputSegmentLists[ps].mFirst->mTFront;
        float sampleMaxT = mInputSegmentLists[ps].mLast->mTBack;
        totalSegmentLength += (sampleMaxT - sampleMinT);
        minT = std::min(minT, sampleMinT);
        maxT = std::max(maxT, sampleMaxT);
    }
    if (totalNumSegments == 0) {
        return nullptr;
    }
    float avgLength = totalSegmentLength / totalNumSegments;
    unsigned newNumSegments = (unsigned)ceil((maxT - minT) / avgLength);
    float deltaT = ((maxT - minT) / newNumSegments);

    // Create the merged output segment list.
    VolumeOutputSegment* firstOutputSegment = nullptr;
    VolumeOutputSegment* prevOutputSegment = nullptr;
    for (unsigned s = 0; s < newNumSegments; s++) {
        VolumeOutputSegment* outputSegment =
            (VolumeOutputSegment*)malloc(deepBuffer.getVolumeOutputSegmentSize());
        if (!firstOutputSegment) {
            firstOutputSegment = outputSegment;
        }
        if (prevOutputSegment) {
            prevOutputSegment->mNext = outputSegment;
        }
        outputSegment->mNext = nullptr;
        // T values for output segments are just evenly spaced
        outputSegment->mTFront = minT + deltaT * s;
        outputSegment->mTBack = minT + deltaT * (s + 1);
        outputSegment->mTtoZ = -(mAccumRayZ / mNumPixelSamples);
        // The output sigmaT is merged below.
        outputSegment->mSigmaT = scene_rdl2::math::Color(0, 0, 0);
        // Radiance and AOVs are merged below.
        for (int chan = 0; chan < deepBuffer.mNumChannels; chan++) {
            outputSegment->mChannels[chan] = 0.f;
        }
        outputSegment->mNumUnblockedSamples = 0;
        prevOutputSegment = outputSegment;
    }

    // We have numPixelSamples input segment lists that look like this:
    //    +--------------------+------------+-------------+
    //  +-------------+----------------+---+----------------+
    //         +-----------+-------------------+---------------+
    // etc.
    // Each segment has a sigmaT (extinction coeff) value.  We need to remap
    // all of the sigmaTs to the one collapsed output segment list, which has
    // newNumSegments segments of equal length:
    // +----------+----------+----------+----------+----------+
    //
    // The first step is to align the segments in the input segment lists to
    // the output segment.  E.g. the top diagram becomes this:
    // +----------+----------+----------+----------+----------+
    // +----------+----------+----------+----------+----------+
    // +----------+----------+----------+----------+----------+
    //
    // We still have numPixelSamples input segment lists but the segments themselves
    // will line up with the output segments.  We need to adjust the input unaligned
    // sigmaT values based on their overlap with the aligned input segments.
    // There are several cases that need to be handled.

    // Store the aligned input segment sigmaTs:
    scene_rdl2::math::Color *alignedSigmaTs = new scene_rdl2::math::Color[mNumPixelSamples * newNumSegments];

    // For each aligned (dest) segment we iterate over the pixel samples and find the unaligned
    // (src) segments that overlap.  We compute the relative amount of overlap between the
    // aligned (dest) and unaligned (src) and sum in the sigmaT * relative overlap.
    // Relative overlap is in the range [0,1].

    VolumeOutputSegment* destSegment = firstOutputSegment;
    unsigned destIdx = 0;
    while (destSegment) {
        float destTFront = destSegment->mTFront;
        float destTBack = destSegment->mTBack;

        for (unsigned ps = 0; ps < mNumPixelSamples; ps++) {

            scene_rdl2::math::Color& destSigmaT = alignedSigmaTs[ps * newNumSegments + destIdx];
            destSigmaT = scene_rdl2::math::Color(0.f);

            VolumeInputSegment* srcSeg = mInputSegmentLists[ps].mFirst;
            while (srcSeg) {

                if (destTBack < srcSeg->mTFront) {
                    //                       tfront          tback
                    // src:                     +--------------+
                    // dest:        +-------+
                    //           tfront   tback
                    //
                    // no overlap, dest segment is completely in front of src
                    // so we can exit the loop as the dest segment will be in front
                    // of the rest of the src segments, since the segments are in t order
                    break;
                }

                float overlap = scene_rdl2::math::clamp(srcSeg->mTBack, destTFront, destTBack) -
                                scene_rdl2::math::clamp(srcSeg->mTFront, destTFront, destTBack);

                // sum in a fraction of the src segment's sigmaT based on the amount
                //  of overlap with the dest segment
                float relativeOverlap = overlap / (destTBack - destTFront);
                destSigmaT += srcSeg->mSigmaT * relativeOverlap;

                srcSeg = srcSeg->mNext;
            }
        }
        destSegment = destSegment->mNext;
        destIdx++;
    }

    // Now that we have numSamples segment lists, and those segments are aligned to the
    // output segments, we can combine the sigmaT values.  We are ultimately
    // computing transmittance with Beer's law with the sigmaT values:
    //
    // transmittance = e^(-sigmaT * dt)
    //
    // Say we computed the transmittance for two samples independently, and then
    // averaged the results.  This is analogous to what Moonray does to compute its
    // volume alpha for flat images.  i.e.
    //
    // transmittance = (e^(-sigmaT0 * dt) + e^(-sigmaT1 * dt)) / 2
    //
    // We want to combine sigmaT0 and sigmaT1 into a combined sigmaT2 that still
    // gives us the same transmittance.  Thus:
    //
    // e^(-sigmaT2 * dt) = (e^(-sigmaT0 * dt) + e^(-sigmaT1 * dt)) / 2
    //
    // Solve for sigmaT2:
    //
    // -sigmaT2 * dt = log((e^(-sigmaT0 * dt) + e^(-sigmaT1 * dt)) / 2)
    //
    // sigmaT2 = -log((e^(-sigmaT0 * dt) + e^(-sigmaT1 * dt)) / 2) / dt
    //
    // This gives us the solution for an isolated segment, but the total transmittance
    // is actually the individual segments' transmittance multiplied together.
    // i.e. for 2 segments A and B in the same sample:
    //
    //      +----------------------------+------------------------------+
    //              segment A                      segment B
    //
    // total = transA * transB
    //
    // Say we have combined sigmaT values sigmaTA and sigmaTB:
    //
    //      +----------------------------+------------------------------+
    //          segment A (sigmaTA)          segment B (sigmaTB)
    //
    // total = exp(-sigmaTA * dt) * exp(-sigmaTB * dt)
    //
    // ... and with the original, uncombined sigmaT values ...
    //
    //      +----------------------------+------------------------------+
    //          segment A (sigmaT0A)         segment B (sigmaT0B)
    //      +----------------------------+------------------------------+
    //          segment A (sigmaT1A)         segment B (sigmaT1B)
    //
    // total = (exp(-sigmaT0A * dt) * exp(-sigmaT0B * dt) +
    //          exp(-sigmaT1A * dt) * exp(-sigmaT1B * dt)) / 2
    //
    // Combine equations and merge exponents together:
    //
    // exp(-(sigmaTA + sigmaTB) * dt) = (exp(-(sigmaT0A + sigmaT0B) * dt) +
    //                                   exp(-(sigmaT1A + sigmaT1B) * dt)) / 2
    //
    // sigmaTA + sigmaTB = -log((exp(-(sigmaT0A + sigmaT0B) * dt) +
    //                           exp(-(sigmaT1A + sigmaT1B) * dt)) / 2) / dt
    //
    // We need to solve for sigmaTA and sigmaTB.  This is not yet possible because
    // we have two unknowns but only one equation.  We need another constraint.
    // The key is that we traverse the volume from front to back, and we don't
    // need to know all of the sT values as we go.  i.e. the first segment only depends
    // on sigmaTA, and not sigmaTB, so we can solve that segment first.
    //
    // total0 = transmittanceA(sigmaTA)
    // total1 = transmittanceA(sigmaTA) * transmittanceB(sigmaTB)
    //
    // We can solve the first equation for sigmaTA using the previous sigmaT2 derivation:
    //
    // sigmaTA = -log((exp(-sigmaT0A * dt) + exp(-sigmaT1A * dt)) / 2) / dt
    //
    // Now that we know sigmaTA, we can solve for sigmaTB:
    //
    // sigmaTB = -log((exp(-(sigmaT0A + sigmaT0B) * dt) +
    //                 exp(-(sigmaT1A + sigmaT1B) * dt)) / 2) / dt
    //           - sigmaTA
    //
    // We can extend this to a third segment:
    //
    // sigmaTC = -log((exp(-(sigmaT0A + sigmaT0B + sigmaT0C) * dt) +
    //                 exp(-(sigmaT1A + sigmaT1B + sigmaT1C) * dt)) / 2) / dt
    //           - (sigmaTA + sigmaTB)
    //
    // There is a clear pattern now and we can extend this to an arbitrary number of
    // segments by keeping running sums of sT values.  In the code below we use
    // srcSigmaTSums[] (for sigmaT0A + sigmaT0B, sigmaT1A + sigmaT1B etc) and destSigmaSum
    // (for sigmaTA + sigmaTB etc).
    //
    // This can also be extended to handle an arbitrary number of samples by simply adding
    // more exp() terms inside the log() and dividing by numPixelSamples instead of by 2.
    {
        scene_rdl2::math::Color srcSigmaTSums[samplesPerPixel];  // per sample
        for (unsigned i = 0; i < samplesPerPixel; i++) {
            srcSigmaTSums[i] = scene_rdl2::math::Color(0.f);
        }
        scene_rdl2::math::Color destSigmaSum(0.f);
        unsigned destIdx = 0;
        VolumeOutputSegment *outputSegment = firstOutputSegment;
        while (outputSegment) {
            scene_rdl2::math::Color expSum(0.f);

            // The number of pixel samples that intersected the volume but weren't
            //  blocked by a hard surface.
            unsigned numUnblockedSamples = 0;

            for (unsigned ps = 0; ps < samplesPerPixel; ps++) {
                // We know we cast samplesPerPixel rays into the volume, but not all
                //  may have intersected the volume.  Thus, mNumPixelSamples may be
                //  less than samplesPerPixel.  The remaining samples missed the volume.
                if (ps < mNumPixelSamples) {
                    // Intersected the volume.
                    // Check if this pixel sample is blocked by a hard surface.
                    // Each pixel sample has its single 8x8 coverage mask bit set in mMaskBits.
                    bool isBlocked = deepBuffer.isBlockedByHardSurface(
                            hardSegments,
                            mMaskBits[ps],
                            outputSegment->mTFront);

                    // If it's blocked, we need to ignore it completely or it will
                    // incorrectly weight the merged sigmaT value.  This is because
                    // it is missing information as moonray terminates ray traversal
                    // at the first hard surface inside the volume, so we don't know
                    // the properties of the volume behind.  Else...
                    if (!isBlocked) {
                        srcSigmaTSums[ps] += alignedSigmaTs[ps * newNumSegments + destIdx];
                        expSum += exp(-srcSigmaTSums[ps] * deltaT);
                        numUnblockedSamples++;
                    }

                } else {
                    // Missed the volume.  Assume zero sigmaT (vacuum).
                    // expSum += exp(scene_rdl2::math::Color(0.f) * deltaT);
                    expSum += scene_rdl2::math::Color(1.f);
                    numUnblockedSamples++;
                }
            }
            // scene_rdl2::math::Color sigmaT = -log(expSum / std::min(mSamplesPerPixel, VolumePixelBuffer::mMaxPixelSamples)) /
            //                     deltaT - destSigmaSum;
            // Corrected for blocked samples:

            /*
            scene_rdl2::math::Color sigmaT = -log(expSum / numUnblockedSamples) / deltaT - destSigmaSum;
            outputSegment->mSigmaT = sigmaT;
            outputSegment->mNumUnblockedSamples = numUnblockedSamples;
            destSigmaSum += sigmaT;
            */

            // The above code has potential precision issues with the running destSigmaSum.
            // It may become much larger than sigmaT and precision may be lost.
            // Note that if we expand the destSigmaSum += sigmaT expression:
            //
            // destSigmaSum = destSigmaSum + -log(expSum / numUnblockedSamples) / deltaT - destSigmaSum
            //
            // Note cancellation of destSigmaSum term.  Simplify:
            //
            // destSigmaSum = -log(expSum / numUnblockedSamples) / deltaT
            //
            // This avoids error accumulation.  The code can be rewritten as:
            scene_rdl2::math::Color logTerm = -log(expSum / numUnblockedSamples) / deltaT;
            scene_rdl2::math::Color sigmaT = logTerm - destSigmaSum;
            outputSegment->mSigmaT = sigmaT;
            outputSegment->mNumUnblockedSamples = numUnblockedSamples;
            destSigmaSum = logTerm;

            outputSegment = outputSegment->mNext;
            destIdx++;
        }
    }

    // We're done with this as the sigmaTs are in the output segments now.
    delete[] alignedSigmaTs;

    // Find the largest sigmaT value for the output segments.
    float largestSigmaT = 0;
    VolumeOutputSegment *outputSegment = firstOutputSegment;
    while (outputSegment) {
        // Note that the luminance is used here as it simplifies things.
        float sigmaT = luminance(outputSegment->mSigmaT);
        if (sigmaT > largestSigmaT) {
            largestSigmaT = sigmaT;
        }
        outputSegment = outputSegment->mNext;
    }

    // The maximum allowable difference between segments' sigmaT values for
    //  segments that can be merged together.
    float maxDelta = largestSigmaT / volCompressionRes;

    // Optimize the segment list by merging segments with similar sigmaT values (< maxDelta).
    outputSegment = firstOutputSegment;

    // Delete any segments at the front of the volume with zero sigmaTs.  We don't
    // want to be merging in segments with non-zero sigmaTs into these because
    // these segments will no longer be empty space and will cause compositing artifacts.
    while (outputSegment) {
        if (luminance(outputSegment->mSigmaT) > 0.f) {
            break;
        }
        VolumeOutputSegment *nextSegment = outputSegment->mNext;
        free(outputSegment);
        outputSegment = nextSegment;
        firstOutputSegment = outputSegment;
    }

    unsigned currCount = 0;     // The number of segments in this run of merged segments
    float currLuminance;    // The sigmaT luminance for this current run of merged segments
    while (outputSegment) {
        // For each outputSegment we look at the next segment and see if it can be
        // merged with the current outputSegment.
        if (currCount == 0) {
            // We have started a new run of segments.
            currCount = 1;
            currLuminance = luminance(outputSegment->mSigmaT);
        }
        VolumeOutputSegment *nextSegment = outputSegment->mNext;
        if (nextSegment) {
            // Careful: we must check that the sigmaT values are similar AND that
            // the two segments being merged have the same amount of hard surface
            // occlusion.  If the number of blocked samples differs, that means that
            // there is a hard surface between the two segments and thus we shouldn't
            // combine the segments.  Technically we could but it produces better results
            // if we don't.
            if (fabs(luminance(nextSegment->mSigmaT) - currLuminance) <= maxDelta &&
                     nextSegment->mNumUnblockedSamples == outputSegment->mNumUnblockedSamples) {
                // The next segment is within the maxDelta tolerance, so merge it
                //  into the current output segment.  We sum in the sigmaT and adjust
                //  the tBack value, adjust the linked list next pointer, and delete
                //  the merged (next) segment.
                outputSegment->mSigmaT += nextSegment->mSigmaT;
                outputSegment->mTBack = nextSegment->mTBack;
                outputSegment->mNext = nextSegment->mNext;
                free(nextSegment);
                currCount++;
                continue;  // Stay on the current outputSegment by bypassing the
                           //  outputSegment = outputSegment->mNext; at the end of the
                           //  loop body.
            } else {
                // The next segment is outside the maxDelta tolerance, so finish
                //  up the current run of segments.  We just average the sigmaTs by dividing
                //  the sum by the count of segments in this run of merged segments.
                outputSegment->mSigmaT /= currCount;
                currCount = 0;
            }
        } else if (currCount > 1) {
            // No next segment (we're at the end of the list), but we might
            //  need to finish up the current segment
            outputSegment->mSigmaT /= currCount;
        }

        outputSegment = outputSegment->mNext;
    }

    // We now have an optimized list of volume output segments, each with a sigmaT
    // value and tFront/tBack distances.  The final step is to merge in all the
    // radiance+aov sample data (VolumeSamples) into these segments.
    // Each VolumeSample was computed at a ray distance mT and has an overall
    // mTransmittance value, which is the total transmittance from the camera to
    // the location of the sample.  There's a problem... the mTransmittance is the
    // total transmittance for the original volume, not our optimized output segments
    // which are slightly different than the original volume.  If we just use the
    // original radiance+aov values here, the composited result will be slightly too
    // bright/too dark due to the transmittance mismatch.  So, we scale the radiance+aov
    // values by the relative difference between the original transmittance and the
    // transmittance of the merged output segments.  This corrects for the difference
    // and prevents the final composited radiance from being too bright or dark.
    //
    // Another detail is that by merging in the VolumeSamples into the output segments,
    // we are *moving* the sample t location slightly to the back of the output segment,
    // i.e. its tBack location.  This also requires a correction to the radiance
    // as the transmittance is reduced slightly from moving the sample further further
    // from the camera.
    //
    // Conveniently, the radiance correction calculation below handles both problems
    // at once.

    // Keep track of the total transmittance as we traverse the output samples
    scene_rdl2::math::Color totalTransmittance(1.f);
    outputSegment = firstOutputSegment;
    while (outputSegment) {

        // For each outputSegment, find all the VolumeSamples that overlap it.
        // If the VolumeSample's t value is greater than the tBack of the last
        // output segment, it belongs to the last output segment.
        VolumeSample* sample = mSampleList;
        VolumeSample* prevSample = nullptr;
        while (sample) {
            if ((sample->mT > outputSegment->mTFront && sample->mT <= outputSegment->mTBack) ||
                (!outputSegment->mNext && sample->mT > outputSegment->mTBack)) {

                // Derivation:
                // The final pixel value in the flat render after applying mTransmittance is:
                // pixel = luminance(sample->mTransmittance) * sample->mChannels
                //
                // The final pixel value in the deep render is given by:
                // pixel = luminance(totalTransmittance) *
                //         luminance(exp(-outputSegment->mSigmaT * deltaT)) *
                //         outputSegment->mChannels
                //
                // Note that luminance(totalTransmittance) is the transmittance up to
                // the front of the current output segment, then we also need to
                // attenuate by the thickness of the current output segment with the exp() term
                // as the sample is located at the back of the output segment.
                //
                // We want flat pixel = deep pixel so we can solve for outputSegment->mChannels:
                //
                // outputSegment->mChannels = luminance(sample->mTransmittance) * sample->mChannels /
                //                            (luminance(totalTransmittance) *
                //                             luminance(exp(-outputSegment->mSigmaT * deltaT)))
                //
                // outputSegment->mChannels = sample->mChannels * ratio
                //
                // ratio = luminance(sample->mTransmittance) /
                //          (luminance(totalTransmittance) *
                //           luminance(exp(-outputSegment->mSigmaT * deltaT)))
                //
                // But, our ratio term in the code doesn't have the exp() term...
                // OpenDCX requires the channels to be premultiplied by the transmittance
                // of the segment, which means that:
                //
                // outputSegment->mChannels = sample->mChannels * ratio * luminance(exp(-outputSegment->mSigmaT * deltaT))
                //
                // The exp() term cancels out and we can avoid computing it.

                // Correction ratio of original transmittance and new transmittance
                float ratio = luminance(sample->mTransmittance) / luminance(totalTransmittance);

                for (int chan = 0; chan < deepBuffer.mNumChannels; chan++) {
                    // Apply correction and sum this into the current output segment.
                    outputSegment->mChannels[chan] += sample->mChannels[chan] * ratio;
                }

                // Remove the sample from the list as it can only overlap one output
                // segment.
                if (prevSample) {
                    prevSample->mNext = sample->mNext;
                } else {
                    // we're at the start of the list
                    mSampleList = sample->mNext;
                }
                // don't change prevSample here to the current sample because the
                // current sample has been deleted
                sample = sample->mNext;
                continue;
            }

            // else sample doesn't overlap segment, move to the next sample
            prevSample = sample;
            sample = sample->mNext;
        }

        // We need to divide by the number of unblocked pixel samples to get the
        //  correct scaling for the channel data.
        for (int chan = 0; chan < deepBuffer.mNumChannels; chan++) {
            if (outputSegment->mNumUnblockedSamples > 0) {
                outputSegment->mChannels[chan] /= outputSegment->mNumUnblockedSamples;
            } else {
                // If no samples just set to zero.
                outputSegment->mChannels[chan] = 0.f;
            }
        }

        // Update the total transmittance based on Beer's Law and then move to
        // the next output segment.
        float deltaT = outputSegment->mTBack - outputSegment->mTFront;
        totalTransmittance *= exp(-outputSegment->mSigmaT * deltaT);
        outputSegment = outputSegment->mNext;
    }

    // Return the head of the output segment linked list.
    return firstOutputSegment;

    // Finished.  Wasn't that fun?
}

void
DeepBuffer::writeVolumeSegments(VolumeOutputSegment* outputSegments,
                                const OPENDCX_INTERNAL_NAMESPACE::ChannelSet& chanSet,
                                const std::vector<std::pair<unsigned, unsigned> >& channelsToOutput,
                                const std::vector<OPENDCX_INTERNAL_NAMESPACE::ChannelIdx>& deepIDChannels,
                                const std::vector<OPENDCX_INTERNAL_NAMESPACE::ChannelIdx>& channelIdxs,
                                OPENDCX_INTERNAL_NAMESPACE::DeepPixel& dcxpixel) const
{
    // Iterate over the volume output segments and write them to the deep file.

    VolumeOutputSegment *vs = outputSegments;
    while (vs) {
        OPENDCX_INTERNAL_NAMESPACE::DeepSegment ds;
        OPENDCX_INTERNAL_NAMESPACE::Pixelf dp(chanSet);

        // Convert t to z
        ds.Zf = vs->mTFront * vs->mTtoZ;
        ds.Zb = vs->mTBack * vs->mTtoZ;

        ds.index = -1;   // unassigned

        // Set the pixel mask to full coverage for volume pixels
        ds.metadata.spmask = 0xffffffffffffffffull;

        // Set default compositing mode, appropriate for volumes.
        ds.metadata.flags  = 0;

        // Apply Beer's Law to compute the transmittance between the front
        // and back of the segment.
        float deltaT = vs->mTBack - vs->mTFront;
        scene_rdl2::math::Color transmittance = exp(-vs->mSigmaT * deltaT);

        // Convert 3-channel transmittance to single-channel alpha
        scene_rdl2::math::Color alpha3 = scene_rdl2::math::Color(1, 1, 1) - transmittance;
        float alpha = luminance(alpha3);
        dp[OPENDCX_INTERNAL_NAMESPACE::Chan_A] = alpha;

        for (size_t i = 0; i < mDeepIDChannels.size(); i++) {
            // copy in the deep ID data
            dp[deepIDChannels[i]] = 0;  // No IDs yet...
        }

        // Copy the RGB beauty values (channels[0-2]) to the appropriate aov channels.
        int destChannel = 3;
        for (const auto &entry: (*mAovSchema)) {
            if (entry.type() == pbr::AOV_TYPE_BEAUTY) {
                vs->mChannels[destChannel] = vs->mChannels[0];
                vs->mChannels[destChannel + 1] = vs->mChannels[1];
                vs->mChannels[destChannel + 2] = vs->mChannels[2];
            } else if (entry.type() == pbr::AOV_TYPE_ALPHA) {
                vs->mChannels[destChannel] = alpha;
            }
            destChannel += entry.numChannels();
        }

        // Copy the channel data into the deep pixel
        unsigned i = 0;
        for (unsigned a = 0; a < channelsToOutput.size(); a++) {
            for (unsigned chan = channelsToOutput[a].first; chan <= channelsToOutput[a].second; chan++) {
                dp[channelIdxs[i]] = vs->mChannels[chan];
                i++;
            }
        }

        dcxpixel.append(ds, dp);

        vs = vs->mNext;
    }
}

void
DeepBuffer::writeHardSurfaceSegments(int idx,
                                     const OPENDCX_INTERNAL_NAMESPACE::ChannelSet& chanSet,
                                     const std::vector<std::pair<unsigned, unsigned> >& channelsToOutput,
                                     const std::vector<OPENDCX_INTERNAL_NAMESPACE::ChannelIdx>& deepIDChannels,
                                     const std::vector<OPENDCX_INTERNAL_NAMESPACE::ChannelIdx>& channelIdxs,
                                     OPENDCX_INTERNAL_NAMESPACE::DeepPixel& dcxpixel) const
{
   for (int layer = 0; layer < mMaxLayers; layer++) {
        // output each layer separately
        HardSurfaceSegment *hs = mHardSurfaceSegments[layer][idx];
        while (hs) {
            OPENDCX_INTERNAL_NAMESPACE::DeepSegment ds;
            OPENDCX_INTERNAL_NAMESPACE::Pixelf dp(chanSet);

            ds.Zf = -hs->mTFront * hs->mRayZ;
            ds.Zb = -hs->mTBack * hs->mRayZ;

            ds.index = -1;   // unassigned
            ds.metadata.spmask = hs->mMask;
            ds.metadata.flags  = OPENDCX_INTERNAL_NAMESPACE::DeepFlags::LINEAR_INTERP;  // set hard surface
            float alpha = hs->mAlpha / hs->mTotalWeight;
            dp[OPENDCX_INTERNAL_NAMESPACE::Chan_A] = alpha;
            for (size_t i = 0; i < mDeepIDChannels.size(); i++) {
                // copy in the deep ID data
                dp[deepIDChannels[i]] = getID(hs, i);
            }

            // Copy the RGB beauty values (channels[0-2]) to the appropriate aov channels.
            // Note that we start at 3.  Channels 0-2 in the deep buffer are storage for the
            // beauty channels, but these need to be output as AOVs which occupy channels 3+.
            // Note that we don't actually output channels 0-2.
            // This copying is similar to what happens in the flat render output code,
            // see pbr/core/Aov.cc::aovSetBeautyAndAlpha().
            int destChannel = 3;
            for (const auto &entry: (*mAovSchema)) {
                if (entry.type() == pbr::AOV_TYPE_BEAUTY) {
                    setChannel(hs, destChannel, getChannel(hs, 0));
                    setChannel(hs, destChannel + 1, getChannel(hs, 1));
                    setChannel(hs, destChannel + 2, getChannel(hs, 2));
                } else if (entry.type() == pbr::AOV_TYPE_ALPHA) {
                    setChannel(hs, destChannel, alpha);
                }
                destChannel += entry.numChannels();
            }

            unsigned i = 0;
            for (unsigned a = 0; a < channelsToOutput.size(); a++) {
                for (unsigned chan = channelsToOutput[a].first; chan <= channelsToOutput[a].second; chan++) {
                    // copy in the channel data, normalizing it by dividing by mTotalWeight
                    dp[channelIdxs[i]] = getChannel(hs, chan) / hs->mTotalWeight;
                    i++;
                }
            }

            dcxpixel.append(ds, dp);

            hs = hs->mNext;
        }
    }
}

bool
DeepBuffer::sortSegmentByZ(const DeepBuffer::HardSurfaceSegment *lhs,
                           const DeepBuffer::HardSurfaceSegment *rhs)
{
    return lhs->mZ < rhs->mZ;
}

void
DeepBuffer::writeHardSurfaceSegmentsNoMask(int idx,
                                           int filmX, int filmY,
                                           const scene_rdl2::fb_util::PixelBuffer<unsigned>& samplesCount,
                                           const OPENDCX_INTERNAL_NAMESPACE::ChannelSet& chanSet,
                                           const std::vector<std::pair<unsigned, unsigned> >& channelsToOutput,
                                           const std::vector<OPENDCX_INTERNAL_NAMESPACE::ChannelIdx>& deepIDChannels,
                                           const std::vector<OPENDCX_INTERNAL_NAMESPACE::ChannelIdx>& channelIdxs,
                                           OPENDCX_INTERNAL_NAMESPACE::DeepPixel& dcxpixel) const
{
    // Output each layer separately
    for (int layer = 0; layer < mMaxLayers; layer++) {

        HardSurfaceSegment *hs = mHardSurfaceSegments[layer][idx];

        // Copy the hard surfaces linked list into an array for easier sorting.
        // We need to output the surfaces in front to back order because we need
        // to compute the total accumulated alpha as we go.  This is different than
        // the subpixel mask case where we don't need to sort at all.
        // First we need to count them.
        unsigned numSegments = 0;
        while (hs) {
            numSegments++;
            hs = hs->mNext;
        }

        if (numSegments == 0) {
            // No hard surfaces left to write for this pixel.
            //  We can skip deeper layers because they will be empty.
            return;
        }

        // Copy into an array and compute the Z value from the ray direction and t
        HardSurfaceSegment *segments[numSegments];
        hs = const_cast<HardSurfaceSegment*>(mHardSurfaceSegments[layer][idx]);
        numSegments = 0;
        while (hs) {
            segments[numSegments++] = hs;
            hs->mZ = -hs->mTBack * hs->mRayZ;
            hs = hs->mNext;
        }

        // Sort segments in increasing Z order
        std::sort(segments, segments + numSegments, sortSegmentByZ);

        // Output the sorted segments.
        // We need to keep track of the total coverage as we output the surfaces in front
        //  to back order.  This is used to compute the alpha value.
        float totalCoverage = 0.f;
        for (unsigned seg = 0; seg < numSegments; seg++) {
            hs = segments[seg];

            OPENDCX_INTERNAL_NAMESPACE::DeepSegment ds;
            OPENDCX_INTERNAL_NAMESPACE::Pixelf dp(chanSet);

            // Set Zf = Zb for hard surfaces
            ds.Zf = ds.Zb = hs->mZ;

            ds.index = -1;   // unassigned

            // Note that these aren't actually output in the deep file because these
            // channels aren't part of the ChanSet when not outputting masks.  We still
            // set them to keep the OpenDCX API in a consistent state just in case.
            ds.metadata.spmask = 0xffffffffffffffffull;
            ds.metadata.flags  = OPENDCX_INTERNAL_NAMESPACE::DeepFlags::LINEAR_INTERP;  // set hard surface

            // Get the number of samples for this pixel.  This varies
            // when using adaptive sampling.
            int samplesPerPixel = samplesCount.getPixel(filmX, filmY);
            // Deeper layers have fewer samples
            int samplesDivision = 1 << (layer * 2);  // 1, 4, 16, 64 ...
            samplesPerPixel /= samplesDivision;
            if (samplesPerPixel == 0) {
                samplesPerPixel = 1;
            }

            // Compute the fractional coverage for the current surface.
            // Note that the mTotalWeight is really the number of samples that
            // hit this surface.
            float coverage = (float)hs->mTotalWeight / (float)samplesPerPixel;

            // Compute the alpha value that considers the current surface's coverage
            //  and the total coverage from other surfaces in front of it.
            // This is the magic formula that lets us replace coverage masks with
            // alpha and get identical results when compositing, most of the time.
            // I derived this formula empirically after experimenting with alpha blending
            // on a spreadsheet.  It seems to produce correct results in NUKE.
            float alpha = coverage / (1 - totalCoverage);
            totalCoverage += coverage;

            dp[OPENDCX_INTERNAL_NAMESPACE::Chan_A] = alpha;

            for (size_t i = 0; i < mDeepIDChannels.size(); i++) {
                // Copy in the deep ID data.
                dp[deepIDChannels[i]] = getID(hs, i);
            }

            // Copy the RGB beauty values (channels[0-2]) to the appropriate aov channels.
            int destChannel = 3;
            for (const auto &entry: (*mAovSchema)) {
                if (entry.type() == pbr::AOV_TYPE_BEAUTY) {
                    setChannel(hs, destChannel, getChannel(hs, 0));
                    setChannel(hs, destChannel + 1, getChannel(hs, 1));
                    setChannel(hs, destChannel + 2, getChannel(hs, 2));
                } else if (entry.type() == pbr::AOV_TYPE_ALPHA) {
                    setChannel(hs, destChannel, alpha);
                }
                destChannel += entry.numChannels();
            }

            unsigned i = 0;
            for (unsigned a = 0; a < channelsToOutput.size(); a++) {
                for (unsigned chan = channelsToOutput[a].first; chan <= channelsToOutput[a].second; chan++) {
                    // Copy in the channel data, normalizing it by dividing by mTotalWeight
                    // (which is actually the number of samples, so we're just taking the average
                    // value) and premultiply the alpha.
                    dp[channelIdxs[i]] = getChannel(hs, chan) / hs->mTotalWeight * alpha;
                    i++;
                }
            }

            // Write the deep surface.
            dcxpixel.append(ds, dp);

            if (totalCoverage >= 1.f) {
                // If we have reached full total coverage, don't process any more
                // segments because they are not visible.
                break;
            }
        }
    }
}

void
DeepBuffer::write(const std::string& filename,
                  const std::vector<int>& aovs,
                  const std::vector<std::string>& aovChannelNames,
                  const scene_rdl2::fb_util::PixelBuffer<unsigned>& samplesCount,
                  const scene_rdl2::math::HalfOpenViewport& aperture, const scene_rdl2::math::HalfOpenViewport& region,
                  const scene_rdl2::rdl2::Metadata *metadata) const
{
    // Create an OpenDCX deep file and copy all of the HardSurfaceSegments to it

    OPENDCX_INTERNAL_NAMESPACE::ChannelSet chanSet;
    chanSet.insert(OPENDCX_INTERNAL_NAMESPACE::Chan_ZFront);
    chanSet.insert(OPENDCX_INTERNAL_NAMESPACE::Chan_ZBack);
    if (mFormat == DeepFormat::OpenDCX2_0) {
        // We only need the channels containing the mask bits and flags if we
        // are using the subpixel masks.
        chanSet.insert(OPENDCX_INTERNAL_NAMESPACE::Chan_SpBits1);   // first 32 bits of 64-bit 8x8 mask
        chanSet.insert(OPENDCX_INTERNAL_NAMESPACE::Chan_SpBits2);   // second 32 bits of 64-bit 8x8 mask
        chanSet.insert(OPENDCX_INTERNAL_NAMESPACE::Chan_DeepFlags); // flags controlling mask behavior when compositing
    }
    chanSet.insert(OPENDCX_INTERNAL_NAMESPACE::Chan_A);

    OPENDCX_INTERNAL_NAMESPACE::ChannelContext chanCtx;

    // Add in all of the deep ID channels
    std::vector<OPENDCX_INTERNAL_NAMESPACE::ChannelIdx> deepIDChannels;
    for (size_t i = 0; i < mDeepIDChannels.size(); i++) {
        chanCtx.addChannelAlias (mDeepIDChannels[i],  // channel name
                                 "deep",              // layer name
                                 OPENDCX_INTERNAL_NAMESPACE::Chan_Invalid,   // channel index, assign next available
                                 0,                   // position
                                 "",                  // io_name,
                                 Imf::FLOAT,          // io_type
                                 0);                  // part
        deepIDChannels.push_back(chanCtx.getChannel(std::string("deep.") + mDeepIDChannels[i]));
        chanSet.insert(deepIDChannels[i]);
    }

    // We have multiple ranges of deep channels to output depending on whether the
    // beauty and which AOVs are selected for output.  Note that these are the
    // channels in the deep data, not the AOV channel ids.
    std::vector<std::pair<unsigned, unsigned> > channelsToOutput;
    std::vector<OPENDCX_INTERNAL_NAMESPACE::ChannelIdx> channelIdxs;

    {
        int channelNameIdx = 0;
        for (size_t a = 0; a < aovs.size(); a++) {
            // If we're outputting AOVs, figure out first and last deep channels to output for each aov.
            unsigned firstChannel = 3;  // 0-2 are the beauty channels
            for (size_t i = 0; i < aovs[a]; i++) {
                firstChannel += mChannelsPerAOV[i];
            }
            unsigned lastChannel = firstChannel + mChannelsPerAOV[aovs[a]] - 1;
            channelsToOutput.push_back(std::pair<unsigned, unsigned>(firstChannel, lastChannel));

            for (int cn = 0; cn < mChannelsPerAOV[aovs[a]]; cn++) {
                // Need to split the aov channel names e.g. aovName.red
                size_t dotPos = aovChannelNames[channelNameIdx].find(".");
                std::string channel = aovChannelNames[channelNameIdx].substr(dotPos + 1);
                std::string layer = aovChannelNames[channelNameIdx].substr(0, dotPos);
                chanCtx.addChannelAlias(channel, layer,
                                        OPENDCX_INTERNAL_NAMESPACE::Chan_Invalid, 0, "", Imf::HALF, 0);
                channelIdxs.push_back(chanCtx.getChannel(aovChannelNames[channelNameIdx]));
                channelNameIdx++;
            }
        }
    }

    for (size_t i = 0; i < channelIdxs.size(); i++) {
        chanSet.insert(channelIdxs[i]);
    }

    IMATH_NAMESPACE::Box2i displayWindow(IMATH_NAMESPACE::V2i(aperture.min().x, aperture.min().y),
                                         IMATH_NAMESPACE::V2i(aperture.max().x - 1, aperture.max().y - 1));

    IMATH_NAMESPACE::Box2i dataWindow(IMATH_NAMESPACE::V2i(region.min().x, region.min().y),
                                      IMATH_NAMESPACE::V2i(region.max().x - 1, region.max().y - 1));

    OPENDCX_INTERNAL_NAMESPACE::DeepImageOutputTile *outputTile = new OPENDCX_INTERNAL_NAMESPACE::DeepImageOutputTile(displayWindow,
                                               dataWindow,
                                               true,
                                               chanSet,
                                               chanCtx);
    Imf::Header header;
    if (metadata) {
        fillHeaderMetadata(metadata, header);
    }
    outputTile->setOutputFile(filename.c_str(), header);

    if (mFormat == DeepFormat::OpenEXR2_0) {
        std::cout << "Writing hard deep surfaces without masks." << std::endl;
    }

    // Iterate over the pixels/HardSurfaceSegments and copy their data to the OpenDCX
    //  file.
    // Note that the film coordinates (filmX, filmY) are not necessarily the same as the
    //  image coordinates (outX, outY) in that the image coordinates sometimes have an
    //  offset applied.
    int idx = 0;
    for (int outY = outputTile->minY(), filmY = 0; outY <= outputTile->maxY(); outY++, filmY++) {
        for (int outX = outputTile->minX(), filmX = 0; outX <= outputTile->maxX(); outX++, filmX++, idx++) {

            // if mHardSurfaceSegments[0] doesn't have segments then we can assume
            // there are no deeper layers
            if (!mHardSurfaceSegments[0][idx] && !mVolumeOutputSegments[idx]) {
                outputTile->clearDeepPixel(outX, outY);
                continue;
            }

            OPENDCX_INTERNAL_NAMESPACE::DeepPixel dcxpixel(chanSet);

            writeVolumeSegments(mVolumeOutputSegments[idx],
                                chanSet,
                                channelsToOutput,
                                deepIDChannels,
                                channelIdxs,
                                dcxpixel);

            if (mFormat == DeepFormat::OpenDCX2_0) {
                writeHardSurfaceSegments(idx,
                                         chanSet,
                                         channelsToOutput,
                                         deepIDChannels,
                                         channelIdxs,
                                         dcxpixel);
            } else {
                writeHardSurfaceSegmentsNoMask(idx,
                                               filmX, filmY,
                                               samplesCount,
                                               chanSet,
                                               channelsToOutput,
                                               deepIDChannels,
                                               channelIdxs,
                                               dcxpixel);
            }

            outputTile->setDeepPixel(outX, outY, dcxpixel);
        }
        outputTile->writeScanline(outY, true);
    }

    delete outputTile;
}

size_t
DeepBuffer::getMemoryUsage() const
{
    // Currently not called by anything, used for debugging purposes only.

    size_t total = mWidth * mHeight * sizeof(HardSurfaceSegment*);

    // count up all the HardSurfaceSegments
    for (size_t i = 0; i < mWidth * mHeight; i++) {
        for (size_t layer = 0; layer < mMaxLayers; layer++) {
            HardSurfaceSegment *current = mHardSurfaceSegments[layer][i];
            while (current) {
                current = current->mNext;
                total += getHardSurfaceSegmentSize();
            }
        }
    }

    // count up all the VolumeOutputSegments
    for (size_t i = 0; i < mWidth * mHeight; i++) {
        VolumeOutputSegment *current = mVolumeOutputSegments[i];
        while (current) {
            current = current->mNext;
            total += getVolumeOutputSegmentSize();
        }
    }

    return total;
}

}
}

