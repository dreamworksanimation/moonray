// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <moonray/rendering/pbr/core/Aov.h>

#include <scene_rdl2/common/fb_util/PixelBuffer.h>
#include <scene_rdl2/common/math/Color.h>
#include <scene_rdl2/common/math/Vec3.h>

#include <moonray/deepfile/DcxChannelContext.h>
#include <moonray/deepfile/DcxChannelSet.h>
#include <moonray/deepfile/DcxDeepImageTile.h>
#include <OpenEXR/ImfHeader.h>
#include <tbb/mutex.h>

namespace moonray {

namespace pbr {
    class VolumeProperties;
}

namespace pbr {

// Code for building a deep image and outputting it to an OpenDCX file.

enum class DeepFormat
{
    OpenEXR2_0 = 0,
    OpenDCX2_0 = 1
};

class DeepBuffer
{
public:
    DeepBuffer();
    ~DeepBuffer();

    void initDeep(unsigned width, unsigned height,
                  DeepFormat format,
                  float curvatureTolerance,
                  float zTolerance,
                  unsigned mVolCompressionRes,
                  unsigned numRenderThreads,
                  const pbr::AovSchema &aovSchema,
                  const std::vector<std::string>& deepIDChannels,
                  int maxLayers);

    void clear();

    unsigned getWidth() const { return mWidth; }
    unsigned getHeight() const { return mHeight; }

    // 1, 2, 4, or 8 -> 1x1, 2x2, 4x4, or 8x8
    // The actual resolution of the deep pixels is still 8x8, but samples are
    //  duplicated to mimic a lower resolution.  This is useful if we are rendering
    //  with fewer than 64 samples per pixel as we don't have enough samples to
    //  fully populate an 8x8 subpixel grid.
    void setSubpixelRes(unsigned subpixelRes) { mSubpixelRes = subpixelRes; }

    void setSamplesPerPixel(unsigned samplesPerPixel) { mSamplesPerPixel = samplesPerPixel; }

    // Adds sample data to a deep pixel.  Can be called repeatedly to accumulate
    // channel values.  Used in the Film's bundled handlers.  Thread safe.
    void addSample(pbr::TLState *pbrTls,
                   unsigned x,                // pixel x coordinate
                   unsigned y,                // pixel y coordinate
                   float sx,                  // x subpixel offset [0, 1)
                   float sy,                  // y subpixel offset [0, 1)
                   int layer,                 // depth layer
                   const float *deepIDs,      // one or more deep IDs for the current surface
                   float t,                   // t value of the current surface
                   float rayZ,                // ray.z value, used for converting t->z
                   const scene_rdl2::math::Vec3f& normal, // surface normal
                   float alpha,               // alpha
                   const int *channels,       // list of channel indices to write to
                   int numChannels,           // number of channel indices
                   const float *values,       // channel data, one value per channel
                   float scale,               // scaling value to apply to channel data
                   float weight);             // sample weight, i.e. pathPixelWeight

    // Adds sample color data to a deep pixel.  Convenience method that takes
    // separate rgb + aov data.  Used for scalar renderFrame().
    void addSample(pbr::TLState *pbrTls,
                   unsigned x, unsigned y, float sx, float sy,
                   int layer,
                   const float *deepIDs, float t, float rayZ,
                   const scene_rdl2::math::Vec3f& normal, float alpha,
                   const scene_rdl2::math::Color& rgb, const float *aovs,
                   float scale, float weight);

    void addVolumeSegments(pbr::TLState *pbrTls,
                           unsigned x, unsigned y, float sx, float sy,
                           float rayZ,
                           const pbr::VolumeProperties *vp,
                           size_t numVps);

    void addVolumeSample(pbr::TLState *pbrTls,
                         unsigned x, unsigned y,
                         float t,
                         const scene_rdl2::math::Color& transmittance,
                         const scene_rdl2::math::Color& rgb, const float *aovs);

    void finishPixel(unsigned threadIdx);

    // Write the deep buffer to a deep file using OpenDCX
    void write(const std::string& filename,
               const std::vector<int>& aovs,       // aov channels
               const std::vector<std::string>& aovChannelNames,
               const scene_rdl2::fb_util::PixelBuffer<unsigned>& samplesCount,
               const scene_rdl2::math::HalfOpenViewport& aperture, const scene_rdl2::math::HalfOpenViewport& region,
               const scene_rdl2::rdl2::Metadata *metadata) const;

    size_t getMemoryUsage() const;

    DeepFormat getFormat() const { return mFormat; }

private:
    // The other addSample methods potentially duplicate the sample data to
    //  simulate a 1x1, 2x2, or 4x4 subpixel resolution.  This is the method that
    //  actually adds the (un)duplicated deep samples to the buffer.

    // Lock free version, suitable for use in scalar mode.
    void addSample8x8(unsigned x, unsigned y, unsigned subpixelX, unsigned subpixelY,
                      int layer,
                      const float *ids, float t, float rayZ,
                      const scene_rdl2::math::Vec3f& normal, float alpha,
                      const int *channels, int numChannels,
                      const float *values,
                      float scale, float weight);

    // Thread safe version suitable for use in vector mode.
    void addSample8x8Safe(unsigned x, unsigned y, unsigned subpixelX, unsigned subpixelY,
                          int layer,
                          const float *ids, float t, float rayZ,
                          const scene_rdl2::math::Vec3f& normal, float alpha,
                          const int *channels, int numChannels,
                          const float *values,
                          float scale, float weight);

    /* Each pixel has a linked list of HardSurfaceSegments.  Pixels may be empty and
     * have no HardSurfaceSegments assigned, in which case mHardSurfaceSegments will
     * have a nullptr for that pixel.
     */
    struct HardSurfaceSegment
    {
        HardSurfaceSegment *mNext;  // next segment in the linked list
        float mTFirst;              // First T distance value
        float mTFront;              // T front distance value
        float mTBack;               // T back distance value
        float mRayZ;                // The ray's Z component, used for converting t->z
        float mZ;                   // Computed z value used for sorting surfaces
        uint64_t mMask;             // 64-bit 8x8 subpixel mask
        scene_rdl2::math::Vec3f mNormal;        // surface normal
        float mAlpha;               // alpha
        float mTotalWeight;         // total weight, used for combining multiple samples
        float mIDsAndChannels[0];   // deep IDs and channels
        /* The deep IDs and channels data is variable-length (although constant
         * for a given scene), and depends on the number of IDs and beauty+AOV
         * channels being rendered.  Accessors are provided below to correctly
         * index into this data.
         */
    };

    // One vector<HardSurfaceSegment *> per depth layer, up to mMaxLayers
    std::vector< std::vector<HardSurfaceSegment *> > mHardSurfaceSegments;

    static bool sortSegmentByZ(const HardSurfaceSegment *lhs, const HardSurfaceSegment *rhs);

    size_t getHardSurfaceSegmentSize() const {
        return sizeof(HardSurfaceSegment) + (mDeepIDChannels.size() + mNumChannels) * sizeof(float);
    }

    inline float& getID(HardSurfaceSegment *segment, unsigned i) {
        return segment->mIDsAndChannels[i];
    }
    inline float getID(const HardSurfaceSegment *segment, unsigned i) const {
        return segment->mIDsAndChannels[i];
    }

    inline float& getChannel(HardSurfaceSegment *segment, unsigned i) {
        return segment->mIDsAndChannels[mDeepIDChannels.size() + i];
    }
    inline float getChannel(const HardSurfaceSegment *segment, unsigned i) const {
        return segment->mIDsAndChannels[mDeepIDChannels.size() + i];
    }
    inline void setChannel(HardSurfaceSegment *segment, unsigned i, float value) const {
        segment->mIDsAndChannels[mDeepIDChannels.size() + i] = value;
    }

    // Traverse a HardSurfaceSegment linked list and check if the specified
    // subpixel (mask bit) is blocked by the hard surface at the distance value
    // t.
    bool isBlockedByHardSurface(const HardSurfaceSegment *firstSegment,
                                uint64_t maskBit,
                                float t) const
    {
        uint64_t mask = 0;
        const HardSurfaceSegment *current = firstSegment;
        while (current) {
            if (current->mTBack < t) {
                mask |= current->mMask;
                if (mask & maskBit) {
                    return true;
                }
            }
            current = current->mNext;
        }
        return false;
    }

    void writeHardSurfaceSegments(int idx,
                                  const OPENDCX_INTERNAL_NAMESPACE::ChannelSet& chanSet,
                                  const std::vector<std::pair<unsigned, unsigned> >& channelsToOutput,
                                  const std::vector<OPENDCX_INTERNAL_NAMESPACE::ChannelIdx>& deepIDChannels,
                                  const std::vector<OPENDCX_INTERNAL_NAMESPACE::ChannelIdx>& channelIdxs,
                                  OPENDCX_INTERNAL_NAMESPACE::DeepPixel& dcxpixel) const;

    void writeHardSurfaceSegmentsNoMask(int idx,
                                        int filmX, int filmY,
                                        const scene_rdl2::fb_util::PixelBuffer<unsigned>& samplesCount,
                                        const OPENDCX_INTERNAL_NAMESPACE::ChannelSet& chanSet,
                                        const std::vector<std::pair<unsigned, unsigned> >& channelsToOutput,
                                        const std::vector<OPENDCX_INTERNAL_NAMESPACE::ChannelIdx>& deepIDChannels,
                                        const std::vector<OPENDCX_INTERNAL_NAMESPACE::ChannelIdx>& channelIdxs,
                                        OPENDCX_INTERNAL_NAMESPACE::DeepPixel& dcxpixel) const;

    /*
     * Deep Volumes Overview:
     *
     * The PathIntegratorVolume sends both volume segments (VolumeInputSegment)
     * and radiance/aov samples along the ray (VolumeSample) to the DeepBuffer.
     * These are stored in a VolumePixelBuffer structure until all of a pixel's segments
     * and samples are collected.  The segments and samples are then merged to
     * produce a much smaller set of VolumeOutputSegments which are written to the
     * deep file.
     *
     * For each pixel, there is a list of VolumeInputSegments for each pixel sample.
     * There is only one list of VolumeSamples for the pixel, for all pixel samples.
     * One list of VolumeOutputSegments is output for each pixel from merging the many
     * lists of VolumeInputSegments and the single list of VolumeSamples.
     * The VolumeOutputSegments are a simplified representation
     * of the actual volume.  They are processed to consume minimal space in
     * the deep file but still produce a compositing result that closely matches
     * the flat render output from the original volume.  There will inevitably
     * be minor differences due to this simplification.
     *
     * The amount of segment merging / deep file size is adjustable so a tradeoff can
     * be made between deep file size and maintaining the detail of the original
     * volume.  Great care is taken with the math to minimize the visible effects of segment
     * merging.
     *
     * Volume and hard surface data may be mixed within a deep file and these
     * should interact/composite properly.  An example would be a flaming sword
     * which is a combination of hard surface (the blade) and volume (the fire)
     * elements.
     *
     * A simplifying assumption is made that the coverage mask for a volume segment
     * in the deep file is either full coverage or no coverage.  Volumes tend to
     * have "soft" edges so the lack of subpixel-accurate coverage information is not
     * visually noticeable.  The code has special treatment of volume segments that
     * are partially overlapped by hard surfaces.
     */

    // Segments received from the ray-marching code in the VolumeIntegrator.
    // No channel data is associated with these.  mSigmaT is extinction coefficient
    // of the volume for the segment, which is used to calculate transmittance with
    // Beer's law.
    struct VolumeInputSegment
    {
        VolumeInputSegment *mNext;
        float mTFront;
        float mTBack;
        scene_rdl2::math::Color mSigmaT;
    };

    struct VolumeInputSegmentList
    {
        VolumeInputSegment *mFirst;
        VolumeInputSegment *mLast;
        unsigned mNumSegments;
    };

    // One sample of the channel data (radiance + aovs) at a location mT along
    // the ray inside the volume.  mTransmittance gives the transmittance from the
    // camera to the location mT.  The sample may have an arbitrary number of RGB
    // and aov channels, hence the zero-length array which is a placeholder for
    // the channel data.
    struct VolumeSample
    {
        VolumeSample *mNext;
        float mT;
        scene_rdl2::math::Color mTransmittance;
        float mChannels[0];
    };

    // The size of the VolumeSample depends on the number of channels.
    // Note that we can only dynamically allocate this data structure due to the
    // unknown size at compile time.  This is due to the number of aov channels
    // which varies per scene context.
    size_t getVolumeSampleSize() const {
        return sizeof(VolumeSample) + mNumChannels * sizeof(float);
    }

    // VolumeInputSegments and VolumeSamples are merged together to produce
    // VolumeOutputSegments which are output to the deep file.
    struct VolumeOutputSegment
    {
        VolumeOutputSegment *mNext;
        float mTFront;
        float mTBack;

        // mTtoZ is a scaling value that converts ray T values to deep Z values.
        // We work in ray T space right up until the deep file export.  This
        // keeps things consistent between hard surfaces and volumes.
        float mTtoZ;

        // The combined sigmaT for the segment, created by merging the sigmaTs of
        // multiple VolumeInputSegments.
        scene_rdl2::math::Color mSigmaT;

        // VolumeOutputSegments can be partially occluded by hard surfaces.
        // The amount of occlusion is given by mNumBlockedSamples / mNumPixelSamples.
        // We need this to compensate for the missing (blocked / occluded) samples.
        unsigned mNumUnblockedSamples;

        // Again, variable number of channels (beauty + aov) that varies per
        // scene context.
        float mChannels[0];
    };
    // This works the same as the HardSurfaceSegments.  There is one linked list of
    // VolumeOutputSegments per pixel.
    mutable std::vector<VolumeOutputSegment *> mVolumeOutputSegments;

    // Similarly to VolumeSample, the size depends on the number of channels
    // and is only known at run time.
    size_t getVolumeOutputSegmentSize() const {
        return sizeof(VolumeOutputSegment) + mNumChannels * sizeof(float);
    }

    // One of these per thread.  Keeps track of the data pertaining to the current
    // pixel being rendered.  Note that this only works with batch render mode and NOT
    // progressive render mode because batch render mode fully renders a pixel before
    // moving on to the next one but progressive render mode only partially renders
    // the pixel and makes multiple passes.
    // Supporting progressive render mode would require a VolumePixelBuffer for
    // EACH pixel in the image which would require a prohibitive amount of memory.
    // With batch render mode we only need one of these per thread as each thread
    // works on one pixel at a time.  The buffer can then be cleared and re-used
    // for the next pixel.
    struct VolumePixelBuffer
    {
        // x, y coords of the current pixel being worked on by the thread.
        unsigned mCurrentX;
        unsigned mCurrentY;

        // Total number of pixel samples for this pixel...
        unsigned mNumPixelSamples;

        // Max number of pixel samples... we just discard any samples over this limit.
        static const unsigned mMaxPixelSamples = 1024;

        // For each pixel sample we accumulate the ray.z value and then divide it
        // by the number of pixel samples to produce an average ray Z value for
        // the pixel.  This is needed to properly handle camera motion blur where the
        // ray.z value varies significantly between the pixel samples.
        float mAccumRayZ;

        // One linked list of VolumeInputSegments per pixel sample.
        // We keep track of the first and last segments in the list, and the
        // number of segments.
        VolumeInputSegmentList mInputSegmentLists[mMaxPixelSamples];

        // Which subpixel mask bit does the subpixel overlap?
        uint64_t mMaskBits[mMaxPixelSamples];

        // Linked list of VolumeSamples for the pixel.  Not associated with any
        // particular pixel sample as these all get merged into the VolumeOutputSegments.
        // We don't care about the order of these and don't need to keep track of first
        //  + last pointers like the VolumeInputSegments.
        VolumeSample *mSampleList;

        void clear();

        // Merges the VolumeInputSegments and the VolumeSamples together into a
        // list of VolumeOutputSegments.  Merging/compression is applied based on the
        // volCompressionRes.  The hard surfaces need to be considered to handle
        // partial occlusion of VolumeOutputSegments by hard surfaces properly.
        VolumeOutputSegment* mergeSegments(const DeepBuffer& deepBuffer,
                                           unsigned samplesPerPixel,
                                           unsigned volCompressionRes,
                                           const HardSurfaceSegment *hardSegments);
    };
    std::vector<VolumePixelBuffer> mVolumePixelBuffers;  // one per thread

    // Finally, write the VolumeOutputSegments to the DeepPixel.  Note that the
    // ID channels are not currently used.
    void writeVolumeSegments(VolumeOutputSegment* outputSegments,
                             const OPENDCX_INTERNAL_NAMESPACE::ChannelSet& chanSet,
                             const std::vector<std::pair<unsigned, unsigned> >& channelsToOutput,
                             const std::vector<OPENDCX_INTERNAL_NAMESPACE::ChannelIdx>& deepIDChannels,
                             const std::vector<OPENDCX_INTERNAL_NAMESPACE::ChannelIdx>& channelIdxs,
                             OPENDCX_INTERNAL_NAMESPACE::DeepPixel& dcxpixel) const;

/* Deep pixels are independent of each other, so there is no threading hazard
 * when different threads are writing to different pixels.  It is possible
 * (although uncommon) that multiple threads might write to the same pixel,
 * so that needs to be protected with a mutex.  The simple solution is to
 * have one mutex per pixel.  This would consume a lot of memory, so instead
 * there is an array of 225 mutexes for the entire image that are shared
 * between the pixels.  Some pixels will share the same mutex, which results
 * in some extra locking... however testing has shown the overhead is negligible
 * once there are >= 64 mutexes for ~20 rendering threads.  This may need to be
 * increased as the number of threads increases though.
 * The mutexes form a 15x15 tile that repeats across the image, and the
 * getMutexIdx() logic maps a screen coordinate to a [0, 225] tile index.
 * Note that the tile size is not a multiple of the rendering 8x8 tile size.
 * This reduces mutex sharing during rendering.
 */
    static const int mMutexTileSize = 15;
    // force mutex to be cache-line aligned for speed
    struct CACHE_ALIGN AlignedMutex : public tbb::mutex {};
    AlignedMutex *mPixelMutex;
    int getMutexIdx(unsigned x, unsigned y) const {
        return (y % mMutexTileSize) * mMutexTileSize + (x % mMutexTileSize);
    }

    // Copies the rdl2 metadata into the exr header.
    void fillHeaderMetadata(const scene_rdl2::rdl2::Metadata *metadata, Imf::Header& header) const;

    // deep image dimensions
    unsigned mWidth;
    unsigned mHeight;

    const pbr::AovSchema *mAovSchema;

    // the names of the deep ID channels
    std::vector<std::string> mDeepIDChannels;

    // the number of channels per AOV, typically 1-3
    std::vector<unsigned int> mChannelsPerAOV;

    // total number of beauty + aov channels
    unsigned mNumChannels;

    // curvature tolerance (in degrees) that is used to split highly curved deep segments
    float mCurvatureTolerance;

    // hard surface samples will be merged together provided their z values differ
    // by less than this tolerance value
    float mZTolerance;

    // Number of levels / bins used when combining VolumeInputSegments with similar
    // mSigmaT values.  Must be > 0 and can be arbitrarily high.  A value of 1 usually combines
    // all segments into one segment with an average sigmaT value.
    unsigned mVolCompressionRes;

    // 1, 2, 4, or 8 -> 1x1, 2x2, 4x4, or 8x8
    unsigned mSubpixelRes;

    unsigned mSamplesPerPixel;

    unsigned mNumRenderThreads;

    DeepFormat mFormat;

    int mMaxLayers;
};

}
}

