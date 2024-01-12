// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

//
//
#pragma once
#include <scene_rdl2/common/math/MathUtil.h>
#include <scene_rdl2/render/util/SortUtil.h>

namespace moonray {
namespace mcrt_common {

//-----------------------------------------------------------------------------

// PREFETCH_DISTANCE is the number of bytes in advance to prefetch (not yet implemented).
// numElems is the number of AOS elements we want to transpose.
// TODO: @@@ add actual prefetching logic.
template<unsigned SRC_AOS_SIZE,
         unsigned SRC_AOS_STRIDE,
         unsigned DST_SOA_STRIDE,
         unsigned PREFETCH_DISTANCE>
inline void convertAOSToSOA_AVX512(unsigned numElems,
                                   const uint32_t *__restrict src,
                                   uint32_t *__restrict dst)
{
    MNRY_STATIC_ASSERT(SRC_AOS_SIZE <= SRC_AOS_STRIDE);

    MNRY_STATIC_ASSERT(SRC_AOS_SIZE % (AVX512_VLEN * 4) == 0);
    MNRY_STATIC_ASSERT(SRC_AOS_SIZE % (AVX512_VLEN * 4) == 0);
    MNRY_STATIC_ASSERT(SRC_AOS_STRIDE % (AVX512_VLEN * 4) == 0);
    MNRY_STATIC_ASSERT(DST_SOA_STRIDE % (AVX512_VLEN * 4) == 0);

    MNRY_ASSERT(isAligned(src, AVX512_SIMD_MEMORY_ALIGNMENT));
    MNRY_ASSERT(isAligned(dst, AVX512_SIMD_MEMORY_ALIGNMENT));

    const unsigned numFullBlocks = numElems / AVX512_VLEN;
    const unsigned numChunksPerBlock = SRC_AOS_SIZE / AVX512_SIMD_MEMORY_ALIGNMENT;

    struct ALIGN(AVX512_SIMD_MEMORY_ALIGNMENT) AOSData
    {
        uint8_t mData[SRC_AOS_STRIDE];
    };

    const AOSData *aosData = (const AOSData *)src;

    //
    // Transpose all full blocks.
    //
    for (unsigned iblock = 0; iblock < numFullBlocks; ++iblock) {

        const CACHE_ALIGN uint32_t *__restrict srcRows[AVX512_VLEN] =
        {
            (const uint32_t *)(&aosData[0]),
            (const uint32_t *)(&aosData[1]),
            (const uint32_t *)(&aosData[2]),
            (const uint32_t *)(&aosData[3]),
            (const uint32_t *)(&aosData[4]),
            (const uint32_t *)(&aosData[5]),
            (const uint32_t *)(&aosData[6]),
            (const uint32_t *)(&aosData[7]),
            (const uint32_t *)(&aosData[8]),
            (const uint32_t *)(&aosData[9]),
            (const uint32_t *)(&aosData[10]),
            (const uint32_t *)(&aosData[11]),
            (const uint32_t *)(&aosData[12]),
            (const uint32_t *)(&aosData[13]),
            (const uint32_t *)(&aosData[14]),
            (const uint32_t *)(&aosData[15]),
        };

        aosData += AVX_VLEN;

        for (unsigned i = 0; i < numChunksPerBlock; ++i) {

            uint32_t *__restrict dstChunk = dst + (AVX512_VLEN * AVX512_VLEN) * i;

            scene_rdl2::math::transposeAOSToSOA_16x16(srcRows, dstChunk);

            for(unsigned j=0; j<AVX512_VLEN; j++)
                srcRows[j] += AVX512_VLEN;
        }

        // Take destination stride into account.
        dst += DST_SOA_STRIDE >> 2;
    }

    //
    // Transpose remaining entries into a partially filled block.
    //
    unsigned remainingEntries = numElems - (numFullBlocks * AVX512_VLEN);

    if (remainingEntries) {

        MNRY_ASSERT(remainingEntries < AVX512_VLEN);

        const CACHE_ALIGN uint32_t *__restrict srcRows[AVX512_VLEN];

        // Fill in valid entries.
        for (unsigned i = 0; i < remainingEntries; ++i) {
            srcRows[i] = (const uint32_t *)(&aosData[i]);
        }

        // Duplicate final entries to fill out trailing lanes.
        const uint32_t *finalRow = (const uint32_t *)(&aosData[remainingEntries - 1]);
        for (unsigned i = remainingEntries; i < AVX512_VLEN; ++i) {
            srcRows[i] = finalRow;
        }

        // Process remaining chunks in this block whilst prefetching from the next block.
        for (unsigned i = 0; i < numChunksPerBlock; ++i) {

            scene_rdl2::math::transposeAOSToSOA_16x16(srcRows, dst);

            for(unsigned j=0; j<AVX512_VLEN; j++)
                srcRows[j] += AVX512_VLEN;

            dst += (AVX512_VLEN * AVX512_VLEN);
        }
    }
}

// SORT_INDEX_STRIDE is the stride between each sort_indices index in bytes.
template<unsigned SRC_AOS_SIZE,
         unsigned SRC_AOS_STRIDE,
         unsigned DST_SOA_STRIDE,
         unsigned SORT_INDEX_STRIDE,
         unsigned PREFETCH_DISTANCE>
inline void convertAOSToSOAIndexed_AVX512(unsigned numElems,
                                          const uint32_t *__restrict src,
                                          uint32_t *__restrict dst,
                                          const uint32_t *sortIndices)
{
    MNRY_STATIC_ASSERT(SRC_AOS_SIZE <= SRC_AOS_STRIDE);
    MNRY_STATIC_ASSERT(SRC_AOS_SIZE % (AVX512_VLEN * 4) == 0);
    MNRY_STATIC_ASSERT(SRC_AOS_STRIDE % (AVX512_VLEN * 4) == 0);
    MNRY_STATIC_ASSERT(DST_SOA_STRIDE % (AVX512_VLEN * AVX512_SIMD_REGISTER_SIZE) == 0);
    MNRY_STATIC_ASSERT(SORT_INDEX_STRIDE % sizeof(uint32_t) == 0);

    MNRY_ASSERT(scene_rdl2::util::isAligned(src, AVX512_SIMD_MEMORY_ALIGNMENT));
    MNRY_ASSERT(scene_rdl2::util::isAligned(dst, AVX512_SIMD_MEMORY_ALIGNMENT));

    const unsigned numFullBlocks = numElems / AVX512_VLEN;
    const unsigned numChunksPerBlock = SRC_AOS_SIZE / AVX512_SIMD_MEMORY_ALIGNMENT;

    struct ALIGN(AVX512_SIMD_MEMORY_ALIGNMENT) AOSData
    {
        uint8_t mData[SRC_AOS_STRIDE];
    };

    const AOSData *aosData = (const AOSData *)src;

    //
    // Transpose all full blocks.
    //
    for (unsigned iblock = 0; iblock < numFullBlocks; ++iblock) {

        const CACHE_ALIGN uint32_t *__restrict srcRows[AVX512_VLEN] =
        {
            (const uint32_t *)(&aosData[sortIndices[0]]),
            (const uint32_t *)(&aosData[sortIndices[SORT_INDEX_STRIDE >> 2]]),
            (const uint32_t *)(&aosData[sortIndices[SORT_INDEX_STRIDE >> 1]]),
            (const uint32_t *)(&aosData[sortIndices[3 * (SORT_INDEX_STRIDE >> 2)]]),
            (const uint32_t *)(&aosData[sortIndices[SORT_INDEX_STRIDE]]),
            (const uint32_t *)(&aosData[sortIndices[5 * (SORT_INDEX_STRIDE >> 2)]]),
            (const uint32_t *)(&aosData[sortIndices[6 * (SORT_INDEX_STRIDE >> 2)]]),
            (const uint32_t *)(&aosData[sortIndices[7 * (SORT_INDEX_STRIDE >> 2)]]),
            (const uint32_t *)(&aosData[sortIndices[SORT_INDEX_STRIDE << 1]]),
            (const uint32_t *)(&aosData[sortIndices[9  * (SORT_INDEX_STRIDE >> 2)]]),
            (const uint32_t *)(&aosData[sortIndices[10 * (SORT_INDEX_STRIDE >> 2)]]),
            (const uint32_t *)(&aosData[sortIndices[11 * (SORT_INDEX_STRIDE >> 2)]]),
            (const uint32_t *)(&aosData[sortIndices[12 * (SORT_INDEX_STRIDE >> 2)]]),
            (const uint32_t *)(&aosData[sortIndices[13 * (SORT_INDEX_STRIDE >> 2)]]),
            (const uint32_t *)(&aosData[sortIndices[14 * (SORT_INDEX_STRIDE >> 2)]]),
            (const uint32_t *)(&aosData[sortIndices[15 * (SORT_INDEX_STRIDE >> 2)]])
        };
        sortIndices += AVX512_VLEN * (SORT_INDEX_STRIDE >> 2);

        for (unsigned i = 0; i < numChunksPerBlock; ++i) {

            uint32_t *__restrict dstChunk = dst + (AVX512_VLEN * AVX512_VLEN) * i;

            scene_rdl2::math::transposeAOSToSOA_16x16(srcRows, dstChunk);

            for(unsigned j=0; j<AVX512_VLEN; j++)
                srcRows[j] += AVX512_VLEN;
        }

        // Take destination stride into account.
        dst += DST_SOA_STRIDE >> 2;
    }

    //
    // Transpose remaining entries into a partially filled block.
    //
    unsigned remainingEntries = numElems - (numFullBlocks * AVX512_VLEN);

    if (remainingEntries) {

        MNRY_ASSERT(remainingEntries < AVX512_VLEN);

        const CACHE_ALIGN uint32_t *__restrict srcRows[AVX512_VLEN];

        // Fill in valid entries.
        for (unsigned i = 0; i < remainingEntries; ++i) {
            srcRows[i] = (const uint32_t *)(&aosData[sortIndices[i * (SORT_INDEX_STRIDE >> 2)]]);
        }

        // Duplicate final entries to fill out trailing lanes.
        const uint32_t *finalRow = (const uint32_t *)(&aosData[sortIndices[(remainingEntries - 1) * (SORT_INDEX_STRIDE >> 2)]]);
        for (unsigned i = remainingEntries; i < AVX512_VLEN; ++i) {
            srcRows[i] = finalRow;
        }

        // Process remaining chunks in this block whilst prefetching from the next block.
        for (unsigned i = 0; i < numChunksPerBlock; ++i) {

            scene_rdl2::math::transposeAOSToSOA_16x16(srcRows, dst);

            for(unsigned j=0; j<AVX512_VLEN; j++)
                srcRows[j] += AVX512_VLEN;

            dst += (AVX512_VLEN * AVX512_VLEN);
        }
    }
}

template<unsigned SRC_SOA_SIZE,         // DST_AOS_SIZE is derived from this.
         unsigned SRC_SOA_STRIDE,
         unsigned INDEX_STRIDE,
         unsigned PREFETCH_DISTANCE>
inline void
convertSOAToAOSIndexed_AVX512(unsigned numElems,
                              const uint32_t *indices,
                              const uint32_t *__restrict src,
                              uint32_t **__restrict dst)
{
    const unsigned DST_AOS_SIZE = SRC_SOA_SIZE / AVX512_VLEN;

    MNRY_STATIC_ASSERT(SRC_SOA_SIZE <= SRC_SOA_STRIDE);
    MNRY_STATIC_ASSERT(SRC_SOA_SIZE % (AVX512_SIMD_REGISTER_SIZE * AVX512_VLEN) == 0);
    MNRY_STATIC_ASSERT(SRC_SOA_STRIDE % (AVX512_SIMD_REGISTER_SIZE * AVX512_VLEN) == 0);
    MNRY_STATIC_ASSERT(DST_AOS_SIZE % AVX512_SIMD_REGISTER_SIZE == 0);
    MNRY_STATIC_ASSERT(INDEX_STRIDE % sizeof(uint32_t) == 0);

    MNRY_ASSERT(scene_rdl2::util::isAligned(src, AVX512_SIMD_MEMORY_ALIGNMENT));
    MNRY_ASSERT(scene_rdl2::util::isAligned(dst, AVX512_SIMD_MEMORY_ALIGNMENT));
    MNRY_ASSERT(numElems);

    CACHE_ALIGN uint32_t dummy[DST_AOS_SIZE >> 2];

    const unsigned numChunksPerBlock = SRC_SOA_SIZE / (AVX512_VLEN * AVX512_VLEN * 4);

    unsigned currIdx = 0;

    do {
        // Find indices within current SOA block.
        unsigned baseIdx = indices[currIdx];
        unsigned endIdx = (baseIdx + AVX512_VLEN) & ~AVX512_VLEN_MASK;

        CACHE_ALIGN uint32_t *__restrict dstRows[AVX512_VLEN] = { dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy };

        do {
            dstRows[indices[currIdx] & AVX512_VLEN_MASK] = dst[currIdx];
            ++currIdx;
            --numElems;
        } while (numElems && indices[currIdx] < endIdx);

        const uint32_t *soaBlock = src + ((SRC_SOA_STRIDE * (baseIdx >> AVX512_VLEN_SHIFT)) / 4);
        MNRY_ASSERT(scene_rdl2::util::isAligned(soaBlock, AVX512_SIMD_MEMORY_ALIGNMENT));

        for (unsigned i = 0; i < numChunksPerBlock; ++i) {

            const uint32_t *__restrict srcChunk = soaBlock + (AVX512_VLEN * AVX512_VLEN) * i;

            scene_rdl2::math::transposeSOAToAOS_16x16(srcChunk, dstRows);

            for(unsigned j = 0; j < AVX512_VLEN; ++j)
                dstRows[j] += AVX512_VLEN;
        }

    } while (numElems);
}

//-----------------------------------------------------------------------------

template<unsigned SRC_AOS_SIZE,
         unsigned SRC_AOS_STRIDE,
         unsigned DST_SOA_STRIDE,
         unsigned PREFETCH_DISTANCE>
inline void
convertAOSToSOA_AVX(unsigned numElems,
                    const uint32_t *__restrict src,
                    uint32_t *__restrict dst)
{
    MNRY_STATIC_ASSERT(SRC_AOS_SIZE <= SRC_AOS_STRIDE);
    MNRY_STATIC_ASSERT(SRC_AOS_SIZE % (AVX_VLEN * 4) == 0);
    MNRY_STATIC_ASSERT(SRC_AOS_STRIDE % (AVX_VLEN * 4) == 0);
    MNRY_STATIC_ASSERT(DST_SOA_STRIDE % (AVX_VLEN * AVX_SIMD_REGISTER_SIZE) == 0);

    MNRY_ASSERT(scene_rdl2::util::isAligned(src, AVX_SIMD_MEMORY_ALIGNMENT));
    MNRY_ASSERT(scene_rdl2::util::isAligned(dst, AVX_SIMD_MEMORY_ALIGNMENT));

    const unsigned numFullBlocks = numElems / AVX_VLEN;
    const unsigned numChunksPerBlock = SRC_AOS_SIZE / AVX_SIMD_MEMORY_ALIGNMENT;

    struct ALIGN(AVX_SIMD_MEMORY_ALIGNMENT) AOSData
    {
        uint8_t mData[SRC_AOS_STRIDE];
    };

    const AOSData *aosData = (const AOSData *)src;

    //
    // Transpose all full blocks.
    //
    for (unsigned iblock = 0; iblock < numFullBlocks; ++iblock) {

        const CACHE_ALIGN uint32_t *__restrict srcRows[AVX_VLEN] =
        {
            (const uint32_t *)(&aosData[0]),
            (const uint32_t *)(&aosData[1]),
            (const uint32_t *)(&aosData[2]),
            (const uint32_t *)(&aosData[3]),
            (const uint32_t *)(&aosData[4]),
            (const uint32_t *)(&aosData[5]),
            (const uint32_t *)(&aosData[6]),
            (const uint32_t *)(&aosData[7]),
        };

        aosData += AVX_VLEN;

        for (unsigned i = 0; i < numChunksPerBlock; ++i) {

            uint32_t *__restrict dstChunk = dst + (AVX_VLEN * AVX_VLEN) * i;

            scene_rdl2::math::transposeAOSToSOA_8x8(srcRows, dstChunk);

            srcRows[0] += AVX_VLEN;
            srcRows[1] += AVX_VLEN;
            srcRows[2] += AVX_VLEN;
            srcRows[3] += AVX_VLEN;
            srcRows[4] += AVX_VLEN;
            srcRows[5] += AVX_VLEN;
            srcRows[6] += AVX_VLEN;
            srcRows[7] += AVX_VLEN;
        }

        // Take destination stride into account.
        dst += DST_SOA_STRIDE >> 2;
    }

    //
    // Transpose remaining entries into a partially filled block.
    //
    unsigned remainingEntries = numElems - (numFullBlocks * AVX_VLEN);

    if (remainingEntries) {

        MNRY_ASSERT(remainingEntries < AVX_VLEN);

        const CACHE_ALIGN uint32_t *__restrict srcRows[AVX_VLEN];

        // Fill in valid entries.
        for (unsigned i = 0; i < remainingEntries; ++i) {
            srcRows[i] = (const uint32_t *)(&aosData[i]);
        }

        // Duplicate final entries to fill out trailing lanes.
        const uint32_t *finalRow = (const uint32_t *)(&aosData[remainingEntries - 1]);
        for (unsigned i = remainingEntries; i < AVX_VLEN; ++i) {
            srcRows[i] = finalRow;
        }

        // Process remaining chunks in this block whilst prefetching from the next block.
        for (unsigned i = 0; i < numChunksPerBlock; ++i) {

            scene_rdl2::math::transposeAOSToSOA_8x8(srcRows, dst);

            srcRows[0] += AVX_VLEN;
            srcRows[1] += AVX_VLEN;
            srcRows[2] += AVX_VLEN;
            srcRows[3] += AVX_VLEN;
            srcRows[4] += AVX_VLEN;
            srcRows[5] += AVX_VLEN;
            srcRows[6] += AVX_VLEN;
            srcRows[7] += AVX_VLEN;

            dst += (AVX_VLEN * AVX_VLEN);
        }
    }
}

template<unsigned SRC_AOS_SIZE,
         unsigned SRC_AOS_STRIDE,
         unsigned DST_SOA_STRIDE,
         unsigned SORT_INDEX_STRIDE,
         unsigned PREFETCH_DISTANCE>
inline void
convertAOSToSOAIndexed_AVX(unsigned numElems,
                           const uint32_t *__restrict src,
                           uint32_t *__restrict dst,
                           const uint32_t *sortIndices)
{
    MNRY_STATIC_ASSERT(SRC_AOS_SIZE <= SRC_AOS_STRIDE);
    MNRY_STATIC_ASSERT(SRC_AOS_SIZE % (AVX_VLEN * 4) == 0);
    MNRY_STATIC_ASSERT(SRC_AOS_STRIDE % (AVX_VLEN * 4) == 0);
    MNRY_STATIC_ASSERT(DST_SOA_STRIDE % (AVX_VLEN * AVX_SIMD_REGISTER_SIZE) == 0);
    MNRY_STATIC_ASSERT(SORT_INDEX_STRIDE % sizeof(uint32_t) == 0);

    MNRY_ASSERT(scene_rdl2::util::isAligned(src, AVX_SIMD_MEMORY_ALIGNMENT));
    MNRY_ASSERT(scene_rdl2::util::isAligned(dst, AVX_SIMD_MEMORY_ALIGNMENT));

    const unsigned numFullBlocks = numElems / AVX_VLEN;
    const unsigned numChunksPerBlock = SRC_AOS_SIZE / AVX_SIMD_MEMORY_ALIGNMENT;

    struct ALIGN(AVX_SIMD_MEMORY_ALIGNMENT) AOSData
    {
        uint8_t mData[SRC_AOS_STRIDE];
    };

    const AOSData *aosData = (const AOSData *)src;

    //
    // Transpose all full blocks.
    //
    for (unsigned iblock = 0; iblock < numFullBlocks; ++iblock) {

        const CACHE_ALIGN uint32_t *__restrict srcRows[AVX_VLEN] =
        {
            (const uint32_t *)(&aosData[sortIndices[0]]),
            (const uint32_t *)(&aosData[sortIndices[SORT_INDEX_STRIDE >> 2]]),
            (const uint32_t *)(&aosData[sortIndices[SORT_INDEX_STRIDE >> 1]]),
            (const uint32_t *)(&aosData[sortIndices[3 * (SORT_INDEX_STRIDE >> 2)]]),
            (const uint32_t *)(&aosData[sortIndices[SORT_INDEX_STRIDE]]),
            (const uint32_t *)(&aosData[sortIndices[5 * (SORT_INDEX_STRIDE >> 2)]]),
            (const uint32_t *)(&aosData[sortIndices[6 * (SORT_INDEX_STRIDE >> 2)]]),
            (const uint32_t *)(&aosData[sortIndices[7 * (SORT_INDEX_STRIDE >> 2)]]),
        };
        sortIndices += AVX_VLEN * (SORT_INDEX_STRIDE >> 2);

        for (unsigned i = 0; i < numChunksPerBlock; ++i) {

            uint32_t *__restrict dstChunk = dst + (AVX_VLEN * AVX_VLEN) * i;

            scene_rdl2::math::transposeAOSToSOA_8x8(srcRows, dstChunk);

            srcRows[0] += AVX_VLEN;
            srcRows[1] += AVX_VLEN;
            srcRows[2] += AVX_VLEN;
            srcRows[3] += AVX_VLEN;
            srcRows[4] += AVX_VLEN;
            srcRows[5] += AVX_VLEN;
            srcRows[6] += AVX_VLEN;
            srcRows[7] += AVX_VLEN;
        }

        // Take destination stride into account.
        dst += DST_SOA_STRIDE >> 2;
    }

    //
    // Transpose remaining entries into a partially filled block.
    //
    unsigned remainingEntries = numElems - (numFullBlocks * AVX_VLEN);

    if (remainingEntries) {

        MNRY_ASSERT(remainingEntries < AVX_VLEN);

        const CACHE_ALIGN uint32_t *__restrict srcRows[AVX_VLEN];

        // Fill in valid entries.
        for (unsigned i = 0; i < remainingEntries; ++i) {
            srcRows[i] = (const uint32_t *)(&aosData[sortIndices[i * (SORT_INDEX_STRIDE >> 2)]]);
        }

        // Duplicate final entries to fill out trailing lanes.
        const uint32_t *finalRow = (const uint32_t *)(&aosData[sortIndices[(remainingEntries - 1) * (SORT_INDEX_STRIDE >> 2)]]);
        for (unsigned i = remainingEntries; i < AVX_VLEN; ++i) {
            srcRows[i] = finalRow;
        }

        // Process remaining chunks in this block whilst prefetching from the next block.
        for (unsigned i = 0; i < numChunksPerBlock; ++i) {

            scene_rdl2::math::transposeAOSToSOA_8x8(srcRows, dst);

            srcRows[0] += AVX_VLEN;
            srcRows[1] += AVX_VLEN;
            srcRows[2] += AVX_VLEN;
            srcRows[3] += AVX_VLEN;
            srcRows[4] += AVX_VLEN;
            srcRows[5] += AVX_VLEN;
            srcRows[6] += AVX_VLEN;
            srcRows[7] += AVX_VLEN;

            dst += (AVX_VLEN * AVX_VLEN);
        }
    }
}

template<unsigned SRC_SOA_SIZE,         // DST_AOS_SIZE is derived from this.
         unsigned SRC_SOA_STRIDE,
         unsigned INDEX_STRIDE,
         unsigned PREFETCH_DISTANCE>
inline void
convertSOAToAOSIndexed_AVX(unsigned numElems,
                           const uint32_t *indices,
                           const uint32_t *__restrict src,
                           uint32_t **__restrict dst)
{
    const unsigned DST_AOS_SIZE = SRC_SOA_SIZE / AVX_VLEN;

    MNRY_STATIC_ASSERT(SRC_SOA_SIZE <= SRC_SOA_STRIDE);
    MNRY_STATIC_ASSERT(SRC_SOA_SIZE % (AVX_SIMD_REGISTER_SIZE * AVX_VLEN) == 0);
    MNRY_STATIC_ASSERT(SRC_SOA_STRIDE % (AVX_SIMD_REGISTER_SIZE * AVX_VLEN) == 0);
    MNRY_STATIC_ASSERT(DST_AOS_SIZE % AVX_SIMD_REGISTER_SIZE == 0);
    MNRY_STATIC_ASSERT(INDEX_STRIDE % sizeof(uint32_t) == 0);

    MNRY_ASSERT(scene_rdl2::util::isAligned(src, AVX_SIMD_MEMORY_ALIGNMENT));
    MNRY_ASSERT(scene_rdl2::util::isAligned(dst, AVX_SIMD_MEMORY_ALIGNMENT));
    MNRY_ASSERT(numElems);

    CACHE_ALIGN uint32_t dummy[DST_AOS_SIZE >> 2];

    const unsigned numChunksPerBlock = SRC_SOA_SIZE / (AVX_VLEN * AVX_VLEN * 4);

    unsigned currIdx = 0;

    do {
        // Find indices within current SOA block.
        unsigned baseIdx = indices[currIdx];
        unsigned endIdx = (baseIdx + AVX_VLEN) & ~AVX_VLEN_MASK;

        CACHE_ALIGN uint32_t *__restrict dstRows[AVX_VLEN] = { dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy };

        do {
            dstRows[indices[currIdx] & AVX_VLEN_MASK] = dst[currIdx];
            ++currIdx;
            --numElems;
        } while (numElems && indices[currIdx] < endIdx);

        const uint32_t *soaBlock = src + ((SRC_SOA_STRIDE * (baseIdx >> AVX_VLEN_SHIFT)) / 4);
        MNRY_ASSERT(scene_rdl2::util::isAligned(soaBlock, AVX_SIMD_MEMORY_ALIGNMENT));

        for (unsigned i = 0; i < numChunksPerBlock; ++i) {

            const uint32_t *__restrict srcChunk = soaBlock + (AVX_VLEN * AVX_VLEN) * i;

            scene_rdl2::math::transposeSOAToAOS_8x8(srcChunk, dstRows);

            dstRows[0] += AVX_VLEN;
            dstRows[1] += AVX_VLEN;
            dstRows[2] += AVX_VLEN;
            dstRows[3] += AVX_VLEN;
            dstRows[4] += AVX_VLEN;
            dstRows[5] += AVX_VLEN;
            dstRows[6] += AVX_VLEN;
            dstRows[7] += AVX_VLEN;
        }

    } while (numElems);
}

//-----------------------------------------------------------------------------

} // namespace mcrt_common
} // namespace moonray

