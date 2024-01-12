// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

//
//
#include "TestAosSoa.h"
#include <moonray/rendering/mcrt_common/SOAUtil.h>
#include <moonray/common/time/Ticker.h>

namespace moonray {
namespace mcrt_common {

//----------------------------------------------------------------------------

//
// Here is a very simple implementation which produces the correct results.
// It serves as a baseline for performance and correctness.
//

void
doAVXRefAOSToSOA( unsigned numElems,
                  const AOSData *__restrict aosData,
                  SOABlock8 *__restrict soaBlocks,
                  SortOrder *sortOrder,
                  scene_rdl2::alloc::Arena *arena,
                  Ticks *ticks )
{
    ticks->mPreSort = time::getTicks();

    // Sorting phase:
    std::sort(sortOrder, sortOrder + numElems, [](const SortOrder &a, const SortOrder &b) -> bool {
        return a.mSortKey < b.mSortKey;
    });

    ticks->mPostSort = time::getTicks();

    // Transposition phase:
    for (unsigned i = 0; i < numElems; ++i) {

        unsigned blockIdx = i >> AVX_VLEN_SHIFT;
        unsigned laneIdx = i & AVX_VLEN_MASK;

        const AOSData *aos = aosData + sortOrder[i].mElemIdx;

        for (unsigned j = 0; j < sizeof(AOSData) / sizeof(uint32_t); ++j) {
            soaBlocks[blockIdx].mData[j][laneIdx] = aos->mData[j];
        }
    }

    // Smear final entry over trailing SOA entries.
    if ((numElems & AVX_VLEN_MASK) != 0) {

        SOABlock8 &finalSoa = soaBlocks[numElems >> AVX_VLEN_SHIFT];
        unsigned finalLaneIdx = (numElems - 1) & AVX_VLEN_MASK;

        for (unsigned i = 0; i < sizeof(AOSData) / sizeof(uint32_t); ++i) {
            uint32_t ref = finalSoa.mData[i][finalLaneIdx];
            for (unsigned j = finalLaneIdx + 1; j < AVX_VLEN; ++j) {
                finalSoa.mData[i][j] = ref;
            }
        }
    }

    ticks->mPostTranspose = time::getTicks();
}

//----------------------------------------------------------------------------

void
doAVXOptAOSToSOA( unsigned numElems,
                  const AOSData *__restrict aosData,
                  SOABlock8 *__restrict soaBlocks,
                  SortOrder *sortOrder,
                  scene_rdl2::alloc::Arena *arena,
                  Ticks *ticks)
{
    ticks->mPreSort = time::getTicks();

    scene_rdl2::util::inPlaceRadixSort32(numElems, sortOrder, arena);

    ticks->mPostSort = time::getTicks();

    convertAOSToSOAIndexed_AVX<sizeof(AOSData),
                               sizeof(AOSData),
                               sizeof(SOABlock8),
                               sizeof(SortOrder),
                               0>
        (numElems, (const uint32_t *)aosData, (uint32_t *)soaBlocks, &sortOrder[0].mElemIdx);

    ticks->mPostTranspose = time::getTicks();
}

//----------------------------------------------------------------------------

void
doAVXOptSOAToAOS( unsigned numElems,
                  const SOABlock8 *__restrict soaBlocks,
                  AOSData **__restrict aosData,
                  uint32_t *indices,
                  scene_rdl2::alloc::Arena *arena,
                  Ticks *ticks )
{
    ticks->mPreSort = ticks->mPostSort = time::getTicks();

    convertSOAToAOSIndexed_AVX<sizeof(SOABlock8),
                               sizeof(SOABlock8),
                               sizeof(uint32_t),
                               0>
        (numElems, indices, (const uint32_t *)soaBlocks, (uint32_t **)aosData);

    ticks->mPostTranspose = time::getTicks();
}

//----------------------------------------------------------------------------

} // namespace mcrt_common
} // namespace moonray

