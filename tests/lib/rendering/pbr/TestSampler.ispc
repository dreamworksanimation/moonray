// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#include <moonray/rendering/pbr/sampler/SequenceID.isph>
#include <moonray/rendering/pbr/sampler/Sampling.isph>
#include <scene_rdl2/common/platform/Platform.isph>

int32 toInt32(const uniform int32 v[])
{
    int32 r;
    for (uniform int i = 0; i < VLEN; ++i) {
        r = insert(r, i, v[i]);
    }
    return r;
}

unsigned int toUInt(const uniform unsigned int v[])
{
    unsigned int r;
    for (uniform int i = 0; i < VLEN; ++i) {
        r = insert(r, i, v[i]);
    }
    return r;
}

float toFloat(const uniform float v[])
{
    float r;
    for (uniform int i = 0; i < VLEN; ++i) {
        r = insert(r, i, v[i]);
    }
    return r;
}

void fromInt32(uniform int32 r[], varying int32 v)
{
    for (uniform int i = 0; i < VLEN; ++i) {
        r[i] = extract(v, i);
    }
}

void fromFloat(uniform float r[], varying float v)
{
    for (uniform int i = 0; i < VLEN; ++i) {
        r[i] = extract(v, i);
    }
}

// DO NOT USE FOR PRODUCTION. Slow. Only for testing from C++.
export void PBRTest_SequenceID0(const uniform int32 seed[],
                                uniform int32 output[])
{
    const varying unsigned int32 iseed = (unsigned int32) toInt32(seed);

    varying SequenceID sid;
    SequenceID_init(sid);

    const varying unsigned int32 r = SequenceID_getHash(sid, iseed);
    fromInt32(output, (int32) r);
}

// DO NOT USE FOR PRODUCTION. Slow. Only for testing from C++.
export void PBRTest_SequenceID1(const uniform int32 seed[],
                                uniform int32 output[],
                                uniform unsigned int32 hash0)
{
    const varying unsigned int32 iseed = (unsigned int32) toInt32(seed);

    varying SequenceID sid;
    SequenceID_init(sid, hash0);

    const varying unsigned int32 r = SequenceID_getHash(sid, iseed);
    fromInt32(output, (int32) r);
}

// DO NOT USE FOR PRODUCTION. Slow. Only for testing from C++.
export void PBRTest_SequenceID2(const uniform int32 seed[],
                                uniform int32 output[],
                                uniform unsigned int32 hash0,
                                uniform unsigned int32 hash1)
{
    const varying unsigned int32 iseed = (unsigned int32) toInt32(seed);

    varying SequenceID sid;
    SequenceID_init(sid, hash0, hash1);

    const varying unsigned int32 r = SequenceID_getHash(sid, iseed);
    fromInt32(output, (int32) r);
}

// DO NOT USE FOR PRODUCTION. Slow. Only for testing from C++.
export void PBRTest_SequenceID3(const uniform int32 seed[],
                                uniform int32 output[],
                                uniform unsigned int32 hash0,
                                uniform unsigned int32 hash1,
                                uniform unsigned int32 hash2)
{
    const varying unsigned int32 iseed = (unsigned int32) toInt32(seed);

    varying SequenceID sid;
    SequenceID_init(sid, hash0, hash1, hash2);

    const varying unsigned int32 r = SequenceID_getHash(sid, iseed);
    fromInt32(output, (int32) r);
}

// DO NOT USE FOR PRODUCTION. Slow. Only for testing from C++.
export void PBRTest_SequenceID4(const uniform int32 seed[],
                                uniform int32 output[],
                                uniform unsigned int32 hash0,
                                uniform unsigned int32 hash1,
                                uniform unsigned int32 hash2,
                                uniform unsigned int32 hash3)
{
    const varying unsigned int32 iseed = (unsigned int32) toInt32(seed);

    varying SequenceID sid;
    SequenceID_init(sid, hash0, hash1, hash2, hash3);

    const varying unsigned int32 r = SequenceID_getHash(sid, iseed);
    fromInt32(output, (int32) r);
}

// DO NOT USE FOR PRODUCTION. Slow. Only for testing from C++.
export void PBRTest_SequenceID5(const uniform int32 seed[],
                                uniform int32 output[],
                                uniform unsigned int32 hash0,
                                uniform unsigned int32 hash1,
                                uniform unsigned int32 hash2,
                                uniform unsigned int32 hash3,
                                uniform unsigned int32 hash4)
{
    const varying unsigned int32 iseed = (unsigned int32) toInt32(seed);

    varying SequenceID sid;
    SequenceID_init(sid, hash0, hash1, hash2, hash3, hash4);

    const varying unsigned int32 r = SequenceID_getHash(sid, iseed);
    fromInt32(output, (int32) r);
}

#if 0
// DO NOT USE FOR PRODUCTION. Slow. Only for testing from C++.
// The memory layout is easier to deal with if we only return one lane at a
// time (it's test code -- I don't care if we're doing redudnant work).
// Preconditions: lane must be less than VLEN.
void PBRTest_IntegratorSample1D(varying IntegratorSample1D sampler,
                                uniform float output0[],
                                uniform unsigned int32 lane,
                                uniform unsigned int32 numSamples)
{
    const uniform unsigned int32 depth = 0;

    for (uniform unsigned int32 s = 0; s < numSamples; ++s) {
        varying float data;
        getSample(sampler, data, depth); // Can we overload this? I think so...
        output0[s] = extract(data, lane);
    }
}

// DO NOT USE FOR PRODUCTION. Slow. Only for testing from C++.
// The memory layout is easier to deal with if we only return one lane at a
// time (it's test code -- I don't care if we're doing redudnant work).
// Preconditions: lane must be less than VLEN.
void PBRTest_IntegratorSample2D(varying IntegratorSample2D sampler,
                                uniform float output0[],
                                uniform float output1[],
                                uniform unsigned int32 lane,
                                uniform unsigned int32 numSamples)
{
    const uniform unsigned int32 depth = 0;

    for (uniform unsigned int32 s = 0; s < numSamples; ++s) {
        varying float data[2];
        getSample(sampler, data, depth); // Can we overload this? I think so...
        output0[s] = extract(data[0], lane);
        output1[s] = extract(data[1], lane);
    }
}

// DO NOT USE FOR PRODUCTION. Slow. Only for testing from C++.
// The memory layout is easier to deal with if we only return one lane at a
// time (it's test code -- I don't care if we're doing redundant work).
// Preconditions: lane must be less than VLEN.
export void PBRTest_IntegratorSample1DIndefiniteSize(uniform float output0[],
                                                     uniform unsigned int32 lane,
                                                     uniform unsigned int32 numSamples)
{
    varying SequenceID sid;
    SequenceID_init(sid, 23, 97, 89);

    varying IntegratorSample1D sampler;
    IntegratorSample1D_init(sampler, sid);

    PBRTest_IntegratorSample1D(sampler, output0, lane, numSamples);
}

export void PBRTest_IntegratorSample1DDefiniteSize(uniform float output0[],
                                                   uniform unsigned int32 lane,
                                                   uniform unsigned int32 numSamples)
{
    varying SequenceID sid;
    SequenceID_init(sid, 23, 97, 89);

    varying IntegratorSample1D sampler;
    IntegratorSample1D_init(sampler, sid);
    sampler.mTotalSamples = numSamples;

    PBRTest_IntegratorSample1D(sampler, output0, lane, numSamples);
}

// DO NOT USE FOR PRODUCTION. Slow. Only for testing from C++.
// The memory layout is easier to deal with if we only return one lane at a
// time (it's test code -- I don't care if we're doing redundant work).
// Preconditions: lane must be less than VLEN.
export void PBRTest_IntegratorSample2DIndefiniteSize(uniform float output0[],
                                                     uniform float output1[],
                                                     uniform unsigned int32 lane,
                                                     uniform unsigned int32 numSamples)
{
    varying SequenceID sid;
    SequenceID_init(sid, 23, 97, 89);

    varying IntegratorSample2D sampler;
    IntegratorSample2D_init(sampler, sid);

    PBRTest_IntegratorSample2D(sampler, output0, output1, lane, numSamples);
}

export void PBRTest_IntegratorSample2DDefiniteSize(uniform float output0[],
                                                   uniform float output1[],
                                                   uniform unsigned int32 lane,
                                                   uniform unsigned int32 numSamples)
{
    varying SequenceID sid;
    SequenceID_init(sid, 23, 97, 89);

    varying IntegratorSample2D sampler;
    IntegratorSample2D_init(sampler, sid);
    sampler.mTotalSamples = numSamples;

    PBRTest_IntegratorSample2D(sampler, output0, output1, lane, numSamples);
}
#endif

// DO NOT USE FOR PRODUCTION. Slow. Only for testing from C++.
export void PBRTest_testPermutationSequence(const uniform uint32 indexMax, // Arrays need to be at least this size
                                            const uniform uint32 seed,
                                            const uniform uint32 permutationSize,
                                            uniform uint32 output[])
{
    foreach (idx = 0 ... indexMax) {
        output[idx] = permuteSequence(seed, idx, permutationSize);
    }
}

// DO NOT USE FOR PRODUCTION. Slow. Only for testing from C++.
export void PBRTest_testPermutationSequenceFast(const uniform uint32 indexMax, // Arrays need to be at least this size
                                                const uniform uint32 seed,
                                                const uniform uint32 permutationSize,
                                                uniform uint32 output[])
{
    foreach (idx = 0 ... indexMax) {
        output[idx] = permuteSequenceFast(seed, idx, permutationSize);
    }
}

// DO NOT USE FOR PRODUCTION. Slow. Only for testing from C++.
export void PBRTest_testPermutation(const uniform uint32 indexMax, // Arrays need to be at least this size
                                    const uniform uint32 seed,
                                    const uniform uint32 permutationSize,
                                    uniform uint32 output[])
{
    foreach (idx = 0 ... indexMax) {
        output[idx] = permuteIndexAbsolute(seed, idx, permutationSize);
    }
}

// DO NOT USE FOR PRODUCTION. Slow. Only for testing from C++.
export void PBRTest_testPermutationFast(const uniform uint32 indexMax, // Arrays need to be at least this size
                                        const uniform uint32 seed,
                                        const uniform uint32 permutationSize,
                                        uniform uint32 output[])
{
    foreach (idx = 0 ... indexMax) {
        output[idx] = permuteIndexFastAbsolute(seed, idx, permutationSize);
    }
}

