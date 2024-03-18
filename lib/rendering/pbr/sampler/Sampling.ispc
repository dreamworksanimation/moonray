// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#include "Permute.isph"
#include "RSequence.isph"
#include "Sampling.isph"
#include "SequenceID.isph"
#include <moonray/rendering/pbr/Types.isph>

// "Correlated Multi-Jittered Sampling"
// Andrew Kensler
// Pixar Technical Memo 13-01, 2013
varying float randfloat(varying unsigned int i, uniform unsigned int p)
{
    i ^= p;
    i ^= i >> 17;
    i ^= i >> 10; i *= 0xb36534e5;
    i ^= i >> 12;
    i ^= i >> 21; i *= 0x93fc4795;
    i ^= 0xdf6e307f;
    i ^= i >> 17; i *= 1 | p >> 18;
    return bitsToFloat(useLowOrderBits(i));
}

// Warps samples to quadratic B-spline filter.
// Separable: run once for x and y dimensions.
// Input: samples in [0, 1].
// Output: samples in [-1.5, 1.5]
export void PBR_quadraticBSplineWarp(uniform unsigned int size,
                                     uniform float samples[])
{
    foreach (i = 0 ... size) {
        const float r = samples[i];

        if (r < 1.0f/6.0f) {
            samples[i] = pow(6.0f * r, 1.0f/3.0f) - 3.0f/2.0f;
        } else if (r < 5.0f/6.0f) {
            float u = (6.0f*r - 3.0f) / 4.0f;
            for (uniform int j = 0; j < 4; ++j) {
                u = (8*u*u*u - 12*r + 6)/(12*u*u - 9);
            }
            samples[i] = u;
        } else {
            samples[i] = 3.0f/2.0f - pow(6.0f * (1.0f - r), 1.0f/3.0f);
        }
    }
}

// Input r in [0, 1)
// Newton's method used to approximate a portion of the cubicBSplineWarp for
// the middle curves.
inline float distb1(float r)
{
    float u = r;
    for (uniform int j = 0; j < 5; ++j) {
        u = (11*r + u*u*(6 + u*(8 - 9*u)))/(4 + 12*u*(1 + u*(1 - u)));
    }
    return u;
}

// Warps samples to cubic B-spline filter.
// Separable: run once for x and y dimensions.
// Input: samples in [0, 1].
// Output: samples in [-2, 2]
export void PBR_cubicBSplineWarp(uniform unsigned int size,
                                 uniform float samples[])
{
    foreach (i = 0 ... size) {
        const float r = samples[i];

        if (r < 1.0f/24.0f) {
            samples[i] = pow(24.0f * r, 0.25f) - 2.0f;
        } else if (r < 0.5f) {
            samples[i] = distb1(24.0f/11.0f * (r - 1.0f/24.0f)) - 1.0f;
        } else if (r < 23.0f/24.0f) {
            samples[i] = 1.0f - distb1(24.0f/11.0f * (23.0f/24.0f - r));
        } else {
            samples[i] = 2.0f - pow(24.0f * (1.0f - r), 0.25f);
        }
    }
}

export void PBR_filterOffset(uniform unsigned int size, uniform float values[],
                             uniform float widthMult)
{
    foreach (i = 0 ... size) {
        values[i] *= widthMult;
        values[i] += 0.5f;
    }
}

// Inputs u and v in [0, 1].
// Outputs u and v in [-1, 1].
export void PBR_toUnitDisk(uniform unsigned int size, uniform float u[],
                           uniform float v[])
{
    foreach (i = 0 ... size) {
        if (u[i] == 0.5f && v[i] == 0.5f) {
            u[i] = 0.0f;
            v[i] = 0.0f;
        } else {
            float phi,r;
            const float a = 2.0f * u[i] - 1.0f;
            const float b = 2.0f * v[i] - 1.0f;

            if (abs(a) > abs(b)) {
                r = a;
                phi = (PI/4)*(b/a);
            } else {
                r = b;
                phi = (PI/2) - (PI/4)*(a/b);
            }

            float cosPhi;
            float sinPhi;

            sincos(phi, &sinPhi, &cosPhi);

            u[i] = r*cosPhi;
            v[i] = r*sinPhi;
        }
    }
}

varying float PBR_cranleyPattersonRotationMulti(varying float data,
                                                varying int32 seed)
{
    const varying float x = bitsToFloat(useLowOrderBits(seed));
    assert(x >= 0.0f);
    assert(x <  1.0f);

    data += x;
    data = (data >= 1.0f) ? data - 1.0f : data;

    assert(data >= 0.0f);
    assert(data <  1.0f);

    return data;
}

export void PBR_cranleyPattersonRotation(uniform unsigned int size,
                                         uniform float data[],
                                         uniform int32 seed)
{
    const uniform float x = bitsToFloat(useLowOrderBits(seed));
    assert(x >= 0.0f);
    assert(x <  1.0f);
    foreach (i = 0 ... size) {
        assert(data[i] >= 0.0f);
        assert(data[i] <  1.0f);
        data[i] += x;
        data[i] = (data[i] >= 1.0f) ? data[i] - 1.0f : data[i];
        assert(data[i] >= 0.0f);
        assert(data[i] <  1.0f);
    }
}

export void PBR_correlatedMultiJitter1D(uniform unsigned int arraySize,
                                        uniform float output[],
                                        uniform unsigned int totalSamples,
                                        uniform unsigned int startSampleN,
                                        uniform unsigned int scramble)
{
    // The permutation function can hang if the value to permute is larger than
    // the permutation size, so we have to be careful to generate the right
    // number of samples.
    assert(totalSamples >= startSampleN);
    const uniform unsigned int samplesToGenerate = min(arraySize, totalSamples - startSampleN);

    foreach (i = 0 ... samplesToGenerate) {
        const unsigned int s = permute(i+startSampleN, totalSamples, scramble * 0xa511e9b3);
        const float j = randfloat(s, scramble * 0x711ad6a5);
        output[i] = (s+j) / totalSamples;
    }
}

export void PBR_correlatedMultiJitter2D(uniform unsigned int arraySize,
                                        uniform float output1[],
                                        uniform float output2[],
                                        uniform unsigned int totalSamples,
                                        uniform unsigned int startSampleN,
                                        uniform unsigned int scramble)
{
    const uniform int m = sqrt((uniform float)totalSamples);
    const uniform int n = (totalSamples + m - 1) / m;

    // The permutation function can hang if the value to permute is larger than
    // the permutation size, so we have to be careful to generate the right
    // number of samples.
    assert(totalSamples >= startSampleN);
    const uniform unsigned int samplesToGenerate = min(arraySize, totalSamples - startSampleN);
    //print("STG: %\n", samplesToGenerate);
    foreach (i = 0 ... samplesToGenerate) {
        const int s = permute(i+startSampleN, totalSamples, scramble * 0x51633e2d);
        const int sx = permute(s % m, m, scramble * 0x68bc21eb);
        const int sy = permute(s / m, n, scramble * 0x02e5be93);
        const float jx = randfloat(s, scramble * 0x967a889b);
        const float jy = randfloat(s, scramble * 0x368cc8b7);
        output1[i] = (sx + (sy + jx) / n) / m;
        output2[i] = (s + jy) / totalSamples;
    }
}

void
partitioned1D(const varying IntegratorSample1D& is,
              varying float& output,
              const uniform FrameState& fs)
{
    const uniform uint32 kNumSequences = 4096u;
    const uniform uint32 kNumSamplesPerSequence = 1024u;

    if (is.mSampleNumber >= kNumSamplesPerSequence) {
        // We use a different sampler when we're going to run out of pre-computed samples.
        // We could partially fill in samples until we run out, but that complicates the logic.
        const varying uint32_t scramble = SequenceID_getHash(is.mSequenceID, 0xdf4f4915);
        const varying uint32_t sampleNum = is.mSampleNumber - kNumSamplesPerSequence;
        output = jitteredR1(sampleNum, scramble);
    } else {
        const varying uint32_t scramble = SequenceID_getHash(is.mSequenceID, 0x0740eb57);
        const varying uint32 seqNum = scramble & (kNumSequences - 1u); // scramble3 % kNumSequences;
        const uniform float* const uniform p = fs.mSamples1D;
        output = p[seqNum * kNumSamplesPerSequence + is.mSampleNumber];
    }

    PBR_cranleyPattersonRotationMulti(output, SequenceID_getHash(is.mSequenceID, 0x5b748587));
}

void
partitioned2D(const varying IntegratorSample2D& is,
              varying float& output1,
              varying float& output2,
              const uniform FrameState& fs)
{
    const uniform uint32 kNumSequences = 4096u;
    const uniform uint32 kNumSamplesPerSequence = 1024u;

    if (is.mSampleNumber >= kNumSamplesPerSequence) {
        // We use a different sampler when we're going to run out of pre-computed samples.
        // We could partially fill in samples until we run out, but that complicates the logic.
        const varying uint32_t scramble = SequenceID_getHash(is.mSequenceID, 0xa81972b7);
        const varying uint32_t sampleNum = is.mSampleNumber - kNumSamplesPerSequence;
        const varying Sample2D s = jitteredR2(sampleNum, scramble);
        output1 = s.u;
        output2 = s.v;
    } else {
        const varying uint32_t scramble = SequenceID_getHash(is.mSequenceID, 0x4a770fdf);
        const varying uint32 seqNum = scramble & (kNumSequences - 1u); // scramble % kNumSequences;
        const uniform Sample2D* const uniform p = (const uniform Sample2D* uniform)fs.mSamples2D;
        const varying Sample2D s = p[seqNum * kNumSamplesPerSequence + is.mSampleNumber];
        output1 = s.u;
        output2 = s.v;
    }

    PBR_cranleyPattersonRotationMulti(output1, SequenceID_getHash(is.mSequenceID, 0xd0944adb));
    PBR_cranleyPattersonRotationMulti(output2, SequenceID_getHash(is.mSequenceID, 0x0662aa53));
}


