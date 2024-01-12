// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#include "NPoint.h"
#include "util.h"
#include "DynamicHyperGrid.h"
#include "ProgressBar.h"

#include <iostream>
#include <vector>
#include <functional>
#include <algorithm>
#include <string>
#include <chrono> 
#include <cstdint>
#include <random>
#include <limits>
#include <fstream>
#include <chrono>
#include <deque>

struct Sample
{
    static constexpr unsigned sDims = 2;
    using InternalType = double; // internal type

    double x, y;

    double& operator[](int i) {
        switch (i) {
            case 0: return x;
            default: return y;
        }
    }

    const double& operator[](int i) const {
        switch (i) {
            case 0: return x;
            default: return y;
        }
    }
};

using RNG  = std::mt19937_64;
using Grid = DynamicHyperGrid<Sample, 2>;

using std::uint_fast64_t;
using std::uint64_t;
using std::size_t;

// Returns a seed used for RNG (currently we use mt19937_64).
// If PMJ02_RANDOM_SEED is not defined, we'll use a seed that
// was empirically chosen with the std::mt19937_64 random number generator.
uint_fast64_t getSeed()
{
#if defined(PMJ02_RANDOM_SEED)
    std::random_device rd;
    std::uniform_int_distribution<uint_fast64_t> dist;
    return dist(rd);
#else
    return 8074967353239662646;
#endif
}

// Checks if an unsigned 64 bit number is a power of 4 or not.
bool isPowerOf4(uint64_t n)
{
    // Check that it's a power of 2. This is technically not necessary
    // as this function is used in context where n is guaranteed to be a power of 2.
    if (n == 0 || (n & (n - 1)) != 0) { return false; }
    // Check that the number of trailing zeroes is even:
    return n & 0x5555555555555555ULL;
}

class BitVec
{
public:
    BitVec(size_t numBits)
    {
        const size_t size = (numBits + BIT_CNT - 1) / BIT_CNT;
        m_bits.assign(size, 0);
    }

    void unsetResize(size_t numBits) 
    {
        const size_t newSize = (numBits + BIT_CNT - 1) / BIT_CNT;
        m_bits.assign(newSize, 0);
    }

    void set(size_t i)
    {
        const size_t blockIdx = i / BIT_CNT;
        const size_t bitIdx   = i % BIT_CNT;
        m_bits[blockIdx] |= 1ULL << bitIdx;
    }

    bool isSet(size_t i) const
    {
        const size_t blockIdx = i / BIT_CNT;
        const size_t bitIdx   = i % BIT_CNT;
        return (m_bits[blockIdx] & (1ULL << bitIdx)) != 0;
    }

private:
    // Empirically, this performs better than uint8_t or uint32_t (and actually vector<bool> for that matter).
    static const size_t   BIT_CNT = 64;
    std::vector<uint64_t> m_bits;
};

class Strata
{
public:
    Strata() :
        m_dim(1), m_numSamples(1), m_level(1), m_strata(1)
    {
    }

    int getDim() const { return m_dim; }

    // Subdivides the strata. This requires updating the tree again, so we need to pass the samples along.
    void subdivideStrata(const std::vector<Sample>& samples)
    {
        // Subdividing the strata doubles the number of samples: ({1x1}, {1x2, 2x1}, {2x2, 1x4, 4x1}, ...)
        m_numSamples *= 2;
        // The dimension only changes when we swap from a power of 4 (that is, after the "square" strata).
        // So, dim is: (1, 2, 2, 4, 4, 8, 8, ...)
        if (!isPowerOf4(m_numSamples)) {
            m_dim *= 2;
        }

        // At each level of the stratum, we have 1 extra level:
        ++m_level;
        m_strata.unsetResize(m_numSamples * m_level);

        // Then, update the new stratum with the old values.
        // Might we worth parallelizing this...
        for (const Sample& s : samples) {
            updateStrata(s);
        }
    }

    // Generates a sample in a specific stratum, updating the tree structure.
    Sample genSample(Grid& grid, uint64_t numCandidates, uint64_t x, uint64_t y, RNG& rng, std::vector<uint64_t>& xoffsets, std::vector<uint64_t>& yoffsets)
    {
        // First, we get stratum:
        xoffsets.clear();
        yoffsets.clear();
        getValidOffsets(x, y, xoffsets, yoffsets);

        const double width = 1.0 / m_numSamples;

        std::uniform_int_distribution<size_t> xDist(0, xoffsets.size() - 1);
        std::uniform_int_distribution<size_t> yDist(0, yoffsets.size() - 1);

        // Generate the candidate samples:
        Sample bestCandidate;
        double farthestD2 = 0.0;
        for (uint64_t i = 0; i < numCandidates; ++i) {
            // Get a random strata:
            const uint64_t xoffset = xoffsets[xDist(rng)];
            const uint64_t yoffset = yoffsets[yDist(rng)];

            // Generate a sample point for this strata:
            const Sample candidate {
                std::uniform_real_distribution<double>(width * xoffset, width * (xoffset + 1))(rng), 
                std::uniform_real_distribution<double>(width * yoffset, width * (yoffset + 1))(rng)
            };

            // Get it's nearest distance:
            double nearestD2 = std::numeric_limits<double>::max();
            const auto nearFn = [&candidate, &nearestD2](const Sample& s) {
                nearestD2 = std::min(nearestD2, wrappedDistanceSquared(candidate, s));
            };
            grid.visitNeighbors(nearFn, candidate);

            if (nearestD2 > farthestD2) {
                farthestD2 = nearestD2;
                bestCandidate = candidate;
            }
        }

        grid.add(bestCandidate);

        // Update the existence of this sample:
        updateStrata(bestCandidate);

        return bestCandidate;
    }

private:
    void updateStrata(const Sample& sample)
    {
        // Here, we essentially loop over all of the elementary intervals for the given number of
        // samples. For instance, if we have 16 samples, we'd loop over: {(16, 1), (8, 2), (4, 4), (2, 8), (1, 16)}.
        // At each "level", we update the strata with true if the sample belongs to it.
        for (uint64_t level = 0, nx = m_numSamples, ny = 1;
             nx >= 1;
             nx /= 2, ny *= 2, ++level) 
        {
            const uint64_t x = sample.x * nx;
            const uint64_t y = sample.y * ny;
            const uint64_t index = (level * m_numSamples) + (y * nx + x);
            m_strata.set(index);
        }
    }

    void getXOffsets(uint64_t x, uint64_t y, uint64_t level, std::vector<uint64_t>& xoffsets) const
    {
        // Check, at the specific grid index, the elementary interval is filled. We need the "nx" term in order
        // to do this, we simply have to do: 2^(m_level - level - 1) or 1 << (m_level - level -1):
        const uint64_t index = (level * m_numSamples) + (y * (1ULL << (m_level - level - 1ULL)) + x);
        if (!m_strata.isSet(index)) {
            // We reached the lowest level, so we are done (e.g. for 16 samples: (16, 1))
            if (level == 0) {
                xoffsets.emplace_back(x);
            } else {
                getXOffsets(x*2,     y/2, level-1, xoffsets);
                getXOffsets(x*2 + 1, y/2, level-1, xoffsets);
            }
        }
    }

    void getYOffsets(uint64_t x, uint64_t y, uint64_t level, std::vector<uint64_t>& yoffsets) const
    {
        // Same as above, basically, only we work backwards. This is because of the ordering of the elementary
        // intervals in m_strata (see updateStrata for details).
        // Only, we need to keep track of the number of columns ourselves. That is,
        // if nx == 1, then we are done (e.g. for 16 samples: (1, 16))
        const uint64_t nx    = 1ULL << (m_level - level - 1ULL);
        const uint64_t index = (level * m_numSamples) + (y * nx + x);
        if (!m_strata.isSet(index)) {
            if (nx == 1) {
                yoffsets.emplace_back(y);
            } else {
                getYOffsets(x/2, y*2,     level+1, yoffsets);
                getYOffsets(x/2, y*2 + 1, level+1, yoffsets);
            }
        }
    }

    void getValidOffsets(uint64_t x, uint64_t y, std::vector<uint64_t>& xoffsets, std::vector<uint64_t>& yoffsets) const
    {
        // If we have an even number of levels, we don't have a square elementary interval. To handle that,
        // we must half one of the grid sides. For instance, if m_level == 4, then {8x1, 4x2, 2x4, 1x8} are the
        // elementary intervals. So for x, we wish to start at (4x2), and at (2x4) for y.
        if (m_level % 2 == 0) {
            getXOffsets(x,   y/2, m_level/2 - 1, xoffsets);
            getYOffsets(x/2, y,   m_level/2, yoffsets);
        } else {
            getXOffsets(x, y, m_level/2, xoffsets);
            getYOffsets(x, y, m_level/2, yoffsets);
        }
    }

private:
    uint64_t         m_dim;        // This is sqrt(n) if n is a power of 4.
    uint64_t         m_numSamples; // The total number of samples we're currently working with.
    uint64_t         m_level;      // The current level we are dealing with.
    BitVec           m_strata;     // All of the stratum go here.
};

std::vector<Sample> genSamples(uint64_t numSamples, uint64_t numCandidates, RNG& rng)
{
    std::vector<Sample> samples;
    samples.reserve(numSamples);

    Grid grid;

    ProgressBar progress(numSamples);

    // Keep memory allocated here so we don't keep reallocating memory.
    std::vector<uint64_t> xoffsets, yoffsets;

    // Generate initial sampling point:
    Strata strata;
    samples.emplace_back(strata.genSample(grid, numCandidates, 0, 0, rng, xoffsets, yoffsets));

    std::bernoulli_distribution coinflip(0.5);

    for (uint64_t n = 1; n < numSamples; n *= 4) {
        strata.subdivideStrata(samples);

        std::uniform_int_distribution<uint64_t> shuffleDist(0, n - 1);

        // Now we go through and calculate the diagonal values at our specific grid level:
        uint64_t oldShuffle = shuffleDist(rng);
        for (uint64_t i = 0; (i < n) && (n + i < numSamples); ++i) {
            const Sample& s = samples[i ^ oldShuffle];
            const uint64_t x = s.x * strata.getDim();
            const uint64_t y = s.y * strata.getDim();
            samples.emplace_back(strata.genSample(grid, numCandidates, x ^ 1, y ^ 1, rng, xoffsets, yoffsets));

            progress.update();
            progress.draw();
        }

        strata.subdivideStrata(samples);

        // Now we should have stratum that are square. When picking diagonals here, we use Simon Brown's
        // rust implementation idea of randomly swapping a value.
        oldShuffle = shuffleDist(rng);
        const uint64_t flip = coinflip(rng);
        for (uint64_t i = 0; (i < n) && (2 * n + i < numSamples); ++i) {
            const Sample& s = samples[i ^ oldShuffle];
            const uint64_t x = s.x * strata.getDim();
            const uint64_t y = s.y * strata.getDim();
            samples.emplace_back(strata.genSample(grid, numCandidates, x ^ flip, y ^ flip ^ 1, rng, xoffsets, yoffsets));

            progress.update();
            progress.draw();
        }

        // Now we find the diagonal of the quads we placed above. To do this, we need to pick a diagonal for
        // the numbers we just appended. Those happen to be at the 2n position, so we can do [(n << 1) | (i)] to
        // index these values (it might make more sense to just perform an addition here...)
        oldShuffle = shuffleDist(rng);
        for (uint64_t i = 0; (i < n) && (3 * n + i < numSamples); ++i) {
            const Sample& s = samples[(n << 1) | (i ^ oldShuffle)];
            const uint64_t x = s.x * strata.getDim();
            const uint64_t y = s.y * strata.getDim();
            samples.emplace_back(strata.genSample(grid, numCandidates, x ^ 1, y ^ 1, rng, xoffsets, yoffsets));

            progress.update();
            progress.draw();
        }
    }

    std::cout << std::endl;

    return samples;
}

// Example usage of this program:
// ./pmj02 4096 2048 pmj02.txt
// The above will generate 4096 * 64 * 64 samples, using 2048 candidates to pick the best sample, and writing the data to pmj02.txt.
// Note that it's 4096 * 64 * 64 because we split the samples on a 64 by 64 grid.
int main(int argc, char** argv)
{
    if (argc != 4) {
        std::cout << "Usage: <num_samples_per_grid> <num_candidates> <output_file_name>\n";
        return 1;
    }

    const char* fileName = argv[3];

    const uint64_t numSamples    = std::stoull(argv[1]) * 64 * 64; // 64 by 64 grid.
    const uint64_t numCandidates = std::stoull(argv[2]);

    RNG rng(getSeed());
    const auto points = genSamples(numSamples, numCandidates, rng);

    // Write the points (this is like NPoint's "writePoints", but support for double precision).

    std::ofstream outputfile(fileName);

    IOStateSaver stateraii(outputfile);
    outputfile.exceptions(std::ios_base::failbit);
    outputfile.precision(9);
    outputfile.setf(std::ios_base::fixed | std::ios_base::showpoint);

    for (const auto& p : points) {
        outputfile << p.x << ' ' << p.y << '\n';
    }
}