// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#include "ArgumentParser.h"
#include "NPoint.h"
#include "DynamicHyperGrid.h"
#include "PerfectPowerArray.h"
#include "ProgressBar.h"
#include "util.h"

#include <algorithm>
#include <array>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <queue>
#include <string>
#include <vector>
#include <cmath>
#include <cstddef>

#define DIMENSIONS 2
#define GRID_SIZE 64
#include "pd_generation.h"

const unsigned kDims = DIMENSIONS;
const unsigned kN1DStrata = GRID_SIZE;
const float kStratumWidth = 1.0f / kN1DStrata;

typedef DynamicHyperGrid<NPoint<kDims>, kDims> DynamicGrid;

template <unsigned D, typename RNG>
inline NPoint<D> sampleWithinStratum(const NPoint<D>& min, const NPoint<D>& max, RNG& rng)
{
    NPoint<D> r;
    for (unsigned d = 0; d < D; ++d) {
        do {
            r[d] = lerp(canonical(rng), min[d], max[d]);
        } while (r[d] >= max[d]);
        assert(r[d] >= 0.0f);
        assert(r[d] <  1.0f);
    }
    return r;
}

template <unsigned D>
class StratumSampler
{
    NPoint<D> mStratumMin;
    NPoint<D> mStratumMax;

public:
    StratumSampler(unsigned stratumXIdx, unsigned stratumYIdx) :
        mStratumMin({stratumXIdx * kStratumWidth, stratumYIdx * kStratumWidth}),
        mStratumMax({mStratumMin[0] + kStratumWidth, mStratumMin[1] + kStratumWidth})
    {
    }

    template <typename RNG>
    NPoint<D> operator()(RNG& rng)
    {
        return sampleWithinStratum<D>(mStratumMin, mStratumMax, rng);
    }
};

class FibonacciSampler
{
    const std::vector<NPoint<2>>& mGenPoints;

    void polarToCartesian(float r, float phi, float& x, float& y)
    {
        float sinPhi;
        float cosPhi;
#if defined(__clang__)
    __sincosf(phi, &sinPhi, &cosPhi);
#else
    sincosf(phi, &sinPhi, &cosPhi);
#endif
        x = r * cosPhi;
        y = r * sinPhi;
    }

public:
    explicit FibonacciSampler(const std::vector<NPoint<2>>& p) : mGenPoints(p) {}

    template <typename RNG>
    NPoint<2> operator()(RNG& rng)
    {
        const auto size = mGenPoints.size();

        if (size == 0) {
            return generateRandomPoint<2>(rng);
        } else {
            typedef std::uniform_int_distribution<std::size_t> int_distr_t;
            typedef int_distr_t::param_type int_param_t;

            int_distr_t intDist;

            const float tau = (1.0f + std::sqrt(5.0f)) / 2.0f;
            const float phi = (2.0f * pi) * (1.0f - 1.0f/tau);

            const std::size_t kNumFibPoints = 30;
            const std::size_t j = intDist(rng, int_param_t(0, kNumFibPoints - 1u));

            float rj = std::sqrt(static_cast<float>(j)/static_cast<float>(kNumFibPoints));
            rj += 1.0f;
            rj /= 4.0f; // Map to [0, 0.5]
            const float thetaj = j * phi;

            float x;
            float y;
            polarToCartesian(rj, thetaj, x, y);

            // Pick a random point
            NPoint<2> p = mGenPoints[intDist(rng, int_param_t(0, size - 1u))];
            p += NPoint<2>({x, y});

            // Wrap point
            while (p[0] < 0.0f) {
                p[0] += 1.0f;
            }
            while (p[1] < 0.0f) {
                p[1] += 1.0f;
            }
            while (p[0] >= 1.0f) {
                p[0] -= 1.0f;
            }
            while (p[1] >= 1.0f) {
                p[1] -= 1.0f;
            }

            return p;
        }
    }
};

template <unsigned D>
class CanonicalSampler
{
public:
    explicit CanonicalSampler(const std::vector<NPoint<2>>&) { }

    template <typename RNG>
    NPoint<D> operator()(RNG& rng)
    {
        return generateRandomPoint<D>(rng);
    }
};

template <unsigned D, typename RNG, typename CandidateGenerator, typename DistanceFunction = WrappedDistanceSquared<D>>
NPoint<D> bestCandidatePoint(const DynamicGrid& grid, unsigned nCandidates, RNG& rng, CandidateGenerator candidateGenerator, DistanceFunction distance)
{
    typedef NPoint<D> Point;

    static std::vector<Point> otherPoints;

    Point bestCandidate;
    float maxDist2 = 0.0f;
    for (unsigned k = 0; k < nCandidates; ++k) {
        const Point candidate = candidateGenerator(rng);

        otherPoints.clear();
        auto f = [/*&otherPoints*/](const Point& p) { otherPoints.push_back(p); };
        grid.visitNeighbors(f, candidate);
        if (otherPoints.empty()) {
            grid.visitAll(f);
        }

        float sampleDist2 = std::numeric_limits<float>::max();
        for (const auto& otherPoint : otherPoints) {
            const float d2 = distance(candidate, otherPoint);
            sampleDist2 = std::min(d2, sampleDist2);
        }

        if (sampleDist2 > maxDist2) {
            bestCandidate = candidate;
            maxDist2 = sampleDist2;
        }
    }

    return bestCandidate;
}

#if DIMENSIONS == 2
template <typename RNG>
void populateWithPoisson(DynamicGrid& grid, RNG& rng)
{
    // TODO: have this function take an rng.
    const auto points = generatePoissonDisk(0, GRID_SIZE);

    typedef PerfectPowerArray<bool, DIMENSIONS> BoolGrid;
    BoolGrid boolGrid(GRID_SIZE, false);

    for (const auto& p : points) {
        grid.add(p);
        addSampleToGrid(p, true, boolGrid);
    }

    for (unsigned y = 0; y < kN1DStrata; ++y) {
        for (unsigned x = 0; x < kN1DStrata; ++x) {
            if (!boolGrid[x][y]) {
                grid.add(bestCandidatePoint<kDims>(grid, 500, rng, StratumSampler<kDims>(x, y)));
            }
        }
    }
}

#endif

template <typename SamplerFunction, typename DistanceFunction = WrappedDistanceSquared<kDims>>
std::vector<NPoint<kDims>> bestCandidate(uint32_t seed, unsigned numDesired)
{
    typedef NPoint<kDims> Point;

    // We can't just grab the points from the final grid, as tempting as it is,
    // because the grid has no notion of the order in which the points were
    // added.
    std::vector<Point> points;
    std::mt19937 rng(seed);

    SamplerFunction sampler(points);

    ProgressBar progress(numDesired);
    progress.draw();

    DynamicGrid grid;

    const Point initialPoint = sampler(rng);
    //const Point initialPoint({0.0f, 0.0f});
    grid.add(initialPoint);
    points.push_back(initialPoint);
    progress.update();
    progress.draw();

    for (unsigned i = 0; i < numDesired - 1; ++i) {
        const DistanceFunction distance(i + 1u);
        const unsigned nCandidates = 500 + std::pow(static_cast<float>(i), 0.85f);
        const Point p = bestCandidatePoint<kDims>(grid, nCandidates, rng, sampler, distance);
        grid.add(p);
        points.push_back(p);
        progress.update();
        progress.draw();
    }

    std::cout << '\n';
    return points;
}

int main(int argc, const char* argv[])
{
    uint32_t seed = 0;
    unsigned count = 1024;
    std::string filename("points.dat");

    enum {
        basic,
        disk,
        fibonacci,
        minkowski,
        chebyshev
    } genFunction = basic;

    try {
        ArgumentParser parser(argc, argv);
        if (parser.has("-seed")) {
            seed = parser.getModifier<uint32_t>("-seed", 0);
        }
        if (parser.has("-out")) {
            filename = parser.getModifier<std::string>("-out", 0);
        }
        if (parser.has("-count")) {
            count = parser.getModifier<unsigned>("-count", 0);
        }
        if (parser.has("-mode")) {
            const std::string mode = parser.getModifier<std::string>("-mode", 0);
            if (mode == "basic") {
                genFunction = basic;
            } else if (mode == "disk") {
                genFunction = disk;
            } else if (mode == "fibonacci") {
                genFunction = fibonacci;
            } else if (mode == "minkowski") {
                genFunction = minkowski;
            } else if (mode == "chebyshev") {
                genFunction = chebyshev;
            } else {
                throw ArgumentException("Unknown option to '-mode': " + mode);
            }
        }
    } catch (std::exception& e) {
        std::cerr << e.what() << '\n';
        return EXIT_FAILURE;
    }

    std::vector<NPoint<kDims>> points;
    switch (genFunction) {
        case basic:
            points = bestCandidate<CanonicalSampler<kDims>>(seed, count);
            break;
        case disk:
            points = bestCandidate<CanonicalSampler<kDims>, DiskWrappedDistanceSquared>(seed, count);
            break;
        case fibonacci:
            points = bestCandidate<FibonacciSampler>(seed, count);
            break;
        case minkowski:
            points = bestCandidate<CanonicalSampler<kDims>, MinkowskiWrappedDistanceSquared>(seed, count);
            break;
        case chebyshev:
            points = bestCandidate<CanonicalSampler<kDims>, ChebyshevWrappedDistanceSquared>(seed, count);
            break;
    }

    std::ofstream outs(filename);
    writePoints(points, outs);
}
