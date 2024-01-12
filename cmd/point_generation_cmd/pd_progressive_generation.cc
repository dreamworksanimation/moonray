// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#include "ArgumentParser.h"
#include "NPoint.h"
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

typedef std::mt19937 RNG;

const unsigned kNoSample = std::numeric_limits<unsigned>::max();

template <typename T>
inline void addSampleToGrid(const NPoint<1>& p, T val, PerfectPowerArray<T, 1>& grid)
{
    const unsigned u0 = gridIdx(p[0], grid.size());
    grid[u0] = val;
}

template <typename T>
inline void addSampleToGrid(const NPoint<2>& p, T val, PerfectPowerArray<T, 2>& grid)
{
    const unsigned u0 = gridIdx(p[0], grid.size());
    const unsigned u1 = gridIdx(p[1], grid.size());
    grid[u0][u1] = val;
}

constexpr float kRelativeRadius = 0.75f;
static_assert(kRelativeRadius >= 0.65f && kRelativeRadius <= 0.85f,
    "The relative radius should be large, but not too large, as we "
    "want to avoid a regular configuration.");

const unsigned kDims = 2u;
std::vector<NPoint<kDims>> run(uint32_t seed, unsigned numDesired)
{
    typedef PerfectPowerArray<std::size_t, kDims> Grid;
    typedef NPoint<kDims> Point;

    RNG rng(seed);

    std::vector<Point> points;

    ProgressBar progress(numDesired);
    progress.draw();

    std::size_t lastGridSize = std::numeric_limits<std::size_t>::max();
    std::unique_ptr<Grid> grid;

    unsigned pointsAdded = 0;
    unsigned poissonPointsAdded = 0;
    for (unsigned i = 0; i < numDesired; ++i) {
        const unsigned nPoints = i + 1u;

        // We're going to be placing Q.size() + 1 points. The radius
        // max is the maximum radius of placing N points.
        const float radius = 2.0f * kRelativeRadius * radiusMax<kDims>(nPoints);

        // We want the cell size to be r/sqrt(n), where r is the radius
        // and n is the dimensions. This allows, at most, one point per
        // cell.
        const float cellWidth = radius/std::sqrt(static_cast<float>(kDims));

        // Taking the ceiling will reduce the radius, still ensuring
        // that we can only get one point per cell.
        const unsigned numCells = static_cast<unsigned>(std::ceil(1.0f / cellWidth));

        // We don't need to create a new grid each time. Sometimes, the
        // grid doesn't change size. If we don't create a new grid, we
        // just have to add the last point to the already existing
        // grid.
        if (numCells != lastGridSize) {
            grid.reset(new Grid(numCells, kNoSample));
            for (std::size_t q = 0; q < points.size(); ++q) {
                addSampleToGrid(points[q], q, *grid);
            }
            lastGridSize = numCells;
        }

        Point bestCandidate;
        float farthestD2 = 0.0f;
        const unsigned ncandidates = std::max(5'000'000u * kDims, 50'000u);
        for (unsigned k = 0; k < ncandidates; ++k) {
            const Point candidate = generateRandomPoint<kDims>(rng);

            // Check to see if candidate within 'radius' of other
            // points. We check ourselves and the neighbors on either
            // side.
            constexpr unsigned nneighbors = powu(3, kDims);
            std::array<unsigned, nneighbors> otherPointIndices;
            unsigned arrayIdx = 0;
            auto f = [&otherPointIndices, &arrayIdx](unsigned idx) { otherPointIndices[arrayIdx++] = idx; };
            const auto gridIdx = toGridLocation(candidate, grid->size());
            grid->visitNeighbors(f, gridIdx.data());

            float nearestD2 = std::numeric_limits<float>::max();
            for (unsigned idx = 0; idx < nneighbors; ++idx) {
                const auto otherPointIdx = otherPointIndices[idx];
                if (otherPointIdx != kNoSample) {
                    const Point& otherPoint = points[otherPointIdx];
                    const float d2 = wrappedDistanceSquared(candidate, otherPoint);
                    nearestD2 = std::min(d2, nearestD2);
                }
            }

            if (nearestD2 >= radius*radius) {
                // Is Poisson. We can stop looking.
                bestCandidate = candidate;
                ++poissonPointsAdded;
                break;
            }

            if (nearestD2 > farthestD2) {
                farthestD2 = nearestD2;
                bestCandidate = candidate;
            }
        }
        ++pointsAdded;
        points.push_back(bestCandidate);
        addSampleToGrid(bestCandidate, points.size() - 1u, *grid);

        progress.update();
        progress.draw();
    }

    std::cout << '\n';
    std::cout << ((poissonPointsAdded * 100) / pointsAdded) << "% Poisson distributed\n";

    return points;
}

int main(int argc, const char* argv[])
{
    uint32_t seed = 0;
    unsigned count = 1024;
    std::string filename("points.dat");

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
    } catch (std::exception& e) {
        std::cerr << e.what() << '\n';
        return EXIT_FAILURE;
    }

    const auto points = run(seed, count);

    std::ofstream outs(filename);
    writePoints(points, outs);
}
