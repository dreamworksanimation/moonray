// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#include "NPoint.h"
#include "DynamicHyperGrid.h"
#include "ProgressBar.h"
#include "util.h"
#include "ArgumentParser.h"

#include <fstream>
#include <iostream>

#ifndef prints
#define prints(x) std::cout << #x << ": " << (x) << std::endl
#endif

constexpr unsigned kDims = 3;
constexpr float kRadiusWeight = 0.166f;
static_assert(kRadiusWeight >= 0.0f && kRadiusWeight <= 1.0f, "We have to be consistent");

void testGrid1()
{
    std::ofstream outs("points.dat");
    if (!outs) {
        std::cerr << "Unable to open points file\n";
        return;
    }

    std::mt19937 rng;
    DynamicHyperGrid<NPoint<kDims>, kDims> grid;
    for (int i = 0; i < 10000000; ++i) {
        grid.add(generateRandomPoint<kDims>(rng));
    }

    std::vector<NPoint<kDims>> points;
    grid.visitProjectedNeighbors([&points](const NPoint<kDims>& p) {
        points.push_back(p);
    }, NPoint<kDims>(0.75f));

    for (const auto& p : points) {
        outs << p << '\n';
    }

    std::sort(points.begin(), points.end());
    auto unique_end = std::unique(points.begin(), points.end());
    if (unique_end == points.end()) {
        std::cout << "No dups\n";
    } else {
        std::cout << "Dups\n";
    }

    std::cout.precision(8);
    for (auto it = points.begin(); it != unique_end; ++it) {
        std::cout << std::fixed << *it << '\n';
    }
    std::cout << "------------------\n";
    for (auto it = unique_end; it != points.end(); ++it) {
        std::cout << std::fixed << *it << '\n';
    }
}

void testGrid2()
{
    std::mt19937 rng;
    DynamicHyperGrid<NPoint<kDims>, kDims> grid;
    for (int i = 0; i < 1000000; ++i) {
        grid.add(generateRandomPoint<kDims>(rng));
    }

    std::vector<NPoint<kDims>> points;
    grid.visitProjectedNeighbors([&points, counter = 0](const NPoint<kDims>& p) mutable {
        points.push_back(p);
        return counter++ >= 30;
    }, NPoint<kDims>(0.75f));

    std::cout << points.size() << '\n';
    //for (const auto& p : points) {
        //std::cout << p << '\n';
    //}
}

template <unsigned D>
unsigned effectiveDimension(const NPoint<D>& point)
{
    unsigned effDim = 0;
    for (unsigned i = 0; i < D; ++i) {
        if (point[i] > 0.0f) {
            ++effDim;
        }
    }
    return effDim;
}

std::vector<NPoint<kDims>> run(std::uint32_t seed, unsigned numDesired)
{
    using Point = NPoint<kDims>;
    using Grid = DynamicHyperGrid<Point, kDims>;
    using RNG = std::mt19937;

    RNG rng(seed);

    std::vector<Point> points;

    ProgressBar progress(numDesired);
    progress.draw();

    Grid grid;
    const auto projectionVectors = getProjectionVectors<kDims>();

    const Point initialPoint = generateRandomPoint<kDims>(rng);
    grid.add(initialPoint);
    points.push_back(initialPoint);
    progress.update();

    // We may get better results if we best-candidate our Poisson points, but it's probably very slow.
    unsigned pointsAdded = 1;
    float radiusWeight = kRadiusWeight;
    for (pointsAdded = 1; pointsAdded < numDesired; ) {
        const float radiusFullDimension = radiusMax<kDims>(pointsAdded+0);

        Point toAdd;
        constexpr unsigned ncandidates = std::max(5'000'000u * kDims, 50'000u);
        bool addPoint = false;
        for (unsigned k = 0; k < ncandidates; ++k) {
            const Point candidate = generateRandomPoint<kDims>(rng);

            bool poisson = true;
            for (const auto& projectionVector : projectionVectors) {
                const auto effDim = effectiveDimension(projectionVector);
                if (effDim == 0) {
                    continue;
                }
                const float radiusSubDimension = radiusMax(pointsAdded + 0, 1.0f, effDim);
                const float rj = radiusWeight * radiusSubDimension / radiusFullDimension;
                auto poissonCheck = [&candidate, &projectionVector, rj](const Point& p) {
                    const Point h0 = hadamard(projectionVector, candidate);
                    const Point h1 = hadamard(projectionVector, p);
                    if (wrappedDistanceSquared(h0, h1) < (rj * rj)) {
                        return SearchResult::TERMINATED_EARLY;
                    }
                    return SearchResult::SEARCH_COMPLETE;
                };

                if (grid.visitProjectedNeighbors(poissonCheck, candidate) == SearchResult::TERMINATED_EARLY) {
                    poisson = false;
                    break;
                }
            }

            if (poisson) {
                toAdd = candidate;
                addPoint = true;
                break;
            }
        }
        if (addPoint) {
            grid.add(toAdd);
            points.push_back(toAdd);
            progress.update();
            progress.draw();
            ++pointsAdded;
        } else {
            radiusWeight -= 0.001f;
        }
    }

    return points;
}

int main(int argc, const char* argv[])
{
    //testGrid1();
    //testGrid2();

#if 1
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
#endif
}
