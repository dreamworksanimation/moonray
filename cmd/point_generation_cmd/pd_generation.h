// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Array.h"
#include "NPoint.h"
#include "ProgressBar.h"

#include <algorithm>
#include <array>
#include <iostream>
#include <limits>
#include <random>
#include <vector>
#include <cmath>
#include <cstddef>

#ifndef DIMENSIONS
#error DIMENSIONS must be defined before including this file
#endif

#define REPEAT5(n) n,n,n,n,n
#define REPEAT4(n) n,n,n,n
#define REPEAT3(n) n,n,n
#define REPEAT2(n) n,n
#define REPEAT1(n) n
#define XREPEAT(n,d) REPEAT##d(n)
#define REPEAT(n,d) XREPEAT(n,d)

template <typename Generator, typename T>
const T& uniformSelection(const std::vector<T>& v, Generator& g)
{
    typedef typename std::vector<T>::size_type size_type;
    typedef typename std::uniform_int_distribution<size_type> dist_type;
    typedef typename dist_type::param_type param_type;

    static std::uniform_int_distribution<size_type> dist;
    return v[dist(g, param_type(0, v.size() - 1u))];
}

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

template <typename T>
inline void addSampleToGrid(const NPoint<3>& p, T val, PerfectPowerArray<T, 3>& grid)
{
    const unsigned u0 = gridIdx(p[0], grid.size());
    const unsigned u1 = gridIdx(p[1], grid.size());
    const unsigned u2 = gridIdx(p[2], grid.size());
    grid[u0][u1][u2] = val;
}

template <typename T>
inline void addSampleToGrid(const NPoint<5>& p, T val, PerfectPowerArray<T, 5>& grid)
{
    const unsigned u0 = gridIdx(p[0], grid.size());
    const unsigned u1 = gridIdx(p[1], grid.size());
    const unsigned u2 = gridIdx(p[2], grid.size());
    const unsigned u3 = gridIdx(p[3], grid.size());
    const unsigned u4 = gridIdx(p[4], grid.size());
    grid[u0][u1][u2][u3][u4] = val;
}

inline std::vector<NPoint<DIMENSIONS>> generatePoissonDisk(uint32_t seed, uint32_t gridSize)
{
    const unsigned kCandidatePoints = 30u * DIMENSIONS;
    const unsigned kNoSample = std::numeric_limits<unsigned>::max();

    typedef NPoint<DIMENSIONS> Point;

    // We want the cell size to be r/sqrt(n), where r is the radius and n is
    // the dimensions. This allows, at most, one point per cell. We're
    // specifying the number of cells, and not the radius, so we have to figure
    // out r from this.
    const float cellWidth = 1.0f / static_cast<float>(gridSize);
    const float r = cellWidth * std::sqrt(static_cast<float>(DIMENSIONS)); 

    std::mt19937 rng(seed);

    typedef PerfectPowerArray<unsigned, DIMENSIONS> Grid;

    Grid grid(gridSize, kNoSample); 
    std::vector<unsigned> activeList;
    std::vector<Point> points;

    ProgressBar progress(powu(gridSize, DIMENSIONS)); 

    // Choose initial sample uniformly.
    const Point initialPoint = generateRandomPoint<DIMENSIONS>(rng);
    activeList.push_back(0);
    points.push_back(initialPoint);

    progress.update(); // For initial point.

    // Insert index 0 into grid
    addSampleToGrid(initialPoint, 0u, grid);

    while (!activeList.empty()) {
        // Choose uniformly from active list in [0, activeList.size())
        const unsigned randIdx = uniformSelection(activeList, rng);
        const Point& basisPoint = points[randIdx];

        bool foundNewPoint = false;

        // Generate candidate point
        for (unsigned i = 0; i < kCandidatePoints && !foundNewPoint; ++i) {
            const Point candidate = generateRandomPoint<DIMENSIONS>(basisPoint, r, rng);

            // Check to see if candidate within 'r' of other points.
            // We check ourselves and the neighbors on either side.
            // 1D: 3 cells.
            // 2D: 9 cells.
            // 3D: 27 cells.
            // etc.
            constexpr unsigned nneighbors = powu(3, DIMENSIONS);
            std::array<unsigned, nneighbors> otherPointIndices;
            unsigned arrayIdx = 0;
            auto f = [&otherPointIndices, &arrayIdx](unsigned idx) { otherPointIndices[arrayIdx++] = idx; };
            const auto gridIdx = toGridLocation(candidate, gridSize);
            grid.visitNeighbors(f, gridIdx.data());

            bool valid = true;
            for (unsigned idx = 0; idx < nneighbors; ++idx) {
                const auto otherPointIdx = otherPointIndices[idx];
                if (otherPointIdx != kNoSample) {
                    const Point& otherPoint = points[otherPointIdx];
                    if (wrappedDistanceSquared(candidate, otherPoint) < r*r) {
                        valid = false;
                        break;
                    }
                }
            }

            if (valid) {
                points.push_back(candidate);
                const unsigned newindex = points.size() - 1u;
                activeList.push_back(newindex);
                addSampleToGrid(candidate, newindex, grid);
                foundNewPoint = true;
                progress.update();
                progress.draw();
            }
        }

        if (!foundNewPoint) {
            // Remove i from activeList.
            const auto it = std::find(activeList.begin(), activeList.end(), randIdx);
            activeList.erase(it);
        }
    }
    std::cout << '\n';

    return points;
}

