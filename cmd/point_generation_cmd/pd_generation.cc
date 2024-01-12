// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#include "ArgumentParser.h"
#include "Array.h"
#include "NPoint.h"
#include "ProgressBar.h"
#include "PerfectPowerArray.h"

#include <algorithm>
#include <array>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>
#include <cmath>
#include <cstddef>

//#define DIMENSIONS 1
//#define GRID_SIZE 1540

#define DIMENSIONS 2
//#define GRID_SIZE 55

#include "pd_generation.h"

int main(int argc, const char* argv[])
{
    uint32_t seed = 0;
    uint32_t gridSize = 100;
    std::string filename = "points.dat";
    bool quiet = false;

    try {
        ArgumentParser parser(argc, argv);
        if (parser.has("-seed")) {
            seed = parser.getModifier<uint32_t>("-seed", 0);
        }
        if (parser.has("-out")) {
            filename = parser.getModifier<std::string>("-out", 0);
        }
        if (parser.has("-grid_size")) {
            gridSize = parser.getModifier<uint32_t>("-grid_size", 0);
        }
        if (parser.has("-quiet")) {
            quiet = true;
        }
    } catch (std::exception& e) {
        std::cerr << e.what() << '\n';
        return EXIT_FAILURE;
    }

    if (!quiet) {
        std::cout << "Generating Poisson points in " << DIMENSIONS << " dimensions.\n";
        std::cout << "Storing in file " << filename << '\n';
    }
    const auto points = generatePoissonDisk(seed, gridSize);

    std::ofstream outs(filename);
    writePoints(points, outs);
}

