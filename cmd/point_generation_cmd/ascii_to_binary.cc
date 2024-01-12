// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#include "NPoint.h"

#include <cstdlib>
#include <iostream>
#include <istream>
#include <ostream>
#include <iterator>
#include <fstream>
#include <string>
#include <vector>

template <unsigned D>
std::vector<NPoint<D>> readAsciiPoints(std::istream& ins)
{
    return std::vector<NPoint<D>>(std::istream_iterator<NPoint<D>>(ins), std::istream_iterator<NPoint<D>>());
}

template <unsigned D>
void convert(std::istream& ins, std::ostream& outs, bool includeSize)
{
    const auto v = readAsciiPoints<D>(ins);

    if (includeSize) {
        const uint32_t size = v.size();
        outs.write(reinterpret_cast<const char*>(&size), sizeof(size));
    }
    for (const auto& p : v) {
        for (unsigned i = 0; i < D; ++i) {
            const float f = p[i];
            outs.write(reinterpret_cast<const char*>(&f), sizeof(f));
        }
    }
}

int main(int argc, char* argv[])
{
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] << " <include size (0|1)> <dimension (int)> <ascii file> <binary file>\n";
        return EXIT_FAILURE;
    }

    try {
        const bool includeSize = std::stoul(std::string(argv[1]));
        const unsigned long dimension = std::stoul(std::string(argv[2]));
        const std::string asciiFileName(argv[3]);
        const std::string binFileName(argv[4]);

        if (asciiFileName == binFileName) {
            std::cerr << "File names are the same. This is a really bad idea.\n";
            return EXIT_FAILURE;
        }

        std::ifstream ins(asciiFileName);
        if (!ins) {
            std::cerr << "Unable to open '" << asciiFileName << "' for reading.\n";
            return EXIT_FAILURE;
        }

        std::ofstream outs(binFileName, std::ios::binary);
        if (!outs) {
            std::cerr << "Unable to open '" << binFileName << "' for writing.\n";
            return EXIT_FAILURE;
        }

        ins.exceptions(std::ifstream::badbit);
        outs.exceptions(std::ofstream::badbit);

        switch (dimension) {
            case 1:
                convert<1>(ins, outs, includeSize);
                break;
            case 2:
                convert<2>(ins, outs, includeSize);
                break;
            case 3:
                convert<3>(ins, outs, includeSize);
                break;
            case 5:
                convert<5>(ins, outs, includeSize);
                break;
            default:
                std::cerr << "I don't know how to handle dimension " << dimension << '\n';
                return EXIT_FAILURE;
        }
    } catch (std::exception& e) {
        std::cerr << "Exception: " << e.what() << '\n';
        return EXIT_FAILURE;
    } catch (...) {
        std::cerr << "Unknown error.\n";
        return EXIT_FAILURE;
    }
}

