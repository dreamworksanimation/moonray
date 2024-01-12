// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#include "TestActivePixelMask.h"
#include <moonray/rendering/rndr/adaptive/ActivePixelMask.h>

#include <iostream>
#include <utility>

namespace moonray {
namespace rndr {
namespace unittest {

namespace {

// clang-format off
constexpr std::pair<unsigned, unsigned> accessMatrix[64] = {
    { 0,  0}, { 0,  4}, { 2,  2}, { 4,  4}, { 6,  6}, { 2,  6}, { 4,  0}, { 6,  2},
    { 1,  1}, { 1,  5}, { 3,  3}, { 5,  5}, { 7,  7}, { 3,  7}, { 5,  1}, { 7,  3},
    { 1,  0}, { 1,  4}, { 3,  2}, { 5,  4}, { 7,  6}, { 3,  6}, { 5,  0}, { 7,  2},
    { 0,  2}, { 0,  6}, { 2,  0}, { 4,  6}, { 6,  4}, { 2,  4}, { 4,  2}, { 6,  0},
    { 1,  3}, { 1,  7}, { 3,  1}, { 5,  7}, { 7,  5}, { 3,  5}, { 5,  3}, { 7,  1},
    { 0,  3}, { 0,  7}, { 2,  1}, { 4,  7}, { 6,  5}, { 2,  5}, { 4,  3}, { 6,  1},
    { 0,  1}, { 0,  5}, { 2,  3}, { 4,  5}, { 6,  7}, { 2,  7}, { 4,  1}, { 6,  3},
    { 1,  2}, { 1,  6}, { 3,  0}, { 5,  6}, { 7,  4}, { 3,  4}, { 5,  2}, { 7,  0}
};

// clang-format on
// These values are inclusive
void checkFill(const ActivePixelMask& mask, unsigned x0, unsigned y0, unsigned x1, unsigned y1)
{
    if (x0 > x1) {
        std::swap(x0, x1);
    }
    if (y0 > y1) {
        std::swap(y0, y1);
    }

    for (unsigned y = 0; y < ActivePixelMask::height; ++y) {
        for (unsigned x = 0; x < ActivePixelMask::width; ++x) {
            if (x >= x0 && x <= x1 && y >= y0 && y <= y1) {
                CPPUNIT_ASSERT(mask(x, y));
            } else {
                CPPUNIT_ASSERT(!mask(x, y));
            }
        }
    }
}
} // anonymous namespace

void
TestActivePixelMask::testAccess()
{
    ActivePixelMask a;

    CPPUNIT_ASSERT(a.empty());
    CPPUNIT_ASSERT(!a);

    // Randomly access the array so that we don't accidentally encounter some strange neighboring access bug.
    constexpr auto size = ActivePixelMask::width * ActivePixelMask::height;
    for (unsigned i = 0; i < size; ++i) {
        unsigned x;
        unsigned y;
        for (unsigned j = i; j < size; ++j) {
            std::tie(x, y) = accessMatrix[j];
            CPPUNIT_ASSERT(!a(x, y));
        }
        for (unsigned j = 0; j < i; ++j) {
            std::tie(x, y) = accessMatrix[j];
            CPPUNIT_ASSERT(a(x, y));
        }
        std::tie(x, y) = accessMatrix[i];
        a.set(x, y);
    }

    CPPUNIT_ASSERT(a.full());
}

void
TestActivePixelMask::testFill()
{

    // Fill in any gaps in the axis-aligned bounding-box of the selection set.
    // +--------+
    // |.x......|
    // |........|
    // |.....x..|
    // +--------+
    //
    // becomes
    //
    // +--------+
    // |.xxxxx..|
    // |.xxxxx..|
    // |.xxxxx..|
    // +--------+

    constexpr int min_x = 2;
    constexpr int max_x = 5;
    constexpr int min_y = 4;
    constexpr int max_y = 6;
    ActivePixelMask a;
    a.set(min_x, min_y);
    a.set(max_x, max_y);

    CPPUNIT_ASSERT(a.get(min_x, min_y));
    CPPUNIT_ASSERT(a.get(max_x, max_y));
    a.fillGaps();
    for (unsigned y = 0; y < ActivePixelMask::height; ++y) {
        for (unsigned x = 0; x < ActivePixelMask::width; ++x) {
            const bool set = x >= min_x && x <= max_x && y >= min_y && y <= max_y;
            CPPUNIT_ASSERT(a(x, y) == set);
        }
    }

    ActivePixelMask d;
    d.fillGaps();
    CPPUNIT_ASSERT(d.empty());

    constexpr auto size = ActivePixelMask::width * ActivePixelMask::height;
    for (unsigned i = 0; i < size; ++i) {
        for (unsigned j = i; j < size; ++j) {
            unsigned x0;
            unsigned y0;
            unsigned x1;
            unsigned y1;

            // Help! I'm stuck in C++14! Oh, C++ structured binding, how I wish I could use you!
            std::tie(x0, y0) = accessMatrix[i];
            std::tie(x1, y1) = accessMatrix[j];

            d.clear();

            d.set(x0, y0);
            d.set(x1, y1);
            d.fillGaps();
            checkFill(d, x0, y0, x1, y1);
        }
    }
}

void
TestActivePixelMask::testDilate()
{
    ActivePixelMask a;
    a.set(3, 2);
    a.set(4, 6);
    a.set(0, 7);

    ActivePixelMask b;
    b.set(4, 6);
    b.set(5, 5);

    ActivePixelMask c = a | b;
    CPPUNIT_ASSERT(c(3, 2));
    CPPUNIT_ASSERT(c(4, 6));
    CPPUNIT_ASSERT(c(0, 7));
    CPPUNIT_ASSERT(c(5, 5));

    c.dilate();
    auto check_neighbors = [](const ActivePixelMask& apm, int x, int y) {
        // We dilate orthogonal neighbors, not diagonals, so these are not nested loops, but we do check our center
        // element twice.
        for (int yd = -1; yd <= 1; ++yd) {
            const int y_access = y + yd;
            if (y_access < 0 || y_access >= static_cast<int>(ActivePixelMask::height)) {
                continue;
            }
            CPPUNIT_ASSERT(apm(x, y_access));
        }
        for (int xd = -1; xd <= 1; ++xd) {
            const int x_access = x + xd;
            if (x_access < 0 || x_access >= static_cast<int>(ActivePixelMask::width)) {
                continue;
            }
            CPPUNIT_ASSERT(apm(x_access, y));
        }
    };

    check_neighbors(c, 3, 2);
    check_neighbors(c, 4, 6);
    check_neighbors(c, 0, 7);
    check_neighbors(c, 5, 5);
}

} // namespace unittest
} // namespace rndr
} // namespace moonray

