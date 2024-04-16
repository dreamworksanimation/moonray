// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#include "ArgumentParser.h"
#include "NPoint.h"

#include "tbb/blocked_range.h"
#include "tbb/parallel_reduce.h"

#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>

const unsigned kDimensions = 2;

typedef NPoint<kDimensions> Point;
typedef std::vector<Point> PointContainer;
typedef std::mt19937 reng;
typedef reng::result_type SeedType;

bool contained(const Point& toTest, const Point& bounds)
{
    for (unsigned i = 0; i < kDimensions; ++i) {
        if (toTest[i] >= bounds[i]) {
            return false;
        }
    }

    return true;
}

inline float volume(const Point& p)
{
    float r = 1.0f;
    for (unsigned i = 0; i < kDimensions; ++i) {
        r *= p[i];
    }
    return r;
}

// Measure the star discrepancy from one sample point (a single box) in the
// space [0, 1)^2.
inline float measureStarDiscrepancy(const Point& measurePoint, const PointContainer& points)
{
    const unsigned n = std::count_if(points.cbegin(), points.cend(),
                                    [&measurePoint](const Point& p) { return contained(p, measurePoint); });
    return std::abs(static_cast<float>(n)/static_cast<float>(points.size()) - volume(measurePoint));
}

class MeasureStarDiscrepancySplit
{
public:
    explicit MeasureStarDiscrepancySplit(const PointContainer& points) :
        mLUB(0.0f),
        mRand(sSeed++),
        mPoints(points)
    {
    }

    MeasureStarDiscrepancySplit(MeasureStarDiscrepancySplit& x, tbb::split) :
        mLUB(0.0f),
        mRand(sSeed++), // Call with new seed
        mPoints(x.mPoints)
    {
    }

    void operator()(const ::tbb::blocked_range<unsigned>& r)
    {
        float lub = mLUB;
        const auto end = r.end();
        for (auto i = r.begin(); i != end; ++i) {
            lub = std::max(lub, measureStarDiscrepancy(generateRandomPoint<kDimensions>(mRand), mPoints));
        }
        mLUB = lub;
    }

    void join(const MeasureStarDiscrepancySplit& y)
    {
        mLUB = std::max(mLUB, y.mLUB);
    }

    float getLUB() const
    {
        return mLUB;
    }

    static tbb::atomic<SeedType> sSeed;

private:
    float mLUB;
    reng mRand;
    const PointContainer& mPoints;
};

tbb::atomic<SeedType> MeasureStarDiscrepancySplit::sSeed;

// Estimate the star discrepancy for a set of points by repeatedly random
// sampling.
float measureStarDiscrepancy(const PointContainer& points)
{
    const unsigned numSamples = 100000u;
    MeasureStarDiscrepancySplit msd(points);
    tbb::parallel_reduce(tbb::blocked_range<unsigned>(0, numSamples), msd);
    return msd.getLUB();
}

template <typename Generator>
PointContainer generatePoints(Generator generator, std::size_t numPoints)
{
    PointContainer points;
    points.reserve(numPoints);

    for (std::size_t s = 0; s < numPoints; ++s) {
        points.push_back(generator(s));
        assert(points.back()[0] >= 0.0f && points.back()[0] < 1.0f);
        assert(points.back()[1] >= 0.0f && points.back()[1] < 1.0f);
    }

    return points;
}

PointContainer readPoints(std::istream& ins, unsigned count)
{
    PointContainer r;

    for (unsigned i = 0; i < count && ins; ++i) {
        Point p;
        ins >> p;
        r.push_back(p);
    }
    return r;
}

PointContainer readPoints(const std::string& filename, unsigned count)
{
    std::ifstream ins(filename);
    return readPoints(ins, count);
}

int main(int argc, const char* argv[])
{
    unsigned count = std::numeric_limits<unsigned>::max();
    std::string filename;
    try {
        ArgumentParser parser(argc, argv);
        if (!parser.has("-in")) {
            std::cerr << "Usage: " << argv[0] << " -in <filename>\n";
            return EXIT_FAILURE;
        }
        filename = parser.getModifier<std::string>("-in", 0);
        if (parser.has("-count")) {
            count = parser.getModifier<unsigned>("-count", 0);
        }
    } catch (std::exception& e) {
        std::cerr << e.what() << '\n';
        return EXIT_FAILURE;
    }

    const float d = measureStarDiscrepancy(readPoints(filename, count));
    std::cout << d << '\n';
}


