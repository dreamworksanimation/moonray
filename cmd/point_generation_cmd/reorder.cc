// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#include "ArgumentParser.h"
#include "DynamicHyperGrid.h"
#include "NPoint.h"
#include "PointContainer2D.h"
#include "ProgressBar.h"

#include <tbb/parallel_reduce.h>
#include <tbb/blocked_range.h>

#include <algorithm>
#include <cassert>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#define DIMENSIONS 2

#if DIMENSIONS == 1
const unsigned kDims = 1u;
#elif DIMENSIONS == 2
const unsigned kDims = 2u;
#elif DIMENSIONS == 3
const unsigned kDims = 3u;
#elif DIMENSIONS == 5
const unsigned kDims = 5u;
#else
#error Not supported (but easy to add)
#endif

typedef NPoint<kDims> Point;
typedef DynamicHyperGrid<Point, kDims> Grid;

template <unsigned D>
class CachedDistPoint : public NPoint<D>
{
public:
    explicit CachedDistPoint(std::initializer_list<float> args) :
        NPoint<D>(args),
        mDist2UpperBound(std::numeric_limits<float>::max())
    {
    }

    CachedDistPoint(const NPoint<D-1>& p, float f) :
        NPoint<D>(p, f),
        mDist2UpperBound(std::numeric_limits<float>::max())
    {
    }

    CachedDistPoint(const Point& p) :
        NPoint<D>(p),
        mDist2UpperBound(std::numeric_limits<float>::max())
    {
    }

    float mDist2UpperBound;
};

typedef std::vector<Point> PointContainer;

typedef std::default_random_engine RNG;

inline void addSampleToGrid(const Point& p, Grid& grid)
{
    grid.add(p);
}

template <typename DistanceFunction>
float findClosest(const Point& p, const Grid& grid, DistanceFunction distance)
{
    // Point to grid indices.
    PointContainer pointsToCheck;
    static constexpr auto maxNeighbors = Grid::maxNeighboringItems();
    pointsToCheck.reserve(maxNeighbors);
    auto f = [&pointsToCheck](const Point& c)
    {
        pointsToCheck.push_back(c);
    };

    // If we don't have a populated neighbor, we want to visit all locations,
    // to make sure we still get the closest point. Yay, O(N^2)!
    grid.visitNeighbors(f, p);
    if (pointsToCheck.empty()) {
        grid.visitAll(f);
    }

    float sampleDist2 = std::numeric_limits<float>::max();
    for (const auto& sample : pointsToCheck) {
        const float thisDist2 = distance(p, sample);
        if (thisDist2 < sampleDist2) {
            sampleDist2 = thisDist2;
        }
    }

    return sampleDist2;
}

template <unsigned D>
std::vector<NPoint<D>> readPoints(std::istream& ins)
{
    return std::vector<NPoint<D>>(std::istream_iterator<NPoint<D>>(ins), std::istream_iterator<NPoint<D>>());
}

template <unsigned D>
std::vector<NPoint<D>> readPoints(const std::string& filename)
{
    if (filename == "-") {
        return readPoints<D>(std::cin);
    } else {
        std::ifstream ins(filename);
        return readPoints<D>(ins);
    }
}

std::vector<float> readFloats(std::istream& ins)
{
    return std::vector<float>(std::istream_iterator<float>(ins), std::istream_iterator<float>());
}

std::vector<float> readFloats(const std::string& filename)
{
    std::ifstream ins(filename);
    return readFloats(ins);
}

// Find a (farish) random point.
template <typename Container, typename DistanceFunction>
typename Container::iterator stochasticFindNext(Container& list, const Grid& grid, RNG& rng, DistanceFunction distance)
{
    typedef typename Container::iterator iterator;
    typedef typename Container::size_type size_type;
    const size_type upperLimit = list.size() / 4u;

    // For each point in list (up to a limit):
    //
    // If it is farther than radius to each point in the grid, add to candidate
    // list.
    //
    // Randomly choose one element from candidate list.

    struct ValueType
    {
        ValueType(iterator it, float d) : iter(it), distance(d) {}
        typename Container::iterator iter;
        float distance;
    };

    std::vector<ValueType> candidates;
    candidates.reserve(upperLimit);

    // We want to make a min-heap, so we can readily access the smallest value
    // we've saved, so swap it out.
    auto comp = [](const ValueType& a, const ValueType& b) { return a.distance > b.distance; };

    for (auto it = list.begin(); it != list.end(); ++it) {
        const float c = findClosest(*it, grid, distance);
        if (candidates.empty() || candidates.size() < upperLimit) {
            candidates.emplace_back(it, c);
            if (candidates.size() == upperLimit) {
                std::make_heap(candidates.begin(), candidates.end(), comp);
            }
        } else if (c > candidates.front().distance) {
            std::pop_heap(candidates.begin(), candidates.end(), comp);
            candidates.back() = ValueType(it, c);
            std::push_heap(candidates.begin(), candidates.end(), comp);
        }
    }

    std::vector<float> distances;
    distances.reserve(candidates.size());
    std::transform(candidates.begin(), candidates.end(), std::back_inserter(distances),
            [](const ValueType& v) {
                return v.distance;
            });

    std::discrete_distribution<size_type> dist(distances.begin(), distances.end());
    return candidates.at(dist(rng)).iter;
}

#if 0
template <typename Container>
typename Container::iterator findFarthest(Container& list, const Grid& grid)
{
    float maxDist2 = 0.0f;
    typename Container::iterator ret;
    for (auto it = list.begin(); it != list.end(); ++it) {
        // Find closest point for p
        if (maxDist2 <= it->mDist2UpperBound) {
            const float c = findClosest(*it, grid);
            it->mDist2UpperBound = c;

            // If distance between closest and p is farthest yet, save p
            if (c > maxDist2) {
                maxDist2 = c;
                ret = it;
            }
        }
    }

    return ret;
}
#else

template <typename Container, typename DistanceFunction>
struct FindFarthest
{
    float mMaxDist2;
    const Grid& mGrid;
    DistanceFunction mDist;
    typename Container::iterator mEnd;
    typename Container::iterator mFar;

    explicit FindFarthest(const Grid& g, typename Container::iterator end, DistanceFunction dist) :
        mMaxDist2(0.0f),
        mGrid(g),
        mDist(dist),
        mEnd(end),
        mFar(end)
    {
    }

    FindFarthest(FindFarthest& s, tbb::split) :
        mMaxDist2(0.0f),
        mGrid(s.mGrid),
        // We don't allow dynamically changing distance functions. It ruins the cache.
        mDist(0),
        mEnd(s.mEnd),
        mFar(mEnd)
    {
    }

    void operator()(const tbb::blocked_range<typename Container::iterator>& r)
    {
        float maxDist2 = mMaxDist2;
        for (auto a = r.begin(); a != r.end(); ++a) {
            if (maxDist2 <= a->mDist2UpperBound) {
                const float c = findClosest(*a, mGrid, mDist);
                a->mDist2UpperBound = c;
                if (c > maxDist2) {
                    maxDist2 = c;
                    mFar = a;
                }
            }
        }
        mMaxDist2 = maxDist2;
    }

    void join(FindFarthest& rhs)
    {
        if (mFar == mEnd || mMaxDist2 < rhs.mMaxDist2) {
            mMaxDist2 = rhs.mMaxDist2;
            mFar = rhs.mFar;
        }
    }
};

template <typename Container, typename DistanceFunction>
inline typename Container::iterator findFarthest(Container& list, const Grid& grid, DistanceFunction distance)
{
    typedef typename Container::iterator iterator;
    FindFarthest<Container, DistanceFunction> furthest(grid, list.end(), distance);
    tbb::parallel_reduce(tbb::blocked_range<iterator>(list.begin(), list.end()),
                         furthest);
    return furthest.mFar;
}
#endif

// Although streams are supposed to be movable, GCC 4.8/4.9 does not implement
// this functionality.
std::unique_ptr<std::ofstream> createOutputStream(const std::string& filename)
{
    std::unique_ptr<std::ofstream> outs(new std::ofstream(filename));
    outs->exceptions(std::ios_base::failbit);
    outs->precision(9);
    outs->setf(std::ios_base::fixed | std::ios_base::showpoint);
    return outs;
}

// Precondition: initialPoint has been removed from the unprocessed list.
template <typename Container, typename DistanceFunction>
void reorder(const std::string& outFileName, Container unprocessed, const Point& initialPoint, RNG& reng, DistanceFunction distance, unsigned stochasticCount)
{
    auto outs = createOutputStream(outFileName);

    Grid grid;
    // Add initial point to grid.
    grid.add(initialPoint);
    (*outs) << initialPoint << '\n';

    ProgressBar progress(unprocessed.size());
    progress.draw();
    for (unsigned i = 0; !unprocessed.empty(); ++i) {
        const bool stochastic = i < stochasticCount && i % 2 == 0;
        const auto toAdd = (stochastic) ?
                   stochasticFindNext(unprocessed, grid, reng, distance) :
                   findFarthest(unprocessed, grid, distance);

        assert(toAdd != unprocessed.end());

        grid.add(*toAdd);
        (*outs) << *toAdd << '\n';

        unprocessed.erase(toAdd);
        progress.update();
        progress.draw();
    }
    std::cout << '\n';
}

template <typename DistanceFunction>
void reorderFile(const std::string& inFileName, const std::string& outFileName, unsigned stochasticCount)
{
    RNG reng;

    // Read in points, perhaps from stdin.
    PointContainer points = readPoints<kDims>(inFileName);

    // This is an odd way to make the first point of the re-ordered sequence
    // the first point of the original sequence, in case there is something
    // special about that point.
    std::reverse(points.begin(), points.end());
    const Point initialPoint = points.back();

    // Remove initial point from unprocessed list.
    points.pop_back();

    // We don't allow dynamically changing distance functions. It ruins the cache.
    DistanceFunction distance(0);
    reorder(outFileName,
            std::vector<CachedDistPoint<kDims>>(points.cbegin(), points.cend()),
            initialPoint,
            reng,
            distance,
            stochasticCount);
}

#if 0
void addDimension(const std::string& filename1, const std::string& filename2)
{
    RNG reng;

    // Read in points, perhaps from stdin.
    auto points = readPoints<kDims - 1u>(filename1);
    auto floats = readFloats(filename2);

    // Shuffle points so that stochasticFindNext is unbiased.
    std::shuffle(points.begin(), points.end(), reng);
    std::shuffle(floats.begin(), floats.end(), reng);
    Point initialPoint(points.back(), floats.back());

    // Remove initial point from unprocessed list.
    points.pop_back();
    floats.pop_back();

    /*
    reorder(PointManager2D<CachedDistPoint<kDims>>(points.cbegin(),
                                                   points.cend(),
                                                   floats.cbegin(),
                                                   floats.cend()),
                                                   initialPoint,
                                                   reng);;
                                                   */
}
#endif

int main(int argc, const char* argv[])
{
    unsigned stochasticCount = 0;
    std::string inName = "-";
    std::string outName = "reordered.dat";

    enum {
        basic,
        disk,
        chebyshev
    } distanceFunction = basic;

    try {
        ArgumentParser parser(argc, argv);
        if (parser.has("-in")) {
            inName = parser.getModifier<std::string>("-in", 0);
        }
        if (parser.has("-out")) {
            outName = parser.getModifier<std::string>("-out", 0);
        }
        if (parser.has("-stochastic")) {
            stochasticCount = parser.getModifier<unsigned>("-stochastic", 0);
        }
        if (parser.has("-mode")) {
            const std::string mode = parser.getModifier<std::string>("-mode", 0);
            if (mode == "basic") {
                distanceFunction = basic;
            } else if (mode == "disk") {
                distanceFunction = disk;
            } else if (mode == "chebyshev") {
                distanceFunction = chebyshev;
            } else {
                throw ArgumentException("Unknown option to '-mode': " + mode);
            }
        }
    } catch (std::exception& e) {
        std::cerr << e.what() << '\n';
        return EXIT_FAILURE;
    }

    switch (distanceFunction) {
        case basic:
            reorderFile<WrappedDistanceSquared<kDims>>(inName, outName, stochasticCount);
            break;
        case disk:
            reorderFile<DiskWrappedDistanceSquared>(inName, outName, stochasticCount);
        case chebyshev:
            reorderFile<ChebyshevWrappedDistanceSquared>(inName, outName, stochasticCount);
            break;
    }
}

