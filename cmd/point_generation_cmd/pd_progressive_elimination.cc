// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#include "ArgumentParser.h"
#include "NPoint.h"
#include "PerfectPowerArray.h"
#include "ProgressBar.h"
#include "r_sequence.h"
#include "util.h"
#include "StaticHyperGrid.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <future>
#include <iostream>
#include <limits>
#include <memory>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#define PRINT(v) std::cout << #v << ": " << (v) << '\n'

const unsigned kDims = 2u;
typedef NPoint<kDims> Point;

struct PointData
{
    int index;
    float weight;
    Point point;
    auto operator[](unsigned idx) const { return point[idx]; }
};

using Grid = StaticHyperGrid<PointData*, kDims>;

bool operator==(const PointData& a, const PointData& b)
{
    return a.point == b.point;
}

template <typename T>
T sqr(const T& t)
{
    return t*t;
}

float rminCalc(float rmax, float ncandidates, float ndesired)
{
    constexpr float beta = 0.65f;
    constexpr float gamma = 1.5f;
    const float N = ndesired;
    const float M = ncandidates;
    return rmax * (1.0f - std::pow(N/M, gamma)) * beta;
}

class WeightFunction
{
public:
    WeightFunction(float rmin, float rmax)
    : mRmin(rmin)
    , mRmax(rmax)
    {
    }

    float operator()(const Point& a, const Point& b) const
    {
        constexpr float alpha = 8.0f;
        const float d2 = wrappedDistanceSquared(a, b);
        const float dhat = (d2 > sqr(2.0f*mRmin)) ? std::min(std::sqrt(d2), 2.0f*mRmax) : 2.0f*mRmin;
        return std::pow(1.0f - dhat/(2.0f*mRmax), alpha);
    }

private:
    float mRmin;
    float mRmax;
};

template <typename RandomIt>
void max_heapify(RandomIt first, RandomIt last, RandomIt element)
{
    using std::swap; // Allow ADL
    using difference_type = typename std::iterator_traits<RandomIt>::difference_type;
    if (first == last) {
        return;
    }

    const difference_type length = std::distance(first, last);
    const difference_type i = std::distance(first, element);
    const difference_type left  = 2*i + 1;
    const difference_type right = 2*i + 2;

    difference_type largest = i;

    if (left < length && first[largest]->weight < first[left]->weight) {
        largest = left;
    }
    if (right < length && first[largest]->weight < first[right]->weight) {
        largest = right;
    }

    if (largest != i) {
        swap(first[i], first[largest]);
        swap(first[i]->index, first[largest]->index);
        // Leave index
        max_heapify(first, last, first + largest);
    }
}

template <typename RandomIt>
void pop_heap(RandomIt first, RandomIt last)
{
    using std::swap; // Allow ADL
    using difference_type = typename std::iterator_traits<RandomIt>::difference_type;
    if (first == last) {
        return;
    }

    RandomIt lastNode = std::prev(last);
    swap(*first, *lastNode);
    swap((*first)->index, (*lastNode)->index);
    // Leave index

    max_heapify(first, lastNode, first);
}

template <typename RandomIt>
void push_heap(RandomIt first, RandomIt last)
{
    using std::swap; // Allow ADL
    using difference_type = typename std::iterator_traits<RandomIt>::difference_type;
    if (first == last) {
        return;
    }

    difference_type i = std::distance(first, last) - 1;
    difference_type parent = (i-1)/2;

    while (first[parent]->weight < first[i]->weight) {
        swap(first[i], first[parent]);
        swap(first[i]->index, first[parent]->index);
        // Leave index
        i = parent;
        parent = (i-1)/2;
    }
}

template <typename RandomIt>
void make_heap(RandomIt first, RandomIt last)
{
    using difference_type = typename std::iterator_traits<RandomIt>::difference_type;
    const difference_type length = std::distance(first, last);

    for (difference_type i = length/2; i >= 0; --i) {
        max_heapify(first, last, first + i);
    }
}

template <typename RandomIt>
bool is_heap(RandomIt first, RandomIt last)
{
    using difference_type = typename std::iterator_traits<RandomIt>::difference_type;

    if (first == last) {
        return true;
    }

    const difference_type length = std::distance(first, last);
    for (difference_type i = 1; i < length; ++i) {
        const difference_type parent = (i-1)/2;
        if (first[parent]->weight < first[i]->weight) {
            return false;
        }
    }
    return true;
}

template <typename RandomIt>
bool consistent_indices(RandomIt first, RandomIt last)
{
    using difference_type = typename std::iterator_traits<RandomIt>::difference_type;

    const difference_type length = std::distance(first, last);
    for (difference_type i = 0; i < length; ++i) {
        if (first[i]->index != i) {
            return false;
        }
    }
    return true;
}

// Generate n*desired samples
// Mix of r-sequences, random, and uniform?
//
// Create a grid such that each cell is at least as wide as our target 2rmax.
// Need a grid that can hold any number in cell
//
// For each point, calculate weight with neighboring points
//
// Pull top from heap.
// Lookup neighbors.
// For each neighbor n:
//     Decrement weight of n
//     Re-heap n

StaticHyperGrid<PointData*, kDims> createGrid(float rmax)
{
    // Create a grid such that each cell is at least as wide as our target 2rmax.
    const float cellWidth = 2.0f*rmax;
    const int ncells = static_cast<int>(std::floor(1.0f/cellWidth));
    return StaticHyperGrid<PointData*, kDims>(std::max(1, ncells));
}

void doElimination(unsigned nremove,
                   std::vector<PointData*>& heap,
                   Grid& grid,
                   const float rmax,
                   const WeightFunction& weightFunction,
                   std::vector<Point>* progressivePoints = nullptr)
{
    ProgressBar progress(nremove);
    progress.draw();
    for (unsigned i = 0; i < nremove && !heap.empty(); ++i) {
        // Pull from heap, p.
        assert(::is_heap(heap.cbegin(), heap.cend()));
        assert(::consistent_indices(heap.cbegin(), heap.cend()));
        ::pop_heap(heap.begin(), heap.end());
        PointData* const toRemove = heap.back();
        heap.pop_back();

        if (progressivePoints) {
            progressivePoints->push_back(toRemove->point);
        }

        // Lookup p in grid.
        grid.visitNeighbors([&toRemove, &heap, weightFunction, rmax](const PointData* other) {
            if (toRemove != other) {
                const float d2 = wrappedDistanceSquared(toRemove->point, other->point);
                if (d2 <= sqr(2.0f*rmax)) {
                    // For each neighbor n of p, remove weight from n
                    heap[other->index]->weight -= weightFunction(toRemove->point, other->point);
                    // Reheap n
                    ::max_heapify(heap.begin(), heap.end(), heap.begin() + other->index);
                }
            }
        }, toRemove);
        grid.remove(toRemove);
        if ((i + 1) % 16384 == 0) {
            progress.update(16384);
            progress.draw();
        }
    }
    progress.draw();
}

std::vector<PointData> generateCandidates(uint32_t seed, unsigned ncandidates)
{
    std::mt19937 rng(seed);

    std::vector<PointData> candidates;
    candidates.reserve(ncandidates);

    RSequenceGenerator<kDims> rseq;
    const float r = 1.0f/std::sqrt(static_cast<float>(ncandidates));
    const float rseqSeed = static_cast<float>(seed)/static_cast<float>(std::numeric_limits<decltype(seed)>::max());
    for (unsigned i = 0; i < ncandidates; ++i) {
        PointData pd;
        pd.index = i;
#if 1
        const float f0 = std::min(OneMinusEpsilon, static_cast<float>(drand48()));
        const float f1 = std::min(OneMinusEpsilon, static_cast<float>(drand48()));
        pd.point = Point({f0, f1});
#else
        pd.point = generateRandomPointInDisk(rseq(i, 0.5f), r, rng);
#endif
        candidates.push_back(pd);
    }

    return candidates;
}

std::vector<PointData> run(std::vector<PointData>& candidates,
                           unsigned ndesired,
                           std::vector<Point>* progressivePoints = nullptr)
{
    const float sqrt3 = std::sqrt(3.0f);
    const unsigned ncandidates = candidates.size();

    // TODO: this is 2D sphere packing.
    const float rmax = std::sqrt(1.0f/(2.0f * sqrt3 * ndesired));
    const float rmin = rminCalc(rmax, ncandidates, ndesired);

    Grid grid = createGrid(rmax);
    std::vector<PointData*> heap;
    heap.reserve(ncandidates);
    for (auto& candidate : candidates) {
        grid.add(std::addressof(candidate));
        heap.push_back(std::addressof(candidate));
    }

    WeightFunction weightFunction(rmin, rmax);

    std::cout << "\nComputing weights\n";
    ProgressBar progressWeight(ncandidates);
    progressWeight.draw();
    int count = 0;
    for (auto& candidate : candidates) {
        candidate.weight = 0.0f;
        grid.visitNeighbors([&candidate, weightFunction, rmax](PointData* other) {
            if (candidate.point != other->point) {
                const float d2 = wrappedDistanceSquared(candidate.point, other->point);
                if (d2 <= sqr(2.0f*rmax)) {
                    candidate.weight += weightFunction(candidate.point, other->point);
                }
            }
        }, std::addressof(candidate));

        ++count;
        if (count % 16384 == 0) {
            progressWeight.update(16384);
            progressWeight.draw();
        }
    }
    progressWeight.draw();

    std::cout << "\nCreating heap\n";
    ::make_heap(heap.begin(), heap.end());
    assert(::is_heap(heap.cbegin(), heap.cend()));
    assert(::consistent_indices(heap.cbegin(), heap.cend()));

    if (progressivePoints) {
        progressivePoints->reserve(ndesired);
    }
    doElimination(ncandidates - ndesired, heap, grid, rmax, weightFunction, progressivePoints);

    std::vector<PointData> ret;
    ret.reserve(ndesired);
    assert(heap.size() == ndesired);
    for (auto p : heap) {
        ret.push_back(*p);
    }
    return ret;
}

std::vector<Point> run(uint32_t seed, unsigned ndesired)
{
    std::cout << "Populating candidates\n";
    const unsigned ncandidates = ndesired * 32;

    std::vector<PointData> candidates = generateCandidates(seed, ncandidates);

    std::cout << "Creating initial population\n";
    std::vector<PointData> population = run(candidates, ndesired);

    std::cout << "\nCreating progressive points\n";
    std::vector<Point> points;
    points.reserve(ndesired);

    unsigned ptarget = ndesired/2;
    for (unsigned i = 1; ptarget > 0; ++i) {
        ptarget = ndesired/(1<<i);
        population = run(population, ptarget, &points);
    }
    std::reverse(points.begin(), points.end());
    return points;
}

int main(int argc, char* argv[])
{
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

#if 0
#if 0
    std::vector<PointData> p(9);
    p[0].weight =  17;
    p[0].index  =   0;
    ::push_heap(p.begin(), p.begin() + 1);
    p[1].weight =   3;
    p[1].index  =   1;
    ::push_heap(p.begin(), p.begin() + 2);
    p[2].weight =  36;
    p[2].index  =   2;
    ::push_heap(p.begin(), p.begin() + 3);
    p[3].weight =  19;
    p[3].index  =   3;
    ::push_heap(p.begin(), p.begin() + 4);
    p[4].weight =   2;
    p[4].index  =   4;
    ::push_heap(p.begin(), p.begin() + 5);
    p[5].weight =   7;
    p[5].index  =   5;
    ::push_heap(p.begin(), p.begin() + 6);
    p[6].weight = 100;
    p[6].index  =   6;
    ::push_heap(p.begin(), p.begin() + 7);
    p[7].weight =  25;
    p[7].index  =   7;
    ::push_heap(p.begin(), p.begin() + 8);
    p[8].weight =   1;
    p[8].index  =   8;
    ::push_heap(p.begin(), p.begin() + 9);

#elif 1
    std::vector<PointData> p(9);
    p[0].weight =  17;
    p[0].index  =   0;
    p[1].weight =   3;
    p[1].index  =   1;
    p[2].weight =  36;
    p[2].index  =   2;
    p[3].weight =  19;
    p[3].index  =   3;
    p[4].weight =   2;
    p[4].index  =   4;
    p[5].weight =   7;
    p[5].index  =   5;
    p[6].weight = 100;
    p[6].index  =   6;
    p[7].weight =  25;
    p[7].index  =   7;
    p[8].weight =   1;
    p[8].index  =   8;
    ::make_heap(p.begin(), p.end());
#endif

#if 0
    p[2].weight = 18;
    ::max_heapify(p.begin(), p.end(), p.begin() + 2);

    p[7].weight = 6;
    ::max_heapify(p.begin(), p.end(), p.begin() + 6);
#endif

    while (!p.empty()) {
        ::pop_heap(p.begin(), p.end());
        std::cout << p.back().weight << std::endl;
        p.pop_back();
    }
#endif
}
