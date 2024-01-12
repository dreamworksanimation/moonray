// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "StaticVector.h"
#include "util.h"

#include <array>
#include <cassert>
#include <cmath>
#include <initializer_list>
#include <istream>
#include <limits>
#include <ostream>
#include <random>
#include <type_traits>
#include <vector>

constexpr float OneMinusEpsilon = 0x1.fffffep-1;

template <unsigned D>
class NPoint
{
public:
    static constexpr unsigned sDims = D;
    using InternalType = float; // internal type

    template <typename Iter>
    NPoint(Iter first, Iter /*last*/)
    {
        for (unsigned i = 0; i < D-1u; ++i) {
            mP[i] = *(first + i);
        }
        mF = *(first + (D-1u));
    }

    explicit NPoint(std::initializer_list<float> args) :
        NPoint(args.begin(), args.end())
    {
    }

    NPoint(const NPoint<D-1>& p, float f) :
        mP(p),
        mF(f)
    {
    }

    explicit NPoint(float f) :
        mP(f),
        mF(f)
    {
    }

    NPoint() = default;

    float* data()
    {
        static_assert(std::is_standard_layout<NPoint<D>>::value,
                "Our memory should pretend it's an array.");
        return reinterpret_cast<float*>(std::addressof(mP));
    }

    const float* data() const
    {
        static_assert(std::is_standard_layout<NPoint<D>>::value,
                "Our memory should pretend it's an array.");
        return reinterpret_cast<const float*>(std::addressof(mP));
    }

    float operator[](unsigned idx) const
    {
        assert(idx < D);
        return data()[idx];
    }

    float& operator[](unsigned idx)
    {
        assert(idx < D);
        return data()[idx];
    }

    const NPoint<D-1u>& strip() const
    {
        return mP;
    }

    float back() const
    {
        return mF;
    }

    NPoint& operator+=(const NPoint& other)
    {
        for (unsigned i = 0; i < D; ++i) {
            (*this)[i] += other[i];
        }
        return *this;
    }

private:
    NPoint<D-1u> mP;
    float mF;
};

template <>
class NPoint<1>
{
public:
    static constexpr unsigned sDims = 1;

    explicit NPoint(std::initializer_list<float> args) :
        mF(*(args.begin()))
    {
    }

    explicit NPoint(float f) :
        mF(f)
    {
    }

    NPoint() = default;

    float* data()
    {
        return std::addressof(mF);
    }

    const float* data() const
    {
        return std::addressof(mF);
    }

#ifdef NDEBUG
    float operator[](unsigned) const
#else
    float operator[](unsigned idx) const
#endif
    {
        assert(idx == 0);
        return mF;
    }

#ifdef NDEBUG
    float& operator[](unsigned)
#else
    float& operator[](unsigned idx)
#endif
    {
        assert(idx == 0);
        return mF;
    }

private:
    float mF;
};

template <unsigned D>
struct ProjectionVectorCalculator
{
    static auto get()
    {
        using Container = std::array<NPoint<D>, powu(2, D)>;
        using size_type = typename Container::size_type;
        Container ret;

        const auto last = ProjectionVectorCalculator<D-1>::get();
        for (size_type i = 0; i < last.size(); ++i) {
            ret[i*2+0] = NPoint<D>(last[i], 0.0f);
            ret[i*2+1] = NPoint<D>(last[i], 1.0f);
        }
        return ret;
    }
};

template <>
struct ProjectionVectorCalculator<1>
{
    static auto get()
    {
        using Container = std::array<NPoint<1>, 2>;
        const Container ret = { NPoint<1>(0.0f), NPoint<1>(1.0f) };
        return ret;
    }
};

// This will return an array of all 2^N combinations of 0 and 1 in each
// dimension. E.g., D = 3, you will get an array of 8 3D points:
// 0, 0, 0
// 0, 0, 1
// 0, 1, 0
// 0, 1, 1
// 1, 0, 0
// 1, 0, 1
// 1, 1, 0
// 1, 1, 1
template <unsigned D>
auto getProjectionVectors()
{
    return ProjectionVectorCalculator<D>::get();
}

template <unsigned D>
bool operator==(const NPoint<D>&a, const NPoint<D>&b)
{
    return std::equal(a.data(), a.data() + D, b.data());
}

template <unsigned D>
bool operator!=(const NPoint<D>&a, const NPoint<D>&b)
{
    return !(a == b);
}

// Component-wise multiplication
template <unsigned D>
NPoint<D> hadamard(const NPoint<D>&a, const NPoint<D>&b)
{
    NPoint<D> ret;
    for (unsigned i = 0; i < D; ++i) {
        ret[i] = a[i] * b[i];
    }
    return ret;
}

template <unsigned D, unsigned current>
struct PointLess
{
    static bool cmp(const NPoint<D>& a, const NPoint<D>& b) noexcept
    {
        if (a[current-1u] == b[current-1u]) {
            return PointLess<D, current - 1u>::cmp(a, b);
        } else {
            return a[current-1u] < b[current-1u];
        }
    }
};

template <unsigned D>
struct PointLess<D, 1>
{
    static bool cmp(const NPoint<D>& a, const NPoint<D>& b) noexcept
    {
        return a[0] < b[0];
    }
};

template <unsigned D>
inline bool operator<(const NPoint<D>& a, const NPoint<D>& b) noexcept
{
    return PointLess<D, D>::cmp(a, b);
}

template <unsigned D>
std::ostream& operator<<(std::ostream& outs, const NPoint<D>& p)
{
    outs << p[0];
    for (unsigned i = 1; i < D; ++i) {
        outs << ' ' << p[i];
    }
    return outs;
}

template <unsigned D>
std::istream& operator>>(std::istream& ins, NPoint<D>& p)
{
    for (unsigned i = 0; i < D; ++i) {
        ins >> p[i];
    }
    return ins;
}

template <unsigned D>
float dot(const NPoint<D>& a, const NPoint<D>& b)
{
    return std::inner_product(a.data(), a.data() + D, b.data(), 0.0f);
}

template <typename Generator>
inline float canonical(Generator& g)
{
    // This should be in [0, 1), but there is a bug in the standard
    // specification that allows for 1.
    static std::uniform_real_distribution<float> dist;
    return std::min(dist(g), OneMinusEpsilon);
}

template <unsigned D, typename RNGType>
inline NPoint<D> generateRandomPoint(RNGType& rng)
{
    NPoint<D> p;
    for (unsigned i = 0; i < D; ++i) {
        p[i] = canonical(rng);
    }
    return p;
}

// This generates a random point in N-dimensional space within distance [r, 2r]
// from "basis".
//
// Here, 'o' is the basis point, and the filled area is where we're generating
// points. 2D case represented, 'cause, stuff.
//
//                 xxxxxxx
//               xx.......xx
//              x...........x
//             x.............x
//            x...............x
//           x......xxxxx......x
//           x.....x     x.....x
//          x.....x       x.....x
//          x....x         x....x
//          x....x         x....x
//          x....x    o    x....x
//          x....x         x....x
//          x....x         x....x
//          x.....x       x.....x
//           x.....x     x.....x
//           x......xxxxx......x
//            x...............x
//             x.............x
//              x...........x
//               xx.......xx
//                 xxxxxxx
//
// This can be done more efficiently in 2D space, but it uses rejection
// sampling because it was easy (and fast enough) in ND space.
template <unsigned D, typename RNGType>
inline NPoint<D> generateRandomPoint(const NPoint<D>& basis, float r, RNGType& rng)
{
    std::uniform_real_distribution<float> dist(-2.0f*r, 2.0f*r);

    // Rejection sampling.
    // Loop until we get a valid sample.
    for ( ; ; ) {
        NPoint<D> p;
        for (unsigned i = 0; i < D; ++i) {
            const float rand = dist(rng);
            p[i] = basis[i] + rand;

            // These should only execute once at most (in theory), but there
            // are some numerical precision issues.
            while (p[i] < 0.0f) {
                p[i] += 1.0f;
            }
            while (p[i] >= 1.0f) {
                p[i] -= 1.0f;
            }

            assert(p[i] >= 0.0f);
            assert(p[i] <  1.0f);
        }
        const float rSquared = r*r;
        const float r2Squared = 2.0f*r * 2.0f*r;
        const float wd2 = wrappedDistanceSquared(p, basis);

        if (wd2 < r2Squared && wd2 > rSquared) {
            return p;
        }
    }
}

template <unsigned D, typename RNGType>
inline NPoint<D> generateRandomPointInDisk(const NPoint<D>& basis, float r, RNGType& rng)
{
    //std::uniform_real_distribution<float> dist(-r, +r);
    std::normal_distribution<float> dist(0.0f, 0.68f * r);

    // Rejection sampling.
    // Loop until we get a valid sample.
    for ( ; ; ) {
        NPoint<D> p;
        for (unsigned i = 0; i < D; ++i) {
            const float rand = dist(rng);
            p[i] = basis[i] + rand;

            // These should only execute once at most (in theory), but there
            // are some numerical precision issues.
            while (p[i] < 0.0f) {
                p[i] += 1.0f;
            }
            while (p[i] >= 1.0f) {
                p[i] -= 1.0f;
            }

            assert(p[i] >= 0.0f);
            assert(p[i] <  1.0f);
        }
        const float rSquared = r*r;
        const float wd2 = wrappedDistanceSquared(p, basis);

        if (wd2 < rSquared) {
            return p;
        }
    }
}

template <unsigned D>
inline void writePoints(const std::vector<NPoint<D>>& points, std::ostream& outs)
{
    IOStateSaver stateraii(outs);

    outs.exceptions(std::ios_base::failbit);
    outs.precision(9);
    outs.setf(std::ios_base::fixed | std::ios_base::showpoint);

    for (const auto& p : points) {
        outs << p << '\n';
    }
}

inline unsigned gridIdx(float v, unsigned gridSize)
{
    const unsigned r = static_cast<unsigned>(v * gridSize);
    assert(v < 1.0f);
    assert(v >= 0.0f);
    assert(r < gridSize);
    return r;
}

template <unsigned D>
std::array<std::size_t, D> toGridLocation(const NPoint<D>& p, unsigned gridSize)
{
    std::array<std::size_t, D> a;
    for (unsigned i = 0; i < D; ++i) {
        a[i] = gridIdx(p[i], gridSize);
    }
    return a;
}

inline NPoint<2> fromConcentricDisk(const NPoint<2>& p)
{
    float r;
    float phi;
    cartesianToPolar(p[0], p[1], r, phi);

    if (phi < -pi/4.0f) {
        // in range [-pi/4,7pi/4]
        phi += 2.0f * pi;
    }

    float a;
    float b;

    if (phi < pi/4.0f) {
        a = r;
        b = phi * a / (pi/4.0f);
    } else if (phi < 3.0f*pi/4.0f) {
        b = r;
        a = -(phi - pi/2.0f) * b / (pi/4.0f);
    } else if (phi < 5.0f*pi/4.0f) {
        a = -r;
        b = (phi - pi)*a / (pi/4.0f);
    } else {
        b = -r;
        a = -(phi - 3.0f*pi/2.0f) * b / (pi/4.0f);
    }

    const float x = (a + 1.0f) / 2.0f;
    const float y = (b + 1.0f) / 2.0f;

    return NPoint<2>({x, y});
}

inline NPoint<2> toConcentricDisk(const NPoint<2>& p)
{

    float a = 2.0f*p[0] - 1.0f;
    float b = 2.0f*p[1] - 1.0f;

    if (a == 0.0f && b == 0.0f) {
        return NPoint<2>({0.0f, 0.0f});
    }

    float phi;
    float r;

    if (std::abs(a) > std::abs(b)) {
        r = a;
        phi = (pi/4.0f)*(b/a);
    } else {
        r = b;
        phi = (pi/2.0f) - (pi/4.0f)*(a/b);
    }

    float x;
    float y;
    polarToCartesian(r, phi, x, y);
    return NPoint<2>({x, y});
}

template <typename F>
F wrapped1DDist(F a, F b)
{
    const auto d = std::abs(a - b);
    if (d < static_cast<F>(0.5)) {
        return d;
    } else {
        return static_cast<F>(1.0) - std::max(a, b) + std::min(a, b);
    }
}

template <typename P>
auto wrappedDistanceSquared(const P& a, const P& b)
{
    typename P::InternalType d2 = 0.0;
    for (unsigned i = 0; i < P::sDims; ++i) {
        const auto d = wrapped1DDist(a[i], b[i]);
        d2 += d*d;
    }
    return d2;
}

template <unsigned D>
float distanceSquared(const NPoint<D>& a, const NPoint<D>& b)
{
    float d2 = 0.0f;
    for (unsigned i = 0; i < D; ++i) {
        d2 += (a[i] - b[i]) * (a[i] - b[i]);
    }
    return d2;
}

template <unsigned D>
struct WrappedDistanceSquared
{
    explicit WrappedDistanceSquared(unsigned /*count*/) {}

    float operator()(const NPoint<D>&a, const NPoint<D>& b) const
    {
        return wrappedDistanceSquared(a, b);
    }
};

struct ChebyshevWrappedDistanceSquared
{
    explicit ChebyshevWrappedDistanceSquared(unsigned /*count*/) {}

    float operator()(const NPoint<2>& a, const NPoint<2>& b) const
    {
        float d2 = 0.0f;
        for (unsigned i = 0; i < 2; ++i) {
            const float d = wrapped1DDist(a[i], b[i]);
            d2 = std::max(d, d2);
        }
        return d2;
    }
};

class MinkowskiWrappedDistanceSquared
{
    float mP;

public:
    explicit MinkowskiWrappedDistanceSquared(unsigned count) :
        // An exponent of 2 is Euclidian distance. We start with a value that
        // favors diagonal distance, and move to Euclidian over time.
        mP(std::min(2.0f, 0.5f + count/8.0f))
    {
    }

    float operator()(const NPoint<2>& a, const NPoint<2>& b) const
    {
        float d2 = 0.0f;
        for (unsigned i = 0; i < 2; ++i) {
            const float d = wrapped1DDist(a[i], b[i]);
            d2 += std::pow(d, mP);
        }
        // 2/mP instead of 1/mP because this is the distance squared.
        return std::pow(d2, 2.0f/mP);
    }
};

struct DiskWrappedDistanceSquared
{
    explicit DiskWrappedDistanceSquared(unsigned /*count*/) {}

    float operator()(const NPoint<2>& a, const NPoint<2>& b) const
    {
        const float d = wrappedDistanceSquared(a, b);
        const float mx = 0.5f*0.5f;
        return mx - std::abs(d - mx);
    }
};

namespace std {

template <unsigned D>
struct hash<NPoint<D>>
{
public:
    typedef NPoint<D> argument_type;
    typedef std::size_t result_type;

    result_type operator()(argument_type const& p) const;

private:
    static void hash_combine(result_type& seed, result_type v);
};

template <unsigned D>
inline void hash<NPoint<D>>::hash_combine(typename hash<NPoint<D>>::result_type& seed,
                                            typename hash<NPoint<D>>::result_type v)
{
    seed ^= v + 0x9e3779b9u + (seed<<6u) + (seed>>2u);
}

template <unsigned D>
inline typename hash<NPoint<D>>::result_type hash<NPoint<D>>::operator()(typename hash<NPoint<D>>::argument_type const& p) const
{
    result_type s = result_type();
    for (unsigned i = 0; i < D; ++i) {
        hash_combine(s, std::hash<float>()(p[i]));
    }
    return s;
}

} // namespace std


