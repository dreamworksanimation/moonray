// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cmath>
#include <ios>

const float pi = acos(-1.0f);

template <class T>
constexpr std::add_const_t<T>& as_const(T& t) noexcept
{
    return t;
}

class IOStateSaver
{
public:
    IOStateSaver(std::ios& stream) :
        mStream(stream),
        mFlags(stream.flags()),
        mPrec(stream.precision()),
        mWidth(stream.width())
    {
    }

    ~IOStateSaver()
    {
        mStream.flags(mFlags);
        mStream.precision(mPrec);
        mStream.width(mWidth);
   }
 
private:
    std::ios& mStream;
    std::ios::fmtflags mFlags;
    std::streamsize mPrec;
    std::streamsize mWidth;
};

constexpr unsigned powu(unsigned base, unsigned p)
{
    return (p == 0) ? 1 : base * powu(base, p - 1);
}

inline float lerp(float u, float min, float max)
{
    return (1.0f - u) * min + u * max;
}

inline void polarToCartesian(float r, float phi, float& x, float& y)
{
    float sinPhi;
    float cosPhi;

#if defined(__clang__)
    __sincosf(phi, &sinPhi, &cosPhi);
#else
    sincosf(phi, &sinPhi, &cosPhi);
#endif
    
    x = r*cosPhi;
    y = r*sinPhi;
}

inline void cartesianToPolar(float x, float y, float& r, float& phi)
{
    r = std::sqrt(x*x + y*y);
    phi = std::atan2(y, x);
}

// These are the maximum radius of disks packed in a D-dimensional space.
template <unsigned D>
float radiusMax(unsigned ndisks, float volume = 1.0f);

template <>
inline float radiusMax<1>(unsigned ndisks, float volume)
{
    return volume / (2.0f * static_cast<float>(ndisks));
}

// Taken from "A Comparison of Methods for Generating Poisson Disk
// Distributions" by Lagae and Dutré. This computes the maximum radius of disks
// in optimal disk (circle) packing (spoiler: it's a hex lattice). This assumes
// the radius to the edges of other disks, but the Poisson radius is to other
// points, so it needs to be doubled after the fact.
template <>
inline float radiusMax<2>(unsigned ndisks, float volume)
{
    static const float kSqrt3 = std::sqrt(3.0f);
    return std::sqrt(volume / (2.0f * kSqrt3 * static_cast<float>(ndisks)));
}

// Korkin A., Zolotarev G.: Sur les formes quadratiques positives.
// Mathematische Annalen 11 (1877), 242–292
template <>
inline float radiusMax<3>(unsigned ndisks, float volume)
{
    static const float kSqrt2 = std::sqrt(2.0f);
    return std::pow(volume / (4.0f * kSqrt2 * static_cast<float>(ndisks)), 1.0f/3.0f);
}

template <>
inline float radiusMax<4>(unsigned ndisks, float volume)
{
    return std::pow(volume / (8.0f * static_cast<float>(ndisks)), 1.0f/4.0f);
}

inline float radiusMax(unsigned ndisks, float volume, unsigned dimensions)
{
    switch (dimensions) {
        case 1: return radiusMax<1>(ndisks, volume);
        case 2: return radiusMax<2>(ndisks, volume);
        case 3: return radiusMax<3>(ndisks, volume);
        case 4: return radiusMax<4>(ndisks, volume);
        default:
            assert(!"Not implemented");
            return -1.0f;
    }
}
