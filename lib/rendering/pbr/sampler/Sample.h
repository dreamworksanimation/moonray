// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Sample.hh"
#include <scene_rdl2/common/platform/Platform.h>

#include <istream>
#include <ostream>

namespace moonray {
namespace pbr {

///
/// @class Sample Sample.h <rendering/pbr/sampler/Sample.h>
/// @brief This class holds sample values for pixel/lens/time
///
struct Sample
{
    SAMPLE_MEMBERS;

    static uint32_t hudValidation(bool verbose) { SAMPLE_VALIDATION; }
};

finline std::ostream& operator<<(std::ostream& outs, const Sample& s)
{
    return outs << s.pixelX << ' '
                << s.pixelY << ' '
                << s.lensU  << ' '
                << s.lensV  << ' '
                << s.time;
}

// Not robust -- used for testing.
inline bool operator==(const Sample& a, const Sample& b)
{
    return a.pixelX == b.pixelX &&
           a.pixelY == b.pixelY &&
           a.lensU  == b.lensU  &&
           a.lensV  == b.lensV  &&
           a.time   == b.time;
}

struct Sample3D
{
    SAMPLE_3D_MEMBERS;

    static uint32_t hudValidation(bool verbose) { SAMPLE_3D_VALIDATION; }
};

struct Sample2D
{
    SAMPLE_2D_MEMBERS;

    static uint32_t hudValidation(bool verbose) { SAMPLE_2D_VALIDATION; }
};

finline std::ostream& operator<<(std::ostream& outs, const Sample2D& s)
{
    return outs << s.u << ' ' << s.v;
}

finline std::istream& operator>>(std::istream& ins, Sample2D& s)
{
    return ins >> s.u >> s.v;
}

finline std::ostream& operator<<(std::ostream& outs, const Sample3D& s)
{
    return outs << s.u << ' ' << s.v << ' ' << s.w;
}

finline std::istream& operator>>(std::istream& ins, Sample3D& s)
{
    return ins >> s.u >> s.v >> s.w;
}

// These functions are generic accessors for various sample types. We could use
// array access, but it gets really obscure in cases like 5D samples if we're
// not using the items by name.
finline float  getPrimaryValue0(const Sample& s)   { return s.pixelX; } 
finline float& getPrimaryValue0(Sample& s)         { return s.pixelX; } 
finline float  getPrimaryValue1(const Sample& s)   { return s.pixelY; }
finline float& getPrimaryValue1(Sample& s)         { return s.pixelY; }
finline float  getPrimaryValue0(const Sample3D& s) { return s.u; }
finline float& getPrimaryValue0(Sample3D& s)       { return s.u; }
finline float  getPrimaryValue1(const Sample3D& s) { return s.v; }
finline float& getPrimaryValue1(Sample3D& s)       { return s.v; }
finline float  getPrimaryValue0(const Sample2D& s) { return s.u; }
finline float& getPrimaryValue0(Sample2D& s)       { return s.u; }
finline float  getPrimaryValue1(const Sample2D& s) { return s.v; }
finline float& getPrimaryValue1(Sample2D& s)       { return s.v; }
finline float  getLensValue0(const Sample& s)      { return s.lensU; }
finline float& getLensValue0(Sample& s)            { return s.lensU; }
finline float  getLensValue1(const Sample& s)      { return s.lensV; } 
finline float& getLensValue1(Sample& s)            { return s.lensV; } 
finline float  getLensValue0(const Sample2D& s)    { return s.u; } 
finline float& getLensValue0(Sample2D& s)          { return s.u; }
finline float  getLensValue1(const Sample2D& s)    { return s.v; }
finline float& getLensValue1(Sample2D& s)          { return s.v; }

template <typename T>
struct GenerateRandomPointImpl;

template <typename T, typename RNG>
T generateRandomPoint(RNG& rng)
{
    // Users, don't touch this!
    // Specialize GenerateRandomPointImpl instead.
    return GenerateRandomPointImpl<T>::template apply<RNG>(rng);
}

template <>
struct GenerateRandomPointImpl<Sample>
{
    template <typename RNG>
    static Sample apply(RNG& rng)
    {
        Sample s;
        s.pixelX = rng();
        s.pixelY = rng();
        s.lensU  = rng();
        s.lensV  = rng();
        s.time   = rng();
        return s;
    }
};

template <>
struct GenerateRandomPointImpl<Sample3D>
{
    template <typename RNG>
    static Sample3D apply(RNG& rng)
    {
        Sample3D s;
        s.u = rng();
        s.v = rng();
        s.w = rng();
        return s;
    }
};

template <>
struct GenerateRandomPointImpl<Sample2D>
{
    template <typename RNG>
    static Sample2D apply(RNG& rng)
    {
        Sample2D s;
        s.u = rng();
        s.v = rng();
        return s;
    }
};

} // namespace pbr
} // namespace moonray

