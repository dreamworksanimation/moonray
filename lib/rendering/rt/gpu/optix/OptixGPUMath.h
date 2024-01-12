// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cuda_runtime.h>
#include <math.h>
#include <optix.h>

// On the GPU side, we want to use the native CUDA float2/float3/float4 types
// which means we can't use the regular Moonray math library.  This means we
// need to reimplement a few basic vector math things here.  We also implement
// a bunch of the spline/curve functions here and extend Optix's Aabb functionality.


// These normally come from cstdint.h or limits but there's problems with pulling
// those complex headers into NVCC so instead of fighting with those I just copied
// the two values I needed into here.
#define FLOAT_EPSILON    1.192092896e-07f
#define FLOAT_MAX        3.402823466e+38f

#ifndef __CUDACC__
// code not compiled by NVCC

// CUDA defines rsqrtf() but the regular host-side math includes don't
inline __host__
float rsqrtf(const float a)
{
    return 1.f / sqrtf(a);
}

// end code not compiled by NVCC
#endif

// Extended versions of clamp/min/max

inline __host__ __device__
float clampf(const float val, const float low, const float high)
{
    return fminf(high, fmaxf(low, val));
}

inline __host__ __device__
float fmaxf(const float a, const float b, const float c)
{
    return fmaxf(fmaxf(a, b), c);
}

inline __host__ __device__
float fmaxf(const float a, const float b, const float c, const float d)
{
    return fmaxf(fmaxf(a, b), fmaxf(c, d));
}

inline __host__ __device__
float fminf(const float a, const float b, const float c)
{
    return fminf(fminf(a, b), c);
}

inline __host__ __device__
float fminf(const float a, const float b, const float c, const float d)
{
    return fminf(fminf(a, b), fminf(c, d));
}

// Other misc useful functions

inline __host__ __device__
float lerpf(const float v1, const float v2, const float t)
{
    return (1.f - t) * v1 + t * v2;
}

inline __host__ __device__
float4 lerp(const float4& v1, const float4& v2, const float t)
{
    return {(1.f - t) * v1.x + t * v2.x,
            (1.f - t) * v1.y + t * v2.y,
            (1.f - t) * v1.z + t * v2.z,
            (1.f - t) * v1.w + t * v2.w};
}

inline __host__ __device__
void swapf(float& a, float& b)
{
    float tmp = a;
    a = b;
    b = tmp;
}

// float3 operators and functions

inline __host__ __device__
float3 operator+(const float3& a, const float3& b)
{
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

inline __host__ __device__
float3 operator+(const float3& a, const float b)
{
    return {a.x + b, a.y + b, a.z + b};
}

inline __host__ __device__
float3 operator-(const float3& a, const float3& b)
{
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

inline __host__ __device__
float3 operator-(const float3& a, const float b)
{
    return {a.x - b, a.y - b, a.z - b};
}

inline __host__ __device__
float3 operator*(const float3& v, const float a)
{
    return {v.x * a, v.y * a, v.z * a};
}

inline __host__ __device__
float3 operator*(const float a, const float3& v)
{
    return {v.x * a, v.y * a, v.z * a};
}

inline __host__ __device__
float3 operator/(const float3& v, const float a)
{
    return {v.x / a, v.y / a, v.z / a};
}

inline __host__ __device__
void operator/=(float3& v, const float a)
{
    v.x /= a; v.y /= a; v.z /= a;
}

inline __host__ __device__
float dot(const float3& a, const float3& b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline __host__ __device__
float3 cross(const float3& a, const float3& b)
{
    return {a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x};
}

inline __host__ __device__
float length(const float3& v)
{
    return sqrtf(dot(v, v));
}

inline __host__ __device__
float3 normalize(const float3& v)
{
    return v * rsqrtf(dot(v, v));
}

inline __host__ __device__
void makeCoordSystem(const float3& v1, float3* v2, float3* v3)
{
    // v1 must be normalized
    if (fabs(v1.x) > fabs(v1.y)) {
        // cheaper than full normalize as we know one component is zero
        float s = rsqrtf(v1.x * v1.x + v1.z * v1.z);
        *v2 = {-v1.z * s, 0.f, v1.x * s};
    } else {
        float s = rsqrtf(v1.y * v1.y + v1.z * v1.z);
        *v2 = {0.f, v1.z * s, -v1.y * s};
    }

    *v3 = cross(v1, *v2);
}

inline __host__ __device__
float maxAbsComponent(const float3& v)
{
    return fmaxf(fabsf(v.x), fabsf(v.y), fabsf(v.z));
}

// float4 operators and functions

inline __host__ __device__
float4 operator+(const float4& a, const float4& b)
{
    return {a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w};
}

inline __host__ __device__
float4 operator-(const float4& v)
{
    return {-v.x, -v.y, -v.z, -v.w};
}

inline __host__ __device__
float4 operator-(const float4& a, const float4& b)
{
    return {a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w};
}

inline __host__ __device__
float4 operator*(const float4& v, const float a)
{
    return {v.x * a, v.y * a, v.z * a, v.w * a};
}

inline __host__ __device__
float4 operator*(const float a, const float4& v)
{
    return {v.x * a, v.y * a, v.z * a, v.w * a};
}

inline __host__ __device__
float3 make_float3(const float4& v)
{
    return {v.x, v.y, v.z};
}

inline __host__ __device__
float4 make_float4(const float3& v, const float a)
{
    return {v.x, v.y, v.z, a};
}

// OptixAabb functions

inline __host__ __device__
OptixAabb emptyAabb()
{
    return {FLOAT_MAX, FLOAT_MAX, FLOAT_MAX, -FLOAT_MAX, -FLOAT_MAX, -FLOAT_MAX};
}

inline __host__ __device__
OptixAabb expandAabb(const OptixAabb& bb, const float3& p)
{
    return {p.x < bb.minX ? p.x : bb.minX,
            p.y < bb.minY ? p.y : bb.minY,
            p.z < bb.minZ ? p.z : bb.minZ,
            p.x > bb.maxX ? p.x : bb.maxX,
            p.y > bb.maxY ? p.y : bb.maxY,
            p.z > bb.maxZ ? p.z : bb.maxZ};
}

inline __host__ __device__
OptixAabb combineAabbs(const OptixAabb& bb0, const OptixAabb& bb1)
{
    return {bb0.minX < bb1.minX ? bb0.minX : bb1.minX,
            bb0.minY < bb1.minY ? bb0.minY : bb1.minY,
            bb0.minZ < bb1.minZ ? bb0.minZ : bb1.minZ,
            bb0.maxX > bb1.maxX ? bb0.maxX : bb1.maxX,
            bb0.maxY > bb1.maxY ? bb0.maxY : bb1.maxY,
            bb0.maxZ > bb1.maxZ ? bb0.maxZ : bb1.maxZ};
}

// Curve functions

enum CurveType
{
    BEZIER,
    BSPLINE,
    LINEAR
};

// https://en.wikipedia.org/wiki/B%C3%A9zier_curve
inline __host__ __device__
float4 getBezierBasis(const float t)
{
    const float t1 = 1.f - t;
    return {t1 * t1 * t1,
            3.f * t1 * t1 * t,
            3.f * t1 * t * t,
            t * t * t};
}

inline __host__ __device__
float4 getBezierBasisDerivative(const float t)
{
    const float t1 = 1.f - t;
    return {-3.f * t1 * t1,
            -6.f * t1 * t + 3.f * t1 * t1,
            -3.f * t * t + 6.f * t1 * t,
             3.f * t * t};
}

inline __host__ __device__
float4 evalBezier(const float4 cp[4], const float t)
{
    float4 basis = getBezierBasis(t);
    return basis.x * cp[0] + basis.y * cp[1] + basis.z * cp[2] + basis.w * cp[3];
}

inline __host__ __device__
float3 evalBezierDerivative(const float4 cp[4], const float t)
{
    float4 basis = getBezierBasisDerivative(t);
    return make_float3(basis.x * cp[0] + basis.y * cp[1] + basis.z * cp[2] + basis.w * cp[3]);
}

// From embree bspline_curve.h
inline __host__ __device__
float4 getBsplineBasis(const float t)
{
    const float t1 = 1.0f - t;
    return {(1.f / 6.f) * t1 * t1 * t1,
            (1.f / 6.f) * ((4.0f*(t1*t1*t1) + (t*t*t)) + (12.0f*((t1*t)*t1) + 6.0f*((t*t1)*t))),
            (1.f / 6.f) * ((4.0f*(t*t*t) + (t1*t1*t1)) + (12.0f*((t*t1)*t) + 6.0f*((t1*t)*t1))),
            (1.f / 6.f) * t * t * t};
}

inline __host__ __device__
float4 getBsplineBasisDerivative(const float t)
{
    const float t1 = 1.0f - t;
    return {0.5f * -t1 * t1,
            0.5f * (-t * t - 4.0f * (t * t1)),
            0.5f * (t1 * t1 + 4.0f * (t1 * t)),
            0.5f * (t * t)};
}

inline __host__ __device__
float4 evalBspline(const float4 cp[4], const float t)
{
    float4 basis = getBsplineBasis(t);
    return basis.x * cp[0] + basis.y * cp[1] + basis.z * cp[2] + basis.w * cp[3];
}

inline __host__ __device__
float3 evalBsplineDerivative(const float4 cp[4], const float t)
{
    float4 basis = getBsplineBasisDerivative(t);
    return make_float3(basis.x * cp[0] + basis.y * cp[1] + basis.z * cp[2] + basis.w * cp[3]);
}

// Affine transform

namespace moonray {
namespace rt {

struct OptixGPUXform
{
    OptixGPUXform() = default;

    __host__ __device__
    static OptixGPUXform identityXform()
    {
        return {1.f, 0.f, 0.f, 0.f,
                0.f, 1.f, 0.f, 0.f,
                0.f, 0.f, 1.f, 0.f};
    }

    __host__ __device__
    static OptixGPUXform rotateToZAxisXform(const float3& dir)
    {
        // transform to space where dir.z is the +Z axis
        float3 vz = normalize(dir);
        float3 vx, vy;
        makeCoordSystem(vz, &vx, &vy);
        // form the matrix:
        // [ vx.x, vy.x, vz.x, 0 ]
        // [ vx.y, vy.y, vz.y, 0 ]
        // [ vx.z, vy.z, vz.z, 0 ]
        // take the transpose to invert it
        return {vx.x, vx.y, vx.z, 0.f,
                vy.x, vy.y, vy.z, 0.f,
                vz.x, vz.y, vz.z, 0.f};
    }

    __host__ __device__
    float3 transformVector(const float3& v) const
    {
        return {m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z,
                m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z,
                m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z};
    }

    __host__ __device__
    float3 transformPoint(const float3& p) const
    {
        return {m[0][0] * p.x + m[0][1] * p.y + m[0][2] * p.z + m[0][3],
                m[1][0] * p.x + m[1][1] * p.y + m[1][2] * p.z + m[1][3],
                m[2][0] * p.x + m[2][1] * p.y + m[2][2] * p.z + m[2][3]};
    }

    __host__ __device__
    OptixAabb transformAabb(const OptixAabb& bb) const
    {
        // TODO: Not the most efficient way to do this, but the easiest
        // see Graphics Gems pg 548 for a faster way.  This slow way is what we do in
        // CPU-side math::transformBounds()...
        OptixAabb result = emptyAabb();
        result = expandAabb(result, transformPoint({bb.minX, bb.minY, bb.minZ}));
        result = expandAabb(result, transformPoint({bb.minX, bb.minY, bb.maxZ}));
        result = expandAabb(result, transformPoint({bb.minX, bb.maxY, bb.minZ}));
        result = expandAabb(result, transformPoint({bb.minX, bb.maxY, bb.maxZ}));
        result = expandAabb(result, transformPoint({bb.maxX, bb.minY, bb.minZ}));
        result = expandAabb(result, transformPoint({bb.maxX, bb.minY, bb.maxZ}));
        result = expandAabb(result, transformPoint({bb.maxX, bb.maxY, bb.minZ}));
        result = expandAabb(result, transformPoint({bb.maxX, bb.maxY, bb.maxZ}));
        return result;
    }

    __host__ __device__
    void toOptixTransform(float transform[12]) const
    {
        transform[0] = m[0][0]; transform[1] = m[0][1];  transform[2] = m[0][2];  transform[3] = m[0][3];
        transform[4] = m[1][0]; transform[5] = m[1][1];  transform[6] = m[1][2];  transform[7] = m[1][3];
        transform[8] = m[2][0]; transform[9] = m[2][1]; transform[10] = m[2][2]; transform[11] = m[2][3];
    }

    float m[3][4];
    // This is an affine 4x4 transformation matrix so the last row is always 0 0 0 1
    // and we don't need to store it.  This convention is opposite to what we use
    // in scene_rdl2 where an affine matrix has the translation in the last row and
    // 0 0 0 1 in the last column.  We use a different convention than scene_rdl2 to match
    // up with what Optix is expecting.
};

} // namespace rt
} // namespace moonray

