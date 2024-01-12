// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

//
#pragma once

#include <scene_rdl2/common/math/Color.h>
#include <scene_rdl2/common/math/Mat4.h>
#include <scene_rdl2/common/math/Math.h>
#include <scene_rdl2/render/util/BitUtils.h>

namespace moonray {
namespace pbr {

// Helper function adapted from PBRT3 that emulates the behavior of
// std::upper_bound(). The implementation here adds some bounds checking
// for corner cases (e.g., making sure that a valid interval is selected
// even in the case the predicate evaluates to true or false for all entries),
// which would normally have to follow a call to std::upper_bound()
template <typename Predicate>
int findInterval(int size, const Predicate& pred)
{
    int first = 0, len = size;
    while (len > 0) {
        int half = len >> 1, middle = first + half;
        if (pred(middle)) {
            first = middle + 1;
            len -= half + 1;
        } else {
            len = half;
        }
    }
    return scene_rdl2::math::clamp(first - 1, 0, size - 2);
}


/// sample 1D value t between range a and b and the distribution is
/// propotional to \exp^{-\sigma(t-a)}
/// For detail reference see:
/// "Importance Sampling Techniques for Path Tracing in Participating Media"
/// EGSR2012 Christopher Kulla and Marcos Fajardo
finline float
sampleDistanceExponential(float u, float sigma, float a, float b)
{
    return a - scene_rdl2::math::log(1.0f - u * (1.0f - scene_rdl2::math::exp(sigma * (a - b)))) / sigma;
}


/// pdf for above distance sampling
finline float
pdfDistanceExponential(float t, float sigma, float a, float b)
{
    return sigma / (scene_rdl2::math::exp((t - a) * sigma) - scene_rdl2::math::exp((t - b) * sigma));
}


/// sample 1D value t between range a and b and the distribution is
/// propotional to 1/(D^2+t^2) where D is the distance from a pivot point
/// to the line formed by extanding ab
/// For detail reference see:
/// "Importance Sampling Techniques for Path Tracing in Participating Media"
/// EGSR2012 Christopher Kulla and Marcos Fajardo
finline float
sampleEquiAngular(float u, float D, float thetaA, float thetaB)
{
    return D * scene_rdl2::math::tan((1 - u) * thetaA + u * thetaB);
}


/// pdf for above equi-angular sampling
finline float
pdfEquiAngular(float t, float D, float thetaA, float thetaB)
{
    return D / ((thetaB - thetaA) * (D * D + t * t));
}


/// Map samples from a square to a unit radius disk with the same pdf
finline void
squareSampleToCircle(float r1, float r2, float *u, float *v)
{
    float r = scene_rdl2::math::sqrt(r1);
    float angle = 2.0f * scene_rdl2::math::sPi * r2;
    *u = r * scene_rdl2::math::cos(angle);
    *v = r * scene_rdl2::math::sin(angle);
}

finline void
squareSampleToCircle(float xy[2], float uv[2])
{
    squareSampleToCircle(xy[0], xy[1], &uv[0], &uv[1]);
}


/// Map samples from a square to a unit radius disk with the same pdf
/// keeping the distribution properties of the original samples. Based on
/// Shirley and Chiu's algorithm.
/// The code has been updated to use a new version of the algorithm posted on
/// Shirley's site.
finline void
toUnitDisk(float& u, float& v)
{
    if (u == 0.5f && v == 0.5f) {
        u = 0.0f;
        v = 0.0f;
    } else {
        float phi,r;
        const float a = 2.0f*u - 1.0f;
        const float b = 2.0f*v - 1.0f;

        if (scene_rdl2::math::abs(a) > scene_rdl2::math::abs(b)) {
            r = a;
            phi = (scene_rdl2::math::sPi/4.0f)*(b/a);
        } else {
            r = b;
            phi = (scene_rdl2::math::sPi/2.0f) - (scene_rdl2::math::sPi/4.0f)*(a/b);
        }

        float cosPhi;
        float sinPhi;

        scene_rdl2::math::sincos(phi, &sinPhi, &cosPhi);

        u = r*cosPhi;
        v = r*sinPhi;
    }
}


namespace detail {
// move to pbr
/// Pixel filtering as described in "Generation of Stratified Samples for
/// B-Spline Pixel Filtering", by Stark, Shirley, and Ashikhmin
__forceinline float
distb1(float r)
{
    float u = r;
    for (int j = 0; j < 5; ++j) {
        u = (11.0f * r + u * u * (6.0f + u * (8.0f - 9.0f * u)))
                / (4.0f + 12.0f * u * (1 + u * (1.0f - u)));
    }
    return u;
}
}


/// Pixel filtering as described in "Generation of Stratified Samples for
/// B-Spline Pixel Filtering", by Stark, Shirley, and Ashikhmin
finline float
cubicBSplineWarp(float r)
{
    if (r < 1.0f / 24.0f) {
        return scene_rdl2::math::pow(24.0f * r, 0.25f) - 2.0f;
    } else if (r < 0.5f) {
        return detail::distb1(24.0f / 11.0f * (r - 1.0f / 24.0f)) - 1.0f;
    } else if (r < 23.0f / 24.0f) {
        return 1.0f - detail::distb1(24.0f / 11.0f * (23.0f / 24.0f - r));
    } else {
        return 2.0f - scene_rdl2::math::pow(24.0f * (1.0f - r), 0.25f);
    }
}


/// Pixel filtering as described in "Generation of Stratified Samples for
/// B-Spline Pixel Filtering", by Stark, Shirley, and Ashikhmin
finline float
quadraticBSplineWarp(float r)
{
    if (r < 1.0f / 6.0f) {
        return scene_rdl2::math::pow(6.0f * r, 1.0f / 3.0f) - 3.0f / 2.0f;
    } else if (r < 5.0f / 6.0f) {
        float u = (6.0f * r - 3.0f) / 4.0f;
        for (int j = 0; j < 4; ++j) {
            u = (8.0f * u * u * u - 12.0f * r + 6.0f) / (12.0f * u * u - 9.0f);
        }
        return u;
    } else {
        return 3.0f / 2.0f - scene_rdl2::math::pow(6.0f * (1.0f - r), 1.0f / 3.0f);
    }
}


finline bool
isSampleValid(const scene_rdl2::math::Color &c, float pdf)
{
    bool validSample = (pdf != 0.0f)  &&  !scene_rdl2::math::isExactlyZero(c);

    return validSample;
}


finline bool
isSampleInvalid(const scene_rdl2::math::Color &c, float pdf)
{
    bool invalidSample = (pdf == 0.0f)  ||  scene_rdl2::math::isExactlyZero(c);

    return invalidSample;
}


finline float
powerHeuristic(float f, float g)
{
    return (f * f) / (f * f + g * g);
}

// Returns false if matrix had non-uniform scale, or true otherwise.
finline bool
extractUniformScale(const scene_rdl2::math::Mat4f &mat, float *scale)
{
    float x = mat.row0().length();
    float y = mat.row1().length();
    float z = mat.row2().length();

    if (!scene_rdl2::math::isEqual(x, y, 0.0001f) || !scene_rdl2::math::isEqual(x, z, 0.0001f)) {
        return false;
    }

    *scale = (x + y + z) * (1.0f / 3.0f);
    return true;
}

finline scene_rdl2::math::Vec3f
extractNonUniformScale(const scene_rdl2::math::Mat4f &mat)
{
    return scene_rdl2::math::Vec3f(mat.row0().length(),
                                   mat.row1().length(),
                                   mat.row2().length());

}

// If integrating over a surface instead of solid angle, we need to scale
// by this factor. If the return value could result in a denormalized result,
// return 0. Also see absAreaToSolidAngleScale.
finline float
areaToSolidAngleScale(const scene_rdl2::math::Vec3f &wi, const scene_rdl2::math::Vec3f &lightNormal, float dist)
{
    float denom = -dot(wi, lightNormal);
    return (denom < 0.00001f) ? 0.0f : (dist * dist) / denom;
}

// Like the areaToSolidAngleScale function but use this one if there is a
// possibility that wi and lightNormal are within 90 degrees of each other.
// If that's not possible, then using the above version is slightly faster since
// it bypasses the abs call.
finline float
absAreaToSolidAngleScale(const scene_rdl2::math::Vec3f &wi, const scene_rdl2::math::Vec3f &lightNormal, float dist)
{
    float denom = scene_rdl2::math::abs(dot(wi, lightNormal));
    return (denom < 0.00001f) ? 0.0f : (dist * dist) / denom;
}


finline bool
intersectUnitSphere(const scene_rdl2::math::Vec3f &pos, const scene_rdl2::math::Vec3f &dir,
                    scene_rdl2::math::Vec3f *hit, float *dist)
{
    MNRY_ASSERT(isNormalized(dir));

    float b = scene_rdl2::math::dot(dir, pos) * 2.0f;
    float c = pos.lengthSqr() - 1.0f;

    // Exit if ray outside sphere (c > 0) and is pointing away (b > 0).
    if (c > 0.0f && b > 0.0f) {
        return false;
    }

    // Find quadratic discriminant
    float discrim = b * b - 4.0f * c;
    if (discrim < 0.0f) {
        return false;
    }

    float sqrtDiscrim = scene_rdl2::math::sqrt(discrim);
    float t0 = -0.5f * ((b < 0.0f) ? (b - sqrtDiscrim) : (b + sqrtDiscrim));
    float t1 = c / t0;

    if (t0 > t1) {
        std::swap(t0, t1);
    }

    if (t1 <= 0.0f) {
        return false;
    }

    float t = (t0 < 0.0f) ? t1 : t0;

    // Mathematically, we shouldn't need to normalize here. In practice, it helps
    // with robustness since it ensures that all components are within [-1, 1].
    // This is not the case under normal circumstances due to floating point
    // inaccuracies.
    *hit = normalize(pos + dir * t);
    *dist = t;

    return true;
}


// The word "reduce" in this context is used in a algoritmic sense, i.e. take
// multiple values and combine them somehow to form a single value. This is opposed
// to the more common usage which would be to lessen something. We're not actually
// lessening the transparency here.
finline float
reduceTransparency(const scene_rdl2::math::Color &color)
{
    return scene_rdl2::math::reduce_avg(color);
}


// Clamp radiance by scaling it down to avoid hue-shift problems
finline scene_rdl2::math::Color
smartClamp(const scene_rdl2::math::Color &radiance, float clamp)
{
    if (clamp > 0.0f) {
        float maxComponent = scene_rdl2::math::reduce_max(radiance);
        if (maxComponent > clamp) {
            float factor = clamp * scene_rdl2::math::rcp(maxComponent);
            return radiance * factor;
        } else {
            return radiance;
        }
    } else {
        return radiance;
    }
}


// Returns a higher precision inverse when only rotations and translations
// are present in the matrix.  This method first checks for this condition and
// falls back to the generic inverse function if the condition fails.  If the
// generic inverse is used, res will contain false.
template<typename Scalar>
scene_rdl2::math::Mat4<scene_rdl2::math::Vec4<Scalar> >
rtInverse(const scene_rdl2::math::Mat4<scene_rdl2::math::Vec4<Scalar> > &m, bool &res)
{
    scene_rdl2::math::QuaternionT<Scalar> rot;
    scene_rdl2::math::Mat3<scene_rdl2::math::Vec3<Scalar> > scale;
    const scene_rdl2::math::Mat3<scene_rdl2::math::Vec3<Scalar> > m3(m.vx.x, m.vx.y, m.vx.z,
                                             m.vy.x, m.vy.y, m.vy.z,
                                             m.vz.x, m.vz.y, m.vz.z);
    scene_rdl2::math::decompose(m3, scale, rot);
    const Scalar TOL = 1e-5;
    if (!scene_rdl2::math::isOne(scale[0][0], TOL) || !scene_rdl2::math::isOne(scale[1][1], TOL) ||
        !scene_rdl2::math::isOne(scale[2][2], TOL)) {
        // fallback to generic inverse
        res = false;
        return m.inverse();
    }

    // otherwise the inverse is just tInv * rTranpose
    scene_rdl2::math::Mat3<scene_rdl2::math::Vec3<Scalar> > rInv(m3.transposed());
    scene_rdl2::math::Vec3<Scalar>              tInv(-m.vw.x, -m.vw.y, -m.vw.z);

    res = true;
    return
        scene_rdl2::math::Mat4<scene_rdl2::math::Vec4<Scalar> >(1,      0,      0,      0,
                                        0,      1,      0,      0,
                                        0,      0,      1,      0,
                                        tInv.x, tInv.y, tInv.z, 1) *

        scene_rdl2::math::Mat4<scene_rdl2::math::Vec4<Scalar> >(rInv[0].x, rInv[0].y, rInv[0].z, 0,
                                        rInv[1].x, rInv[1].y, rInv[1].z, 0,
                                        rInv[2].x, rInv[2].y, rInv[2].z, 0,
                                        0,         0,         0,         1);
}


template<typename Scalar>
scene_rdl2::math::Mat4<scene_rdl2::math::Vec4<Scalar> >
rtInverse(const scene_rdl2::math::Mat4<scene_rdl2::math::Vec4<Scalar> > &m)
{
    bool isRotTrans;
    return rtInverse(m, isRotTrans);
}

// check if lane is active
finline bool
isActive(int32_t lanemask, int i)
{
    return ((1 << i) & lanemask);
}

} // namespace pbr
} // namespace moonray

