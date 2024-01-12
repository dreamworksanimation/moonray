// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <scene_rdl2/common/math/BBox.h>
#include <scene_rdl2/common/math/Mat4.h>
#include <scene_rdl2/common/math/Vec3.h>
#include <array>

namespace moonray {
namespace mcrt_common {

// Some camera types may implement a frustum, i.e. orthographic and perspective
//  cameras.  A frustum is defined by right, left, bottom, top, far, and near
//  clipping planes, plus the viewport in raster space and the camera to screen xform.
struct Frustum
{
    // Performs a simple test to check if a BBox overlaps the frustum.  This test
    //  may produce false positives, but may NOT produce false negatives.
    bool testBBoxOverlaps(const scene_rdl2::math::BBox3f& bbox) const;

    // Projects a camera-space 3D point to a 3D point on the z=-1 plane, where
    //  the result's x, y coordinates are the pixel's raster coordinates.
    scene_rdl2::math::Vec3f projectToViewport(const scene_rdl2::math::Vec3f& p) const;

    // Computes a 6-bit Cohen-Sutherland outcode value where each bit is the
    // result of a point/plane half-space test with one of the clip planes.
    // The bit is set if the point is behind the plane (on the side opposite the
    // normal defined by the plane equation coefficients).
    // A point on the plane is classified as 'inside'.
    // Bit i corresponds to mClipPlanes[i].
    int computeOutcode(const scene_rdl2::math::Vec3f& p) const;

    // Clip the convex planar poly defined by (xyzIn, numIn) to one plane of the frustum,
    // outputting results in xyzOut. Returns the number of points in the clipped poly.
    // The output array xyzOut must be big enough to hold numIn+1 points.
    int clipPolyToPlane(scene_rdl2::math::Vec3f* xyzOut, const scene_rdl2::math::Vec3f* xyzIn, int numIn, int planeIdx) const;

    // Clip the convex planar poly defined by (xyzIn, stIn, numIn) to one plane of the frustum,
    // outputting results in (xyzOut, stOut). Returns the number of points in the clipped poly.
    // The output arrays (xyzOut, stOut) must be big enough to hold numIn+1 points.
    int clipPolyToPlane(scene_rdl2::math::Vec3f* xyzOut, scene_rdl2::math::Vec2f* stOut,
                        const scene_rdl2::math::Vec3f* xyzIn, const scene_rdl2::math::Vec2f* stIn, int numIn, int planeIdx) const;

    // Clip the convex planar poly defined by (xyzIn, numIn) to the planes of the frustum specified by
    // whichPlanes, outputting results in xyzOut. Returns the number of points in the clipped poly.
    // The output array xyzOut must be big enough to hold numIn+6 points.
    // If the whichPlanes mask is omitted, the function will clip to all 6 planes.
    // If the clipping operation is preceded by outcode testing, and polygons whose AND-of-outcodes have
    // been trivially rejected, then the OR-of-outcodes can be supplied as the whichPlanes value, assuming
    // outcodes have been generated using Frustum::computeOutcode().
    // If supplied, whichPlanes must be non-zero (otherwise why bother clippng?)
    int clipPoly(scene_rdl2::math::Vec3f* xyzOut, const scene_rdl2::math::Vec3f* xyzIn, int numIn, int whichPlanes = 0x3F) const;

    // Clip the convex planar poly defined by (xyzIn, stIn, numIn) to the planes of the frustum specified by
    // whichPlanes, outputting results in (xyzOut, stOut). Returns the number of points in the clipped poly.
    // The output arrays (xyzOut, stOut) must be big enough to hold numIn+6 points.
    // If the whichPlanes mask is omitted, the function will clip to all 6 planes.
    // If the clipping operation is preceded by outcode testing, and polygons whose AND-of-outcodes have
    // been trivially rejected, then the OR-of-outcodes can be supplied as the whichPlanes value, assuming
    // outcodes have been generated using Frustum::computeOutcode().
    // If supplied, whichPlanes must be non-zero (otherwise why bother clippng?)
    int clipPoly(scene_rdl2::math::Vec3f* xyzOut, scene_rdl2::math::Vec2f* stOut,
                 const scene_rdl2::math::Vec3f* xyzIn, const scene_rdl2::math::Vec2f* stIn, int numIn, int whichPlanes = 0x3F) const;
    
    // Transform all of the clipping planes by the given transform
    void transformClipPlanes(const scene_rdl2::math::Mat4d& transform);

    // Camera space to screen space xform
    scene_rdl2::math::Mat4f mC2S;

    // 6 planes with 4 coefficients each, i.e. Ax + By + Cz + D = 0;
    //  Ordering: right, left, bottom, top, far, near
    float mClipPlanes[6][4];

    // Viewport in raster space (pixels)
    std::array<int, 4> mViewport;
};

} // namespace mcrt_common
} // namespace moonray


