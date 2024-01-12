// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#include "Frustum.h"

namespace moonray {
namespace mcrt_common {

using namespace scene_rdl2::math;

bool
Frustum::testBBoxOverlaps(const BBox3f& bbox) const
{
    for (int i = 0; i < 6; i++) {
        // Perform a plane-AABB overlap test as described in "Real Time Rendering",
        //  section 10.7.1.  We only calculate the "max" box corner (the one closest
        //  to the plane on the "back" side of the plane.)
        // If the AABB is completely outside any of the frustum planes, the AABB is
        //  completely outside the frustum.  Else, it is not guaranteed to be completely
        //  outside.  Note that we are testing against the *infinite* planes that describe
        //  the frustum, not the *finite* frustum sides so we can fail to reject some
        //  AABBs.  This works well enough most of the time though, and failing to reject
        //  an AABB won't cause an error in the output render - just additional memory and
        //  processing.
        float x = mClipPlanes[i][0] < 0 ? bbox.lower[0] : bbox.upper[0];
        float y = mClipPlanes[i][1] < 0 ? bbox.lower[1] : bbox.upper[1];
        float z = mClipPlanes[i][2] < 0 ? bbox.lower[2] : bbox.upper[2];
        // xyz is the corner of the AABB that is closest to the plane on the negative (back) side.
        // Just plug into the clip plane's equation to see if it's actually behind the plane.
        if (x * mClipPlanes[i][0] + y * mClipPlanes[i][1] + z * mClipPlanes[i][2] + mClipPlanes[i][3] < 0) {
            return false;
        }
    }
    return true;
}

Vec3f
Frustum::projectToViewport(const Vec3f& p) const
{
    // transformH divides by w to put everything on the z=-1 plane
    Vec3f pv = transformH(mC2S, p);
    // need to scale by viewport size as pv is normalized to [-1, 1] range
    pv[0] = mViewport[0] + (pv[0] + 1) * 0.5f * (mViewport[2] - mViewport[0]);
    pv[1] = mViewport[1] + (pv[1] + 1) * 0.5f * (mViewport[3] - mViewport[1]);
    return pv;
}

int
Frustum::computeOutcode(const Vec3f& p) const
{
    int outcode = 0;
    for (int i = 0; i < 6; i++) {
        if (dot((const Vec3f)mClipPlanes[i], p) + mClipPlanes[i][3] < 0.0f) {
            // point is outside this frustum clip plane
            outcode |= 1 << i;
        }
    }
    return outcode;
}

#define MAX_POINTS 4    // Todo: better handling of max number of input points

int
Frustum::clipPolyToPlane(Vec3f* xyzOut, const Vec3f* xyzIn, int numIn, int planeIdx) const
{
    // Compute plane distances
    float d[MAX_POINTS + 5];
    for (int i = 0; i < numIn; i++) {
        d[i] = dot((const Vec3f)mClipPlanes[planeIdx], xyzIn[i]) + mClipPlanes[planeIdx][3];
    }

    // Clip poly
    int numOut = 0;
    for (int i0 = numIn - 1, i1 = 0; i1 < numIn; i0 = i1, i1++) {
        bool clipEdge = false;
        if (d[i0] >= 0.0f) {
            // Point i0 is inside => copy it
            xyzOut[numOut++] = xyzIn[i0];

            // If (i0,i1) is an exiting edge, clip it
            if (d[i1] < 0.0f) clipEdge = true;

            // If (i0,i1) is an entering edge, clip it
        } else if (d[i1] >= 0.0f) clipEdge = true;

        if (clipEdge) {
            // Compute and output edge intersection with plane
            float t = d[i0] / (d[i0] - d[i1]);
            xyzOut[numOut++] = xyzIn[i0] + t * (xyzIn[i1] - xyzIn[i0]);
        }
    }

    return numOut;
}

int
Frustum::clipPolyToPlane(Vec3f* xyzOut, Vec2f* stOut, const Vec3f* xyzIn, const Vec2f* stIn, int numIn, int planeIdx) const
{
    // Compute plane distances
    float d[MAX_POINTS + 5];
    for (int i = 0; i < numIn; i++) {
        d[i] = dot((const Vec3f)mClipPlanes[planeIdx], xyzIn[i]) + mClipPlanes[planeIdx][3];
    }

    // Clip poly
    int numOut = 0;
    for (int i0 =numIn - 1, i1 = 0; i1 < numIn; i0 = i1, i1++) {
        bool clipEdge = false;
        if (d[i0] >= 0.0f) {
            // Point i0 is inside => copy it
            xyzOut[numOut  ] = xyzIn[i0];
            stOut [numOut++] = stIn [i0];

            // If (i0,i1) is an exiting edge, clip it
            if (d[i1] < 0.0f) clipEdge = true;

            // If (i0,i1) is an entering edge, clip it
        } else if (d[i1] >= 0.0f) clipEdge = true;

        if (clipEdge) {
            // Compute and output edge intersection with plane
            float t = d[i0] / (d[i0] - d[i1]);
            xyzOut[numOut  ] = xyzIn[i0] + t * (xyzIn[i1] - xyzIn[i0]);
            stOut [numOut++] = stIn [i0] + t * (stIn [i1] - stIn [i0]);
        }
    }

    return numOut;
}

static inline bool isPlaneCountEven(int whichPlanes)
{
    // __popcnt() returns the number of bits set, so here we're testing for an even number of bits set in the lower 6
    return (__popcnt(whichPlanes & 0x3F) & 1) == 0;
}

int
Frustum::clipPoly(Vec3f* xyzOut, const Vec3f* xyzIn, int count, int whichPlanes) const
{
    // Set up output buffers (xyz0 is temporary, xyz1 is the user buffer).
    // Each plane can introduce at most one vertex into the polygon. Once we've clipped to 5 planes,
    // which is the last time the temp buffer would be used, the poly size is at most MAX_POINTS + 5.
    Vec3f xyz0[MAX_POINTS + 5], *xyz1 = xyzOut;

    // Choose initial output buffer depending on parity of number of planes we're clipping to
    if (isPlaneCountEven(whichPlanes)) xyzOut = xyz0;

    // Clip to all planes indicated by whichPlanes
    for (int i = 0; i < 6; i++) {
        if (whichPlanes & (1 << i)) {
            count = clipPolyToPlane(xyzOut, xyzIn, count, i);
            if (count == 0) break;
            xyzIn  = xyzOut;
            xyzOut = (xyzOut == xyz0) ? xyz1 : xyz0;
        }
    }

    return count;
}

int
Frustum::clipPoly(Vec3f* xyzOut, Vec2f* stOut, const Vec3f* xyzIn, const Vec2f* stIn, int count, int whichPlanes) const
{
    // Set up output buffers (xyz0/st0 are temporary, xyz1/st1 are the user buffers)
    // Each plane can introduce at most one vertex into the polygon. Once we've clipped to 5 planes,
    // which is the last time the temp buffer would be used, the poly size is at most MAX_POINTS + 5.
    Vec3f xyz0[MAX_POINTS + 5], *xyz1 = xyzOut;
    Vec2f st0 [MAX_POINTS + 5], *st1  = stOut;

    // Choose initial output buffers depending on parity of number of planes we're clipping to
    if (isPlaneCountEven(whichPlanes)) { xyzOut = xyz0, stOut = st0; }

    // Clip to all planes indicated by whichPlanes
    for (int i = 0; i < 6; i++) {
        if (whichPlanes & (1 << i)) {
            count = clipPolyToPlane(xyzOut, stOut, xyzIn, stIn, count, i);
            if (count == 0) break;
            xyzIn  = xyzOut;
            stIn   = stOut;
            xyzOut = (xyzOut == xyz0) ? xyz1 : xyz0;
            stOut  = (stOut  == st0 ) ? st1  : st0;
        }
    }

    return count;
}

void
Frustum::transformClipPlanes(const Mat4d& transform) {

    for (int i = 0; i < 6; ++i) { 
        Vec4d clipPlane(mClipPlanes[i][0], mClipPlanes[i][1], mClipPlanes[i][2], mClipPlanes[i][3]);
        // multiply the clipPlane by the inverse transpose of the transform
        Vec4d res = transform.inverse() * clipPlane;

        // set the new clipping plane coefficients
        mClipPlanes[i][0] = res.x;
        mClipPlanes[i][1] = res.y;
        mClipPlanes[i][2] = res.z;
        mClipPlanes[i][3] = res.w;
    }
}

} // namespace mcrt_common
} // namespace moonray

