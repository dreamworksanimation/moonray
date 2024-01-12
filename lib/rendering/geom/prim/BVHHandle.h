// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file BVHHandle.h
/// $Id$
///

#pragma once

#include <embree4/rtcore.h>

namespace moonray {
namespace geom {
namespace internal {

// BVHHandle is the embree side representation of a Primitive, it provides
// functionality to remove/update a Primitive in embree BVH. Each Primitive
// contains a BVHHandle after BVH is constructed and the life cycle of
// the embree representation is bound to its corresponding Primitive
class BVHHandle {
public:

    BVHHandle(RTCScene& parentScene, uint32_t geomID):
        mParentScene(parentScene), mGeomID(geomID) {}

    ~BVHHandle() {
        if (mParentScene != nullptr && mGeomID != RTC_INVALID_GEOMETRY_ID) {
            rtcDetachGeometry(mParentScene, mGeomID);
            rtcReleaseGeometry(rtcGetGeometry(mParentScene, mGeomID));
        }
        mParentScene = nullptr;
        mGeomID = RTC_INVALID_GEOMETRY_ID;
    }

    void update() {
        if (mParentScene != nullptr && mGeomID != RTC_INVALID_GEOMETRY_ID) {
            rtcCommitGeometry(rtcGetGeometry(mParentScene, mGeomID));
        }
    }

    uint32_t getGeomID() const { return mGeomID; }

private:
    RTCScene mParentScene;
    uint32_t mGeomID;
};
 
} // namespace internal
} // namespace geom
} // namespace moonray

