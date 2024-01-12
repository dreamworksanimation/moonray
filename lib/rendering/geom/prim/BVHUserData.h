// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

//
/// @file BVHUserData.h
/// $Id$
///

#pragma once

#include "Primitive.h"
#include <embree4/rtcore.h>

namespace moonray {
namespace geom {
namespace internal {

// @brief BVHUserData is the user data we pass to embree
struct BVHUserData {
public:
    // @brief The intersection Filter Manager is a container for all intersection
    // filters for a primitive.
    class IntersectionFilterManager {
    public:
        IntersectionFilterManager() {}
        ~IntersectionFilterManager() {}

        /// static filter functions that are registered with embree.

        static void intersectionFilter(const RTCFilterFunctionNArguments* args)
        {
            const BVHUserData* userData =
                (const BVHUserData*)args->geometryUserPtr;
            const IntersectionFilterManager* fm = userData->mFilterManager;
            for (auto& filter : fm->mIntersectionFilters) {
                filter(args);
            }
        }

        static void occlusionFilter(const RTCFilterFunctionNArguments* args)
        {
            const BVHUserData* userData =
                (const BVHUserData*)args->geometryUserPtr;
            const IntersectionFilterManager* fm = userData->mFilterManager;
            for (auto& filter : fm->mOcclusionFilters) {
                filter(args);
            }
        }

        /// add filter function to list of filter functions
        void addIntersectionFilter(RTCFilterFunctionN filterFunction)
        {
            mIntersectionFilters.push_back(filterFunction);
        }

        void addOcclusionFilter(RTCFilterFunctionN filterFunction)
        {
            mOcclusionFilters.push_back(filterFunction);
        }

        /// filter function lists
        std::vector<RTCFilterFunctionN> mIntersectionFilters;
        std::vector<RTCFilterFunctionN> mOcclusionFilters;
    };

    BVHUserData(const scene_rdl2::rdl2::Layer* layer, const Primitive* primitive,
             const IntersectionFilterManager* filterManager) :
        mLayer(layer), mPrimitive(primitive), mFilterManager(filterManager)
    {
        // should have layer and primitive but instances do not have
        // Filter Managers
        MNRY_ASSERT(layer);
        MNRY_ASSERT(primitive);
    }

    ~BVHUserData()
    {
        delete mFilterManager;
    }

    const scene_rdl2::rdl2::Layer* mLayer;
    const Primitive* mPrimitive;
    const IntersectionFilterManager* mFilterManager;
};

} // namespace internal
} // namespace geom
} // namespace moonray


