// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file PrimitivePrivateAccess.h
/// $Id$
///
#pragma once

#include <moonray/rendering/bvh/shading/Xform.h>
#include <moonray/rendering/geom/Primitive.h>
#include <moonray/rendering/geom/Procedural.h>
#include <moonray/rendering/geom/SharedPrimitive.h>
#include <moonray/rendering/geom/TransformedPrimitive.h>

namespace moonray {
namespace geom {
namespace internal {

class PrimitivePrivateAccess {
public:
    static void transformPrimitive(geom::Primitive* handle,
            const MotionBlurParams& motionBlurParams,
            const shading::XformSamples& prim2render) {
        handle->transformPrimitive(motionBlurParams, prim2render);
    }

    static Primitive* getPrimitiveImpl(geom::Primitive* handle) {
        return handle->getPrimitiveImpl();
    }

    static void setBVHScene(geom::SharedPrimitive& handle, void* bvhScene) {
        handle.setBVHScene(bvhScene);
    }

    static void* getBVHScene(geom::SharedPrimitive& handle) {
        return handle.getBVHScene();
    }

    static void transformToReference(geom::Procedural* handle) {
        handle->transformToReference();
    }
};

} // namespace internal
} // namespace geom
} // namespace moonray


