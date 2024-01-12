// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file BakeCamera.h

#pragma once

#include "Camera.h"

#include <moonray/rendering/texturing/sampler/TextureSampler.h>

#include <scene_rdl2/common/math/Mat3.h>
#include <scene_rdl2/common/math/Mat4.h>
#include <scene_rdl2/common/math/Vec2.h>
#include <scene_rdl2/common/math/Vec3.h>
#include <scene_rdl2/common/math/Vec4.h>
#include <scene_rdl2/common/math/Vec3fa.h>
#include <scene_rdl2/scene/rdl2/Types.h>

#include <string>

namespace moonray {

namespace mcrt_common { class RayDifferential; }

namespace pbr {

class BakeCamera : public Camera
{
public:
    explicit BakeCamera(const scene_rdl2::rdl2::Camera *rdlCamera);
    ~BakeCamera();

private:
    // use OIIO to sample explicitly supplied normal maps.  if we
    // need to take shade time normal mapping into account, we will
    // need to use a two pass solution.  the first pass creates the normal
    // map, the second pass will use the normal map to determine directions.
    class NormalMap
    {
    public:
        NormalMap(): mFilename(""), mTextureHandle(nullptr), mTextureSystem(nullptr)
        {
        }

        std::string mFilename;
        texture::TextureHandle *mTextureHandle;
        OIIO::TextureSystem *mTextureSystem;
    };

    // These need to match up with the enum declared in
    // dso/camera/BakeCamera/attributes.cc
    enum Mode {
        MODE_FROM_CAMERA_TO_SURFACE = 0,
        MODE_FROM_SURFACE_ALONG_NORMAL,
        MODE_FROM_SURFACE_ALONG_REFLECTION_VECTOR,
        MODE_ABOVE_SURFACE_REVERSE_NORMAL
    };

    bool getIsDofEnabledImpl() const override;
    void updateImpl(const scene_rdl2::math::Mat4d &world2render) override;
    void createRayImpl(mcrt_common::RayDifferential *dstRay,
                       float x,
                       float y,
                       float time,
                       float lensU,
                       float lensV) const override;
    void bakeUvMapsImpl() override;
    void getRequiredPrimAttributesImpl(shading::PerGeometryAttributeKeySet &keys) const override;

    // interpolate the position map
    bool interpolatePosMap(const scene_rdl2::math::Vec2f &uv, scene_rdl2::math::Vec3f &posResult,
                           scene_rdl2::math::Vec3f &nrmResult) const;

    // has the user provided a valid normal map
    bool haveNormalMap() const { return mNormalMap.mTextureHandle; }

    // does our baking mode require a normal?
    bool needNormals() const { return getRdlCamera()->get(mAttrMode); }

    // do we require the mNrmMap? (i.e. baked normals).
    bool needNrmMap() const
    {
        // we need to compute state shading normals (nrm  map)
        // if we need normals, don't have a user supplied normal map or
        // the user supplied normal map is specified in tangent space.
        return needNormals() &&
            (!haveNormalMap() ||
             getRdlCamera()->get(mAttrNormalMapSpace) == 1 /* tangent space */);
    }

    bool computeDpdu(const scene_rdl2::math::Vec3f &P, const scene_rdl2::math::Vec2f &uv,
                     scene_rdl2::math::Vec3f &dpdu) const;

    // Utility Functions
    // Transform the normal and the point on the geometry if the geometry is moving:
    void xformPoint(float time,
                    scene_rdl2::math::Vec3f& P,
                    scene_rdl2::math::Vec3f& N) const;
    // Create Rays from surface along reflection vector
    void fromSurfaceAlongReflectionVector(float bias,
                                          const scene_rdl2::math::Vec3f& cameraPos,
                                          const scene_rdl2::math::Vec3f& P,
                                          const scene_rdl2::math::Vec3f& N,
                                          scene_rdl2::math::Vec3f& rayOrg,
                                          scene_rdl2::math::Vec3f& rayDir) const;
    // Create Rays from Camera to Surface
    void fromCameraToSurface(float bias,
                             const scene_rdl2::math::Vec3f& cameraPos,
                             const scene_rdl2::math::Vec3f& P,
                             const scene_rdl2::math::Vec3f& N,
                             scene_rdl2::math::Vec3f& rayOrg,
                             scene_rdl2::math::Vec3f& rayDir) const;
    // Create Rays from Surface Along Normals
    void fromSurfaceAlongNormal(float bias,
                                const scene_rdl2::math::Vec3f& P,
                                const scene_rdl2::math::Vec3f& N,
                                scene_rdl2::math::Vec3f& rayOrg,
                                scene_rdl2::math::Vec3f& rayDir) const;
    // Create Rays from Above Surface pointing towards reverse normals
    void fromAboveSurfaceNormal(float bias,
                                const scene_rdl2::math::Vec3f& P,
                                const scene_rdl2::math::Vec3f& N,
                                scene_rdl2::math::Vec3f& rayOrg,
                                scene_rdl2::math::Vec3f& rayDir) const;

    // dimenions of map we are creating
    unsigned int mWidth;
    unsigned int mHeight;

    // position map is the fundamental structure that maps
    // uv coordinates on the geomety to 3D camera space positions.
    // The nrm map maps uv coordinates to normals in camera space.
    // This map does not take material normal mapping into account, but
    // does respect displacement and explicitly provided mesh normals.
    //
    // Not all uv coordinates exist on a piece of geometry.  For any
    // pixel p, mPosMap[p].w is used to determine if geometry exists
    // at the corresponding uv coordinate.  If non-zero, geometry
    // exists, if 0, then geometry does not exist.  For uv coordinates
    // where geometry does not exist, an invalid ray is generated, and
    // the integrator is expected to handle this.
    scene_rdl2::math::Vec3fa *mPosMap;
    scene_rdl2::math::Vec3f  *mNrmMap;

    // position map width and height - by default, equal to the
    // dimenions of the map we are creating.  but this can be adjusted
    // with the "map factor" attribute.
    unsigned int mPosMapWidth;
    unsigned int mPosMapHeight;

    // explicitly supplied normal map
    NormalMap mNormalMap;

    // cached transformations used to transfrom the outgoing rays according to the geometry
    scene_rdl2::math::Mat4d mRenderToGeom;
    scene_rdl2::math::Mat4d mWorldToRender;

    // cached decomposition, used to more efficiently interpolate the geometry position.
    scene_rdl2::math::XformComponent<scene_rdl2::math::Mat3d> mGeomToWorldOpen;
    scene_rdl2::math::XformComponent<scene_rdl2::math::Mat3d> mGeomToWorldClose;

    // rdl attribute keys
    scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::SceneObject *> mAttrGeometry;
    scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Int> mAttrUdim;
    scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::String> mAttrUvAttribute;
    scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Int> mAttrMode;
    scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float> mAttrBias;
    scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Bool> mAttrUseRelativeBias;
    scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float> mAttrMapFactor;
    scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::String> mAttrNormalMap;
    scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Int> mAttrNormalMapSpace;
};

} // namespace pbr
} // namespace moonray

