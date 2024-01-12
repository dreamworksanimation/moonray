// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file Primitive.h
/// $Id$
///

#pragma once

#ifndef GEOM_PRIMITIVE_HAS_BEEN_INCLUDED
#define GEOM_PRIMITIVE_HAS_BEEN_INCLUDED

#include <moonray/rendering/geom/Types.h>
#include <moonray/rendering/geom/prim/BVHHandle.h>
#include <moonray/rendering/geom/prim/EmissionDistribution.h>
#include <moonray/rendering/geom/prim/GridSampler.h>
#include <moonray/rendering/geom/prim/VDBVelocity.h>

#include <moonray/rendering/bvh/shading/Intersection.h>
#include <moonray/rendering/mcrt_common/Ray.h>

#include <scene_rdl2/common/platform/Platform.h>
#include <scene_rdl2/render/util/Memory.h>
#include <scene_rdl2/scene/rdl2/rdl2.h>

#include <openvdb/openvdb.h>
#include <openvdb/Grid.h>

#include <unordered_map>

namespace scene_rdl2 {

namespace rdl2 {
class VolumeShader;
}
}

namespace moonray {

namespace pbr {
class Camera;
}

namespace mcrt_common {
class ThreadLocalState;
struct Frustum;
}

namespace geom {

struct MotionBlurParams;

namespace internal {

class VolumeAssignmentTable;
class VolumeSampleInfo;

/// Tessellation parameters
struct TessellationParams {
    TessellationParams(const scene_rdl2::rdl2::Layer *rdlLayer,
        const std::vector<mcrt_common::Frustum>& frustums,
        const scene_rdl2::math::Mat4d& world2render,
        bool enableDisplacement,
        bool fastGeomUpdate,
        bool isBaking,
        const VolumeAssignmentTable* volumeAssignmentTable) :
            mRdlLayer(rdlLayer), mFrustums(frustums),
            mWorld2Render(world2render),
            mEnableDisplacement(enableDisplacement),
            mFastGeomUpdate(fastGeomUpdate),
            mIsBaking(isBaking),
            mVolumeAssignmentTable(volumeAssignmentTable) {}

    const scene_rdl2::rdl2::Layer *mRdlLayer;
    const std::vector<mcrt_common::Frustum>& mFrustums;
    const scene_rdl2::math::Mat4d& mWorld2Render;
    bool mEnableDisplacement;
    bool mFastGeomUpdate;
    bool mIsBaking;
    const VolumeAssignmentTable* mVolumeAssignmentTable;
};

/// @brief A Primitive is the actual geometry to be rendered.
class Primitive
{
public:
    typedef enum {
        POLYMESH,
        CURVES,
        INSTANCE,
        QUADRIC,
        VDB_VOLUME} PrimitiveType;

    /// Constructor / Destructor
    /// A Primitive stores a geometry primitive to be rendered. The Procedural
    /// creating the primitive must construct / update it passing vertex "P"
    /// and "N" defined in local-space(local space of the primitive).
    Primitive() : mRdlGeometry(nullptr), mIsReference(false), mFeatureSize(1.0f), mDensityColor(1.0f)
    {
        // VDBVelocity stores shutter open and close data, an openvdb velocity grid,
        // and openvdb velocity samplers. It is needed for motion blur when the primitive
        // is bound to a volume shader.
        mVdbVelocity.reset(new VDBVelocity());
    }

    virtual ~Primitive() = default;

    Primitive(const Primitive&) = delete;

    Primitive &operator=(const Primitive&) = delete;

    Primitive& operator=(Primitive&&) = delete;

    /// Query the primitive type
    virtual PrimitiveType getType() const = 0;

    /// Get the memory usage of this Primitive in byte
    virtual size_t getMemory() const;

    /// Get the motion sample count for this primitive
    /// (only support at most 2 motion samples at this moment)
    virtual size_t getMotionSamplesCount() const = 0;

    /// The primitive needs to implement getSubPrimitiveCount,
    /// getBoundsFunction, getIntersectFunction, getOccludedFunction
    /// for BVH as ray tracing kernel if canIntersect return true,
    /// otherwise it needs to implement tessellate to generate sub primitives
    /// renderer can put in BVH for ray tracing
    virtual bool canIntersect() const = 0;

    /// This method tessellates the primitive into sub primitives that
    /// can be ray-traced. (triangles, bezier spans...etc)
    virtual void tessellate(const TessellationParams& tessellationParams)
    {
        MNRY_ASSERT(0, "not implemented");
    }

    /// query the number of sub-primitive (face for mesh, span for curves,
    /// point for points...etc) of this Primitive
    virtual size_t getSubPrimitiveCount() const
    {
        MNRY_ASSERT(0, "not implemented");
        return 0;
    }

    /// Provide the bounding box calculation (for each sub-primitive) callback
    /// if the Primitive is intersectable
    virtual RTCBoundsFunction getBoundsFunction() const
    {
        MNRY_ASSERT(0, "not implemented");
        return nullptr;
    }

    /// Provide the sub-primitive/ray intersection test callback
    /// if the Primitive is intersectable
    virtual RTCIntersectFunctionN getIntersectFunction() const
    {
        MNRY_ASSERT(0, "not implemented");
        return nullptr;
    }

    /// Provide the sub-primitive/ray occlusion test callback
    /// if the Primitive is intersectable
    virtual RTCOccludedFunctionN getOccludedFunction() const
    {
        MNRY_ASSERT(0, "not implemented");
        return nullptr;
    }

    /// This method is responsible for looking up scene_rdl2::rdl2::Material in scene_rdl2::rdl2::Layer
    /// based on ray.primID.
    virtual const scene_rdl2::rdl2::Material *
    getIntersectionMaterial(const scene_rdl2::rdl2::Layer *pRdlLayer,
            const mcrt_common::Ray&ray) const
    {
        MNRY_ASSERT(0, "not implemented");
        return nullptr;
    }

    /// This method is responsible for initializing the Intersection
    /// data structure (except for mP which is initialized automatically
    /// from the ray hit distance).
    virtual void postIntersect(mcrt_common::ThreadLocalState &tls,
            const scene_rdl2::rdl2::Layer *pRdlLayer, const mcrt_common::Ray &ray,
            shading::Intersection &intersection) const
    {
        MNRY_ASSERT(0, "not implemented");
    }

    /// Calculate the intersection curvature dnds, dndt
    /// return false if the curvature of this Primitive can't be calculated
    virtual bool computeIntersectCurvature(const mcrt_common::Ray &ray,
            const shading::Intersection &intersection, Vec3f &dnds, Vec3f &dndt) const
    {
        return false;
    }

    /// Calculate the axis-aligned bounding box
    virtual BBox3f computeAABB() const
    {
        MNRY_ASSERT(0, "not implemented");
        return BBox3f();
    }

    /// Calculate the axis-aligned bounding box at a specified time step
    virtual BBox3f computeAABBAtTimeStep(int timeStep) const
    {
        MNRY_ASSERT(0, "not implemented");
        return BBox3f();
    }

    virtual void initVolumeSampleInfo(VolumeSampleInfo* info,
            const Vec3f& rayOrg, const Vec3f& rayDir, float time,
            const scene_rdl2::rdl2::VolumeShader* volumeShader,
            int volumeId) const = 0;

    /// Query whether thre are multiple frame samples for this Primitive
    bool isMotionBlurOn() const
    {
        return getMotionSamplesCount() > 1;
    }

    /// Query whether the BVH side representation get constructed
    bool isBVHInitialized() const
    {
        return mBVHHandle != nullptr;
    }

    /// Bind the BVH side representation to this corresponding primitive
    void setBVHHandle(std::unique_ptr<BVHHandle>&& bvhHandle)
    {
        MNRY_ASSERT_REQUIRE(bvhHandle.get() != nullptr);
        mBVHHandle = std::move(bvhHandle);
    }

    /// Update BVH side representation after this primitive got deformed
    void updateBVHHandle()
    {
        MNRY_ASSERT_REQUIRE(isBVHInitialized());
        mBVHHandle->update();
    }

    /// Query the geometry ID in BVH for this primitive
    uint32_t getGeomID() const
    {
        return mBVHHandle->getGeomID();
    }

    void setRdlGeometry(const scene_rdl2::rdl2::Geometry* rdlGeometry)
    {
        mRdlGeometry = rdlGeometry;
    }

    const scene_rdl2::rdl2::Geometry* getRdlGeometry() const
    {
        return mRdlGeometry;
    }

    void setIsReference(bool isReferenced) {
        mIsReference = isReferenced;
    }

    bool getIsReference() const {
        return mIsReference;
    }

    float getFeatureSize() const {
        return mFeatureSize;
    }

    void setInstanceFeatureSize(int volumeId, float featureSize) {
        mInstanceFeatureSize[volumeId] = featureSize;
    }

    float getInstanceFeatureSize(int volumeId) const {
        float featureSize = mFeatureSize;
        std::unordered_map<int, float>::const_iterator itr = mInstanceFeatureSize.find(volumeId);
        if (itr != mInstanceFeatureSize.end()) {
            featureSize = itr->second;
        }
        return featureSize;
    }

    virtual std::unique_ptr<EmissionDistribution>
    computeEmissionDistribution(const scene_rdl2::rdl2::VolumeShader* volumeShader) const;

    // query whether the primitive contains specified assignment id
    virtual bool hasAssignment(int assignmentId) const = 0;

    // Bake the density field of the volume shader into a vdb grid for
    // optimized sampling during render time. This accommodates only one
    // volume shader per primitive.
    virtual void bakeVolumeShaderDensityMap(const scene_rdl2::rdl2::VolumeShader* volumeShader,
                                            const scene_rdl2::math::Mat4f& primToRender,
                                            const geom::MotionBlurParams& motionBlurParams,
                                            const VolumeAssignmentTable* volumeAssignmentTable,
                                            const int assignmentId);

    // If this primitive is bound to a volume shader with a map binding, and the
    // primitive exhibits motion blur, we must bake the velocity grid so that the
    // volume shader can be blurred with the primitive. Only Mesh geometries
    // currently support velocity grids.
    // interiorBandwidth is the interior thickness of the velocity grid in voxel units.
    virtual void createVelocityGrid(const float interiorBandwidth,
                                    const geom::MotionBlurParams& motionBlurParams,
                                    const std::vector<int>& volumeIds)
    {
        MNRY_ASSERT(0, "not implemented");
    }

    // This sample position is used for sampling a primitive's vdb grids. It must be in the vdb grids'
    // space. VdbVolume primitives have several vdb grids: e.g. density, albedo, and emission.
    // For VdbVolume, these grids have a unique "grid space", aka local space. Mesh primitives have a
    // "baked density" grid, whose space is conveniently the same as render space.
    virtual scene_rdl2::math::Vec3f evalVolumeSamplePosition(mcrt_common::ThreadLocalState* tls,
                                                 uint32_t volumeId,
                                                 const Vec3f& pSample,
                                                 float time) const;

    // This sample position is used for sampling map shaders. It must be in render space.
    // TODO: use "time" parameter for xform motion blur of "baked density" grid in Mesh Primitives.
    virtual scene_rdl2::math::Vec3f transformVolumeSamplePosition(const Vec3f& pSample, float /* time */) const
    {
        return pSample;
    }

    // Evaluate volume density from volume shader or vdb grid. For VdbVolumes, pSample
    // is in world space. For all other primitives pSample is in render space. This is
    // because the density grid is stored in world space in VdbVolume and it is stored in
    // render space for all other primitives.
    virtual scene_rdl2::math::Color evalDensity(mcrt_common::ThreadLocalState* tls,
                                    uint32_t volumeId,
                                    const Vec3f& pSample,
                                    const float rayVolumeDepth,
                                    const scene_rdl2::rdl2::VolumeShader* const volumeShader) const;

    // Performs voxel value lookup. VdbVolume can have voxel grids for extinction,
    // albedo, and temperature. Meshes have voxel grids only for extinction.
    virtual void evalVolumeCoefficients(mcrt_common::ThreadLocalState* tls,
                                        uint32_t volumeId,
                                        const Vec3f& pSample,
                                        scene_rdl2::math::Color* extinction,
                                        scene_rdl2::math::Color* albedo,
                                        scene_rdl2::math::Color* temperature,
                                        bool highQuality,
                                        const float rayVolumeDepth,
                                        const scene_rdl2::rdl2::VolumeShader* const volumeShader) const;

    virtual scene_rdl2::math::Color evalTemperature(mcrt_common::ThreadLocalState* tls,
                                        uint32_t volumeId,
                                        const Vec3f& pSample) const
    {
        return scene_rdl2::math::Color(1.0f);
    }

protected:
    std::unique_ptr<BVHHandle> mBVHHandle;
    const scene_rdl2::rdl2::Geometry* mRdlGeometry;
    bool mIsReference;

    //////////////////// Baked Volume Shader //////////////////////

    // The minimum distance between significant field value changes.
    // This corresponds to the voxel size of a VDB grid.
    float mFeatureSize;

    VDBSampler<openvdb::Vec3SGrid> mBakedDensitySampler;
    openvdb::Vec3SGrid::Ptr mBakedDensityGrid;
    scene_rdl2::math::Color mDensityColor;

    // velocity field used for advection based motion blur
    std::unique_ptr<VDBVelocity> mVdbVelocity;

    ////////  Instanced Volumes
    std::unordered_map<int, float> mInstanceFeatureSize;
};

} // namespace internal
} // namespace geom
} // namespace moonray

#endif /* GEOM_PRIMITIVE_HAS_BEEN_INCLUDED */

