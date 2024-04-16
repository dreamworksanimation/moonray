// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "EmbreeAccelerator.h"
#include "GeometryManagerExecTracker.h"
#include "gpu/GPUAccelerator.h"

#include <moonray/common/mcrt_util/Average.h>
#include <moonray/rendering/mcrt_common/Frustum.h>
#include <moonray/rendering/geom/MotionBlurParams.h>
#include <moonray/rendering/geom/prim/NamedPrimitive.h>
#include <moonray/rendering/geom/Primitive.h>
#include <moonray/rendering/geom/prim/EmissiveRegion.h>
#include <moonray/rendering/geom/prim/ShadowLinking.h>
#include <scene_rdl2/common/grid_util/RenderPrepStats.h>
#include <scene_rdl2/common/math/Xform.h>
#include <scene_rdl2/common/platform/Platform.h>
#include <scene_rdl2/scene/rdl2/SceneContext.h>
#include <scene_rdl2/scene/rdl2/Layer.h>

#include <tbb/concurrent_unordered_set.h>
#include <functional>

namespace moonray {

namespace geom {
class BakedMesh;
class BakedCurves;

namespace internal {
class VolumeAssignmentTable;
}
}

namespace rt {

enum class ChangeFlag;

typedef tbb::concurrent_unordered_map<geom::internal::Primitive *, tbb::atomic<unsigned int>> PrimitiveReferenceCountMap;

struct GeometryManagerStats
{
    std::function<void (const std::string& str)> logString;
    std::function<void (const std::string& str)> logDebugString;
    std::function<void (const std::string& sceneClass, const std::string& name)> logGeneratingProcedurals;
    util::AverageDouble mTessellationTime;
    util::AverageDouble mBuildAcceleratorTime;
    util::AverageDouble mBuildGPUAcceleratorTime;
    double mBuildProceduralTime;
    double mRtcCommitTime;
    std::vector<std::pair<geom::internal::NamedPrimitive*, double> > mPerPrimitiveTessellationTime;

    GeometryManagerExecTracker mGeometryManagerExecTracker;

    void reset()
    {
        mTessellationTime.reset();
        mBuildAcceleratorTime.reset();
        mBuildGPUAcceleratorTime.reset();
        mBuildProceduralTime = 0.0;
        mRtcCommitTime = 0.0;
        mPerPrimitiveTessellationTime.clear();

        mGeometryManagerExecTracker.initLoadGeometries(0);
        mGeometryManagerExecTracker.initFinalizeChange(0);
    }
};

/// Various options for GeometryManager.
struct
GeometryManagerOptions
{
    GeometryManagerStats stats;
    AcceleratorOptions accelOptions;
};

/**
 * GeometryManager manages all geometries related to ray tracing. It take cares
 * of loading procedurals for geometries, handling and propagate updates for
 * geometry data. It manages an acceleration data structure for ray intersection
 * queries.
 */
class GeometryManager
{
public:
    typedef std::vector<const std::vector<float>*> VertexBufferArray;

    enum class GM_RESULT // loadGeometries(), finalizeChanges() execution result
    {
        CANCELED, // canceled middle of the renderPrep phase
        FINISHED  // renderPrep phase has been completed
    };

    /// Requires a SceneContext and Layer to access geometry data.
    GeometryManager(scene_rdl2::rdl2::SceneContext* sceneContext,
                    const GeometryManagerOptions& options);

    ~GeometryManager() = default;

    /// Load geometries and their procedurals from the SceneContext and Layer.
    GM_RESULT loadGeometries(scene_rdl2::rdl2::Layer* layer, const ChangeFlag flag,
                             const scene_rdl2::math::Mat4d& world2render,
                             const int currentFrame,
                             const geom::MotionBlurParams& motionBlurParams,
                             const unsigned threadCount,
                             const shading::PerGeometryAttributeKeySet &perGeometryAttributes);

    /// Updates the geometry data with new mesh name and vertex buffers.
    void updateGeometryData(scene_rdl2::rdl2::Layer* layer,
            scene_rdl2::rdl2::Geometry* geometry,
            const std::vector<std::string>& meshNames,
            const VertexBufferArray& vertexBuffers,
            const scene_rdl2::math::Mat4d& world2render,
            const int currentFrame,
            const geom::MotionBlurParams& motionBlurParams,
            const unsigned threadCount);

    void bakeGeometry(scene_rdl2::rdl2::Layer* layer,
            const geom::MotionBlurParams& motionBlurParams,
            const std::vector<mcrt_common::Frustum>& frustums,
            const scene_rdl2::math::Mat4d& world2render,
            std::vector<std::unique_ptr<geom::BakedMesh>>& bakedMeshes,
            std::vector<std::unique_ptr<geom::BakedCurves>>& bakedCurves,
            const scene_rdl2::rdl2::Camera* globalDicingCamera);

    /// Client must call this method to commit all the changes in geometries
    /// before ray tracing.
    GM_RESULT finalizeChanges(scene_rdl2::rdl2::Layer* layer,
                              const geom::MotionBlurParams& motionBlurParams,
                              const std::vector<mcrt_common::Frustum>& frustums,
                              const scene_rdl2::math::Mat4d& world2render,
                              OptimizationTarget accelMode,
                              const scene_rdl2::rdl2::Camera* dicingCamera,
                              bool updateSceneBVH);

    void updateShadowLinkings(const scene_rdl2::rdl2::Layer* layer);

    finline bool isGPUEnabled() const
    {
        return mGPUAccelerator != nullptr;
    }

    void getEmissiveRegions(const scene_rdl2::rdl2::Layer* layer,
            std::vector<geom::internal::EmissiveRegion>& emissiveRegions) const;

    /// Returns the internal state whether there are changes pending and not
    /// finalized.
    finline bool isValid() const
    {
        return mChangeStatus == ChangeFlag::NONE;
    }

    // This is for a call back function in order to report renderPrep progress information
    // to the downstream computation. This functionality is only used under arras context.
    using RenderPrepStatsCallBack = std::function<void(const scene_rdl2::grid_util::RenderPrepStats &rPrepStats)>;
    using RenderPrepCancelCallBack = std::function<bool()>;
    void setStageIdAndCallBackLoadGeometries(int stageId,
                                             const RenderPrepStatsCallBack &statsCallBack,
                                             const RenderPrepCancelCallBack& cancelCallBack) {
        mOptions.stats.mGeometryManagerExecTracker.initLoadGeometries(stageId);
        mOptions.stats.mGeometryManagerExecTracker.setRenderPrepStatsCallBack(statsCallBack);
        mOptions.stats.mGeometryManagerExecTracker.setRenderPrepCancelCallBack(cancelCallBack);
    }
    void setStageIdAndCallBackFinalizeChange(int stageId,
                                             const RenderPrepStatsCallBack &statsCallBack,
                                             const RenderPrepCancelCallBack &cancelCallBack) {
        mOptions.stats.mGeometryManagerExecTracker.initFinalizeChange(stageId);
        mOptions.stats.mGeometryManagerExecTracker.setRenderPrepStatsCallBack(statsCallBack);
        mOptions.stats.mGeometryManagerExecTracker.setRenderPrepCancelCallBack(cancelCallBack);
    }
    const GeometryManagerStats& getStatistics() const
    {
        return mOptions.stats;
    }

    void resetStatistics()
    {
        mOptions.stats.reset();
    }

    finline void setChangeFlag(ChangeFlag flag)
    {
        mChangeStatus = flag;
    }

    finline void compareAndSwapFlag(ChangeFlag swapFlag, ChangeFlag compareFlag)
    {
        mChangeStatus.compare_and_swap(swapFlag, compareFlag);
    }

    void updateGPUAccelerator(bool allowUnsupportedFeatures, const scene_rdl2::rdl2::Layer* layer);

    const geom::internal::VolumeAssignmentTable * getVolumeAssignmentTable() const
    {
        return mVolumeAssignmentTable.get();
    }

    finline const rt::EmbreeAccelerator* getEmbreeAccelerator() const
    {
        return mEmbreeAccelerator.get();
    }

    finline const rt::GPUAccelerator* getGPUAccelerator() const
    {
        return mGPUAccelerator.get();
    }

    GeometryManagerExecTracker &getGeometryManagerExecTracker() { return mOptions.stats.mGeometryManagerExecTracker; }


    // Gets the frustums and transform from the dicing camera, returns whether a dicing camera exists
    bool getDicingCameraFrustums(std::vector<mcrt_common::Frustum>* frustums,
                                 scene_rdl2::math::Mat4d* dicingWorld2Render,
                                 const scene_rdl2::rdl2::Camera* globalDicingCamera,
                                 const geom::internal::NamedPrimitive* prim,
                                 const scene_rdl2::math::Mat4d& mainWorld2Render);

private:

    /// Tessellates the Geometries in the provided GeometrySets
    GM_RESULT tessellate(scene_rdl2::rdl2::Layer* layer,
                         geom::InternalPrimitiveList& primitivesToTessellate,
                         const std::vector<mcrt_common::Frustum>& frustums,
                         const scene_rdl2::math::Mat4d& world2render,
                         const geom::MotionBlurParams& motionBlurParams,
                         const scene_rdl2::rdl2::Camera* globalDicingCamera);

    /// Add/update geometries in the provided GeometrySets to the
    /// spatial accelerator
    void updateAccelerator(const scene_rdl2::rdl2::Layer* layer,
            const scene_rdl2::rdl2::SceneContext::GeometrySetVector& geometrySets,
            const scene_rdl2::rdl2::Layer::GeometryToRootShadersMap& g2s,
            OptimizationTarget accelMode);

    // Private Data
private:
    /// EXTERNAL REFERENCE (Not Owned by GeometryManager)
    /// Keep a reference to rdl2 SceneContext for queries.
    scene_rdl2::rdl2::SceneContext* mSceneContext;

    /// INTERNAL DATA
    typedef tbb::concurrent_unordered_set<const scene_rdl2::rdl2::GeometrySet*> TbbSetOfGeometrySet;

    /// The set of GeometrySet that get deformed during an update cycle, and
    /// which require a BVH update
    TbbSetOfGeometrySet mDeformedGeometrySets;

    // The Embree spatial accelerator for ray intersection.
    std::unique_ptr<EmbreeAccelerator> mEmbreeAccelerator;

    // The GPU spatial accelerator for ray intersection.
    std::unique_ptr<GPUAccelerator> mGPUAccelerator;

    GeometryManagerOptions mOptions;

    typedef tbb::atomic<ChangeFlag> ChangeFlagAtomic;

    /// Tracks current change status
    ChangeFlagAtomic mChangeStatus;

    std::unique_ptr<geom::internal::VolumeAssignmentTable> mVolumeAssignmentTable;
};

// For use with shadow suppression between specified geometries
struct LabeledGeometrySet {
    std::vector<const scene_rdl2::rdl2::Geometry *> mGeometries;
    std::string mLabel;
};

struct ShadowExclusionMapping {
    std::unordered_set<int> mCasterParts;
    std::unordered_set<LabeledGeometrySet *> mLabeledGeoSets;
};


} // namespace rt
} // namespace moonray

