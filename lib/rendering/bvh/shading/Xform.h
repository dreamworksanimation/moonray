// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

//

#pragma once

#include <moonray/rendering/bvh/shading/ispc/Xform_ispc_stubs.h>

#include <scene_rdl2/common/math/simd.h>
#include <scene_rdl2/render/util/AlignedAllocator.h>
#include <scene_rdl2/scene/rdl2/rdl2.h>

#include <vector>

namespace moonray {
namespace shading {

template <typename T>
    using Vector = std::vector<T, scene_rdl2::alloc::AlignedAllocator<T, SIMD_MEMORY_ALIGNMENT>>;

typedef std::vector<scene_rdl2::math::Xform3f> XformSamples;

class State;

/**
 * @class Xform
 * @brief Xform, a utility library for shaders - for space transforms.
 *
 * Xform can be used to transform a point, normal or vector from one space
 * to another (refer to each transform function for the list of supported spaces).
 * It encapsulates the transformation matrices and utilizes them during shading.
 * The intended utilization of this class is to create an instance in the shader's
 * update() function. Doing so will set up the right transform data once.
 *
 * In order to use this library at sample time in ISPC, you should a pointer to the data needs
 * to be held onto, in the ISPC shared structure of the shader. The pointer is obtained
 * by calling getIspcXform().
 *
 */

class Xform
{
public:
   /**
    * @brief Construct a Xform object with a custom object, camera or window
    * @param shader Pointer to the current shader's rdl2 dso class
    * @param customObject Override the geometry to use for object space transforms
    *     or pass nullptr to use the object space of the shading point's object
    * @param customCamera Override the camera to use for camera space transforms
    *     or pass nullptr to use the scene's current active camera
    * @param customWindow Override the window to use for screen space transforms
    *     or pass nullptr to use the scene's current aspect ratio to form a window
    *
    */
    Xform(const scene_rdl2::rdl2::SceneObject *shader,
          const scene_rdl2::rdl2::Node *customObject,
          const scene_rdl2::rdl2::Camera *customCamera,
          const std::array<float, 4> *customWindow);
  
   /**
    * @brief Construct a Xform object with an matrix representing
    *     an object or projector's transform
    * @param shader Pointer to the current shader's rdl2 dso class
    * @param objectMatrix 4x4 matrix representing and object
    *     or projection's transform
    */
    Xform(const scene_rdl2::rdl2::SceneObject *shader,
          const scene_rdl2::math::Mat4d& objectMatrix);

    /**
     * @brief Construct a Xform object with default transforms
     *
     * This constructor uses object space of the shading point's object,
     * scene's current active camera and scene's current aspect ratio to form a window
     *
     * @param shader Pointer to the current shader's rdl2 dso class
     */
    explicit Xform(const scene_rdl2::rdl2::SceneObject *shader);

    DISALLOW_COPY_OR_ASSIGNMENT(Xform);

    /**
     * @brief Transform a 3D point from source space to destination space
     *
     * The spaces supported are: Render, World, Camera, Screen and Object
     * @param srcSpace Source space (enum, begins with SHADER_SPACE)
     * @param destSpace Destination space (enum, begins with SHADER_SPACE)
     * @param state Current shading state
     * @param inPoint 3D point to transform
     */
    scene_rdl2::math::Vec3f transformPoint(const int srcSpace,
                               const int dstSpace,
                               const State &state,
                               const scene_rdl2::math::Vec3f inPoint) const;

    /**
     * @brief Transform a normal from source space to destination space
     *
     * The spaces supported are: Render, World, Camera and Object
     * @param srcSpace Source space (enum, begins with SHADER_SPACE)
     * @param destSpace Destination space (enum, begins with SHADER_SPACE)
     * @param state Current shading state
     * @param inNormal 3D point to transform
     */
    scene_rdl2::math::Vec3f transformNormal(const int srcSpace,
                                const int dstSpace,
                                const State &state,
                                const scene_rdl2::math::Vec3f inNormal) const;

    /**
     * @brief Transform a vector from source space to destination space
     *
     * The spaces supported are: Render, World, Camera and Object
     * @param srcSpace Source space (enum, begins with SHADER_SPACE)
     * @param destSpace Destination space (enum, begins with SHADER_SPACE)
     * @param state Current shading state
     * @param inVector 3D point to transform
     */
    scene_rdl2::math::Vec3f transformVector(const int srcSpace,
                                const int dstSpace,
                                const State &state,
                                const scene_rdl2::math::Vec3f inVector) const;

    /**
     * Get a pointer to transform data, required to use the library functions
     * in ISPC during shade.
     */
    const ispc::Xform* getIspcXform() const;

private:
    void precalculateMatrices(const scene_rdl2::rdl2::SceneObject *shader,
                              const scene_rdl2::math::Mat4d& r2oMatrix,
                              const scene_rdl2::rdl2::Camera *customCamera,
                              const std::array<float, 4> *customWindow);

    ispc::Xform mIspc;
};

} // namespace shading
} // namespace moonray

