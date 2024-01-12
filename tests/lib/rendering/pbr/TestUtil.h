// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file TestUtil.h
/// $Id$
///

#pragma once

#include "TestUtil_ispc_stubs.h"

#include <moonray/rendering/pbr/core/Util.h>

#include <moonray/rendering/mcrt_common/ThreadLocalState.h>
#include <moonray/rendering/pbr/core/PbrTLState.h>
#include <moonray/rendering/pbr/sampler/PixelScramble.h>
#include <scene_rdl2/scene/rdl2/rdl2.h>
#include <scene_rdl2/common/math/Color.h>
#include <scene_rdl2/common/math/Mat4.h>
#include <scene_rdl2/common/math/MathUtil.h>
#include <scene_rdl2/common/math/Vec3.h>
#include <scene_rdl2/render/util/AlignedAllocator.h>
#include <scene_rdl2/render/util/Arena.h>
#include <scene_rdl2/render/util/Random.h>
#include <scene_rdl2/render/util/Ref.h>

#include <cppunit/extensions/HelperMacros.h>

#include <tbb/blocked_range2d.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>

#include <cstdarg>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <vector>

namespace moonray {
namespace pbr {

// Un-comment the line below to use CPPUNIT_ASSERT upon failure
#define PBR_TEST_BSDF_CPPUNIT_ASSERT

// Un-comment the line below to print debug values (verbose)
#define PBR_TEST_BSDF_DEBUG

// Uncomment this when we implement savePaMatFile
// #define PBR_SAVE_MAT_FILES

// Enable this to turn off parallel_for threading for ease of debugging.
#if 0
namespace tbb {
    template<typename Range, typename Body>
    finline void
    disableThreadedParallelFor(const Range &range, const Body &body)
    {
        body(range);
    }
}
#define parallel_for    disableThreadedParallelFor
#endif


typedef std::vector<float, scene_rdl2::alloc::AlignedAllocator<float, SIMD_MEMORY_ALIGNMENT>> FloatArray;
typedef std::vector<scene_rdl2::math::Vec3f, scene_rdl2::alloc::AlignedAllocator<scene_rdl2::math::Vec3f,
                                                                                 SIMD_MEMORY_ALIGNMENT>> Vec3Array;
typedef std::vector<scene_rdl2::math::Color, scene_rdl2::alloc::AlignedAllocator<scene_rdl2::math::Color,
                                                                                 SIMD_MEMORY_ALIGNMENT>> ColorArray;


//----------------------------------------------------------------------------

inline void
printInfo(const char *format, ...)
{
    va_list args;
    va_start(args, format);
    vfprintf(stderr, format, args);
    fprintf(stderr, "\n");
    va_end(args);
}


inline void
printDebug(const char *format, ...)
{
#ifdef PBR_TEST_BSDF_DEBUG
    va_list args;
    va_start(args, format);
    vfprintf(stderr, format, args);
    fprintf(stderr, "\n");
    va_end(args);
#endif
}


inline void
printWarning(const char *format, ...)
{
    va_list args;
    va_start(args, format);
    fprintf(stderr, "Warning: ");
    vfprintf(stderr, format, args);
    fprintf(stderr, "\n");
    va_end(args);
}


inline void
printError(const char *format, ...)
{
    va_list args;
    va_start(args, format);
    fprintf(stderr, "Error: ");
    vfprintf(stderr, format, args);
    fprintf(stderr, "\n");
    va_end(args);
    fflush(stderr);
#ifdef PBR_TEST_BSDF_CPPUNIT_ASSERT
    CPPUNIT_ASSERT(0);
#endif
}


inline bool
testAssert(bool condition, const char *format, ...)
{
    bool error = !condition;
    if (error) {
        va_list args;
        va_start(args, format);
        fprintf(stderr, "Error: ");
        vfprintf(stderr, format, args);
        fprintf(stderr, "\n");
        va_end(args);
        fflush(stderr);
#ifdef PBR_TEST_BSDF_CPPUNIT_ASSERT
        CPPUNIT_ASSERT(0);
#endif
    }

    return error;
}


inline bool
asCppBool(bool ispcBool)
{
    int i = ispcBool ? 1 : 0;
    return (bool)i;
}


finline bool
isValidDirection(const scene_rdl2::math::Vec3f &v)
{
    if (!scene_rdl2::math::isFinite(v)) {
        return false;
    }
    return scene_rdl2::math::isNormalized(v);
}


finline bool
isValidColor(const scene_rdl2::math::Color &c)
{
    if (!scene_rdl2::math::isFinite(c)) {
        return false;
    }
    return (c[0] >= 0.0f  &&  c[1] >= 0.0f  &&  c[2] >= 0.0f);
}


finline bool
isValidPdf(float pdf)
{
    return (finite(pdf)  &&  pdf > 0.0f);
}


finline bool
isValidPdfIntegral(float integral, float tolerance)
{
    return scene_rdl2::math::isOne(integral, tolerance);
}


finline bool
isValidEvalIntegral(const scene_rdl2::math::Color &integral, float tolerance)
{
    return (integral - scene_rdl2::math::Color(tolerance) < scene_rdl2::math::Color(1.0f, 1.0f, 1.0f));
}


finline float
computeError(float v1, float v2)
{
    MNRY_ASSERT(v1 >= 0.0f);
    MNRY_ASSERT(v2 >= 0.0f);

    float max = scene_rdl2::math::max(v1, v2);
    if (max > 1.0f) {
        return scene_rdl2::math::abs(v2 - v1) / max;
    } else {
        return scene_rdl2::math::abs(v2 - v1);
    }
}


//----------------------------------------------------------------------------

finline void
save2dTxtFile(const std::string &filename, const FloatArray &u, const FloatArray &v)
{
    MNRY_ASSERT_REQUIRE(u.size() == v.size());

    std::ofstream file;
    file.open(filename.c_str());

    for (size_t i = 0; i < u.size(); i++) {
        file << u[i] << " " << v[i] << " 0.0" << std::endl;
    }

    file.close();
}


#ifdef PBR_SAVE_MAT_FILES

finline int
savePaMatFile(const std::string &filename, const FloatArray &u, const FloatArray &v)
{
    char *name = FX_new_name(filename.c_str(), ".mat");
    if (FX_write_test(name, true) == NULL) {
        // #### Error reporting!!!
        PDI_error("Cannot write to matrix file '%s'\n", name);
        return -1;
    }
    MAT_FILE *fd = MAT_FILE_open(name, "w");
    if (fd == NULL) {
        // #### Error reporting!!!
        MAT_error("Cannot open matrix file '%s'\n", name);
        return -1;
    }

    int particleCount = static_cast<int>(u.size());

    // Write 3 attributes at a time
    int paAttribute[3];
    MAT *mat[3];
    mat[0] = MAT_create(1, particleCount);
    mat[1] = MAT_create(1, particleCount);
    mat[2] = MAT_create(1, particleCount);
    if (mat[0] == NULL  ||  mat[1] == NULL  ||  mat[2] == NULL) {
        MAT_error("Cannot create matrix");
        return -1;
    }


    // Position
    paAttribute[0] = PA_NAME_POSITION_X;
    paAttribute[1] = PA_NAME_POSITION_Y;
    paAttribute[2] = PA_NAME_POSITION_Z;
    for (int i=0; i < particleCount; i++) {
        Vec3d position(u[i], v[i], 0.0);
        mat[0]->data[i] = position[0];
        mat[1]->data[i] = position[1];
        mat[2]->data[i] = position[2];
    }
    MAT_set_name(mat[0], PA_get_attribute_name(paAttribute[0]));
    MAT_set_name(mat[1], PA_get_attribute_name(paAttribute[1]));
    MAT_set_name(mat[2], PA_get_attribute_name(paAttribute[2]));
    if (!MAT_write(fd, mat[0])  ||
        !MAT_write(fd, mat[1])  ||
        !MAT_write(fd, mat[2])) {
        MAT_error("Cannot write matrix");
        return -1;
    }


    MAT_destroy(mat[0]);
    MAT_destroy(mat[1]);
    MAT_destroy(mat[2]);

    MAT_FILE_close(fd);

    return particleCount;
}

#endif  // PBR_SAVE_MAT_FILES

finline bool
isEqualLuminance(const scene_rdl2::math::Color &a, const scene_rdl2::math::Color &b, float eps =
                 float(scene_rdl2::math::sEpsilon))
{
    return scene_rdl2::math::isEqual<float>(scene_rdl2::math::luminance(a), scene_rdl2::math::luminance(b), eps);
}

// Test if the lengths of two vectors are within a certain percentage of each other.
template<typename T> finline bool
isEqualLength(const T &a, const T &b, float eps = float(scene_rdl2::math::sEpsilon))
{
    return scene_rdl2::math::isEqual<float>(a.length(), b.length(), eps);
}

// Test if the angle between two vectors is within a set tolerance.
template<typename T> finline bool
isEqualDirection(const T &a, const T &b, float toleranceInDegrees)
{
    return dot(normalize(a), normalize(b)) > scene_rdl2::math::cos(scene_rdl2::math::deg2rad(toleranceInDegrees));
}

// Test if uvs match. Uvs are assumed wrapped so for example, 0 == 1 == -1 == 2
// and so on.
finline bool
isEqualWrappedUv(const scene_rdl2::math::Vec2f &a, const scene_rdl2::math::Vec2f &b, float eps =
                 float(scene_rdl2::math::sEpsilon))
{
    scene_rdl2::math::Vec2f diff = a - b;
    float diffU = scene_rdl2::math::fmod(diff.x, 1.f);
    float diffV = scene_rdl2::math::fmod(diff.y, 1.f);

    return (scene_rdl2::math::isZero(diffU, eps) || scene_rdl2::math::isOne(diffU, eps) ||
            scene_rdl2::math::isEqual(diffU, -1.f, eps)) &&
           (scene_rdl2::math::isZero(diffV, eps) || scene_rdl2::math::isOne(diffV, eps) ||
            scene_rdl2::math::isEqual(diffV, -1.f, eps));
}


//----------------------------------------------------------------------------

finline scene_rdl2::rdl2::Light *
makeRectLightSceneObject(const char *name,
                         scene_rdl2::rdl2::SceneContext *sc,
                         const scene_rdl2::math::Mat4f &xform,
                         const scene_rdl2::math::Color &color,
                         float width,
                         float height,
                         const char *texturePath,
                         bool normalized = true)
{
    scene_rdl2::rdl2::SceneObject *object = sc->createSceneObject("RectLight", std::string(name));

    object->beginUpdate();
    object->set<scene_rdl2::rdl2::Mat4d>(scene_rdl2::rdl2::Node::sNodeXformKey, scene_rdl2::math::toDouble(xform));
    object->set<scene_rdl2::rdl2::Rgb>("color", color);
    object->set<scene_rdl2::rdl2::Bool>("normalized", normalized);
    object->set<scene_rdl2::rdl2::Float>("width", width);
    object->set<scene_rdl2::rdl2::Float>("height", height);
    if (texturePath) {
        object->set<scene_rdl2::rdl2::String>("texture", scene_rdl2::rdl2::String(texturePath));
    }
    object->endUpdate();

    return object->asA<scene_rdl2::rdl2::Light>();
}

finline scene_rdl2::rdl2::Light *
makeCylinderLightSceneObject(const char *name,
                         scene_rdl2::rdl2::SceneContext *sc,
                         const scene_rdl2::math::Mat4f &xform,
                         const scene_rdl2::math::Color &color,
                         float radius,
                         float height,
                         const char *texturePath,
                         bool normalized = true)
{
    scene_rdl2::rdl2::SceneObject *object = sc->createSceneObject("CylinderLight", std::string(name));

    object->beginUpdate();
    object->set<scene_rdl2::rdl2::Mat4d>(scene_rdl2::rdl2::Node::sNodeXformKey, scene_rdl2::math::toDouble(xform));
    object->set<scene_rdl2::rdl2::Rgb>("color", color);
    object->set<scene_rdl2::rdl2::Bool>("normalized", normalized);
    object->set<scene_rdl2::rdl2::Float>("radius", radius);
    object->set<scene_rdl2::rdl2::Float>("height", height);
    if (texturePath) {
        object->set<scene_rdl2::rdl2::String>("texture", scene_rdl2::rdl2::String(texturePath));
    }
    object->endUpdate();

    return object->asA<scene_rdl2::rdl2::Light>();
}

finline scene_rdl2::rdl2::Light *
makeDiskLightSceneObject(const char *name,
                         scene_rdl2::rdl2::SceneContext *sc,
                         const scene_rdl2::math::Mat4f &xform,
                         const scene_rdl2::math::Color &color,
                         float radius,
                         const char *texturePath,
                         bool normalized = true)
{
    scene_rdl2::rdl2::SceneObject *object = sc->createSceneObject("DiskLight", std::string(name));

    object->beginUpdate();
    object->set<scene_rdl2::rdl2::Mat4d>(scene_rdl2::rdl2::Node::sNodeXformKey, scene_rdl2::math::toDouble(xform));
    object->set<scene_rdl2::rdl2::Rgb>("color", color);
    object->set<scene_rdl2::rdl2::Bool>("normalized", normalized);
    object->set<scene_rdl2::rdl2::Float>("radius", radius);
    if (texturePath) {
        object->set<scene_rdl2::rdl2::String>("texture", scene_rdl2::rdl2::String(texturePath));
    }
    object->endUpdate();

    return object->asA<scene_rdl2::rdl2::Light>();
}

finline scene_rdl2::rdl2::Light *
makeSphereLightSceneObject(const char *name,
                           scene_rdl2::rdl2::SceneContext *sc,
                           const scene_rdl2::math::Mat4f &xform,
                           const scene_rdl2::math::Color &color,
                           float radius,
                           const char *texturePath,
                           bool normalized = true)
{
    scene_rdl2::rdl2::SceneObject *object = sc->createSceneObject("SphereLight", std::string(name));

    object->beginUpdate();
    object->set<scene_rdl2::rdl2::Mat4d>(scene_rdl2::rdl2::Node::sNodeXformKey, scene_rdl2::math::toDouble(xform));
    object->set<scene_rdl2::rdl2::Rgb>("color", color);
    object->set<scene_rdl2::rdl2::Bool>("normalized", normalized);
    object->set<scene_rdl2::rdl2::Float>("radius", radius);
    if (texturePath) {
        object->set<scene_rdl2::rdl2::String>("texture", scene_rdl2::rdl2::String(texturePath));
    }
    object->endUpdate();

    return object->asA<scene_rdl2::rdl2::Light>();
}

finline scene_rdl2::rdl2::Light *
makeSpotLightSceneObject(const char *name,
                         scene_rdl2::rdl2::SceneContext *sc,
                         const scene_rdl2::math::Mat4f &xform,
                         const scene_rdl2::math::Color &color,
                         float lens_radius,
                         float aspect_ratio,
                         float inner_cone_angle,
                         float outer_cone_angle,
                         const char *texturePath,
                         bool normalized = true)
{
    scene_rdl2::rdl2::SceneObject *object = sc->createSceneObject("SpotLight", std::string(name));

    object->beginUpdate();
    object->set<scene_rdl2::rdl2::Mat4d>(scene_rdl2::rdl2::Node::sNodeXformKey, scene_rdl2::math::toDouble(xform));
    object->set<scene_rdl2::rdl2::Rgb>("color", color);
    object->set<scene_rdl2::rdl2::Bool>("normalized", normalized);
    object->set<scene_rdl2::rdl2::Float>("lens radius", lens_radius);
    object->set<scene_rdl2::rdl2::Float>("aspect ratio", aspect_ratio);
    object->set<scene_rdl2::rdl2::Float>("inner cone angle", inner_cone_angle);
    object->set<scene_rdl2::rdl2::Float>("outer cone angle", outer_cone_angle);
    if (texturePath) {
        object->set<scene_rdl2::rdl2::String>("texture", scene_rdl2::rdl2::String(texturePath));
    }
    object->endUpdate();

    return object->asA<scene_rdl2::rdl2::Light>();
}

finline scene_rdl2::rdl2::Light *
makeDistantLightSceneObject(const char *name,
                            scene_rdl2::rdl2::SceneContext *sc,
                            const scene_rdl2::math::Mat4f &xform,
                            const scene_rdl2::math::Color &radiance,
                            float angular_extent)
{
    scene_rdl2::rdl2::SceneObject *object = sc->createSceneObject("DistantLight", std::string(name));

    object->beginUpdate();
    object->set<scene_rdl2::rdl2::Mat4d>(scene_rdl2::rdl2::Node::sNodeXformKey, scene_rdl2::math::toDouble(xform));
    object->set<scene_rdl2::rdl2::Rgb>("color", radiance);
    object->set<scene_rdl2::rdl2::Bool>("normalized", false);
    object->set<scene_rdl2::rdl2::Float>("angular extent", angular_extent);
    object->endUpdate();

    return object->asA<scene_rdl2::rdl2::Light>();
}

finline scene_rdl2::rdl2::Light *
makeEnvLightSceneObject(const char *name,
                        scene_rdl2::rdl2::SceneContext *sc,
                        const scene_rdl2::math::Mat4f &xform,
                        const scene_rdl2::math::Color &radiance,
                        const char *texturePath,
                        bool upperHemisphereOnly)
{
    scene_rdl2::rdl2::SceneObject *object = sc->createSceneObject("EnvLight", std::string(name));

    object->beginUpdate();
    object->set<scene_rdl2::rdl2::Mat4d>(scene_rdl2::rdl2::Node::sNodeXformKey, scene_rdl2::math::toDouble(xform));
    object->set<scene_rdl2::rdl2::Rgb>("color", radiance);
    if (texturePath) {
        object->set<scene_rdl2::rdl2::String>("texture", scene_rdl2::rdl2::String(texturePath));
    }
    object->set<scene_rdl2::rdl2::Bool>("sample upper hemisphere only", upperHemisphereOnly);
    object->endUpdate();

    return object->asA<scene_rdl2::rdl2::Light>();
}

finline scene_rdl2::rdl2::Light *
makeMeshLightSceneObject(const char *name,
                        scene_rdl2::rdl2::SceneContext *sc,
                        const scene_rdl2::math::Mat4f &xform,
                        const scene_rdl2::math::Color &radiance,
                        const char *texturePath,
                        bool normalized = true)
{
    // light
    scene_rdl2::rdl2::SceneObject *object = sc->createSceneObject("MeshLight", std::string(name));
    object->beginUpdate();
    object->set<scene_rdl2::rdl2::Mat4d>(scene_rdl2::rdl2::Node::sNodeXformKey, scene_rdl2::math::toDouble(xform));
    object->set<scene_rdl2::rdl2::Rgb>("color", radiance);

    // image map
    if (texturePath) {
        scene_rdl2::rdl2::SceneObject *imageMap = sc->createSceneObject("ImageMap", "texture");
        imageMap->beginUpdate();
        imageMap->set<scene_rdl2::rdl2::String>("texture", scene_rdl2::rdl2::String(texturePath));
        imageMap->endUpdate();
        object->set("map_shader", imageMap);
    }
    object->endUpdate();

    return object->asA<scene_rdl2::rdl2::Light>();
}


//----------------------------------------------------------------------------

// Wrapper around tbb::parallel_reduce with common plumbing code for iterating
// over a unit square domain. By systematically iterating over the domain we can
// get a more accurate estimate of the integral than just by picking random
// numbers. Fortunately we don't care about aliasing issues here.
template<typename T, typename FUNC>
finline T
doReductionOverUnitSquare(const T &identity, int samplesPerAxis,
        int grainSizePerAxis, FUNC func)
{
    // turn an x, y coordinate into a uniform distributed number within [0, 1)^2
    float scl, ofs;
    scene_rdl2::math::getScaleOffset<float>(0, samplesPerAxis - 1, 0.f, 0.99999f, &scl, &ofs);

    return tbb::parallel_reduce(

        // blocked range
        tbb::blocked_range2d<unsigned>(0u, samplesPerAxis, grainSizePerAxis,
                                       0u, samplesPerAxis, grainSizePerAxis),

        // identity
        identity,

        // body
        [&](const tbb::blocked_range2d<unsigned> &range, const T &current) -> T
        {
            return func(range, current, scl, ofs);
        },

        // reduction
        [](const T &a, const T &b)
        {
            return a + b;
        }
    );
}


finline void
initIspcRange(ispc::Range2d &ispcRange, const tbb::blocked_range2d<unsigned> &tbbRange)
{
    ispcRange.mRowBegin = tbbRange.rows().begin();
    ispcRange.mRowEnd   = tbbRange.rows().end();
    ispcRange.mColBegin = tbbRange.cols().begin();
    ispcRange.mColEnd   = tbbRange.cols().end();
}

//----------------------------------------------------------------------------

finline void
generateGGXNormals(float aRandomness, uint32_t size, Vec3Array& normals)
{
    constexpr std::uint32_t seed = 0xdaedfeeb;
    scene_rdl2::util::Random rand(seed);

    normals.resize(size);
    for (size_t i = 0; i < size; ++i) {
        float r1 = rand.getNextFloat();
        float r2 = rand.getNextFloat();

        float theta = scene_rdl2::math::atan(aRandomness*scene_rdl2::math::sqrt(r1) *
                                             scene_rdl2::math::rsqrt(1.0f - r1)); // GGX
        float phi = 2.0f * scene_rdl2::math::sPi * r2;

        float sinTheta, cosTheta;
        float sinPhi, cosPhi;

        scene_rdl2::math::sincos(theta, &sinTheta, &cosTheta);
        scene_rdl2::math::sincos(phi, &sinPhi, &cosPhi);

        scene_rdl2::math::Vec3f wn;
        wn.x = cosPhi * sinTheta;
        wn.y = sinPhi * sinTheta;
        wn.z = cosTheta;

        normals[i] = wn;
    }
}

finline void
generateWeightedFlakeColors(uint32_t size, ColorArray& colors)
{
    const float oneOverSize = 1.0f / size;
    colors.resize(size);
    for (size_t i = 0; i < size; ++i) {
        colors[i] = scene_rdl2::math::sWhite * oneOverSize;
    }
}

inline void
setupThreadLocalData()
{
    // Create arena block pool which is shared between all threads.
    scene_rdl2::util::Ref<scene_rdl2::alloc::ArenaBlockPool> arenaBlockPool =
        scene_rdl2::util::alignedMallocCtorArgs<scene_rdl2::alloc::ArenaBlockPool>(CACHE_LINE_SIZE);

    mcrt_common::TLSInitParams initParams;
    initParams.mUnitTests = true;
    initParams.mArenaBlockPool = arenaBlockPool.get();
    initParams.initPbrTls = pbr::TLState::allocTls;

    mcrt_common::initTLS(initParams);
}

inline void
cleanupThreadLocalData()
{
    mcrt_common::cleanUpTLS();
}
//----------------------------------------------------------------------------

} // namespace pbr
} // namespace moonray

