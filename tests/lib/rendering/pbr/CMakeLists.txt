# Copyright 2023-2024 DreamWorks Animation LLC
# SPDX-License-Identifier: Apache-2.0

set(target moonray_rendering_pbr_tests)

# ----------------------------------------
# compile some ispc sources to object files
set(objLib ${target}_objlib)
add_library(${objLib} OBJECT)

target_sources(${objLib}
    PRIVATE
        BsdfFactory.ispc
        TestBsdf.ispc
        TestBsdfOneSampler.ispc
        TestBssrdf.ispc
        TestDistribution.ispc
        TestLightSetSampler.ispc
        TestLights.ispc
        TestLightUtil.ispc
        TestSampler.ispc
        TestUtil.ispc
)

file(RELATIVE_PATH relBinDir ${CMAKE_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR})
set_target_properties(${objLib} PROPERTIES
    ISPC_HEADER_SUFFIX _ispc_stubs.h
    ISPC_HEADER_DIRECTORY /${relBinDir}
    ISPC_INSTRUCTION_SETS avx2-i32x8
)

target_link_libraries(${objLib}
    PRIVATE
        Moonray::rendering_bvh_shading
        Moonray::rendering_pbr)

# Set standard compile/link options
Moonray_ispc_compile_options(${objLib})
Moonray_link_options(${objLib})
# ----------------------------------------

add_executable(${target})

target_sources(${target}
    PRIVATE
        main.cc
        TestBsdfCommon.cc
        TestBsdfCommonTaskOneSampler.cc
        TestBsdfCommonTaskSampler.cc
        TestBsdfOneSampler.cc
        TestBsdfOneSamplerv.cc
        TestBsdfSampler.cc
        TestBsdfv.cc
        TestBsdfvTask.cc
        TestBssrdf.cc
        TestDebugRays.cc
        TestDistribution.cc
        TestLights.cc
        TestLightSetSampler.cc
        TestLightTree.cc
        TestLightUtil.cc
        TestSampler.cc
        TestSceneContext.cc
        # pull in our ispc object files
        $<TARGET_OBJECTS:${objLib}>
)

target_include_directories(${target}
    PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
)

target_link_libraries(${target}
    PRIVATE
        Moonray::common_mcrt_macros
        Moonray::common_mcrt_util
        Moonray::common_time
        Moonray::rendering_bvh_shading
        Moonray::rendering_geom
        Moonray::rendering_mcrt_common
        Moonray::rendering_pbr
        Moonray::rendering_rndr
        Moonray::rendering_shading
        SceneRdl2::common_except
        SceneRdl2::common_fb_util
        SceneRdl2::common_math
        SceneRdl2::pdevunit
        SceneRdl2::render_util
        SceneRdl2::scene_rdl2
        TBB::tbb
)

# Set standard compile/link options
Moonray_cxx_compile_definitions(${target})
Moonray_cxx_compile_features(${target})
Moonray_cxx_compile_options(${target})
Moonray_link_options(${target})

target_link_libraries(${target} PRIVATE atomic)

# ---------------------------------------------------
# This test needs to load certain DSOs at runtime
set(dsoTargets
        CylinderLight
        DiskLight
        DistantLight
        EnvLight
        ImageMap
        MeshLight
        RectLight
        SphereLight
        SpotLight
)

# Make sure the DSOs are built when this target is built
add_dependencies(${target} ${dsoTargets})

# We need to provide the path to the DSOs this test requires at runtime.
# To enable the tests to run after building but before installing, we
# generate a list of colon-separated paths to the BINARY_DIR of each DSO.
set(pathList "")
foreach(dsoTarget ${dsoTargets})
    get_target_property(binDir ${dsoTarget} BINARY_DIR)
    list(APPEND pathList ${binDir})
endforeach()
list(JOIN pathList ":" rdl2DsoPath)

# The final path is passed via compiler definition to the sources
target_compile_definitions(${target}
    PRIVATE
        RDL2DSO_PATH="${rdl2DsoPath}"
)
# ---------------------------------------------------

add_test(NAME ${target} COMMAND ${target})
set_tests_properties(${target} PROPERTIES LABELS "unit")
