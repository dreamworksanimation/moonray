# Copyright 2023-2024 DreamWorks Animation LLC
# SPDX-License-Identifier: Apache-2.0

set(ispc_target test_rendering_pbr_common_ispc)

add_library(${ispc_target} STATIC)

target_sources(${ispc_target}
    PRIVATE
        BsdfFactory.ispc
        TestBsdf.ispc
        TestBsdfOneSampler.ispc
        TestUtil.ispc
)

file(RELATIVE_PATH relBinDir ${CMAKE_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR})
set_target_properties(${ispc_target} PROPERTIES
    ISPC_HEADER_SUFFIX _ispc_stubs.h
    ISPC_HEADER_DIRECTORY /${relBinDir}
    ISPC_INSTRUCTION_SETS avx2-i32x8
)

target_include_directories(${ispc_target}
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)

target_link_libraries(${ispc_target}
    PUBLIC
        Moonray::rendering_mcrt_common
        Moonray::rendering_rndr
        Moonray::rendering_mcrt_common
        Moonray::rendering_pbr
        Moonray::rendering_shading
        SceneRdl2::scene_rdl2
        SceneRdl2::common_math
        SceneRdl2::pdevunit
        SceneRdl2::render_util
        TBB::tbb
)

# Set standard compile/link options
Moonray_ispc_compile_options(${ispc_target})
Moonray_link_options(${ispc_target})
# ----------------------------------------

set(cc_target test_rendering_pbr_common)

add_library(${cc_target} STATIC)

target_sources(${cc_target}
    PRIVATE
        TestBsdfvTask.cc
        TestBsdfCommonTaskSampler.cc
        TestBsdfCommonTaskOneSampler.cc
        TestBsdfOneSamplerv.cc
        TestBsdfCommon.cc
)

target_include_directories(${cc_target}
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)

target_link_libraries(${cc_target}
    PUBLIC
        ${ispc_target}
        Moonray::rendering_mcrt_common
        Moonray::rendering_rndr
        Moonray::rendering_mcrt_common
        Moonray::rendering_pbr
        Moonray::rendering_shading
        SceneRdl2::scene_rdl2
        SceneRdl2::common_math
        SceneRdl2::pdevunit
        SceneRdl2::render_util
        TBB::tbb
)

# Set standard compile/link options
Moonray_cxx_compile_definitions(${cc_target})
Moonray_cxx_compile_features(${cc_target})
Moonray_cxx_compile_options(${cc_target})
Moonray_link_options(${cc_target})
