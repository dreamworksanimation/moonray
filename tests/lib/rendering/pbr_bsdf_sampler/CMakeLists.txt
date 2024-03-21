# Copyright 2023-2024 DreamWorks Animation LLC
# SPDX-License-Identifier: Apache-2.0

set(target moonray_rendering_pbr_bsdf_sampler)

add_executable(${target})

target_sources(${target}
    PRIVATE
        main.cc
        TestBsdfSampler.cc
        # pull in our common object files
        $<TARGET_OBJECTS:test_rendering_pbr_common>
        $<TARGET_OBJECTS:test_rendering_pbr_common_ispc>
)

target_include_directories(${target}
    PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
)

target_link_libraries(${target}
    PRIVATE
        test_rendering_pbr_common
        test_rendering_pbr_common_ispc
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
Moonray_cxx_compile_definitions(${target})
Moonray_cxx_compile_features(${target})
Moonray_cxx_compile_options(${target})
Moonray_link_options(${target})

add_test(NAME ${target} COMMAND ${target})
set_tests_properties(${target} PROPERTIES LABELS "unit")
