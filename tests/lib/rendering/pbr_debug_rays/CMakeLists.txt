# Copyright 2023-2024 DreamWorks Animation LLC
# SPDX-License-Identifier: Apache-2.0

set(target moonray_rendering_pbr_debug_rays)

add_executable(${target})

target_sources(${target}
    PRIVATE
        main.cc
        TestDebugRays.cc
)

target_include_directories(${target}
    PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
)

target_link_libraries(${target}
    PRIVATE
        Moonray::common_mcrt_macros
        Moonray::common_mcrt_util
        Moonray::rendering_pbr
        Moonray::rendering_rndr
        SceneRdl2::render_util
        SceneRdl2::common_math
        SceneRdl2::pdevunit
)

# Set standard compile/link options
Moonray_cxx_compile_definitions(${target})
Moonray_cxx_compile_features(${target})
Moonray_cxx_compile_options(${target})
Moonray_link_options(${target})

add_test(NAME ${target} COMMAND ${target})
set_tests_properties(${target} PROPERTIES LABELS "unit")
