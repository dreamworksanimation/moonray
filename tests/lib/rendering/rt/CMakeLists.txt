# Copyright 2023-2024 DreamWorks Animation LLC
# SPDX-License-Identifier: Apache-2.0

set(target moonray_rendering_rt_tests)

add_executable(${target})

target_sources(${target}
    PRIVATE
        main.cc
        test_rt.cc
)

target_link_libraries(${target}
    PRIVATE
        Moonray::rendering_geom
        Moonray::rendering_pbr
        Moonray::rendering_rndr
        Moonray::rendering_rt
        Moonray::rendering_shading
        SceneRdl2::pdevunit
        SceneRdl2::scene_rdl2
        TBB::tbb
)

# Set standard compile/link options
Moonray_cxx_compile_definitions(${target})
Moonray_cxx_compile_features(${target})
Moonray_cxx_compile_options(${target})
Moonray_link_options(${target})

# Build the DSOs needed by the tests
moonray_dso_simple(TestRtGeometry
    SKIP_INSTALL
    SOURCE_DIR dso/geometry/TestRtGeometry
    DEPENDENCIES
        Moonray::rendering_geom
        SceneRdl2::scene_rdl2
)
moonray_dso_simple(TestRtMaterial
    SKIP_INSTALL
    SOURCE_DIR dso/material/TestRtMaterial
    DEPENDENCIES SceneRdl2::scene_rdl2
)

add_test(NAME ${target} COMMAND ${target})
set_tests_properties(${target} PROPERTIES LABELS "unit")
