# Copyright 2023-2024 DreamWorks Animation LLC
# SPDX-License-Identifier: Apache-2.0

set(target moonray_rendering_geom_prim_tests)

add_executable(${target})

target_sources(${target}
    PRIVATE
        main.cc
        TestInterpolator.cc
        TestPrimAttr.cc
        TestPrimUtils.cc
)

target_link_libraries(${target}
    PRIVATE
        Moonray::rendering_bvh_shading
        Moonray::rendering_geom
        Moonray::rendering_mcrt_common
        Moonray::rendering_shading
        SceneRdl2::common_math
        SceneRdl2::pdevunit
        SceneRdl2::scene_rdl2
        SceneRdl2::render_util
        TBB::tbb
)

# Set standard compile/link options
Moonray_cxx_compile_definitions(${target})
Moonray_cxx_compile_features(${target})
Moonray_cxx_compile_options(${target})
Moonray_link_options(${target})

# Build the DSOs needed by the tests
moonray_dso_simple(rendering_geom_prim_TestGeometry
    SKIP_INSTALL
    DSO_NAME TestGeometry
    SOURCE_DIR dso/geometry/TestGeometry
    DEPENDENCIES
        Moonray::rendering_geom
        Moonray::rendering_shading
        SceneRdl2::scene_rdl2
)

moonray_dso_simple(rendering_geom_prim_TestMaterial
    SKIP_INSTALL
    DSO_NAME TestMaterial
    SOURCE_DIR dso/material/TestMaterial
    DEPENDENCIES
        Moonray::rendering_geom
        SceneRdl2::scene_rdl2
)

add_test(NAME ${target} COMMAND ${target})
set_tests_properties(${target} PROPERTIES LABELS "unit")
