# Copyright 2023-2024 DreamWorks Animation LLC
# SPDX-License-Identifier: Apache-2.0

set(target moonray_rendering_pbr_light_tree)

add_executable(${target})

target_sources(${target}
    PRIVATE
        main.cc
        TestLightTree.cc
)

target_include_directories(${target}
    PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
)

target_link_libraries(${target}
    PRIVATE
        Moonray::rendering_rndr
        Moonray::rendering_pbr
        SceneRdl2::pdevunit
)

# Set standard compile/link options
Moonray_cxx_compile_definitions(${target})
Moonray_cxx_compile_features(${target})
Moonray_cxx_compile_options(${target})
Moonray_link_options(${target})

add_test(NAME ${target} COMMAND ${target})
set_tests_properties(${target} PROPERTIES LABELS "unit")
