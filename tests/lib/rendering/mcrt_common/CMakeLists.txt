# Copyright 2023-2024 DreamWorks Animation LLC
# SPDX-License-Identifier: Apache-2.0

set(target moonray_rendering_mcrt_common_tests)

add_executable(${target})

target_sources(${target}
    PRIVATE
        AVXTest.cc
        main.cc
        TestAosSoa.cc
)

target_link_libraries(${target}
    PRIVATE
        Moonray::common_time
        Moonray::rendering_mcrt_common
        SceneRdl2::common_math
        SceneRdl2::pdevunit
        SceneRdl2::render_util
        SceneRdl2::scene_rdl2
)

# Set standard compile/link options
Moonray_cxx_compile_definitions(${target})
Moonray_cxx_compile_features(${target})
Moonray_cxx_compile_options(${target})
Moonray_link_options(${target})

add_test(NAME ${target} COMMAND ${target})
set_tests_properties(${target} PROPERTIES LABELS "unit")
