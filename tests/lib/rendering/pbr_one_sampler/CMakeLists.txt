# Copyright 2023-2024 DreamWorks Animation LLC
# SPDX-License-Identifier: Apache-2.0

set(target moonray_rendering_pbr_one_sampler)

# ----------------------------------------
# compile some ispc sources to object files
set(objLib ${target}_objlib)
add_library(${objLib} OBJECT)

target_sources(${objLib}
    PRIVATE
        TestBsdfOneSampler.ispc
)

file(RELATIVE_PATH relBinDir ${CMAKE_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR})
set_target_properties(${objLib} PROPERTIES
    ISPC_HEADER_SUFFIX _ispc_stubs.h
    ISPC_HEADER_DIRECTORY /${relBinDir}
    ISPC_INSTRUCTION_SETS avx2-i32x8
)

target_link_libraries(${objLib}
    PRIVATE
        test_rendering_pbr_common_ispc
        Moonray::rendering_pbr)

# Set standard compile/link options
Moonray_ispc_compile_options(${objLib})
Moonray_link_options(${objLib})
# ----------------------------------------

add_executable(${target})

target_sources(${target}
    PRIVATE
        main.cc
        TestBsdfOneSampler.cc
        # pull in our ispc object files
        $<TARGET_OBJECTS:test_rendering_pbr_common>
        $<TARGET_OBJECTS:test_rendering_pbr_common_ispc>
)

target_include_directories(${target}
    PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
)

target_link_libraries(${target}
    PRIVATE
        Moonray::rendering_mcrt_common
        Moonray::rendering_pbr
        Moonray::rendering_rndr
        Moonray::rendering_shading
        SceneRdl2::common_math
        SceneRdl2::pdevunit
        SceneRdl2::render_util
        SceneRdl2::scene_rdl2
        TBB::tbb
        test_rendering_pbr_common
)

# Set standard compile/link options
Moonray_cxx_compile_definitions(${target})
Moonray_cxx_compile_features(${target})
Moonray_cxx_compile_options(${target})
Moonray_link_options(${target})

add_test(NAME ${target} COMMAND ${target})
set_tests_properties(${target} PROPERTIES LABELS "unit")
