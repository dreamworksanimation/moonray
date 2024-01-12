# Copyright 2023-2024 DreamWorks Animation LLC
# SPDX-License-Identifier: Apache-2.0

set(target moonray_rendering_shading_tests)

# ----------------------------------------
# compile some ispc sources to object files
set(objLib ${target}_objlib)
add_library(${objLib} OBJECT)

target_sources(${objLib}
    PRIVATE
        TestHair.ispc
)

file(RELATIVE_PATH relBinDir ${CMAKE_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR})
set_target_properties(${objLib} PROPERTIES
    ISPC_HEADER_SUFFIX _ispc_stubs.h
    ISPC_HEADER_DIRECTORY /${relBinDir}
    ISPC_INSTRUCTION_SETS avx2-i32x8
)

target_link_libraries(${objLib}
    PRIVATE
        Moonray::rendering_shading
        SceneRdl2::common_platform)

# Set standard compile/link options
Moonray_ispc_compile_options(${objLib})
Moonray_link_options(${objLib})
# ----------------------------------------

add_executable(${target})

target_sources(${target}
    PRIVATE
        main.cc
        TestHair.cc
        # pull in our ispc object files
        $<TARGET_OBJECTS:${objLib}>
)

target_include_directories(${target}
    PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
)

target_link_libraries(${target}
    PRIVATE
        Moonray::rendering_shading
        SceneRdl2::common_math
        SceneRdl2::pdevunit
        SceneRdl2::render_util
)

# Set standard compile/link options
Moonray_cxx_compile_definitions(${target})
Moonray_cxx_compile_features(${target})
Moonray_cxx_compile_options(${target})
Moonray_link_options(${target})

add_test(NAME ${target} COMMAND ${target})
set_tests_properties(${target} PROPERTIES LABELS "unit")
add_subdirectory(ispc)
