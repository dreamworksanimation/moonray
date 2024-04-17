# Copyright 2023-2024 DreamWorks Animation LLC
# SPDX-License-Identifier: Apache-2.0

# ###################################################
# moonray_rendering_shading_ispc_tests
# ###################################################
set(target moonray_rendering_shading_ispc_tests)

# ----------------------------------------
# compile some ispc sources to object files
set(objLib ${target}_objlib)
add_library(${objLib} OBJECT)

target_sources(${objLib}
    PRIVATE
        TestPrimitiveAttribute.ispc
)

file(RELATIVE_PATH relBinDir ${CMAKE_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR})
set_target_properties(${objLib} PROPERTIES
    ISPC_HEADER_SUFFIX _ispc_stubs.h
    ISPC_HEADER_DIRECTORY /${relBinDir}
    ISPC_INSTRUCTION_SETS ${GLOBAL_ISPC_INSTRUCTION_SETS}
    LINKER_LANGUAGE CXX
)

target_link_libraries(${objLib}
    PRIVATE
        Moonray::shading_ispc
        SceneRdl2::common_math
        SceneRdl2::render_util)

# Set standard compile/link options
Moonray_ispc_compile_options(${objLib})
Moonray_link_options(${objLib})

get_target_property(objLibDeps ${objLib} DEPENDENCY)
if(NOT objLibDeps STREQUAL "")
    add_dependencies(${objLibDeps} 
        Moonray::shading_ispc
        SceneRdl2::common_math
        SceneRdl2::render_util
    )
endif()
# ----------------------------------------

add_executable(${target})

get_target_property(ISPC_TARGET_OBJECTS ${objLib} TARGET_OBJECTS)
target_sources(${target}
    PRIVATE
        main.cc
        TestBase.cc
        TestDisplace.cc
        TestEvalAttribute.cc
        TestPrimitiveAttribute.cc
        # pull in our ispc object files
        ${ISPC_TARGET_OBJECTS}
)

target_include_directories(${target}
    PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
)

target_link_libraries(${target}
    PRIVATE
        Moonray::common_mcrt_macros
        Moonray::rendering_bvh_shading
        Moonray::rendering_mcrt_common
        Moonray::rendering_rndr
        Moonray::rendering_shading
        Moonray::texturing_sampler
        Moonray::shading_ispc
        SceneRdl2::common_math
        SceneRdl2::pdevunit
        SceneRdl2::scene_rdl2
        SceneRdl2::render_util

        $<$<AND:$<CXX_COMPILER_ID:GNU>,$<VERSION_GREATER_EQUAL:$<CXX_COMPILER_VERSION>,6>>:
            atomic
        >

        $<$<CXX_COMPILER_ID:Clang>:
            atomic
        >
)

add_dependencies(${target} ${objLib})

# Set standard compile/link options
Moonray_cxx_compile_definitions(${target})
Moonray_cxx_compile_features(${target})
Moonray_cxx_compile_options(${target})
Moonray_link_options(${target})

# Build the DSOs needed by the tests
function(make_test_dso name dir)
    moonray_ispc_dso(${name}
        SKIP_INSTALL TEST_DSO
        SOURCE_DIR ${dir}
        DEPENDENCIES
            Moonray::shading_ispc
            SceneRdl2::scene_rdl2
            SceneRdl2::render_util)
    if(CMAKE_XCODE_BUILD_SYSTEM)
        add_dependencies(${name}_objlib_ispc_dep ${objLib})
    endif()
endfunction()

make_test_dso(TestDisplacement dso/displacement/TestDisplacement)
make_test_dso(TestCheckerMap dso/map/TestCheckerMap)
make_test_dso(TestDebugMap dso/map/TestDebugMap)
make_test_dso(TestMandelbrot dso/map/TestMandelbrot)
make_test_dso(TestMap dso/map/TestMap)
make_test_dso(TestNoise dso/map/TestNoise)
make_test_dso(TestMaterial dso/material/TestMaterial)

# ---------------------------------------------------
# This test needs to load certain other DSOs at runtime
set(dsoTargets
        BaseMaterial
)

# Make sure the DSOs are built when this target is built
add_dependencies(${target} ${dsoTargets})

# We need to provide the path to the DSOs this test requires at runtime.
# To enable the tests to run after building but before installing, we
# generate a list of colon-separated paths to the BINARY_DIR of each DSO.
set(pathList "")
foreach(dsoTarget ${dsoTargets})
    list(APPEND pathList $<TARGET_FILE_DIR:${dsoTarget}>)
endforeach()
list(JOIN pathList ":" rdl2DsoPath)

# The final path is passed via compiler definition to the sources
target_compile_definitions(${target}
    PRIVATE
        RDL2DSO_PATH="${rdl2DsoPath}"
)
# ---------------------------------------------------

add_test(NAME ${target} COMMAND ${target})
set_tests_properties(${target} PROPERTIES
    LABELS "unit"
    WORKING_DIRECTORY $<TARGET_FILE_DIR:${target}>
)
