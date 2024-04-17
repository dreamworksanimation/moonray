# Copyright 2023-2024 DreamWorks Animation LLC
# SPDX-License-Identifier: Apache-2.0

set(target brdf_resample)

add_executable(${target})

target_sources(${target}
    PRIVATE
        main.cc
)

target_link_libraries(${target}
    PRIVATE
        ${PROJECT_NAME}::rendering_rndr
        ${PROJECT_NAME}::rendering_shading
        SceneRdl2::render_logging
        SceneRdl2::render_util

        $<$<AND:$<CXX_COMPILER_ID:GNU>,$<VERSION_GREATER_EQUAL:$<CXX_COMPILER_VERSION>,6>>:
            atomic
        >

        $<$<CXX_COMPILER_ID:Clang>:
            atomic
        >
)

# Set standard compile/link options
Moonray_cxx_compile_definitions(${target})
Moonray_cxx_compile_features(${target})
Moonray_cxx_compile_options(${target})
Moonray_link_options(${target})

install(TARGETS ${target}
    RUNTIME DESTINATION bin)
