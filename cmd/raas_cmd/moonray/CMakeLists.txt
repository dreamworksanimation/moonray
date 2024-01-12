# Copyright 2023-2024 DreamWorks Animation LLC
# SPDX-License-Identifier: Apache-2.0

option(MOONRAY_ENABLE_ATHENA_LOGGING "Enable logging CSV stream to Athena service" NO)

set(target moonray)

add_executable(${target})

target_sources(${target}
    PRIVATE
        moonray.cc
)

target_link_libraries(${target}
    PRIVATE
        ${PROJECT_NAME}::application
        ${PROJECT_NAME}::common_mcrt_util
        ${PROJECT_NAME}::rendering_rndr
        ${PROJECT_NAME}::rendering_pbr
        ${PROJECT_NAME}::rendering_shading
        SceneRdl2::render_util
        SceneRdl2::scene_rdl2
        OpenImageIO::OpenImageIO
        atomic
        ${MKL}
)

if (MOONRAY_ENABLE_ATHENA_LOGGING)
    target_compile_definitions(${target}
        PRIVATE ENABLE_ATHENA_LOGGING)
endif()

# Set standard compile/link options
Moonray_cxx_compile_definitions(${target})
Moonray_cxx_compile_features(${target})
Moonray_cxx_compile_options(${target})
Moonray_link_options(${target})

install(TARGETS ${target}
    RUNTIME DESTINATION bin)
