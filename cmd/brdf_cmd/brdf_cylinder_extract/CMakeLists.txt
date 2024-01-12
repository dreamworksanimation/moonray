# Copyright 2023-2024 DreamWorks Animation LLC
# SPDX-License-Identifier: Apache-2.0

set(target brdf_cylinder_extract)

add_executable(${target})

target_sources(${target}
    PRIVATE
        EdgeDetect.cc
        Image.cc
        main.cc

)

target_link_libraries(${target}
    PRIVATE
        ${PROJECT_NAME}::rendering_rndr
        ${PROJECT_NAME}::rendering_shading
        SceneRdl2::render_util
        SceneRdl2::render_logging
        OpenImageIO::OpenImageIO
)

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    if (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 6)
        target_link_libraries(${target} PRIVATE atomic)
    endif()
endif()

# Set standard compile/link options
Moonray_cxx_compile_definitions(${target})
Moonray_cxx_compile_features(${target})
Moonray_cxx_compile_options(${target})
Moonray_link_options(${target})

install(TARGETS ${target}
    RUNTIME DESTINATION bin)
