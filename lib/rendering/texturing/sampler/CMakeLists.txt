# Copyright 2023-2024 DreamWorks Animation LLC
# SPDX-License-Identifier: Apache-2.0

set(component texturing_sampler)

set(installIncludeDir ${PACKAGE_NAME}/rendering/texturing/sampler)
set(exportGroup ${PROJECT_NAME}Targets)

add_library(${component} SHARED "")
add_library(${PROJECT_NAME}::${component} ALIAS ${component})

# ----------------------------------------
# compile some ispc sources to object files
set(objLib ${component}_objlib)

add_library(${objLib} OBJECT)

target_sources(${objLib}
    PRIVATE
        TextureTLState.ispc
)

file(RELATIVE_PATH relBinDir ${CMAKE_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR})
set_target_properties(${objLib} PROPERTIES
    ISPC_HEADER_SUFFIX _ispc_stubs.h
    ISPC_HEADER_DIRECTORY /${relBinDir}
    ISPC_INSTRUCTION_SETS avx2-i32x8
)

target_link_libraries(${objLib}
    PRIVATE
        ${PROJECT_NAME}::rendering_mcrt_common
        SceneRdl2::common_platform
)

# Set standard compile/link options
Moonray_ispc_compile_options(${objLib})
Moonray_link_options(${objLib})

# ----------------------------------------

target_sources(${component}
    PRIVATE
        TextureSampler.cc
        TextureTLState.cc
        # pull in our ispc object files
        $<TARGET_OBJECTS:${objLib}>
)

set_property(TARGET ${component}
    PROPERTY PUBLIC_HEADER
        TextureSampler.h
        TextureTLState.h
        TextureTLState.hh
        TextureTLState.isph
        ${CMAKE_CURRENT_BINARY_DIR}/TextureTLState_ispc_stubs.h
)

target_include_directories(${component}
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(${component}
    PUBLIC
        SceneRdl2::common_grid_util
        SceneRdl2::render_util
        SceneRdl2::scene_rdl2
        ${PROJECT_NAME}::common_mcrt_macros
        ${PROJECT_NAME}::rendering_mcrt_common
        ${PROJECT_NAME}::statistics
        JPEG::JPEG
        OpenImageIO::OpenImageIO
        OpenVDB::OpenVDB
)

# If at Dreamworks add a SConscript stub file so others can use this library.
SConscript_Stub(${component})

# Set standard compile/link options
Moonray_cxx_compile_definitions(${component})
Moonray_cxx_compile_features(${component})
Moonray_cxx_compile_options(${component})
Moonray_link_options(${component})

# -------------------------------------
# Install the target and the export set
# -------------------------------------
include(GNUInstallDirs)

# install the target
install(TARGETS ${component}
    COMPONENT ${component}
    EXPORT ${exportGroup}
    LIBRARY
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        NAMELINK_SKIP
    RUNTIME
        DESTINATION ${CMAKE_INSTALL_BINDIR}
    ARCHIVE
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
    PUBLIC_HEADER
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${installIncludeDir}
)

# # install the export set
# install(
#     EXPORT ${exportGroup}
#     NAMESPACE ${PROJECT_NAME}::
#     DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}-${PROJECT_VERSION}
# )
