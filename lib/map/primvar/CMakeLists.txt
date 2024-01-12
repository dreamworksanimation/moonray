# Copyright 2023-2024 DreamWorks Animation LLC
# SPDX-License-Identifier: Apache-2.0

set(component map_primvar)

set(installIncludeDir ${PACKAGE_NAME}/map/primvar)
set(exportGroup ${PROJECT_NAME}Targets)

add_library(${component} SHARED "")
add_library(${PROJECT_NAME}::${component} ALIAS ${component})

# ----------------------------------------
# compile some ispc sources to object files
set(objLib ${component}_objlib)

add_library(${objLib} OBJECT)

target_sources(${objLib}
    PRIVATE
        ispc/Primvar.ispc
)

file(RELATIVE_PATH relBinDir ${CMAKE_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR})
set_target_properties(${objLib} PROPERTIES
    ISPC_HEADER_SUFFIX _ispc_stubs.h
    ISPC_HEADER_DIRECTORY /${relBinDir}/ispc
    ISPC_INSTRUCTION_SETS avx2-i32x8
)
# We need to create the destination directory for the ispc_stubs.h
# files before building the objects
add_custom_target(${objLib}_make_dir COMMAND
    ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/ispc)

add_dependencies(${objLib} ${objLib}_make_dir)

target_link_libraries(${objLib}
    PRIVATE
        ${PROJECT_NAME}::shading_ispc
        SceneRdl2::scene_rdl2
        SceneRdl2::common_platform)

# Set standard compile/link options
Moonray_ispc_compile_options(${objLib})
Moonray_link_options(${objLib})

# ----------------------------------------

target_sources(${component}
    PRIVATE
        Primvar.cc
        # pull in our ispc object files
        $<TARGET_OBJECTS:${objLib}>
)

set_property(TARGET ${component}
    PROPERTY PUBLIC_HEADER
        Primvar.h
)

set_property(TARGET ${component}
    PROPERTY PRIVATE_HEADER
        ispc/Primvar.isph
        ${CMAKE_CURRENT_BINARY_DIR}/ispc/Primvar_ispc_stubs.h
)

target_include_directories(${component}
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
        $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(${component}
    PUBLIC
        ${PROJECT_NAME}::shading_ispc
        SceneRdl2::scene_rdl2
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
    PRIVATE_HEADER
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${installIncludeDir}/ispc
)

# # install the export set
# install(
#     EXPORT ${exportGroup}
#     NAMESPACE ${PROJECT_NAME}::
#     DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}-${PROJECT_VERSION}
# )
