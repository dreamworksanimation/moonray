# Copyright 2023-2024 DreamWorks Animation LLC
# SPDX-License-Identifier: Apache-2.0

set(component bvh_shading_ispc)

set(installIncludeDir ${PACKAGE_NAME}/rendering/bvh/shading/ispc)
set(exportGroup ${PROJECT_NAME}Targets)

add_library(${component} SHARED "")
add_library(${PROJECT_NAME}::${component} ALIAS ${component})

# If we don't request the C++ linker, cmake-3.21 will complain about
# a missing variable CMAKE_ISPC_CREATE_SHARED_LIBRARY
set_target_properties(${component}
        PROPERTIES
            LINKER_LANGUAGE CXX)

target_sources(${component}
    PRIVATE
        AttributeTable.ispc
        Intersection.ispc
        MipSelector.ispc
        ShadingTLState.ispc
        Xform.ispc
)

set_property(TARGET ${component}
    PROPERTY PUBLIC_HEADER
        Intersection.hh
        Intersection.isph
        ShadingTLState.isph
        State.isph
        ThreadLocalObjectState.isph
        Xform.isph
        ${CMAKE_CURRENT_BINARY_DIR}/AttributeTable_ispc_stubs.h
        ${CMAKE_CURRENT_BINARY_DIR}/Intersection_ispc_stubs.h
        ${CMAKE_CURRENT_BINARY_DIR}/Intersection_ispc_stubs.h
        ${CMAKE_CURRENT_BINARY_DIR}/MipSelector_ispc_stubs.h
        ${CMAKE_CURRENT_BINARY_DIR}/ShadingTLState_ispc_stubs.h
        ${CMAKE_CURRENT_BINARY_DIR}/Xform_ispc_stubs.h
)

# set some ISPC related properties
file(RELATIVE_PATH relBinDir ${CMAKE_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR})
set_target_properties(${component} PROPERTIES
    ISPC_HEADER_SUFFIX _ispc_stubs.h
    ISPC_HEADER_DIRECTORY /${relBinDir}
    ISPC_INSTRUCTION_SETS avx2-i32x8
    # If we don't request the C++ linker, cmake will complain about
    # a missing variable CMAKE_ISPC_CREATE_SHARED_LIBRARY. This seems
    # to be a bug in CMake-3.23's ISPC support
    LINKER_LANGUAGE CXX
)

target_include_directories(${component}
    PUBLIC
        # find the generated ispc headers during build stage
        $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(${component}
    PUBLIC
        ${PROJECT_NAME}::rendering_mcrt_common
        SceneRdl2::common_platform
)

# If at Dreamworks add a SConscript stub file so others can use this library.
SConscript_Stub(${component})

# Set standard compile/link options
Moonray_ispc_compile_options(${component})
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
