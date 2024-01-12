# Copyright 2023-2024 DreamWorks Animation LLC
# SPDX-License-Identifier: Apache-2.0

set(component common_mcrt_macros)

set(installIncludeDir ${PACKAGE_NAME}/common/mcrt_macros)
set(exportGroup ${PROJECT_NAME}Targets)

add_library(${component} INTERFACE "")
add_library(${PROJECT_NAME}::${component} ALIAS ${component})

set_property(TARGET ${component}
    PROPERTY PUBLIC_HEADER
        moonray_nonvirt_baseclass.h
        moonray_static_check.h
)

target_include_directories(${component}
    INTERFACE
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# If at Dreamworks add a SConscript stub file so others can use this library.
SConscript_Stub(${component})

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
