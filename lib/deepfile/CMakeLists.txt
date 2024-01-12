# Copyright 2023-2024 DreamWorks Animation LLC
# SPDX-License-Identifier: Apache-2.0

set(component deepfile)

set(installIncludeDir ${PACKAGE_NAME}/deepfile)
set(exportGroup ${PROJECT_NAME}Targets)

add_library(${component} SHARED "")
add_library(Moonray::${component} ALIAS ${component})

target_sources(${component}
    PRIVATE
        DcxChannelSet.cpp
        DcxDeepImageTile.cpp
        DcxDeepPixel.cpp
        DcxDeepTile.cpp
        DcxImageFormat.cpp
        DcxPixelTile.cpp
)

set_property(TARGET ${component}
    PROPERTY PUBLIC_HEADER
        DcxAPI.h
        DcxChannelAlias.h
        DcxChannelContext.h
        DcxChannelDefs.h
        DcxChannelSet.h
        DcxDeepFlags.h
        DcxDeepImageTile.h
        DcxDeepPixel.h
        DcxDeepTile.h
        DcxImageFormat.h
        DcxPixel.h
        DcxPixelTile.h
        DcxSpMask.h
        version.h
)

target_include_directories(${component}
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(${component}
    PUBLIC
        ${OPENEXRIEX}
        ${IMATHIMATH}
        ${OPENEXRILMTHREAD}
        ${OPENEXROPENEXR}
        ${OPENEXRUTIL}
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
