# Copyright 2023-2024 DreamWorks Animation LLC
# SPDX-License-Identifier: Apache-2.0

add_subdirectory(ispc)

set(component rendering_bvh_shading)

set(installIncludeDir ${PACKAGE_NAME}/rendering/bvh/shading)
set(exportGroup ${PROJECT_NAME}Targets)

add_library(${component} SHARED "")
add_library(${PROJECT_NAME}::${component} ALIAS ${component})

target_sources(${component}
    PRIVATE
        AttributeKey.cc
        Attributes.cc
        AttributeTable.cc
        InstanceAttributes.cc
        Interpolator.cc
        Intersection.cc
        Log.cc
        MipSelector.cc
        PrimitiveAttribute.cc
        ShadingTLState.cc
        State.cc
        Xform.cc
)

set_property(TARGET ${component}
    PROPERTY PUBLIC_HEADER
        Log.h
        AttributeKey.h
        AttributeTable.h
        PrimitiveAttribute.h
        Attributes.h
        Xform.h
        State.h
        ShadingTLState.hh
)

target_include_directories(${component}
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(${component}
    PUBLIC
        ${PROJECT_NAME}::bvh_shading_ispc
        ${PROJECT_NAME}::texturing_sampler
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
)

# # install the export set
# install(
#     EXPORT ${exportGroup}
#     NAMESPACE ${PROJECT_NAME}::
#     DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}-${PROJECT_VERSION}
# )
