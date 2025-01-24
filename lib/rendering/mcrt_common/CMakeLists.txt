# Copyright 2023-2024 DreamWorks Animation LLC
# SPDX-License-Identifier: Apache-2.0

set(component rendering_mcrt_common)

set(installIncludeDir ${PACKAGE_NAME}/rendering/mcrt_common)
set(exportGroup ${PROJECT_NAME}Targets)

add_library(${component} SHARED "")
add_library(${PROJECT_NAME}::${component} ALIAS ${component})

# ----------------------------------------
# compile some ispc sources to object files
set(objLib ${component}_objlib)

add_library(${objLib} OBJECT)

target_sources(${objLib}
    PRIVATE
        Ray.ispc
        ThreadLocalState.ispc
)

file(RELATIVE_PATH relBinDir ${CMAKE_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR})
set_target_properties(${objLib} PROPERTIES
    ISPC_HEADER_SUFFIX _ispc_stubs.h
    ISPC_HEADER_DIRECTORY /${relBinDir}
    ISPC_INSTRUCTION_SETS ${GLOBAL_ISPC_INSTRUCTION_SETS}
    LINKER_LANGUAGE CXX
)

target_link_libraries(${objLib}
    PRIVATE SceneRdl2::common_platform)

# Set standard compile/link options
Moonray_ispc_compile_options(${objLib})
Moonray_link_options(${objLib})

get_target_property(objLibDeps ${objLib} DEPENDENCY)
if(NOT objLibDeps STREQUAL "")
    add_dependencies(${objLibDeps} 
        SceneRdl2::common_platform
    )
endif()

# ----------------------------------------

get_target_property(ISPC_TARGET_OBJECTS ${objLib} TARGET_OBJECTS)
target_sources(${component}
    PRIVATE
	AffinityManager.cc
        Frustum.cc
        ProfileAccumulator.cc
        ProfileAccumulatorHandles.cc
        Ray.cc
        ThreadLocalState.cc
        Util.cc

        # pull in our ispc object files
        ${ISPC_TARGET_OBJECTS}
)

set_property(TARGET ${component}
    PROPERTY PUBLIC_HEADER
        AffinityManager.h
        ExecutionMode.h
        Frustum.h
        ThreadLocalState.hh
        Util.isph
        ${CMAKE_CURRENT_BINARY_DIR}/Ray_ispc_stubs.h
        ${CMAKE_CURRENT_BINARY_DIR}/ThreadLocalState_ispc_stubs.h
)

target_include_directories(${component}
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(${component}
    PUBLIC
        embree
        ${PROJECT_NAME}::common_mcrt_macros
        ${PROJECT_NAME}::common_time
        ${PROJECT_NAME}::deepfile
        ${OPENEXRUTIL}
        SceneRdl2::common_grid_util
        SceneRdl2::common_math
        SceneRdl2::common_platform
        SceneRdl2::render_logging
        SceneRdl2::render_util
        SceneRdl2::scene_rdl2
        TBB::tbb
)

add_dependencies(${component} ${objLib})

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
