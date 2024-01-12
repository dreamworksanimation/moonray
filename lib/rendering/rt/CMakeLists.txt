# Copyright 2023-2024 DreamWorks Animation LLC
# SPDX-License-Identifier: Apache-2.0

set(component rendering_rt)

set(installIncludeDir ${PACKAGE_NAME}/rendering/rt)
set(exportGroup ${PROJECT_NAME}Targets)

add_library(${component} SHARED "")
add_library(${PROJECT_NAME}::${component} ALIAS ${component})

if (MOONRAY_USE_CUDA)
    # ----------------------------------------
    # compile optix sources to ptx files
    set(optixLib ${component}_optix)

    add_library(${optixLib} OBJECT)

    target_sources(${optixLib}
        PRIVATE
            gpu/optix/OptixGPUPrograms.cu
    )
    set_target_properties(${optixLib}
        PROPERTIES
            CUDA_PTX_COMPILATION ON
            CUDA_ARCHITECTURES 61
            CUDA_STANDARD 11
    )

    target_compile_options(${optixLib}
        PRIVATE
            --use_fast_math
    )

    target_link_libraries(${optixLib}
        PUBLIC
            OptiX::OptiX
    )
endif()

#---------------------------------------

target_sources(${component}
    PRIVATE
        EmbreeAccelerator.cc
        GeomContext.cc
        GeometryManager.cc
        GeometryManagerExecTracker.cc
        IntersectionFilters.cc
        gpu/GPUAccelerator.cc
)

if (MOONRAY_USE_CUDA)
    target_sources(${component}
        PRIVATE
            gpu/optix/OptixGPUAccelerator.cc
            gpu/optix/OptixGPUInstance.cc
            gpu/optix/OptixGPUPrimitive.cc
            gpu/optix/OptixGPUPrimitiveGroup.cc
            gpu/optix/OptixGPUUtils.cc
    )
endif()

target_include_directories(${component}
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(${component}
    PUBLIC
        ${PROJECT_NAME}::common_mcrt_util
        ${PROJECT_NAME}::rendering_bvh_shading
        ${PROJECT_NAME}::rendering_geom
        ${PROJECT_NAME}::rendering_mcrt_common
        ${PROJECT_NAME}::rendering_shading
        SceneRdl2::common_math
        SceneRdl2::render_util
        SceneRdl2::scene_rdl2
        embree
        TBB::tbb
)

if (MOONRAY_USE_CUDA)
    target_link_libraries(${component}
        PUBLIC
            CUDA::cudart
            OptiX::OptiX
    )
endif()

# If at Dreamworks add a SConscript stub file so others can use this library.
SConscript_Stub(${component})

# TODO: Rename this option to prepend MOONRAY_ ?
# TODO: Use CACHE var with STRINGS property, see
# moonray/lib/rendering/pbr/CMakeLists.txt
option(RT_TARGET_ARCHITECTURE "Architecture for rt library (SSE,AVX,AVX2)" AVX2)
if(${RT_TARGET_ARCHITECTURE} MATCHES "^AVX2$")
    target_compile_definitions(${component}
        PUBLIC __TARGET_AVX2__ __AVX2__)
elseif(${RT_TARGET_ARCHITECTURE} MATCHES "^AVX$")
    target_compile_definitions(${component}
        PUBLIC __TARGET_AVX__ __AVX__)
endif()

if (MOONRAY_USE_CUDA)
    target_compile_definitions(${component}
        PRIVATE MOONRAY_USE_CUDA)
endif()

# Set standard compile/link options
Moonray_cxx_compile_definitions(${component})
Moonray_cxx_compile_features(${component})
Moonray_cxx_compile_options(${component})
Moonray_link_options(${component})

# -------------------------------------
# Install the target and the export set
# -------------------------------------
include(GNUInstallDirs)

if (MOONRAY_USE_CUDA)
    # install optix ptx files
    install(
        FILES $<TARGET_OBJECTS:${optixLib}>
        DESTINATION shaders
    )
endif()

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
