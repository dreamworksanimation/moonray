# Copyright 2023-2024 DreamWorks Animation LLC
# SPDX-License-Identifier: Apache-2.0

set(component rendering_rt)

set(installIncludeDir ${PACKAGE_NAME}/rendering/rt)
set(exportGroup ${PROJECT_NAME}Targets)

add_library(${component} SHARED "")
add_library(${PROJECT_NAME}::${component} ALIAS ${component})

if(MOONRAY_USE_OPTIX)
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

if (MOONRAY_USE_METAL)
    # ----------------------------------------
    # compile Metal sources to air files
    set(metalLib ${component}_metal)

    add_library(${metalLib} STATIC)

    set(srcName gpu/Metal/MetalGPUPrograms.metal)

    target_sources(${metalLib}
        PRIVATE
            ${srcName}
    )
    set_source_files_properties(${srcName} 
        PROPERTIES 
            LANGUAGE METAL
            COMPILE_FLAGS "
                -std=metal3.1
                -ffast-math
                -gline-tables-only
                -frecord-sources"
    )

    set_target_properties(${metalLib}
        PROPERTIES
            LINKER_LANGUAGE METAL)

    add_dependencies(${component} ${metalLib})
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
        gpu/GPUAccelerator.h
)

if(MOONRAY_USE_OPTIX)
    target_sources(${component}
        PRIVATE
            gpu/optix/OptixGPUAccelerator.cc
            gpu/optix/OptixGPUInstance.cc
            gpu/optix/OptixGPUPrimitive.cc
            gpu/optix/OptixGPUPrimitiveGroup.cc
            gpu/optix/OptixGPUUtils.cc

            gpu/optix/OptixGPUAccelerator.h
            gpu/optix/OptixGPUInstance.h
            gpu/optix/OptixGPUPrimitive.h
            gpu/optix/OptixGPUPrimitiveGroup.h
            gpu/optix/OptixGPUUtils.h
    )
endif()

if (MOONRAY_USE_METAL)
    set_source_files_properties(gpu/GPUAccelerator.cc PROPERTIES COMPILE_FLAGS "-x objective-c++")
    target_sources(${component}
        PRIVATE
            gpu/Metal/MetalGPUAccelerator.mm
            gpu/Metal/MetalGPUInstance.mm
            gpu/Metal/MetalGPUPrimitive.mm
            gpu/Metal/MetalGPUPrimitiveGroup.mm
            gpu/Metal/MetalGPUUtils.mm

            gpu/Metal/MetalGPUAccelerator.h
            gpu/Metal/MetalGPUInstance.h
            gpu/Metal/MetalGPUPrimitive.h
            gpu/Metal/MetalGPUPrimitiveGroup.h
            gpu/Metal/MetalGPUUtils.h
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

if (IsDarwinPlatform)
    # Undefined symbols for architecture arm64:
    #   "moonray::pbr::ImageDistribution::~ImageDistribution()"
    #   "moonray::pbr::PerspectiveCamera::PerspectiveCamera(scene_rdl2::rdl2::Camera const*)"
    #   "moonray::pbr::Camera::update(scene_rdl2::math::Mat4<scene_rdl2::math::Vec4<double>> const&)"
    #   "vtable for moonray::pbr::ProjectiveCamera"
    set_target_properties(${component} PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
endif()

if(MOONRAY_USE_OPTIX)
    target_link_libraries(${component}
        PUBLIC
            CUDA::cudart
            OptiX::OptiX
    )
endif()

if (MOONRAY_USE_METAL)
    target_link_libraries(${component}
        PUBLIC
            Metal::Metal
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

if(MOONRAY_USE_OPTIX)
    target_compile_definitions(${component}
        PRIVATE MOONRAY_USE_OPTIX)
elseif(MOONRAY_USE_METAL)
    target_compile_definitions(${component}
        PRIVATE MOONRAY_USE_METAL)
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

if(MOONRAY_USE_OPTIX)
    # install optix ptx files
    install(
        FILES $<TARGET_OBJECTS:${optixLib}>
        DESTINATION shaders
    )
elseif(MOONRAY_USE_METAL)
    # install Metal lib files
    install(
        FILES $<TARGET_FILE_DIR:${metalLib}>/default.metallib
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
