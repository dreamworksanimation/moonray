# Copyright 2023-2024 DreamWorks Animation LLC
# SPDX-License-Identifier: Apache-2.0

add_subdirectory(ispc)

set(component rendering_shading)

set(installIncludeDir ${PACKAGE_NAME}/rendering/shading)
set(exportGroup ${PROJECT_NAME}Targets)

add_library(${component} SHARED "")
add_library(${PROJECT_NAME}::${component} ALIAS ${component})

target_sources(${component}
    PRIVATE
        BasicTexture.cc
        BsdfBuilder.cc
        BsdfComponent.cc
        Displacement.cc
        EvalAttribute.cc
        EvalShader.cc
        Geometry.cc
        Material.cc
        OpenVdbSampler.cc
        OpenVdbUtil.cc
        PbrValidity.cc
        RampControl.cc
        ShadingUtil.cc
        Texture.cc
        UdimTexture.cc
        bsdf/Bsdf.cc
        bsdf/BsdfSlice.cc
        bsdf/BsdfTable.cc
        bsdf/BsdfTableAniso.cc
        bsdf/Bsdfv.cc
        bsdf/Fresnel.cc
        bsdf/LightStage.cc
        bsdf/ashikhmin_shirley/BsdfAshikhminShirley.cc
        bsdf/cook_torrance/BsdfCookTorrance.cc
        bsdf/cook_torrance/BsdfTransmissionCookTorrance.cc
        bsdf/cook_torrance/energy_compensation/CookTorranceEnergyCompensation.cc
        bsdf/cook_torrance/energy_compensation/TransmissionCookTorranceEnergyCompensation.cc
        bsdf/fabric/BsdfDwaFabric.cc
        bsdf/fabric/BsdfFabric.cc
        bsdf/fabric/BsdfFabricVelvet.cc
        bsdf/fabric/BsdfKajiyaKayFabric.cc
        bsdf/fabric/VelvetAlbedo.cc
        bsdf/hair/BsdfHair.cc
        bsdf/hair/BsdfHairDiffuse.cc
        bsdf/hair/BsdfHairLobes.cc
        bsdf/hair/BsdfHairOneSampler.cc
        bsdf/hair/HairState.cc
        bsdf/npr/BsdfToon.cc
        bsdf/under/BsdfUnderClearcoat.cc
        bsdf/under/BsdfUnderClearcoatTransmission.cc
        bsdf/ward/BsdfWard.cc
        bssrdf/Bssrdf.cc
        bssrdf/Dipole.cc
        bssrdf/MultiLayer.cc
        bssrdf/NormalizedDiffusion.cc
        bssrdf/VolumeSubsurface.cc
)

set_property(TARGET ${component}
    PROPERTY PUBLIC_HEADER
        BasicTexture.h
        BsdfBuilder.h
        BsdfComponent.h
        BsdfUtil.hh
        Displacement.h
        EvalAttribute.h
        EvalShader.h
        MaterialApi.h
        MapApi.h
        OpenVdbSampler.h
        RampControl.h
        Shading.h
        ShadingUtil.h
        ColorCorrect.h
        Iridescence.h
        UdimTexture.h
        Util.h
)

target_include_directories(${component}
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(${component}
    PUBLIC
        ${PROJECT_NAME}::bvh_shading_ispc
        ${PROJECT_NAME}::common_file_resource
        ${PROJECT_NAME}::common_mcrt_util
        ${PROJECT_NAME}::rendering_bvh_shading
        ${PROJECT_NAME}::shading_eval_ispc
        ${PROJECT_NAME}::shading_ispc
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
