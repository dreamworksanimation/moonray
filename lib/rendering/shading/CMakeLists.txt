# Copyright 2023-2024 DreamWorks Animation LLC
# SPDX-License-Identifier: Apache-2.0

add_subdirectory(ispc)

set(component rendering_shading)

find_package(OpenColorIO REQUIRED)

set(installIncludeDir ${PACKAGE_NAME}/rendering/shading)
set(exportGroup ${PROJECT_NAME}Targets)

add_library(${component} SHARED "")
add_library(${PROJECT_NAME}::${component} ALIAS ${component})

target_sources(${component}
    PRIVATE
        AttributeKey.cc
        Attributes.cc
        AttributeTable.cc
        BasicTexture.cc
        BsdfBuilder.cc
        BsdfComponent.cc
        Displacement.cc
        EvalAttribute.cc
        EvalShader.cc
        Geometry.cc
        InstanceAttributes.cc
        Interpolator.cc
        Intersection.cc
        Log.cc
        Material.cc
        MipSelector.cc
        OpenVdbSampler.cc
        OpenVdbUtil.cc
        PbrValidity.cc
        PrimitiveAttribute.cc
        RampControl.cc
        ShadingTLState.cc
        ShadingUtil.cc
        State.cc
        Texture.cc
        TextureColorManagement.cc
        UdimTexture.cc
        Xform.cc
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
        AttributeKey.h
        Attributes.h
        AttributeTable.h
        BasicTexture.h
        BsdfBuilder.h
        BsdfComponent.h
        BsdfUtil.hh
        Displacement.h
        EvalAttribute.h
        EvalShader.h
        Log.h
        MaterialApi.h
        MapApi.h
        OpenVdbSampler.h
        PrimitiveAttribute.h
        RampControl.h
        Shading.h
        ShadingTLState.hh
        ShadingUtil.h
        State.h
        ColorCorrect.h
        Iridescence.h
        TextureColorManagement.h
        UdimTexture.h
        Util.h
        Xform.h
)

target_include_directories(${component}
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(${component}
    PUBLIC
        ${PROJECT_NAME}::common_file_resource
        ${PROJECT_NAME}::common_mcrt_util
        ${PROJECT_NAME}::shading_eval_ispc
        ${PROJECT_NAME}::shading_ispc
        ${PROJECT_NAME}::texturing_sampler
        OpenColorIO::OpenColorIO
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
