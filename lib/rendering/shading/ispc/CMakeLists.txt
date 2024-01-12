# Copyright 2023-2024 DreamWorks Animation LLC
# SPDX-License-Identifier: Apache-2.0

include(GNUInstallDirs)

set(installIncludeDir ${PACKAGE_NAME}/rendering/shading/ispc)
set(exportGroup ${PROJECT_NAME}Targets)

# =====================================================================================
# shading_eval_ispc: component that lib/rendering/pbr depends on to initiate shading
# =====================================================================================

# -------------------------------------------------------------------
# TODO: In our SCons build, Shadingv.cc is included in shading_eval_ispc, but it
# actually pulls headers from lib/rendering/shading and lib/rendering/shading/ispc.
#   ... so it really should be build *after* those are built and included
# as part of rendering_shading lib, ( I guess )
# -------------------------------------------------------------------

add_library(shading_eval_ispc SHARED "")
add_library(${PROJECT_NAME}::shading_eval_ispc ALIAS shading_eval_ispc)

# If we don't request the C++ linker, cmake-3.21 will complain about
# a missing variable CMAKE_ISPC_CREATE_SHARED_LIBRARY
set_target_properties(shading_eval_ispc
        PROPERTIES
            LINKER_LANGUAGE CXX)

target_sources(shading_eval_ispc
    PRIVATE
        BasicTexture.ispc
        EvalAttribute.ispc
        OpenVdbUtil.ispc
        Shading.ispc
        UdimTexture.ispc
)

set_property(TARGET shading_eval_ispc
    PROPERTY PUBLIC_HEADER
        BasicTexture.isph
        EvalAttribute.isph
        Shading.isph
        UdimTexture.isph
        ${CMAKE_CURRENT_BINARY_DIR}/BasicTexture_ispc_stubs.h
        ${CMAKE_CURRENT_BINARY_DIR}/EvalAttribute_ispc_stubs.h
        ${CMAKE_CURRENT_BINARY_DIR}/OpenVdbUtil_ispc_stubs.h
        ${CMAKE_CURRENT_BINARY_DIR}/Shading_ispc_stubs.h
        ${CMAKE_CURRENT_BINARY_DIR}/UdimTexture_ispc_stubs.h
)

file(RELATIVE_PATH relBinDir ${CMAKE_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR})
set_target_properties(shading_eval_ispc PROPERTIES
    ISPC_HEADER_SUFFIX _ispc_stubs.h
    ISPC_HEADER_DIRECTORY /${relBinDir}
    ISPC_INSTRUCTION_SETS avx2-i32x8
    # If we don't request the C++ linker, cmake will complain about
    # a missing variable CMAKE_ISPC_CREATE_SHARED_LIBRARY. This seems
    # to be a bug in CMake-3.23's ISPC support
    LINKER_LANGUAGE CXX
)

target_include_directories(shading_eval_ispc
    PUBLIC
        # find the generated ispc headers during build stage
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
        $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(shading_eval_ispc
    PUBLIC
        ${PROJECT_NAME}::texturing_sampler
        SceneRdl2::common_platform
        SceneRdl2::scene_rdl2
)

# If at Dreamworks add a SConscript stub file so others can use this library.
SConscript_Stub(shading_eval_ispc)

# Set standard compile/link options
Moonray_ispc_compile_options(shading_eval_ispc)
Moonray_link_options(shading_eval_ispc)

# -------------------------------------------------------------------

# install the target
install(TARGETS shading_eval_ispc
    COMPONENT shading_eval_ispc
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

# -------------------------------------------------------------------


# =====================================================================================
# shading_ispc: component that shaders depend on to define shaders
# =====================================================================================
add_library(shading_ispc SHARED "")
add_library(${PROJECT_NAME}::shading_ispc ALIAS shading_ispc)

# ----------------------------------------
# We'd like the *_ispc_stubs.h files to be placed into the build
# tree with the same folder organization as their source. This pattern
# is used in our SCons builds, and so our #include paths are expecting it.
# To facilate this, we split the sources into multiple targets based on their
# folder structure and build them in chunks, setting the path for the stubs
# files accordingly
set(sources00
        BsdfBuilder.ispc
        BsdfComponent.ispc
        Closure.ispc
        EvalShader.ispc
        LobeAttenuator.ispc
        PbrValidity.ispc
        RampControl.ispc
        ShaderDataAux.ispc
        Util.ispc)
set(sources01
        bsdf/Bsdf.ispc
        bsdf/BsdfEyeCaustic.ispc
        bsdf/BsdfIridescence.ispc
        bsdf/BsdfLambert.ispc
        bsdf/BsdfMirror.ispc
        bsdf/BsdfOrenNayar.ispc
        bsdf/BsdfRetroreflection.ispc
        bsdf/BsdfSlice.ispc
        bsdf/BsdfStochasticFlakes.ispc
        bsdf/BsdfUtil.ispc
        bsdf/Bsdfv.ispc
        bsdf/Fresnel.ispc)
set(sources02
        bsdf/ashikhmin_shirley/BsdfAshikhminShirley.ispc)
set(sources03
        bsdf/cook_torrance/BsdfCookTorrance.ispc
        bsdf/cook_torrance/BsdfTransmissionCookTorrance.ispc)
set(sources04
        bsdf/cook_torrance/energy_conservation/CookTorranceEnergyCompensation.ispc
        bsdf/cook_torrance/energy_conservation/TransmissionCookTorranceEnergyCompensation.ispc)
set(sources05
        bsdf/fabric/BsdfFabric.ispc
        bsdf/fabric/BsdfFabricVelvet.ispc
        bsdf/fabric/VelvetAlbedo.ispc)
set(sources06
        bsdf/hair/BsdfHair.ispc
        bsdf/hair/BsdfHairDiffuse.ispc
        bsdf/hair/BsdfHairLobes.ispc
        bsdf/hair/BsdfHairOneSampler.ispc
        bsdf/hair/HairState.ispc)
set(sources07
        bsdf/npr/BsdfFlatDiffuse.ispc
        bsdf/npr/BsdfToon.ispc)
set(sources08
        bsdf/under/BsdfUnder.ispc
        bsdf/under/BsdfUnderClearcoat.ispc
        bsdf/under/BsdfUnderClearcoatTransmission.ispc)
set(sources09
        bsdf/ward/BsdfWard.ispc)
set(sources10
        bssrdf/Bssrdf.ispc
        bssrdf/Dipole.ispc
        bssrdf/NormalizedDiffusion.ispc
        bssrdf/VolumeSubsurface.ispc)

file(RELATIVE_PATH relBinDir ${CMAKE_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR})

function(make_ispc_target target sources stubsDestination)
    add_library(${target} OBJECT)

    target_sources(${target} PRIVATE ${sources})
    foreach(source ${sources})
        string(REGEX REPLACE "(.*)\.ispc"
            "${CMAKE_CURRENT_BINARY_DIR}/\\1_ispc_stubs.h"
            header ${source})
        list(APPEND headers ${header})
    endforeach()

    set_target_properties(${target} PROPERTIES
        ISPC_HEADER_SUFFIX _ispc_stubs.h
        ISPC_HEADER_DIRECTORY /${relBinDir}/${stubsDestination}
        ISPC_INSTRUCTION_SETS avx2-i32x8
    )

    set_property(TARGET ${target}
        PROPERTY PUBLIC_HEADER
        ${headers}
    )

    target_include_directories(${target}
        PRIVATE
            $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
            $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
            $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    )

    target_link_libraries(${target}
        PRIVATE
            SceneRdl2::common_platform
            SceneRdl2::scene_rdl2
    )

    # Set standard compile/link options
    Moonray_ispc_compile_options(${target})
    Moonray_link_options(${target})

    # We need to create the destination directory for the ispc_stubs.h
    # files before building the objects
    add_custom_target(${target}_make_dir COMMAND
        ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/${stubsDestination})

    add_dependencies(${target} ${target}_make_dir)

    install(TARGETS ${target}
        EXPORT ${exportGroup}
        PUBLIC_HEADER
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${installIncludeDir}
            )
endfunction()

make_ispc_target(shading_ispc_00 "${sources00}" .)
make_ispc_target(shading_ispc_01 "${sources01}" bsdf)
make_ispc_target(shading_ispc_02 "${sources02}" bsdf/ashikhmin_shirley)
make_ispc_target(shading_ispc_03 "${sources03}" bsdf/cook_torrance)
make_ispc_target(shading_ispc_04 "${sources04}" bsdf/cook_torrance/energy_conservation)
make_ispc_target(shading_ispc_05 "${sources05}" bsdf/fabric)
make_ispc_target(shading_ispc_06 "${sources06}" bsdf/hair)
make_ispc_target(shading_ispc_07 "${sources07}" bsdf/npr)
make_ispc_target(shading_ispc_08 "${sources08}" bsdf/under)
make_ispc_target(shading_ispc_09 "${sources09}" bsdf/ward)
make_ispc_target(shading_ispc_10 "${sources10}" bssrdf)
# ----------------------------------------




# ----------------------------------------
# There is a bug in the optimization pass with ISPC 1.9.2 which causes certain
# generated code to crash in opt builds. The workaround is to move the offending
# code out into a separate file and build it in debug.
add_library(eval_attribute_debug_hack_ispc OBJECT)

target_sources(eval_attribute_debug_hack_ispc PRIVATE EvalAttributeDebugHack.ispc)

set_target_properties(eval_attribute_debug_hack_ispc PROPERTIES
    ISPC_HEADER_SUFFIX _ispc_stubs.h
    ISPC_HEADER_DIRECTORY /${relBinDir}
    ISPC_INSTRUCTION_SETS avx2-i32x8
)

target_include_directories(eval_attribute_debug_hack_ispc
    PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
        $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
)

target_link_libraries(eval_attribute_debug_hack_ispc
    PRIVATE
        SceneRdl2::common_platform
        SceneRdl2::scene_rdl2
)

# If at Dreamworks add a SConscript stub file so others can use this library.
SConscript_Stub(eval_attribute_debug_hack_ispc)

# Set standard compile/link options
Moonray_ispc_compile_options(eval_attribute_debug_hack_ispc)
Moonray_link_options(eval_attribute_debug_hack_ispc)

# ... now disable optimizations for this hack file
target_compile_options(eval_attribute_debug_hack_ispc
    PRIVATE
        -DDEBUG
        -O0
)

# ----------------------------------------


target_sources(shading_ispc
    PRIVATE
        BsdfLabels.cc
        bsdf/BsdfUtil.cc
        Shadingv.cc
        # pull in our ispc object files
        $<TARGET_OBJECTS:shading_ispc_00>
        $<TARGET_OBJECTS:shading_ispc_01>
        $<TARGET_OBJECTS:shading_ispc_02>
        $<TARGET_OBJECTS:shading_ispc_03>
        $<TARGET_OBJECTS:shading_ispc_04>
        $<TARGET_OBJECTS:shading_ispc_05>
        $<TARGET_OBJECTS:shading_ispc_06>
        $<TARGET_OBJECTS:shading_ispc_07>
        $<TARGET_OBJECTS:shading_ispc_08>
        $<TARGET_OBJECTS:shading_ispc_09>
        $<TARGET_OBJECTS:shading_ispc_10>
        $<TARGET_OBJECTS:eval_attribute_debug_hack_ispc>
)

set_property(TARGET shading_ispc
    PROPERTY PUBLIC_HEADER
        AovLabels.isph
        BasicTexture.isph
        BsdfBuilder.isph
        BsdfComponent.isph
        Closure.isph
        EvalAttribute.isph
        EvalShader.isph
        MapApi.isph
        MaterialApi.isph
        RampControl.isph
        ShaderMacros.isph
        Shading.isph
        ShaderDataAux.isph
        Ior.isph
        ColorCorrect.isph
        UdimTexture.isph
        Util.isph
)

target_include_directories(shading_ispc
    PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
    PUBLIC
        # find the generated ispc headers during build stage
        $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(shading_ispc
    PRIVATE
        shading_eval_ispc
        ${PROJECT_NAME}::texturing_sampler
    PUBLIC
        OpenImageIO::OpenImageIO
        SceneRdl2::common_platform
        SceneRdl2::scene_rdl2
)

# If at Dreamworks add a SConscript stub file so others can use this library.
SConscript_Stub(shading_ispc)

# Set standard compile/link options
Moonray_cxx_compile_definitions(shading_ispc)
Moonray_cxx_compile_features(shading_ispc)
Moonray_cxx_compile_options(shading_ispc)
Moonray_link_options(shading_ispc)

# Install some public headers that really should not be public
# at all.  Fixing this will require reworking/completing the vector
# material shading API
install(
    FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/PbrValidity.isph
    DESTINATION
        ${CMAKE_INSTALL_INCLUDEDIR}/${installIncludeDir}
)
install(
    FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/bsdf/Fresnel.hh
        ${CMAKE_CURRENT_SOURCE_DIR}/bsdf/Fresnel.isph
    DESTINATION
        ${CMAKE_INSTALL_INCLUDEDIR}/${installIncludeDir}/bsdf
)
install(
    FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/bsdf/fabric/VelvetAlbedo.isph
    DESTINATION
        ${CMAKE_INSTALL_INCLUDEDIR}/${installIncludeDir}/bsdf/fabric
)
install(
    FILES
        ${CMAKE_CURRENT_BINARY_DIR}/bssrdf/Bssrdf_ispc_stubs.h
    DESTINATION
        ${CMAKE_INSTALL_INCLUDEDIR}/${installIncludeDir}/bssrdf
)

# -------------------------------------
# Install the target and the export set
# -------------------------------------

# install the target
install(TARGETS shading_ispc
    COMPONENT shading_ispc
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
