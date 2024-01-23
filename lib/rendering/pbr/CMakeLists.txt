# Copyright 2023-2024 DreamWorks Animation LLC
# SPDX-License-Identifier: Apache-2.0

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# Aov.hh needs STL_VERSION set to the major version of GCC (e.g. 9)
# This must be set for clang, gcc and ispc compiles. All our compiles are at
# least STL version 6.
set(stlVersion 6)

set(component rendering_pbr)

set(installIncludeDir ${PACKAGE_NAME}/rendering/pbr)
set(exportGroup ${PROJECT_NAME}Targets)

add_library(${component} SHARED "")
add_library(${PROJECT_NAME}::${component} ALIAS ${component})

# ----------------------------------------
# Create a 'core' directory in the build tree for generated files
add_custom_target(pbr_make_dirs
    COMMAND
        ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/core
    COMMAND
        ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/sampler
)

# ----------------------------------------
# We'd like the *_ispc_stubs.h files to be placed into the build
# tree with the same folder organization as their source. This pattern
# is used in our SCons builds, and so our #include paths are expecting it.
# To facilate this, we split the sources into multiple targets based on their
# folder structure and build them in chunks, setting the path for the stubs
# files accordingly
set(sources00
        core/Aov.ispc
        core/Distribution.ispc
        core/ImageColorCorrect.ispc
        core/PbrTLState.ispc
        core/RayState.ispc
        core/Statistics.ispc
        core/Util.ispc)
set(sources01
        integrator/BsdfOneSampler.ispc
        integrator/BsdfSampler.ispc
        integrator/LightSetSampler.ispc
        integrator/PathIntegratorBundled.ispc
        integrator/PathIntegratorMultiSampler.ispc
        integrator/PathIntegratorUtil.ispc)
set(sources02
        light/CylinderLight.ispc
        light/DiskLight.ispc
        light/DistantLight.ispc
        light/EnvLight.ispc
        light/Light.ispc
        light/LightAccelerator.ispc
        light/LightSet.ispc
        light/LightTree.ispc
        light/LightUtil.ispc
        light/MeshLight.ispc
        light/RectLight.ispc
        light/SphereLight.ispc
        light/SpotLight.ispc)
set(sources03
        lightfilter/BarnDoorLightFilter.ispc
        lightfilter/ColorRampLightFilter.ispc
        lightfilter/CombineLightFilter.ispc
        lightfilter/CookieLightFilter.ispc
        lightfilter/CookieLightFilter_v2.ispc
        lightfilter/DecayLightFilter.ispc
        lightfilter/EllipticalFalloff.ispc
        lightfilter/IntensityLightFilter.ispc
        lightfilter/LightFilterList.ispc
        lightfilter/RodLightFilter.ispc
        lightfilter/VdbLightFilter.ispc)
set(sources04
        sampler/Sampling.ispc)
set(sources05
        Types.ispc)

file(RELATIVE_PATH relBinDir ${CMAKE_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR})

function(make_ispc_target target sources stubsDestination)
    add_library(${target} OBJECT)

    target_sources(${target} PRIVATE ${sources})

    set_target_properties(${target} PROPERTIES
        ISPC_HEADER_SUFFIX _ispc_stubs.h
        ISPC_HEADER_DIRECTORY /${relBinDir}/${stubsDestination}
        ISPC_INSTRUCTION_SETS avx2-i32x8
    )

    target_link_libraries(${target}
        PRIVATE
            embree
            ${PROJECT_NAME}::bvh_shading_ispc
            ${PROJECT_NAME}::common_mcrt_macros
            ${PROJECT_NAME}::rendering_lpe
            ${PROJECT_NAME}::rendering_mcrt_common
            ${PROJECT_NAME}::rendering_shading
            ${PROJECT_NAME}::shading_ispc
            ${PROJECT_NAME}::shading_eval_ispc
            SceneRdl2::common_math
            SceneRdl2::common_platform
            SceneRdl2::math_ispc
            SceneRdl2::render_util
            SceneRdl2::scene_rdl2
    )

    target_include_directories(${target}
        PRIVATE
            $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
            $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    )

    # Set standard compile/link options
    Moonray_ispc_compile_options(${target})

    # Extra definitions needed by this target
    target_compile_definitions(${target}
        PUBLIC
            STL_VERSION=${stlVersion}
    )

    # We need to create the destination directory for the ispc_stubs.h
    # files before building the objects
    add_custom_target(${target}_make_dir COMMAND
        ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/${stubsDestination})

    add_dependencies(${target} ${target}_make_dir)
endfunction()

make_ispc_target(rendering_pbr_ispc_00 "${sources00}" core)
make_ispc_target(rendering_pbr_ispc_01 "${sources01}" integrator)
make_ispc_target(rendering_pbr_ispc_02 "${sources02}" light)
make_ispc_target(rendering_pbr_ispc_03 "${sources03}" lightfilter)
make_ispc_target(rendering_pbr_ispc_04 "${sources04}" sampler)
make_ispc_target(rendering_pbr_ispc_05 "${sources05}" .)

# ----------------------------------------
# yacc/bison and f/lex

set(yaccTarget ${component}_yacc)
bison_target(${yaccTarget} core/AovMat.yy core/AovMatParser.cc
             COMPILE_FLAGS "-d -p aovMat_yy"
             DEFINES_FILE ${CMAKE_CURRENT_BINARY_DIR}/core/AovMatParser.hh
)
set(aovMatParser_cc ${BISON_${yaccTarget}_OUTPUT_SOURCE})
set(aovMatParser_hh ${BISON_${yaccTarget}_OUTPUT_HEADER})

set(lexTarget ${component}_lex)
flex_target(${lexTarget} core/AovMat.ll core/AovMatLexer.cc
             COMPILE_FLAGS "-PaovMat_yy"
)
set(aovMatLexer_cc ${FLEX_${lexTarget}_OUTPUTS})
set(aovMatLexer_hh ${FLEX_${lexTarget}_OUTPUT_HEADER})

add_flex_bison_dependency(${lexTarget} ${yaccTarget})

# -----------------------------------------
# create object files from sample points

function(symbolsFromBinFilename binFile resultStart resultEnd)
    string(REGEX REPLACE "[\/\.\-]" "_" base ${binFile})
    string(CONCAT start "_binary_" ${base} "_start")
    string(CONCAT end "_binary_" ${base} "_end")
    set(${resultStart} ${start} PARENT_SCOPE)
    set(${resultEnd} ${end} PARENT_SCOPE)
endfunction()

set(MOONRAY_SAMPLING "PMJ02" CACHE STRING "Sampling scheme (only PMJ02 currently exposed)")
# TODO: Expose other sample options, add support below
set_property(CACHE MOONRAY_SAMPLING PROPERTY STRINGS PMJ02 #[[PPD BC RANDOM]])

set(headerDefs "LIB_RENDERING_PBR_SAMPLING_USE_${MOONRAY_SAMPLING}")

if(${MOONRAY_SAMPLING} MATCHES "^PMJ02$")
    set(bin_ppd_2D_integrator   ${CMAKE_CURRENT_SOURCE_DIR}/sampler/ppd_2D_integrator_points_4096_sequences_of_1024.bin)
    set(bin_ppd_lens            ${CMAKE_CURRENT_SOURCE_DIR}/sampler/ppd_lens_points_961_sequences_of_1024.bin)
    set(bin_bc_1D_integrator    ${CMAKE_CURRENT_SOURCE_DIR}/sampler/bc_1D_integrator_points_4096_sequences_of_1024.bin)
    set(bin_bc_time             ${CMAKE_CURRENT_SOURCE_DIR}/sampler/bc_time_points_841_sequences_of_1024.bin)

    # We need to handle the pmj02 pixel samples separately because pmj02_best_candidate_4096.bin
    # is too large to commit to github so it has been split into 40M chunks. We first need to
    # cat it back together before converting it to an object file.
    # run "split --bytes=40M --numeric-suffix=1 --suffix-length=1 pmj02_best_candidate_4096.bin pmj02_best_candidate_4096.bin."
    # if a new version is created to split it apart again.
    set(chunks
            sampler/pmj02_best_candidate_4096.bin.1
            sampler/pmj02_best_candidate_4096.bin.2
            sampler/pmj02_best_candidate_4096.bin.3
            sampler/pmj02_best_candidate_4096.bin.4)

    # Write to a temp file and only update the final file if it is changed
    set(temp_pmj02_best_candidate ${CMAKE_CURRENT_BINARY_DIR}/sampler/pmj02_best_candidate_4096.bin.tmp)
    set(bin_pmj02_best_candidate ${CMAKE_CURRENT_BINARY_DIR}/sampler/pmj02_best_candidate_4096.bin)

    # The OUTPUT_FILE in the below COMMAND must exist before we attempt to write to it.
    # Also, We need to create the 'sampler' directory for file (if it doesn't yet exist)
    # before we atempt to TOUCH it
    file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/sampler)
    file(TOUCH ${temp_pmj02_best_candidate})

    execute_process(COMMAND ${CMAKE_COMMAND} -E cat ${chunks}
                    OUTPUT_FILE ${temp_pmj02_best_candidate}
                    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                    COMMAND_ERROR_IS_FATAL ANY)

    # Copy to final .bin file
    execute_process(COMMAND ${CMAKE_COMMAND} -E copy_if_different ${temp_pmj02_best_candidate} ${bin_pmj02_best_candidate})

    # Add to list for objcopy ( .bin -> .o )
    list(APPEND binFiles
        ${bin_pmj02_best_candidate}
        ${bin_ppd_2D_integrator}
        ${bin_ppd_lens}
        ${bin_bc_1D_integrator}
        ${bin_bc_time})

    symbolsFromBinFilename(${bin_pmj02_best_candidate} bin_pmj02_best_candidate_start bin_pmj02_best_candidate_end)
    symbolsFromBinFilename(${bin_ppd_2D_integrator} bin_ppd_2D_integrator_start bin_ppd_2D_integrator_end)
    symbolsFromBinFilename(${bin_ppd_lens} bin_ppd_lens_start bin_ppd_lens_end)
    symbolsFromBinFilename(${bin_bc_1D_integrator} bin_bc_1D_integrator_start bin_bc_1D_integrator_end)
    symbolsFromBinFilename(${bin_bc_time} bin_bc_time_start bin_bc_time_end)

    list(APPEND headerDefs
          "SAMPLES_PMJ02_BEST_CANDIDATE_4096_BIN_START ${bin_pmj02_best_candidate_start}"
          "SAMPLES_PMJ02_BEST_CANDIDATE_4096_BIN_END ${bin_pmj02_best_candidate_end}"
          "SAMPLES_PPD_2D_INTEGRATOR_BIN_START ${bin_ppd_2D_integrator_start}"
          "SAMPLES_PPD_2D_INTEGRATOR_BIN_END ${bin_ppd_2D_integrator_end}"
          "SAMPLES_PPD_LENS_BIN_START ${bin_ppd_lens_start}"
          "SAMPLES_PPD_LENS_BIN_END ${bin_ppd_lens_end}"
          "SAMPLES_BC_1D_INTEGRATOR_BIN_START ${bin_bc_1D_integrator_start}"
          "SAMPLES_BC_1D_INTEGRATOR_BIN_END ${bin_bc_1D_integrator_end}"
          "SAMPLES_BC_TIME_BIN_START ${bin_bc_time_start}"
          "SAMPLES_BC_TIME_BIN_END ${bin_bc_time_end}"
    )
endif()

include(${PROJECT_NAME}SamplingDefinitions)
writeSamplingDefinitionsHeader("${headerDefs}") # passing a list requires quotes

# translate ".bin" source files to ".o" files
foreach(binFile ${binFiles})
    get_filename_component(baseName ${binFile} NAME_WLE)
    set(objFile sampler/${baseName}.o)
    add_custom_command(
        OUTPUT  ${objFile}
        DEPENDS ${binFile}
        COMMAND objcopy --input-target=binary --binary-architecture=i386
                        --output-target=elf64-x86-64
                        "${binFile}"
                        "${objFile}"
    )
    # build a list of the results
    list(APPEND samplerObjs ${objFile})
endforeach()

# --------------------------------------------------------------

add_dependencies(${component} pbr_make_dirs)

target_sources(${component}
    PRIVATE
        camera/Camera.cc
        camera/BakeCamera.cc
        camera/DomeMaster3DCamera.cc
        camera/OrthographicCamera.cc
        camera/PerspectiveCamera.cc
        camera/ProjectiveCamera.cc
        camera/SphericalCamera.cc
        camera/LensDistribution.cc
        core/Aov.cc
        core/Cryptomatte.cc
        core/DebugRay.cc
        core/DeepBuffer.cc
        core/Distribution.cc
        core/PbrTLState.cc
        core/RayState.cc
        core/Scene.cc
        core/Statistics.cc
        handlers/RayHandlers.cc
        handlers/RayHandlerUtils.cc
        handlers/ShadeBundleHandler.cc
        handlers/XPURayHandlers.cc
        integrator/BsdfOneSampler.cc
        integrator/BsdfSampler.cc
        integrator/LightSetSampler.cc
        integrator/PathGuide.cc
        integrator/PathIntegrator.cc
        integrator/PathIntegratorMultiSampler.cc
        integrator/PathIntegratorSubsurface.cc
        integrator/PathIntegratorUtil.cc
        integrator/PathIntegratorVolume.cc
        integrator/Picking.cc
        light/CylinderLight.cc
        light/DiskLight.cc
        light/DistantLight.cc
        light/EnvLight.cc
        light/Light.cc
        light/LightAccelerator.cc
        light/LightSet.cc
        light/LightTree.cc
        light/LightTreeUtil.cc
        light/LightUtil.cc
        light/MeshLight.cc
        light/RectLight.cc
        light/SphereLight.cc
        light/SpotLight.cc
        lightfilter/BarnDoorLightFilter.cc
        lightfilter/ColorRampLightFilter.cc
        lightfilter/CombineLightFilter.cc
        lightfilter/CookieLightFilter.cc
        lightfilter/CookieLightFilter_v2.cc
        lightfilter/DecayLightFilter.cc
        lightfilter/EllipticalFalloff.cc
        lightfilter/IntensityLightFilter.cc
        lightfilter/RodLightFilter.cc
        lightfilter/VdbLightFilter.cc
        sampler/Moebius.cc
        sampler/PixelScramble.cc
        sampler/Sampler.cc
        Types.cc

        # lex and yacc sources
        ${aovMatLexer_cc}
        ${aovMatParser_cc}

        # pull in our ispc object files
        $<TARGET_OBJECTS:rendering_pbr_ispc_00>
        $<TARGET_OBJECTS:rendering_pbr_ispc_01>
        $<TARGET_OBJECTS:rendering_pbr_ispc_02>
        $<TARGET_OBJECTS:rendering_pbr_ispc_03>
        $<TARGET_OBJECTS:rendering_pbr_ispc_04>
        $<TARGET_OBJECTS:rendering_pbr_ispc_05>

        # sampler object files
        ${samplerObjs}
)

target_include_directories(${component}
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
        $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(${component}
    PUBLIC
        embree
        ${PROJECT_NAME}::common_mcrt_macros
        ${PROJECT_NAME}::deepfile
        ${PROJECT_NAME}::rendering_geom
        ${PROJECT_NAME}::rendering_lpe
        ${PROJECT_NAME}::rendering_mcrt_common
        ${PROJECT_NAME}::rendering_rt
        ${PROJECT_NAME}::rendering_shading
        ${PROJECT_NAME}::shading_eval_ispc
        ${PROJECT_NAME}::texturing_sampler
        OpenImageIO::OpenImageIO
        Random123::Random123
        SceneRdl2::common_except
        SceneRdl2::render_util
        SceneRdl2::scene_rdl2
        TBB::tbb
)

# If at Dreamworks add a SConscript stub file so others can use this library.
SConscript_Stub(${component})

# Set standard compile/link options
Moonray_cxx_compile_definitions(${component})
Moonray_cxx_compile_features(${component})
Moonray_cxx_compile_options(${component})
Moonray_link_options(${component})

# Address what appears to be a compiler bug causing the pbr unit test to crash for variant 7
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION STREQUAL "6.3.1")
    set_source_files_properties(core/PbrTLState.cc
        PROPERTIES COMPILE_OPTIONS "$<$<CONFIG:RELWITHDEBINFO>:-O2>"
    )
endif()

# Extra definitions needed by this target
target_compile_definitions(${component}
    PUBLIC
        STL_VERSION=${stlVersion}
)

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
