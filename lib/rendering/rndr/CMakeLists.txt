# Copyright 2023-2024 DreamWorks Animation LLC
# SPDX-License-Identifier: Apache-2.0

set(component rendering_rndr)

set(installIncludeDir ${PACKAGE_NAME}/rendering/rndr)
set(exportGroup ${PROJECT_NAME}Targets)

add_library(${component} SHARED "")
add_library(${PROJECT_NAME}::${component} ALIAS ${component})

target_sources(${component}
    PRIVATE
        statistics/AthenaCSVStream.cc
        statistics/SocketStream.cc
        adaptive/AdaptiveRegions.cc
        adaptive/AdaptiveRegionTree.cc
        AdaptiveRenderTileInfo.cc
        AdaptiveRenderTilesTable.cc
        AttributeOverrides.cc
        CheckpointController.cc
        CheckpointSigIntHandler.cc
        DebugSamplesRecArray.cc
        DisplayFilterDriver.cc
        Error.cc
        ExrUtils.cc
        Film.cc
        ImageWriteCache.cc
        ImageWriteDriver.cc
        OiioReader.cc
        OiioUtils.cc
        PixelBufferUtils.cc
        PixSampleRuntimeVerify.cc
        ProcKeeper.cc
        RenderContext.cc
        RenderContextConsoleDriver.cc
        RenderDriver.cc
        RenderDriverCheckpointUtil.cc
        RenderDriverSnapshotDelta.cc
        RenderFrame.cc
        RenderFrameCheckpointResume.cc
        RenderFramePasses.cc
        RenderOptions.cc
        RenderOutputDriver.cc
        RenderOutputDriverImplParser.cc
        RenderOutputDriverImplRead.cc
        RenderOutputDriverImplWrite.cc
        RenderOutputWriter.cc
        RenderPrepExecTracker.cc
        RenderProgressEstimation.cc
        RenderStatistics.cc
        RenderTimingRecord.cc
        ResumeHistoryMetaData.cc
        SampleIdBuff.cc
        TileSampleSpecialEvent.cc
        TileScheduler.cc
        TileWorkQueue.cc
        TileWorkQueueRuntimeVerify.cc
)
set(MOONRAY_EXEC_MODE_DEFAULT "AUTO" CACHE STRING "MoonRay's default execution mode")
set_property(CACHE MOONRAY_EXEC_MODE_DEFAULT PROPERTY STRINGS AUTO SCALAR VECTORIZED XPU)

set_property(
    SOURCE RenderOptions.cc
    PROPERTY COMPILE_DEFINITIONS
        MOONRAY_EXEC_MODE_DEFAULT=$CACHE{MOONRAY_EXEC_MODE_DEFAULT}
)

set_property(TARGET ${component}
    PROPERTY PUBLIC_HEADER
        PixelBufferUtils.h
        RenderContext.h
        RenderPrepExecTracker.h
        RenderProgressEstimation.h
        RenderOptions.h
        RenderOutputDriver.h
        RenderStatistics.h
        TileScheduler.h
        Types.h
        rndr.h
)

set_property(TARGET ${component}
    PROPERTY PRIVATE_HEADER
        statistics/ArrasLogStream.h
        statistics/AthenaCSVStream.h
)

target_include_directories(${component}
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(${component}
    PUBLIC
        SceneRdl2::common_except
        SceneRdl2::common_fb_util
        SceneRdl2::common_grid_util
        SceneRdl2::common_math
        SceneRdl2::scene_rdl2
        SceneRdl2::render_util
        ${PROJECT_NAME}::common_mcrt_util
        ${PROJECT_NAME}::common_mcrt_macros
        ${PROJECT_NAME}::common_time
        ${PROJECT_NAME}::rendering_bvh_shading
        ${PROJECT_NAME}::rendering_displayfilter
        ${PROJECT_NAME}::rendering_mcrt_common
        ${PROJECT_NAME}::rendering_geom
        ${PROJECT_NAME}::rendering_pbr
        ${PROJECT_NAME}::rendering_rt
        ${PROJECT_NAME}::statistics
        ${PROJECT_NAME}::texturing_sampler
        OpenImageIO::OpenImageIO
        OpenVDB::OpenVDB
        TBB::tbb
        atomic
        embree
)

# If at Dreamworks add a SConscript stub file so others can use this library.
SConscript_Stub(${component})

# Set standard compile/link options
Moonray_cxx_compile_definitions(${component})
Moonray_cxx_compile_features(${component})
Moonray_cxx_compile_options(${component})
Moonray_link_options(${component})

# Extra definitions needed by this target
target_compile_definitions(${component}
    PUBLIC
        # This flag is required for TBB to support waiting for workers during terminate()
        # See RenderContext::terminateAndJoinThreads() for details
        __TBB_SUPPORTS_WORKERS_WAITING_IN_TERMINATE
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
    PRIVATE_HEADER
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${installIncludeDir}/statistics
)

# # install the export set
# install(
#     EXPORT ${exportGroup}
#     NAMESPACE ${PROJECT_NAME}::
#     DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}-${PROJECT_VERSION}
# )
