# Copyright 2023-2024 DreamWorks Animation LLC
# SPDX-License-Identifier: Apache-2.0

set(component rendering_geom)

set(installIncludeDir ${PACKAGE_NAME}/rendering/geom)
set(exportGroup ${PROJECT_NAME}Targets)

add_library(${component} SHARED "")
add_library(${PROJECT_NAME}::${component} ALIAS ${component})

if(MOONRAY_BUILD_AMORPHOUS_VOLUME)
    set(amorphous_source AmorphousVolume.cc prim/AmorphousVolume.cc)
    set(amorphous_header AmorphousVolume.h)
endif()

target_sources(${component}
    PRIVATE
        Api.cc
        Box.cc
        Curves.cc
        Instance.cc
        IntersectionInit.cc
        InstanceProceduralLeaf.cc
        LocalMotionBlur.cc
        Points.cc
        PolygonMesh.cc
        Primitive.cc
        PrimitiveGroup.cc
        PrimitiveUserData.cc
        PrimitiveVisitor.cc
        Procedural.cc
        ProceduralLeaf.cc
        ProceduralNode.cc
        SharedPrimitive.cc
        Sphere.cc
        SubdivisionMesh.cc
        VdbVolume.cc
        TransformedPrimitive.cc
        prim/BezierSpanChains.cc
        prim/BSpline.cc
        prim/Box.cc
        prim/CubicSpline.cc
        prim/Curves.cc
        prim/EmissionDistribution.cc
        prim/GeomTLState.cc
        prim/Instance.cc
        prim/LineSegments.cc
        prim/Mesh.cc
        prim/MeshTessellationUtil.cc
        prim/NamedPrimitive.cc
        prim/OpenSubdivMesh.cc
        prim/Points.cc
        prim/PolyMesh.cc
        prim/PolyMeshCalcNv.cc
        prim/Primitive.cc
        prim/QuadMesh.cc
        prim/Sphere.cc
        prim/TriMesh.cc
        prim/Util.cc
        prim/VdbVolume.cc
        prim/VolumeAssignmentTable.cc
        ${amorphous_source}
)
set_property(TARGET ${component}
    PROPERTY PRIVATE_HEADER
        internal/InterleavedTraits.h
        internal/VertexBufferAllocator.h
)

set_property(TARGET ${component}
    PROPERTY PUBLIC_HEADER
        Api.h
        Box.h
        BakedAttribute.h
        Curves.h
        Instance.h
        InstanceProceduralLeaf.h
        LayerAssignmentId.h
        LocalMotionBlur.h
        MotionBlurParams.h
        Points.h
        PolygonMesh.h
        Primitive.h
        PrimitiveGroup.h
        ProceduralContext.h
        PrimitiveVisitor.h
        Procedural.h
        ProceduralLeaf.h
        SharedPrimitive.h
        Sphere.h
        State.h
        SubdivisionMesh.h
        VdbVolume.h
        TransformedPrimitive.h
        Types.h
        VertexBuffer.h
        ${amorphous_header}
)

target_include_directories(${component}
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(${component}
    PUBLIC
        embree
        OpenImageIO::OpenImageIO
        OpenSubDiv::OpenSubDiv
        OpenVDB::OpenVDB
        ${PROJECT_NAME}::common_file_resource
        ${PROJECT_NAME}::common_mcrt_util
        ${PROJECT_NAME}::rendering_bvh_shading
        ${PROJECT_NAME}::rendering_mcrt_common
        ${PROJECT_NAME}::rendering_shading
        ${PROJECT_NAME}::shading_eval_ispc
        ${PROJECT_NAME}::texturing_sampler
        SceneRdl2::common_math
        SceneRdl2::render_logging
        SceneRdl2::render_util
        SceneRdl2::scene_rdl2
        TBB::tbb
        ${AMORPHOUS}
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
    PRIVATE_HEADER
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${installIncludeDir}/internal
)

# # install the export set
# install(
#     EXPORT ${exportGroup}
#     NAMESPACE ${PROJECT_NAME}::
#     DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}-${PROJECT_VERSION}
# )
