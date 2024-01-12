# Copyright 2023-2024 DreamWorks Animation LLC
# SPDX-License-Identifier: Apache-2.0

moonray_dso_simple(RdlCurveGeometry
    DEPENDENCIES
        ${PROJECT_NAME}::rendering_geom
        SceneRdl2::render_util
        SceneRdl2::scene_rdl2)
