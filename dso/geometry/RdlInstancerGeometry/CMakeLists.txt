# Copyright 2023-2024 DreamWorks Animation LLC
# SPDX-License-Identifier: Apache-2.0

moonray_dso_simple(RdlInstancerGeometry
    DEPENDENCIES
        Moonray::rendering_geom
        Moonray::rendering_shading
        SceneRdl2::render_util
        SceneRdl2::scene_rdl2)
