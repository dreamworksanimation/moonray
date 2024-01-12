# Copyright 2023-2024 DreamWorks Animation LLC
# SPDX-License-Identifier: Apache-2.0

moonray_ispc_dso(UsdTransform2d
    DEPENDENCIES
        Moonray::rendering_shading
        Moonray::shading_ispc
        SceneRdl2::scene_rdl2)
