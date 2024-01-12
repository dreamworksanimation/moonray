# Copyright 2023-2024 DreamWorks Animation LLC
# SPDX-License-Identifier: Apache-2.0

moonray_ispc_dso(ImageMap
    DEPENDENCIES
        Moonray::rendering_pbr
        Moonray::rendering_shading
        Moonray::shading_eval_ispc
        Moonray::shading_ispc
        Moonray::texturing_sampler
        SceneRdl2::render_util
        SceneRdl2::scene_rdl2)
