# Copyright 2023-2024 DreamWorks Animation LLC
# SPDX-License-Identifier: Apache-2.0

moonray_ispc_dso(NormalDisplacement
    DEPENDENCIES
        ${PROJECT_NAME}::rendering_shading
        ${PROJECT_NAME}::shading_ispc
        ${PROJECT_NAME}::texturing_sampler
        SceneRdl2::scene_rdl2)
