# Copyright 2023 DreamWorks Animation LLC
# SPDX-License-Identifier: Apache-2.0

moonray_ispc_dso(CombineDisplacement
    DEPENDENCIES
        ${PROJECT_NAME}::rendering_shading
        ${PROJECT_NAME}::shading_ispc
        ${PROJECT_NAME}::texturing_sampler
        SceneRdl2::scene_rdl2)
