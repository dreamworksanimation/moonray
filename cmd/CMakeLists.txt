# Copyright 2023-2024 DreamWorks Animation LLC
# SPDX-License-Identifier: Apache-2.0

if (MOONRAY_BUILD_BRDF_CMDS)
    add_subdirectory(brdf_cmd)
endif()

if (MOONRAY_BUILD_POINT_GENERATION_CMDS)
    add_subdirectory(point_generation_cmd)
endif()

add_subdirectory(denoise_cmd)
add_subdirectory(raas_cmd)
