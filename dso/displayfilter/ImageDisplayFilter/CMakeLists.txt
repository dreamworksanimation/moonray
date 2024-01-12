# Copyright 2023-2024 DreamWorks Animation LLC
# SPDX-License-Identifier: Apache-2.0

moonray_ispc_dso(ImageDisplayFilter
    DEPENDENCIES
        Moonray::rendering_displayfilter
        OpenImageIO::OpenImageIO
        SceneRdl2::common_math
        SceneRdl2::scene_rdl2)
