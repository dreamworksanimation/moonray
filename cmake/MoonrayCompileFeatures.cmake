# Copyright 2023-2024 DreamWorks Animation LLC
# SPDX-License-Identifier: Apache-2.0

function(${PROJECT_NAME}_cxx_compile_features target)
    if (NOT CMAKE_CXX_COMPILER_ID STREQUAL Intel)
        target_compile_features(${target}
            PRIVATE
            cxx_std_17
            )
    endif()
endfunction()
