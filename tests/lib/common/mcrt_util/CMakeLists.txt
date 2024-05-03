# Copyright 2023-2024 DreamWorks Animation LLC
# SPDX-License-Identifier: Apache-2.0

set(target moonray_common_mcrt_util_tests)

add_executable(${target})

target_sources(${target}
    PRIVATE
        main.cc
        test_aligned_element_array.cc
        test_atomic_functions.cc
        test_ring_buffer.cc
        test_wait.cc
)

if (NOT IsDarwinPlatform)
    set(PlatformSpecificLibs atomic)
endif()

target_link_libraries(${target}
    PRIVATE
        Moonray::common_mcrt_util
        pthread
        SceneRdl2::pdevunit
        ${PlatformSpecificLibs}
)

# Set standard compile/link options
Moonray_cxx_compile_definitions(${target})
Moonray_cxx_compile_features(${target})
Moonray_cxx_compile_options(${target})
Moonray_link_options(${target})

add_test(NAME ${target} COMMAND ${target})
set_tests_properties(${target} PROPERTIES
    LABELS "unit"
    WORKING_DIRECTORY $<TARGET_FILE_DIR:${target}>
)
