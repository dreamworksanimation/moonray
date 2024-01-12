# Copyright 2023-2024 DreamWorks Animation LLC
# SPDX-License-Identifier: Apache-2.0

# ===========================================
set(target blue_noise_pd_progressive_generation)
add_executable(${target})
target_sources(${target} PRIVATE
    DynamicHyperGrid.h
    NPoint.h
    blue_noise_pd_progressive_generation.cc
)
target_compile_features(${target} PRIVATE cxx_std_17)
target_link_options(${target} PRIVATE -Wl,--enable-new-dtags)


# ===========================================
set(target pd_progressive_gen)
add_executable(${target})
target_sources(${target} PRIVATE
    ArgumentParser.h
    NPoint.h
    PerfectPowerArray.h
    ProgressBar.h
    pd_progressive_generation.cc
    util.h
)
target_compile_features(${target} PRIVATE cxx_std_17)
target_link_options(${target} PRIVATE -Wl,--enable-new-dtags)


# ===========================================
set(target pd_gen)
add_executable(${target})
target_sources(${target} PRIVATE
    Array.h
    NPoint.h
    PerfectPowerArray.h
    ProgressBar.h
    pd_generation.cc
    pd_generation.h
)
target_compile_features(${target} PRIVATE cxx_std_17)
target_link_options(${target} PRIVATE -Wl,--enable-new-dtags)


# ===========================================
set(target reorder)
add_executable(${target})
target_sources(${target} PRIVATE
    DynamicHyperGrid.h
    NPoint.h
    PointContainer2D.h
    ProgressBar.h
    StaticVector.h
    reorder.cc
)
target_compile_features(${target} PRIVATE cxx_std_17)
target_link_libraries(${target} PRIVATE TBB::tbb)
target_link_options(${target} PRIVATE -Wl,--enable-new-dtags)


# ===========================================
set(target ascii_to_binary)
add_executable(${target})
target_sources(${target} PRIVATE
    NPoint.h
    ascii_to_binary.cc
)
target_compile_features(${target} PRIVATE cxx_std_17)
target_link_options(${target} PRIVATE -Wl,--enable-new-dtags)


# ===========================================
set(target stratified_best_candidate_gen)
add_executable(${target})
target_sources(${target} PRIVATE
    Array.h
    DynamicHyperGrid.h
    NPoint.h
    ProgressBar.h
    StaticVector.h
    pd_generation.h
    stratified_best_candidate.cc
)
target_compile_features(${target} PRIVATE cxx_std_17)
target_link_options(${target} PRIVATE -Wl,--enable-new-dtags)


# ===========================================
set(target discrepency)
add_executable(${target})
target_sources(${target} PRIVATE
    NPoint.h
    discrepancy.cc
)
target_compile_features(${target} PRIVATE cxx_std_17)
target_link_libraries(${target} PRIVATE TBB::tbb)
target_link_options(${target} PRIVATE -Wl,--enable-new-dtags)


# ===========================================
set(target pmj02)
add_executable(${target})
target_sources(${target} PRIVATE
    NPoint.h
    pmj02.cc
)
target_compile_features(${target} PRIVATE cxx_std_17)
target_link_options(${target} PRIVATE -Wl,--enable-new-dtags)


install(TARGETS
        blue_noise_pd_progressive_generation
        pd_progressive_gen
        pd_gen
        reorder
        ascii_to_binary
        stratified_best_candidate_gen
        discrepency
        pmj02
    RUNTIME DESTINATION
        bin/point_generation_cmd
)
