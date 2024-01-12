// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#include "Sample.h"
#include "Sampler.h"

#include <fstream>
#include <iterator>

// These are symbols that are created through the 'objcopy' command, which
// takes our arbitrary binary data and shoves it into ELF format. They symbols
// can be set to any type, but we're writing out samples, so it's easiest to
// treat them as so.
//
// Part of the symbol names are generated based on the file name (periods and
// other non-symbol-legal characters are converted to underscores). The easiest
// way to find the symbols is to look at the binary created by objcopy.
// E.g 'readelf -s --wide primary_samples_binary.o'

namespace moonray {
namespace pbr {

namespace {

template <typename Container>
finline Container containerFromFile(const std::string& filename)
{
    typedef typename Container::value_type value_type;
    std::ifstream ins(filename);
    return Container(std::istream_iterator<value_type>(ins), std::istream_iterator<value_type>());
}

} // namespace

#if defined(USE_POISSON_PIXEL)
#error USE_POISSON_PIXEL defined, but these symbols should not be hard-coded
// extern "C" moonray::pbr::Sample2D
    // _binary_moonray_lib_rendering_pbr_sampler_points_poisson_points2D_4_2mil_reordered_bin_start;
// extern "C" moonray::pbr::Sample2D
    // _binary_moonray_lib_rendering_pbr_sampler_points_poisson_points2D_4_2mil_reordered_bin_end;

// PixelPartition kPixelPartition(
    // &_binary_moonray_lib_rendering_pbr_sampler_points_poisson_points2D_4_2mil_reordered_bin_start,
    // &_binary_moonray_lib_rendering_pbr_sampler_points_poisson_points2D_4_2mil_reordered_bin_end
// );
#endif

#if defined(USE_PPD_PIXEL)
#error USE_PPD_PIXEL defined, but these symbols should not be hard-coded
// extern "C" moonray::pbr::Sample2D _binary_lib_rendering_pbr_sampler_ppd_points2D_4_2mil_bin_start;
// extern "C" moonray::pbr::Sample2D _binary_lib_rendering_pbr_sampler_ppd_points2D_4_2mil_bin_end;

// PixelPartition kPixelPartition(
    // &_binary_lib_rendering_pbr_sampler_ppd_points2D_4_2mil_bin_start,
    // &_binary_lib_rendering_pbr_sampler_ppd_points2D_4_2mil_bin_end
// );
#endif

#if defined(USE_PMJ02_PIXEL)
extern "C" moonray::pbr::Sample2D SAMPLES_PMJ02_BEST_CANDIDATE_4096_BIN_START;
extern "C" moonray::pbr::Sample2D SAMPLES_PMJ02_BEST_CANDIDATE_4096_BIN_END;

PixelPartition kPixelPartition(
    &SAMPLES_PMJ02_BEST_CANDIDATE_4096_BIN_START,
    &SAMPLES_PMJ02_BEST_CANDIDATE_4096_BIN_END
);
#endif

#if defined(USE_LHS_5D)
#error USE_LHS_5D defined, but these symbols should not be hard-coded
// extern "C" moonray::pbr::Sample _binary_moonray_lib_rendering_pbr_sampler_points_4_1_5d_pure_lhs_ordered_bin_start;
// extern "C" moonray::pbr::Sample _binary_moonray_lib_rendering_pbr_sampler_points_4_1_5d_pure_lhs_ordered_bin_end;

// PixelPartition kPixelPartition(
    // &_binary_moonray_lib_rendering_pbr_sampler_points_4_1_5d_pure_lhs_ordered_bin_start,
    // &_binary_moonray_lib_rendering_pbr_sampler_points_4_1_5d_pure_lhs_ordered_bin_end
// );
#endif

#if defined(USE_LHS_PD_5D)
#error USE_LHS_PD_5D defined, but these symbols should not be hard-coded
// extern "C" moonray::pbr::Sample _binary_moonray_lib_rendering_pbr_sampler_points_5d_557596_pd_lhs_reordered_bin_start;
// extern "C" moonray::pbr::Sample _binary_moonray_lib_rendering_pbr_sampler_points_5d_557596_pd_lhs_reordered_bin_end;

// PixelPartition kPixelPartition(
    // &_binary_moonray_lib_rendering_pbr_sampler_points_5d_557596_pd_lhs_reordered_bin_start,
    // &_binary_moonray_lib_rendering_pbr_sampler_points_5d_557596_pd_lhs_reordered_bin_end
// );
#endif

#if defined(USE_GENERAL_BEST_CANDIDATE)
#error USE_GENERAL_BEST_CANDIDATE defined, but these symbols should not be hard-coded
// extern "C" moonray::pbr::Sample2D _binary_moonray_lib_rendering_pbr_sampler_points_best_candidate_4_2mil_bin_start;
// extern "C" moonray::pbr::Sample2D _binary_moonray_lib_rendering_pbr_sampler_points_best_candidate_4_2mil_bin_end;

// PixelPartition kPixelPartition(
    // &_binary_moonray_lib_rendering_pbr_sampler_points_best_candidate_4_2mil_bin_start,
    // &_binary_moonray_lib_rendering_pbr_sampler_points_best_candidate_4_2mil_bin_end
// );
#endif

#if defined(USE_STRATIFIED_BEST_CANDIDATE)
#error USE_STRATIFIED_BEST_CANDIDATE defined, but these symbols should not be hard-coded
// extern "C" moonray::pbr::Sample2D _binary_lib_rendering_pbr_sampler_stratified_best_candidate_bin_start;
// extern "C" moonray::pbr::Sample2D _binary_lib_rendering_pbr_sampler_stratified_best_candidate_bin_end;

// PixelPartition kPixelPartition(
    // &_binary_lib_rendering_pbr_sampler_stratified_best_candidate_bin_start,
    // &_binary_lib_rendering_pbr_sampler_stratified_best_candidate_bin_end
// );
#endif

#if defined(USE_BC_LENS)
#error USE_BC_LENS defined, but these symbols should not be hard-coded
// extern "C" moonray::pbr::Sample2D _binary_lib_rendering_pbr_sampler_bc_lens_points_961_sequences_of_1024_bin_start;
// extern "C" moonray::pbr::Sample2D _binary_lib_rendering_pbr_sampler_bc_lens_points_961_sequences_of_1024_bin_end;

// LensPartition kLensPartition(
    // &_binary_lib_rendering_pbr_sampler_bc_lens_points_961_sequences_of_1024_bin_start,
    // &_binary_lib_rendering_pbr_sampler_bc_lens_points_961_sequences_of_1024_bin_end
// );
#endif

#if defined(USE_PPD_LENS)
extern "C" moonray::pbr::Sample2D SAMPLES_PPD_LENS_BIN_START;
extern "C" moonray::pbr::Sample2D SAMPLES_PPD_LENS_BIN_END;

LensPartition kLensPartition(
    &SAMPLES_PPD_LENS_BIN_START,
    &SAMPLES_PPD_LENS_BIN_END
);
#endif

#if defined(USE_BC_TIME)
extern "C" float SAMPLES_BC_TIME_BIN_START;
extern "C" float SAMPLES_BC_TIME_BIN_END;

TimePartition kTimePartition(
   &SAMPLES_BC_TIME_BIN_START,
   &SAMPLES_BC_TIME_BIN_END
);

#endif

#if defined(USE_BC_INTEGRATOR_1D)
extern "C" float SAMPLES_BC_1D_INTEGRATOR_BIN_START;
extern "C" float SAMPLES_BC_1D_INTEGRATOR_BIN_END;

const std::vector<float> k1DSampleTable(
    &SAMPLES_BC_1D_INTEGRATOR_BIN_START,
    &SAMPLES_BC_1D_INTEGRATOR_BIN_END
);
#endif

#if defined(USE_BC_INTEGRATOR_2D)
#error USE_BC_INTEGRATOR_2D defined, but these symbols should not be hard-coded
// extern "C" moonray::pbr::Sample2D _binary_lib_rendering_pbr_sampler_bc_2D_integrator_points_4096_sequences_of_1024_bin_start;
// extern "C" moonray::pbr::Sample2D _binary_lib_rendering_pbr_sampler_bc_2D_integrator_points_4096_sequences_of_1024_bin_end;

// extern const std::vector<Sample2D> k2DSampleTable(
    // &_binary_lib_rendering_pbr_sampler_bc_2D_integrator_points_4096_sequences_of_1024_bin_start,
    // &_binary_lib_rendering_pbr_sampler_bc_2D_integrator_points_4096_sequences_of_1024_bin_end
// );
#endif

#if defined(USE_PPD_INTEGRATOR_2D)
extern "C" moonray::pbr::Sample2D SAMPLES_PPD_2D_INTEGRATOR_BIN_START;
extern "C" moonray::pbr::Sample2D SAMPLES_PPD_2D_INTEGRATOR_BIN_END;

extern const std::vector<Sample2D> k2DSampleTable(
    &SAMPLES_PPD_2D_INTEGRATOR_BIN_START,
    &SAMPLES_PPD_2D_INTEGRATOR_BIN_END
);
#endif

// TODO: Move to binary files if we end up using these.
/*
#if defined(USE_PD_LENS)
#error USE_PD_LENS defined, but these paths aren't valid outside DWA
// This is code for testing a seqeunce. In production cases, this needs to be converted to the binary form and not be
// dependent on a file location.
extern LensPartition kLensPartition = containerFromFile<LensPartition>(
    "/work/rd/raas/points/pd_lens_points_961_sequences_of_1024.dat");
#endif

#if defined(USE_PD_TIME)
#error USE_PD_TIME defined, but these paths aren't valid outside DWA
// This is code for testing a seqeunce. In production cases, this needs to be converted to the binary form and not be
// dependent on a file location.
extern TimePartition kTimePartition = containerFromFile<TimePartition>(
    "/work/rd/raas/points/pd_time_points_841_sequences_of_1024.dat");
#endif

#if defined(USE_PD_INTEGRATOR_1D)
#error USE_PD_INTEGRATOR_1D defined, but these paths aren't valid outside DWA
// This is code for testing a seqeunce. In production cases, this needs to be converted to the binary form and not be
// dependent on a file location.
extern const std::vector<float> k1DSampleTable = containerFromFile<std::vector<float>>(
    "/work/rd/raas/points/pd_1D_integrator_points_4096_sequences_of_1024.dat");
#endif

#if defined(USE_PD_INTEGRATOR_2D)
#error USE_PD_INTEGRATOR_2D defined, but these paths aren't valid outside DWA
// This is code for testing a seqeunce. In production cases, this needs to be converted to the binary form and not be
// dependent on a file location.
extern const std::vector<Sample2D> k2DSampleTable = containerFromFile<std::vector<Sample2D>>(
    "/work/rd/raas/points/pd_2D_integrator_points_4096_sequences_of_1024.dat");
#endif
*/

} // namespace pbr
} // namespace moonray

