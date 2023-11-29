# Copyright 2023 DreamWorks Animation LLC
# SPDX-License-Identifier: Apache-2.0

# Find PresenZ
#
# Imported targets
# ----------------
# This module defines the following imported targets:
#
# ``PresenZ::PresenZ``
#   The PresenZ library, if found
#
# Result variables
# ----------------
# ``PresenZ_INCLUDE_DIRS``
#   where to find ColorGrad.h etc...
# ``PresenZ_LIBRARIES``
#   the libraries to link against to use PresenZ
#
find_path(PresenZ_INCLUDE_DIR
  NAMES PzCameraApi.h
  PATH_SUFFIXES API
  HINTS $ENV{PRESENZ_ROOT}/include)

# need to find <API/PzPhaseApi.h>
set(PresenZ_INCLUDE_DIRS ${PresenZ_INCLUDE_DIR}/..)

find_library(PresenZ_LIBRARIES
  NAMES PresenZ
  HINTS $ENV{PRESENZ_ROOT}/lib )
mark_as_advanced(PresenZ_INCLUDE_DIR PresenZ_INCLUDE_DIRS PresenZ_LIBRARIES)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(PresenZ
  REQUIRED_VARS PresenZ_LIBRARIES PresenZ_INCLUDE_DIRS
)

if (PresenZ_FOUND AND NOT TARGET PresenZ::PresenZ)
    add_library(PresenZ::PresenZ UNKNOWN IMPORTED)
    set_target_properties(PresenZ::PresenZ PROPERTIES
      IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
      IMPORTED_LOCATION "${PresenZ_LIBRARIES}"
      INTERFACE_INCLUDE_DIRECTORIES "${PresenZ_INCLUDE_DIRS}")
endif()

