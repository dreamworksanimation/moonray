// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#ifndef INCLUDED_DCX_API_H
#define INCLUDED_DCX_API_H

/* Windows compatibility.
   When compiling the library DWA_DCX_EXPORT is defined with -D
   When compiling programs DWA_DCX_EXPORT is undefined
*/
#ifndef DCX_EXPORT
#   if defined(_WIN32)
#       if defined(DWA_OPENDCX_EXPORTS)
#           define DCX_EXPORT __declspec(dllexport)
#       else
#           define DCX_EXPORT __declspec(dllimport)
#       endif
#   else
#       define DCX_EXPORT
#   endif
#endif

#include <limits>  // for std::numeric_limits

#include "version.h"

//==================================================================================
// Define epsilon for floats & doubles - for convenience
#undef  EPSILONf
#define EPSILONf std::numeric_limits<float>::epsilon()//0.0001f
#undef  EPSILONd
#define EPSILONd std::numeric_limits<double>::epsilon()//0.000001

// Define infinity for floats & doubles - for convenience
#undef  INFINITYf
#define INFINITYf std::numeric_limits<float>::infinity()//1e+37
#undef  INFINITYd
#define INFINITYd std::numeric_limits<double>::infinity()//1e+37

#define GNU_CONST_DECL constexpr


#endif // INCLUDED_DCX_API_H
