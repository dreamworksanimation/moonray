// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#ifndef OPENDCX_VERSION_H
#define OPENDCX_VERSION_H

//==================================================================================
// Define our version symbols
// Keep major & minor in sync with OpenEXR / IlmBase!
// Patch will be used to increment the OpenDCX release number.
#define OPENDCX_VERSION        "2.02.03"
#define OPENDCX_VERSION_INT    20203
#define OPENDCX_VERSION_MAJOR  2
#define OPENDCX_VERSION_MINOR  2
#define OPENDCX_VERSION_PATCH  3

//==================================================================================
// Define namespace macros
#ifndef OPENDCX_NAMESPACE
#define OPENDCX_NAMESPACE deepfile
#endif

#ifndef OPENDCX_INTERNAL_NAMESPACE
#define OPENDCX_INTERNAL_NAMESPACE moonray::OPENDCX_NAMESPACE
#endif

//
// We need to be sure that we import the internal namespace into the public one.
// To do this, we use the small bit of code below which initially defines
// OPENDCX_INTERNAL_NAMESPACE (so it can be referenced) and then defines
// OPENDCX_NAMESPACE and pulls the internal symbols into the public
// namespace.
//

namespace moonray { namespace OPENDCX_NAMESPACE {}}
namespace moonray {
    namespace OPENDCX_NAMESPACE {
        using namespace OPENDCX_INTERNAL_NAMESPACE;
    }
}

//
// There are identical pairs of HEADER/SOURCE ENTER/EXIT macros so that
// future extension to the namespace mechanism is possible without changing
// project source code.
//

#define OPENDCX_INTERNAL_NAMESPACE_HEADER_ENTER namespace moonray { namespace OPENDCX_NAMESPACE {
#define OPENDCX_INTERNAL_NAMESPACE_HEADER_EXIT } }

#define OPENDCX_INTERNAL_NAMESPACE_SOURCE_ENTER  namespace moonray { namespace OPENDCX_NAMESPACE {
#define OPENDCX_INTERNAL_NAMESPACE_SOURCE_EXIT } }


#endif // OPENDCX_VERSION_H
