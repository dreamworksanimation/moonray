// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#pragma once

// This does not behave the same way. In true "if constexpr" methods, the non-true parts are not evaluated by the
// compiler. Therefore, this can not be used in cases where non-true cases may not compile.
#if defined(__cpp_if_constexpr)
#define IF_CONSTEXPR if constexpr
#else
#define IF_CONSTEXPR if
#endif

#define NO_DISCARD [[gnu::warn_unused_result]]
#if defined(__has_cpp_attribute) && __has_cpp_attribute(nodiscard)
    #undef NO_DISCARD
    #define NO_DISCARD [[nodiscard]]
#endif

