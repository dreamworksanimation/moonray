// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

// ---------------------------------------------------------------------------

#ifndef MOONRAY_NONVIRT_BASECLASS_INCLUDED
#define MOONRAY_NONVIRT_BASECLASS_INCLUDED

//
// These macros are meant to be used when inheriting from simple POD classes. They prevent
// the warning about a base class not having a virtual destructor. The implication is you
// shouldn't be holding onto a dynamically allocated instance of the derived class
// using a pointer to this base class as this will result in the derived class'
// destructor(s) not getting called on deletion. These macros are for when you know you
// won't violate this rule.
//

#if defined(__ICC)

//
// use these defines to bracket a region of code where a class derives from a non-virtual
// class. (Most of the time this is not what you want to do, but for small objects it can be.)
// Keep the region as small as possible
//
#define MOONRAY_START_INHERIT_FROM_NONVIRTUAL_BASECLASS   __pragma(warning(disable:444))
#define MOONRAY_FINISH_INHERIT_FROM_NONVIRTUAL_BASECLASS  __pragma(warning(default:444))

// simpler version for one-line cases
#define MOONRAY_INHERIT_FROM_NONVIRTUAL_BASECLASS(CODE)   __pragma(warning(disable:444)); CODE; __pragma(warning(default:444))

#else // gcc does not support these compiler warnings

#define MOONRAY_START_INHERIT_FROM_NONVIRTUAL_BASECLASS
#define MOONRAY_FINISH_INHERIT_FROM_NONVIRTUAL_BASECLASS
#define MOONRAY_INHERIT_FROM_NONVIRTUAL_BASECLASS(CODE) CODE

#endif

#endif // MOONRAY_NONVIRT_BASECLASS_INCLUDED

// ---------------------------------------------------------------------------


