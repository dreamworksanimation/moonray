// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

// ---------------------------------------------------------------------------

#ifndef MOONRAY_STATIC_CHECK_INCLUDED
#define MOONRAY_STATIC_CHECK_INCLUDED

/** THIS IS A TMP FILE!!!
 *  This is a copy of engine/foundation/moonray_static_check.h from foundation folio,
 *  put here in order to resolve conflicts
 *  due to moonray_dev and moonray_raas_dev using different version of Foundation.
 */


// Description:
//
//   Writes to static variables are a major cause of race conditions in
// threaded code. The Intel compiler has warnings that can trigger for
// the following accesses to static variables:
//
//   static int x=0;
//   if(x>1) {}; // warning #1710: reference to statically allocated variable
//   x = 1;      // warning #1711: assignment to statically allocated variable
//   int* p=&x;  // warning #1712: address taken of statically allocated variable
//
// 
// 1712 currently triggers in a lot of system-level code that cannot
// be avoided. There is an open issue with Intel about this.
//
// To catch this condition the warning can be enabled by default in
// Intel compiler builds and treated as an error. The macro in this
// file can then be used to disable this warning under two conditions:
//
//    * the code is 100% guaranteed to be threadsafe
//    * the code is not 100% guaranteed to be threadsafe
//
// There should be no code in the second category long term, it is
// only provided as a temporary convenience for code that will soon
// be cleaned up. Code in the first category can be tagged as
// threadsafe for one of the following reasons:
// 
//    * the variable is guaranteed never to be modified and accessed
//      from more than one thread simultaneously
//    * it is protected by a lock or other threading primitive
//    * it can be proven to be safe some other way
//
// Needless to say it is important to be sure that code tagged as safe
// really is and remains safe. The tagged code should be commented to
// indicate why it is threadsafe, eg:
//
//   static atomic<int> x = 0;
//   MOONRAY_THREADSAFE_STATIC_WRITE(x = 1; ) // safe since x is atomic variable
//
//      or:
//
//   static atomic<int> x = 0;
//   MOONRAY_START_THREADSAFE_STATIC_WRITE
//   x = 1; // safe since x is atomic variable
//   [...]  // more threadsafe code
//   MOONRAY_FINISH_THREADSAFE_STATIC_WRITE
//
// Since this option only exists with the Intel compiler, it can be
// enabled only for that compiler, and is a noop on gcc.


#if defined(__ICC)

// use these defines to bracket a region of code that has safe static
// accesses. Keep the region as small as possible
#define MOONRAY_START_THREADSAFE_STATIC_REFERENCE   __pragma(warning(disable:1710))
#define MOONRAY_FINISH_THREADSAFE_STATIC_REFERENCE  __pragma(warning(default:1710))
#define MOONRAY_START_THREADSAFE_STATIC_WRITE       __pragma(warning(disable:1711))
#define MOONRAY_FINISH_THREADSAFE_STATIC_WRITE      __pragma(warning(default:1711))
#define MOONRAY_START_THREADSAFE_STATIC_ADDRESS     __pragma(warning(disable:1712))
#define MOONRAY_FINISH_THREADSAFE_STATIC_ADDRESS    __pragma(warning(default:1712))

// use these defines to bracket a region of code that has unsafe
// static accesses. Keep the region as small as possible
#define MOONRAY_START_NON_THREADSAFE_STATIC_REFERENCE   __pragma(warning(disable:1710))
#define MOONRAY_FINISH_NON_THREADSAFE_STATIC_REFERENCE  __pragma(warning(default:1710))
#define MOONRAY_START_NON_THREADSAFE_STATIC_WRITE       __pragma(warning(disable:1711))
#define MOONRAY_FINISH_NON_THREADSAFE_STATIC_WRITE      __pragma(warning(default:1711))
#define MOONRAY_START_NON_THREADSAFE_STATIC_ADDRESS     __pragma(warning(disable:1712))
#define MOONRAY_FINISH_NON_THREADSAFE_STATIC_ADDRESS    __pragma(warning(default:1712))

// simpler version for one-line cases
#define MOONRAY_THREADSAFE_STATIC_REFERENCE(CODE) __pragma(warning(disable:1710)); CODE; __pragma(warning(default:1710))
#define MOONRAY_THREADSAFE_STATIC_WRITE(CODE)     __pragma(warning(disable:1711)); CODE; __pragma(warning(default:1711))
#define MOONRAY_THREADSAFE_STATIC_ADDRESS(CODE)   __pragma(warning(disable:1712)); CODE; __pragma(warning(default:1712))

#else // gcc does not support these compiler warnings

#define MOONRAY_START_THREADSAFE_STATIC_REFERENCE
#define MOONRAY_FINISH_THREADSAFE_STATIC_REFERENCE
#define MOONRAY_START_THREADSAFE_STATIC_WRITE
#define MOONRAY_FINISH_THREADSAFE_STATIC_WRITE
#define MOONRAY_START_THREADSAFE_STATIC_ADDRESS
#define MOONRAY_FINISH_THREADSAFE_STATIC_ADDRESS

#define MOONRAY_START_NON_THREADSAFE_STATIC_REFERENCE
#define MOONRAY_FINISH_NON_THREADSAFE_STATIC_REFERENCE
#define MOONRAY_START_NON_THREADSAFE_STATIC_WRITE
#define MOONRAY_FINISH_NON_THREADSAFE_STATIC_WRITE
#define MOONRAY_START_NON_THREADSAFE_STATIC_ADDRESS
#define MOONRAY_FINISH_NON_THREADSAFE_STATIC_ADDRESS


#define MOONRAY_THREADSAFE_STATIC_REFERENCE(CODE) CODE
#define MOONRAY_THREADSAFE_STATIC_WRITE(CODE) CODE
#define MOONRAY_THREADSAFE_STATIC_ADDRESS(CODE) CODE

#endif

#endif // MOONRAY_STATIC_CHECK_INCLUDED

// ---------------------------------------------------------------------------


