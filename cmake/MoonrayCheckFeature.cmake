# Copyright 2025 DreamWorks Animation LLC
# SPDX-License-Identifier: Apache-2.0

#
# Testing __int128 atomic operation is lock-free or not and
# NO_16BYTE_ATOMIC_LOCK_FREE directive is defined if __int128 is not lock-free.
#

file(WRITE "${CMAKE_BINARY_DIR}/checkInt128AtomicLockFree.cc"
"
#include <atomic>
int main()
{
    return (__atomic_is_lock_free(sizeof(__int128), 0x0)) ? 0 : 1;
}
")

message(STATUS "Using C++ compiler for try_run: ${CMAKE_CXX_COMPILER}")

set_source_files_properties("${CMAKE_BINARY_DIR}/checkInt128AtomicLockFree.cc" PROPERTIES LANGUAGE CXX)
try_run(ATOMIC128_RUN_RESULT ATOMIC128_COMPILE_RESULT
  "${CMAKE_BINARY_DIR}"
  "${CMAKE_BINARY_DIR}/checkInt128AtomicLockFree.cc"
  LINK_LIBRARIES atomic
)

if(ATOMIC128_COMPILE_RESULT EQUAL FALSE)
  message(WARNING "Compile failed for checking int128 atomic lock-free operation")
else()
  if(ATOMIC128_RUN_RESULT EQUAL 0)
    message(STATUS "int128 atomic operation is lock-free")
  else()
    message(STATUS "int128 atomic operation is not lock-free, define NO_16BYTE_ATOMIC_LOCK_FREE directive")
    add_compile_definitions(NO_16BYTE_ATOMIC_LOCK_FREE)
  endif()
endif()  
