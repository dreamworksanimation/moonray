// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

//
#include "Util.h"
#include <execinfo.h>  // backtrace
#include <tbb/mutex.h>
#include <sys/syscall.h>

#include <cstring>

namespace moonray {
namespace mcrt_common {

void
threadSleep()
{
    usleep(500);
}

void
threadYield()
{
    __TBB_Yield();
}

void
debugPrintThreadID(const char *contextString)
{
    if (!contextString) contextString = "-- Thread ID = ";
    pid_t tid = syscall(SYS_gettid);

    // This printing is thread safe.
    std::printf("%s%d\n", contextString, tid);
    std::fflush(stdout);
}

void
debugPrintCallstack(const char *contextString)
{
    static tbb::mutex mutex;

    mutex.lock();

    if (!contextString) contextString = "-- Callstack:\n";
    std::printf("%s\n", contextString);

    const int MAX_BACKTRACE_SYMBOLS = 64;
    void *pointers[MAX_BACKTRACE_SYMBOLS];

    size_t size = backtrace(pointers, MAX_BACKTRACE_SYMBOLS);
    char **strings = backtrace_symbols(pointers, size);

    char indent[MAX_BACKTRACE_SYMBOLS + 1];
    memset(indent, ' ', sizeof(indent));

    int wrapperIndex = -1;

    // Iterate to size - 1 since final may not be null terminated.
    for(size_t i = 0; i < size - 1; ++i) {
        // Indent backtrace.
        indent[i - wrapperIndex - 1] = 0;
        char *p = strrchr(strings[i], '/');
        std::printf("%s%s\n", indent, p ? (p + 1) : strings[i]);
        indent[i - wrapperIndex - 1] = ' ';
    }

    std::printf("\n");
    std::fflush(stdout);

    free(strings);

    mutex.unlock();
}

// Functions exposed to ISPC:
extern "C"
{

void
CPP_debugPrintThreadID()
{
    debugPrintThreadID(nullptr);
}

void
CPP_debugPrintCallstack()
{
    debugPrintCallstack(nullptr);
}

}

} // namespace mcrt_common
} // namespace moonray

