# Copyright 2023-2024 DreamWorks Animation LLC
# SPDX-License-Identifier: Apache-2.0

function(${PROJECT_NAME}_cxx_compile_definitions target)
    if(CMAKE_BINARY_DIR MATCHES ".*refplat-vfx2020.*")
        # Use openvdb abi version 7 for vfx2020 to match up with Houdini 18.5
        set(openvdb_abi OPENVDB_ABI_VERSION_NUMBER=7)
    endif()

    target_compile_definitions(${target}
        PUBLIC
            $<$<CONFIG:DEBUG>:
                DEBUG                               # Enables extra validation/debugging code

                # Definitions for printing debug info
                TSLOG_LEVEL=TSLOG_MSG_DEBUG
                TSLOG_SHOW_PID
                TSLOG_SHOW_TID
                TSLOG_SHOW_TIME
            >
            $<$<CONFIG:RELWITHDEBINFO>:
                BOOST_DISABLE_ASSERTS               # Disable BOOST_ASSERT macro
            >
            $<$<CONFIG:RELEASE>:
                BOOST_DISABLE_ASSERTS               # Disable BOOST_ASSERT macro
            >

        PUBLIC
            ${GLOBAL_COMPILE_DEFINITIONS}
            ${GLOBAL_CPP_FLAGS}
            GL_GLEXT_PROTOTYPES=1                   # This define makes function symbols to be available as extern declarations.
            TBB_SUPPRESS_DEPRECATED_MESSAGES        # Suppress 'deprecated' messages from TBB

            # OpenVdb defs should probably propagate to users of some lib/rendering headers
            ${openvdb_abi}                          # Which version of the openvdb ABI to use
            OPENVDB_USE_BLOSC                       # Denotes whether VDB was built with Blosc support. Shouldn't this be defined by openvdb?
            OPENVDB_USE_LOG4CPLUS                   # Should openvdb use log4cplus (vs. std::cerr) for log messages?
            _LIBCPP_ENABLE_CXX17_REMOVED_AUTO_PTR=1 # Clang - enable auto_ptr when targeting c++17
            _LIBCPP_ENABLE_CXX17_REMOVED_RANDOM_SHUFFLE=1 # Clang - ensure std::random_shuffle is available
    )
    if(MOONRAY_DWA_BUILD)
        target_compile_definitions(${target}
            PUBLIC
                DWA_OPENVDB                         # Enables some SIMD computations in DWA's version of openvdb
        )
    endif()
endfunction()
