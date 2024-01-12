// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
/*
 * Call initialize() to enable file resource abstractions, such
 * as the OIIO file system implementation.  When initialized, all
 * file access should go through the FileResource abstraction.
 */

namespace moonray {
namespace file_resource {
    /*
     * Returns true if initialization was successful
     */
    bool initialize();

    /* Get a file resource object by name.
     * The returned object should be deleted by
     * the caller, which will free anything in use by the object
     */
    class FileResource;
    FileResource* getFileResource(const std::string& name);

    bool fileExists(const std::string & name);
}
}


