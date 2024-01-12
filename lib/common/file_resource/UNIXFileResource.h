// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

//

#pragma once

#include "FileResource.h"
#include <string>

/** UNIX implementation of FileResource interface
 */
namespace moonray {
namespace file_resource {

class UNIXFileResource : public FileResource
{
public:
    UNIXFileResource(const std::string& filepath);
    virtual ~UNIXFileResource();

    // implementation of FileResource
    std::string userLabel() const;
    bool exists() const;
    std::istream* openIStream();
    std::ostream* openOStream();

    bool supportsIndexing() const;
    FileResource* getIndexed(float indexVal) const;
    std::string fileName() const;
    std::string extension() const;
private:
    std::string mPath;
    bool mHasPound;
};

} // namespace file_resource
} // namespace moonray
    

