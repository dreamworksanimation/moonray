// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

//

#include "file_resource.h"

#include "UNIXFileResource.h"

#include <memory>

namespace moonray {
namespace file_resource {

class Impl
{
public:
    Impl() : mInitialized(false) {}

    bool initialize();
    FileResource* getFileResource(const std::string& name);

private:

    bool mInitialized;
};

bool 
Impl::initialize()
{
    if (!mInitialized) {
        mInitialized = true;
    }
    return true;
}

FileResource*
Impl::getFileResource(const std::string& name)
{
    // treat name as a UNIX path
    return new UNIXFileResource(name);
}

Impl sImpl;

bool initialize()
{
    return sImpl.initialize();
}

FileResource* getFileResource(const std::string& name)
{
    return sImpl.getFileResource(name);
}

bool fileExists(const std::string& name)
{
    std::unique_ptr<FileResource> resource(getFileResource(name));
    return resource->exists();
}

} // namespace file_resource
} // namespace moonray

//
