// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

//

#pragma once

#include <iostream>

/** FileResource is base class for different implementations of
 * file-like objects.
 *
 * indexing: some FileResources may represent an indexed set of files.
 * This could be done by a substitution symbol (e.g. #) or through
 * a container/directory object. getIndexed() returns a new FileResource*
 * for the file at a given index value.
 */
namespace moonray {
namespace file_resource {

class FileResource
{
public:

    virtual ~FileResource() {}

    /* return a string label for this resource to
       be reported to a user (e.g. in an error message)
    */
    virtual std::string userLabel() const = 0;

    /* Return true if the resource exists
    **/
    virtual bool exists() const = 0;

    /* return a stream to read data from this file.
       the stream should be deleted by the caller.
       returns NULL if stream cannot be obtained. */
    virtual std::istream* openIStream() = 0;
   
    /* return a stream to write data to this file.
       the stream should be deleted by the caller.
       returns NULL if stream cannot be obtained. */
    virtual std::ostream* openOStream() = 0;

    /* return true if this represents an indexed set of
       files */
    virtual bool supportsIndexing() const = 0;

    /* return a new FileResource corresponding to a specific
       index value. returns null pointer if this doesn't support
       indexing or index is out of range.
       the returned object should be deleted by the caller
    */
    virtual FileResource* getIndexed(float indexVal) const = 0;

    virtual std::string fileName() const = 0;
    /* return the extension 
    */
    virtual std::string extension() const = 0;
};

} // namespace file_resource
} // namespace moonray

//
