// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <map>
#include <set>
#include <string>
#include <vector>

namespace moonray {


/**
 * The ChangeWatcher class allows you to easily install watchers on files
 * (powered by inotify in the kernel) and check for changes to those files.
 */
class ChangeWatcher
{
public:
    ChangeWatcher() {}
    virtual ~ChangeWatcher() {}
    
    static ChangeWatcher *CreateChangeWatcher();

    /**
     * Install a file watcher for the file at the given path. Any modification
     * to this file on the file system will cause hasChanged() to return true,
     * or waitForChange() to unblock.
     *
     * @param   filePath    The path to a file you wish to watch.
     */
    virtual void watchFile(const std::string& filePath) = 0;

    /// Returns true if any of the watched files have changed since the last
    /// time hasChanged() was called. This is non-blocking and will return
    /// immediately. Adds the filenames of the changed files to changedFiles.
    virtual bool hasChanged(std::set<std::string> * changedFiles = nullptr) = 0;

    /// Waits (blocks) until one of the watched files is changed.
    virtual void waitForChange() = 0;
};

} // namespace moonray

