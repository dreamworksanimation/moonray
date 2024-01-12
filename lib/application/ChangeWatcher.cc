// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#include "ChangeWatcher.h"

#include <scene_rdl2/render/util/Files.h>
#include <scene_rdl2/common/platform/Platform.h>
#include <scene_rdl2/common/except/exceptions.h>
#include <scene_rdl2/render/util/Strings.h>

#include <algorithm>
#include <cstring>
#include <string>
#include <vector>
#include <poll.h>
#include <sys/inotify.h>
#include <unistd.h>

namespace moonray {

/*
 * You would think that we want to just set up an inotify watcher on the file
 * in question, watch for IN_MODIFY events, and call it a day. Oh, how simple
 * and lovely that would be...
 *
 * However, many text editors that use some form of swap files (vim, emacs,
 * gedit, etc) do not just modify the file in place. They may end up doing
 * other operations (like moves) between the swap file and the actual file.
 *
 * Unfortunately, if the originally watched file is moved, the watcher is
 * automatically removed. This means you'll get a single notification on first
 * edit, followed by nothing for subsequent edits. You can't just add the
 * watcher again each time, because you have no simple way of efficiently
 * synchronizing with the editor to know when that file will reappear. Ugh.
 *
 * The solution to this fantastic little problem comes to us from deep in the
 * Gentoo forums. Rather than watching the file itself, watch the directory
 * the file is in. You can filter notifications for that file by looking at
 * the "name" member of the inotify_event struct. This way, you'll get
 * notifications for that file whether it's modified in place, moved from a
 * swap file, or teleported to Mars and back. Well, maybe not that last one.
 */

ChangeWatcher::ChangeWatcher() :
    mNotifyDesc(-1)
{
    // Initialize inotify in non-blocking mode.
    mNotifyDesc = inotify_init1(IN_NONBLOCK);
    if (mNotifyDesc < 0) {
        throw scene_rdl2::except::RuntimeError(scene_rdl2::util::buildString(
                "Failed to initialize inotify for file watching: ",
                std::strerror(errno)));
    }

    // Allocate some space for the read buffer. We'll allocate enough for one
    // event struct, plus the name string at the end of it. If we need more
    // space for the name, we'll reallocate this later.
    mReadBuffer.resize(sizeof(inotify_event) + 1024, 0);
}

ChangeWatcher::~ChangeWatcher()
{
    // Remove all watchers by iterating over unique keys in the multimap.
    for (auto iter = mFileFilter.begin(); iter != mFileFilter.end();
            iter = mFileFilter.upper_bound(iter->first)) {
        inotify_rm_watch(mNotifyDesc, iter->first);
    }

    // Close the notification descriptor.
    if (mNotifyDesc >= 0) {
        close(mNotifyDesc);
    }
}

void
ChangeWatcher::watchFile(const std::string& filePath)
{
    // Different text editors perform different routines to save a file. These
    // routines generate multiple inotify events. We watch for IN_CLOSE_WRITE
    // events because all editors that were tested (vim, gedit, emacs, nano,
    // gvim, Nedit, kWrite, Houdini) generate a single IN_CLOSE_WRITE event on the
    // original file near the end of the routine. We watch for IN_MOVED_TO
    // because sed -i generates that event on the original file.
    // Gedit performs both and IN_CLOSE_WRITE and a IN_MOVED_TO on the original
    // file so there is a very slim chance Gedit may detect 2 scene changes from
    // one edit.

    // Split the file path into the basename and dirname.
    auto components = scene_rdl2::util::splitPath(filePath);
    const auto& dirName = components.first;
    const auto& fileName = components.second;

    // Add the watcher to the directory. If it already exists, we'll get the
    // same watch descriptor back.
    int watchDesc = inotify_add_watch(mNotifyDesc, dirName.c_str(), IN_CLOSE_WRITE | IN_MOVED_TO);
    if (watchDesc < 0) {
        throw scene_rdl2::except::RuntimeError(scene_rdl2::util::buildString(
                "Failed to add file watcher for '", filePath, "': ",
                std::strerror(errno)));
    }

    // Find the entry in the file filter corresponding to this watcher.
    auto range = mFileFilter.equal_range(watchDesc);
    if (range.first == mFileFilter.end() || std::none_of(range.first, range.second,
            [&fileName](const std::pair<int, std::pair<std::string, std::string> >& item) { return item.second.first == fileName; })) {
        // Either this is the first entry, or it doesn't exist yet, so add it.
        mFileFilter.insert(std::make_pair(watchDesc, std::make_pair(fileName, dirName)));
    }
}

bool
ChangeWatcher::hasChanged(std::set<std::string> * const changedFiles)
{
    // We read any notifications out of the buffer, because we want to swallow
    // them. But we only indicate a change has occurred if the event came from
    // a watched file.
    bool changeOccurred = false;
    ssize_t readLen = 0;

    // We set up inotify in non-blocking mode, so this read should never block.
tryread:
    readLen = read(mNotifyDesc, &mReadBuffer[0], mReadBuffer.size());
    if (readLen < 0) {
        if (errno == EAGAIN) {
            // No notification, so nothing has changed since the last read().
            // Something might have changed on the previous iteration though.
            return changeOccurred;
        } else if (errno == EINVAL) {
            // Buffer too small, double it and try again.
            mReadBuffer.resize(mReadBuffer.size() * 2, 0);
            goto tryread;
        } else {
            throw scene_rdl2::except::RuntimeError(scene_rdl2::util::buildString(
                    "Failed to read from file watcher descriptor: ",
                    std::strerror(errno)));
        }
    } else if (readLen > 0) {
        int offset = 0;
        while (offset < readLen) {
            // Read the next inotify event out of the buffer.
            inotify_event* event =
                reinterpret_cast<inotify_event*>(&mReadBuffer[offset]);

            // Was there a filename attached to the event?
            if (event->len > 0) {
                // Get the range of files for this watch descriptor (directory).
                auto range = mFileFilter.equal_range(event->wd);
                if (range.first == mFileFilter.end()) {
                    // For some reason we got an event for watch descriptor we
                    // didn't keep a record of... that's most certainly a logic
                    // error.
                    throw scene_rdl2::except::RuntimeError("Received an inotify notification"
                            " for an untracked watch descriptor.");
                }

                // Does this notification match any of our watched files and path?
                auto it = std::find_if(range.first, range.second,
                        [event](const std::pair<int, std::pair<std::string, std::string> >& item) {
                    return item.second.first == event->name;
                });
                // A match was found, now add it to the list
                if (it != range.second) {
                    if (changedFiles) {
                        changedFiles->insert(it->second.second + '/' + it->second.first);
                    }
                    changeOccurred = true;
                }
            }

            // Advance to the next event in the buffer.
            offset += sizeof(inotify_event) + event->len;
        }

        // Wait here for 0.1 second for any further notifications to appear,
        // then try reading again.  Keep reading until no more notifications.
        usleep(100000);
        goto tryread;
    }

    return changeOccurred;
}

void
ChangeWatcher::waitForChange()
{
    struct pollfd fds[1];
    fds[0].fd = mNotifyDesc;
    fds[0].events = POLLIN | POLLPRI;

    do {
        // Block until file descriptor is ready to read from.
        if (poll(fds, 1, -1) < 0) {
            throw scene_rdl2::except::RuntimeError(scene_rdl2::util::buildString(
                    "Polling on file watcher failed: ", std::strerror(errno)));
        }
    } while (!hasChanged());
}
} // namespace moonray

