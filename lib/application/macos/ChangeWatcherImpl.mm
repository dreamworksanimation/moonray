// Copyright 2023 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#ifndef __APPLE__
#error This file is for AppleOS platforms only
#endif

#include "../ChangeWatcher.h"

#include <scene_rdl2/render/util/Files.h>
#include <scene_rdl2/common/platform/Platform.h>
#include <scene_rdl2/common/except/exceptions.h>
#include <scene_rdl2/render/util/Strings.h>

#include <CoreServices/CoreServices.h>
#include <iostream>
#include <algorithm>
#include <cstring>
#include <string>
#include <vector>
#include <unistd.h>

namespace moonray {

class ChangeWatcherMacOS : public ChangeWatcher
{
public:
    ChangeWatcherMacOS();
    virtual ~ChangeWatcherMacOS();

    virtual void watchFile(const std::string& filePath);
    virtual bool hasChanged(std::set<std::string> * changedFiles = nullptr);
    virtual void waitForChange();

private:
    
    static void callback(ConstFSEventStreamRef stream,
                         void *callbackData,
                         size_t eventCount,
                         void *pathsVoid,
                         const FSEventStreamEventFlags flags[],
                         const FSEventStreamEventId ids[]);

    std::map<std::string, FSEventStreamRef> mEventStreams;
    std::multimap<ConstFSEventStreamRef, std::pair<std::string, std::string> > mFileFilter;
    std::set<std::string> mChangedFiles;
    std::mutex mChangedFilesLock;
};

ChangeWatcher *ChangeWatcher::CreateChangeWatcher() {
    return new ChangeWatcherMacOS();
}

void ChangeWatcherMacOS::callback(
  ConstFSEventStreamRef stream,
  void *callbackData,
  size_t eventCount,
  void *pathsVoid,
  const FSEventStreamEventFlags flags[],
  const FSEventStreamEventId ids[])
{
    ChangeWatcherMacOS *self = static_cast<ChangeWatcherMacOS*>(callbackData);
    char const **paths = (char const**)pathsVoid;

    for(int i = 0; i < eventCount; ++i) {
        if (!(flags[i] & (kFSEventStreamEventFlagItemModified|
                          kFSEventStreamEventFlagItemRenamed|
                          kFSEventStreamEventFlagItemXattrMod))) {
            continue;
        }

        auto range = self->mFileFilter.equal_range(stream);
        if (range.first == self->mFileFilter.end()) {
            // For some reason we got an event for watch descriptor we
            // didn't keep a record of... that's most certainly a logic
            // error.
            throw scene_rdl2::except::RuntimeError("Received an inotify notification"
                    " for an untracked watch descriptor.");
        }

        // Does this notification match any of our watched files and path?
        auto components = scene_rdl2::util::splitPath(paths[i]);
        const auto& dirName = components.first;
        const auto& fileName = components.second;
        
        auto it = std::find_if(range.first, range.second,
                [dirName, fileName](const std::pair<ConstFSEventStreamRef, std::pair<std::string, std::string> >& item) {
            return item.second.first == fileName && item.second.second == dirName;
        });
        // A match was found, now add it to the list
        if (it != range.second) {
            std::lock_guard<std::mutex> lock(self->mChangedFilesLock);
            self->mChangedFiles.insert(it->second.second + '/' + it->second.first);
        }
    }
}

ChangeWatcherMacOS::ChangeWatcherMacOS()
{
}

ChangeWatcherMacOS::~ChangeWatcherMacOS()
{
    for (auto iter = mFileFilter.begin(); iter != mFileFilter.end();
        iter = mFileFilter.upper_bound(iter->first)) {
        FSEventStreamRef stream = const_cast<FSEventStreamRef>(iter->first);
        FSEventStreamStop(stream);
        FSEventStreamInvalidate(stream);
        FSEventStreamRelease(stream);
    }
}

void
ChangeWatcherMacOS::watchFile(const std::string& filePath)
{
    // Split the file path into the basename and dirname.
    auto components = scene_rdl2::util::splitPath(filePath);
    const auto& dirName = components.first;
    const auto& fileName = components.second;

    // Is this a directory we're already watching?
    auto iter = mEventStreams.find(dirName);
    FSEventStreamRef stream;
    if (iter == mEventStreams.end()) {
        CFAbsoluteTime latency = 1.0;

        CFStringRef pathToWatch = CFStringCreateWithCString(kCFAllocatorDefault, dirName.c_str(),
                                                            kCFStringEncodingUTF8);
        CFArrayRef pathsToWatch = CFArrayCreate(NULL, (const void **)&pathToWatch, 1, NULL);

        FSEventStreamContext context;
        memset(&context, 0x00, sizeof(context));
        context.info = this;
        
        stream = FSEventStreamCreate(
                    NULL, &callback, &context, pathsToWatch,
                    kFSEventStreamEventIdSinceNow, latency, kFSEventStreamCreateFlagFileEvents);

        mEventStreams[dirName] = stream;

        //FSEventStreamScheduleWithRunLoop(stream, CFRunLoopGetCurrent(), kCFRunLoopDefaultMode);
        FSEventStreamSetDispatchQueue(
            stream, dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0));
        FSEventStreamStart(stream);
    }
    else {
        stream = iter->second;
    }
    
    // Find the entry in the file filter corresponding to this watcher.
    auto range = mFileFilter.equal_range(stream);
    if (range.first == mFileFilter.end() || std::none_of(range.first, range.second,
            [&fileName](const std::pair<ConstFSEventStreamRef, std::pair<std::string, std::string> >& item) { return item.second.first == fileName; })) {
        // Either this is the first entry, or it doesn't exist yet, so add it.
        mFileFilter.insert(std::make_pair(stream, std::make_pair(fileName, dirName)));
    }
}

bool
ChangeWatcherMacOS::hasChanged(std::set<std::string> * const changedFiles)
{
    // We read any notifications out of the buffer, because we want to swallow
    // them. But we only indicate a change has occurred if the event came from
    // a watched file.
    std::lock_guard<std::mutex> lock(mChangedFilesLock);
    bool changeOccurred = mChangedFiles.size();
    
    if (changedFiles) {
        *changedFiles = mChangedFiles;
    }
    mChangedFiles.clear();

    return changeOccurred;
}

void
ChangeWatcherMacOS::waitForChange()
{
    while (!hasChanged()) {
        usleep(100000);
    }
}
} // namespace moonray

