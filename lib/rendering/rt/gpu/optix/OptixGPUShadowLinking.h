// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace moonray {
namespace rt {

struct ShadowLinkLight
{
    int mCasterId;
    unsigned long long mLightId;
};

struct ShadowLinkReceiver
{
    int mCasterId;
    int mReceiverId;
    bool mIsComplemented;
};

} // namespace rt
} // namespace moonray
