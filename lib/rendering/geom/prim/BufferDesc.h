// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file BufferDesc.h
/// $Id$
///

#pragma once

namespace moonray {
namespace geom {
namespace internal {

class BufferDesc {
public:
    explicit BufferDesc(const void* data = nullptr, size_t offset = 0, size_t stride = 0):
        mData(data), mOffset(offset), mStride(stride) {}

    const void* mData;
    size_t mOffset;
    size_t mStride;
};

} // namespace internal
} // namespace geom
} // namespace moonray


