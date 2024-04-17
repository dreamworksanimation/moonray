// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#pragma once


#include <Metal/Metal.h>
#include <simd/vector.h>
typedef uint32_t cudaError_t;
typedef uint8_t* CUdeviceptr;
typedef uint32_t CUstream;
#define cudaSuccess 0

#include <vector>

#include <scene_rdl2/common/platform/Platform.h>

namespace moonray {
namespace rt {

// MetalGPUBuffer manages a buffer on the GPU, with methods to copy to/from that
// GPU buffer.  You can't manipulate GPU memory directly from the host side.

template <typename T>
class MetalGPUBuffer
{
public:

    MetalGPUBuffer(id<MTLDevice> device) :

        mCount {0},
        mPtr {nullptr}

        ,mDevice(device)
        ,mBuffer {nil}

    {}

    // move constructor and assignment
    MetalGPUBuffer(MetalGPUBuffer&& other)
    {
        mCount = other.mCount;
        mPtr = other.mPtr;
        other.mCount = 0;
        other.mPtr = nullptr;

        mDevice = other.mDevice;
        mBuffer = other.mBuffer;
    }

    MetalGPUBuffer& operator=(MetalGPUBuffer&& other)
    {
        if (this != &other) {
            free();
            mCount = other.mCount;
            mPtr = other.mPtr;
            other.mCount = 0;
            other.mPtr = nullptr;
            other.mDevice = mDevice;
            other.mBuffer = nil;

        }
        return *this;
    }

    // non-copyable
    MetalGPUBuffer(const MetalGPUBuffer&) = delete;
    MetalGPUBuffer& operator=(const MetalGPUBuffer&) = delete;

    ~MetalGPUBuffer()
    {
        free();
    }

    // you MUST check for cudaSuccess for sufficient GPU VRAM
    cudaError_t alloc(const size_t count)
    {
        free();
        if (count > 0) {
            mCount = count;

            mBuffer = [mDevice newBufferWithLength:sizeInBytes()
                                           options:MTLResourceStorageModeShared];
            mPtr = static_cast<T*>([mBuffer contents]);
        }
        return cudaSuccess;
    }

    void free()
    {
        [mBuffer release];
        mBuffer = nil;

        mCount = 0;
        mPtr = nullptr;
    }

    // you MUST check for cudaSuccess for sufficient GPU VRAM
    cudaError_t resize(const size_t count)
    {
        if (count != mCount) {
            free();
            return alloc(count);
        }
        // do nothing if the size hasn't changed
        return cudaSuccess;
    }

    void upload(const T* src)
    {
        // we already know the size from the previous alloc()
        MNRY_ASSERT(mPtr != nullptr);
        MNRY_ASSERT(src != nullptr);

        memcpy(mPtr, src, sizeInBytes());
//        [mBuffer didModifyRange:NSMakeRange(0, sizeInBytes())];

    }

    void upload(const T* src, const size_t count)
    {
        // count is <= the buffer count
        MNRY_ASSERT(mPtr != nullptr);
        MNRY_ASSERT(src != nullptr);
        MNRY_ASSERT(count <= mCount);
        // *should* always succeed unless there is a bug

        memcpy(mPtr, src, sizeInBytes());
//        [mBuffer didModifyRange:NSMakeRange(0, sizeInBytes())];

    }

    void upload(const std::vector<T>& v)
    {
        MNRY_ASSERT(mPtr != nullptr);
        MNRY_ASSERT(mCount == v.size());
        upload(v.data());
    }

    // you MUST check for cudaSuccess for sufficient GPU VRAM
    cudaError_t allocAndUpload(const T* src)
    {
        MNRY_ASSERT(src != nullptr);
        cudaError_t err = alloc(1);
        if (err != cudaSuccess) {
            return err;
        }
        upload(src);
        return cudaSuccess;
    }

    // you MUST check for cudaSuccess for sufficient GPU VRAM
    cudaError_t allocAndUpload(const T* src, const size_t count)
    {
        MNRY_ASSERT(src != nullptr);
        cudaError_t err = alloc(count);
        if (err != cudaSuccess) {
            return err;
        }
        if (count > 0) {
            upload(src);
        }
        return cudaSuccess;
    }

    // you MUST check for cudaSuccess for sufficient GPU VRAM
    cudaError_t allocAndUpload(const std::vector<T>& v)
    {
        cudaError_t err = alloc(v.size());
        if (err != cudaSuccess) {
            return err;
        }
        if (v.size() > 0) {
            upload(v.data());
        }
        return cudaSuccess;
    }

    void download(T* dst) const
    {
        MNRY_ASSERT(mPtr != nullptr);
        MNRY_ASSERT(dst != nullptr);

        memcpy(dst, mPtr, sizeInBytes());
    }

    void download(T* dst, const size_t count) const
    {
        // count is <= the buffer count
        MNRY_ASSERT(mPtr != nullptr);
        MNRY_ASSERT(dst != nullptr);
        MNRY_ASSERT(count <= mCount);

        memcpy(dst, mPtr, count * sizeof(T));

    }

    size_t sizeInBytes() const
    {
        return mCount * sizeof(T);
    }

    size_t count() const
    {
        return mCount;
    }

    // Pointer to memory on GPU.
    T* ptr() const
    {
        return (T*)[mBuffer gpuAddress];
    }


    T* cpu_ptr() const
    {
        return mPtr;
    }



    id<MTLBuffer> deviceptr() const
    {
        return mBuffer;
    }


    
private:
    size_t mCount;
    T *mPtr;

    id<MTLDevice> mDevice;
    id<MTLBuffer> mBuffer;

};

} // namespace rt
} // namespace moonray

