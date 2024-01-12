// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <scene_rdl2/common/platform/Platform.h>

#include <cuda_runtime.h>
#include <optix.h>  // For CUdeviceptr
#include <vector>

namespace moonray {
namespace rt {

// OptixGPUBuffer manages a buffer on the GPU, with methods to copy to/from that
// GPU buffer.  You can't manipulate GPU memory directly from the host side.

template <typename T>
class OptixGPUBuffer
{
public:
    OptixGPUBuffer() :
        mCount {0},
        mPtr {nullptr}
    {}

    // move constructor and assignment
    OptixGPUBuffer(OptixGPUBuffer&& other)
    {
        mCount = other.mCount;
        mPtr = other.mPtr;
        other.mCount = 0;
        other.mPtr = nullptr;
    }

    OptixGPUBuffer& operator=(OptixGPUBuffer&& other)
    {
        if (this != &other) {
            free();
            mCount = other.mCount;
            mPtr = other.mPtr;
            other.mCount = 0;
            other.mPtr = nullptr;
        }
        return *this;
    }

    // non-copyable
    OptixGPUBuffer(const OptixGPUBuffer&) = delete;
    OptixGPUBuffer& operator=(const OptixGPUBuffer&) = delete;

    ~OptixGPUBuffer()
    {
        free();
    }

    // you MUST check for cudaSuccess for sufficient GPU VRAM
    cudaError_t alloc(const size_t count)
    {
        free();
        if (count > 0) {
            mCount = count;
            return cudaMalloc(&mPtr, sizeInBytes());
        }
        return cudaSuccess;
    }

    void free()
    {
        if (mPtr != nullptr) {
            // Careful: If mPtr == nullptr then CUDA might not have been initialized so
            // we don't try to call it, i.e. GPU initialization has failed and we are just
            // cleaning up.
            cudaError_t err = cudaFree(mPtr);
            MNRY_ASSERT(err == cudaSuccess);
        }
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
        // *should* always succeed unless there is a bug
        cudaError_t err = cudaMemcpy(mPtr, (void*)const_cast<T*>(src), sizeInBytes(),
                                     cudaMemcpyHostToDevice);
        MNRY_ASSERT(err == cudaSuccess);
    }

    void upload(const T* src, const size_t count)
    {
        // count is <= the buffer count
        MNRY_ASSERT(mPtr != nullptr);
        MNRY_ASSERT(src != nullptr);
        MNRY_ASSERT(count <= mCount);
        // *should* always succeed unless there is a bug
        cudaError_t err = cudaMemcpy(mPtr, (void*)const_cast<T*>(src), count * sizeof(T),
                                     cudaMemcpyHostToDevice);
        MNRY_ASSERT(err == cudaSuccess);
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
        // *should* always succeed unless there is a bug
        cudaError_t err = cudaMemcpy((void*)dst, mPtr, sizeInBytes(), cudaMemcpyDeviceToHost);
        MNRY_ASSERT(err == cudaSuccess);
    }

    void download(T* dst, const size_t count) const
    {
        // count is <= the buffer count
        MNRY_ASSERT(mPtr != nullptr);
        MNRY_ASSERT(dst != nullptr);
        MNRY_ASSERT(count <= mCount);
        // *should* always succeed unless there is a bug
        cudaError_t err = cudaMemcpy((void*)dst, mPtr, count * sizeof(T), cudaMemcpyDeviceToHost);
        MNRY_ASSERT(err == cudaSuccess);
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
    T* ptr()
    {
        return mPtr;
    }

    const T* ptr() const
    {
        return mPtr;
    }

    // CUdeviceptr = unsigned long long
    // Some parts of the Optix API want this instead of a raw pointer
    CUdeviceptr deviceptr() const
    {
        return reinterpret_cast<CUdeviceptr>(mPtr);
    }

private:
    size_t mCount;
    T *mPtr;
};

} // namespace rt
} // namespace moonray

