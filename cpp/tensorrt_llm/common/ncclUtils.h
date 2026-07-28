/*
 * Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/runtime/utils/multiDeviceUtils.h"

#if ENABLE_MULTI_DEVICE
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime_api.h>
#include <nccl.h>
#include <torch/extension.h>
#endif

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#if ENABLE_MULTI_DEVICE

// TLLM_NCCL_CHECK (throw on failure) is provided by multiDeviceUtils.h.

// Warn-only variant: log a warning on NCCL failure but do not throw or abort.
// Use for cleanup/secondary operations where an NCCL error is non-fatal (e.g. ncclMemFree on an error path).
#define TLLM_NCCL_CHECK_WARN(cmd)                                                                                      \
    do                                                                                                                 \
    {                                                                                                                  \
        ncclResult_t const _tllm_nccl_warn_r = (cmd);                                                                  \
        if (TLLM_UNLIKELY(_tllm_nccl_warn_r != ncclSuccess))                                                           \
        {                                                                                                              \
            TLLM_LOG_WARNING(                                                                                          \
                "NCCL error in %s (%s:%d): %s", #cmd, __FILE__, __LINE__, ncclGetErrorString(_tllm_nccl_warn_r));      \
        }                                                                                                              \
    } while (0)

TRTLLM_NAMESPACE_BEGIN

namespace common::nccl_util
{

//==============================================================================
// NCCL Resource Management
//==============================================================================

// Resource cleanup function type. Called before the NCCL communicator is destroyed.
using ResourceCleanupFunc = std::function<void()>;

// Manages resources associated with NCCL communicators. Thread-safe singleton that maintains
// a pool of resources per NCCL comm. Resources are automatically cleaned up when the
// communicator is destroyed.
class NcclCommResourceManager
{
public:
    static NcclCommResourceManager& getInstance() noexcept;

    // Register a resource cleanup function for a specific NCCL communicator.
    // The cleanup function will be called before ncclCommDestroy.
    // Thread-safe: Uses global mutex to serialize all operations.
    void registerResource(ncclComm_t comm, ResourceCleanupFunc cleanup, char const* debugName = nullptr);

    // Cleanup all resources associated with a communicator. Called automatically by
    // the shared_ptr deleter before ncclCommDestroy.
    // Thread-safe: Uses global mutex to serialize cleanup operations.
    // Order-preserving: Resources are cleaned up in registration order.
    void cleanupResources(ncclComm_t comm) noexcept;

    // Check if a communicator has registered resources.
    bool hasResources(ncclComm_t comm) const noexcept;

    // Get the number of resources registered for a communicator.
    size_t getResourceCount(ncclComm_t comm) const noexcept;

    NcclCommResourceManager(NcclCommResourceManager const&) = delete;
    NcclCommResourceManager& operator=(NcclCommResourceManager const&) = delete;
    NcclCommResourceManager(NcclCommResourceManager&&) = delete;
    NcclCommResourceManager& operator=(NcclCommResourceManager&&) = delete;

private:
    NcclCommResourceManager() = default;
    ~NcclCommResourceManager();

    using ResourceEntry = std::pair<ResourceCleanupFunc, std::string>;

    mutable std::mutex mMutex;
    std::unordered_map<ncclComm_t, std::vector<ResourceEntry>> mCommResources;
    std::atomic<bool> mIsDestroying{false};
};

// RAII helper to register a resource with a NCCL communicator.
// Automatically registers cleanup function on construction.
template <typename ResourceType>
class NcclCommResource
{
public:
    NcclCommResource(ncclComm_t comm, ResourceType&& resource, std::function<void(ResourceType&)> cleanup,
        char const* debugName = nullptr)
        : mComm(comm)
        , mResource(std::forward<ResourceType>(resource))
        , mCleanup(std::move(cleanup))
        , mRegistered(true)
    {
        // Register with the manager
        NcclCommResourceManager::getInstance().registerResource(
            comm,
            [this]()
            {
                if (mCleanup)
                {
                    mCleanup(mResource);
                }
            },
            debugName);
    }

    ResourceType& get()
    {
        return mResource;
    }

    ResourceType const& get() const
    {
        return mResource;
    }

    NcclCommResource(NcclCommResource const&) = delete;
    NcclCommResource& operator=(NcclCommResource const&) = delete;
    NcclCommResource(NcclCommResource&&) = delete;
    NcclCommResource& operator=(NcclCommResource&&) = delete;

private:
    ncclComm_t mComm;
    ResourceType mResource;
    std::function<void(ResourceType&)> mCleanup;
    bool mRegistered;
};

//==============================================================================
// NCCL Version Check
//==============================================================================

// Returns true if NCCL window buffers (ncclMemAlloc / ncclCommWindowRegister)
// are supported for the given real SM version, integrated-device flag, and runtime NCCL version.
// Exposed for focused unit testing of platform/version gates.
bool isNcclWindowSupportedForPlatform(int realSmVersion, bool isIntegrated, int ncclRuntimeVersion);

// Returns true if the compile-time and runtime NCCL versions support window buffers
// and the current CUDA device is not in a known-unsupported platform/version set.
bool isNcclWindowSupported();

// Device-explicit variant. Window support can differ between CUDA devices in
// the same process, so callers that own a tensor should use its device index.
bool isNcclWindowSupported(int device);

//==============================================================================
// NCCL Window Buffer Allocation
//==============================================================================

#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 28, 0)

// Represents a buffer with an associated NCCL window
struct NCCLWindowBuffer
{
    void* ptr;           // Device pointer (same as UBBuffer.addr)
    int handle;          // Buffer handle/index (for compatibility with UB interface)
    size_t size;         // Size in bytes
    ncclWindow_t window; // NCCL window handle
    int device;          // Exact CUDA device that owns the allocation and communicator

    NCCLWindowBuffer(void* p = nullptr, int h = -1, size_t s = 0, ncclWindow_t w = nullptr, int d = -1)
        : ptr(p)
        , handle(h)
        , size(s)
        , window(w)
        , device(d)
    {
    }

    [[nodiscard]] bool isValid() const
    {
        return ptr != nullptr && handle >= 0 && size > 0 && window != nullptr;
    }

    [[nodiscard]] bool invalid() const
    {
        return !isValid();
    }

    // Alias for compatibility with UBBuffer interface
    void* addr() const
    {
        return ptr;
    }
};

// A logical acquisition of a registered window. Registration metadata remains
// immutable and searchable; only requestBuffer can mint a releasable lease.
struct NCCLWindowLease : public NCCLWindowBuffer
{
    uint64_t generation{0};
    cudaStream_t homeStream{nullptr};

    NCCLWindowLease() = default;

    NCCLWindowLease(NCCLWindowBuffer const& buffer, uint64_t generation_, cudaStream_t homeStream_)
        : NCCLWindowBuffer(buffer)
        , generation(generation_)
        , homeStream(homeStream_)
    {
    }
};

// Manages NCCL window-registered buffers with pooling and automatic cleanup.
// Buffers are tied to the lifetime of their associated NCCL communicator.
class NCCLWindowAllocator
{
public:
    static NCCLWindowAllocator& getInstance();

    // Request a buffer for the given communicator and size.
    // If an unused buffer of at least the requested size exists for this communicator, it will be reused.
    // Uses best-fit strategy: selects the smallest available buffer that meets the size requirement.
    // Otherwise, a new buffer is allocated and registered.
    NCCLWindowLease requestBuffer(ncclComm_t comm, size_t size, int device = -1);

    // Search for a buffer by pointer. Returns an invalid buffer if not found.
    // This matches the UBManager.search_buffer() interface.
    NCCLWindowBuffer searchBuffer(ncclComm_t comm, void* ptr, int device = -1);

    // Release a buffer back to the pool for potential reuse
    // Release is conditional on the exact logical lease generation.
    void releaseBuffer(ncclComm_t comm, void* ptr, int device, uint64_t generation);

    void releaseBuffer(ncclComm_t comm, NCCLWindowLease const& lease)
    {
        releaseBuffer(comm, lease.ptr, lease.device, lease.generation);
    }

    // Get the window handle for a specific buffer pointer
    ncclWindow_t getWindow(ncclComm_t comm, void* ptr, int device = -1) const;

    // Get the size of a specific buffer pointer
    size_t getSize(ncclComm_t comm, void* ptr, int device = -1) const;

    // Get buffer info by pointer
    NCCLWindowBuffer getBufferInfo(ncclComm_t comm, void* ptr, int device = -1) const;

    // Get the number of buffers allocated for a communicator
    size_t getBufferCount(ncclComm_t comm, int device = -1) const;

    // Get the number of buffers in use for a communicator
    size_t getBufferInUseCount(ncclComm_t comm, int device = -1) const;

    // Create a serial reuse domain on the current stream. The returned handle
    // is process-local and is intended to be owned by one CUDA graph runner.
    uint64_t createReuseDomain(int device = -1);

    // Begin/end a graph capture scope for an existing serial domain. begin
    // must run after CUDA capture has started so the active cudaGraph_t can be
    // obtained and retained.
    uint64_t beginCapture(uint64_t domainId);
    void endCapture(uint64_t captureId);

    // Close a domain after its graph execs have been reset. This synchronizes
    // the teardown-only replay lane and returns its arena to the eager pool.
    void closeReuseDomain(uint64_t domainId);

    // Keep a communicator alive while the active graph domain owns window
    // addresses from it. This is a no-op outside a window capture scope.
    void retainCommForActiveCapture(std::shared_ptr<ncclComm_t> const& comm);

    // Check if a communicator is valid (non-null)
    // Note: We don't track cleaned-up comms because NCCL can reuse memory addresses.
    // All non-null comms are considered valid and will be registered when first used.
    bool isCommValid(ncclComm_t comm) const noexcept;

    NCCLWindowAllocator(NCCLWindowAllocator const&) = delete;
    NCCLWindowAllocator& operator=(NCCLWindowAllocator const&) = delete;
    NCCLWindowAllocator(NCCLWindowAllocator&&) = delete;
    NCCLWindowAllocator& operator=(NCCLWindowAllocator&&) = delete;

private:
    friend class NCCLWindowAllocatorTestAccess;

    NCCLWindowAllocator() = default;
    ~NCCLWindowAllocator() = default;

    // Allocate a new buffer and register it with NCCL as a window
    NCCLWindowBuffer allocateAndRegisterBuffer(ncclComm_t comm, size_t size, int handle, int device);

    // Record a failed new symmetric allocation (assumes mMutex is already locked).
    void recordSymmetricFailureLocked(ncclComm_t comm, int device, size_t size);

    using CudaGetLastErrorFunc = cudaError_t (*)();

    // Drain the sticky CUDA error left by a failed symmetric allocation.
    static cudaError_t clearCudaErrorIfSymmetricAllocationFailed(
        int localAllocOk, CudaGetLastErrorFunc getLastError = cudaGetLastError) noexcept;

    // Search for a buffer by pointer (assumes mMutex is already locked)
    NCCLWindowBuffer searchBufferLocked(ncclComm_t comm, void* ptr, int device) const;

    // Register cleanup function for all buffers associated with a communicator
    void registerBufferCleanup(ncclComm_t comm, int device);

    // Cleanup all buffers for a specific communicator
    void cleanupBuffersForComm(ncclComm_t comm, int device);

    struct BufferEntry
    {
        NCCLWindowBuffer buffer;
        bool inUse;
        uint64_t generation;
        cudaStream_t homeStream;
        uint64_t domainId{0};
        uint64_t lastCaptureId{0};
        bool persistent{false};
    };

    struct GraphBindingRecord
    {
        std::atomic<bool> retired{false};
        bool accounted{false};
        uint64_t captureId{0};
    };

    struct ReuseDomain
    {
        uint64_t id{0};
        int device{-1};
        cudaStream_t replayStream{nullptr};
        bool closing{false};
        size_t liveBindings{0};
        std::vector<std::unique_ptr<GraphBindingRecord>> bindings;
        std::unordered_map<ncclComm_t, std::shared_ptr<ncclComm_t>> commOwners;
    };

    struct CaptureState
    {
        std::shared_ptr<ReuseDomain> domain;
        uint64_t captureId{0};
        cudaGraph_t graph{nullptr};
        cudaStream_t captureStream{nullptr};
        bool graphBound{false};
        std::vector<BufferEntry*> touchedEntries;
    };

    struct PoolKey
    {
        ncclComm_t comm;
        int device;

        bool operator==(PoolKey const& other) const noexcept
        {
            return comm == other.comm && device == other.device;
        }
    };

    struct PoolKeyHash
    {
        size_t operator()(PoolKey const& key) const noexcept
        {
            size_t seed = std::hash<ncclComm_t>{}(key.comm);
            return seed ^ (std::hash<int>{}(key.device) + 0x9e3779b9U + (seed << 6U) + (seed >> 2U));
        }
    };

    static int resolveDevice(int device);
    static void graphBindingDestructor(void* userData);
    void ensureGraphBindingLocked(CaptureState& capture);
    void touchEntryLocked(BufferEntry& entry, CaptureState& capture);
    void drainRetiredBindingsLocked(ReuseDomain& domain);

    mutable std::mutex mMutex;
    std::unordered_map<PoolKey, std::vector<std::unique_ptr<BufferEntry>>, PoolKeyHash> mBufferPool;
    std::unordered_set<PoolKey, PoolKeyHash> mRegisteredPools;
    // Smallest request size that is known to fail collectively for each communicator.
    // Requests below the recorded size may still succeed and already-pooled buffers are always
    // reused before consulting this cache.
    std::unordered_map<PoolKey, size_t, PoolKeyHash> mMinSymmetricFailureSize;
    std::unordered_map<uint64_t, std::shared_ptr<ReuseDomain>> mReuseDomains;
    std::atomic<uint64_t> mNextDomainId{1};
    static thread_local std::unique_ptr<CaptureState> mActiveCapture;
};

// RAII wrapper for NCCL window buffers
class ScopedNCCLWindowBuffer
{
public:
    ScopedNCCLWindowBuffer(std::shared_ptr<ncclComm_t> comm, size_t size, int device = -1)
        : mComm(std::move(comm))
        , mBuffer{}
    {
        if (mComm && *mComm)
        {
            mBuffer = NCCLWindowAllocator::getInstance().requestBuffer(*mComm, size, device);
        }
    }

    ~ScopedNCCLWindowBuffer()
    {
        if (mBuffer.isValid())
        {
            NCCLWindowAllocator::getInstance().releaseBuffer(
                *mComm, mBuffer.ptr, mBuffer.device, mBuffer.generation);
        }
    }

    void* getPtr() const
    {
        return mBuffer.ptr;
    }

    size_t getSize() const
    {
        return mBuffer.size;
    }

    ncclWindow_t getWindow() const
    {
        return mBuffer.window;
    }

    NCCLWindowBuffer const& getBuffer() const
    {
        return mBuffer;
    }

    ScopedNCCLWindowBuffer(ScopedNCCLWindowBuffer const&) = delete;
    ScopedNCCLWindowBuffer& operator=(ScopedNCCLWindowBuffer const&) = delete;
    ScopedNCCLWindowBuffer(ScopedNCCLWindowBuffer&&) = delete;
    ScopedNCCLWindowBuffer& operator=(ScopedNCCLWindowBuffer&&) = delete;

private:
    std::shared_ptr<ncclComm_t> mComm;
    NCCLWindowLease mBuffer;
};

// Creates a PyTorch tensor backed by an NCCL window buffer.
// The tensor will automatically release the buffer back to the pool when destroyed.
// This is analogous to torch_ext::create_userbuffers_tensor() but for NCCLWindowAllocator.
inline std::pair<torch::Tensor, NCCLWindowBuffer> createNCCLWindowTensor(
    std::shared_ptr<ncclComm_t> comm, at::IntArrayRef shape, torch::ScalarType dtype, c10::Device device)
{
    TLLM_CHECK_WITH_INFO(device.is_cuda(), "NCCL window tensors require a CUDA device");
    if (!device.has_index())
    {
        device = c10::Device(c10::DeviceType::CUDA, c10::cuda::current_device());
    }
    c10::cuda::CUDAGuard deviceGuard(device);

    // Calculate buffer size
    int64_t buffer_size
        = std::accumulate(shape.begin(), shape.end(), 1LL, std::multiplies<int64_t>()) * torch::elementSize(dtype);

    // Calculate strides
    std::vector<int64_t> strides_vec(shape.size());
    if (!shape.empty())
    {
        strides_vec[shape.size() - 1] = 1;
        for (int64_t i = static_cast<int64_t>(shape.size()) - 1; i >= 1; --i)
        {
            strides_vec[i - 1] = strides_vec[i] * shape[i];
        }
    }

    // Request buffer from allocator
    auto& allocator = NCCLWindowAllocator::getInstance();
    NCCLWindowLease lease;

    if (!comm || !*comm)
    {
        TLLM_LOG_DEBUG("[createNCCLWindowTensor] null comm; returning invalid buffer");
        return std::make_pair(torch::Tensor(), NCCLWindowBuffer());
    }

    try
    {
        allocator.retainCommForActiveCapture(comm);
        lease = allocator.requestBuffer(*comm, buffer_size, device.index());
    }
    catch (std::exception const& e)
    {
        TLLM_LOG_DEBUG("[createNCCLWindowTensor] requestBuffer failed; returning invalid buffer: %s", e.what());
        return std::make_pair(torch::Tensor(), NCCLWindowBuffer());
    }

    // Defensive validation: ensure buffer is valid before proceeding
    if (!lease.isValid())
    {
        TLLM_LOG_DEBUG("[createNCCLWindowTensor] invalid buffer returned from requestBuffer; returning invalid buffer");
        return std::make_pair(torch::Tensor(), NCCLWindowBuffer());
    }

    // Create custom deleter that releases the buffer
    auto deleter
        = [comm, ptr = lease.ptr, deviceIndex = lease.device, generation = lease.generation](void*) noexcept
    {
        try
        {
            NCCLWindowAllocator::getInstance().releaseBuffer(*comm, ptr, deviceIndex, generation);
        }
        catch (std::exception const& e)
        {
            TLLM_LOG_WARNING("[createNCCLWindowTensor] Failed to release buffer %p on device %d: %s", ptr, deviceIndex,
                e.what());
        }
        catch (...)
        {
            TLLM_LOG_WARNING(
                "[createNCCLWindowTensor] Failed to release buffer %p on device %d", ptr, deviceIndex);
        }
    };

    // Create tensor from the buffer
    auto tensor = torch::from_blob(lease.ptr, shape, strides_vec, deleter, torch::dtype(dtype).device(device));

    return std::make_pair(tensor, static_cast<NCCLWindowBuffer const&>(lease));
}

inline std::pair<torch::Tensor, NCCLWindowBuffer> createNCCLWindowTensor(
    std::shared_ptr<ncclComm_t> comm, at::IntArrayRef shape, torch::ScalarType dtype)
{
    return createNCCLWindowTensor(
        std::move(comm), shape, dtype, c10::Device(c10::DeviceType::CUDA, c10::cuda::current_device()));
}

#endif // NCCL_VERSION_CODE >= NCCL_VERSION(2, 28, 0)

} // namespace common::nccl_util

TRTLLM_NAMESPACE_END

#endif // ENABLE_MULTI_DEVICE
