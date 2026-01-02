/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/common/ncclUtils.h"

#if ENABLE_MULTI_DEVICE

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/common/opUtils.h"
#include <iostream>
#include <limits>
#include <stdexcept>

namespace tensorrt_llm::common::nccl_util
{

//==============================================================================
// NcclCommResourceManager Implementation
//==============================================================================

NcclCommResourceManager& NcclCommResourceManager::getInstance() noexcept
{
    static NcclCommResourceManager instance;
    return instance;
}

void NcclCommResourceManager::registerResource(ncclComm_t comm, ResourceCleanupFunc cleanup, char const* debugName)
{
    if (!comm)
    {
        TLLM_LOG_WARNING("[NCCLUtil] Attempted to register resource for null NCCL comm");
        return;
    }

    std::lock_guard<std::mutex> lock(mMutex);
    auto& resources = mCommResources[comm];
    resources.emplace_back(std::move(cleanup), debugName ? debugName : "unnamed");

    TLLM_LOG_TRACE("[NCCLUtil] Registered resource '%s' for NCCL comm %p (total: %zu)",
        debugName ? debugName : "unnamed", static_cast<void*>(comm), resources.size());
}

void NcclCommResourceManager::cleanupResources(ncclComm_t comm) noexcept
{
    if (!comm)
    {
        return;
    }

    std::vector<ResourceEntry> resourcesToClean;

    {
        std::lock_guard<std::mutex> lock(mMutex);
        auto it = mCommResources.find(comm);
        if (it == mCommResources.end())
        {
            // Nothing registered for this comm, nothing to clean up
            return;
        }

        // Move resources out (preserves order) and remove from map
        resourcesToClean = std::move(it->second);
        mCommResources.erase(it);

        TLLM_LOG_TRACE(
            "[NCCLUtil] Cleaning up %zu resources for NCCL comm %p", resourcesToClean.size(), static_cast<void*>(comm));
    }

    // Clean up outside the lock to avoid deadlocks if cleanup functions try to access the manager
    // Order is preserved: resources are cleaned up in registration order
    for (auto& [cleanup, name] : resourcesToClean)
    {
        try
        {
            TLLM_LOG_TRACE(
                "[NCCLUtil] Cleaning up resource '%s' for NCCL comm %p", name.c_str(), static_cast<void*>(comm));
            cleanup();
        }
        catch (std::exception const& e)
        {
            TLLM_LOG_ERROR("[NCCLUtil] Exception during cleanup of resource '%s' for NCCL comm %p: %s", name.c_str(),
                static_cast<void*>(comm), e.what());
        }
        catch (...)
        {
            TLLM_LOG_ERROR("[NCCLUtil] Unknown exception during cleanup of resource '%s' for NCCL comm %p",
                name.c_str(), static_cast<void*>(comm));
        }
    }
}

bool NcclCommResourceManager::hasResources(ncclComm_t comm) const noexcept
{
    std::lock_guard<std::mutex> lock(mMutex);
    return mCommResources.find(comm) != mCommResources.end();
}

size_t NcclCommResourceManager::getResourceCount(ncclComm_t comm) const noexcept
{
    std::lock_guard<std::mutex> lock(mMutex);
    auto it = mCommResources.find(comm);
    return it != mCommResources.end() ? it->second.size() : 0;
}

//==============================================================================
// NCCLHelper Implementation
//==============================================================================

NCCLHelper& NCCLHelper::getInstance()
{
    static NCCLHelper instance;
    return instance;
}

NCCLHelper::NCCLHelper()
    : mLibraryHandle(nullptr)
    , mNCCLCommWindowRegister(nullptr)
    , mNCCLMemAlloc(nullptr)
    , mIsLoaded(false)
{
    loadNCCLLibrary();
}

NCCLHelper::~NCCLHelper()
{
    if (mLibraryHandle)
    {
#ifdef _WIN32
        FreeLibrary(mLibraryHandle);
#else
        dlclose(mLibraryHandle);
#endif
        mLibraryHandle = nullptr;
    }
}

void NCCLHelper::loadNCCLLibrary()
{
    try
    {
#ifdef _WIN32
        char const* libraryNames[] = {"nccl.dll"};
#else
        char const* libraryNames[] = {"libnccl.so"};
#endif

        for (auto const* name : libraryNames)
        {
            mLibraryHandle = loadLibraryHandle(name);
            if (mLibraryHandle)
            {
                TLLM_LOG_INFO("Successfully loaded NCCL library: %s", name);
                break;
            }
        }

        if (!mLibraryHandle)
        {
            TLLM_LOG_WARNING("Failed to load NCCL library");
            return;
        }

        // Load the required symbols
        mNCCLCommWindowRegister
            = reinterpret_cast<ncclCommWindowRegisterFunc>(getSymbolAddress(mLibraryHandle, "ncclCommWindowRegister"));

        mNCCLMemAlloc = reinterpret_cast<ncclMemAllocFunc>(getSymbolAddress(mLibraryHandle, "ncclMemAlloc"));

        if (mNCCLCommWindowRegister == nullptr)
        {
            TLLM_LOG_WARNING("Failed to load ncclCommWindowRegister symbol, NCCL symmetric will not be supported.");
        }

        if (mNCCLMemAlloc == nullptr)
        {
            TLLM_LOG_WARNING("Failed to load ncclMemAlloc symbol, NCCL symmetric will not be supported.");
        }

        if (mNCCLCommWindowRegister != nullptr && mNCCLMemAlloc != nullptr)
        {
            mIsLoaded = true;
        }
        else
        {
            TLLM_LOG_WARNING(
                "Failed to load required NCCL symbols (both ncclCommWindowRegister and ncclMemAlloc are required)");
        }
    }
    catch (std::exception const& e)
    {
        TLLM_LOG_WARNING("Exception while loading NCCL library: %s", e.what());
    }
}

void* NCCLHelper::loadLibraryHandle(char const* libName)
{
#ifdef _WIN32
    return LoadLibraryA(libName);
#else
    return dlopen(libName, RTLD_LAZY | RTLD_GLOBAL);
#endif
}

void* NCCLHelper::getSymbolAddress(void* handle, char const* symbolName)
{
    if (!handle)
    {
        return nullptr;
    }

#ifdef _WIN32
    return GetProcAddress(static_cast<HMODULE>(handle), symbolName);
#else
    return dlsym(handle, symbolName);
#endif
}

NCCLHelper::ncclCommWindowRegisterFunc NCCLHelper::getNCCLCommWindowRegister()
{
    return mNCCLCommWindowRegister;
}

NCCLHelper::ncclMemAllocFunc NCCLHelper::getNCCLMemAlloc()
{
    return mNCCLMemAlloc;
}

bool NCCLHelper::isLoaded() const
{
    return mIsLoaded;
}

//==============================================================================
// NCCLWindowAllocator Implementation
//==============================================================================

NCCLWindowAllocator& NCCLWindowAllocator::getInstance()
{
    static NCCLWindowAllocator instance;
    return instance;
}

NCCLWindowBuffer NCCLWindowAllocator::requestBuffer(ncclComm_t comm, size_t size)
{
    int rank = -1;
    if (comm != nullptr)
    {
        ncclCommUserRank(comm, &rank);
    }
    std::cout << "[NCCLWindowAllocator::requestBuffer] Rank " << rank
              << ": Starting requestBuffer, comm=" << static_cast<void*>(comm) << ", size=" << size << std::endl
              << std::flush;

    TLLM_CHECK_WITH_INFO(comm != nullptr, "NCCL communicator cannot be null");
    TLLM_CHECK_WITH_INFO(size > 0, "Buffer size must be greater than 0");

    int handle;
    {
        std::cout << "[NCCLWindowAllocator::requestBuffer] Rank " << rank << ": Acquiring mMutex" << std::endl
                  << std::flush;
        std::lock_guard<std::mutex> lock(mMutex);
        std::cout << "[NCCLWindowAllocator::requestBuffer] Rank " << rank << ": Acquired mMutex" << std::endl
                  << std::flush;

        // Register cleanup callback for this communicator if not already registered
        // This is cheap even if no buffers exist yet - cleanup will just return early
        std::cout << "[NCCLWindowAllocator::requestBuffer] Rank " << rank << ": Calling registerBufferCleanup"
                  << std::endl
                  << std::flush;
        registerBufferCleanup(comm);
        std::cout << "[NCCLWindowAllocator::requestBuffer] Rank " << rank << ": registerBufferCleanup completed"
                  << std::endl
                  << std::flush;

        // Check if we have an available buffer of at least the requested size for this communicator
        // Use deterministic first-fit: find the first buffer that's >= requested size (by handle order)
        // This ensures all ranks reuse the same buffers in the same order
        auto& commBuffers = mBufferPool[comm];
        std::cout << "[NCCLWindowAllocator::requestBuffer] Rank " << rank
                  << ": Checking existing buffers, pool size=" << commBuffers.size() << std::endl
                  << std::flush;
        auto firstFit = commBuffers.end();

        for (auto it = commBuffers.begin(); it != commBuffers.end(); ++it)
        {
            if (!it->inUse && it->buffer.size >= size)
            {
                firstFit = it;
                break; // Use first fit for deterministic behavior
            }
        }

        if (firstFit != commBuffers.end())
        {
            firstFit->inUse = true;
            std::cout << "[NCCLWindowAllocator::requestBuffer] Rank " << rank
                      << ": Reusing buffer, handle=" << firstFit->buffer.handle << ", ptr=" << firstFit->buffer.ptr
                      << ", size=" << firstFit->buffer.size << std::endl
                      << std::flush;
            TLLM_LOG_TRACE(
                "[NCCLUtil] Reusing NCCL window buffer for comm %p: handle=%d, ptr=%p, size=%zu (requested: %zu)",
                static_cast<void*>(comm), firstFit->buffer.handle, firstFit->buffer.ptr, firstFit->buffer.size, size);
            return firstFit->buffer;
        }

        // No available buffer found, will allocate a new one
        handle = static_cast<int>(commBuffers.size());
        std::cout << "[NCCLWindowAllocator::requestBuffer] Rank " << rank
                  << ": No available buffer found, will allocate new, handle=" << handle << std::endl
                  << std::flush;
    }

    // Release the pool mutex before calling allocateAndRegisterBuffer (which will acquire NCCL op mutex)
    // We need to release mMutex here because allocateAndRegisterBuffer will acquire the NCCL op mutex
    // and we don't want to hold both mutexes at the same time
    std::cout << "[NCCLWindowAllocator::requestBuffer] Rank " << rank
              << ": Released mMutex, calling allocateAndRegisterBuffer" << std::endl
              << std::flush;
    TLLM_LOG_TRACE(
        "[NCCLUtil] Allocating new NCCL window buffer for comm %p, size=%zu", static_cast<void*>(comm), size);
    NCCLWindowBuffer buffer = allocateAndRegisterBuffer(comm, size, handle);
    std::cout << "[NCCLWindowAllocator::requestBuffer] Rank " << rank
              << ": allocateAndRegisterBuffer completed, buffer.ptr=" << buffer.ptr << ", buffer.size=" << buffer.size
              << std::endl
              << std::flush;

    {
        // Re-acquire the lock for push_back
        std::cout << "[NCCLWindowAllocator::requestBuffer] Rank " << rank << ": Re-acquiring mMutex for push_back"
                  << std::endl
                  << std::flush;
        std::lock_guard<std::mutex> relock(mMutex);
        mBufferPool[comm].push_back({buffer, true});
        std::cout << "[NCCLWindowAllocator::requestBuffer] Rank " << rank << ": Added buffer to pool" << std::endl
                  << std::flush;
    }

    std::cout << "[NCCLWindowAllocator::requestBuffer] Rank " << rank << ": requestBuffer completed successfully"
              << std::endl
              << std::flush;
    return buffer;
}

NCCLWindowBuffer NCCLWindowAllocator::searchBuffer(ncclComm_t comm, void* ptr) const
{
    int rank = -1;
    if (comm != nullptr)
    {
        ncclCommUserRank(comm, &rank);
    }
    std::cout << "[NCCLWindowAllocator::searchBuffer] Rank " << rank
              << ": Starting searchBuffer, comm=" << static_cast<void*>(comm) << ", ptr=" << ptr << std::endl
              << std::flush;

    if (!comm || !ptr)
    {
        std::cout << "[NCCLWindowAllocator::searchBuffer] Rank " << rank
                  << ": Invalid comm or ptr, returning invalid buffer" << std::endl
                  << std::flush;
        return NCCLWindowBuffer();
    }

    std::cout << "[NCCLWindowAllocator::searchBuffer] Rank " << rank << ": Acquiring mMutex" << std::endl << std::flush;
    std::lock_guard<std::mutex> lock(mMutex);
    auto result = searchBufferLocked(comm, ptr);
    std::cout << "[NCCLWindowAllocator::searchBuffer] Rank " << rank
              << ": searchBufferLocked completed, isValid=" << result.isValid() << ", ptr=" << result.ptr << std::endl
              << std::flush;
    return result;
}

void NCCLWindowAllocator::releaseBuffer(ncclComm_t comm, void* ptr)
{
    if (!comm || !ptr)
    {
        return;
    }

    std::lock_guard<std::mutex> lock(mMutex);
    auto commIt = mBufferPool.find(comm);
    if (commIt == mBufferPool.end())
    {
        TLLM_LOG_WARNING(
            "[NCCLUtil] Attempted to release buffer %p for unknown comm %p", ptr, static_cast<void*>(comm));
        return;
    }

    for (auto& entry : commIt->second)
    {
        if (entry.buffer.ptr == ptr)
        {
            entry.inUse = false;
            TLLM_LOG_TRACE("[NCCLUtil] Released NCCL window buffer for comm %p: ptr=%p", static_cast<void*>(comm), ptr);
            return;
        }
    }

    TLLM_LOG_WARNING("[NCCLUtil] Attempted to release unknown buffer %p for comm %p", ptr, static_cast<void*>(comm));
}

ncclWindow_t NCCLWindowAllocator::getWindow(ncclComm_t comm, void* ptr) const
{
    std::lock_guard<std::mutex> lock(mMutex);
    NCCLWindowBuffer buffer = searchBufferLocked(comm, ptr);
    return buffer.isValid() ? buffer.window : nullptr;
}

size_t NCCLWindowAllocator::getSize(ncclComm_t comm, void* ptr) const
{
    std::lock_guard<std::mutex> lock(mMutex);
    NCCLWindowBuffer buffer = searchBufferLocked(comm, ptr);
    return buffer.isValid() ? buffer.size : 0;
}

NCCLWindowBuffer NCCLWindowAllocator::getBufferInfo(ncclComm_t comm, void* ptr) const
{
    std::lock_guard<std::mutex> lock(mMutex);
    return searchBufferLocked(comm, ptr);
}

size_t NCCLWindowAllocator::getBufferCount(ncclComm_t comm) const
{
    std::lock_guard<std::mutex> lock(mMutex);
    auto commIt = mBufferPool.find(comm);
    return commIt != mBufferPool.end() ? commIt->second.size() : 0;
}

size_t NCCLWindowAllocator::getBufferInUseCount(ncclComm_t comm) const
{
    std::lock_guard<std::mutex> lock(mMutex);
    auto commIt = mBufferPool.find(comm);
    if (commIt == mBufferPool.end())
    {
        return 0;
    }

    size_t count = 0;
    for (auto const& entry : commIt->second)
    {
        if (entry.inUse)
        {
            ++count;
        }
    }
    return count;
}

bool NCCLWindowAllocator::isCommValid(ncclComm_t comm) const noexcept
{
    // Simply check for null - all non-null comms are valid
    // We don't track cleaned-up comms because NCCL can reuse memory addresses,
    // making pointer-based tracking unreliable. New comms will be registered when used.
    return comm != nullptr;
}

std::mutex& NCCLWindowAllocator::getNcclOpMutex(ncclComm_t comm) const
{
    std::lock_guard<std::mutex> lock(mMutex);
    auto it = mNcclOpMutexes.find(comm);
    if (it == mNcclOpMutexes.end())
    {
        mNcclOpMutexes[comm] = std::make_unique<std::mutex>();
        return *mNcclOpMutexes[comm];
    }
    return *it->second;
}

NCCLWindowBuffer NCCLWindowAllocator::allocateAndRegisterBuffer(ncclComm_t comm, size_t size, int handle)
{
    int rank = -1;
    if (comm != nullptr)
    {
        ncclCommUserRank(comm, &rank);
    }
    std::cout << "[NCCLWindowAllocator::allocateAndRegisterBuffer] Rank " << rank
              << ": Starting allocateAndRegisterBuffer, comm=" << static_cast<void*>(comm) << ", size=" << size
              << ", handle=" << handle << std::endl
              << std::flush;

    // Serialize NCCL operations - NCCL is not thread-safe
    // This mutex ensures only one thread per communicator can perform NCCL operations at a time
    std::cout << "[NCCLWindowAllocator::allocateAndRegisterBuffer] Rank " << rank << ": Acquiring NCCL op mutex"
              << std::endl
              << std::flush;
    std::lock_guard<std::mutex> ncclLock(getNcclOpMutex(comm));
    std::cout << "[NCCLWindowAllocator::allocateAndRegisterBuffer] Rank " << rank << ": Acquired NCCL op mutex"
              << std::endl
              << std::flush;

    NCCLWindowBuffer buffer;
    buffer.handle = handle;

    // Get NCCL helper for dynamic symbol loading
    std::cout << "[NCCLWindowAllocator::allocateAndRegisterBuffer] Rank " << rank << ": Getting NCCLHelper instance"
              << std::endl
              << std::flush;
    auto& ncclHelper = NCCLHelper::getInstance();
    if (!ncclHelper.isLoaded())
    {
        std::cout << "[NCCLWindowAllocator::allocateAndRegisterBuffer] Rank " << rank
                  << ": ERROR - NCCL library not loaded" << std::endl
                  << std::flush;
        TLLM_THROW("NCCL library could not be loaded for dynamic symbol access");
    }
    std::cout << "[NCCLWindowAllocator::allocateAndRegisterBuffer] Rank " << rank << ": NCCLHelper loaded successfully"
              << std::endl
              << std::flush;

    auto ncclMemAllocFunc = ncclHelper.getNCCLMemAlloc();
    auto ncclCommWindowRegisterFunc = ncclHelper.getNCCLCommWindowRegister();

    // Defensive checks: both function pointers must be non-null
    if (ncclMemAllocFunc == nullptr)
    {
        std::cout << "[NCCLWindowAllocator::allocateAndRegisterBuffer] Rank " << rank
                  << ": ERROR - ncclMemAlloc is null" << std::endl
                  << std::flush;
        TLLM_THROW("ncclMemAlloc function pointer is null, cannot allocate NCCL window buffer");
    }

    if (ncclCommWindowRegisterFunc == nullptr)
    {
        std::cout << "[NCCLWindowAllocator::allocateAndRegisterBuffer] Rank " << rank
                  << ": ERROR - ncclCommWindowRegister is null" << std::endl
                  << std::flush;
        TLLM_THROW("ncclCommWindowRegister function pointer is null, cannot register NCCL window buffer");
    }
    std::cout << "[NCCLWindowAllocator::allocateAndRegisterBuffer] Rank " << rank << ": Function pointers validated"
              << std::endl
              << std::flush;

    // Allocate device memory using ncclMemAlloc
    std::cout << "[NCCLWindowAllocator::allocateAndRegisterBuffer] Rank " << rank
              << ": Calling ncclMemAlloc, size=" << size << std::endl
              << std::flush;
    ncclResult_t allocResult = ncclMemAllocFunc(&buffer.ptr, size);
    if (allocResult != ncclSuccess)
    {
        std::cout << "[NCCLWindowAllocator::allocateAndRegisterBuffer] Rank " << rank
                  << ": ERROR - ncclMemAlloc failed with error=" << allocResult << std::endl
                  << std::flush;
        TLLM_THROW("ncclMemAlloc failed with error: %d", allocResult);
    }
    buffer.size = size;
    std::cout << "[NCCLWindowAllocator::allocateAndRegisterBuffer] Rank " << rank
              << ": ncclMemAlloc completed, buffer.ptr=" << buffer.ptr << ", buffer.size=" << buffer.size << std::endl
              << std::flush;

    // Synchronize all ranks before the collective ncclCommWindowRegister call.
    // ncclCommWindowRegister is a collective operation that requires all ranks to participate.
    // Without synchronization, if ranks are out of sync (e.g., after autotuning), the call will hang.
    // Note: We already hold the NCCL op mutex, so we inline the synchronization here
    int nRanks;
    std::cout << "[NCCLWindowAllocator::allocateAndRegisterBuffer] Rank " << rank
              << ": Calling ncclCommCount (before barrier)" << std::endl
              << std::flush;
    NCCLCHECK_THROW(ncclCommCount(comm, &nRanks));
    std::cout << "[NCCLWindowAllocator::allocateAndRegisterBuffer] Rank " << rank
              << ": ncclCommCount completed, nRanks=" << nRanks << std::endl
              << std::flush;

    if (nRanks > 1)
    {
        std::cout << "[NCCLWindowAllocator::allocateAndRegisterBuffer] Rank " << rank
                  << ": nRanks > 1, performing barrier allreduce (before registration)" << std::endl
                  << std::flush;
        // Create a dummy buffer for the barrier allreduce
        void* dummyBuffer;
        TLLM_CUDA_CHECK(cudaMalloc(&dummyBuffer, sizeof(int)));
        int dummyValue = 0;
        TLLM_CUDA_CHECK(cudaMemcpy(dummyBuffer, &dummyValue, sizeof(int), cudaMemcpyHostToDevice));
        // Use the default stream (nullptr) for the barrier allreduce
        // Perform a dummy allreduce as a barrier to synchronize all ranks
        std::cout << "[NCCLWindowAllocator::allocateAndRegisterBuffer] Rank " << rank
                  << ": About to call ncclAllReduce (barrier before registration)" << std::endl
                  << std::flush;
        NCCLCHECK_THROW(ncclAllReduce(dummyBuffer, dummyBuffer, 1, ncclInt, ncclSum, comm, nullptr));
        std::cout << "[NCCLWindowAllocator::allocateAndRegisterBuffer] Rank " << rank
                  << ": ncclAllReduce (barrier before registration) completed" << std::endl
                  << std::flush;
        // Synchronize the default stream to ensure the allreduce completes
        std::cout << "[NCCLWindowAllocator::allocateAndRegisterBuffer] Rank " << rank
                  << ": Synchronizing default stream" << std::endl
                  << std::flush;
        TLLM_CUDA_CHECK(cudaStreamSynchronize(nullptr));
        TLLM_CUDA_CHECK(cudaFree(dummyBuffer));
        std::cout << "[NCCLWindowAllocator::allocateAndRegisterBuffer] Rank " << rank
                  << ": Barrier (before registration) completed" << std::endl
                  << std::flush;
    }

    // Register the buffer with NCCL as a window
    std::cout << "[NCCLWindowAllocator::allocateAndRegisterBuffer] Rank " << rank
              << ": About to call ncclCommWindowRegister, buffer.ptr=" << buffer.ptr << ", size=" << size << std::endl
              << std::flush;
    ncclResult_t regResult
        = ncclCommWindowRegisterFunc(comm, buffer.ptr, size, &buffer.window, NCCL_WIN_COLL_SYMMETRIC);
    if (regResult != ncclSuccess)
    {
        std::cout << "[NCCLWindowAllocator::allocateAndRegisterBuffer] Rank " << rank
                  << ": ERROR - ncclCommWindowRegister failed with error=" << regResult << std::endl
                  << std::flush;
        ncclMemFree(buffer.ptr);
        TLLM_THROW("ncclCommWindowRegister failed with error: %d", regResult);
    }
    std::cout << "[NCCLWindowAllocator::allocateAndRegisterBuffer] Rank " << rank
              << ": ncclCommWindowRegister completed, buffer.window=" << static_cast<void*>(buffer.window) << std::endl
              << std::flush;

    // Synchronize all ranks after the collective ncclCommWindowRegister call.
    // This ensures all ranks have completed registration before proceeding.
    if (nRanks > 1)
    {
        std::cout << "[NCCLWindowAllocator::allocateAndRegisterBuffer] Rank " << rank
                  << ": nRanks > 1, performing barrier allreduce (after registration)" << std::endl
                  << std::flush;
        // Create a dummy buffer for the barrier allreduce
        void* dummyBuffer;
        TLLM_CUDA_CHECK(cudaMalloc(&dummyBuffer, sizeof(int)));
        int dummyValue = 0;
        TLLM_CUDA_CHECK(cudaMemcpy(dummyBuffer, &dummyValue, sizeof(int), cudaMemcpyHostToDevice));
        // Perform a dummy allreduce as a barrier to synchronize all ranks
        std::cout << "[NCCLWindowAllocator::allocateAndRegisterBuffer] Rank " << rank
                  << ": About to call ncclAllReduce (barrier after registration)" << std::endl
                  << std::flush;
        NCCLCHECK_THROW(ncclAllReduce(dummyBuffer, dummyBuffer, 1, ncclInt, ncclSum, comm, nullptr));
        std::cout << "[NCCLWindowAllocator::allocateAndRegisterBuffer] Rank " << rank
                  << ": ncclAllReduce (barrier after registration) completed" << std::endl
                  << std::flush;
        // Synchronize the default stream to ensure the allreduce completes
        std::cout << "[NCCLWindowAllocator::allocateAndRegisterBuffer] Rank " << rank
                  << ": Synchronizing default stream" << std::endl
                  << std::flush;
        TLLM_CUDA_CHECK(cudaStreamSynchronize(nullptr));
        TLLM_CUDA_CHECK(cudaFree(dummyBuffer));
        std::cout << "[NCCLWindowAllocator::allocateAndRegisterBuffer] Rank " << rank
                  << ": Barrier (after registration) completed" << std::endl
                  << std::flush;
    }

    std::cout << "[NCCLWindowAllocator::allocateAndRegisterBuffer] Rank " << rank
              << ": allocateAndRegisterBuffer completed successfully, handle=" << handle << ", ptr=" << buffer.ptr
              << ", size=" << buffer.size << ", window=" << static_cast<void*>(buffer.window) << std::endl
              << std::flush;
    TLLM_LOG_TRACE("[NCCLUtil] Allocated and registered NCCL window buffer: handle=%d, ptr=%p, size=%zu, window=%p",
        handle, buffer.ptr, size, static_cast<void*>(buffer.window));

    return buffer;
}

NCCLWindowBuffer NCCLWindowAllocator::searchBufferLocked(ncclComm_t comm, void* ptr) const
{
    auto commIt = mBufferPool.find(comm);
    if (commIt == mBufferPool.end())
    {
        return NCCLWindowBuffer();
    }

    for (auto const& entry : commIt->second)
    {
        if (entry.buffer.ptr == ptr)
        {
            return entry.buffer;
        }
    }

    return NCCLWindowBuffer();
}

void NCCLWindowAllocator::registerBufferCleanup(ncclComm_t comm)
{
    // Don't register if already registered
    if (mRegisteredComms.find(comm) != mRegisteredComms.end())
    {
        return;
    }

    mRegisteredComms.insert(comm);

    // Register cleanup with the resource manager
    NcclCommResourceManager::getInstance().registerResource(
        comm, [this, comm]() { this->cleanupBuffersForComm(comm); }, "NCCLWindowAllocator");
}

void NCCLWindowAllocator::cleanupBuffersForComm(ncclComm_t comm) noexcept
{
    if (!comm)
    {
        return;
    }

    // Synchronize CUDA to ensure all operations using these buffers are complete
    // before we deregister windows and free memory
    cudaError_t cudaErr = cudaDeviceSynchronize();
    if (cudaErr != cudaSuccess)
    {
        TLLM_LOG_WARNING("[NCCLUtil] cudaDeviceSynchronize failed with error: %d before cleanup for comm %p", cudaErr,
            static_cast<void*>(comm));
        // Continue anyway - the sync failure might be from a previous error
    }

    std::lock_guard<std::mutex> lock(mMutex);

    // Check if we've already cleaned up this communicator
    if (mRegisteredComms.find(comm) == mRegisteredComms.end())
    {
        // Already cleaned up or never registered
        return;
    }

    auto commIt = mBufferPool.find(comm);
    if (commIt == mBufferPool.end())
    {
        // No buffers to clean up, but mark as cleaned
        mRegisteredComms.erase(comm);
        return;
    }

    TLLM_LOG_TRACE(
        "[NCCLUtil] Cleaning up %zu NCCL window buffers for comm %p", commIt->second.size(), static_cast<void*>(comm));

    // Check for buffers still in use - this shouldn't happen if cleanup is called properly,
    // but we log a warning if it does
    size_t inUseCount = 0;
    for (auto const& entry : commIt->second)
    {
        if (entry.inUse)
        {
            ++inUseCount;
        }
    }
    if (inUseCount > 0)
    {
        TLLM_LOG_WARNING(
            "[NCCLUtil] Cleaning up %zu buffers still marked as in-use for comm %p. "
            "This may indicate buffers weren't properly released before cleanup.",
            inUseCount, static_cast<void*>(comm));
    }

    for (auto& entry : commIt->second)
    {
        if (entry.buffer.isValid())
        {
            // Deregister the window - the communicator is still valid at this point
            // (cleanup happens before ncclCommDestroy), but we need to be careful
            // if buffers are still in use by active operations
            if (entry.buffer.window && comm)
            {
                // Note: Even if buffer is marked inUse, we must deregister since
                // the communicator is being destroyed. The communicator is valid,
                // but we should handle potential errors gracefully.
                ncclResult_t result = ncclCommWindowDeregister(comm, entry.buffer.window);
                if (result != ncclSuccess)
                {
                    TLLM_LOG_WARNING(
                        "[NCCLUtil] ncclCommWindowDeregister failed with error: %d for comm %p, "
                        "window %p (buffer inUse: %d)",
                        result, static_cast<void*>(comm), static_cast<void*>(entry.buffer.window), entry.inUse);
                }
            }

            // Free device memory using ncclMemFree
            // This should be safe even if deregister failed
            if (entry.buffer.ptr)
            {
                try
                {
                    ncclResult_t ncclResult = ncclMemFree(entry.buffer.ptr);
                    if (ncclResult != ncclSuccess)
                    {
                        TLLM_LOG_WARNING("[NCCLUtil] ncclMemFree failed with error: %d", ncclResult);
                    }
                }
                catch (...)
                {
                    TLLM_LOG_ERROR("[NCCLUtil] Exception during ncclMemFree for ptr %p", entry.buffer.ptr);
                }
            }

            TLLM_LOG_TRACE(
                "[NCCLUtil] Freed NCCL window buffer: ptr=%p, size=%zu", entry.buffer.ptr, entry.buffer.size);
        }
    }

    mBufferPool.erase(commIt);
    mRegisteredComms.erase(comm);
}

} // namespace tensorrt_llm::common::nccl_util

#endif // ENABLE_MULTI_DEVICE
