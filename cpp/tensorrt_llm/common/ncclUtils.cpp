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

#include "tensorrt_llm/common/ncclUtils.h"

#if ENABLE_MULTI_DEVICE

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/logger.h"
#include <limits>
#include <stdexcept>
#include <thread>

namespace
{

// RAII guard for cudaMalloc. Frees the pointer on destruction, logging a warning on failure.
struct CudaMallocGuard
{
    void* ptr{nullptr};

    explicit CudaMallocGuard(void* p) noexcept
        : ptr(p)
    {
    }

    ~CudaMallocGuard()
    {
        if (ptr)
        {
            TLLM_CUDA_CHECK_WARN(cudaFree(ptr));
        }
    }

    void* release() noexcept
    {
        void* p = ptr;
        ptr = nullptr;
        return p;
    }

    CudaMallocGuard(CudaMallocGuard const&) = delete;
    CudaMallocGuard& operator=(CudaMallocGuard const&) = delete;
};

// RAII guard for ncclMemAlloc. Frees the pointer on destruction, logging a warning on failure.
struct NcclMemGuard
{
    void* ptr{nullptr};

    explicit NcclMemGuard(void* p) noexcept
        : ptr(p)
    {
    }

    ~NcclMemGuard()
    {
        if (ptr)
        {
            TLLM_NCCL_CHECK_WARN(ncclMemFree(ptr));
        }
    }

    void* release() noexcept
    {
        void* p = ptr;
        ptr = nullptr;
        return p;
    }

    NcclMemGuard(NcclMemGuard const&) = delete;
    NcclMemGuard& operator=(NcclMemGuard const&) = delete;
};

} // namespace

namespace tensorrt_llm::common::nccl_util
{

namespace
{

#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 28, 0)
constexpr int kNcclWindowMinRuntimeVersion = NCCL_VERSION(2, 28, 0);
constexpr int kNcclGb10WindowFixedVersion = NCCL_VERSION(2, 30, 4);
constexpr int kGb10RealSmVersion = 121;

bool isGb10Platform(int realSmVersion, bool isIntegrated)
{
    return realSmVersion == kGb10RealSmVersion && isIntegrated;
}
#endif

bool queryNcclWindowSupported(int device)
{
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 28, 0)
    c10::cuda::CUDAGuard deviceGuard(static_cast<c10::DeviceIndex>(device));

    int version = 0;
    if (ncclGetVersion(&version) != ncclSuccess)
    {
        TLLM_LOG_WARNING("[NCCLUtil] Failed to query NCCL runtime version; falling back to regular tensors.");
        return false;
    }

    if (version < kNcclWindowMinRuntimeVersion)
    {
        TLLM_LOG_WARNING(
            "[NCCLUtil] NCCL runtime version %d.%d.%d does not support window buffers; falling back to regular "
            "tensors.",
            version / 10000, (version % 10000) / 100, version % 100);
        return false;
    }

    if (version >= kNcclGb10WindowFixedVersion)
    {
        return true;
    }

    int isIntegrated = 0;
    cudaError_t const integratedErr = cudaDeviceGetAttribute(&isIntegrated, cudaDevAttrIntegrated, device);
    if (integratedErr != cudaSuccess)
    {
        TLLM_LOG_WARNING(
            "[NCCLUtil] Failed to query CUDA integrated-device attribute for device %d while checking NCCL window "
            "support: %s; falling back to regular tensors.",
            device, cudaGetErrorString(integratedErr));
        return false;
    }

    int realSmVersion = -1;
    try
    {
        realSmVersion = tensorrt_llm::common::getSMVersion(/*queryRealSmArch=*/true);
    }
    catch (std::exception const& e)
    {
        TLLM_LOG_WARNING(
            "[NCCLUtil] Failed to query real CUDA SM version while checking NCCL window support: %s; falling back "
            "to regular tensors.",
            e.what());
        return false;
    }

    bool const supported = !isGb10Platform(realSmVersion, isIntegrated != 0);
    if (!supported)
    {
        TLLM_LOG_WARNING(
            "[NCCLUtil] Disabling NCCL window buffers on integrated SM %d with NCCL runtime version %d.%d.%d; "
            "GB10 requires NCCL 2.30.4 or newer for symmetric window registration.",
            realSmVersion, version / 10000, (version % 10000) / 100, version % 100);
    }
    return supported;
#else
    (void) device;
    return false;
#endif
}

} // namespace

bool isNcclWindowSupportedForPlatform(int realSmVersion, bool isIntegrated, int ncclRuntimeVersion)
{
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 28, 0)
    if (ncclRuntimeVersion < kNcclWindowMinRuntimeVersion)
    {
        return false;
    }

    return !(ncclRuntimeVersion < kNcclGb10WindowFixedVersion && isGb10Platform(realSmVersion, isIntegrated));
#else
    (void) realSmVersion;
    (void) isIntegrated;
    (void) ncclRuntimeVersion;
    return false;
#endif
}

bool isNcclWindowSupported()
{
    return isNcclWindowSupported(c10::cuda::current_device());
}

bool isNcclWindowSupported(int device)
{
    static std::mutex supportCheckMutex;
    static std::unordered_map<int, bool> windowSupportByDevice;

    std::lock_guard<std::mutex> lock(supportCheckMutex);
    auto const [it, inserted] = windowSupportByDevice.try_emplace(device, false);
    if (inserted)
    {
        it->second = queryNcclWindowSupported(device);
    }
    return it->second;
}

//==============================================================================
// NcclCommResourceManager Implementation
//==============================================================================

NcclCommResourceManager& NcclCommResourceManager::getInstance() noexcept
{
    static NcclCommResourceManager instance;
    return instance;
}

NcclCommResourceManager::~NcclCommResourceManager()
{
    // Mark that we're in destruction to prevent cleanup attempts from deleters
    // that may run during static destruction
    mIsDestroying.store(true, std::memory_order_release);

    // Proactively clean up all resources before destruction
    // This ensures cleanup happens in a controlled manner before static destruction
    std::vector<std::pair<ncclComm_t, std::vector<ResourceEntry>>> allResources;

    {
        std::lock_guard<std::mutex> lock(mMutex);
        // Move all resources out of the map
        allResources.reserve(mCommResources.size());
        for (auto& [comm, resources] : mCommResources)
        {
            allResources.emplace_back(comm, std::move(resources));
        }
        mCommResources.clear();
    }

    // Clean up all resources outside the lock
    // Note: We don't call ncclCommDestroy here - that's the responsibility
    // of the shared_ptr deleter. We just clean up registered resources.
    for (auto& [comm, resources] : allResources)
    {
        for (auto& [cleanup, name] : resources)
        {
            try
            {
                cleanup();
            }
            catch (...)
            {
                // Ignore exceptions during destruction
            }
        }
    }
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

    // Check if we're in the process of being destroyed
    // If so, skip cleanup - the destructor will handle it proactively
    if (mIsDestroying.load(std::memory_order_acquire))
    {
        return;
    }

    std::vector<ResourceEntry> resourcesToClean;

    {
        // During static destruction, mutex and logging may not be safe.
        // Use try-catch to handle any issues gracefully.
        try
        {
            std::lock_guard<std::mutex> lock(mMutex);

            // Double-check after acquiring lock (destruction may have started)
            if (mIsDestroying.load(std::memory_order_acquire))
            {
                return;
            }

            auto it = mCommResources.find(comm);
            if (it == mCommResources.end())
            {
                // Nothing registered for this comm, nothing to clean up
                return;
            }

            // Move resources out (preserves order) and remove from map
            resourcesToClean = std::move(it->second);
            mCommResources.erase(it);

            // Logging may fail during static destruction, so wrap in try-catch
            try
            {
                TLLM_LOG_TRACE("[NCCLUtil] Cleaning up %zu resources for NCCL comm %p", resourcesToClean.size(),
                    static_cast<void*>(comm));
            }
            catch (...)
            {
                // Ignore logging failures during static destruction
            }
        }
        catch (...)
        {
            // If mutex access fails during static destruction, just return.
            // This prevents segfaults when the singleton is being destroyed.
            return;
        }
    }

    // Clean up outside the lock to avoid deadlocks if cleanup functions try to access the manager
    // Order is preserved: resources are cleaned up in registration order
    for (auto& [cleanup, name] : resourcesToClean)
    {
        try
        {
            // Logging may fail during static destruction, so wrap in try-catch
            try
            {
                TLLM_LOG_TRACE(
                    "[NCCLUtil] Cleaning up resource '%s' for NCCL comm %p", name.c_str(), static_cast<void*>(comm));
            }
            catch (...)
            {
                // Ignore logging failures during static destruction
            }
            cleanup();
        }
        catch (std::exception const& e)
        {
            try
            {
                TLLM_LOG_ERROR("[NCCLUtil] Exception during cleanup of resource '%s' for NCCL comm %p: %s",
                    name.c_str(), static_cast<void*>(comm), e.what());
            }
            catch (...)
            {
                // Ignore logging failures during static destruction
            }
        }
        catch (...)
        {
            try
            {
                TLLM_LOG_ERROR("[NCCLUtil] Unknown exception during cleanup of resource '%s' for NCCL comm %p",
                    name.c_str(), static_cast<void*>(comm));
            }
            catch (...)
            {
                // Ignore logging failures during static destruction
            }
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
// NCCLWindowAllocator Implementation
//==============================================================================

#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 28, 0)

thread_local std::unique_ptr<NCCLWindowAllocator::CaptureState> NCCLWindowAllocator::mActiveCapture;

NCCLWindowAllocator& NCCLWindowAllocator::getInstance()
{
    static NCCLWindowAllocator instance;
    return instance;
}

int NCCLWindowAllocator::resolveDevice(int device)
{
    if (device >= 0)
    {
        return device;
    }

    int currentDevice = -1;
    TLLM_CUDA_CHECK(cudaGetDevice(&currentDevice));
    return currentDevice;
}

uint64_t NCCLWindowAllocator::createReuseDomain(int device)
{
    device = resolveDevice(device);
    c10::cuda::CUDAGuard deviceGuard(static_cast<c10::DeviceIndex>(device));

    auto domain = std::make_shared<ReuseDomain>();
    domain->id = mNextDomainId.fetch_add(1, std::memory_order_relaxed);
    domain->device = device;
    domain->replayStream = at::cuda::getCurrentCUDAStream(device).stream();

    std::lock_guard<std::mutex> lock(mMutex);
    mReuseDomains.emplace(domain->id, domain);
    TLLM_LOG_DEBUG("[NCCLUtil] Created window reuse domain %llu on device %d, replay stream %p",
        static_cast<unsigned long long>(domain->id), device, static_cast<void*>(domain->replayStream));
    return domain->id;
}

uint64_t NCCLWindowAllocator::beginCapture(uint64_t domainId)
{
    TLLM_CHECK_WITH_INFO(!mActiveCapture, "An NCCL window capture scope is already active on this thread");

    std::shared_ptr<ReuseDomain> domain;
    {
        std::lock_guard<std::mutex> lock(mMutex);
        auto const domainIt = mReuseDomains.find(domainId);
        TLLM_CHECK_WITH_INFO(domainIt != mReuseDomains.end(), "Unknown NCCL window reuse domain %llu",
            static_cast<unsigned long long>(domainId));
        domain = domainIt->second;
        TLLM_CHECK_WITH_INFO(!domain->closing, "NCCL window reuse domain %llu is closing",
            static_cast<unsigned long long>(domainId));
    }

    c10::cuda::CUDAGuard deviceGuard(static_cast<c10::DeviceIndex>(domain->device));
    cudaStream_t const stream = at::cuda::getCurrentCUDAStream(domain->device).stream();
    cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
    unsigned long long captureId = 0;
    cudaGraph_t graph = nullptr;
    TLLM_CUDA_CHECK(cudaStreamGetCaptureInfo_v2(stream, &status, &captureId, &graph, nullptr, nullptr));
    TLLM_CHECK_WITH_INFO(status == cudaStreamCaptureStatusActive && graph != nullptr,
        "NCCL window capture scope must begin inside an active CUDA graph capture");

    auto capture = std::make_unique<CaptureState>();
    capture->domain = std::move(domain);
    capture->captureId = captureId;
    capture->graph = graph;
    capture->captureStream = stream;
    mActiveCapture = std::move(capture);
    return captureId;
}

void NCCLWindowAllocator::endCapture(uint64_t captureId)
{
    TLLM_CHECK_WITH_INFO(mActiveCapture && mActiveCapture->captureId == captureId,
        "Mismatched NCCL window capture scope end for capture %llu", static_cast<unsigned long long>(captureId));

    {
        std::lock_guard<std::mutex> lock(mMutex);
        for (auto* entry : mActiveCapture->touchedEntries)
        {
            if (entry->inUse && entry->lastCaptureId == captureId)
            {
                // A lease that survives the captured body may be a graph
                // output or persistent state. Do not temporally alias it.
                entry->persistent = true;
            }
        }
    }

    mActiveCapture.reset();
}

void NCCLWindowAllocator::graphBindingDestructor(void* userData)
{
    auto* binding = static_cast<GraphBindingRecord*>(userData);
    binding->retired.store(true, std::memory_order_release);
}

void NCCLWindowAllocator::drainRetiredBindingsLocked(ReuseDomain& domain)
{
    for (auto& binding : domain.bindings)
    {
        if (!binding->accounted && binding->retired.load(std::memory_order_acquire))
        {
            binding->accounted = true;
            TLLM_CHECK(domain.liveBindings > 0);
            --domain.liveBindings;
        }
    }
}

void NCCLWindowAllocator::ensureGraphBindingLocked(CaptureState& capture)
{
    if (capture.graphBound)
    {
        return;
    }

    auto& domain = *capture.domain;
    TLLM_CHECK_WITH_INFO(!domain.closing, "NCCL window reuse domain %llu is closing",
        static_cast<unsigned long long>(domain.id));

    auto binding = std::make_unique<GraphBindingRecord>();
    binding->captureId = capture.captureId;
    auto* bindingPtr = binding.get();
    domain.bindings.push_back(std::move(binding));
    ++domain.liveBindings;

    cudaUserObject_t userObject = nullptr;
    auto const createResult = cudaUserObjectCreate(
        &userObject, bindingPtr, &NCCLWindowAllocator::graphBindingDestructor, 1, cudaUserObjectNoDestructorSync);
    if (createResult != cudaSuccess)
    {
        --domain.liveBindings;
        domain.bindings.pop_back();
        TLLM_CUDA_CHECK(createResult);
    }

    auto const retainResult
        = cudaGraphRetainUserObject(capture.graph, userObject, 1, cudaGraphUserObjectMove);
    if (retainResult != cudaSuccess)
    {
        // The caller still owns the reference when MOVE fails. Releasing it
        // schedules the atomic-only destructor; the stable record stays in the
        // domain until normal host-side retirement draining.
        TLLM_CUDA_CHECK_WARN(cudaUserObjectRelease(userObject, 1));
        TLLM_CUDA_CHECK(retainResult);
    }

    capture.graphBound = true;
    TLLM_LOG_DEBUG("[NCCLUtil] Bound window reuse domain %llu to CUDA capture %llu",
        static_cast<unsigned long long>(domain.id), static_cast<unsigned long long>(capture.captureId));
}

void NCCLWindowAllocator::touchEntryLocked(BufferEntry& entry, CaptureState& capture)
{
    auto& domain = *capture.domain;
    TLLM_CHECK_WITH_INFO(entry.domainId == 0 || entry.domainId == domain.id,
        "NCCL window buffer %p belongs to reuse domain %llu, not active domain %llu", entry.buffer.ptr,
        static_cast<unsigned long long>(entry.domainId), static_cast<unsigned long long>(domain.id));
    ensureGraphBindingLocked(capture);
    entry.domainId = domain.id;
    entry.lastCaptureId = capture.captureId;
    if (std::find(capture.touchedEntries.begin(), capture.touchedEntries.end(), &entry)
        == capture.touchedEntries.end())
    {
        capture.touchedEntries.push_back(&entry);
    }
}

void NCCLWindowAllocator::retainCommForActiveCapture(std::shared_ptr<ncclComm_t> const& comm)
{
    if (!mActiveCapture || !comm || !*comm)
    {
        return;
    }
    std::lock_guard<std::mutex> lock(mMutex);
    mActiveCapture->domain->commOwners.try_emplace(*comm, comm);
}

void NCCLWindowAllocator::closeReuseDomain(uint64_t domainId)
{
    std::shared_ptr<ReuseDomain> domain;
    {
        std::lock_guard<std::mutex> lock(mMutex);
        auto const domainIt = mReuseDomains.find(domainId);
        if (domainIt == mReuseDomains.end())
        {
            return;
        }
        domain = domainIt->second;
        domain->closing = true;
    }

    c10::cuda::CUDAGuard deviceGuard(static_cast<c10::DeviceIndex>(domain->device));
    TLLM_CUDA_CHECK(cudaStreamSynchronize(domain->replayStream));

    constexpr size_t maxRetirementYields = 100000;
    for (size_t attempt = 0; attempt < maxRetirementYields; ++attempt)
    {
        {
            std::lock_guard<std::mutex> lock(mMutex);
            drainRetiredBindingsLocked(*domain);
            if (domain->liveBindings == 0)
            {
                break;
            }
        }
        std::this_thread::yield();
    }

    std::unordered_map<ncclComm_t, std::shared_ptr<ncclComm_t>> commOwners;
    {
        std::lock_guard<std::mutex> lock(mMutex);
        drainRetiredBindingsLocked(*domain);
        TLLM_CHECK_WITH_INFO(domain->liveBindings == 0,
            "NCCL window reuse domain %llu still has %zu live CUDA graph binding(s); reset its graph execs before "
            "closing it",
            static_cast<unsigned long long>(domainId), domain->liveBindings);

        for (auto& [poolKey, entries] : mBufferPool)
        {
            for (auto& entry : entries)
            {
                if (entry->domainId != domainId)
                {
                    continue;
                }
                TLLM_CHECK_WITH_INFO(!entry->inUse,
                    "NCCL window reuse domain %llu still has a live tensor lease for buffer %p",
                    static_cast<unsigned long long>(domainId), entry->buffer.ptr);
                entry->domainId = 0;
                entry->lastCaptureId = 0;
                entry->persistent = false;
                entry->homeStream = domain->replayStream;
            }
        }

        commOwners = std::move(domain->commOwners);
        mReuseDomains.erase(domainId);
    }
    // Drop communicator pins outside the allocator lock. A final owner may
    // synchronously invoke communicator resource cleanup.
    commOwners.clear();
}

NCCLWindowLease NCCLWindowAllocator::requestBuffer(ncclComm_t comm, size_t size, int device)
{
    device = resolveDevice(device);
    c10::cuda::CUDAGuard deviceGuard(static_cast<c10::DeviceIndex>(device));

    if (!isNcclWindowSupported(device))
    {
        return NCCLWindowLease();
    }

    TLLM_CHECK_WITH_INFO(comm != nullptr, "NCCL communicator cannot be null");
    TLLM_CHECK_WITH_INFO(size > 0, "Buffer size must be greater than 0");

    std::lock_guard<std::mutex> lock(mMutex);

    // Register cleanup callback for this communicator if not already registered
    // This is cheap even if no buffers exist yet - cleanup will just return early
    registerBufferCleanup(comm, device);

    // Check if we have an available buffer of at least the requested size for this communicator
    // Use best-fit: find the smallest buffer that's >= requested size
    PoolKey const poolKey{comm, device};
    auto& commBuffers = mBufferPool[poolKey];
    cudaStream_t const currentStream = at::cuda::getCurrentCUDAStream(device).stream();
    CaptureState* const capture = mActiveCapture.get();
    if (capture)
    {
        TLLM_CHECK_WITH_INFO(capture->domain->device == device,
            "NCCL window reuse domain %llu belongs to device %d, but allocation requested device %d",
            static_cast<unsigned long long>(capture->domain->id), capture->domain->device, device);
    }
    for (auto& [domainId, domain] : mReuseDomains)
    {
        (void) domainId;
        drainRetiredBindingsLocked(*domain);
    }
    auto bestFit = commBuffers.end();
    size_t bestFitSize = std::numeric_limits<size_t>::max();

    for (auto it = commBuffers.begin(); it != commBuffers.end(); ++it)
    {
        auto const& entry = **it;
        bool eligible = false;
        if (capture)
        {
            bool const unbound = entry.domainId == 0;
            bool const sameDomainScratch = entry.domainId == capture->domain->id && !entry.persistent;
            bool const sameCaptureOrdered
                = entry.lastCaptureId != capture->captureId || entry.homeStream == currentStream;
            eligible = unbound || (sameDomainScratch && sameCaptureOrdered);
        }
        else
        {
            bool const streamOrdered = entry.homeStream == nullptr || entry.homeStream == currentStream;
            eligible = entry.domainId == 0 && streamOrdered;
        }
        if (!entry.inUse && eligible && entry.buffer.size >= size && entry.buffer.size < bestFitSize)
        {
            bestFit = it;
            bestFitSize = entry.buffer.size;
        }
    }

    if (bestFit != commBuffers.end())
    {
        auto& entry = **bestFit;
        if (capture)
        {
            touchEntryLocked(entry, *capture);
        }
        entry.inUse = true;
        ++entry.generation;
        entry.homeStream = currentStream;
        TLLM_LOG_TRACE(
            "[NCCLUtil] Reusing NCCL window buffer for comm %p on device %d: "
            "handle=%d, ptr=%p, size=%zu, generation=%llu (requested: %zu)",
            static_cast<void*>(comm), device, entry.buffer.handle, entry.buffer.ptr, entry.buffer.size,
            static_cast<unsigned long long>(entry.generation), size);
        return NCCLWindowLease(entry.buffer, entry.generation, entry.homeStream);
    }

    // If a previous allocateAndRegisterBuffer call collectively failed for this comm at a size
    // no larger than this request, do not retry the known-failing new allocation path. Smaller
    // requests and already-pooled buffers can still use NCCL windows.
    auto const failureIt = mMinSymmetricFailureSize.find(poolKey);
    if (failureIt != mMinSymmetricFailureSize.end() && size >= failureIt->second)
    {
        TLLM_LOG_DEBUG(
            "[NCCLUtil] Skipping NCCL window allocation for comm %p on device %d, "
            "size=%zu; known failure threshold=%zu",
            static_cast<void*>(comm), device, size, failureIt->second);
        return NCCLWindowLease();
    }

    // No available buffer found, avoid registration during CUDA graph capture
    cudaStreamCaptureStatus captureStatus = cudaStreamCaptureStatusNone;
    auto const captureErr = cudaStreamIsCapturing(currentStream, &captureStatus);
    if (captureErr != cudaSuccess)
    {
        TLLM_LOG_DEBUG("[NCCLUtil] cudaStreamIsCapturing failed: %s", cudaGetErrorString(captureErr));
    }
    bool const isCapturing = captureErr == cudaSuccess && captureStatus != cudaStreamCaptureStatusNone;
    if (isCapturing)
    {
        TLLM_LOG_DEBUG("[NCCLUtil] Skipping NCCL window allocation during capture for comm %p (requested: %zu)",
            static_cast<void*>(comm), size);
        return NCCLWindowLease();
    }

    // No available buffer found, allocate a new one
    TLLM_LOG_TRACE(
        "[NCCLUtil] Allocating new NCCL window buffer for comm %p on device %d, size=%zu",
        static_cast<void*>(comm), device, size);
    int handle = static_cast<int>(commBuffers.size());
    NCCLWindowBuffer buffer = allocateAndRegisterBuffer(comm, size, handle, device);
    // Only cache valid buffers. allocateAndRegisterBuffer returns an empty buffer when any rank
    // failed ncclMemAlloc (collective fallback to plain allreduce); caching it would leak a
    // permanently "in use" empty entry per request because releaseBuffer is a no-op for nullptr.
    if (buffer.isValid())
    {
        commBuffers.push_back(std::make_unique<BufferEntry>(BufferEntry{buffer, true, 1, currentStream}));
    }
    else
    {
        // The collective allreduce inside allocateAndRegisterBuffer agreed that this request
        // cannot use symmetric memory on at least one rank. Remember the smallest failing
        // request size so repeated too-large autotuner probes do not keep stressing this path.
        recordSymmetricFailureLocked(comm, device, size);
    }

    return buffer.isValid() ? NCCLWindowLease(buffer, 1, currentStream) : NCCLWindowLease();
}

void NCCLWindowAllocator::recordSymmetricFailureLocked(ncclComm_t comm, int device, size_t size)
{
    PoolKey const poolKey{comm, device};
    auto failureIt = mMinSymmetricFailureSize.find(poolKey);
    if (failureIt == mMinSymmetricFailureSize.end())
    {
        mMinSymmetricFailureSize.emplace(poolKey, size);
    }
    else if (size < failureIt->second)
    {
        failureIt->second = size;
    }
}

cudaError_t NCCLWindowAllocator::clearCudaErrorIfSymmetricAllocationFailed(
    int localAllocOk, CudaGetLastErrorFunc getLastError) noexcept
{
    if (localAllocOk == 0)
    {
        return getLastError();
    }
    return cudaSuccess;
}

NCCLWindowBuffer NCCLWindowAllocator::searchBuffer(ncclComm_t comm, void* ptr, int device)
{
    if (!comm || !ptr)
    {
        return NCCLWindowBuffer();
    }

    device = resolveDevice(device);
    c10::cuda::CUDAGuard deviceGuard(static_cast<c10::DeviceIndex>(device));
    std::lock_guard<std::mutex> lock(mMutex);
    auto commIt = mBufferPool.find(PoolKey{comm, device});
    if (commIt == mBufferPool.end())
    {
        return NCCLWindowBuffer();
    }
    for (auto const& entry : commIt->second)
    {
        if (entry->buffer.ptr == ptr)
        {
            if (mActiveCapture)
            {
                TLLM_CHECK_WITH_INFO(mActiveCapture->domain->device == device,
                    "NCCL window reuse domain %llu belongs to device %d, but buffer lookup used device %d",
                    static_cast<unsigned long long>(mActiveCapture->domain->id), mActiveCapture->domain->device,
                    device);
                touchEntryLocked(*entry, *mActiveCapture);
            }
            return entry->buffer;
        }
    }
    return NCCLWindowBuffer();
}

void NCCLWindowAllocator::releaseBuffer(ncclComm_t comm, void* ptr, int device, uint64_t generation)
{
    if (!comm || !ptr)
    {
        return;
    }

    device = resolveDevice(device);
    c10::cuda::CUDAGuard deviceGuard(static_cast<c10::DeviceIndex>(device));
    std::lock_guard<std::mutex> lock(mMutex);
    PoolKey const poolKey{comm, device};
    auto commIt = mBufferPool.find(poolKey);
    if (commIt == mBufferPool.end())
    {
        TLLM_LOG_WARNING("[NCCLUtil] Attempted to release buffer %p for unknown comm %p on device %d", ptr,
            static_cast<void*>(comm), device);
        return;
    }

    for (auto const& entryPtr : commIt->second)
    {
        auto& entry = *entryPtr;
        if (entry.buffer.ptr == ptr)
        {
            if (generation == 0 || generation != entry.generation)
            {
                TLLM_LOG_WARNING(
                    "[NCCLUtil] Ignoring stale release for comm %p on device %d: "
                    "ptr=%p, generation=%llu, active generation=%llu",
                    static_cast<void*>(comm), device, ptr, static_cast<unsigned long long>(generation),
                    static_cast<unsigned long long>(entry.generation));
                return;
            }
            if (!entry.inUse)
            {
                TLLM_LOG_WARNING(
                    "[NCCLUtil] Ignoring duplicate release for comm %p on device %d: ptr=%p, generation=%llu",
                    static_cast<void*>(comm), device, ptr, static_cast<unsigned long long>(entry.generation));
                return;
            }
            entry.inUse = false;
            TLLM_LOG_TRACE(
                "[NCCLUtil] Released NCCL window buffer for comm %p on device %d: ptr=%p, generation=%llu, stream=%p",
                static_cast<void*>(comm), device, ptr, static_cast<unsigned long long>(entry.generation),
                static_cast<void*>(entry.homeStream));
            return;
        }
    }

    TLLM_LOG_WARNING("[NCCLUtil] Attempted to release unknown buffer %p for comm %p on device %d", ptr,
        static_cast<void*>(comm), device);
}

ncclWindow_t NCCLWindowAllocator::getWindow(ncclComm_t comm, void* ptr, int device) const
{
    device = resolveDevice(device);
    c10::cuda::CUDAGuard deviceGuard(static_cast<c10::DeviceIndex>(device));
    std::lock_guard<std::mutex> lock(mMutex);
    NCCLWindowBuffer buffer = searchBufferLocked(comm, ptr, device);
    return buffer.isValid() ? buffer.window : nullptr;
}

size_t NCCLWindowAllocator::getSize(ncclComm_t comm, void* ptr, int device) const
{
    device = resolveDevice(device);
    c10::cuda::CUDAGuard deviceGuard(static_cast<c10::DeviceIndex>(device));
    std::lock_guard<std::mutex> lock(mMutex);
    NCCLWindowBuffer buffer = searchBufferLocked(comm, ptr, device);
    return buffer.isValid() ? buffer.size : 0;
}

NCCLWindowBuffer NCCLWindowAllocator::getBufferInfo(ncclComm_t comm, void* ptr, int device) const
{
    device = resolveDevice(device);
    c10::cuda::CUDAGuard deviceGuard(static_cast<c10::DeviceIndex>(device));
    std::lock_guard<std::mutex> lock(mMutex);
    return searchBufferLocked(comm, ptr, device);
}

size_t NCCLWindowAllocator::getBufferCount(ncclComm_t comm, int device) const
{
    device = resolveDevice(device);
    std::lock_guard<std::mutex> lock(mMutex);
    auto commIt = mBufferPool.find(PoolKey{comm, device});
    return commIt != mBufferPool.end() ? commIt->second.size() : 0;
}

size_t NCCLWindowAllocator::getBufferInUseCount(ncclComm_t comm, int device) const
{
    device = resolveDevice(device);
    std::lock_guard<std::mutex> lock(mMutex);
    auto commIt = mBufferPool.find(PoolKey{comm, device});
    if (commIt == mBufferPool.end())
    {
        return 0;
    }

    size_t count = 0;
    for (auto const& entry : commIt->second)
    {
        if (entry->inUse)
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

NCCLWindowBuffer NCCLWindowAllocator::allocateAndRegisterBuffer(ncclComm_t comm, size_t size, int handle, int device)
{
    c10::cuda::CUDAGuard deviceGuard(static_cast<c10::DeviceIndex>(device));

    // Step 1: Pre-allocate the rank-sync flag before ncclMemAlloc. ncclMemAlloc can fail
    // asymmetrically with ncclUnhandledCudaError on configurations where the symmetric/VMM path
    // is unavailable; that failure may leave a sticky CUDA last-error on the device. If we
    // deferred this cudaMalloc until after the failure, the sticky error would propagate into
    // cudaMalloc, TLLM_CUDA_CHECK would throw, and the failing rank would never reach the
    // collective ncclAllReduce(min) below, hanging every other rank that did succeed.
    int* rankSyncFlag = nullptr;
    TLLM_CUDA_CHECK(cudaMalloc(&rankSyncFlag, sizeof(int)));
    CudaMallocGuard flagGuard{rankSyncFlag}; // frees rankSyncFlag on any early return or exception
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    TLLM_CUDA_CHECK(cudaMemsetAsync(rankSyncFlag, 0, sizeof(int), stream));

    // Step 2: Allocate symmetric memory. This per-rank, non-collective call can fail
    // asymmetrically. When it fails, NCCL may leave a sticky CUDA error behind; clear it before
    // the stream-ordered flag copy and collective fallback so the failing rank still reaches
    // ncclAllReduce with the other ranks.
    void* ncclPtr = nullptr;
    TLLM_NCCL_CHECK_WARN(ncclMemAlloc(&ncclPtr, size));
    int const localAllocOk = (ncclPtr != nullptr) ? 1 : 0;
    NcclMemGuard ncclGuard{ncclPtr}; // frees ncclPtr on any early return or exception
    clearCudaErrorIfSymmetricAllocationFailed(localAllocOk);

    // Step 3: ncclCommWindowRegister is collective. If any rank skips it, all other ranks hang.
    // Populate flag, reduce with min across ranks (0 if any rank failed), then read back.
    // The flag is initialized to 0, so H2D failure is non-fatal and conservatively falls back
    // to regular NCCL while still reaching the collective. allreduce and D2H failures throw.
    if (localAllocOk != 0)
    {
        TLLM_CUDA_CHECK_WARN(
            cudaMemcpyAsync(rankSyncFlag, &localAllocOk, sizeof(localAllocOk), cudaMemcpyHostToDevice, stream));
    }
    TLLM_NCCL_CHECK(ncclAllReduce(rankSyncFlag, rankSyncFlag, 1, ncclInt32, ncclMin, comm, stream));
    TLLM_CUDA_CHECK_WARN(cudaStreamSynchronize(stream));

    int allAllocOk = 0;
    TLLM_CUDA_CHECK(cudaMemcpy(&allAllocOk, rankSyncFlag, sizeof(int), cudaMemcpyDeviceToHost));
    // flagGuard frees rankSyncFlag here at end of its scope

    if (!allAllocOk)
    {
        if (localAllocOk)
        {
            TLLM_LOG_WARNING(
                "[NCCLUtil] ncclMemAlloc failed on at least one other rank; "
                "freeing local allocation (size=%zu) and aborting window registration on all ranks.",
                size);
        }
        return NCCLWindowBuffer{}; // ncclGuard frees ncclPtr
    }

    // Step 4: Register with NCCL as a window. This is collective, so all ranks must reach it.
    // Failure here is non-fatal: warn and fall back to regular allreduce.
    // ncclGuard frees ncclPtr on return.
    ncclWindow_t window = nullptr;
    ncclResult_t const regResult = ncclCommWindowRegister(comm, ncclPtr, size, &window, NCCL_WIN_COLL_SYMMETRIC);
    TLLM_NCCL_CHECK_WARN(regResult);
    if (regResult != ncclSuccess)
    {
        return NCCLWindowBuffer{};
    }

    // Step 5: Success. Transfer ownership to the returned buffer.
    ncclGuard.release();
    NCCLWindowBuffer buffer{ncclPtr, handle, size, window, device};
    TLLM_LOG_TRACE(
        "[NCCLUtil] Allocated and registered NCCL window buffer: "
        "handle=%d, ptr=%p, size=%zu, window=%p, device=%d",
        handle, buffer.ptr, buffer.size, static_cast<void*>(buffer.window), device);
    return buffer;
}

NCCLWindowBuffer NCCLWindowAllocator::searchBufferLocked(ncclComm_t comm, void* ptr, int device) const
{
    auto commIt = mBufferPool.find(PoolKey{comm, device});
    if (commIt == mBufferPool.end())
    {
        return NCCLWindowBuffer();
    }

    for (auto const& entry : commIt->second)
    {
        if (entry->buffer.ptr == ptr)
        {
            return entry->buffer;
        }
    }

    return NCCLWindowBuffer();
}

void NCCLWindowAllocator::registerBufferCleanup(ncclComm_t comm, int device)
{
    PoolKey const poolKey{comm, device};
    // Don't register if already registered
    if (mRegisteredPools.find(poolKey) != mRegisteredPools.end())
    {
        return;
    }

    mRegisteredPools.insert(poolKey);

    // Register cleanup with the resource manager
    NcclCommResourceManager::getInstance().registerResource(
        comm, [this, comm, device]() { this->cleanupBuffersForComm(comm, device); }, "NCCLWindowAllocator");
}

void NCCLWindowAllocator::cleanupBuffersForComm(ncclComm_t comm, int device)
{
    if (!comm)
    {
        return;
    }

    c10::cuda::CUDAGuard deviceGuard(static_cast<c10::DeviceIndex>(device));
    PoolKey const poolKey{comm, device};
    std::vector<cudaStream_t> streams;
    {
        std::lock_guard<std::mutex> lock(mMutex);
        if (mRegisteredPools.find(poolKey) == mRegisteredPools.end())
        {
            return;
        }

        auto const commIt = mBufferPool.find(poolKey);
        if (commIt == mBufferPool.end())
        {
            mRegisteredPools.erase(poolKey);
            mMinSymmetricFailureSize.erase(poolKey);
            return;
        }

        size_t liveLeaseCount = 0;
        size_t graphOwnedCount = 0;
        std::unordered_set<cudaStream_t> uniqueStreams;
        for (auto const& entry : commIt->second)
        {
            liveLeaseCount += entry->inUse ? 1 : 0;
            graphOwnedCount += entry->domainId != 0 ? 1 : 0;
            if (entry->homeStream != nullptr)
            {
                uniqueStreams.insert(entry->homeStream);
            }
        }
        if (liveLeaseCount != 0 || graphOwnedCount != 0)
        {
            // Fail closed: freeing here would turn a lifetime-ordering bug
            // into an address UAF. Correct teardown closes graph domains and
            // releases tensor leases before the final communicator owner.
            TLLM_LOG_ERROR(
                "[NCCLUtil] Refusing to free comm %p window storage with %zu live lease(s) and %zu "
                "graph-owned buffer(s). Graph runners must close before communicator teardown.",
                static_cast<void*>(comm), liveLeaseCount, graphOwnedCount);
            return;
        }
        streams.assign(uniqueStreams.begin(), uniqueStreams.end());
    }

    // Synchronize only streams that carried a window lease. Same-stream reuse
    // is ordered without events; this teardown-only synchronization establishes
    // completion before deregistration without stalling unrelated devices/work.
    for (auto stream : streams)
    {
        cudaError_t const cudaErr = cudaStreamSynchronize(stream);
        if (cudaErr != cudaSuccess)
        {
            TLLM_LOG_WARNING(
                "[NCCLUtil] cudaStreamSynchronize failed with error %d before cleanup for comm %p, stream %p",
                cudaErr, static_cast<void*>(comm), static_cast<void*>(stream));
        }
    }

    std::lock_guard<std::mutex> lock(mMutex);
    auto commIt = mBufferPool.find(poolKey);
    if (commIt == mBufferPool.end())
    {
        return;
    }

    TLLM_LOG_TRACE(
        "[NCCLUtil] Cleaning up %zu NCCL window buffers for comm %p", commIt->second.size(), static_cast<void*>(comm));

    size_t totalBytes = 0;
    for (auto const& entry : commIt->second)
    {
        totalBytes += entry->buffer.size;
    }
    TLLM_LOG_DEBUG("[NCCLUtil] NCCL window allocator teardown for comm %p: %zu buffers, %zu bytes total",
        static_cast<void*>(comm), commIt->second.size(), totalBytes);

    for (auto const& entryPtr : commIt->second)
    {
        auto& entry = *entryPtr;
        if (entry.buffer.isValid())
        {
            if (entry.buffer.window && comm)
            {
                ncclResult_t result = ncclCommWindowDeregister(comm, entry.buffer.window);
                if (result != ncclSuccess)
                {
                    TLLM_LOG_WARNING(
                        "[NCCLUtil] ncclCommWindowDeregister failed with error: %d for comm %p, window %p", result,
                        static_cast<void*>(comm), static_cast<void*>(entry.buffer.window));
                }
            }

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
    mRegisteredPools.erase(poolKey);
    mMinSymmetricFailureSize.erase(poolKey);
}

#endif // NCCL_VERSION_CODE >= NCCL_VERSION(2, 28, 0)

} // namespace tensorrt_llm::common::nccl_util

#endif // ENABLE_MULTI_DEVICE
