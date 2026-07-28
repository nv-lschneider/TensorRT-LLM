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
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/common/opUtils.h"
#include "tensorrt_llm/runtime/utils/mpiUtils.h"

#include <gtest/gtest.h>
#include <c10/cuda/CUDAStream.h>
#include <mutex>
#include <nccl.h>
#include <thread>
#include <vector>

#if ENABLE_MULTI_DEVICE && BUILD_PYT
#include <torch/extension.h>
#endif

#if ENABLE_MULTI_DEVICE

namespace mpi = tensorrt_llm::mpi;
namespace tr = tensorrt_llm::runtime;
namespace nccl_util = tensorrt_llm::common::nccl_util;

using tensorrt_llm::getComm;

namespace tensorrt_llm::common::nccl_util
{
class NCCLWindowAllocatorTestAccess
{
public:
    static void recordSymmetricFailure(NCCLWindowAllocator& allocator, ncclComm_t comm, size_t size)
    {
        std::lock_guard<std::mutex> lock(allocator.mMutex);
        int device = -1;
        TLLM_CUDA_CHECK(cudaGetDevice(&device));
        allocator.recordSymmetricFailureLocked(comm, device, size);
    }

    static cudaError_t clearCudaErrorIfSymmetricAllocationFailed(
        int localAllocOk, NCCLWindowAllocator::CudaGetLastErrorFunc getLastError = cudaGetLastError)
    {
        return NCCLWindowAllocator::clearCudaErrorIfSymmetricAllocationFailed(localAllocOk, getLastError);
    }
};
} // namespace tensorrt_llm::common::nccl_util

namespace
{
int gCudaGetLastErrorCallCount = 0;

cudaError_t fakeCudaGetLastError()
{
    ++gCudaGetLastErrorCallCount;
    return cudaErrorLaunchFailure;
}
} // namespace

TEST(NCCLWindowSupportTest, RuntimeVersionAndGB10Gate)
{
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 28, 0)
    EXPECT_FALSE(nccl_util::isNcclWindowSupportedForPlatform(121, true, NCCL_VERSION(2, 27, 9)));
    EXPECT_FALSE(nccl_util::isNcclWindowSupportedForPlatform(121, true, NCCL_VERSION(2, 29, 2)));
    EXPECT_FALSE(nccl_util::isNcclWindowSupportedForPlatform(121, true, NCCL_VERSION(2, 30, 3)));

    EXPECT_TRUE(nccl_util::isNcclWindowSupportedForPlatform(121, true, NCCL_VERSION(2, 30, 4)));
    EXPECT_TRUE(nccl_util::isNcclWindowSupportedForPlatform(121, false, NCCL_VERSION(2, 29, 2)));
    EXPECT_TRUE(nccl_util::isNcclWindowSupportedForPlatform(120, true, NCCL_VERSION(2, 29, 2)));
    EXPECT_TRUE(nccl_util::isNcclWindowSupportedForPlatform(100, false, NCCL_VERSION(2, 29, 2)));
#else
    GTEST_SKIP() << "NCCL window buffers are not compiled in";
#endif
}

// Helper function to create a split communicator for testing
// This allows us to test cleanup behavior explicitly by controlling the lifetime
std::shared_ptr<ncclComm_t> createSplitComm(ncclComm_t parentComm, int color, int key)
{
    ncclComm_t newComm;
    ncclResult_t result = ncclCommSplit(parentComm, color, key, &newComm, nullptr);
    if (result != ncclSuccess)
    {
        TLLM_THROW("ncclCommSplit failed with error: %d", result);
    }

    // Create a shared_ptr with custom deleter that cleans up resources first
    return std::shared_ptr<ncclComm_t>(new ncclComm_t(newComm),
        [](ncclComm_t* comm)
        {
            if (comm && *comm)
            {
                // STEP 1: Clean up all registered resources FIRST
                tensorrt_llm::common::nccl_util::NcclCommResourceManager::getInstance().cleanupResources(*comm);

                // STEP 2: Now destroy the NCCL communicator
                ncclResult_t result = ncclCommDestroy(*comm);
                if (result != ncclSuccess)
                {
                    TLLM_LOG_WARNING("ncclCommDestroy failed with error: %d", result);
                }

                // STEP 3: Free the memory
                delete comm;
            }
        });
}

//==============================================================================
// NcclCommResourceManager Tests
//==============================================================================

class NcclCommResourceManagerTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        auto& comm = mpi::MpiComm::world();
        mWorldSize = comm.getSize();
        mRank = comm.getRank();

        if (mWorldSize < 2)
        {
            GTEST_SKIP() << "Requires at least 2 ranks (got " << mWorldSize << ")";
        }

        // Set CUDA device for this rank (required before NCCL initialization)
        int deviceCount = 0;
        TLLM_CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
        if (deviceCount > 0)
        {
            int deviceId = mRank % deviceCount;
            TLLM_CUDA_CHECK(cudaSetDevice(deviceId));
        }

        // Create a communicator for testing
        std::set<int> group;
        for (int i = 0; i < mWorldSize; ++i)
        {
            group.insert(i);
        }
        mComm = getComm(group);
    }

    void TearDown() override
    {
        // Communicator cleanup happens automatically via shared_ptr deleter
        mComm.reset();
    }

    int mWorldSize;
    int mRank;
    std::shared_ptr<ncclComm_t> mComm;
};

TEST_F(NcclCommResourceManagerTest, ResourceRegistration)
{
    auto& manager = nccl_util::NcclCommResourceManager::getInstance();

    // Create a separate comm using split for this test
    auto testComm = createSplitComm(*mComm, 0, mRank);

    // Register a resource
    bool cleanupCalled = false;
    manager.registerResource(
        *testComm, [&cleanupCalled]() { cleanupCalled = true; }, "TestResource");

    EXPECT_TRUE(manager.hasResources(*testComm));
    EXPECT_EQ(manager.getResourceCount(*testComm), 1);
    EXPECT_FALSE(cleanupCalled); // Cleanup not called yet

    // Store the raw comm value before destruction
    ncclComm_t rawComm = *testComm;

    // Cleanup should be called when comm is destroyed
    testComm.reset();

    // Verify cleanup was called
    EXPECT_TRUE(cleanupCalled);

    // Verify cleanup: check that the old comm (now destroyed) no longer has resources
    // Note: The comm is destroyed, but we can still check the manager's internal state
    // The cleanup should have removed all resources for this comm
    EXPECT_FALSE(manager.hasResources(rawComm));
    EXPECT_EQ(manager.getResourceCount(rawComm), 0);
}

TEST_F(NcclCommResourceManagerTest, MultipleResources)
{
    auto& manager = nccl_util::NcclCommResourceManager::getInstance();

    // Create a separate comm using split for this test
    auto testComm = createSplitComm(*mComm, 0, mRank);

    std::vector<int> cleanupOrder;
    manager.registerResource(
        *testComm, [&cleanupOrder]() { cleanupOrder.push_back(1); }, "Resource1");
    manager.registerResource(
        *testComm, [&cleanupOrder]() { cleanupOrder.push_back(2); }, "Resource2");
    manager.registerResource(
        *testComm, [&cleanupOrder]() { cleanupOrder.push_back(3); }, "Resource3");

    EXPECT_EQ(manager.getResourceCount(*testComm), 3);

    // Cleanup order should be preserved - destroy comm and verify order
    testComm.reset();

    // Verify cleanup order was preserved (1, 2, 3)
    EXPECT_EQ(cleanupOrder.size(), 3);
    EXPECT_EQ(cleanupOrder[0], 1);
    EXPECT_EQ(cleanupOrder[1], 2);
    EXPECT_EQ(cleanupOrder[2], 3);
}

TEST_F(NcclCommResourceManagerTest, ResourceCount)
{
    auto& manager = nccl_util::NcclCommResourceManager::getInstance();

    // Create a separate comm using split for this test
    auto testComm = createSplitComm(*mComm, 0, mRank);

    EXPECT_FALSE(manager.hasResources(*testComm));
    EXPECT_EQ(manager.getResourceCount(*testComm), 0);

    manager.registerResource(
        *testComm, []() {}, "Test1");
    EXPECT_EQ(manager.getResourceCount(*testComm), 1);

    manager.registerResource(
        *testComm, []() {}, "Test2");
    EXPECT_EQ(manager.getResourceCount(*testComm), 2);

    testComm.reset();
}

//==============================================================================
// NCCLWindowAllocator Tests
//==============================================================================

class NCCLWindowAllocatorTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        auto& comm = mpi::MpiComm::world();
        mWorldSize = comm.getSize();
        mRank = comm.getRank();

        if (mWorldSize < 2)
        {
            GTEST_SKIP() << "Requires at least 2 ranks (got " << mWorldSize << ")";
        }

        // Set CUDA device for this rank (required before NCCL initialization)
        int deviceCount = 0;
        TLLM_CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
        if (deviceCount > 0)
        {
            int deviceId = mRank % deviceCount;
            TLLM_CUDA_CHECK(cudaSetDevice(deviceId));
        }

        // Check if NCCL window buffer support is available
        if (!nccl_util::isNcclWindowSupported())
        {
            GTEST_SKIP() << "NCCL window buffer support is not available";
        }

        std::set<int> group;
        for (int i = 0; i < mWorldSize; ++i)
        {
            group.insert(i);
        }
        mComm = getComm(group);
    }

    void TearDown() override
    {
        // Cleanup happens automatically
        mComm.reset();
    }

    int mWorldSize;
    int mRank;
    std::shared_ptr<ncclComm_t> mComm;
};

TEST_F(NCCLWindowAllocatorTest, BasicAllocation)
{
    auto& allocator = nccl_util::NCCLWindowAllocator::getInstance();

    const size_t bufferSize = 1024 * 1024; // 1MB
    auto buffer = allocator.requestBuffer(*mComm, bufferSize);

    EXPECT_TRUE(buffer.isValid());
    EXPECT_NE(buffer.ptr, nullptr);
    EXPECT_NE(buffer.window, nullptr);
    EXPECT_EQ(buffer.size, bufferSize);
    EXPECT_GE(buffer.handle, 0);
    int currentDevice = -1;
    TLLM_CUDA_CHECK(cudaGetDevice(&currentDevice));
    EXPECT_EQ(buffer.device, currentDevice);

    // Verify we can search for it
    auto found = allocator.searchBuffer(*mComm, buffer.ptr);
    EXPECT_TRUE(found.isValid());
    EXPECT_EQ(found.ptr, buffer.ptr);

    // Release the buffer
    allocator.releaseBuffer(*mComm, buffer);
}

TEST_F(NCCLWindowAllocatorTest, BufferReuse)
{
    auto& allocator = nccl_util::NCCLWindowAllocator::getInstance();

    const size_t bufferSize = 512 * 1024; // 512KB

    // Allocate first buffer
    auto buffer1 = allocator.requestBuffer(*mComm, bufferSize);
    EXPECT_TRUE(buffer1.isValid());
    void* ptr1 = buffer1.ptr;

    // Release it
    allocator.releaseBuffer(*mComm, buffer1);

    // Request another buffer of the same size - should reuse
    auto buffer2 = allocator.requestBuffer(*mComm, bufferSize);
    EXPECT_TRUE(buffer2.isValid());
    EXPECT_EQ(buffer2.ptr, ptr1); // Should be the same buffer

    allocator.releaseBuffer(*mComm, buffer2);
}

TEST_F(NCCLWindowAllocatorTest, StaleGenerationCannotReleaseReusedBuffer)
{
    auto& allocator = nccl_util::NCCLWindowAllocator::getInstance();

    auto first = allocator.requestBuffer(*mComm, 512 * 1024);
    ASSERT_TRUE(first.isValid());
    ASSERT_GT(first.generation, 0);
    allocator.releaseBuffer(*mComm, first.ptr, first.device, first.generation);

    auto second = allocator.requestBuffer(*mComm, 512 * 1024);
    ASSERT_TRUE(second.isValid());
    ASSERT_EQ(second.ptr, first.ptr);
    ASSERT_GT(second.generation, first.generation);

    auto const currentStream = c10::cuda::getCurrentCUDAStream(second.device).stream();
    EXPECT_THROW(
        allocator.recordBufferStreamJoin(
            *mComm, first.ptr, first.device, first.generation, currentStream),
        tensorrt_llm::common::TllmException);

    allocator.releaseBuffer(*mComm, first.ptr, first.device, first.generation);
    EXPECT_EQ(allocator.getBufferInUseCount(*mComm, second.device), 1);

    allocator.releaseBuffer(*mComm, second.ptr, second.device, second.generation);
    EXPECT_EQ(allocator.getBufferInUseCount(*mComm, second.device), 0);
}

TEST_F(NCCLWindowAllocatorTest, EagerDifferentStreamsDoNotAlias)
{
    auto& allocator = nccl_util::NCCLWindowAllocator::getInstance();
    auto testComm = createSplitComm(*mComm, 0, mRank);
    int device = -1;
    TLLM_CUDA_CHECK(cudaGetDevice(&device));

    auto stream1 = c10::cuda::getStreamFromPool(false, device);
    auto stream2 = c10::cuda::getStreamFromPool(false, device);
    ASSERT_NE(stream1.stream(), stream2.stream());

    nccl_util::NCCLWindowLease first;
    {
        c10::cuda::CUDAStreamGuard streamGuard(stream1);
        first = allocator.requestBuffer(*testComm, 256 * 1024, device);
        ASSERT_TRUE(first.isValid());
        allocator.releaseBuffer(*testComm, first);
    }

    nccl_util::NCCLWindowLease second;
    {
        c10::cuda::CUDAStreamGuard streamGuard(stream2);
        second = allocator.requestBuffer(*testComm, 256 * 1024, device);
        ASSERT_TRUE(second.isValid());
        EXPECT_NE(second.ptr, first.ptr);
        allocator.releaseBuffer(*testComm, second);
    }
    TLLM_CUDA_CHECK(cudaStreamSynchronize(stream1.stream()));
    TLLM_CUDA_CHECK(cudaStreamSynchronize(stream2.stream()));
    testComm.reset();
}

TEST_F(NCCLWindowAllocatorTest, DefaultStreamIsARealReuseFrontier)
{
    auto& allocator = nccl_util::NCCLWindowAllocator::getInstance();
    auto testComm = createSplitComm(*mComm, 0, mRank);
    int device = -1;
    TLLM_CUDA_CHECK(cudaGetDevice(&device));

    auto defaultStream = c10::cuda::getDefaultCUDAStream(device);
    auto otherStream = c10::cuda::getStreamFromPool(false, device);

    nccl_util::NCCLWindowLease original;
    {
        c10::cuda::CUDAStreamGuard streamGuard(defaultStream);
        original = allocator.requestBuffer(*testComm, 256 * 1024, device);
        ASSERT_TRUE(original.isValid());
        allocator.releaseBuffer(*testComm, original);

        auto sameLane = allocator.requestBuffer(*testComm, 256 * 1024, device);
        ASSERT_TRUE(sameLane.isValid());
        EXPECT_EQ(sameLane.ptr, original.ptr);
        allocator.releaseBuffer(*testComm, sameLane);
    }

    {
        c10::cuda::CUDAStreamGuard streamGuard(otherStream);
        auto foreignLane = allocator.requestBuffer(*testComm, 256 * 1024, device);
        ASSERT_TRUE(foreignLane.isValid());
        EXPECT_NE(foreignLane.ptr, original.ptr);
        allocator.releaseBuffer(*testComm, foreignLane);
    }

    TLLM_CUDA_CHECK(cudaStreamSynchronize(defaultStream.stream()));
    TLLM_CUDA_CHECK(cudaStreamSynchronize(otherStream.stream()));
    testComm.reset();
}

TEST_F(NCCLWindowAllocatorTest, ObservedConsumerStreamBecomesReuseFrontier)
{
    auto& allocator = nccl_util::NCCLWindowAllocator::getInstance();
    auto testComm = createSplitComm(*mComm, 0, mRank);
    int device = -1;
    TLLM_CUDA_CHECK(cudaGetDevice(&device));

    auto producerStream = c10::cuda::getStreamFromPool(false, device);
    auto consumerStream = c10::cuda::getStreamFromPool(false, device);
    ASSERT_NE(producerStream.stream(), consumerStream.stream());

    nccl_util::NCCLWindowLease original;
    cudaEvent_t produced = nullptr;
    TLLM_CUDA_CHECK(cudaEventCreateWithFlags(&produced, cudaEventDisableTiming));
    {
        c10::cuda::CUDAStreamGuard streamGuard(producerStream);
        original = allocator.requestBuffer(*testComm, 256 * 1024, device);
        ASSERT_TRUE(original.isValid());
        TLLM_CUDA_CHECK(cudaMemsetAsync(original.ptr, 0x11, original.size, producerStream.stream()));
        TLLM_CUDA_CHECK(cudaEventRecord(produced, producerStream.stream()));
    }

    {
        c10::cuda::CUDAStreamGuard streamGuard(consumerStream);
        TLLM_CUDA_CHECK(cudaStreamWaitEvent(consumerStream.stream(), produced));
        auto observed = allocator.searchBuffer(*testComm, original.ptr, device, true);
        ASSERT_TRUE(observed.isValid());
        TLLM_CUDA_CHECK(cudaMemsetAsync(observed.ptr, 0x22, observed.size, consumerStream.stream()));
        allocator.recordBufferStreamJoin(
            *testComm, original.ptr, device, original.generation, producerStream.stream());
        allocator.releaseBuffer(*testComm, original);
    }

    nccl_util::NCCLWindowLease producerBorrow;
    {
        c10::cuda::CUDAStreamGuard streamGuard(producerStream);
        producerBorrow = allocator.requestBuffer(*testComm, 256 * 1024, device);
        ASSERT_TRUE(producerBorrow.isValid());
        EXPECT_NE(producerBorrow.ptr, original.ptr);
        allocator.releaseBuffer(*testComm, producerBorrow);
    }

    {
        c10::cuda::CUDAStreamGuard streamGuard(consumerStream);
        auto consumerBorrow = allocator.requestBuffer(*testComm, 256 * 1024, device);
        ASSERT_TRUE(consumerBorrow.isValid());
        EXPECT_EQ(consumerBorrow.ptr, original.ptr);
        allocator.releaseBuffer(*testComm, consumerBorrow);
        TLLM_CUDA_CHECK(cudaStreamSynchronize(consumerStream.stream()));
    }

    TLLM_CUDA_CHECK(cudaStreamSynchronize(producerStream.stream()));
    TLLM_CUDA_CHECK(cudaEventDestroy(produced));
    testComm.reset();
}

TEST_F(NCCLWindowAllocatorTest, GraphDomainRetainsAndReusesRegisteredBuffer)
{
    auto& allocator = nccl_util::NCCLWindowAllocator::getInstance();
    auto testComm = createSplitComm(*mComm, 0, mRank);
    int device = -1;
    TLLM_CUDA_CHECK(cudaGetDevice(&device));
    auto replayStream = c10::cuda::getStreamFromPool(false, device);
    auto captureStream0 = c10::cuda::getStreamFromPool(false, device);
    auto captureStream1 = c10::cuda::getStreamFromPool(false, device);

    uint64_t domainId = 0;
    uint64_t foreignDomainId = 0;
    void* expectedPtr = nullptr;
    {
        c10::cuda::CUDAStreamGuard replayGuard(replayStream);
        domainId = allocator.createReuseDomain(device);
        allocator.beginPreparation(domainId);
        allocator.retainCommForActiveDomain(testComm);
        auto prepared = allocator.requestBuffer(*testComm, 256 * 1024, device);
        ASSERT_TRUE(prepared.isValid());
        expectedPtr = prepared.ptr;
        allocator.releaseBuffer(*testComm, prepared);
        allocator.endPreparation(domainId);

        // A second serial domain on the same lane must reserve another
        // address even though neither domain has captured a graph yet.
        foreignDomainId = allocator.createReuseDomain(device);
        allocator.beginPreparation(foreignDomainId);
        allocator.retainCommForActiveDomain(testComm);
        auto foreignPrepared = allocator.requestBuffer(*testComm, 256 * 1024, device);
        ASSERT_TRUE(foreignPrepared.isValid());
        EXPECT_NE(foreignPrepared.ptr, expectedPtr);
        allocator.releaseBuffer(*testComm, foreignPrepared);
        allocator.endPreparation(foreignDomainId);
    }

    cudaGraph_t graph0 = nullptr;
    cudaGraphExec_t graphExec0 = nullptr;
    {
        c10::cuda::CUDAStreamGuard captureGuard(captureStream0);
        TLLM_CUDA_CHECK(cudaStreamBeginCapture(captureStream0.stream(), cudaStreamCaptureModeThreadLocal));
        uint64_t const captureId = allocator.beginCapture(domainId);
        allocator.retainCommForActiveDomain(testComm);

        auto firstCapturedLease = allocator.requestBuffer(*testComm, 256 * 1024, device);
        ASSERT_TRUE(firstCapturedLease.isValid());
        EXPECT_EQ(firstCapturedLease.ptr, expectedPtr);
        TLLM_CUDA_CHECK(cudaMemsetAsync(
            firstCapturedLease.ptr, 0x5a, firstCapturedLease.size, captureStream0.stream()));
        allocator.releaseBuffer(*testComm, firstCapturedLease);

        // A later overwrite on the exact same captured stream is ordered and
        // may reuse the transient slot without an event or replay overhead.
        auto secondCapturedLease = allocator.requestBuffer(*testComm, 256 * 1024, device);
        ASSERT_TRUE(secondCapturedLease.isValid());
        EXPECT_EQ(secondCapturedLease.ptr, expectedPtr);
        EXPECT_GT(secondCapturedLease.generation, firstCapturedLease.generation);
        TLLM_CUDA_CHECK(cudaMemsetAsync(
            secondCapturedLease.ptr, 0x3c, secondCapturedLease.size, captureStream0.stream()));
        allocator.releaseBuffer(*testComm, secondCapturedLease);

        allocator.endCapture(captureId);
        TLLM_CUDA_CHECK(cudaStreamEndCapture(captureStream0.stream(), &graph0));
    }
    ASSERT_NE(graph0, nullptr);
    TLLM_CUDA_CHECK(cudaGraphInstantiate(&graphExec0, graph0, nullptr, nullptr, 0));
    ASSERT_NE(graphExec0, nullptr);

    // A second graph variant captured on another side stream shares the same
    // arena because both execs are launched serially on replayStream.
    cudaGraph_t graph1 = nullptr;
    cudaGraphExec_t graphExec1 = nullptr;
    {
        c10::cuda::CUDAStreamGuard captureGuard(captureStream1);
        TLLM_CUDA_CHECK(cudaStreamBeginCapture(captureStream1.stream(), cudaStreamCaptureModeThreadLocal));
        uint64_t const captureId = allocator.beginCapture(domainId);
        allocator.retainCommForActiveDomain(testComm);
        auto variantLease = allocator.requestBuffer(*testComm, 256 * 1024, device);
        ASSERT_TRUE(variantLease.isValid());
        EXPECT_EQ(variantLease.ptr, expectedPtr);
        TLLM_CUDA_CHECK(
            cudaMemsetAsync(variantLease.ptr, 0x7b, variantLease.size, captureStream1.stream()));
        allocator.releaseBuffer(*testComm, variantLease);
        allocator.endCapture(captureId);
        TLLM_CUDA_CHECK(cudaStreamEndCapture(captureStream1.stream(), &graph1));
    }
    ASSERT_NE(graph1, nullptr);
    TLLM_CUDA_CHECK(cudaGraphInstantiate(&graphExec1, graph1, nullptr, nullptr, 0));
    ASSERT_NE(graphExec1, nullptr);

    allocator.quiesceReuseDomain(foreignDomainId);
    allocator.closeReuseDomain(foreignDomainId);

    {
        c10::cuda::CUDAStreamGuard replayGuard(replayStream);
        auto sameLane = allocator.searchBuffer(*testComm, expectedPtr, device, true);
        EXPECT_TRUE(sameLane.isValid());
    }
    {
        c10::cuda::CUDAStreamGuard foreignGuard(captureStream0);
        auto foreignLane = allocator.searchBuffer(*testComm, expectedPtr, device, true);
        EXPECT_FALSE(foreignLane.isValid());
    }

    // The graph owns expectedPtr even though both transient leases ended.
    // Foreign eager work must receive another slot until graph teardown.
    {
        c10::cuda::CUDAStreamGuard replayGuard(replayStream);
        auto foreignLease = allocator.requestBuffer(*testComm, 256 * 1024, device);
        ASSERT_TRUE(foreignLease.isValid());
        EXPECT_NE(foreignLease.ptr, expectedPtr);
        allocator.releaseBuffer(*testComm, foreignLease);

        for (int iteration = 0; iteration < 1000; ++iteration)
        {
            TLLM_CUDA_CHECK(cudaGraphLaunch(graphExec0, replayStream.stream()));
            TLLM_CUDA_CHECK(cudaGraphLaunch(graphExec1, replayStream.stream()));
        }
    }

    // Quiesce before graph reset. closeReuseDomain synchronizes the in-flight
    // replay lane and waits for both CUDA user-object references to retire.
    allocator.quiesceReuseDomain(domainId);
    TLLM_CUDA_CHECK(cudaGraphExecDestroy(graphExec0));
    TLLM_CUDA_CHECK(cudaGraphExecDestroy(graphExec1));
    TLLM_CUDA_CHECK(cudaGraphDestroy(graph0));
    TLLM_CUDA_CHECK(cudaGraphDestroy(graph1));
    allocator.closeReuseDomain(domainId);

    unsigned char finalValue = 0;
    TLLM_CUDA_CHECK(cudaMemcpy(&finalValue, expectedPtr, sizeof(finalValue), cudaMemcpyDeviceToHost));
    EXPECT_EQ(finalValue, 0x7b);

    {
        // Teardown synchronized the old replay lane, so the retired arena is
        // safe to adopt on a different stream without an event.
        c10::cuda::CUDAStreamGuard postCloseGuard(captureStream0);
        auto reused = allocator.requestBuffer(*testComm, 256 * 1024, device);
        ASSERT_TRUE(reused.isValid());
        EXPECT_EQ(reused.ptr, expectedPtr);
        allocator.releaseBuffer(*testComm, reused);
        TLLM_CUDA_CHECK(cudaStreamSynchronize(captureStream0.stream()));
    }
    testComm.reset();
}

TEST_F(NCCLWindowAllocatorTest, BestFitReuse)
{
    auto& allocator = nccl_util::NCCLWindowAllocator::getInstance();

    // Allocate buffers of different sizes
    auto buffer1MB = allocator.requestBuffer(*mComm, 1024 * 1024);
    auto buffer2MB = allocator.requestBuffer(*mComm, 2 * 1024 * 1024);
    auto buffer512KB = allocator.requestBuffer(*mComm, 512 * 1024);

    void* ptr1MB = buffer1MB.ptr;
    void* ptr2MB = buffer2MB.ptr;
    void* ptr512KB = buffer512KB.ptr;

    // Release all
    allocator.releaseBuffer(*mComm, buffer1MB);
    allocator.releaseBuffer(*mComm, buffer2MB);
    allocator.releaseBuffer(*mComm, buffer512KB);

    // Request 768KB - should reuse 1MB (best fit, smallest that fits)
    auto buffer768KB = allocator.requestBuffer(*mComm, 768 * 1024);
    EXPECT_TRUE(buffer768KB.isValid());
    EXPECT_EQ(buffer768KB.ptr, ptr1MB);       // Should reuse 1MB buffer
    EXPECT_EQ(buffer768KB.size, 1024 * 1024); // Original size

    allocator.releaseBuffer(*mComm, buffer768KB);
}

TEST_F(NCCLWindowAllocatorTest, FailureCacheIsSizeAwareForNewAllocations)
{
    auto& allocator = nccl_util::NCCLWindowAllocator::getInstance();
    auto testComm = createSplitComm(*mComm, 0, mRank);

    constexpr size_t failureSize = 1024 * 1024;
    nccl_util::NCCLWindowAllocatorTestAccess::recordSymmetricFailure(allocator, *testComm, failureSize);

    auto smallBuffer = allocator.requestBuffer(*testComm, failureSize / 2);
    ASSERT_TRUE(smallBuffer.isValid());
    EXPECT_EQ(allocator.getBufferCount(*testComm), 1);

    auto failedBuffer = allocator.requestBuffer(*testComm, failureSize);
    EXPECT_FALSE(failedBuffer.isValid());
    EXPECT_EQ(allocator.getBufferCount(*testComm), 1);

    allocator.releaseBuffer(*testComm, smallBuffer);
    testComm.reset();
}

TEST_F(NCCLWindowAllocatorTest, FailureCacheDoesNotDisableReusableBuffers)
{
    auto& allocator = nccl_util::NCCLWindowAllocator::getInstance();
    auto testComm = createSplitComm(*mComm, 0, mRank);

    auto buffer1MB = allocator.requestBuffer(*testComm, 1024 * 1024);
    ASSERT_TRUE(buffer1MB.isValid());
    void* ptr1MB = buffer1MB.ptr;
    allocator.releaseBuffer(*testComm, buffer1MB);

    nccl_util::NCCLWindowAllocatorTestAccess::recordSymmetricFailure(allocator, *testComm, 512 * 1024);

    auto reusedBuffer = allocator.requestBuffer(*testComm, 768 * 1024);
    ASSERT_TRUE(reusedBuffer.isValid());
    EXPECT_EQ(reusedBuffer.ptr, ptr1MB);
    EXPECT_EQ(allocator.getBufferCount(*testComm), 1);
    allocator.releaseBuffer(*testComm, reusedBuffer);

    auto failedBuffer = allocator.requestBuffer(*testComm, 2 * 1024 * 1024);
    EXPECT_FALSE(failedBuffer.isValid());
    EXPECT_EQ(allocator.getBufferCount(*testComm), 1);

    testComm.reset();
}

TEST_F(NCCLWindowAllocatorTest, FailureCacheKeepsSmallestFailureSize)
{
    auto& allocator = nccl_util::NCCLWindowAllocator::getInstance();
    auto testComm = createSplitComm(*mComm, 0, mRank);

    nccl_util::NCCLWindowAllocatorTestAccess::recordSymmetricFailure(allocator, *testComm, 2 * 1024 * 1024);
    nccl_util::NCCLWindowAllocatorTestAccess::recordSymmetricFailure(allocator, *testComm, 1024 * 1024);

    auto smallBuffer = allocator.requestBuffer(*testComm, 768 * 1024);
    ASSERT_TRUE(smallBuffer.isValid());
    EXPECT_EQ(allocator.getBufferCount(*testComm), 1);

    auto failedBuffer = allocator.requestBuffer(*testComm, 1536 * 1024);
    EXPECT_FALSE(failedBuffer.isValid());
    EXPECT_EQ(allocator.getBufferCount(*testComm), 1);

    allocator.releaseBuffer(*testComm, smallBuffer);
    testComm.reset();
}

TEST_F(NCCLWindowAllocatorTest, ClearsCudaErrorAfterLocalAllocationFailure)
{
    auto const clearCudaErrorIfFailed = [](int localAllocOk)
    {
        return nccl_util::NCCLWindowAllocatorTestAccess::clearCudaErrorIfSymmetricAllocationFailed(
            localAllocOk, fakeCudaGetLastError);
    };

    gCudaGetLastErrorCallCount = 0;
    EXPECT_EQ(clearCudaErrorIfFailed(1), cudaSuccess);
    EXPECT_EQ(gCudaGetLastErrorCallCount, 0);

    EXPECT_EQ(clearCudaErrorIfFailed(0), cudaErrorLaunchFailure);
    EXPECT_EQ(gCudaGetLastErrorCallCount, 1);
}

TEST_F(NCCLWindowAllocatorTest, MultipleBuffers)
{
    auto& allocator = nccl_util::NCCLWindowAllocator::getInstance();

    const size_t bufferSize = 256 * 1024;
    std::vector<nccl_util::NCCLWindowLease> leases;

    // Allocate multiple buffers
    for (int i = 0; i < 5; ++i)
    {
        auto buffer = allocator.requestBuffer(*mComm, bufferSize);
        EXPECT_TRUE(buffer.isValid());
        leases.push_back(buffer);
    }

    EXPECT_EQ(allocator.getBufferCount(*mComm), 5);
    EXPECT_EQ(allocator.getBufferInUseCount(*mComm), 5);

    // Release all
    for (auto const& lease : leases)
    {
        allocator.releaseBuffer(*mComm, lease);
    }

    EXPECT_EQ(allocator.getBufferInUseCount(*mComm), 0);
    EXPECT_EQ(allocator.getBufferCount(*mComm), 5); // Buffers still exist, just not in use
}

TEST_F(NCCLWindowAllocatorTest, SearchBuffer)
{
    auto& allocator = nccl_util::NCCLWindowAllocator::getInstance();

    const size_t bufferSize = 128 * 1024;
    auto buffer = allocator.requestBuffer(*mComm, bufferSize);

    // Test searchBuffer
    auto found = allocator.searchBuffer(*mComm, buffer.ptr);
    EXPECT_TRUE(found.isValid());
    EXPECT_EQ(found.ptr, buffer.ptr);
    // Compare against actual allocated size (ncclMemAlloc may allocate more than requested)
    EXPECT_EQ(found.size, buffer.size);
    EXPECT_GE(found.size, bufferSize); // At least the requested size

    // Test search for non-existent buffer
    void* fakePtr = reinterpret_cast<void*>(0xDEADBEEF);
    auto notFound = allocator.searchBuffer(*mComm, fakePtr);
    EXPECT_FALSE(notFound.isValid());

    allocator.releaseBuffer(*mComm, buffer);
}

TEST_F(NCCLWindowAllocatorTest, GetWindowAndSize)
{
    auto& allocator = nccl_util::NCCLWindowAllocator::getInstance();

    const size_t bufferSize = 64 * 1024;
    auto buffer = allocator.requestBuffer(*mComm, bufferSize);

    // Test getWindow
    auto window = allocator.getWindow(*mComm, buffer.ptr);
    EXPECT_NE(window, nullptr);
    EXPECT_EQ(window, buffer.window);

    // Test getSize - compare against actual allocated size (ncclMemAlloc may allocate more than requested)
    auto size = allocator.getSize(*mComm, buffer.ptr);
    EXPECT_EQ(size, buffer.size);
    EXPECT_GE(size, bufferSize); // At least the requested size

    // Test with invalid pointer
    void* fakePtr = reinterpret_cast<void*>(0xDEADBEEF);
    EXPECT_EQ(allocator.getWindow(*mComm, fakePtr), nullptr);
    EXPECT_EQ(allocator.getSize(*mComm, fakePtr), 0);

    allocator.releaseBuffer(*mComm, buffer);
}

TEST_F(NCCLWindowAllocatorTest, GetBufferInfo)
{
    auto& allocator = nccl_util::NCCLWindowAllocator::getInstance();

    const size_t bufferSize = 32 * 1024;
    auto buffer = allocator.requestBuffer(*mComm, bufferSize);

    auto info = allocator.getBufferInfo(*mComm, buffer.ptr);
    EXPECT_TRUE(info.isValid());
    EXPECT_EQ(info.ptr, buffer.ptr);
    EXPECT_EQ(info.size, buffer.size);
    EXPECT_EQ(info.handle, buffer.handle);
    EXPECT_EQ(info.window, buffer.window);

    allocator.releaseBuffer(*mComm, buffer);
}

TEST_F(NCCLWindowAllocatorTest, ScopedBuffer)
{
    const size_t bufferSize = 16 * 1024;

    {
        nccl_util::ScopedNCCLWindowBuffer scopedBuffer(mComm, bufferSize);
        EXPECT_TRUE(scopedBuffer.getBuffer().isValid());
        EXPECT_NE(scopedBuffer.getPtr(), nullptr);
        // Compare against actual allocated size (ncclMemAlloc may allocate more than requested)
        EXPECT_EQ(scopedBuffer.getSize(), scopedBuffer.getBuffer().size);
        EXPECT_GE(scopedBuffer.getSize(), bufferSize); // At least the requested size
        EXPECT_NE(scopedBuffer.getWindow(), nullptr);

        // Buffer should be in use
        auto& allocator = nccl_util::NCCLWindowAllocator::getInstance();
        EXPECT_EQ(allocator.getBufferInUseCount(*mComm), 1);
    }

    // Buffer should be released when scoped buffer goes out of scope
    auto& allocator = nccl_util::NCCLWindowAllocator::getInstance();
    EXPECT_EQ(allocator.getBufferInUseCount(*mComm), 0);
}

TEST_F(NCCLWindowAllocatorTest, CleanupOnCommDestroy)
{
    auto& allocator = nccl_util::NCCLWindowAllocator::getInstance();

    // Create a separate comm using split for this test
    auto testComm = createSplitComm(*mComm, 0, mRank);

    // Store the raw comm value before destruction
    ncclComm_t rawComm = *testComm;

    // Allocate some buffers
    const size_t bufferSize = 8 * 1024;
    auto buffer1 = allocator.requestBuffer(*testComm, bufferSize);
    auto buffer2 = allocator.requestBuffer(*testComm, bufferSize * 2);

    EXPECT_EQ(allocator.getBufferCount(*testComm), 2);
    EXPECT_EQ(allocator.getBufferInUseCount(*testComm), 2);

    // Verify buffers are valid
    EXPECT_TRUE(buffer1.isValid());
    EXPECT_TRUE(buffer2.isValid());

    // Manually release buffers before cleanup to avoid warnings
    allocator.releaseBuffer(*testComm, buffer1);
    allocator.releaseBuffer(*testComm, buffer2);

    // Verify buffers are released but still exist in pool
    EXPECT_EQ(allocator.getBufferInUseCount(*testComm), 0);
    EXPECT_EQ(allocator.getBufferCount(*testComm), 2); // Buffers still exist, just not in use

    // Destroy the communicator - buffers should be cleaned up automatically
    testComm.reset();

    // Verify cleanup: check that the old comm (now destroyed) no longer has buffers
    // Note: The comm is destroyed, but we can still check the allocator's internal state
    // The cleanup should have removed all buffers for this comm
    EXPECT_EQ(allocator.getBufferCount(rawComm), 0);
    EXPECT_EQ(allocator.getBufferInUseCount(rawComm), 0);
    // Note: isCommValid only checks for null, not cleaned-up state, because NCCL can reuse addresses
    // The real check is that buffers are gone, which we verify above
}

TEST_F(NCCLWindowAllocatorTest, CommValidity)
{
    auto& allocator = nccl_util::NCCLWindowAllocator::getInstance();

    // Valid comm should be valid
    EXPECT_TRUE(allocator.isCommValid(*mComm));

    // Null comm should be invalid
    EXPECT_FALSE(allocator.isCommValid(nullptr));
}

//==============================================================================
// Integration Tests
//==============================================================================

TEST_F(NCCLWindowAllocatorTest, MultipleComms)
{
    auto& allocator = nccl_util::NCCLWindowAllocator::getInstance();

    // Create two different communicators using split (different colors)
    auto comm1 = createSplitComm(*mComm, 0, mRank);
    auto comm2 = createSplitComm(*mComm, 1, mRank);

    const size_t bufferSize = 4 * 1024;

    // Allocate buffers from both comms
    auto buffer1 = allocator.requestBuffer(*comm1, bufferSize);
    auto buffer2 = allocator.requestBuffer(*comm2, bufferSize);

    EXPECT_TRUE(buffer1.isValid());
    EXPECT_TRUE(buffer2.isValid());

    // Buffers should be tracked separately per comm
    EXPECT_EQ(allocator.getBufferCount(*comm1), 1);
    EXPECT_EQ(allocator.getBufferCount(*comm2), 1);
    EXPECT_NE(buffer1.ptr, buffer2.ptr); // Different buffers from different comms

    allocator.releaseBuffer(*comm1, buffer1);
    allocator.releaseBuffer(*comm2, buffer2);

    // Clean up comms
    comm1.reset();
    comm2.reset();
}

#if ENABLE_MULTI_DEVICE && BUILD_PYT
//==============================================================================
// createNCCLWindowTensor Tests
//==============================================================================

class CreateNCCLWindowTensorTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        auto& comm = mpi::MpiComm::world();
        mWorldSize = comm.getSize();
        mRank = comm.getRank();

        if (mWorldSize < 2)
        {
            GTEST_SKIP() << "Requires at least 2 ranks (got " << mWorldSize << ")";
        }

        // Set CUDA device for this rank (required before NCCL initialization)
        int deviceCount = 0;
        TLLM_CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
        if (deviceCount > 0)
        {
            int deviceId = mRank % deviceCount;
            TLLM_CUDA_CHECK(cudaSetDevice(deviceId));
        }

        // Check if NCCL window buffer support is available
        if (!nccl_util::isNcclWindowSupported())
        {
            GTEST_SKIP() << "NCCL window buffer support is not available";
        }

        std::set<int> group;
        for (int i = 0; i < mWorldSize; ++i)
        {
            group.insert(i);
        }
        mComm = getComm(group);
    }

    void TearDown() override
    {
        mComm.reset();
    }

    int mWorldSize;
    int mRank;
    std::shared_ptr<ncclComm_t> mComm;
};

TEST_F(CreateNCCLWindowTensorTest, BasicTensorCreation)
{
    using nccl_util::createNCCLWindowTensor;

    // Create a tensor with shape [4, 8] and float32 dtype
    std::vector<int64_t> shape = {4, 8};
    auto [tensor, buffer] = createNCCLWindowTensor(mComm, shape, torch::kFloat32);
    int currentDevice = -1;
    TLLM_CUDA_CHECK(cudaGetDevice(&currentDevice));

    // Verify tensor properties
    EXPECT_TRUE(tensor.defined());
    EXPECT_EQ(tensor.dtype(), torch::kFloat32);
    EXPECT_EQ(tensor.device().type(), torch::kCUDA);
    EXPECT_EQ(tensor.get_device(), currentDevice);
    EXPECT_EQ(tensor.dim(), 2);
    EXPECT_EQ(tensor.size(0), 4);
    EXPECT_EQ(tensor.size(1), 8);
    EXPECT_EQ(tensor.numel(), 4 * 8);

    // Verify buffer properties
    EXPECT_TRUE(buffer.isValid());
    EXPECT_NE(buffer.ptr, nullptr);
    // ncclMemAlloc may allocate more than requested, so check at least the requested size
    EXPECT_GE(buffer.size, 4 * 8 * sizeof(float));
    EXPECT_NE(buffer.window, nullptr);
    EXPECT_EQ(buffer.device, currentDevice);

    // Verify tensor data pointer matches buffer pointer
    EXPECT_EQ(tensor.data_ptr(), buffer.ptr);

    // Tensor should be in use
    auto& allocator = nccl_util::NCCLWindowAllocator::getInstance();
    EXPECT_EQ(allocator.getBufferInUseCount(*mComm), 1);
}

TEST_F(CreateNCCLWindowTensorTest, DifferentDtypes)
{
    using nccl_util::createNCCLWindowTensor;

    std::vector<int64_t> shape = {10};

    // Test float32
    {
        auto [tensor, buffer] = createNCCLWindowTensor(mComm, shape, torch::kFloat32);
        EXPECT_EQ(tensor.dtype(), torch::kFloat32);
        // ncclMemAlloc may allocate more than requested, so check at least the requested size
        EXPECT_GE(buffer.size, 10 * sizeof(float));
        EXPECT_EQ(tensor.data_ptr(), buffer.ptr);
    }

    // Test float16
    {
        auto [tensor, buffer] = createNCCLWindowTensor(mComm, shape, torch::kFloat16);
        EXPECT_EQ(tensor.dtype(), torch::kFloat16);
        // ncclMemAlloc may allocate more than requested, so check at least the requested size
        EXPECT_GE(buffer.size, 10 * sizeof(at::Half));
        EXPECT_EQ(tensor.data_ptr(), buffer.ptr);
    }

    // Test int32
    {
        auto [tensor, buffer] = createNCCLWindowTensor(mComm, shape, torch::kInt32);
        EXPECT_EQ(tensor.dtype(), torch::kInt32);
        // ncclMemAlloc may allocate more than requested, so check at least the requested size
        EXPECT_GE(buffer.size, 10 * sizeof(int32_t));
        EXPECT_EQ(tensor.data_ptr(), buffer.ptr);
    }
}

TEST_F(CreateNCCLWindowTensorTest, DifferentShapes)
{
    using nccl_util::createNCCLWindowTensor;

    // 1D tensor
    {
        std::vector<int64_t> shape = {100};
        auto [tensor, buffer] = createNCCLWindowTensor(mComm, shape, torch::kFloat32);
        EXPECT_EQ(tensor.dim(), 1);
        EXPECT_EQ(tensor.size(0), 100);
        // ncclMemAlloc may allocate more than requested, so check at least the requested size
        EXPECT_GE(buffer.size, 100 * sizeof(float));
    }

    // 3D tensor
    {
        std::vector<int64_t> shape = {2, 3, 4};
        auto [tensor, buffer] = createNCCLWindowTensor(mComm, shape, torch::kFloat32);
        EXPECT_EQ(tensor.dim(), 3);
        EXPECT_EQ(tensor.size(0), 2);
        EXPECT_EQ(tensor.size(1), 3);
        EXPECT_EQ(tensor.size(2), 4);
        // ncclMemAlloc may allocate more than requested, so check at least the requested size
        EXPECT_GE(buffer.size, 2 * 3 * 4 * sizeof(float));
    }

    // 4D tensor
    {
        std::vector<int64_t> shape = {1, 2, 3, 4};
        auto [tensor, buffer] = createNCCLWindowTensor(mComm, shape, torch::kFloat32);
        EXPECT_EQ(tensor.dim(), 4);
        EXPECT_EQ(tensor.numel(), 1 * 2 * 3 * 4);
        // ncclMemAlloc may allocate more than requested, so check at least the requested size
        EXPECT_GE(buffer.size, 1 * 2 * 3 * 4 * sizeof(float));
    }
}

TEST_F(CreateNCCLWindowTensorTest, TensorDeleterReleasesBuffer)
{
    using nccl_util::createNCCLWindowTensor;

    auto& allocator = nccl_util::NCCLWindowAllocator::getInstance();

    {
        std::vector<int64_t> shape = {16, 16};
        auto [tensor, buffer] = createNCCLWindowTensor(mComm, shape, torch::kFloat32);

        EXPECT_EQ(allocator.getBufferInUseCount(*mComm), 1);
        EXPECT_TRUE(buffer.isValid());
        void* bufferPtr = buffer.ptr;

        // Tensor goes out of scope - deleter should release the buffer
    }

    // Buffer should be released (not in use anymore)
    EXPECT_EQ(allocator.getBufferInUseCount(*mComm), 0);

    // Buffer should still exist in the pool (for reuse)
    EXPECT_GE(allocator.getBufferCount(*mComm), 1);
}

TEST_F(CreateNCCLWindowTensorTest, TensorDeleterUsesOwningDevice)
{
    using nccl_util::createNCCLWindowTensor;

    int deviceCount = 0;
    TLLM_CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount < 2)
    {
        GTEST_SKIP() << "Requires at least two CUDA devices";
    }

    int ownerDevice = -1;
    TLLM_CUDA_CHECK(cudaGetDevice(&ownerDevice));
    int const otherDevice = (ownerDevice + 1) % deviceCount;

    auto& allocator = nccl_util::NCCLWindowAllocator::getInstance();
    auto [tensor, buffer] = createNCCLWindowTensor(mComm, {16, 16}, torch::kFloat32);
    ASSERT_TRUE(tensor.defined());
    ASSERT_TRUE(buffer.isValid());
    EXPECT_EQ(buffer.device, ownerDevice);

    TLLM_CUDA_CHECK(cudaSetDevice(otherDevice));
    tensor = torch::Tensor();

    EXPECT_EQ(allocator.getBufferInUseCount(*mComm, ownerDevice), 0);
    int currentDevice = -1;
    TLLM_CUDA_CHECK(cudaGetDevice(&currentDevice));
    EXPECT_EQ(currentDevice, otherDevice);

    TLLM_CUDA_CHECK(cudaSetDevice(ownerDevice));
}

TEST_F(CreateNCCLWindowTensorTest, MultipleTensors)
{
    using nccl_util::createNCCLWindowTensor;

    auto& allocator = nccl_util::NCCLWindowAllocator::getInstance();

    std::vector<int64_t> shape = {8, 8};
    auto [tensor1, buffer1] = createNCCLWindowTensor(mComm, shape, torch::kFloat32);
    auto [tensor2, buffer2] = createNCCLWindowTensor(mComm, shape, torch::kFloat32);
    auto [tensor3, buffer3] = createNCCLWindowTensor(mComm, shape, torch::kFloat32);

    EXPECT_EQ(allocator.getBufferInUseCount(*mComm), 3);
    EXPECT_NE(buffer1.ptr, buffer2.ptr);
    EXPECT_NE(buffer2.ptr, buffer3.ptr);
    EXPECT_NE(buffer1.ptr, buffer3.ptr);

    // All tensors should be valid
    EXPECT_TRUE(tensor1.defined());
    EXPECT_TRUE(tensor2.defined());
    EXPECT_TRUE(tensor3.defined());
}

TEST_F(CreateNCCLWindowTensorTest, TensorStrides)
{
    using nccl_util::createNCCLWindowTensor;

    std::vector<int64_t> shape = {3, 4, 5};
    auto [tensor, buffer] = createNCCLWindowTensor(mComm, shape, torch::kFloat32);

    // Verify strides are correct (row-major order)
    EXPECT_EQ(tensor.stride(0), 4 * 5); // stride for first dimension
    EXPECT_EQ(tensor.stride(1), 5);     // stride for second dimension
    EXPECT_EQ(tensor.stride(2), 1);     // stride for third dimension
}

#endif // ENABLE_MULTI_DEVICE && BUILD_PYT

#endif // ENABLE_MULTI_DEVICE
