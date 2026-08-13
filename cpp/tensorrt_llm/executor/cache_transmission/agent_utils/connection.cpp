/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "connection.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/executor/cache_transmission/cacheSplitConcat.h"
#include "tensorrt_llm/executor/serialization.h"
#include <algorithm>
#include <charconv>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <random>
#include <string>
#include <thread>
#include <unistd.h>
#include <utility>

namespace tensorrt_llm::executor::kv_cache
{
namespace
{

bool mooncakePagedGinDiagEnabled()
{
    return common::getBoolEnv("TRTLLM_MOONCAKE_PAGED_GIN_DIAG");
}

std::string requireStartupEnv(char const* name)
{
    auto const* value = std::getenv(name);
    TLLM_CHECK_WITH_INFO(value != nullptr && value[0] != '\0', "Missing required startup preconnect environment %s", name);
    return value;
}

int getStartupEnvInt(char const* name, int defaultValue)
{
    auto const* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0')
    {
        return defaultValue;
    }
    try
    {
        auto const parsed = std::stoi(value);
        TLLM_CHECK_WITH_INFO(parsed > 0, "Startup preconnect environment %s must be positive", name);
        return parsed;
    }
    catch (std::exception const& e)
    {
        TLLM_THROW("Invalid startup preconnect environment %s=%s: %s", name, value, e.what());
    }
}

void writeStartupFile(std::filesystem::path const& path, std::vector<char> const& payload)
{
    std::filesystem::create_directories(path.parent_path());
    auto temporary = path;
    temporary += "." + std::to_string(static_cast<unsigned long>(::getpid())) + ".tmp";
    {
        std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
        TLLM_CHECK_WITH_INFO(output.is_open(), "Unable to open startup preconnect file %s", temporary.c_str());
        output.write(payload.data(), static_cast<std::streamsize>(payload.size()));
        TLLM_CHECK_WITH_INFO(output.good(), "Unable to write startup preconnect file %s", temporary.c_str());
    }
    std::error_code error;
    std::filesystem::rename(temporary, path, error);
    TLLM_CHECK_WITH_INFO(!error, "Unable to publish startup preconnect file %s: %s", path.c_str(),
        error.message().c_str());
}

void writeStartupMarkerNoThrow(std::filesystem::path const& path, std::string const& message) noexcept
{
    try
    {
        writeStartupFile(path, std::vector<char>(message.begin(), message.end()));
    }
    catch (...)
    {
    }
}

std::vector<char> readStartupFile(std::filesystem::path const& path)
{
    std::ifstream input(path, std::ios::binary);
    TLLM_CHECK_WITH_INFO(input.is_open(), "Unable to open startup preconnect file %s", path.c_str());
    auto payload
        = std::vector<char>(std::istreambuf_iterator<char>{input}, std::istreambuf_iterator<char>{});
    TLLM_CHECK_WITH_INFO(input.good() || input.eof(), "Unable to read startup preconnect file %s", path.c_str());
    return payload;
}

bool startupPathExists(std::filesystem::path const& path)
{
    std::error_code error;
    auto const exists = std::filesystem::is_regular_file(path, error);
    return !error && exists;
}

void waitForStartupPath(std::filesystem::path const& path, std::chrono::steady_clock::time_point deadline)
{
    while (!startupPathExists(path))
    {
        TLLM_CHECK_WITH_INFO(std::chrono::steady_clock::now() < deadline,
            "Timed out waiting for startup preconnect file %s", path.c_str());
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
}

std::vector<char> serializeStartupPeerState(DataTransceiverState const& state, MemoryDesc const& pool)
{
    namespace su = executor::serialize_utils;
    auto const statePayload = Serialization::serialize(state);
    std::ostringstream output;
    su::serialize(statePayload, output);
    MemoryDesc::serialize(pool, output);
    auto const serialized = output.str();
    return std::vector<char>(serialized.begin(), serialized.end());
}

std::pair<DataTransceiverState, MemoryDesc> deserializeStartupPeerState(std::vector<char>& payload)
{
    namespace su = executor::serialize_utils;
    su::VectorWrapBuf<char> buffer(payload);
    std::istream input(&buffer);
    auto statePayload = su::deserialize<std::vector<char>>(input);
    auto pool = MemoryDesc::deserialize(input);
    return {Serialization::deserializeDataTransceiverState(statePayload), pool};
}

std::vector<std::vector<char>> allgatherStartupPayloads(
    mpi::MpiComm const& session, std::vector<char> const& localPayload)
{
    auto const sessionSize = session.getSize();
    TLLM_CHECK_WITH_INFO(sessionSize > 0, "Startup preconnect requires a non-empty MPI session");
    TLLM_CHECK_WITH_INFO(localPayload.size() <= static_cast<size_t>(std::numeric_limits<SizeType32>::max()),
        "Startup preconnect rank payload is too large: %zu bytes", localPayload.size());

    auto const localPayloadSize = static_cast<SizeType32>(localPayload.size());
    std::vector<SizeType32> payloadSizes(sessionSize);
    session.allgather(&localPayloadSize, payloadSizes.data(), 1, mpi::MpiType::kINT32);

    size_t totalSize{0};
    std::vector<int> displacements(sessionSize);
    for (int rank = 0; rank < sessionSize; ++rank)
    {
        TLLM_CHECK_WITH_INFO(totalSize <= static_cast<size_t>(std::numeric_limits<int>::max()),
            "Startup preconnect state bundle exceeds MPI displacement range");
        displacements[rank] = static_cast<int>(totalSize);
        totalSize += static_cast<size_t>(payloadSizes[rank]);
    }
    TLLM_CHECK_WITH_INFO(totalSize <= static_cast<size_t>(std::numeric_limits<int>::max()),
        "Startup preconnect state bundle is too large: %zu bytes", totalSize);

    std::vector<char> gatheredPayloads(totalSize);
    session.allgatherv(localPayload.data(), localPayloadSize, mpi::MpiType::kCHAR, gatheredPayloads.data(),
        payloadSizes, displacements, mpi::MpiType::kCHAR);

    std::vector<std::vector<char>> rankPayloads;
    rankPayloads.reserve(sessionSize);
    for (int rank = 0; rank < sessionSize; ++rank)
    {
        auto const begin = gatheredPayloads.begin() + displacements[rank];
        rankPayloads.emplace_back(begin, begin + payloadSizes[rank]);
    }
    return rankPayloads;
}

std::vector<char> serializeStartupPeerStateBundle(std::vector<std::vector<char>> const& rankPayloads)
{
    namespace su = executor::serialize_utils;
    std::ostringstream output;
    su::serialize(rankPayloads, output);
    auto const serialized = output.str();
    return std::vector<char>(serialized.begin(), serialized.end());
}

std::vector<std::vector<char>> deserializeStartupPeerStateBundle(std::vector<char>& payload)
{
    namespace su = executor::serialize_utils;
    su::VectorWrapBuf<char> buffer(payload);
    std::istream input(&buffer);
    return su::deserialize<std::vector<std::vector<char>>>(input);
}

uint64_t parseStartupPairCount(std::vector<char> const& payload, char const* markerName)
{
    auto const message = std::string(payload.begin(), payload.end());
    auto const token = std::string{"pairs="};
    auto const tokenPos = message.find(token);
    TLLM_CHECK_WITH_INFO(message.rfind("OK ", 0) == 0 && tokenPos != std::string::npos,
        "%s marker has invalid content '%s'", markerName, message.c_str());
    TLLM_CHECK_WITH_INFO(message.find(token, tokenPos + token.size()) == std::string::npos,
        "%s marker contains multiple pair counts: '%s'", markerName, message.c_str());

    auto const valueBegin = tokenPos + token.size();
    auto const valueEnd = message.find('\n', valueBegin);
    TLLM_CHECK_WITH_INFO(valueEnd != std::string::npos && valueEnd + 1 == message.size() && valueEnd > valueBegin,
        "%s marker has an invalid pair-count suffix: '%s'", markerName, message.c_str());

    uint64_t pairCount{0};
    auto const* begin = message.data() + valueBegin;
    auto const* end = message.data() + valueEnd;
    auto const result = std::from_chars(begin, end, pairCount);
    TLLM_CHECK_WITH_INFO(result.ec == std::errc{} && result.ptr == end,
        "%s marker has a non-numeric pair count: '%s'", markerName, message.c_str());
    return pairCount;
}

} // namespace


std::string genUniqueAgentName()
{
    static std::atomic<uint64_t> counter{0};

    // Generate a per-process random suffix to disambiguate agents across containers
    // that may share the same hostname (--network host) and PID namespace.
    static uint64_t const sRandomSuffix = []()
    {
        std::random_device rd;
        return (static_cast<uint64_t>(rd()) << 32) | rd();
    }();

    char hostname[1024];
    gethostname(hostname, sizeof(hostname));
    auto pid = static_cast<uint64_t>(::getpid());
    return std::string(hostname) + "_" + std::to_string(pid) + "_" + std::to_string(sRandomSuffix) + "_"
        + std::to_string(counter++);
}

// NIXL connection is specific, and different from the UCX and mpi connection,
// since NIXL only support one-sided communication. gen send buffer metaData to
// context when it sending requestInfo, but don't send buffer offset, since
// unformmatter has not called yet, it didn't know the cacheSize and offset. We
// assume the recv_size is the same as the send_size. and compute the buffer
// offset according to  the layer num of the selfPPrank ,and previous PP rank's
// layer num, since the buffer size is ratio is equal to the layer num ratio
// except the VSWA case.

template <typename CacheStateT>
auto computeSendOffsetRatio(
    CacheStateT const& peerCacheState, int peerIdx, CacheStateT const& selfCacheState, int connectionIdx)
{
    auto peerTargetInfo = targetIRanks(selfCacheState, peerCacheState, peerIdx);
    size_t offsetLayer = 0;
    for (int i = 0; i < connectionIdx; i++)
    {
        offsetLayer += peerTargetInfo.getPeerPPDomainLayerNum(i);
    }

    size_t selfSendLayer = peerTargetInfo.getPeerPPDomainLayerNum(connectionIdx);
    return std::make_pair(offsetLayer, selfSendLayer);
}

AgentConnection::AgentConnection(
    std::string mAgentName, std::string mRemoteAgentName, AgentConnectionManager* mAgentConnectionManager)
    : mAgentName(mAgentName)
    , mRemoteAgentName(mRemoteAgentName)
    , mAgentConnectionManager(mAgentConnectionManager)
    , mCacheTransBufferManagers(mAgentConnectionManager->getCacheTransBufferManagers())
    , mNeedSendMetadata(true)
{
    TLLM_CHECK(mAgentConnectionManager != nullptr);
    TLLM_CHECK(!mCacheTransBufferManagers.empty());
}

MemoryDesc const& AgentConnection::SenderState::activeBufferDesc() const
{
    TLLM_CHECK(!mCacheReceiverBufferDescs.empty());
    TLLM_CHECK(mActiveBufferIdx < mCacheReceiverBufferDescs.size());
    return mCacheReceiverBufferDescs[mActiveBufferIdx];
}

std::pair<size_t, size_t> const& AgentConnection::SenderState::activeOffsetRatio() const
{
    TLLM_CHECK(!mOffsetRatios.empty());
    TLLM_CHECK(mActiveBufferIdx < mOffsetRatios.size());
    return mOffsetRatios[mActiveBufferIdx];
}

void AgentConnection::SenderState::setActiveBufferIdx(size_t bufferIdx) const
{
    TLLM_CHECK(bufferIdx < mCacheReceiverBufferDescs.size());
    mActiveBufferIdx = bufferIdx;
}

void MemoryDesc::serialize(MemoryDesc const& memoryDesc, std::ostream& os)
{
    namespace su = executor::serialize_utils;
    su::serialize(memoryDesc.mAddr, os);
    su::serialize(memoryDesc.mLen, os);
    su::serialize(memoryDesc.mDeviceId, os);
}

MemoryDesc MemoryDesc::deserialize(std::istream& is)
{
    namespace su = executor::serialize_utils;
    auto addr = su::deserialize<decltype(mAddr)>(is);
    auto len = su::deserialize<decltype(mLen)>(is);
    auto deviceId = su::deserialize<decltype(mDeviceId)>(is);
    return MemoryDesc{addr, len, deviceId};
}

size_t MemoryDesc::serializedSize(MemoryDesc const& memoryDesc)
{
    namespace su = executor::serialize_utils;
    return su::serializedSize(memoryDesc.mAddr) + su::serializedSize(memoryDesc.mLen)
        + su::serializedSize(memoryDesc.mDeviceId);
}

void AgentConnection::send(DataContext const& ctx, void const* data, size_t size) const
{
    MemoryDesc srcDesc{
        reinterpret_cast<uintptr_t>(data), size, static_cast<uint32_t>(mAgentConnectionManager->getDeviceId())};
    MemoryDescs srcDescs{MemoryType::kVRAM, {srcDesc}};
    auto const& dstBaseDesc = mSenderState.activeBufferDesc();
    auto const& offsetRatio = mSenderState.activeOffsetRatio();
    auto offset = size / offsetRatio.second * offsetRatio.first;
    MemoryDesc dstDesc{dstBaseDesc.getAddr() + offset, size, dstBaseDesc.getDeviceId()};
    TLLM_LOG_DEBUG(
        "send dstDesc: %p, size: %ld ,validSegmentIdx: %ld", dstDesc.getAddr(), size, mSenderState.validSegmentIdx);
    MemoryDescs dstDescs{MemoryType::kVRAM, {dstDesc}};
    TransferRequest request{TransferOp::kWRITE, srcDescs, dstDescs, mRemoteAgentName};
    auto status = mAgentConnectionManager->getAgent()->submitTransferRequests(request);
    NotificationSyncInfo syncInfo{mRemoteAgentName, ctx};
    NotificationInfo notificationInfo{syncInfo};
    std::stringstream ss;
    NotificationInfo::serialize(notificationInfo, ss);
    TransferState transferState = status->wait();
    TLLM_CHECK_WITH_INFO(transferState == TransferState::kSUCCESS, "AgentConnection::send failed");
    // TODO: there is a bug in request_with_notify https://github.com/ai-dynamo/nixl/pull/252
    mAgentConnectionManager->getAgent()->notifySyncMessage(mRemoteAgentName, ss.str());
}

void AgentConnection::recv(DataContext const& ctx, void* data, size_t size) const
{

    NotificationSyncInfo syncInfo{mAgentName, ctx};
    mAgentConnectionManager->waitForSyncInfo(mRemoteAgentName, syncInfo, ctx.getTransferTerminate());
}

void AgentConnection::sendRequestAndBufferInfo(batch_manager::RequestInfo& requestInfo,
    std::vector<std::optional<size_t>> const& cacheBufferIds, int connectionIdx,
    std::optional<PagedTransferMetadata> pagedTransferMetadata)
{
    TLLM_CHECK(!common::getEnvTryZCopyForKVCacheTransfer());

    auto const diagRequestId = requestInfo.getRequestId();
    if (mooncakePagedGinDiagEnabled())
    {
        TLLM_LOG_INFO(mpi::MpiComm::world().getRank(),
            "MOONCAKE_PAGED_GIN_DIAG agent_send_request_info_enter request_id=%lu remote=%s connection_idx=%d paged=%d",
            static_cast<unsigned long>(diagRequestId), mRemoteAgentName.c_str(), connectionIdx,
            pagedTransferMetadata.has_value() ? 1 : 0);
    }

    TLLM_CHECK(!cacheBufferIds.empty());
    TLLM_CHECK(cacheBufferIds.size() <= mCacheTransBufferManagers.size());

    auto const& allKinds = mAgentConnectionManager->getBufferKinds();
    std::vector<runtime::ITensor::SharedPtr> preAllocateBuffers;
    std::vector<MemoryDesc> bufferDescs;
    std::vector<std::optional<size_t>> activeCacheBufferIds;
    std::vector<uint8_t> activeKinds;

    for (size_t i = 0; i < cacheBufferIds.size(); i++)
    {
        if (!cacheBufferIds[i].has_value())
        {
            continue;
        }
        auto preAllocateBuffer = mCacheTransBufferManagers[i]->getRecvBuffer(cacheBufferIds[i].value());
        TLLM_CHECK(preAllocateBuffer != nullptr);
        preAllocateBuffers.push_back(preAllocateBuffer);
        activeCacheBufferIds.push_back(cacheBufferIds[i]);
        activeKinds.push_back(allKinds[i]);
    }
    TLLM_CHECK(!activeCacheBufferIds.empty());

    mCacheBufferIds = std::move(activeCacheBufferIds);
    mBufferKinds = activeKinds;

    int deviceId = -1;
    TLLM_CUDA_CHECK(cudaGetDevice(&deviceId));
    TLLM_CHECK(deviceId != -1);
    TLLM_CHECK(deviceId == mAgentConnectionManager->getDeviceId());
    for (auto const& buf : preAllocateBuffers)
    {
        bufferDescs.emplace_back(reinterpret_cast<uintptr_t>(buf->data()), buf->getSizeInBytes(), deviceId);
    }
    std::string address = mAgentConnectionManager->getAgent()->getLocalConnectionInfo();
    std::optional<std::string> metadataOpt = std::nullopt;
    if (mNeedSendMetadata)
    {
        auto metadata = mAgentConnectionManager->getAgent()->getLocalAgentDesc().getBackendAgentDesc();
        metadataOpt = metadata;
        mNeedSendMetadata = false;
    }

    if (pagedTransferMetadata.has_value())
    {
        if (mooncakePagedGinDiagEnabled())
        {
            TLLM_LOG_INFO(mpi::MpiComm::world().getRank(),
                "MOONCAKE_PAGED_GIN_DIAG agent_validate_dest_begin request_id=%lu addr=%p len=%lu",
                static_cast<unsigned long>(diagRequestId),
                reinterpret_cast<void*>(pagedTransferMetadata->mRegisteredMemory.getAddr()),
                pagedTransferMetadata->mRegisteredMemory.getLen());
        }
        validateMemoryForPagedTransfer(pagedTransferMetadata->mRegisteredMemory);
        if (mooncakePagedGinDiagEnabled())
        {
            TLLM_LOG_INFO(mpi::MpiComm::world().getRank(),
                "MOONCAKE_PAGED_GIN_DIAG agent_validate_dest_end request_id=%lu",
                static_cast<unsigned long>(diagRequestId));
        }
    }

    RequestAndBufferInfo requestAndBufferInfo{mAgentName, address, requestInfo, bufferDescs, metadataOpt, connectionIdx,
        activeKinds, std::move(pagedTransferMetadata)};
    std::stringstream ss;
    NotificationInfo notificationInfo{requestAndBufferInfo};
    NotificationInfo::serialize(notificationInfo, ss);
    auto payload = ss.str();
    if (mooncakePagedGinDiagEnabled())
    {
        TLLM_LOG_INFO(mpi::MpiComm::world().getRank(),
            "MOONCAKE_PAGED_GIN_DIAG agent_notify_request_info_begin request_id=%lu remote=%s bytes=%lu",
            static_cast<unsigned long>(diagRequestId), mRemoteAgentName.c_str(), payload.size());
    }
    mAgentConnectionManager->getAgent()->notifySyncMessage(mRemoteAgentName, payload);
    if (mooncakePagedGinDiagEnabled())
    {
        TLLM_LOG_INFO(mpi::MpiComm::world().getRank(),
            "MOONCAKE_PAGED_GIN_DIAG agent_notify_request_info_end request_id=%lu remote=%s",
            static_cast<unsigned long>(diagRequestId), mRemoteAgentName.c_str());
    }
}

void AgentConnection::preconnect() const
{
    TLLM_CHECK(mAgentConnectionManager->supportsPagedTransfer());
    mAgentConnectionManager->getAgent()->preconnectRemoteAgent(mRemoteAgentName);
}

void AgentConnection::preconnect(MemoryDesc const& remotePool) const
{
    TLLM_CHECK(mAgentConnectionManager->supportsPagedTransfer());
    mAgentConnectionManager->getAgent()->preconnectPagedRemoteAgent(
        mRemoteAgentName, mAgentConnectionManager->getPagedPoolMemory(), remotePool);
}

void AgentConnection::setSenderState(std::vector<MemoryDesc> cacheReceiverBufferDescs, int validSegmentIdx,
    std::vector<std::pair<size_t, size_t>> offsetRatios, std::vector<uint8_t> bufferKinds)
{
    TLLM_CHECK(!cacheReceiverBufferDescs.empty());
    TLLM_CHECK(offsetRatios.size() == cacheReceiverBufferDescs.size());
    TLLM_CHECK(bufferKinds.size() == cacheReceiverBufferDescs.size());
    mSenderState.mCacheReceiverBufferDescs = std::move(cacheReceiverBufferDescs);
    mSenderState.validSegmentIdx = validSegmentIdx;
    mSenderState.mOffsetRatios = std::move(offsetRatios);
    mSenderState.setActiveBufferIdx(0);
    mBufferKinds = std::move(bufferKinds);
}

void AgentConnection::sendPagedTransfer(DataContext const& ctx, PagedTransferMetadata const& localMetadata) const
{
    TLLM_CHECK_WITH_INFO(mPagedTransferMetadata.has_value(),
        "MOONCAKE_PAGED_GIN sender did not receive destination paged KV metadata");
    auto const& remoteMetadata = mPagedTransferMetadata.value();
    TLLM_CHECK_WITH_INFO(localMetadata.mPageBytes == remoteMetadata.mPageBytes,
        "MOONCAKE_PAGED_GIN source/destination page size mismatch: %lu vs %lu", localMetadata.mPageBytes,
        remoteMetadata.mPageBytes);
    TLLM_CHECK_WITH_INFO(localMetadata.mPageIndices.size() == remoteMetadata.mPageIndices.size(),
        "MOONCAKE_PAGED_GIN source/destination page table size mismatch: %lu vs %lu",
        localMetadata.mPageIndices.size(), remoteMetadata.mPageIndices.size());
    TLLM_CHECK_WITH_INFO(localMetadata.mLayerPtrs.size() == remoteMetadata.mLayerPtrs.size(),
        "MOONCAKE_PAGED_GIN source/destination layer pointer count mismatch: %lu vs %lu",
        localMetadata.mLayerPtrs.size(), remoteMetadata.mLayerPtrs.size());
    TLLM_CHECK_WITH_INFO(localMetadata.mLayoutFingerprint != 0
            && localMetadata.mLayoutFingerprint == remoteMetadata.mLayoutFingerprint,
        "MOONCAKE_PAGED_GIN source/destination layout fingerprint mismatch: %llu vs %llu",
        static_cast<unsigned long long>(localMetadata.mLayoutFingerprint),
        static_cast<unsigned long long>(remoteMetadata.mLayoutFingerprint));

    if (mooncakePagedGinDiagEnabled())
    {
        TLLM_LOG_INFO(mpi::MpiComm::world().getRank(),
            "MOONCAKE_PAGED_GIN_DIAG agent_validate_source_begin tag=%lu addr=%p len=%lu",
            static_cast<unsigned long>(ctx.getTag()), reinterpret_cast<void*>(localMetadata.mRegisteredMemory.getAddr()),
            localMetadata.mRegisteredMemory.getLen());
    }
    validateMemoryForPagedTransfer(localMetadata.mRegisteredMemory);
    if (mooncakePagedGinDiagEnabled())
    {
        TLLM_LOG_INFO(mpi::MpiComm::world().getRank(),
            "MOONCAKE_PAGED_GIN_DIAG agent_validate_source_end tag=%lu", static_cast<unsigned long>(ctx.getTag()));
    }
    PagedTransferRequest request{localMetadata.mLayerPtrs, remoteMetadata.mLayerPtrs, localMetadata.mPageIndices,
        remoteMetadata.mPageIndices, localMetadata.mPageBytes, mRemoteAgentName};
    if (mooncakePagedGinDiagEnabled())
    {
        TLLM_LOG_INFO(mpi::MpiComm::world().getRank(),
            "MOONCAKE_PAGED_GIN_DIAG agent_submit_paged_begin tag=%lu remote=%s pages=%lu layout=%llu",
            static_cast<unsigned long>(ctx.getTag()), mRemoteAgentName.c_str(), localMetadata.mPageIndices.size(),
            static_cast<unsigned long long>(localMetadata.mLayoutFingerprint));
    }
    mAgentConnectionManager->getAgent()->submitPagedTransferRequest(request);
    if (mooncakePagedGinDiagEnabled())
    {
        TLLM_LOG_INFO(mpi::MpiComm::world().getRank(),
            "MOONCAKE_PAGED_GIN_DIAG agent_submit_paged_end tag=%lu remote=%s", static_cast<unsigned long>(ctx.getTag()),
            mRemoteAgentName.c_str());
    }

    NotificationSyncInfo syncInfo{mRemoteAgentName, ctx};
    NotificationInfo notificationInfo{syncInfo};
    std::stringstream ss;
    NotificationInfo::serialize(notificationInfo, ss);
    mAgentConnectionManager->getAgent()->notifySyncMessage(mRemoteAgentName, ss.str());
}

std::optional<PagedTransferMetadata> const& AgentConnection::getPagedTransferMetadata() const
{
    return mPagedTransferMetadata;
}

void AgentConnection::setPagedTransferMetadata(std::optional<PagedTransferMetadata> pagedTransferMetadata)
{
    mPagedTransferMetadata = std::move(pagedTransferMetadata);
}

void AgentConnection::validateMemoryForPagedTransfer(MemoryDesc const& desc) const
{
    mAgentConnectionManager->validateMemoryForPagedTransfer(desc);
}

void AgentConnection::setHasLoadRemoteAgent(bool hasLoadRemoteAgent)
{
    mHasLoadRemoteAgent = hasLoadRemoteAgent;
}

bool AgentConnection::hasLoadRemoteAgent() const
{
    return mHasLoadRemoteAgent;
}

void AgentConnection::sendReadySignal(DataContext const& ctx, bool isReady) const
{
    ReadySignalInfo readySignalInfo{mRemoteAgentName, ctx, isReady};
    NotificationInfo notificationInfo{readySignalInfo};
    std::stringstream ss;
    NotificationInfo::serialize(notificationInfo, ss);
    mAgentConnectionManager->getAgent()->notifySyncMessage(mRemoteAgentName, ss.str());
}

bool AgentConnection::recvReadySignal(DataContext const& ctx) const
{
    ReadySignalInfo readySignalInfo{mAgentName, ctx, false};
    mAgentConnectionManager->waitForReadySignal(mRemoteAgentName, readySignalInfo, ctx.getTransferTerminate());
    return readySignalInfo.mIsReady;
}

void AgentConnection::activateBuffer(uint8_t kind) const
{
    for (size_t i = 0; i < mBufferKinds.size(); i++)
    {
        if (mBufferKinds[i] == kind)
        {
            mSenderState.setActiveBufferIdx(i);
            return;
        }
    }
}

std::optional<size_t> AgentConnection::getPreAssignedBufferId(uint8_t kind) const
{
    for (size_t i = 0; i < mBufferKinds.size(); i++)
    {
        if (mBufferKinds[i] == kind && i < mCacheBufferIds.size())
        {
            return mCacheBufferIds[i];
        }
    }
    return std::nullopt;
}

AgentConnectionManager::AgentConnectionManager(
    std::vector<batch_manager::BaseTransBufferManager*> cacheTransBufferManagers, CacheState cacheState,
    std::string const& backendType, std::optional<CacheState::RnnCacheState> rnnCacheState,
    std::optional<MemoryDesc> pagedPoolMemory)
    : mCacheState(std::move(cacheState))
    , mBackendType(backendType)
    , mRnnCacheState(std::move(rnnCacheState))
    , mCacheTransBufferManagers(std::move(cacheTransBufferManagers))
    , mRegMemDescs(MemoryType::kVRAM, {})
    , mPagedPoolMemory(std::move(pagedPoolMemory))
{
    TLLM_CUDA_CHECK(cudaGetDevice(&mDeviceId));
    TLLM_CHECK(mDeviceId != -1);

    mAgentName = genUniqueAgentName();
    // Create Agent
    BaseAgentConfig config{mAgentName, true, false, true};
    m_Agent = makeTransferAgent(backendType, &config);
    TLLM_CHECK(!mCacheTransBufferManagers.empty());
    mBufferKinds.reserve(mCacheTransBufferManagers.size());
    std::vector<MemoryDesc> memDescs;
    for (auto* cacheTransBufferManager : mCacheTransBufferManagers)
    {
        TLLM_CHECK(cacheTransBufferManager != nullptr);
        mBufferKinds.push_back(static_cast<uint8_t>(cacheTransBufferManager->getBufferKind()));
        auto recvBufferCount = cacheTransBufferManager->getRecvBufferCount();
        auto sendBufferCount = cacheTransBufferManager->getSendBufferCount();
        for (size_t i = 0; i < recvBufferCount; i++)
        {
            auto recvBuffer = cacheTransBufferManager->getRecvBuffer(i);
            memDescs.emplace_back(recvBuffer->data(), recvBuffer->getSizeInBytes(), mDeviceId);
        }
        for (size_t i = 0; i < sendBufferCount; i++)
        {
            auto sendBuffer = cacheTransBufferManager->getSendBuffer(i);
            memDescs.emplace_back(sendBuffer->data(), sendBuffer->getSizeInBytes(), mDeviceId);
        }
    }
    mRegMemDescs = MemoryDescs{MemoryType::kVRAM, memDescs};
    m_Agent->registerMemory(mRegMemDescs);
    if (supportsPagedTransfer())
    {
        TLLM_CHECK_WITH_INFO(mPagedPoolMemory.has_value(),
            "MOONCAKE_PAGED_GIN requires primary KV-pool memory during connection-manager startup");
        registerMemoryForPagedTransfer(mPagedPoolMemory.value());
    }

    AgentState localAgentState{mAgentName, m_Agent->getLocalConnectionInfo()};
    std::vector<AgentState> agentStates(mpi::MpiComm::session().getSize());
    if (mpi::MpiComm::session().getSize() > 1)
    {

        mpi::MpiComm::session().barrier();
        namespace su = executor::serialize_utils;

        std::ostringstream oStream;
        su::serialize(localAgentState, oStream);
        auto str = oStream.str();
        std::vector<char> buffer(str.begin(), str.end());
        std::vector<SizeType32> sizeofBuffer(mpi::MpiComm::session().getSize());
        SizeType32 bufferSize = buffer.size();
        mpi::MpiComm::session().allgather(&bufferSize, sizeofBuffer.data(), 1, mpi::MpiType::kINT32);
        SizeType32 recvBufferSize = std::accumulate(sizeofBuffer.begin(), sizeofBuffer.end(), 0);
        std::vector<char> recvBuffer(recvBufferSize);
        std::vector<int> displs(mpi::MpiComm::session().getSize());
        for (int r = 0; r < mpi::MpiComm::session().getSize(); r++)
        {
            displs[r] = (r == 0) ? 0 : (displs[r - 1] + sizeofBuffer[r - 1]);
        }
        mpi::MpiComm::session().allgatherv(buffer.data(), bufferSize, mpi::MpiType::kCHAR, recvBuffer.data(),
            sizeofBuffer, displs, mpi::MpiType::kCHAR);

        // deserialize
        for (int i = 0; i < mpi::MpiComm::session().getSize(); i++)
        {
            std::vector<char> serBuffer(
                recvBuffer.begin() + displs[i], recvBuffer.begin() + (displs[i] + sizeofBuffer[i]));
            su::VectorWrapBuf<char> strbuf(serBuffer);
            std::istream is(&strbuf);
            agentStates[i] = su::deserialize<executor::kv_cache::AgentState>(is);
            TLLM_LOG_DEBUG(
                mpi::MpiComm::world().getRank(), " recv  agentStates[%d]: %s", i, agentStates[i].toString().c_str());
        }
    }
    else
    {
        agentStates[0] = localAgentState;
    }
    mCommState = CommState(agentStates, mpi::MpiComm::session().getRank());
    TLLM_LOG_DEBUG(mpi::MpiComm::world().getRank(),
        " ***** AgentConnectionManager::AgentConnectionManager    mCommState: %s", mCommState.toString().c_str());
    runStartupPreconnect();
}

void AgentConnectionManager::runStartupPreconnect()
{
    if (!supportsPagedTransfer() || !common::getBoolEnv("TRTLLM_MOONCAKE_PAGED_GIN_STARTUP_PRECONNECT"))
    {
        return;
    }

    namespace fs = std::filesystem;
    namespace su = executor::serialize_utils;
    static std::atomic<int> nextStartupEpoch{0};
    auto& session = mpi::MpiComm::session();
    auto const sessionRank = session.getRank();
    auto const startupEpoch = nextStartupEpoch.fetch_add(1, std::memory_order_relaxed);
    auto const role = requireStartupEnv("TRTLLM_MOONCAKE_PAGED_GIN_PRECONNECT_ROLE");
    auto const directory = fs::path(requireStartupEnv("TRTLLM_MOONCAKE_PAGED_GIN_PRECONNECT_DIR"));
    auto const instanceId = getStartupEnvInt("TRTLLM_MOONCAKE_PAGED_GIN_PRECONNECT_INSTANCE", 1) - 1;
    auto const contextInstances = getStartupEnvInt("TRTLLM_MOONCAKE_PAGED_GIN_CTX_INSTANCES", 1);
    auto const generationInstances = getStartupEnvInt("TRTLLM_MOONCAKE_PAGED_GIN_GEN_INSTANCES", 1);
    auto const rendezvousTimeout
        = std::chrono::seconds(getStartupEnvInt("TRTLLM_MOONCAKE_PAGED_GIN_RENDEZVOUS_TIMEOUT_SECONDS", 900));
    auto const initTimeout
        = std::chrono::seconds(getStartupEnvInt("TRTLLM_MOONCAKE_PAGED_GIN_INIT_TIMEOUT_SECONDS", 900));
    auto const makeDeadline = [](std::chrono::seconds timeout)
    {
        return std::chrono::steady_clock::now() + timeout;
    };
    auto const contextStatePath = [&](int context)
    {
        // CTX and GEN both rebuild their connection managers after allocating
        // the final NCCL-window KV pool. Epoch 1 republishes this stable path
        // with the exact agents and pool descriptor used during serving. GEN
        // reads it after the epoch-1 CTX acknowledgement, so it cannot observe
        // the provisional epoch-0 descriptor at final preconnect.
        return directory / ("ctx-" + std::to_string(context) + ".states-v2");
    };
    auto const generationMarkerPath = [&](int generation, int epoch, char const* suffix)
    {
        return directory
            / ("gen-" + std::to_string(generation) + "-epoch-" + std::to_string(epoch) + "-v2" + suffix);
    };
    auto const generationStatePath = [&](int generation, int epoch)
    {
        return directory / ("gen-" + std::to_string(generation) + "-epoch-" + std::to_string(epoch) + ".states-v2");
    };
    auto const contextAckPath = [&](int generation, int epoch)
    {
        return directory / ("ctx-ack-gen-" + std::to_string(generation) + "-epoch-" + std::to_string(epoch));
    };

    TLLM_CHECK_WITH_INFO(role == "CTX" || role == "GEN", "Invalid startup preconnect role %s", role.c_str());
    TLLM_LOG_INFO(mpi::MpiComm::world().getRank(),
        "MOONCAKE_PAGED_GIN startup preconnect enter role=%s instance=%d epoch=%d local_rank=%d ctx_instances=%d gen_instances=%d dir=%s",
        role.c_str(), instanceId, startupEpoch, sessionRank, contextInstances, generationInstances, directory.c_str());

    if (role == "CTX")
    {
        TLLM_CHECK_WITH_INFO(startupEpoch == 0 || startupEpoch == 1,
            "Context startup preconnect expected provisional epoch 0 or final epoch 1, got epoch %d", startupEpoch);
        auto localState = DataTransceiverState{mCacheState, mCommState};
        TLLM_CHECK(mPagedPoolMemory.has_value());
        auto localPayload = serializeStartupPeerState(localState, mPagedPoolMemory.value());
        auto rankPayloads = allgatherStartupPayloads(session, localPayload);
        if (sessionRank == 0)
        {
            auto payload = serializeStartupPeerStateBundle(rankPayloads);
            auto const statePath = contextStatePath(instanceId);
            writeStartupFile(statePath, payload);
            TLLM_LOG_INFO(mpi::MpiComm::world().getRank(),
                "MOONCAKE_PAGED_GIN startup preconnect published stable context state instance=%d epoch=%d rank_states=%zu bytes=%zu path=%s",
                instanceId, startupEpoch, rankPayloads.size(), payload.size(), statePath.c_str());

            if (startupEpoch == 0)
            {
                for (int generation = 0; generation < generationInstances; ++generation)
                {
                    auto const readyPath = generationMarkerPath(generation, startupEpoch, ".ready");
                    auto const failedPath = generationMarkerPath(generation, startupEpoch, ".failed");
                    auto const readyDeadline = makeDeadline(rendezvousTimeout);
                    while (!startupPathExists(readyPath))
                    {
                        TLLM_CHECK_WITH_INFO(!startupPathExists(failedPath),
                            "Generation instance %d epoch %d failed during startup preconnect; see %s", generation,
                            startupEpoch, failedPath.c_str());
                        TLLM_CHECK_WITH_INFO(std::chrono::steady_clock::now() < readyDeadline,
                            "Timed out waiting for generation instance %d epoch %d startup preconnect", generation,
                            startupEpoch);
                        std::this_thread::sleep_for(std::chrono::milliseconds(200));
                    }
                    auto const readyPayload = readStartupFile(readyPath);
                    auto const readyMessage = std::string(readyPayload.begin(), readyPayload.end());
                    auto const readyPairCount = parseStartupPairCount(readyPayload, "Generation ready");
                    auto const expectedPairCount = static_cast<uint64_t>(session.getSize())
                        * static_cast<uint64_t>(session.getSize()) * static_cast<uint64_t>(contextInstances);
                    auto const expectedReadyMessage
                        = "OK provisional pairs=" + std::to_string(expectedPairCount) + "\n";
                    TLLM_CHECK_WITH_INFO(readyPairCount == expectedPairCount && readyMessage == expectedReadyMessage,
                        "Generation instance %d epoch %d reported startup result %s, expected %s", generation,
                        startupEpoch, readyMessage.c_str(), expectedReadyMessage.c_str());
                }
            }
        }
        session.barrier();
        if (startupEpoch == 0)
        {
            TLLM_LOG_INFO(mpi::MpiComm::world().getRank(),
                "MOONCAKE_PAGED_GIN provisional startup preconnect complete role=CTX "
                "instance=%d epoch=%d local_rank=%d",
                instanceId, startupEpoch, sessionRank);
            return;
        }
        for (int generation = 0; generation < generationInstances; ++generation)
        {
            std::vector<char> generationBundle;
            if (sessionRank == 0)
            {
                auto const statePath = generationStatePath(generation, 1);
                waitForStartupPath(statePath, makeDeadline(rendezvousTimeout));
                generationBundle = readStartupFile(statePath);
            }
            session.bcast(generationBundle, 0);
            auto peerRankPayloads = deserializeStartupPeerStateBundle(generationBundle);
            TLLM_CHECK_WITH_INFO(
                !peerRankPayloads.empty(), "Startup preconnect received an empty generation state bundle");

            std::optional<std::vector<AgentState>> expectedAgentStates;
            std::vector<Connection const*> connections;
            std::vector<std::pair<DataTransceiverState, MemoryDesc>> peers;
            peers.reserve(peerRankPayloads.size());
            for (size_t peerRank = 0; peerRank < peerRankPayloads.size(); ++peerRank)
            {
                auto peer = deserializeStartupPeerState(peerRankPayloads[peerRank]);
                auto const& peerState = peer.first;
                auto const& peerPool = peer.second;
                TLLM_CHECK(peerState.getCacheState().has_value());
                TLLM_CHECK(peerState.getCommState().has_value());
                auto const& peerCommState = peerState.getCommState().value();
                TLLM_CHECK_WITH_INFO(peerCommState.isAgentState(),
                    "Startup preconnect generation rank %zu did not publish an agent communication state", peerRank);
                TLLM_CHECK_WITH_INFO(peerCommState.getSelfIdx() == static_cast<int>(peerRank),
                    "Startup preconnect generation payload index %zu contains self rank %d", peerRank,
                    peerCommState.getSelfIdx());
                TLLM_CHECK_WITH_INFO(peerCommState.getAgentState().size() == peerRankPayloads.size(),
                    "Startup preconnect generation rank %zu advertises %zu agents, but the bundle contains %zu ranks",
                    peerRank, peerCommState.getAgentState().size(), peerRankPayloads.size());
                TLLM_CHECK_WITH_INFO(peerPool.getAddr() != 0 && peerPool.getLen() != 0,
                    "Startup preconnect generation rank %zu published an empty KV-pool descriptor", peerRank);
                if (!expectedAgentStates.has_value())
                {
                    expectedAgentStates = peerCommState.getAgentState();
                    connections = getConnections(peerCommState);
                }
                else
                {
                    TLLM_CHECK_WITH_INFO(expectedAgentStates.value() == peerCommState.getAgentState(),
                        "Startup preconnect generation rank %zu published an inconsistent agent list", peerRank);
                }
                peers.emplace_back(std::move(peer));
            }

            auto const peerCount = peers.size();
            TLLM_CHECK_WITH_INFO(peerCount == static_cast<size_t>(session.getSize()),
                "Rank-major startup preconnect requires equal local and peer rank counts: local=%d peer=%zu",
                session.getSize(), peerCount);
            TLLM_CHECK_WITH_INFO(connections.size() == peerCount,
                "Startup preconnect loaded %zu generation agents, expected %zu", connections.size(), peerCount);

            auto const localAgentCount = static_cast<uint64_t>(connections.size());

            uint64_t instanceAgentCount{0};
            session.allreduce(
                &localAgentCount, &instanceAgentCount, 1, mpi::MpiType::kUINT64, mpi::MpiOp::SUM);
            auto const expectedAgentCount
                = static_cast<uint64_t>(peerCount) * static_cast<uint64_t>(session.getSize());
            TLLM_CHECK_WITH_INFO(instanceAgentCount == expectedAgentCount,
                "Incomplete final-pool callback coverage: got %lu, expected %lu",
                static_cast<unsigned long>(instanceAgentCount),
                static_cast<unsigned long>(expectedAgentCount));
            TLLM_LOG_INFO(mpi::MpiComm::world().getRank(),
                "MOONCAKE_PAGED_GIN final-pool callbacks ready local_agents=%lu instance_agents=%lu role=CTX "
                "ctx_instance=%d gen_instance=%d local_rank=%d",
                static_cast<unsigned long>(localAgentCount), static_cast<unsigned long>(instanceAgentCount),
                instanceId, generation, sessionRank);

            auto const failedPath = generationMarkerPath(generation, 1, ".failed");
            std::mutex primaryWatchdogMutex;
            std::condition_variable primaryWatchdogCv;
            bool primaryWatchdogDone{false};
            std::thread primaryWatchdog([&]()
            {
                std::unique_lock lock(primaryWatchdogMutex);
                if (!primaryWatchdogCv.wait_for(lock, initTimeout, [&]() { return primaryWatchdogDone; }))
                {
                    writeStartupMarkerNoThrow(failedPath, "CTX primary final-pool preconnect timed out\n");
                    TLLM_LOG_ERROR(mpi::MpiComm::world().getRank(),
                        "MOONCAKE_PAGED_GIN final-pool CTX manager preconnect timed out after %lld seconds "
                        "ctx_instance=%d gen_instance=%d local_rank=%d",
                        static_cast<long long>(initTimeout.count()), instanceId, generation, sessionRank);
                    std::_Exit(124);
                }
            });
            auto stopPrimaryWatchdog = [&]()
            {
                {
                    std::lock_guard lock(primaryWatchdogMutex);
                    primaryWatchdogDone = true;
                }
                primaryWatchdogCv.notify_all();
                primaryWatchdog.join();
            };

            uint64_t localPairCount{0};
            uint64_t instancePairCount{0};
            try
            {
                for (size_t round = 0; round < peerCount; ++round)
                {
                    // Preserve the proven v17 ownership and collective order:
                    // every CTX rank initializes GEN rank 0, then rank 1, etc.
                    auto const peerRank = round;
                    auto const& peerState = peers[peerRank].first;
                    auto const& peerPool = peers[peerRank].second;
                    auto const& peerCommState = peerState.getCommState().value();
                    auto const counterparts = targetIRanks(
                        mCacheState, peerState.getCacheState().value(), peerCommState.getSelfIdx()).mIRanks;
                    if (std::find(counterparts.begin(), counterparts.end(), mCommState.getSelfIdx())
                        != counterparts.end())
                    {
                        auto const* connection
                            = connections.at(static_cast<size_t>(peerCommState.getSelfIdx()));
                        auto const* agentConnection = dynamic_cast<AgentConnection const*>(connection);
                        TLLM_CHECK(agentConnection != nullptr);
                        agentConnection->preconnect(peerPool);
                        ++localPairCount;
                    }
                    session.barrier();
                }
                session.allreduce(
                    &localPairCount, &instancePairCount, 1, mpi::MpiType::kUINT64, mpi::MpiOp::SUM);
                auto const oneToOnePairCount = static_cast<uint64_t>(peerCount);
                auto const allToAllPairCount = oneToOnePairCount * oneToOnePairCount;
                TLLM_CHECK_WITH_INFO(
                    instancePairCount == oneToOnePairCount || instancePairCount == allToAllPairCount,
                    "Unsupported or incomplete final-pool CTX pair coverage: got %lu, expected %lu or %lu",
                    static_cast<unsigned long>(instancePairCount),
                    static_cast<unsigned long>(oneToOnePairCount),
                    static_cast<unsigned long>(allToAllPairCount));
                TLLM_LOG_INFO(mpi::MpiComm::world().getRank(),
                    "MOONCAKE_PAGED_GIN final-pool CTX manager preconnect local_pairs=%lu instance_pairs=%lu "
                    "ctx_instance=%d gen_instance=%d local_rank=%d",
                    static_cast<unsigned long>(localPairCount), static_cast<unsigned long>(instancePairCount),
                    instanceId, generation, sessionRank);
                session.barrier();
                std::vector<char> readyPayload;
                if (sessionRank == 0)
                {
                    auto const ackMessage = "OK callbacks=" + std::to_string(instanceAgentCount)
                        + " pairs=" + std::to_string(instancePairCount) + "\n";
                    writeStartupFile(
                        contextAckPath(generation, 1), std::vector<char>(ackMessage.begin(), ackMessage.end()));

                    auto const readyPath = generationMarkerPath(generation, 1, ".ready");
                    auto const readyDeadline = makeDeadline(rendezvousTimeout + initTimeout);
                    while (!startupPathExists(readyPath))
                    {
                        TLLM_CHECK_WITH_INFO(!startupPathExists(failedPath),
                            "Generation instance %d final-pool preconnect failed; see %s", generation,
                            failedPath.c_str());
                        TLLM_CHECK_WITH_INFO(std::chrono::steady_clock::now() < readyDeadline,
                            "Timed out waiting for generation instance %d final-pool preconnect", generation);
                        std::this_thread::sleep_for(std::chrono::milliseconds(200));
                    }
                    readyPayload = readStartupFile(readyPath);
                }
                session.bcast(readyPayload, 0);
                auto const readyMessage = std::string(readyPayload.begin(), readyPayload.end());
                auto const readyPairCount = parseStartupPairCount(readyPayload, "Generation ready");
                auto const expectedReadyMessage = "OK pairs=" + std::to_string(instancePairCount) + "\n";
                TLLM_CHECK_WITH_INFO(
                    readyPairCount == instancePairCount && readyMessage == expectedReadyMessage,
                    "Generation instance %d reported startup result '%s', expected '%s'", generation,
                    readyMessage.c_str(), expectedReadyMessage.c_str());
                session.barrier();
                stopPrimaryWatchdog();
            }
            catch (...)
            {
                writeStartupMarkerNoThrow(failedPath, "Final-pool CTX manager preconnect failed\n");
                stopPrimaryWatchdog();
                throw;
            }
        }
        TLLM_LOG_INFO(mpi::MpiComm::world().getRank(),
            "MOONCAKE_PAGED_GIN startup preconnect complete role=CTX instance=%d epoch=%d local_rank=%d", instanceId,
            startupEpoch, sessionRank);
        return;
    }

    auto localState = DataTransceiverState{mCacheState, mCommState};
    TLLM_CHECK(mPagedPoolMemory.has_value());
    auto generationRankPayloads
        = allgatherStartupPayloads(session, serializeStartupPeerState(localState, mPagedPoolMemory.value()));
    auto const readyPath = generationMarkerPath(instanceId, startupEpoch, ".ready");
    auto const failedPath = generationMarkerPath(instanceId, startupEpoch, ".failed");
    if (sessionRank == 0)
    {
        writeStartupFile(generationStatePath(instanceId, startupEpoch),
            serializeStartupPeerStateBundle(generationRankPayloads));
    }
    session.barrier();

    TLLM_CHECK_WITH_INFO(startupEpoch == 0 || startupEpoch == 1,
        "Generation startup preconnect expected provisional epoch 0 or final epoch 1, got epoch %d", startupEpoch);
    if (startupEpoch == 0 && instanceId > 0)
    {
        // A 1k/1k launch has three generation instances sharing the same
        // context RPC servers. Prime them one instance at a time so the
        // proven eight-rank v17 warm-up is not turned into a 24-to-1 burst.
        if (sessionRank == 0)
        {
            auto const priorReadyPath = generationMarkerPath(instanceId - 1, startupEpoch, ".ready");
            auto const priorFailedPath = generationMarkerPath(instanceId - 1, startupEpoch, ".failed");
            auto const priorDeadline = makeDeadline(rendezvousTimeout + initTimeout * instanceId);
            while (!startupPathExists(priorReadyPath))
            {
                TLLM_CHECK_WITH_INFO(!startupPathExists(priorFailedPath),
                    "Prior generation instance %d failed provisional preconnect; see %s", instanceId - 1,
                    priorFailedPath.c_str());
                TLLM_CHECK_WITH_INFO(std::chrono::steady_clock::now() < priorDeadline,
                    "Timed out waiting for prior generation instance %d provisional preconnect", instanceId - 1);
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
            }
        }
        session.barrier();
    }
    std::optional<uint64_t> contextPrimaryPairCount;
    if (startupEpoch == 1)
    {
        std::vector<char> contextAckPayload;
        if (sessionRank == 0)
        {
            auto const ackPath = contextAckPath(instanceId, startupEpoch);
            auto const ackTimeout = rendezvousTimeout + initTimeout * (instanceId + 1);
            waitForStartupPath(ackPath, makeDeadline(ackTimeout));
            contextAckPayload = readStartupFile(ackPath);
        }
        session.bcast(contextAckPayload, 0);
        auto const contextAckMessage = std::string(contextAckPayload.begin(), contextAckPayload.end());
        TLLM_CHECK_WITH_INFO(contextAckMessage.rfind("OK callbacks=", 0) == 0,
            "Context primary marker has invalid content '%s'", contextAckMessage.c_str());
        contextPrimaryPairCount = parseStartupPairCount(contextAckPayload, "Context primary");
        TLLM_CHECK_WITH_INFO(
            contextPrimaryPairCount.value() != 0, "Context primary marker reported zero preconnected pairs");
        session.barrier();
    }

    std::vector<char> peerStateBundle;
    if (sessionRank == 0)
    {
        std::vector<std::vector<char>> peerStatePayloads;
        peerStatePayloads.reserve(contextInstances);
        for (int context = 0; context < contextInstances; ++context)
        {
            auto const statePath = contextStatePath(context);
            waitForStartupPath(statePath, makeDeadline(rendezvousTimeout));
            peerStatePayloads.emplace_back(readStartupFile(statePath));
        }
        std::ostringstream output;
        su::serialize(peerStatePayloads, output);
        auto const serialized = output.str();
        peerStateBundle.assign(serialized.begin(), serialized.end());
    }
    session.bcast(peerStateBundle, 0);

    su::VectorWrapBuf<char> bundleBuffer(peerStateBundle);
    std::istream bundleInput(&bundleBuffer);
    auto peerStatePayloads = su::deserialize<std::vector<std::vector<char>>>(bundleInput);
    TLLM_CHECK_WITH_INFO(static_cast<int>(peerStatePayloads.size()) == contextInstances,
        "Startup preconnect received %zu context states, expected %d", peerStatePayloads.size(), contextInstances);

    std::mutex watchdogMutex;
    std::condition_variable watchdogCv;
    bool watchdogDone{false};
    std::thread watchdog([&]()
    {
        std::unique_lock lock(watchdogMutex);
        if (!watchdogCv.wait_for(lock, initTimeout, [&]() { return watchdogDone; }))
        {
            writeStartupMarkerNoThrow(failedPath,
                startupEpoch == 0 ? "NCCL provisional preconnect timed out\n"
                                  : "NCCL final-pool preconnect timed out\n");
            TLLM_LOG_ERROR(mpi::MpiComm::world().getRank(),
                "MOONCAKE_PAGED_GIN %s preconnect timed out after %lld seconds role=GEN instance=%d epoch=%d "
                "local_rank=%d",
                startupEpoch == 0 ? "provisional" : "final-pool", static_cast<long long>(initTimeout.count()),
                instanceId, startupEpoch, sessionRank);
            std::_Exit(124);
        }
    });
    auto stopWatchdog = [&]()
    {
        {
            std::lock_guard lock(watchdogMutex);
            watchdogDone = true;
        }
        watchdogCv.notify_all();
        watchdog.join();
    };

    try
    {
        session.barrier();
        uint64_t localPairCount{0};
        for (auto& peerBundle : peerStatePayloads)
        {
            auto peerRankPayloads = deserializeStartupPeerStateBundle(peerBundle);
            TLLM_CHECK_WITH_INFO(!peerRankPayloads.empty(),
                "Startup preconnect received an empty context state bundle");

            std::optional<std::vector<AgentState>> expectedAgentStates;
            std::vector<Connection const*> connections;
            std::vector<std::pair<DataTransceiverState, MemoryDesc>> peers;
            peers.reserve(peerRankPayloads.size());
            for (size_t peerRank = 0; peerRank < peerRankPayloads.size(); ++peerRank)
            {
                auto peer = deserializeStartupPeerState(peerRankPayloads[peerRank]);
                auto const& peerState = peer.first;
                auto const& peerPool = peer.second;
                TLLM_CHECK(peerState.getCacheState().has_value());
                TLLM_CHECK(peerState.getCommState().has_value());
                auto const& peerCommState = peerState.getCommState().value();
                TLLM_CHECK_WITH_INFO(peerCommState.isAgentState(),
                    "Startup preconnect context rank %zu did not publish an agent communication state", peerRank);
                TLLM_CHECK_WITH_INFO(peerCommState.getSelfIdx() == static_cast<int>(peerRank),
                    "Startup preconnect context payload index %zu contains self rank %d", peerRank,
                    peerCommState.getSelfIdx());
                TLLM_CHECK_WITH_INFO(peerCommState.getAgentState().size() == peerRankPayloads.size(),
                    "Startup preconnect context rank %zu advertises %zu agents, but the bundle contains %zu ranks",
                    peerRank, peerCommState.getAgentState().size(), peerRankPayloads.size());
                TLLM_CHECK_WITH_INFO(peerPool.getAddr() != 0 && peerPool.getLen() != 0,
                    "Startup preconnect context rank %zu published an empty KV-pool descriptor", peerRank);
                if (!expectedAgentStates.has_value())
                {
                    expectedAgentStates = peerCommState.getAgentState();
                    connections = getConnections(peerCommState);
                }
                else
                {
                    TLLM_CHECK_WITH_INFO(expectedAgentStates.value() == peerCommState.getAgentState(),
                        "Startup preconnect context rank %zu published an inconsistent agent list", peerRank);
                }
                peers.emplace_back(std::move(peer));
            }

            auto const peerCount = peers.size();
            TLLM_CHECK_WITH_INFO(peerCount == static_cast<size_t>(session.getSize()),
                "Rank-major startup preconnect requires equal local and peer rank counts: local=%d peer=%zu",
                session.getSize(), peerCount);
            TLLM_CHECK_WITH_INFO(connections.size() == peerCount,
                "Startup preconnect loaded %zu context agents, expected %zu", connections.size(), peerCount);

            uint64_t contextPairCount{0};
            for (size_t round = 0; round < peerCount; ++round)
            {
                // Match the CTX-primary rank order so this reciprocal pass
                // deterministically reuses the established sessions/windows.
                auto const peerRank = round;
                auto const& peerState = peers[peerRank].first;
                auto const& peerPool = peers[peerRank].second;
                auto const& peerCommState = peerState.getCommState().value();
                auto const counterparts
                    = targetIRanks(peerState.getCacheState().value(), mCacheState, mCommState.getSelfIdx()).mIRanks;
                if (std::find(counterparts.begin(), counterparts.end(), peerCommState.getSelfIdx())
                    != counterparts.end())
                {
                    auto const* connection = connections.at(static_cast<size_t>(peerCommState.getSelfIdx()));
                    auto const* agentConnection = dynamic_cast<AgentConnection const*>(connection);
                    TLLM_CHECK(agentConnection != nullptr);
                    agentConnection->preconnect(peerPool);
                    ++contextPairCount;
                }
                session.barrier();
            }
            localPairCount += contextPairCount;
            TLLM_LOG_INFO(mpi::MpiComm::world().getRank(),
                "MOONCAKE_PAGED_GIN %s preconnect context_pairs=%lu context_ranks=%zu role=GEN instance=%d "
                "epoch=%d local_rank=%d",
                startupEpoch == 0 ? "provisional" : "final-pool",
                static_cast<unsigned long>(contextPairCount), peerRankPayloads.size(), instanceId, startupEpoch,
                sessionRank);
        }

        uint64_t instancePairCount{0};
        session.allreduce(
            &localPairCount, &instancePairCount, 1, mpi::MpiType::kUINT64, mpi::MpiOp::SUM);
        auto const oneToOnePairCount
            = static_cast<uint64_t>(session.getSize()) * static_cast<uint64_t>(contextInstances);
        auto const allToAllPairCount = oneToOnePairCount * static_cast<uint64_t>(session.getSize());
        if (contextPrimaryPairCount.has_value())
        {
            TLLM_CHECK_WITH_INFO(instancePairCount == contextPrimaryPairCount.value(),
                "GEN reuse pair coverage %lu does not match CTX-primary coverage %lu",
                static_cast<unsigned long>(instancePairCount),
                static_cast<unsigned long>(contextPrimaryPairCount.value()));
        }
        TLLM_CHECK_WITH_INFO(
            instancePairCount == oneToOnePairCount || instancePairCount == allToAllPairCount,
            "Unsupported or incomplete GEN reuse pair coverage: got %lu, expected %lu or %lu",
            static_cast<unsigned long>(instancePairCount), static_cast<unsigned long>(oneToOnePairCount),
            static_cast<unsigned long>(allToAllPairCount));
        TLLM_LOG_INFO(mpi::MpiComm::world().getRank(),
            "MOONCAKE_PAGED_GIN %s preconnect local_pairs=%lu instance_pairs=%lu role=GEN instance=%d epoch=%d "
            "local_rank=%d",
            startupEpoch == 0 ? "provisional" : "final-pool", static_cast<unsigned long>(localPairCount),
            static_cast<unsigned long>(instancePairCount), instanceId, startupEpoch, sessionRank);
        session.barrier();
        if (sessionRank == 0)
        {
            auto const readyMessage = startupEpoch == 0
                ? "OK provisional pairs=" + std::to_string(instancePairCount) + "\n"
                : "OK pairs=" + std::to_string(instancePairCount) + "\n";
            writeStartupFile(readyPath, std::vector<char>(readyMessage.begin(), readyMessage.end()));
        }
        session.barrier();
        stopWatchdog();
    }
    catch (...)
    {
        writeStartupMarkerNoThrow(
            failedPath, startupEpoch == 0 ? "Provisional preconnect failed\n" : "Final-pool preconnect failed\n");
        stopWatchdog();
        throw;
    }

    TLLM_LOG_INFO(mpi::MpiComm::world().getRank(),
        "MOONCAKE_PAGED_GIN %s preconnect complete role=GEN instance=%d epoch=%d local_rank=%d",
        startupEpoch == 0 ? "provisional" : "final-pool", instanceId, startupEpoch, sessionRank);
}

AgentConnection const* AgentConnectionManager::recvConnectionAndRequestInfo(
    batch_manager::RequestInfo& requestInfo, std::atomic<bool> const& terminateFlag)
{
    while (!terminateFlag.load())
    {
        if (!mIsRunning)
        {
            return nullptr;
        }
        updateUnhandledNotifications();
        std::scoped_lock lock(mNotificationMutex);
        auto it = mUnhandledNotifications.begin();
        while (it != mUnhandledNotifications.end())
        {
            auto& [agent, notifs] = *it;
            auto notifIt = notifs.begin();
            while (notifIt != notifs.end())
            {
                std::stringstream ss(*notifIt);
                NotificationInfo notificationInfo = NotificationInfo::deserialize(ss);
                bool erase = false;
                if (std::holds_alternative<RequestAndBufferInfo>(notificationInfo.mInfo))
                {
                    auto requestAndBufferInfo = std::get<RequestAndBufferInfo>(notificationInfo.mInfo);

                    erase = true;
                    requestInfo = requestAndBufferInfo.mRequestInfo;
                    auto address = requestAndBufferInfo.mAddress;
                    auto bufferDescs = std::move(requestAndBufferInfo.mBufferDescs);
                    auto metadataOpt = requestAndBufferInfo.mMetadata;
                    auto connectionIdx = requestAndBufferInfo.mValidConnectionIdx;
                    auto remoteAgentName = requestAndBufferInfo.mAgentName;
                    auto pagedTransferMetadata = std::move(requestAndBufferInfo.mPagedTransferMetadata);
                    if (mooncakePagedGinDiagEnabled())
                    {
                        TLLM_LOG_INFO(mpi::MpiComm::world().getRank(),
                            "MOONCAKE_PAGED_GIN_DIAG manager_recv_request_info request_id=%lu remote=%s paged=%d buffers=%lu address=%s",
                            static_cast<unsigned long>(requestInfo.getRequestId()), remoteAgentName.c_str(),
                            pagedTransferMetadata.has_value() ? 1 : 0, bufferDescs.size(), address.c_str());
                    }
                    TLLM_LOG_DEBUG(" recv Address:%s", address.c_str());
                    auto connection = connect(remoteAgentName, address, metadataOpt, true);
                    auto bufferKinds = std::move(requestAndBufferInfo.mBufferKinds);

                    std::optional<std::pair<size_t, size_t>> kvOffsetRatio;
                    std::optional<std::pair<size_t, size_t>> rnnOffsetRatio;
                    std::vector<std::pair<size_t, size_t>> offsetRatios;
                    offsetRatios.reserve(bufferDescs.size());

                    for (size_t bi = 0; bi < bufferDescs.size(); bi++)
                    {
                        auto kind = static_cast<batch_manager::BufferKind>(bufferKinds[bi]);
                        switch (kind)
                        {
                        case batch_manager::BufferKind::kKV:
                        case batch_manager::BufferKind::kKV_INDEXER:
                        {
                            if (!kvOffsetRatio)
                            {
                                kvOffsetRatio
                                    = computeSendOffsetRatio(requestInfo.getTransState().getCacheState().value(),
                                        requestInfo.getTransState().getCommState()->getSelfIdx(), mCacheState,
                                        connectionIdx);
                            }
                            offsetRatios.push_back(*kvOffsetRatio);
                            break;
                        }
                        case batch_manager::BufferKind::kRNN:
                        {
                            if (!rnnOffsetRatio)
                            {
                                auto rnnTargetInfo = targetIRanksForRnn(mCacheState,
                                    requestInfo.getTransState().getCacheState().value(),
                                    requestInfo.getTransState().getCommState()->getSelfIdx());
                                size_t rnnOffsetLayer = 0;
                                for (int ri = 0; ri < connectionIdx; ri++)
                                {
                                    rnnOffsetLayer += rnnTargetInfo.getPeerPPDomainLayerNum(ri);
                                }
                                size_t rnnSendLayer = rnnTargetInfo.getPeerPPDomainLayerNum(connectionIdx);
                                rnnOffsetRatio = std::make_pair(rnnOffsetLayer, rnnSendLayer);
                            }
                            offsetRatios.push_back(*rnnOffsetRatio);
                            break;
                        }
                        }
                    }
                    connection->setSenderState(
                        std::move(bufferDescs), connectionIdx, std::move(offsetRatios), std::move(bufferKinds));
                    connection->setPagedTransferMetadata(std::move(pagedTransferMetadata));
                    notifIt = notifs.erase(notifIt);
                    if (notifs.empty())
                    {
                        it = mUnhandledNotifications.erase(it);
                    }
                    return connection;
                }

                if (!erase)
                {
                    notifIt++;
                }
            }
            if (notifs.empty())
            {
                it = mUnhandledNotifications.erase(it);
            }
            else
            {
                it++;
            }
        }
    }
    return nullptr;
}

void AgentConnectionManager::updateUnhandledNotifications()
{
    auto notifiedSyncMessages = m_Agent->getNotifiedSyncMessages();
    std::lock_guard<std::mutex> lock(mNotificationMutex);

    // Merge new notifications with existing ones
    for (auto const& [agent, notifs] : notifiedSyncMessages)
    {
        auto& existingNotifications = mUnhandledNotifications[agent];
        existingNotifications.insert(existingNotifications.end(), std::make_move_iterator(notifs.begin()),
            std::make_move_iterator(notifs.end()));
    }
}

[[nodiscard]] std::vector<Connection const*> AgentConnectionManager::getConnections(CommState const& state)
{
    TLLM_CHECK(state.isAgentState());
    auto ret = std::vector<Connection const*>();
    for (auto&& agentState : state.getAgentState())
    {
        std::string agentName = agentState.mAgentName;
        std::string connectionInfo = agentState.mConnectionInfo;
        ret.emplace_back(connect(agentName, connectionInfo));
    }
    return ret;
}

BaseTransferAgent* AgentConnectionManager::getAgent() const
{
    return m_Agent.get();
}

std::vector<batch_manager::BaseTransBufferManager*> const& AgentConnectionManager::getCacheTransBufferManagers() const
{
    return mCacheTransBufferManagers;
}

std::vector<uint8_t> const& AgentConnectionManager::getBufferKinds() const
{
    return mBufferKinds;
}

bool AgentConnectionManager::supportsPagedTransfer() const
{
    return mBackendType == "mooncake_paged_gin";
}

MemoryDesc const& AgentConnectionManager::getPagedPoolMemory() const
{
    TLLM_CHECK_WITH_INFO(mPagedPoolMemory.has_value(), "Paged KV-pool memory is unavailable");
    return mPagedPoolMemory.value();
}

void AgentConnectionManager::validateMemoryForPagedTransfer(MemoryDesc const& desc) const
{
    TLLM_CHECK_WITH_INFO(supportsPagedTransfer(),
        "validateMemoryForPagedTransfer is only supported by MOONCAKE_PAGED_GIN");
    auto const& pool = getPagedPoolMemory();
    TLLM_CHECK_WITH_INFO(desc.getAddr() == pool.getAddr() && desc.getLen() == pool.getLen()
            && desc.getDeviceId() == pool.getDeviceId(),
        "MOONCAKE_PAGED_GIN serving metadata does not match the startup-registered KV pool: "
        "actual addr=%p len=%lu device=%u expected addr=%p len=%lu device=%u",
        reinterpret_cast<void*>(desc.getAddr()), desc.getLen(), desc.getDeviceId(),
        reinterpret_cast<void*>(pool.getAddr()), pool.getLen(), pool.getDeviceId());
}

void AgentConnectionManager::registerMemoryForPagedTransfer(MemoryDesc const& desc)
{
    TLLM_CHECK_WITH_INFO(supportsPagedTransfer(),
        "registerMemoryForPagedTransfer is only supported by MOONCAKE_PAGED_GIN");
    TLLM_CHECK_WITH_INFO(desc.getAddr() != 0 && desc.getLen() != 0,
        "MOONCAKE_PAGED_GIN cannot register an empty KV memory descriptor");
    std::lock_guard<std::mutex> lock(mPagedRegMutex);
    if (!mPagedRegAddrs.insert(desc.getAddr()).second)
    {
        return;
    }
    MemoryDescs descs{MemoryType::kVRAM, {desc}};
    m_Agent->registerMemory(descs);
    mPagedRegMemDescs.push_back(desc);
}

AgentConnection* AgentConnectionManager::connect(std::string const& remoteAgentName, std::string const& connectionInfo,
    std::optional<std::string> metadata, bool isSender)
{

    TLLM_LOG_DEBUG(
        mpi::MpiComm::world().getRank(), "mAgentName: %s connect to %s", mAgentName.c_str(), remoteAgentName.c_str());
    std::scoped_lock lock(mConnectionsMutex);
    auto it = mConnections.find(remoteAgentName);
    if (it != mConnections.end())
    {
        if (isSender)
        {
            if (!it->second->hasLoadRemoteAgent())
            {
                TLLM_CHECK_WITH_INFO(metadata.has_value(), "should get metadata for sender loadRemtoeAgent");
            }
        }
        if (!it->second->hasLoadRemoteAgent() && metadata.has_value())
        {
            m_Agent->invalidateRemoteAgent(remoteAgentName);
            it->second->setHasLoadRemoteAgent(true);
            TLLM_LOG_DEBUG(mpi::MpiComm::world().getRank(), "set has load remote agent to true");
            m_Agent->loadRemoteAgent(remoteAgentName, AgentDesc{metadata.value()});
        }
        return it->second.get();
    }
    bool hasLoadRemoteAgent = false;
    if (remoteAgentName != mAgentName)
    {
        if (metadata.has_value())
        {
            TLLM_LOG_DEBUG(mpi::MpiComm::world().getRank(), "mAgentName: %s connect to %s with loadRemoteAgent",
                mAgentName.c_str(), remoteAgentName.c_str());
            m_Agent->loadRemoteAgent(remoteAgentName, AgentDesc{metadata.value()});
            hasLoadRemoteAgent = true;
        }
        else
        {
            TLLM_CHECK_WITH_INFO(!isSender, "Sender shouldn't call loadRemoteAgent");
            TLLM_LOG_DEBUG(mpi::MpiComm::world().getRank(), "mAgentName: %s connect to %s with loadRemoteAgent",
                mAgentName.c_str(), remoteAgentName.c_str());
            m_Agent->loadRemoteAgent(remoteAgentName, connectionInfo);
        }
    }
    else
    {
        hasLoadRemoteAgent = true;
    }

    auto connection = std::make_shared<AgentConnection>(mAgentName, remoteAgentName, this);
    mConnections[remoteAgentName] = connection;
    connection->setHasLoadRemoteAgent(hasLoadRemoteAgent);
    return connection.get();
}

CommState const& AgentConnectionManager::getCommState() const
{

    return mCommState;
}

AgentConnection* AgentConnectionManager::recvConnect(DataContext const& ctx, void* data, size_t size)
{

    TLLM_THROW("Not implemented");
    return nullptr;
}

int AgentConnectionManager::getDeviceId() const
{
    return mDeviceId;
}

template <typename NotificationType>
void AgentConnectionManager::waitForNotification(
    std::string const& remoteAgentName, NotificationType& expectedInfo, std::atomic<bool> const& terminateFlag)
{
    while (!terminateFlag.load())
    {

        if (!mIsRunning)
        {
            return;
        }
        updateUnhandledNotifications();
        std::scoped_lock lock(mNotificationMutex);
        auto it = mUnhandledNotifications.begin();
        while (it != mUnhandledNotifications.end())
        {
            auto& [agent, notifs] = *it;
            if (agent != remoteAgentName)
            {
                it++;
                continue;
            }
            auto notifIt = notifs.begin();
            while (notifIt != notifs.end())
            {
                std::stringstream ss(*notifIt);
                NotificationInfo notificationInfo = NotificationInfo::deserialize(ss);
                bool erase = false;
                if constexpr (std::is_same_v<NotificationType, NotificationSyncInfo>)
                {
                    if (std::holds_alternative<NotificationSyncInfo>(notificationInfo.mInfo))
                    {
                        auto notificationData = std::get<NotificationSyncInfo>(notificationInfo.mInfo);
                        if (notificationData.mContext.getTag() == expectedInfo.mContext.getTag()
                            && notificationData.mAgentName == expectedInfo.mAgentName)
                        {
                            erase = true;
                            notifIt = notifs.erase(notifIt);
                            if (notifs.empty())
                            {
                                it = mUnhandledNotifications.erase(it);
                            }
                            return;
                        }
                    }
                }
                else if constexpr (std::is_same_v<NotificationType, ReadySignalInfo>)
                {
                    if (std::holds_alternative<ReadySignalInfo>(notificationInfo.mInfo))
                    {
                        auto readySignalData = std::get<ReadySignalInfo>(notificationInfo.mInfo);
                        if (readySignalData.mContext.getTag() == expectedInfo.mContext.getTag()
                            && readySignalData.mAgentName == expectedInfo.mAgentName)
                        {
                            expectedInfo.mIsReady = readySignalData.mIsReady;

                            erase = true;
                            notifIt = notifs.erase(notifIt);
                            if (notifs.empty())
                            {
                                it = mUnhandledNotifications.erase(it);
                            }
                            return;
                        }
                    }
                }

                if (!erase)
                {
                    notifIt++;
                }
            }
            if (notifs.empty())
            {
                it = mUnhandledNotifications.erase(it);
            }
            else
            {
                it++;
            }
        }
    }
}

// Explicit template instantiations
template void AgentConnectionManager::waitForNotification<NotificationSyncInfo>(
    std::string const& remoteAgentName, NotificationSyncInfo& expectedInfo, std::atomic<bool> const& terminateFlag);
template void AgentConnectionManager::waitForNotification<ReadySignalInfo>(
    std::string const& remoteAgentName, ReadySignalInfo& expectedInfo, std::atomic<bool> const& terminateFlag);

void AgentConnectionManager::waitForSyncInfo(
    std::string const& remoteAgentName, NotificationSyncInfo& syncInfo, std::atomic<bool> const& terminateFlag)
{
    waitForNotification(remoteAgentName, syncInfo, terminateFlag);
}

void AgentConnectionManager::waitForReadySignal(
    std::string const& remoteAgentName, ReadySignalInfo& readySignalInfo, std::atomic<bool> const& terminateFlag)
{
    waitForNotification(remoteAgentName, readySignalInfo, terminateFlag);
}

std::string const& AgentConnectionManager::getAgentName() const
{
    return mAgentName;
}

AgentConnectionManager::~AgentConnectionManager()
{
    mIsRunning = false;
    if (!mPagedRegMemDescs.empty())
    {
        m_Agent->deregisterMemory(MemoryDescs{MemoryType::kVRAM, mPagedRegMemDescs});
    }
    m_Agent->deregisterMemory(mRegMemDescs);
}

bool AgentConnectionManager::isRunning() const
{
    return mIsRunning;
}

} // namespace tensorrt_llm::executor::kv_cache
