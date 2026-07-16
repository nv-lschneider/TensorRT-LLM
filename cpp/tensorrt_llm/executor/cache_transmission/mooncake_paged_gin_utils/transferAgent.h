/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/executor/transferAgent.h"
#include "tent/transfer_engine.h"

#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>

namespace tensorrt_llm::executor::kv_cache
{

class MooncakePagedGinMemoryDesc
{
public:
    explicit MooncakePagedGinMemoryDesc(MemoryDesc desc)
        : mDesc{std::move(desc)}
    {
    }

    void addRef() noexcept
    {
        ++mRefCnt;
    }

    int releaseRef() noexcept
    {
        return --mRefCnt;
    }

    [[nodiscard]] int getRefCount() const noexcept
    {
        return mRefCnt;
    }

    [[nodiscard]] MemoryDesc const& getDesc() const noexcept
    {
        return mDesc;
    }

private:
    MemoryDesc mDesc;
    int mRefCnt{0};
};

class MooncakePagedGinBase64Helper
{
public:
    static std::string encode(std::vector<uint8_t> const& data);
    static std::string encode(std::string const& data);
    static std::vector<uint8_t> decode(std::string const& encoded);
    static std::string decodeToString(std::string const& encoded);

private:
    static std::string const STANDARD_CHARS;

    static std::string encodeInternal(std::vector<uint8_t> const& data, std::string const& chars);
    static std::vector<uint8_t> decodeInternal(std::string const& encoded, std::string const& chars);
    static inline bool isBase64(uint8_t c, std::string const& chars);
    static inline bool isWhitespace(uint8_t c);
};

class MooncakePagedGinTransferAgent final : public BaseTransferAgent
{
public:
    explicit MooncakePagedGinTransferAgent(BaseAgentConfig const& config);
    ~MooncakePagedGinTransferAgent() override;

    void registerMemory(RegisterDescs const& descs) override;
    void deregisterMemory(RegisterDescs const& descs) override;

    void loadRemoteAgent(std::string const& name, AgentDesc const& agentDesc) override;
    void loadRemoteAgent(std::string const& name, ConnectionInfoType const& connectionInfo) override;
    void preconnectRemoteAgent(std::string const& name) override;
    void invalidateRemoteAgent(std::string const& name) override;

    AgentDesc getLocalAgentDesc() override;
    ConnectionInfoType getLocalConnectionInfo() override;

    [[nodiscard]] std::unique_ptr<TransferStatus> submitTransferRequests(TransferRequest const& request) override;
    void submitPagedTransferRequest(PagedTransferRequest const& request) override;

    void notifySyncMessage(std::string const& name, SyncMessage const& syncMessage) override;
    [[nodiscard]] std::unordered_map<std::string, std::vector<SyncMessage>> getNotifiedSyncMessages() override;

    bool checkRemoteDescs(std::string const& name, MemoryDescs const& memoryDescs) override;

private:
    struct AgentInfo
    {
        mooncake::tent::SegmentID segmentId;
        bool preconnected{false};
    };

    [[nodiscard]] mooncake::tent::SegmentID getRemoteSegmentId(std::string const& name) const;
    static void checkStatus(mooncake::tent::Status const& status, char const* operation);

    mutable std::mutex mMutex;
    std::unique_ptr<mooncake::tent::TransferEngine> mEngine;
    std::unordered_map<uintptr_t, std::shared_ptr<MooncakePagedGinMemoryDesc>> mMemRegInfo;
    std::unordered_map<std::string, AgentInfo> mConnectedAgents;
    std::string mLocalAgentName;
};

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreturn-type-c-linkage"
#endif

extern "C"
{
    [[nodiscard]] std::unique_ptr<BaseTransferAgent> createMooncakePagedGinTransferAgent(BaseAgentConfig const* config);
}

#if defined(__clang__)
#pragma clang diagnostic pop
#endif

} // namespace tensorrt_llm::executor::kv_cache
