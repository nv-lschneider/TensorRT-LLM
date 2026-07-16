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

#include "tensorrt_llm/executor/cache_transmission/mooncake_paged_gin_utils/transferAgent.h"

#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/common/ipUtils.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/runtime/utils/mpiUtils.h"
#include "tent/common/types.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <unistd.h>
#include <utility>

namespace tensorrt_llm::executor::kv_cache
{
namespace
{

void setenvIfUnset(char const* key, char const* value)
{
    if (std::getenv(key) == nullptr)
    {
        int const rc = setenv(key, value, 0);
        TLLM_CHECK_WITH_INFO(rc == 0, "setenv(%s) failed", key);
    }
}

bool mooncakePagedGinDiagEnabled()
{
    return common::getBoolEnv("TRTLLM_MOONCAKE_PAGED_GIN_DIAG");
}

std::string jsonEscape(std::string const& value)
{
    std::ostringstream os;
    for (char c : value)
    {
        switch (c)
        {
        case '"': os << "\\\""; break;
        case '\\': os << "\\\\"; break;
        case '\b': os << "\\b"; break;
        case '\f': os << "\\f"; break;
        case '\n': os << "\\n"; break;
        case '\r': os << "\\r"; break;
        case '\t': os << "\\t"; break;
        default: os << c; break;
        }
    }
    return os.str();
}

std::string jsonScalar(std::string const& value)
{
    std::string lower = value;
    std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) { return std::tolower(c); });
    if (lower == "true" || lower == "false")
    {
        return lower;
    }
    if (!value.empty()
        && std::all_of(value.begin(), value.end(), [](unsigned char c) { return std::isdigit(c) || c == '-'; }))
    {
        return value;
    }
    return "\"" + jsonEscape(value) + "\"";
}

std::string writeTentConfig(BaseAgentConfig const& agentConfig)
{
    std::string rpcServerHostname;
    if (auto* explicitIp = std::getenv("TLLM_MOONCAKE_IP_ADDR"))
    {
        rpcServerHostname = explicitIp;
    }
    else
    {
        rpcServerHostname = common::getLocalIp(common::getEnvMooncakeInterface(), mpi::MpiComm::session().getRank());
    }

    std::ostringstream json;
    json << "{\n";
    json << "  \"metadata_type\": \"p2p\",\n";
    if (!rpcServerHostname.empty())
    {
        json << "  \"rpc_server_hostname\": \"" << jsonEscape(rpcServerHostname) << "\",\n";
    }
    json << "  \"log_level\": \"info\",\n";
    json << "  \"transports\": {\n";
    json << "    \"tcp\": {\"enable\": true},\n";
    json << "    \"rdma\": {\"enable\": false},\n";
    json << "    \"shm\": {\"enable\": false},\n";
    json << "    \"nvlink\": {\"enable\": false},\n";
    json << "    \"mnnvl\": {\"enable\": false},\n";
    json << "    \"gds\": {\"enable\": false},\n";
    json << "    \"io_uring\": {\"enable\": false},\n";
    json << "    \"nccl\": {\"enable\": true, \"allow_external_window_buffers\": true, "
            "\"force_gin\": true, \"wait_ack\": false}\n";
    json << "  }";
    for (auto const& [key, value] : agentConfig.backendParams)
    {
        json << ",\n  \"" << jsonEscape(key) << "\": " << jsonScalar(value);
    }
    json << "\n}\n";

    auto const path = "/tmp/trtllm_mooncake_paged_gin_" + std::to_string(getpid()) + "_" + agentConfig.mName + ".json";
    std::ofstream out(path);
    TLLM_CHECK_WITH_INFO(out.good(), "Failed to create TENT config file %s", path.c_str());
    out << json.str();
    out.close();
    TLLM_CHECK_WITH_INFO(out.good(), "Failed to write TENT config file %s", path.c_str());
    return path;
}

mooncake::tent::MemoryOptions makeNcclMemoryOptions()
{
    mooncake::tent::MemoryOptions options;
    options.location = mooncake::tent::kWildcardLocation;
    options.perm = mooncake::tent::kGlobalReadWrite;
    options.type = mooncake::tent::NCCL;
    options.internal = false;
    return options;
}

} // namespace

std::string const MooncakePagedGinBase64Helper::STANDARD_CHARS
    = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
      "abcdefghijklmnopqrstuvwxyz"
      "0123456789+/";

std::string MooncakePagedGinBase64Helper::encode(std::vector<uint8_t> const& data)
{
    return encodeInternal(data, STANDARD_CHARS);
}

std::string MooncakePagedGinBase64Helper::encode(std::string const& data)
{
    return encode(std::vector<uint8_t>(data.begin(), data.end()));
}

std::vector<uint8_t> MooncakePagedGinBase64Helper::decode(std::string const& encoded)
{
    return decodeInternal(encoded, STANDARD_CHARS);
}

std::string MooncakePagedGinBase64Helper::decodeToString(std::string const& encoded)
{
    auto vec = decode(encoded);
    return std::string(vec.begin(), vec.end());
}

std::string MooncakePagedGinBase64Helper::encodeInternal(std::vector<uint8_t> const& data, std::string const& chars)
{
    std::string encoded;
    size_t i = 0;
    size_t j = 0;
    std::array<uint8_t, 3> charArray3{};
    std::array<uint8_t, 4> charArray4{};
    size_t dataLen = data.size();
    uint8_t const* bytes = data.data();

    while (dataLen--)
    {
        charArray3[i++] = *(bytes++);
        if (i == 3)
        {
            charArray4[0] = (charArray3[0] & 0xfc) >> 2;
            charArray4[1] = ((charArray3[0] & 0x03) << 4) + ((charArray3[1] & 0xf0) >> 4);
            charArray4[2] = ((charArray3[1] & 0x0f) << 2) + ((charArray3[2] & 0xc0) >> 6);
            charArray4[3] = charArray3[2] & 0x3f;

            for (i = 0; i < 4; i++)
            {
                encoded += chars[charArray4[i]];
            }
            i = 0;
        }
    }

    if (i > 0)
    {
        for (j = i; j < 3; j++)
        {
            charArray3[j] = '\0';
        }

        charArray4[0] = (charArray3[0] & 0xfc) >> 2;
        charArray4[1] = ((charArray3[0] & 0x03) << 4) + ((charArray3[1] & 0xf0) >> 4);
        charArray4[2] = ((charArray3[1] & 0x0f) << 2) + ((charArray3[2] & 0xc0) >> 6);
        charArray4[3] = charArray3[2] & 0x3f;

        for (j = 0; j < i + 1; j++)
        {
            encoded += chars[charArray4[j]];
        }

        while (i++ < 3)
        {
            encoded += '=';
        }
    }

    return encoded;
}

std::vector<uint8_t> MooncakePagedGinBase64Helper::decodeInternal(
    std::string const& encoded, std::string const& chars)
{
    size_t encodedLen = encoded.size();
    size_t i = 0;
    size_t j = 0;
    size_t in_ = 0;
    std::array<uint8_t, 3> charArray3{};
    std::array<uint8_t, 4> charArray4{};
    std::vector<uint8_t> decoded;

    std::string cleanEncoded;
    for (char c : encoded)
    {
        if (!isWhitespace(c))
        {
            cleanEncoded += c;
        }
    }

    encodedLen = cleanEncoded.size();

    while (encodedLen-- && cleanEncoded[in_] != '=' && isBase64(cleanEncoded[in_], chars))
    {
        charArray4[i++] = cleanEncoded[in_];
        in_++;
        if (i == 4)
        {
            for (i = 0; i < 4; i++)
            {
                charArray4[i] = chars.find(charArray4[i]);
            }

            charArray3[0] = (charArray4[0] << 2) + ((charArray4[1] & 0x30) >> 4);
            charArray3[1] = ((charArray4[1] & 0xf) << 4) + ((charArray4[2] & 0x3c) >> 2);
            charArray3[2] = ((charArray4[2] & 0x3) << 6) + charArray4[3];

            for (i = 0; i < 3; i++)
            {
                decoded.push_back(charArray3[i]);
            }
            i = 0;
        }
    }

    if (i > 0)
    {
        for (j = i; j < 4; j++)
        {
            charArray4[j] = 0;
        }

        for (j = 0; j < 4; j++)
        {
            charArray4[j] = chars.find(charArray4[j]);
        }

        charArray3[0] = (charArray4[0] << 2) + ((charArray4[1] & 0x30) >> 4);
        charArray3[1] = ((charArray4[1] & 0xf) << 4) + ((charArray4[2] & 0x3c) >> 2);
        charArray3[2] = ((charArray4[2] & 0x3) << 6) + charArray4[3];

        for (j = 0; j < i - 1; j++)
        {
            decoded.push_back(charArray3[j]);
        }
    }

    return decoded;
}

bool MooncakePagedGinBase64Helper::isBase64(uint8_t c, std::string const& chars)
{
    return (std::isalnum(c) || (c == chars[62]) || (c == chars[63]));
}

bool MooncakePagedGinBase64Helper::isWhitespace(uint8_t c)
{
    return c == ' ' || c == '\n' || c == '\r' || c == '\t';
}

MooncakePagedGinTransferAgent::MooncakePagedGinTransferAgent(BaseAgentConfig const& config)
    : mLocalAgentName{config.mName}
{
    // Required by the paged GIN path when peers are otherwise local-shared-address candidates.
    setenvIfUnset("MC_NCCL_FORCE_GIN", "1");
    setenvIfUnset("NCCL_CUMEM_ENABLE", "1");

    if (std::getenv("MC_TENT_CONF") != nullptr)
    {
        TLLM_LOG_WARNING(
            "MOONCAKE_PAGED_GIN: MC_TENT_CONF is set and may override PoC NCCL/TCP transport defaults");
    }

    auto const configPath = writeTentConfig(config);
    mEngine = std::make_unique<mooncake::tent::TransferEngine>(configPath);
    std::remove(configPath.c_str());
    TLLM_CHECK_WITH_INFO(mEngine->available(), "MOONCAKE_PAGED_GIN failed to initialize TENT TransferEngine");
    TLLM_LOG_INFO("MOONCAKE_PAGED_GIN TENT segment: %s", mEngine->getSegmentName().c_str());
}

MooncakePagedGinTransferAgent::~MooncakePagedGinTransferAgent()
{
    TLLM_LOG_DEBUG("MooncakePagedGinTransferAgent::~MooncakePagedGinTransferAgent");
}

void MooncakePagedGinTransferAgent::checkStatus(mooncake::tent::Status const& status, char const* operation)
{
    TLLM_CHECK_WITH_INFO(status.ok(), "%s failed: %s", operation, status.ToString().c_str());
}

mooncake::tent::SegmentID MooncakePagedGinTransferAgent::getRemoteSegmentId(std::string const& name) const
{
    std::lock_guard<std::mutex> lock(mMutex);
    auto it = mConnectedAgents.find(name);
    TLLM_CHECK_WITH_INFO(it != mConnectedAgents.end(), "Remote agent %s not found", name.c_str());
    return it->second.segmentId;
}

void MooncakePagedGinTransferAgent::registerMemory(RegisterDescs const& descs)
{
    TLLM_LOG_DEBUG("MooncakePagedGinTransferAgent::registerMemory");
    TLLM_CHECK_WITH_INFO(descs.getType() == MemoryType::kVRAM, "MOONCAKE_PAGED_GIN only supports VRAM registration");

    std::lock_guard<std::mutex> lock(mMutex);
    auto options = makeNcclMemoryOptions();
    for (auto const& desc : descs.getDescs())
    {
        auto it = mMemRegInfo.find(desc.getAddr());
        if (it != mMemRegInfo.end())
        {
            it->second->addRef();
            continue;
        }

        auto status = mEngine->registerLocalMemory(reinterpret_cast<void*>(desc.getAddr()), desc.getLen(), options);
        checkStatus(status, "MOONCAKE_PAGED_GIN registerLocalMemory");

        auto mooncakeDesc = std::make_shared<MooncakePagedGinMemoryDesc>(desc);
        mooncakeDesc->addRef();
        mMemRegInfo[desc.getAddr()] = std::move(mooncakeDesc);
    }
}

void MooncakePagedGinTransferAgent::deregisterMemory(RegisterDescs const& descs)
{
    TLLM_LOG_DEBUG("MooncakePagedGinTransferAgent::deregisterMemory");

    std::lock_guard<std::mutex> lock(mMutex);
    for (auto const& desc : descs.getDescs())
    {
        auto it = mMemRegInfo.find(desc.getAddr());
        if (it == mMemRegInfo.end())
        {
            continue;
        }

        auto const& mooncakeDesc = it->second;
        if (mooncakeDesc->releaseRef() > 0)
        {
            continue;
        }

        auto status = mEngine->unregisterLocalMemory(
            reinterpret_cast<void*>(mooncakeDesc->getDesc().getAddr()), mooncakeDesc->getDesc().getLen());
        checkStatus(status, "MOONCAKE_PAGED_GIN unregisterLocalMemory");
        mMemRegInfo.erase(it);
    }
}

void MooncakePagedGinTransferAgent::loadRemoteAgent(std::string const& name, AgentDesc const& agentDesc)
{
    loadRemoteAgent(name, agentDesc.getBackendAgentDesc());
}

void MooncakePagedGinTransferAgent::loadRemoteAgent(std::string const& name, ConnectionInfoType const& connectionInfo)
{
    TLLM_LOG_DEBUG("MooncakePagedGinTransferAgent::loadRemoteAgent remote=%s connection=%s", name.c_str(),
        connectionInfo.c_str());

    mooncake::tent::SegmentID segmentId{};
    auto status = mEngine->openSegment(segmentId, connectionInfo);
    checkStatus(status, "MOONCAKE_PAGED_GIN openSegment");

    std::lock_guard<std::mutex> lock(mMutex);
    mConnectedAgents[name] = AgentInfo{segmentId, false};
}

void MooncakePagedGinTransferAgent::preconnectRemoteAgent(std::string const& name)
{
    mooncake::tent::SegmentID segmentId{};
    {
        std::lock_guard<std::mutex> lock(mMutex);
        auto const it = mConnectedAgents.find(name);
        TLLM_CHECK_WITH_INFO(it != mConnectedAgents.end(), "Remote agent %s not found", name.c_str());
        if (it->second.preconnected)
        {
            return;
        }
        segmentId = it->second.segmentId;
    }

    TLLM_LOG_INFO("MOONCAKE_PAGED_GIN eager preconnect begin: local=%s remote=%s segment=%lu",
        mLocalAgentName.c_str(), name.c_str(), static_cast<unsigned long>(segmentId));
    auto const status = mEngine->preconnectSegment(segmentId);
    checkStatus(status, "MOONCAKE_PAGED_GIN preconnectSegment");

    {
        std::lock_guard<std::mutex> lock(mMutex);
        auto const it = mConnectedAgents.find(name);
        TLLM_CHECK_WITH_INFO(it != mConnectedAgents.end() && it->second.segmentId == segmentId,
            "Remote agent %s changed during preconnect", name.c_str());
        it->second.preconnected = true;
    }
    TLLM_LOG_INFO("MOONCAKE_PAGED_GIN eager preconnect ready: local=%s remote=%s segment=%lu",
        mLocalAgentName.c_str(), name.c_str(), static_cast<unsigned long>(segmentId));
}

void MooncakePagedGinTransferAgent::invalidateRemoteAgent(std::string const& name)
{
    std::lock_guard<std::mutex> lock(mMutex);
    auto it = mConnectedAgents.find(name);
    if (it == mConnectedAgents.end())
    {
        return;
    }
    auto status = mEngine->closeSegment(it->second.segmentId);
    if (!status.ok())
    {
        TLLM_LOG_WARNING("MOONCAKE_PAGED_GIN closeSegment failed for %s: %s", name.c_str(), status.ToString().c_str());
    }
    mConnectedAgents.erase(it);
}

AgentDesc MooncakePagedGinTransferAgent::getLocalAgentDesc()
{
    return AgentDesc{mEngine->getSegmentName()};
}

ConnectionInfoType MooncakePagedGinTransferAgent::getLocalConnectionInfo()
{
    return mEngine->getSegmentName();
}

std::unique_ptr<TransferStatus> MooncakePagedGinTransferAgent::submitTransferRequests(TransferRequest const& request)
{
    (void) request;
    TLLM_THROW("MOONCAKE_PAGED_GIN does not support contiguous transfer requests");
}

void MooncakePagedGinTransferAgent::submitPagedTransferRequest(PagedTransferRequest const& request)
{
    if (mooncakePagedGinDiagEnabled())
    {
        TLLM_LOG_INFO("MOONCAKE_PAGED_GIN submitPagedTransferRequest: remote=%s layers=%lu pages=%lu page_bytes=%lu",
            request.getRemoteName().c_str(), request.getSrcLayerPtrs().size(), request.getSrcPageIndices().size(),
            request.getPageBytes());
    }
    mooncake::tent::PagedTransferRequest tentRequest;
    tentRequest.target_id = getRemoteSegmentId(request.getRemoteName());
    tentRequest.src_layer_ptrs.reserve(request.getSrcLayerPtrs().size());
    for (auto ptr : request.getSrcLayerPtrs())
    {
        tentRequest.src_layer_ptrs.push_back(reinterpret_cast<void*>(ptr));
    }
    tentRequest.dst_layer_ptrs.reserve(request.getDstLayerPtrs().size());
    for (auto ptr : request.getDstLayerPtrs())
    {
        tentRequest.dst_layer_ptrs.push_back(static_cast<uint64_t>(ptr));
    }
    tentRequest.src_page_indices = request.getSrcPageIndices();
    tentRequest.dst_page_indices = request.getDstPageIndices();
    tentRequest.page_bytes = request.getPageBytes();

    if (mooncakePagedGinDiagEnabled())
    {
        TLLM_LOG_INFO("MOONCAKE_PAGED_GIN transferPagedSync begin: target_id=%lu src0=%p dst0=%p src_page0=%d dst_page0=%d",
            static_cast<unsigned long>(tentRequest.target_id),
            tentRequest.src_layer_ptrs.empty() ? nullptr : tentRequest.src_layer_ptrs.front(),
            reinterpret_cast<void*>(tentRequest.dst_layer_ptrs.empty() ? 0 : tentRequest.dst_layer_ptrs.front()),
            tentRequest.src_page_indices.empty() ? -1 : tentRequest.src_page_indices.front(),
            tentRequest.dst_page_indices.empty() ? -1 : tentRequest.dst_page_indices.front());
    }
    auto status = mEngine->transferPagedSync(tentRequest);
    if (mooncakePagedGinDiagEnabled())
    {
        TLLM_LOG_INFO("MOONCAKE_PAGED_GIN transferPagedSync end: %s", status.ToString().c_str());
    }
    checkStatus(status, "MOONCAKE_PAGED_GIN transferPagedSync");
}

void MooncakePagedGinTransferAgent::notifySyncMessage(std::string const& name, SyncMessage const& syncMessage)
{
    mooncake::tent::Notification notification;
    notification.name = mLocalAgentName;
    notification.msg = MooncakePagedGinBase64Helper::encode(syncMessage);

    if (mooncakePagedGinDiagEnabled())
    {
        TLLM_LOG_INFO("MOONCAKE_PAGED_GIN_DIAG notify_begin remote=%s bytes=%lu", name.c_str(), notification.msg.size());
    }
    auto const targetId = getRemoteSegmentId(name);
    if (mooncakePagedGinDiagEnabled())
    {
        TLLM_LOG_INFO("MOONCAKE_PAGED_GIN_DIAG notify_target remote=%s target_id=%lu", name.c_str(),
            static_cast<unsigned long>(targetId));
    }
    auto status = mEngine->sendNotification(targetId, notification);
    if (mooncakePagedGinDiagEnabled())
    {
        TLLM_LOG_INFO("MOONCAKE_PAGED_GIN_DIAG notify_end remote=%s status=%s", name.c_str(), status.ToString().c_str());
    }
    checkStatus(status, "MOONCAKE_PAGED_GIN sendNotification");
}

std::unordered_map<std::string, std::vector<SyncMessage>> MooncakePagedGinTransferAgent::getNotifiedSyncMessages()
{
    std::vector<mooncake::tent::Notification> notifications;
    auto status = mEngine->receiveNotification(notifications);
    checkStatus(status, "MOONCAKE_PAGED_GIN receiveNotification");

    if (!notifications.empty())
    {
        if (mooncakePagedGinDiagEnabled())
        {
            TLLM_LOG_INFO("MOONCAKE_PAGED_GIN_DIAG receive_notifications count=%lu", notifications.size());
        }
    }
    std::unordered_map<std::string, std::vector<SyncMessage>> notifs;
    for (auto const& notification : notifications)
    {
        if (mooncakePagedGinDiagEnabled())
        {
            TLLM_LOG_INFO("MOONCAKE_PAGED_GIN_DIAG receive_notification from=%s bytes=%lu",
                notification.name.c_str(), notification.msg.size());
        }
        notifs[notification.name].emplace_back(MooncakePagedGinBase64Helper::decodeToString(notification.msg));
    }
    return notifs;
}

bool MooncakePagedGinTransferAgent::checkRemoteDescs(std::string const& name, MemoryDescs const& memoryDescs)
{
    (void) name;
    (void) memoryDescs;
    return true;
}

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreturn-type-c-linkage"
#endif

extern "C"
{
    std::unique_ptr<BaseTransferAgent> createMooncakePagedGinTransferAgent(BaseAgentConfig const* config)
    {
        TLLM_CHECK(config);
        return std::make_unique<MooncakePagedGinTransferAgent>(*config);
    }
}

#if defined(__clang__)
#pragma clang diagnostic pop
#endif

} // namespace tensorrt_llm::executor::kv_cache
