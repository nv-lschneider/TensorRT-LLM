/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "tensorrt_llm/executor/cache_transmission/agent_utils/connection.h"

#include <gtest/gtest.h>

#include <sstream>

namespace texec = tensorrt_llm::executor;

TEST(PagedTransferMetadataTest, SerializationRoundTrip)
{
    texec::kv_cache::PagedTransferMetadata const metadata{{0x1000, 0x2000}, {7, 3, 11}, 1124352,
        0x123456789abcdef0ULL, texec::kv_cache::MemoryDesc{0x1000, 19113984, 2}};

    std::stringstream stream;
    texec::kv_cache::RequestAndBufferInfo::serializePagedTransferMetadata(metadata, stream);
    EXPECT_EQ(stream.str().size(), texec::kv_cache::RequestAndBufferInfo::serializedSizePagedTransferMetadata(metadata));

    auto const restored = texec::kv_cache::RequestAndBufferInfo::deserializePagedTransferMetadata(stream);
    EXPECT_EQ(restored.mLayerPtrs, metadata.mLayerPtrs);
    EXPECT_EQ(restored.mPageIndices, metadata.mPageIndices);
    EXPECT_EQ(restored.mPageBytes, metadata.mPageBytes);
    EXPECT_EQ(restored.mLayoutFingerprint, metadata.mLayoutFingerprint);
    EXPECT_EQ(restored.mRegisteredMemory.getAddr(), metadata.mRegisteredMemory.getAddr());
    EXPECT_EQ(restored.mRegisteredMemory.getLen(), metadata.mRegisteredMemory.getLen());
    EXPECT_EQ(restored.mRegisteredMemory.getDeviceId(), metadata.mRegisteredMemory.getDeviceId());
}
