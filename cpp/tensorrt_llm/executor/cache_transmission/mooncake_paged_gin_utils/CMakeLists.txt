# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION &
# AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0

set(MOONCAKE_TENT_ROOT
    ""
    CACHE PATH "Path to an installed Mooncake TENT tree")

set(TENT_SEARCH_ROOTS)
if(MOONCAKE_TENT_ROOT)
  list(APPEND TENT_SEARCH_ROOTS ${MOONCAKE_TENT_ROOT})
endif()
if(MOONCAKE_ROOT)
  list(APPEND TENT_SEARCH_ROOTS ${MOONCAKE_ROOT})
endif()

find_library(
  TENT_ENGINE_LIB
  NAMES tent_shared
  HINTS ${TENT_SEARCH_ROOTS}
  PATH_SUFFIXES lib build/tent/src build/mooncake-transfer-engine/tent/src)
find_path(
  TENT_INCLUDE_DIR
  NAMES tent/transfer_engine.h
  HINTS ${TENT_SEARCH_ROOTS}
  PATH_SUFFIXES include mooncake-transfer-engine/tent/include)

message(STATUS "Find TENT results:")
message(STATUS "  TENT_ENGINE_LIB = ${TENT_ENGINE_LIB}")
message(STATUS "  TENT_INCLUDE_DIR = ${TENT_INCLUDE_DIR}")

if(TENT_ENGINE_LIB AND TENT_INCLUDE_DIR)
  set(MOONCAKE_PAGED_GIN_WRAPPER_TARGET
      "tensorrt_llm_mooncake_paged_gin_wrapper")

  add_library(${MOONCAKE_PAGED_GIN_WRAPPER_TARGET} SHARED transferAgent.cpp)
  target_compile_options(${MOONCAKE_PAGED_GIN_WRAPPER_TARGET} PRIVATE
                         -Wno-error)
  set_target_properties(
    ${MOONCAKE_PAGED_GIN_WRAPPER_TARGET}
    PROPERTIES CXX_STANDARD "20" CXX_STANDARD_REQUIRED "YES"
               CXX_EXTENSIONS "NO")

  target_include_directories(${MOONCAKE_PAGED_GIN_WRAPPER_TARGET}
                             PRIVATE ${TENT_INCLUDE_DIR})
  target_link_libraries(${MOONCAKE_PAGED_GIN_WRAPPER_TARGET}
                        PRIVATE ${TENT_ENGINE_LIB} CUDA::cudart)

  set(TENT_INCLUDE_DIR
      ${TENT_INCLUDE_DIR}
      PARENT_SCOPE)
endif()
