// SPDX-FileCopyrightText: Copyright (c) 2026 Steve Pederzani
// SPDX-License-Identifier: Apache-2.0

#pragma once

// CUB device algorithms for the .cu sources, which use NVIDIA CUB on CUDA and hipCUB (rocPRIM) on HIP, both under cub::.
// On HIP the CUDA runtime and driver headers come from hip_util.h, through cuda_util.h.

#if defined(__HIP_PLATFORM_AMD__)
#include "hip_util.h"
#include <hipcub/hipcub.hpp>
namespace cub = hipcub;
// cuBQL's builder refers to hipCUB as cub unless that alias already exists.
#define WP_CUB_IS_HIPCUB 1
#else
#include <cub/cub.cuh>
#include <cuda.h>
#include <cuda_runtime_api.h>
#endif
