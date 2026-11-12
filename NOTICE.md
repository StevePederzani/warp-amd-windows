# NOTICE

This repository is a fork of NVIDIA Warp with a HIP backend for AMD GPUs on Windows.

## NVIDIA Warp

Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

Licensed under the Apache License, Version 2.0 (see LICENSE.md). This port is built on tag v1.19.0 of https://github.com/NVIDIA/warp.
The third-party components that Warp bundles are listed with their licenses in `licenses/`.

## ROCm Warp (AMD)

The HIP backend started from ROCm Warp, AMD's port of Warp to AMD Instinct GPUs (https://github.com/ROCm/warp, branch
`amd-integration`, commit 36ed75aa6), licensed under the Apache License, Version 2.0. Its HIP utility layer (`warp/native/hip_util.h`
and much of `warp/native/cuda_util.cpp`) and parts of the build and runtime code are AMD's work, carried here onto Warp 1.19 and
Windows. Files that contain AMD's lines say so in their headers.

## This Port

Copyright (c) 2026 Steve Pederzani. Licensed under the Apache License, Version 2.0.

Every file this port changed carries a line saying so under its license header, and files this port added carry their own copyright
header.
