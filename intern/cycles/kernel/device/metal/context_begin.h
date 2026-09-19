/* SPDX-FileCopyrightText: 2021-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

// clang-format off

/* GPU warp (SIMD group) collectives shared with the CUDA backend, which implements these in
 * kernel/device/cuda/compat.h. */
#define ccl_gpu_simd_sum(value) metal::simd_sum(value)
#define ccl_gpu_simd_max(value) metal::simd_max(value)
#define ccl_gpu_simd_min(value) metal::simd_min(value)
#define ccl_gpu_simd_broadcast(value, lane) metal::simd_broadcast(value, lane)
#define ccl_gpu_simd_shuffle_xor(value, mask) metal::simd_shuffle_xor(value, mask)
#define ccl_gpu_simd_prefix_exclusive_sum(value) metal::simd_prefix_exclusive_sum(value)
#define ccl_gpu_simd_group_barrier() metal::simdgroup_barrier(metal::mem_flags::mem_device)

#ifdef WITH_NANOVDB
#  include "kernel/util/nanovdb.h"
#endif

/* Open the Metal kernel context class
 * Necessary to access resource bindings */
class MetalKernelContext {
  public:
    constant KernelParamsMetal &launch_params_metal;
    constant MetalAncillaries *metal_ancillaries;

    MetalKernelContext(constant KernelParamsMetal &_launch_params_metal, constant MetalAncillaries * _metal_ancillaries)
    : launch_params_metal(_launch_params_metal), metal_ancillaries(_metal_ancillaries)
    {}

    MetalKernelContext(constant KernelParamsMetal &_launch_params_metal)
    : launch_params_metal(_launch_params_metal)
    {}

    /* texture fetch adapter functions */
    using ccl_gpu_image_object_2D = uint64_t;

    template<typename T>
    inline __attribute__((__always_inline__))
    T ccl_gpu_image_object_read_2D(ccl_gpu_image_object_2D tex, const float x, float y) const {
      kernel_assert(0);
      return 0;
    }

    // texture2d
    template<>
    inline __attribute__((__always_inline__))
    float4 ccl_gpu_image_object_read_2D(ccl_gpu_image_object_2D tex, const float x, float y) const {
      const uint tid(tex);
      const uint sid(tex >> 32);
      return ((ccl_global Texture2DParamsMetal*)metal_ancillaries->textures)[tid].tex.sample(metal_samplers[sid], float2(x, y));
    }
    template<>
    inline __attribute__((__always_inline__))
    float ccl_gpu_image_object_read_2D(ccl_gpu_image_object_2D tex, const float x, float y) const {
      const uint tid(tex);
      const uint sid(tex >> 32);
      return ((ccl_global Texture2DParamsMetal*)metal_ancillaries->textures)[tid].tex.sample(metal_samplers[sid], float2(x, y)).x;
    }

#    include "kernel/device/gpu/image.h"

  // clang-format on
