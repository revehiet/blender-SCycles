/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#define __KERNEL_GPU__
#define __KERNEL_CUDA__
#define CCL_NAMESPACE_BEGIN
#define CCL_NAMESPACE_END

#ifndef ATTR_FALLTHROUGH
#  define ATTR_FALLTHROUGH
#endif

/* Manual definitions so we can compile without CUDA toolkit. */

#ifdef __CUDACC_RTC__
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;
#else
#  include <stdint.h>
#endif

#ifdef CYCLES_CUBIN_CC
#  define FLT_MIN 1.175494350822287507969e-38f
#  define FLT_MAX 340282346638528859811704183484516925440.0f
#  define FLT_EPSILON 1.192092896e-07F
#endif

/* Qualifiers */

#define ccl_device __device__ __inline__
#define ccl_device_extern extern "C" __device__
#define ccl_device_inline __device__ __inline__
#define ccl_device_forceinline __device__ __forceinline__
#define ccl_device_noinline __device__ __noinline__
#define ccl_device_noinline_cpu ccl_device
#define ccl_device_inline_method ccl_device
#define ccl_device_template_spec template<> ccl_device_inline
#define ccl_global
#define ccl_inline_constant __constant__
#define ccl_device_constant __constant__ __device__
#define ccl_static_constexpr static constexpr
#define ccl_constant const
#define ccl_gpu_shared __shared__
#define ccl_private
#define ccl_ray_data ccl_private
#define ccl_may_alias
#define ccl_restrict __restrict__
#define ccl_align(n) __align__(n)
#define ccl_optional_struct_init
#define ccl_attr_maybe_unused [[maybe_unused]]

/* No assert supported for CUDA */

#define kernel_assert(cond)

/* GPU thread, block, grid size and index */

#define ccl_gpu_thread_idx_x (threadIdx.x)
#define ccl_gpu_block_dim_x (blockDim.x)
#define ccl_gpu_block_idx_x (blockIdx.x)
#define ccl_gpu_grid_dim_x (gridDim.x)
#define ccl_gpu_warp_size (warpSize)
#define ccl_gpu_thread_mask(thread_warp) uint(0xFFFFFFFF >> (ccl_gpu_warp_size - thread_warp))

#define ccl_gpu_global_id_x() (ccl_gpu_block_idx_x * ccl_gpu_block_dim_x + ccl_gpu_thread_idx_x)
#define ccl_gpu_global_size_x() (ccl_gpu_grid_dim_x * ccl_gpu_block_dim_x)

/* GPU warp synchronization. */

#define ccl_gpu_syncthreads() __syncthreads()
#define ccl_gpu_ballot(predicate) __ballot_sync(0xFFFFFFFF, predicate)

/* GPU warp (SIMD group) collectives, mirroring the Metal SIMD group functions used by the GPU
 * path guiding. Note that the kernel types are not available yet at this point, so the builtin
 * types are used. The active mask is queried so that the callers which validate the group with
 * `ccl_gpu_simd_sum(1u)` observe the number of active lanes, as they do on Metal. */

ccl_device_forceinline unsigned int ccl_gpu_simd_sum(const unsigned int value)
{
  const unsigned int mask = __activemask();
  unsigned int sum = value;
  sum += __shfl_xor_sync(mask, sum, 16);
  sum += __shfl_xor_sync(mask, sum, 8);
  sum += __shfl_xor_sync(mask, sum, 4);
  sum += __shfl_xor_sync(mask, sum, 2);
  sum += __shfl_xor_sync(mask, sum, 1);
  return sum;
}

ccl_device_forceinline float ccl_gpu_simd_sum(const float value)
{
  const unsigned int mask = __activemask();
  float sum = value;
  sum += __shfl_xor_sync(mask, sum, 16);
  sum += __shfl_xor_sync(mask, sum, 8);
  sum += __shfl_xor_sync(mask, sum, 4);
  sum += __shfl_xor_sync(mask, sum, 2);
  sum += __shfl_xor_sync(mask, sum, 1);
  return sum;
}

ccl_device_forceinline float ccl_gpu_simd_max(const float value)
{
  const unsigned int mask = __activemask();
  float maximum = value;
  maximum = fmaxf(maximum, __shfl_xor_sync(mask, maximum, 16));
  maximum = fmaxf(maximum, __shfl_xor_sync(mask, maximum, 8));
  maximum = fmaxf(maximum, __shfl_xor_sync(mask, maximum, 4));
  maximum = fmaxf(maximum, __shfl_xor_sync(mask, maximum, 2));
  maximum = fmaxf(maximum, __shfl_xor_sync(mask, maximum, 1));
  return maximum;
}

ccl_device_forceinline unsigned int ccl_gpu_simd_min(const unsigned int value)
{
  const unsigned int mask = __activemask();
  unsigned int minimum = value;
  minimum = min(minimum, __shfl_xor_sync(mask, minimum, 16));
  minimum = min(minimum, __shfl_xor_sync(mask, minimum, 8));
  minimum = min(minimum, __shfl_xor_sync(mask, minimum, 4));
  minimum = min(minimum, __shfl_xor_sync(mask, minimum, 2));
  minimum = min(minimum, __shfl_xor_sync(mask, minimum, 1));
  return minimum;
}

ccl_device_forceinline unsigned int ccl_gpu_simd_broadcast(const unsigned int value,
                                                           const unsigned int lane)
{
  return __shfl_sync(__activemask(), value, lane);
}

ccl_device_forceinline float ccl_gpu_simd_shuffle_xor(const float value,
                                                      const unsigned int mask_lane)
{
  return __shfl_xor_sync(__activemask(), value, mask_lane);
}

ccl_device_forceinline unsigned int ccl_gpu_simd_prefix_exclusive_sum(const unsigned int value)
{
  const unsigned int mask = __activemask();
  unsigned int inclusive = value;
  inclusive += __shfl_up_sync(mask, inclusive, 16);
  inclusive += __shfl_up_sync(mask, inclusive, 8);
  inclusive += __shfl_up_sync(mask, inclusive, 4);
  inclusive += __shfl_up_sync(mask, inclusive, 2);
  inclusive += __shfl_up_sync(mask, inclusive, 1);
  return inclusive - value;
}

ccl_device_forceinline void ccl_gpu_simd_group_barrier()
{
  __syncwarp();
}

/* GPU texture objects */

typedef unsigned long long CUtexObject;
typedef CUtexObject ccl_gpu_image_object_2D;

template<typename T>
ccl_device_forceinline T ccl_gpu_image_object_read_2D(const ccl_gpu_image_object_2D texobj,
                                                      const float x,
                                                      const float y)
{
  return tex2D<T>(texobj, x, y);
}

/* Use fast math functions */

#define cosf(x) __cosf(((float)(x)))
#define sinf(x) __sinf(((float)(x)))
#define powf(x, y) __powf(((float)(x)), ((float)(y)))
#define tanf(x) __tanf(((float)(x)))
#define logf(x) __logf(((float)(x)))
#define expf(x) __expf(((float)(x)))

/* Half */

typedef unsigned short half;

ccl_device_forceinline half __float2half(const float f)
{
  half val;
  asm("{  cvt.rn.f16.f32 %0, %1;}\n" : "=h"(val) : "f"(f));
  return val;
}

ccl_device_forceinline float __half2float(const half h)
{
  float val;
  asm("{  cvt.f32.f16 %0, %1;}\n" : "=f"(val) : "h"(h));
  return val;
}

/* Types */

#include "util/half.h"
#include "util/types.h"
