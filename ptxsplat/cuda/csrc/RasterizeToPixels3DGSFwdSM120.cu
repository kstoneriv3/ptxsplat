#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAStream.h>

#include "Common.h"
#include "Rasterization.h"

namespace ptxsplat {

struct alignas(16) Sm120ForwardGaussianStage {
    float4 xy_opacity_conic_x;
    float4 conic_yz_id_padding;
};
static_assert(sizeof(Sm120ForwardGaussianStage) == 2 * sizeof(float4));

template <
    uint32_t WARP_PIXEL_WIDTH,
    bool OUTPUT_TRANSPOSE = false,
    bool SKIP_FIRST_PREWRITE_BARRIER = false,
    bool REUSE_GAUSSIAN_STAGE_FOR_OUTPUT = false>
__global__ void rasterize_to_pixels_3dgs_fwd_sm120_soa_kernel(
    const uint32_t I,
    const uint32_t n_isects,
    const vec2 *__restrict__ means2d,
    const vec3 *__restrict__ conics,
    const float *__restrict__ colors,
    const float *__restrict__ opacities,
    const float *__restrict__ backgrounds,
    const bool *__restrict__ masks,
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const int32_t *__restrict__ tile_offsets,
    const int32_t *__restrict__ flatten_ids,
    float *__restrict__ render_colors,
    float *__restrict__ render_alphas,
    int32_t *__restrict__ last_ids
) {
    constexpr uint32_t CDIM = 3;
    constexpr uint32_t TILE_SIZE = 16;
    constexpr uint32_t BLOCK_SIZE = TILE_SIZE * TILE_SIZE;
    constexpr uint32_t BATCH_CAPACITY = 384;
    constexpr uint32_t WARP_SIZE = 32;
    constexpr uint32_t WARP_PIXEL_HEIGHT = WARP_SIZE / WARP_PIXEL_WIDTH;
    constexpr uint32_t WARP_RECTS_ACROSS = TILE_SIZE / WARP_PIXEL_WIDTH;
    static_assert(
        WARP_PIXEL_WIDTH == 16 || WARP_PIXEL_WIDTH == 8 ||
        WARP_PIXEL_WIDTH == 4
    );

    const uint32_t image_id = blockIdx.x;
    const uint32_t tile_id = blockIdx.y * tile_width + blockIdx.z;
    const uint32_t tr = threadIdx.y * TILE_SIZE + threadIdx.x;
    const uint32_t warp = tr / WARP_SIZE;
    const uint32_t lane = tr % WARP_SIZE;
    const uint32_t lane_pixel_x = lane % WARP_PIXEL_WIDTH;
    const uint32_t lane_pixel_y = lane / WARP_PIXEL_WIDTH;
    const uint32_t pixel_x =
        (warp % WARP_RECTS_ACROSS) * WARP_PIXEL_WIDTH + lane_pixel_x;
    const uint32_t pixel_y =
        (warp / WARP_RECTS_ACROSS) * WARP_PIXEL_HEIGHT + lane_pixel_y;
    const uint32_t i = blockIdx.y * TILE_SIZE + pixel_y;
    const uint32_t j = blockIdx.z * TILE_SIZE + pixel_x;

    tile_offsets += image_id * tile_height * tile_width;
    render_colors += image_id * image_height * image_width * CDIM;
    render_alphas += image_id * image_height * image_width;
    last_ids += image_id * image_height * image_width;
    if (backgrounds != nullptr) {
        backgrounds += image_id * CDIM;
    }
    if (masks != nullptr) {
        masks += image_id * tile_height * tile_width;
    }

    const float px = static_cast<float>(j) + 0.5f;
    const float py = static_cast<float>(i) + 0.5f;
    const int32_t pix_id = i * image_width + j;
    const bool inside = i < image_height && j < image_width;
    bool done = !inside;

    const bool tile_enabled = masks == nullptr || masks[tile_id];
    if constexpr (!OUTPUT_TRANSPOSE) {
        if (!tile_enabled) {
            if (inside) {
#pragma unroll
                for (uint32_t k = 0; k < CDIM; ++k) {
                    render_colors[pix_id * CDIM + k] =
                        backgrounds == nullptr ? 0.0f : backgrounds[k];
                }
            }
            return;
        }
    }

    const int32_t range_start = tile_enabled ? tile_offsets[tile_id] : 0;
    const int32_t range_end = tile_enabled
        ? (image_id == I - 1 && tile_id == tile_width * tile_height - 1
               ? n_isects
               : tile_offsets[tile_id + 1])
        : 0;
    const uint32_t num_batches =
        (range_end - range_start + BATCH_CAPACITY - 1) / BATCH_CAPACITY;

    extern __shared__ char shared_storage[];
    Sm120ForwardGaussianStage *stage =
        reinterpret_cast<Sm120ForwardGaussianStage *>(shared_storage);
    float4 *output_color_alpha = REUSE_GAUSSIAN_STAGE_FOR_OUTPUT
        ? reinterpret_cast<float4 *>(stage)
        : reinterpret_cast<float4 *>(stage + BATCH_CAPACITY);
    int32_t *output_last_id = reinterpret_cast<int32_t *>(
        output_color_alpha + BLOCK_SIZE
    );
    float T = 1.0f;
    uint32_t cur_idx = 0;
    float pix_out[CDIM] = {0.0f, 0.0f, 0.0f};

    for (uint32_t b = 0; b < num_batches; ++b) {
        if constexpr (SKIP_FIRST_PREWRITE_BARRIER) {
            if (b != 0 && __syncthreads_count(done) >= BLOCK_SIZE) {
                break;
            }
        } else if (__syncthreads_count(done) >= BLOCK_SIZE) {
            break;
        }

        const uint32_t batch_start = range_start + BATCH_CAPACITY * b;
#pragma unroll
        for (uint32_t load = tr; load < BATCH_CAPACITY; load += BLOCK_SIZE) {
            const uint32_t idx = batch_start + load;
            if (idx < static_cast<uint32_t>(range_end)) {
                const int32_t g = flatten_ids[idx];
                const float2 xy = reinterpret_cast<const float2 *>(means2d)[g];
                const vec3 conic = conics[g];
                stage[load].xy_opacity_conic_x =
                    make_float4(xy.x, xy.y, opacities[g], conic.x);
                stage[load].conic_yz_id_padding =
                    make_float4(conic.y, conic.z, __int_as_float(g), 0.0f);
            }
        }
        __syncthreads();

        const uint32_t batch_size =
            min(BATCH_CAPACITY, static_cast<uint32_t>(range_end) - batch_start);
        for (uint32_t t = 0; t < batch_size && !done; ++t) {
            const float4 xy_opacity_conic_x = stage[t].xy_opacity_conic_x;
            const float4 conic_yz_id = stage[t].conic_yz_id_padding;
            const float delta_x = xy_opacity_conic_x.x - px;
            const float delta_y = xy_opacity_conic_x.y - py;
            const float sigma =
                0.5f * (xy_opacity_conic_x.w * delta_x * delta_x +
                        conic_yz_id.y * delta_y * delta_y) +
                conic_yz_id.x * delta_x * delta_y;
            const float alpha = min(
                0.999f, xy_opacity_conic_x.z * __expf(-sigma)
            );
            if (sigma < 0.0f || alpha < ALPHA_THRESHOLD) {
                continue;
            }

            const float next_T = T * (1.0f - alpha);
            if (next_T <= 1e-4f) {
                done = true;
                break;
            }

            const int32_t g = __float_as_int(conic_yz_id.z);
            const float vis = alpha * T;
            const float *color = colors + g * CDIM;
#pragma unroll
            for (uint32_t k = 0; k < CDIM; ++k) {
                pix_out[k] += color[k] * vis;
            }
            cur_idx = batch_start + t;
            T = next_T;
        }
    }

    if constexpr (OUTPUT_TRANSPOSE) {
        if constexpr (REUSE_GAUSSIAN_STAGE_FOR_OUTPUT) {
            __syncthreads();
        }
        const uint32_t local_pixel_id = pixel_y * TILE_SIZE + pixel_x;
        output_color_alpha[local_pixel_id] = make_float4(
            backgrounds == nullptr ? pix_out[0] : pix_out[0] + T * backgrounds[0],
            backgrounds == nullptr ? pix_out[1] : pix_out[1] + T * backgrounds[1],
            backgrounds == nullptr ? pix_out[2] : pix_out[2] + T * backgrounds[2],
            1.0f - T
        );
        output_last_id[local_pixel_id] = static_cast<int32_t>(cur_idx);
        __syncthreads();

        const uint32_t output_i = blockIdx.y * TILE_SIZE + threadIdx.y;
        const uint32_t output_j = blockIdx.z * TILE_SIZE + threadIdx.x;
        if (output_i < image_height && output_j < image_width) {
            const int32_t output_pix_id = output_i * image_width + output_j;
            const float4 color_alpha = output_color_alpha[tr];
            render_colors[output_pix_id * CDIM] = color_alpha.x;
            render_colors[output_pix_id * CDIM + 1] = color_alpha.y;
            render_colors[output_pix_id * CDIM + 2] = color_alpha.z;
            if (tile_enabled) {
                render_alphas[output_pix_id] = color_alpha.w;
                last_ids[output_pix_id] = output_last_id[tr];
            }
        }
    } else if (inside) {
        render_alphas[pix_id] = 1.0f - T;
#pragma unroll
        for (uint32_t k = 0; k < CDIM; ++k) {
            render_colors[pix_id * CDIM + k] =
                backgrounds == nullptr ? pix_out[k]
                                       : pix_out[k] + T * backgrounds[k];
        }
        last_ids[pix_id] = static_cast<int32_t>(cur_idx);
    }
}

template <
    uint32_t WARP_PIXEL_WIDTH,
    bool OUTPUT_TRANSPOSE = false,
    bool SKIP_FIRST_PREWRITE_BARRIER = false,
    bool REUSE_GAUSSIAN_STAGE_FOR_OUTPUT = false>
void launch_rasterize_to_pixels_3dgs_fwd_sm120_mapping(
    const uint32_t I,
    const uint32_t n_isects,
    const at::Tensor means2d,
    const at::Tensor conics,
    const at::Tensor colors,
    const at::Tensor opacities,
    const at::optional<at::Tensor> backgrounds,
    const at::optional<at::Tensor> masks,
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const at::Tensor tile_offsets,
    const at::Tensor flatten_ids,
    at::Tensor renders,
    at::Tensor alphas,
    at::Tensor last_ids
) {
    constexpr uint32_t batch_capacity = 384;
    constexpr uint32_t block_size = 16 * 16;
    constexpr int64_t output_transpose_size =
        OUTPUT_TRANSPOSE && !REUSE_GAUSSIAN_STAGE_FOR_OUTPUT
        ? block_size * (sizeof(float4) + sizeof(int32_t))
        : 0;
    const dim3 threads = {16, 16, 1};
    const dim3 grid = {I, tile_height, tile_width};
    constexpr int64_t shmem_size =
        batch_capacity * sizeof(Sm120ForwardGaussianStage) +
        output_transpose_size;
    if (cudaFuncSetAttribute(
            rasterize_to_pixels_3dgs_fwd_sm120_soa_kernel<
                WARP_PIXEL_WIDTH,
                OUTPUT_TRANSPOSE,
                SKIP_FIRST_PREWRITE_BARRIER,
                REUSE_GAUSSIAN_STAGE_FOR_OUTPUT>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            shmem_size
        ) != cudaSuccess) {
        AT_ERROR(
            "Failed to set SM120 forward shared memory size (requested ",
            shmem_size,
            " bytes)."
        );
    }

    rasterize_to_pixels_3dgs_fwd_sm120_soa_kernel<
        WARP_PIXEL_WIDTH,
        OUTPUT_TRANSPOSE,
        SKIP_FIRST_PREWRITE_BARRIER,
        REUSE_GAUSSIAN_STAGE_FOR_OUTPUT>
        <<<grid, threads, shmem_size, at::cuda::getCurrentCUDAStream()>>>(
        I,
        n_isects,
        reinterpret_cast<const vec2 *>(means2d.data_ptr<float>()),
        reinterpret_cast<const vec3 *>(conics.data_ptr<float>()),
        colors.data_ptr<float>(),
        opacities.data_ptr<float>(),
        backgrounds.has_value() ? backgrounds.value().data_ptr<float>() : nullptr,
        masks.has_value() ? masks.value().data_ptr<bool>() : nullptr,
        image_width,
        image_height,
        tile_width,
        tile_height,
        tile_offsets.data_ptr<int32_t>(),
        flatten_ids.data_ptr<int32_t>(),
        renders.data_ptr<float>(),
        alphas.data_ptr<float>(),
        last_ids.data_ptr<int32_t>()
    );
}

void launch_rasterize_to_pixels_3dgs_fwd_sm120_kernel(
    const at::Tensor means2d,
    const at::Tensor conics,
    const at::Tensor colors,
    const at::Tensor opacities,
    const at::optional<at::Tensor> backgrounds,
    const at::optional<at::Tensor> masks,
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    const at::Tensor tile_offsets,
    const at::Tensor flatten_ids,
    at::Tensor renders,
    at::Tensor alphas,
    at::Tensor last_ids
) {
    TORCH_INTERNAL_ASSERT(tile_size == 16);
    const uint32_t I = alphas.numel() / (image_height * image_width);
    const uint32_t tile_height = tile_offsets.size(-2);
    const uint32_t tile_width = tile_offsets.size(-1);
    const uint32_t n_isects = flatten_ids.size(0);
    launch_rasterize_to_pixels_3dgs_fwd_sm120_mapping<4, true, false, true>(
        I, n_isects, means2d, conics, colors, opacities, backgrounds, masks,
        image_width, image_height, tile_width, tile_height, tile_offsets,
        flatten_ids, renders, alphas, last_ids
    );
}

} // namespace ptxsplat
