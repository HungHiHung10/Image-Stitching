#include "gpu_backend.cuh"
#include "logging.h"

namespace {

constexpr int kThreads = 256;
constexpr int kScanThreads = 1024;
constexpr int kSeamThreads = 1024;
constexpr int kTile = 16;
constexpr int kExposureTerms = 12;
constexpr int kErrorTerms = 2;
constexpr int kRefitOk = 2;

struct Mat3 {
    double m[9];
};

struct WarpGeometry {
    hd::RgbView a;
    hd::RgbView b;
    Mat3 h_inv;
    float offset_x;
    float offset_y;
    int width;
    int height;
};

struct BlendParams {
    double gain[3];
    double bias[3];
};

struct Chunk {
    int begin;
    int end;
    int offset;
    int total;
};

int blocks_for(long long count, int threads) {
    return static_cast<int>((count + threads - 1) / threads);
}

void check_launch() {
    CHECK_CUDA(cudaGetLastError());
}

__device__ int block_exclusive_scan(int value, int* scratch, int& total) {
    const int tid = threadIdx.x;
    const int threads = blockDim.x;
    scratch[tid] = value;
    __syncthreads();
    for (int offset = 1; offset < threads; offset <<= 1) {
        const int addend = tid >= offset ? scratch[tid - offset] : 0;
        __syncthreads();
        scratch[tid] += addend;
        __syncthreads();
    }
    total = scratch[threads - 1];
    const int exclusive = scratch[tid] - value;
    __syncthreads();
    return exclusive;
}

__device__ Chunk ordered_chunk(const int* flags, int count, int* scratch) {
    const int threads = blockDim.x;
    const int per_thread = (count + threads - 1) / threads;
    Chunk chunk;
    chunk.begin = hd::min_of(count, static_cast<int>(threadIdx.x) * per_thread);
    chunk.end = hd::min_of(count, chunk.begin + per_thread);
    int local = 0;
    for (int i = chunk.begin; i < chunk.end; ++i) local += flags[i] != 0;
    chunk.offset = block_exclusive_scan(local, scratch, chunk.total);
    return chunk;
}

template <int Terms>
__device__ void block_reduce_sums(double (&values)[Terms], unsigned long long& count) {
    __shared__ double sums[Terms][kThreads];
    __shared__ unsigned long long counts[kThreads];
    const int tid = threadIdx.x;
    for (int k = 0; k < Terms; ++k) sums[k][tid] = values[k];
    counts[tid] = count;
    __syncthreads();
    for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            for (int k = 0; k < Terms; ++k) sums[k][tid] += sums[k][tid + stride];
            counts[tid] += counts[tid + stride];
        }
        __syncthreads();
    }
    for (int k = 0; k < Terms; ++k) values[k] = sums[k][0];
    count = counts[0];
}

__device__ bool sample_position(long long sample, int grid_w, int grid_h, const WarpGeometry& g,
                                float& ax, float& ay) {
    if (sample >= static_cast<long long>(grid_w) * grid_h) return false;
    const int px = static_cast<int>(sample % grid_w) * cfg::kExposureStride;
    const int py = static_cast<int>(sample / grid_w) * cfg::kExposureStride;
    ax = px - g.offset_x;
    ay = py - g.offset_y;
    return true;
}

__global__ void match_forward_kernel(const Descriptor256* desc_a, int count_a,
                                     const Descriptor256* desc_b, int count_b,
                                     int* best_index, int* best_distance, int* second_distance) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count_a) return;
    const Descriptor256 query = desc_a[i];
    int best = 256, second = 256, index = -1;
    for (int j = 0; j < count_b; ++j)
        hd::match_keep_top2(hd::hamming256(query.data, desc_b[j].data), j, best, second, index);
    best_index[i] = index;
    best_distance[i] = best;
    second_distance[i] = second;
}

__global__ void match_reverse_kernel(const Descriptor256* desc_a, int count_a,
                                     const Descriptor256* desc_b, int count_b, int* reverse_index) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= count_b) return;
    const Descriptor256 query = desc_b[j];
    int best = 256, index = -1;
    for (int i = 0; i < count_a; ++i)
        hd::match_keep_best(hd::hamming256(query.data, desc_a[i].data), i, best, index);
    reverse_index[j] = index;
}

__global__ void match_flag_kernel(int count_a, const int* best_index, const int* best_distance,
                                  const int* second_distance, const int* reverse_index,
                                  float ratio_threshold, int* flags) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count_a) return;
    const int j = best_index[i];
    flags[i] = hd::match_passes_ratio(j, best_distance[i], second_distance[i], ratio_threshold) &&
               reverse_index[j] == i;
}

__global__ void compact_matches_kernel(const int* flags, int count, const int* best_index,
                                       const int* best_distance, Match* matches, int* match_count) {
    __shared__ int scratch[kScanThreads];
    const Chunk chunk = ordered_chunk(flags, count, scratch);
    int out = chunk.offset;
    for (int i = chunk.begin; i < chunk.end; ++i) {
        if (!flags[i]) continue;
        matches[out].queryIdx = i;
        matches[out].trainIdx = best_index[i];
        matches[out].distance = best_distance[i];
        ++out;
    }
    if (threadIdx.x == 0) *match_count = chunk.total;
}

__global__ void compact_indices_kernel(const int* flags, int count, int* indices, int* index_count) {
    __shared__ int scratch[kScanThreads];
    const Chunk chunk = ordered_chunk(flags, count, scratch);
    int out = chunk.offset;
    for (int i = chunk.begin; i < chunk.end; ++i)
        if (flags[i]) indices[out++] = i;
    if (threadIdx.x == 0) *index_count = chunk.total;
}

__global__ void gather_points_kernel(const Keypoint* keypoints_a, const Keypoint* keypoints_b,
                                     const Match* matches, int count, float* src, float* dst) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= count) return;
    const Keypoint& b = keypoints_b[matches[k].trainIdx];
    const Keypoint& a = keypoints_a[matches[k].queryIdx];
    src[2 * k] = b.x;
    src[2 * k + 1] = b.y;
    dst[2 * k] = a.x;
    dst[2 * k + 1] = a.y;
}

__global__ void hypothesis_kernel(const float* src, const float* dst, int count, int iterations,
                                  uint32_t seed, GpuModel* models, int* valid) {
    const int iter = blockIdx.x * blockDim.x + threadIdx.x;
    if (iter >= iterations) return;
    int sample[4];
    bool ok = hd::ransac_sample(seed, static_cast<uint32_t>(iter), static_cast<uint32_t>(count), sample);
    if (ok) {
        float s[8], d[8];
        for (int k = 0; k < 4; ++k) {
            s[2 * k] = src[2 * sample[k]];
            s[2 * k + 1] = src[2 * sample[k] + 1];
            d[2 * k] = dst[2 * sample[k]];
            d[2 * k + 1] = dst[2 * sample[k] + 1];
        }
        GpuModel& model = models[iter];
        ok = hd::dlt_fit(s, d, 4, model.h) && hd::invert_h(model.h, model.h_inv);
    }
    valid[iter] = ok ? 1 : 0;
}

__global__ void score_kernel(const float* src, const float* dst, int count, const GpuModel* models,
                             const int* valid, float threshold_sq, int* scores) {
    __shared__ double h[9];
    __shared__ double h_inv[9];
    __shared__ int partial[kThreads];
    const int iter = blockIdx.x;
    const int tid = threadIdx.x;
    if (!valid[iter]) {
        if (tid == 0) scores[iter] = 0;
        return;
    }
    if (tid < 9) h[tid] = models[iter].h[tid];
    else if (tid < 18) h_inv[tid - 9] = models[iter].h_inv[tid - 9];
    __syncthreads();
    int local = 0;
    for (int k = tid; k < count; k += kThreads) {
        const double error = hd::symmetric_error_sq(h, h_inv, dst[2 * k], dst[2 * k + 1], src[2 * k], src[2 * k + 1]);
        local += hd::is_inlier(error, threshold_sq) ? 1 : 0;
    }
    partial[tid] = local;
    __syncthreads();
    for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        __syncthreads();
    }
    if (tid == 0) scores[iter] = partial[0];
}

__device__ bool better_score(int score, int iter, int best_score, int best_iter) {
    return score > best_score || (score == best_score && iter < best_iter);
}

__global__ void select_kernel(const int* scores, int iterations, int* best_iter, int* best_score) {
    __shared__ int top_scores[kScanThreads];
    __shared__ int top_iters[kScanThreads];
    const int tid = threadIdx.x;
    int score = -1, iter = iterations;
    for (int i = tid; i < iterations; i += kScanThreads) {
        if (better_score(scores[i], i, score, iter)) {
            score = scores[i];
            iter = i;
        }
    }
    top_scores[tid] = score;
    top_iters[tid] = iter;
    __syncthreads();
    for (int stride = kScanThreads / 2; stride > 0; stride >>= 1) {
        if (tid < stride && better_score(top_scores[tid + stride], top_iters[tid + stride],
                                         top_scores[tid], top_iters[tid])) {
            top_scores[tid] = top_scores[tid + stride];
            top_iters[tid] = top_iters[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        *best_iter = top_iters[0];
        *best_score = top_scores[0];
    }
}

__global__ void inlier_flag_kernel(const float* src, const float* dst, int count, const GpuModel* model,
                                   float threshold_sq, int* flags) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= count) return;
    const double error = hd::symmetric_error_sq(model->h, model->h_inv, dst[2 * k], dst[2 * k + 1],
                                                src[2 * k], src[2 * k + 1]);
    flags[k] = hd::is_inlier(error, threshold_sq) ? 1 : 0;
}

__global__ void gather_inliers_kernel(const float* src, const float* dst, const int* inliers, int count,
                                      float* inlier_src, float* inlier_dst) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= count) return;
    const int k = inliers[t];
    inlier_src[2 * t] = src[2 * k];
    inlier_src[2 * t + 1] = src[2 * k + 1];
    inlier_dst[2 * t] = dst[2 * k];
    inlier_dst[2 * t + 1] = dst[2 * k + 1];
}

__global__ void refit_kernel(const float* inlier_src, const float* inlier_dst, int count,
                             GpuModel* model, int* status) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    int result = 0;
    if (hd::dlt_fit(inlier_src, inlier_dst, count, model->h))
        result = hd::invert_h(model->h, model->h_inv) ? kRefitOk : 1;
    *status = result;
}

__global__ void exposure_stats_kernel(WarpGeometry g, int grid_w, int grid_h, double* partial_sums,
                                      unsigned long long* partial_counts) {
    double terms[kExposureTerms] = {};
    unsigned long long used = 0;
    float ax, ay, a[3], b[3];
    const long long sample = static_cast<long long>(blockIdx.x) * kThreads + threadIdx.x;
    if (sample_position(sample, grid_w, grid_h, g, ax, ay) &&
        hd::overlap_samples(g.a, g.b, g.h_inv.m, ax, ay, a, b) && hd::exposure_usable(a, b)) {
        used = 1;
        for (int c = 0; c < 3; ++c) {
            terms[c] = b[c];
            terms[3 + c] = a[c];
            terms[6 + c] = b[c] * b[c];
            terms[9 + c] = b[c] * a[c];
        }
    }
    block_reduce_sums<kExposureTerms>(terms, used);
    if (threadIdx.x == 0) {
        for (int k = 0; k < kExposureTerms; ++k) partial_sums[blockIdx.x * kExposureTerms + k] = terms[k];
        partial_counts[blockIdx.x] = used;
    }
}

__global__ void exposure_fit_kernel(const double* partial_sums, const unsigned long long* partial_counts,
                                    int blocks, GpuExposureState* state) {
    double terms[kExposureTerms] = {};
    unsigned long long used = 0;
    for (int block = threadIdx.x; block < blocks; block += kThreads) {
        for (int k = 0; k < kExposureTerms; ++k) terms[k] += partial_sums[block * kExposureTerms + k];
        used += partial_counts[block];
    }
    block_reduce_sums<kExposureTerms>(terms, used);
    if (threadIdx.x == 0) {
        state->exposure_samples = used;
        hd::exposure_fit(used, terms, terms + 3, terms + 6, terms + 9, state->gain, state->bias);
    }
}

__global__ void exposure_error_kernel(WarpGeometry g, int grid_w, int grid_h, const GpuExposureState* state,
                                      double* partial_sums, unsigned long long* partial_counts) {
    double terms[kErrorTerms] = {};
    unsigned long long used = 0;
    float ax, ay, a[3], b[3];
    const long long sample = static_cast<long long>(blockIdx.x) * kThreads + threadIdx.x;
    if (sample_position(sample, grid_w, grid_h, g, ax, ay) &&
        hd::overlap_samples(g.a, g.b, g.h_inv.m, ax, ay, a, b)) {
        for (int c = 0; c < 3; ++c) {
            terms[0] += hd::raw_abs_error(a[c], b[c]);
            terms[1] += hd::corrected_abs_error(a[c], b[c], state->gain[c], state->bias[c]);
        }
        used = 3;
    }
    block_reduce_sums<kErrorTerms>(terms, used);
    if (threadIdx.x == 0) {
        for (int k = 0; k < kErrorTerms; ++k) partial_sums[blockIdx.x * kErrorTerms + k] = terms[k];
        partial_counts[blockIdx.x] = used;
    }
}

__global__ void exposure_decide_kernel(const double* partial_sums, const unsigned long long* partial_counts,
                                       int blocks, GpuExposureState* state) {
    double terms[kErrorTerms] = {};
    unsigned long long used = 0;
    for (int block = threadIdx.x; block < blocks; block += kThreads) {
        for (int k = 0; k < kErrorTerms; ++k) terms[k] += partial_sums[block * kErrorTerms + k];
        used += partial_counts[block];
    }
    block_reduce_sums<kErrorTerms>(terms, used);
    if (threadIdx.x == 0) {
        state->error_samples = used;
        state->raw_error = terms[0];
        state->corrected_error = terms[1];
        const hd::ExposureDecision decision = hd::exposure_decide(
            state->exposure_samples, used, terms[0], terms[1], state->gain, state->bias);
        state->compensate = decision.compensate ? 1 : 0;
        state->use_seam = decision.use_seam ? 1 : 0;
        state->overlap_mae = decision.overlap_mae;
    }
}

__global__ void seam_cost_kernel(WarpGeometry g, BlendParams p, float* cost) {
    const int px = blockIdx.x * blockDim.x + threadIdx.x;
    const int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= g.width || py >= g.height) return;
    float a[3], b[3];
    float value = -1.0f;
    if (hd::overlap_samples(g.a, g.b, g.h_inv.m, px - g.offset_x, py - g.offset_y, a, b)) {
        for (int c = 0; c < 3; ++c) b[c] = static_cast<float>(hd::exposure_apply(p.gain[c], p.bias[c], b[c]));
        value = hd::abs_diff_sum(a, b);
    }
    cost[static_cast<size_t>(py) * g.width + px] = value;
}

__global__ void seam_dp_kernel(const float* cost, int width, int height, double* row_a, double* row_b,
                               int8_t* parent, int* seam_x) {
    __shared__ double min_values[kSeamThreads];
    __shared__ int min_indices[kSeamThreads];
    const int tid = threadIdx.x;
    const double inf = hd::infinity_of();
    double* previous = row_a;
    double* current = row_b;
    for (int px = tid; px < width; px += kSeamThreads) previous[px] = inf;
    __syncthreads();
    int first_y = -1, last_y = -1;
    for (int py = 0; py < height; ++py) {
        const float* cost_row = cost + static_cast<size_t>(py) * width;
        int8_t* parent_row = parent + static_cast<size_t>(py) * width;
        int any = 0;
        for (int px = tid; px < width; px += kSeamThreads) {
            const float pixel = cost_row[px];
            double value = inf;
            if (pixel >= 0.0f) {
                any = 1;
                const double pixel_cost = hd::seam_pixel_cost(pixel);
                if (first_y < 0) {
                    value = pixel_cost;
                } else {
                    int direction;
                    const double best = hd::seam_best_previous(previous, width, px, direction);
                    if (hd::is_finite(best)) {
                        value = best + pixel_cost;
                        parent_row[px] = static_cast<int8_t>(direction);
                    }
                }
            }
            current[px] = value;
        }
        if (__syncthreads_or(any)) {
            if (first_y < 0) first_y = py;
            last_y = py;
            double* swap = previous;
            previous = current;
            current = swap;
        }
    }
    double best_value = inf;
    int best_index = width;
    for (int px = tid; px < width; px += kSeamThreads) {
        if (previous[px] < best_value || (previous[px] == best_value && px < best_index)) {
            best_value = previous[px];
            best_index = px;
        }
    }
    min_values[tid] = best_value;
    min_indices[tid] = best_index;
    __syncthreads();
    for (int stride = kSeamThreads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            const double value = min_values[tid + stride];
            const int index = min_indices[tid + stride];
            if (value < min_values[tid] || (value == min_values[tid] && index < min_indices[tid])) {
                min_values[tid] = value;
                min_indices[tid] = index;
            }
        }
        __syncthreads();
    }
    if (tid == 0 && first_y >= 0) {
        int x = min_indices[0];
        for (int py = last_y; py >= first_y; --py) {
            seam_x[py] = x;
            x += parent[static_cast<size_t>(py) * width + x];
            x = hd::seam_clamp_x(x, width);
        }
    }
}

__global__ void warp_blend_kernel(WarpGeometry g, BlendParams p, const int* seam_x, int use_seam,
                                  uint8_t* panorama) {
    const int px = blockIdx.x * blockDim.x + threadIdx.x;
    const int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= g.width || py >= g.height) return;
    const int seam = use_seam ? seam_x[py] : -1;
    hd::warp_blend_pixel(g.a, g.b, g.h_inv.m, g.offset_x, g.offset_y, p.gain, p.bias, seam, px, py,
                         panorama + (static_cast<size_t>(py) * g.width + px) * 3);
}

}

int GpuBackend::read_counter(Counter slot, cudaStream_t stream) const {
    int value = 0;
    CHECK_CUDA(cudaMemcpyAsync(&value, counters_.data() + slot, sizeof(int), cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    return value;
}

int GpuBackend::match(const GpuFeatures& a, const GpuFeatures& b, cudaStream_t stream) {
    if (a.count <= 0 || b.count <= 0) return 0;
    counters_.reserve(kCounterSlots);
    forward_index_.reserve(a.count);
    forward_best_.reserve(a.count);
    forward_second_.reserve(a.count);
    flags_.reserve(a.count);
    matches_.reserve(a.count);
    reverse_index_.reserve(b.count);

    match_forward_kernel<<<blocks_for(a.count, kThreads), kThreads, 0, stream>>>(
        a.descriptors, a.count, b.descriptors, b.count,
        forward_index_.data(), forward_best_.data(), forward_second_.data());
    check_launch();
    match_reverse_kernel<<<blocks_for(b.count, kThreads), kThreads, 0, stream>>>(
        a.descriptors, a.count, b.descriptors, b.count, reverse_index_.data());
    check_launch();
    match_flag_kernel<<<blocks_for(a.count, kThreads), kThreads, 0, stream>>>(
        a.count, forward_index_.data(), forward_best_.data(), forward_second_.data(),
        reverse_index_.data(), cfg::kRatioThreshold, flags_.data());
    check_launch();
    compact_matches_kernel<<<1, kScanThreads, 0, stream>>>(
        flags_.data(), a.count, forward_index_.data(), forward_best_.data(),
        matches_.data(), counters_.data() + kMatchCount);
    check_launch();
    return read_counter(kMatchCount, stream);
}

int GpuBackend::collect_inliers(const GpuModel* model, int match_count, DeviceBuffer<int>& out, Counter slot,
                                cudaStream_t stream) {
    const float threshold_sq = cfg::kInlierThresholdPx * cfg::kInlierThresholdPx;
    inlier_flag_kernel<<<blocks_for(match_count, kThreads), kThreads, 0, stream>>>(
        src_points_.data(), dst_points_.data(), match_count, model, threshold_sq, flags_.data());
    check_launch();
    compact_indices_kernel<<<1, kScanThreads, 0, stream>>>(
        flags_.data(), match_count, out.data(), counters_.data() + slot);
    check_launch();
    return read_counter(slot, stream);
}

bool GpuBackend::estimate_homography(const GpuFeatures& a, const GpuFeatures& b, const Match* matches,
                                     int match_count, Homography3x3& h, int& inlier_count,
                                     cudaStream_t stream) {
    inlier_count = 0;
    if (match_count < 4) return false;
    const int iterations = cfg::kRansacIters;
    const float threshold_sq = cfg::kInlierThresholdPx * cfg::kInlierThresholdPx;
    counters_.reserve(kCounterSlots);
    src_points_.reserve(2 * static_cast<size_t>(match_count));
    dst_points_.reserve(2 * static_cast<size_t>(match_count));
    inlier_src_.reserve(2 * static_cast<size_t>(match_count));
    inlier_dst_.reserve(2 * static_cast<size_t>(match_count));
    flags_.reserve(match_count);
    inliers_.reserve(match_count);
    refined_inliers_.reserve(match_count);
    models_.reserve(iterations);
    valid_.reserve(iterations);
    scores_.reserve(iterations);
    current_model_.reserve(1);
    refined_model_.reserve(1);

    gather_points_kernel<<<blocks_for(match_count, kThreads), kThreads, 0, stream>>>(
        a.keypoints, b.keypoints, matches, match_count, src_points_.data(), dst_points_.data());
    check_launch();
    hypothesis_kernel<<<blocks_for(iterations, kThreads), kThreads, 0, stream>>>(
        src_points_.data(), dst_points_.data(), match_count, iterations, cfg::kRansacSeed,
        models_.data(), valid_.data());
    check_launch();
    score_kernel<<<iterations, kThreads, 0, stream>>>(
        src_points_.data(), dst_points_.data(), match_count, models_.data(), valid_.data(),
        threshold_sq, scores_.data());
    check_launch();
    select_kernel<<<1, kScanThreads, 0, stream>>>(
        scores_.data(), iterations, counters_.data() + kBestIter, counters_.data() + kBestScore);
    check_launch();
    int best[2];
    CHECK_CUDA(cudaMemcpyAsync(best, counters_.data() + kBestIter, sizeof(best), cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    const int best_iter = best[0];
    const int best_score = best[1];
    if (best_score < 4) return false;

    CHECK_CUDA(cudaMemcpyAsync(current_model_.data(), models_.data() + best_iter, sizeof(GpuModel),
                               cudaMemcpyDeviceToDevice, stream));
    int count = collect_inliers(current_model_.data(), match_count, inliers_, kInlierCount, stream);
    for (int pass = 0; pass < cfg::kRefitPasses; ++pass) {
        gather_inliers_kernel<<<blocks_for(count, kThreads), kThreads, 0, stream>>>(
            src_points_.data(), dst_points_.data(), inliers_.data(), count,
            inlier_src_.data(), inlier_dst_.data());
        check_launch();
        refit_kernel<<<1, 1, 0, stream>>>(inlier_src_.data(), inlier_dst_.data(), count,
                                          refined_model_.data(), counters_.data() + kRefitStatus);
        check_launch();
        if (read_counter(kRefitStatus, stream) != kRefitOk) break;
        const int refined = collect_inliers(refined_model_.data(), match_count, refined_inliers_,
                                            kRefinedCount, stream);
        if (refined < 4) break;
        current_model_.swap(refined_model_);
        inliers_.swap(refined_inliers_);
        count = refined;
    }
    GpuModel model;
    current_model_.download(&model, 1, stream);
    CHECK_CUDA(cudaStreamSynchronize(stream));
    for (int i = 0; i < 9; ++i) h.h[i] = model.h[i];
    inlier_count = count;
    return count >= 4;
}

void GpuBackend::warp_and_blend(const GpuImage& a, const GpuImage& b, const Homography3x3& h,
                                BlendInfo* info, cudaStream_t stream) {
    if (info) *info = BlendInfo{};
    layout_ = hd::canvas_layout(h.h, a.width, a.height, b.width, b.height);
    const int width = layout_.width;
    const int height = layout_.height;
    const size_t pixels = static_cast<size_t>(width) * height;
    panorama_.reserve(pixels * 3);
    if (info) {
        info->canvas_w = width;
        info->canvas_h = height;
        info->offset_x = layout_.offset_x;
        info->offset_y = layout_.offset_y;
    }
    WarpGeometry g;
    if (!hd::invert_h(h.h, g.h_inv.m)) {
        panorama_.fill_bytes(0, pixels * 3, stream);
        CHECK_CUDA(cudaStreamSynchronize(stream));
        return;
    }
    g.a = hd::RgbView{a.data, a.width, a.height};
    g.b = hd::RgbView{b.data, b.width, b.height};
    g.offset_x = layout_.offset_x;
    g.offset_y = layout_.offset_y;
    g.width = width;
    g.height = height;

    const int grid_w = (width + cfg::kExposureStride - 1) / cfg::kExposureStride;
    const int grid_h = (height + cfg::kExposureStride - 1) / cfg::kExposureStride;
    const int sample_blocks = blocks_for(static_cast<long long>(grid_w) * grid_h, kThreads);
    partial_sums_.reserve(static_cast<size_t>(sample_blocks) * kExposureTerms);
    partial_counts_.reserve(sample_blocks);
    exposure_.reserve(1);
    seam_x_.reserve(height);

    exposure_stats_kernel<<<sample_blocks, kThreads, 0, stream>>>(
        g, grid_w, grid_h, partial_sums_.data(), partial_counts_.data());
    check_launch();
    exposure_fit_kernel<<<1, kThreads, 0, stream>>>(
        partial_sums_.data(), partial_counts_.data(), sample_blocks, exposure_.data());
    check_launch();
    exposure_error_kernel<<<sample_blocks, kThreads, 0, stream>>>(
        g, grid_w, grid_h, exposure_.data(), partial_sums_.data(), partial_counts_.data());
    check_launch();
    exposure_decide_kernel<<<1, kThreads, 0, stream>>>(
        partial_sums_.data(), partial_counts_.data(), sample_blocks, exposure_.data());
    check_launch();
    GpuExposureState state;
    exposure_.download(&state, 1, stream);
    CHECK_CUDA(cudaStreamSynchronize(stream));

    log_message(LogLevel::Process) << "Blend: exposure="
        << (state.compensate ? "affine" : "unchanged")
        << ", overlap MAE=" << state.overlap_mae
        << ", seam=" << (state.use_seam ? "optimal" : "feather") << "\n";

    BlendParams params;
    for (int c = 0; c < 3; ++c) {
        params.gain[c] = state.gain[c];
        params.bias[c] = state.bias[c];
    }
    const dim3 tile(kTile, kTile);
    const dim3 grid(blocks_for(width, kTile), blocks_for(height, kTile));
    if (state.use_seam) {
        seam_cost_.reserve(pixels);
        seam_parent_.reserve(pixels);
        seam_rows_.reserve(2 * static_cast<size_t>(width));
        seam_parent_.fill_bytes(0, pixels, stream);
        seam_x_.fill_bytes(0xFF, height, stream);
        seam_cost_kernel<<<grid, tile, 0, stream>>>(g, params, seam_cost_.data());
        check_launch();
        seam_dp_kernel<<<1, kSeamThreads, 0, stream>>>(seam_cost_.data(), width, height, seam_rows_.data(),
                                                       seam_rows_.data() + width, seam_parent_.data(),
                                                       seam_x_.data());
        check_launch();
    }
    warp_blend_kernel<<<grid, tile, 0, stream>>>(g, params, seam_x_.data(), state.use_seam, panorama_.data());
    check_launch();

    if (info) {
        info->exposure_applied = state.compensate != 0;
        info->overlap_mae = state.overlap_mae;
        info->seam_used = state.use_seam != 0;
        for (int c = 0; c < 3; ++c) {
            info->gain[c] = state.gain[c];
            info->bias[c] = state.bias[c];
        }
        info->seam_x = state.use_seam ? seam_x_.to_host(height, stream) : std::vector<int>(height, -1);
    }
}

std::vector<Match> GpuBackend::download_matches(int count, cudaStream_t stream) const {
    return matches_.to_host(count, stream);
}

std::vector<int> GpuBackend::download_inliers(int count, cudaStream_t stream) const {
    return inliers_.to_host(count, stream);
}

void GpuBackend::download_panorama(ImageRGB& panorama, cudaStream_t stream) const {
    panorama.width = layout_.width;
    panorama.height = layout_.height;
    panorama.channels = 3;
    panorama.data.resize(static_cast<size_t>(layout_.width) * layout_.height * 3);
    panorama_.download(panorama.data.data(), panorama.data.size(), stream);
    CHECK_CUDA(cudaStreamSynchronize(stream));
}
