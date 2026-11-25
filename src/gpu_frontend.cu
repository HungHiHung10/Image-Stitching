#include "gpu_frontend.cuh"
#define STITCH_DEVICE_BRIEF_PATTERN
#include "brief_pattern.h"
#undef STITCH_DEVICE_BRIEF_PATTERN
#include <cub/device/device_radix_sort.cuh>
#include <algorithm>

namespace {
constexpr int kThreads = 256;
constexpr int kCircleSize = 16;
constexpr int kArcLength = 9;
constexpr int kMaxScore = kCircleSize * 255;
constexpr int kKeyBits = 32 + 12;

__constant__ int kCircle[kCircleSize][2] = {
    {0, 3}, {1, 3}, {2, 2}, {3, 1}, {3, 0}, {3, -1}, {2, -2}, {1, -3},
    {0, -3}, {-1, -3}, {-2, -2}, {-3, -1}, {-3, 0}, {-3, 1}, {-2, 2}, {-1, 3}
};

int blocks_for(int count) { return (count - 1) / kThreads + 1; }

__device__ int index_in_image() { return blockIdx.x * blockDim.x + threadIdx.x; }

__global__ void grayscale(const uint8_t* rgb, uint8_t* gray, int pixels) {
    const int i = index_in_image();
    if (i >= pixels) return;
    gray[i] = static_cast<uint8_t>(0.299f * rgb[3 * i] +
                                  0.587f * rgb[3 * i + 1] + 0.114f * rgb[3 * i + 2]);
}

__global__ void blur_horizontal(const uint8_t* gray, float* output, int w, int h,
                                GaussianWeights weights) {
    const int i = index_in_image();
    if (i >= w * h) return;
    const int x = i % w, row = i - x;
    float acc = 0.0f;
    for (int k = -cfg::kBlurRadius; k <= cfg::kBlurRadius; ++k) {
        const int sx = min(max(x + k, 0), w - 1);
        acc += weights.values[k + cfg::kBlurRadius] * gray[row + sx];
    }
    output[i] = acc;
}

__global__ void blur_vertical(const float* input, uint8_t* output, int w, int h,
                              GaussianWeights weights) {
    const int i = index_in_image();
    if (i >= w * h) return;
    const int x = i % w, y = i / w;
    float acc = 0.0f;
    for (int k = -cfg::kBlurRadius; k <= cfg::kBlurRadius; ++k) {
        const int sy = min(max(y + k, 0), h - 1);
        acc += weights.values[k + cfg::kBlurRadius] * input[sy * w + x];
    }
    output[i] = static_cast<uint8_t>(fminf(fmaxf(acc, 0.0f), 255.0f));
}

__global__ void fast_scores(const uint8_t* gray, uint16_t* scores, int w, int h) {
    const int i = index_in_image();
    if (i >= w * h) return;
    scores[i] = 0;
    const int x = i % w, y = i / w;
    if (x < cfg::kDescriptorBorder || x >= w - cfg::kDescriptorBorder ||
        y < cfg::kDescriptorBorder || y >= h - cfg::kDescriptorBorder) return;
    const int p = gray[i];
    const int low = max(0, p - cfg::kFastThreshold), high = min(255, p + cfg::kFastThreshold);
    int bright = 0, dark = 0;
    for (int k = 0; k < kCircleSize; k += 4) {
        const int value = gray[i + kCircle[k][1] * w + kCircle[k][0]];
        bright += value > high;
        dark += value < low;
    }
    if (bright < 2 && dark < 2) return;

    int values[kCircleSize];
    int response = 0;
    for (int k = 0; k < kCircleSize; ++k) {
        const int value = gray[i + kCircle[k][1] * w + kCircle[k][0]];
        values[k] = value;
        response += abs(value - p);
    }
    bright = dark = 0;
    for (int k = 0; k < kCircleSize + kArcLength - 1; ++k) {
        const int value = values[k % kCircleSize];
        bright = value > high ? bright + 1 : 0;
        dark = value < low ? dark + 1 : 0;
        if (bright == kArcLength || dark == kArcLength) {
            scores[i] = response;
            return;
        }
    }
}

__global__ void collect_candidates(const uint16_t* scores, uint64_t* candidates,
                                   int* count, int w, int h) {
    const int i = index_in_image();
    if (i >= w * h) return;
    const int score = scores[i];
    if (score == 0) return;
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            if ((dx != 0 || dy != 0) && scores[i + dy * w + dx] >= score) return;
        }
    }
    candidates[atomicAdd(count, 1)] = (uint64_t(kMaxScore - score) << 32) | uint32_t(i);
}

__global__ void unpack_keypoints(const uint64_t* sorted, Keypoint* keypoints, int count, int w) {
    const int i = index_in_image();
    if (i >= count) return;
    const uint64_t key = sorted[i];
    const int pixel = static_cast<uint32_t>(key);
    keypoints[i].x = pixel % w;
    keypoints[i].y = pixel / w;
    keypoints[i].response = kMaxScore - static_cast<int>(key >> 32);
    keypoints[i].octave = 0;
}

__global__ void brief(const uint8_t* gray, const Keypoint* keypoints, Descriptor256* descriptors,
                      int count, int w) {
    const int i = index_in_image();
    if (i >= count) return;
    const int center = static_cast<int>(keypoints[i].y) * w + static_cast<int>(keypoints[i].x);
    Descriptor256 descriptor{};
    for (int bit = 0; bit < 256; ++bit) {
        const int8_t* pair = kBriefPattern[bit];
        const int a = gray[center + pair[1] * w + pair[0]];
        const int b = gray[center + pair[3] * w + pair[2]];
        if (a < b) descriptor.data[bit / 8] |= 1u << (bit % 8);
    }
    descriptors[i] = descriptor;
}
}

ImageGray GpuFrontendImage::download_blurred(cudaStream_t stream) const {
    return {width, height, blurred.to_host(size_t(width) * height, stream)};
}

void GpuFrontend::preprocess(GpuFrontendImage& image, cudaStream_t stream) {
    const int pixels = image.width * image.height;
    gray_.reserve(pixels);
    horizontal_.reserve(pixels);
    image.blurred.reserve(pixels);
    const int blocks = blocks_for(pixels);
    grayscale<<<blocks, kThreads, 0, stream>>>(image.rgb.data(), gray_.data(), pixels);
    CHECK_CUDA(cudaGetLastError());
    blur_horizontal<<<blocks, kThreads, 0, stream>>>(gray_.data(), horizontal_.data(), image.width,
                                                     image.height, weights_);
    CHECK_CUDA(cudaGetLastError());
    blur_vertical<<<blocks, kThreads, 0, stream>>>(horizontal_.data(), image.blurred.data(), image.width,
                                                   image.height, weights_);
    CHECK_CUDA(cudaGetLastError());
}

void GpuFrontend::detect(GpuFrontendImage& image, cudaStream_t stream) {
    image.count = 0;
    const int w = image.width, h = image.height;
    const int interior_w = w - 2 * cfg::kDescriptorBorder;
    const int interior_h = h - 2 * cfg::kDescriptorBorder;
    if (interior_w <= 0 || interior_h <= 0) return;

    scores_.reserve(w * h);
    // Strict NMS leaves at most one survivor in each 2x2 tile.
    candidates_.reserve(size_t((interior_w + 1) / 2) * ((interior_h + 1) / 2));
    candidate_count_.reserve(1);
    candidate_count_.fill_bytes(0, 1, stream);
    const int blocks = blocks_for(w * h);
    fast_scores<<<blocks, kThreads, 0, stream>>>(image.blurred.data(), scores_.data(), w, h);
    CHECK_CUDA(cudaGetLastError());
    collect_candidates<<<blocks, kThreads, 0, stream>>>(scores_.data(), candidates_.data(),
                                                        candidate_count_.data(), w, h);
    CHECK_CUDA(cudaGetLastError());
    int count;
    candidate_count_.download(&count, 1, stream);
    CHECK_CUDA(cudaStreamSynchronize(stream));
    if (count == 0) return;

    // Ascending radix keys encode descending response, then ascending y and x.
    sorted_.reserve(count);
    cub::DoubleBuffer<uint64_t> keys(candidates_.data(), sorted_.data());
    size_t bytes = 0;
    CHECK_CUDA(cub::DeviceRadixSort::SortKeys(nullptr, bytes, keys, count, 0, kKeyBits, stream));
    sort_scratch_.reserve(std::max(size_t(1), bytes));
    bytes = sort_scratch_.capacity();
    CHECK_CUDA(cub::DeviceRadixSort::SortKeys(sort_scratch_.data(), bytes, keys, count, 0, kKeyBits, stream));
    image.count = std::min(count, cfg::kMaxKeypoints);
    image.keypoints.reserve(image.count);
    unpack_keypoints<<<blocks_for(image.count), kThreads, 0, stream>>>(keys.Current(), image.keypoints.data(),
                                                                       image.count, w);
    CHECK_CUDA(cudaGetLastError());
}

void GpuFrontend::describe(GpuFrontendImage& image, cudaStream_t stream) {
    if (image.count == 0) return;
    image.descriptors.reserve(image.count);
    brief<<<blocks_for(image.count), kThreads, 0, stream>>>(image.blurred.data(), image.keypoints.data(),
                                                             image.descriptors.data(), image.count, image.width);
    CHECK_CUDA(cudaGetLastError());
}
