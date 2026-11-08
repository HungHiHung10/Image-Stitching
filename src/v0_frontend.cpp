// ============================================================================
// CSC14120 - Parallel Programming | Final Project: Panorama Image Stitching
// Module: v0_frontend.cpp (Sequential Stages 1, 2, 3)
// Author: Vu Tran Phuc - 23120333 (Member 2)
// Lead Role: Front-end & Computer Vision Core Lead
//
// References & Citations:
// 1. Rosten, E., & Drummond, T. (2006). Machine learning for high-speed corner detection. ECCV.
// 2. Calonder, M., et al. (2010). BRIEF: Binary Robust Independent Elementary Features. ECCV.
// 3. ITU-R Recommendation BT.601: Studio encoding parameters of digital television.
// ============================================================================

#include "versions.h"
#include "brief_pattern.h"
#include <cmath>
#include <algorithm>
#include <cstring>

// ----------------------------------------------------------------------------
// STAGE 1: PREPROCESSING (RGB TO GRAYSCALE & GAUSSIAN BLUR)
// ----------------------------------------------------------------------------

ImageGray v0_rgb_to_gray(const ImageRGB& img) {
    ImageGray gray;
    gray.width = img.width;
    gray.height = img.height;
    gray.data.resize(img.width * img.height);

    const int total_pixels = img.width * img.height;
    for (int i = 0; i < total_pixels; ++i) {
        uint8_t r = img.data[3 * i + 0];
        uint8_t g = img.data[3 * i + 1];
        uint8_t b = img.data[3 * i + 2];
        // Rec. 601 standard luma formula: Y = 0.299 R + 0.587 G + 0.114 B
        gray.data[i] = static_cast<uint8_t>(0.299f * r + 0.587f * g + 0.114f * b);
    }
    return gray;
}

ImageGray v0_gaussian_blur(const ImageGray& gray, float sigma) {
    ImageGray blurred = gray;
    const int w = gray.width;
    const int h = gray.height;
    if (w <= 0 || h <= 0) return blurred;

    // 1D Gaussian Kernel generation (size = 5)
    const int kernel_radius = cfg::kBlurRadius;
    const int kernel_size = 2 * kernel_radius + 1;
    float kernel[2 * cfg::kBlurRadius + 1];
    float sum = 0.0f;
    for (int i = -kernel_radius; i <= kernel_radius; ++i) {
        float val = std::exp(-(i * i) / (2.0f * sigma * sigma));
        kernel[i + kernel_radius] = val;
        sum += val;
    }
    for (int i = 0; i < kernel_size; ++i) kernel[i] /= sum;

    // Separable 1D Gaussian Blur: Pass 1 - Horizontal
    std::vector<float> temp(w * h, 0.0f);
    for (int r = 0; r < h; ++r) {
        for (int c = 0; c < w; ++c) {
            float acc = 0.0f;
            for (int k = -kernel_radius; k <= kernel_radius; ++k) {
                int sc = std::min(std::max(c + k, 0), w - 1);
                acc += kernel[k + kernel_radius] * gray.data[r * w + sc];
            }
            temp[r * w + c] = acc;
        }
    }

    // Pass 2 - Vertical
    for (int r = 0; r < h; ++r) {
        for (int c = 0; c < w; ++c) {
            float acc = 0.0f;
            for (int k = -kernel_radius; k <= kernel_radius; ++k) {
                int sr = std::min(std::max(r + k, 0), h - 1);
                acc += kernel[k + kernel_radius] * temp[sr * w + c];
            }
            blurred.data[r * w + c] = static_cast<uint8_t>(std::min(std::max(acc, 0.0f), 255.0f));
        }
    }

    return blurred;
}

// ----------------------------------------------------------------------------
// STAGE 2: KEYPOINT DETECTION (FAST / HARRIS CORNER + NMS + TOP-K)
// ----------------------------------------------------------------------------

std::vector<Keypoint> v0_detect_keypoints(const ImageGray& gray, int max_keypoints) {
    std::vector<Keypoint> keypoints;
    const int w = gray.width;
    const int h = gray.height;
    // BRIEF uses a 31x31 patch. Detect only points that can later receive a
    // valid descriptor so keypoint and descriptor indices stay aligned.
    const int descriptor_border = cfg::kDescriptorBorder;
    if (w <= 2 * descriptor_border || h <= 2 * descriptor_border) return keypoints;

    // FAST-9 Circle Offsets (16 perimeter pixels relative to center)
    const int circle_offsets[16][2] = {
        {0, 3}, {1, 3}, {2, 2}, {3, 1}, {3, 0}, {3, -1}, {2, -2}, {1, -3},
        {0, -3}, {-1, -3}, {-2, -2}, {-3, -1}, {-3, 0}, {-3, 1}, {-2, 2}, {-1, 3}
    };
    const int threshold = cfg::kFastThreshold; // Intensity difference threshold

    std::vector<float> corner_scores(w * h, 0.0f);

    // Step 2.1: FAST corner test for each interior pixel
    for (int r = descriptor_border; r < h - descriptor_border; ++r) {
        for (int c = descriptor_border; c < w - descriptor_border; ++c) {
            uint8_t p = gray.data[r * w + c];
            int lower = std::max(0, p - threshold);
            int upper = std::min(255, p + threshold);

            // Fast rejection using 4 compass points: 0, 4, 8, 12
            int p0  = gray.data[(r + circle_offsets[0][1]) * w + (c + circle_offsets[0][0])];
            int p4  = gray.data[(r + circle_offsets[4][1]) * w + (c + circle_offsets[4][0])];
            int p8  = gray.data[(r + circle_offsets[8][1]) * w + (c + circle_offsets[8][0])];
            int p12 = gray.data[(r + circle_offsets[12][1]) * w + (c + circle_offsets[12][0])];

            int count_b = (p0 > upper) + (p4 > upper) + (p8 > upper) + (p12 > upper);
            int count_d = (p0 < lower) + (p4 < lower) + (p8 < lower) + (p12 < lower);
            // A valid arc of 9 pixels can contain only two of the four compass
            // samples. Requiring three (the common FAST-12 shortcut) rejects
            // valid FAST-9 corners.
            if (count_b < 2 && count_d < 2) continue;

            // Full 16-point circle evaluation
            int circle_vals[16];
            for (int k = 0; k < 16; ++k) {
                circle_vals[k] = gray.data[(r + circle_offsets[k][1]) * w + (c + circle_offsets[k][0])];
            }

            int max_consec_brighter = 0;
            int max_consec_darker = 0;
            for (int start = 0; start < 16; ++start) {
                int cb = 0, cd = 0;
                for (int len = 0; len < 16; ++len) {
                    int val = circle_vals[(start + len) % 16];
                    if (val > upper) { cb++; cd = 0; }
                    else if (val < lower) { cd++; cb = 0; }
                    else { cb = 0; cd = 0; }
                    if (cb > max_consec_brighter) max_consec_brighter = cb;
                    if (cd > max_consec_darker) max_consec_darker = cd;
                }
            }

            if (max_consec_brighter >= 9 || max_consec_darker >= 9) {
                float score = 0.0f;
                for (int k = 0; k < 16; ++k) {
                    score += std::abs(circle_vals[k] - p);
                }
                corner_scores[r * w + c] = score;
            }
        }
    }

    // Step 2.2: Non-Maximum Suppression (NMS) in 3x3 window
    for (int r = descriptor_border; r < h - descriptor_border; ++r) {
        for (int c = descriptor_border; c < w - descriptor_border; ++c) {
            float center_score = corner_scores[r * w + c];
            if (center_score <= 0.0f) continue;

            bool is_max = true;
            for (int dr = -1; dr <= 1 && is_max; ++dr) {
                for (int dc = -1; dc <= 1; ++dc) {
                    if (dr == 0 && dc == 0) continue;
                    if (corner_scores[(r + dr) * w + (c + dc)] >= center_score) {
                        is_max = false;
                        break;
                    }
                }
            }
            if (is_max) {
                keypoints.emplace_back(static_cast<float>(c), static_cast<float>(r), center_score);
            }
        }
    }

    // Step 2.3: Top-K sorting and retention
    const size_t keep = std::min(keypoints.size(), static_cast<size_t>(std::max(max_keypoints, 0)));
    std::partial_sort(keypoints.begin(), keypoints.begin() + keep, keypoints.end(),
                      [](const Keypoint& a, const Keypoint& b) {
                          if (a.response != b.response) return a.response > b.response;
                          if (a.y != b.y) return a.y < b.y;
                          return a.x < b.x;
                      });
    keypoints.resize(keep);

    return keypoints;
}

// ----------------------------------------------------------------------------
// STAGE 3: FEATURE DESCRIPTION (BRIEF 256-BIT BINARY DESCRIPTOR)
// ----------------------------------------------------------------------------

std::vector<Descriptor256> v0_compute_descriptors(const ImageGray& gray,
                                                  const std::vector<Keypoint>& keypoints) {
    std::vector<Descriptor256> descriptors(keypoints.size());
    const int w = gray.width;
    const int h = gray.height;

    for (size_t i = 0; i < keypoints.size(); ++i) {
        const auto& kp = keypoints[i];
        int cx = static_cast<int>(std::round(kp.x));
        int cy = static_cast<int>(std::round(kp.y));
        Descriptor256& desc = descriptors[i];
        std::memset(desc.data, 0, 32);

        // Detection already enforces this border. Keep the guard for callers
        // that construct keypoints themselves.
        if (cx < cfg::kDescriptorBorder || cx >= w - cfg::kDescriptorBorder ||
            cy < cfg::kDescriptorBorder || cy >= h - cfg::kDescriptorBorder) {
            continue;
        }

        // Compute 256 binary intensity comparisons
        for (int b = 0; b < 256; ++b) {
            const int8_t* pair = kBriefPattern[b];
            int p1_val = gray.data[(cy + pair[1]) * w + (cx + pair[0])];
            int p2_val = gray.data[(cy + pair[3]) * w + (cx + pair[2])];
            if (p1_val < p2_val) {
                desc.data[b / 8] |= (1 << (b % 8));
            }
        }
    }

    return descriptors;
}
