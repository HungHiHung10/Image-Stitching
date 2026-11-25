#pragma once
#include "hd_math.h"
#include "pipeline_config.h"
#include <cstddef>

namespace hd {

struct RgbView {
    const uint8_t* data;
    int width;
    int height;
};

struct CanvasLayout {
    int width;
    int height;
    float offset_x;
    float offset_y;
};

struct ExposureDecision {
    bool compensate;
    bool use_seam;
    double overlap_mae;
};

HD inline void match_keep_top2(int distance, int index, int& best, int& second, int& best_index) {
    if (distance < best) {
        second = best;
        best = distance;
        best_index = index;
    } else if (distance < second) {
        second = distance;
    }
}

HD inline void match_keep_best(int distance, int index, int& best, int& best_index) {
    if (distance < best) {
        best = distance;
        best_index = index;
    }
}

HD inline bool match_passes_ratio(int best_index, int best, int second, float ratio_threshold) {
    return best_index >= 0 && second > 0 &&
           (static_cast<float>(best) / static_cast<float>(second)) < ratio_threshold;
}

HD inline bool is_inlier(double symmetric_error, float threshold_sq) {
    return is_finite(symmetric_error) && symmetric_error < threshold_sq;
}

HD inline CanvasLayout canvas_layout(const double h[9], int width_a, int height_a,
                                     int width_b, int height_b) {
    const float corner_x[4] = {0.0f, static_cast<float>(width_b - 1),
                               static_cast<float>(width_b - 1), 0.0f};
    const float corner_y[4] = {0.0f, 0.0f, static_cast<float>(height_b - 1),
                               static_cast<float>(height_b - 1)};
    float min_x = 0.0f, max_x = static_cast<float>(width_a - 1);
    float min_y = 0.0f, max_y = static_cast<float>(height_a - 1);
    for (int i = 0; i < 4; ++i) {
        float x, y;
        project_point(h, corner_x[i], corner_y[i], x, y);
        min_x = min_of(min_x, x);
        max_x = max_of(max_x, x);
        min_y = min_of(min_y, y);
        max_y = max_of(max_y, y);
    }
    CanvasLayout layout;
    layout.width = static_cast<int>(ceil_of(max_x - min_x + 1));
    layout.height = static_cast<int>(ceil_of(max_y - min_y + 1));
    layout.offset_x = -min_x;
    layout.offset_y = -min_y;
    return layout;
}

HD inline bool overlap_samples(const RgbView& image_a, const RgbView& image_b, const double h_inv[9],
                               float ax, float ay, float rgb_a[3], float rgb_b[3]) {
    float bx, by;
    project_point(h_inv, ax, ay, bx, by);
    return sample_bilinear(image_a.data, image_a.width, image_a.height, ax, ay, rgb_a) &&
           sample_bilinear(image_b.data, image_b.width, image_b.height, bx, by, rgb_b);
}

HD inline bool exposure_usable(const float rgb_a[3], const float rgb_b[3]) {
    float mean_a = (rgb_a[0] + rgb_a[1] + rgb_a[2]) / 3.0f;
    float mean_b = (rgb_b[0] + rgb_b[1] + rgb_b[2]) / 3.0f;
    float max_a = max_of(rgb_a[0], max_of(rgb_a[1], rgb_a[2]));
    float max_b = max_of(rgb_b[0], max_of(rgb_b[1], rgb_b[2]));
    return !(mean_a <= cfg::kDarkMeanMax || mean_b <= cfg::kDarkMeanMax ||
             max_a >= cfg::kSaturatedMin || max_b >= cfg::kSaturatedMin);
}

HD inline void exposure_fit(std::size_t samples, const double sx[3], const double sy[3],
                            const double sxx[3], const double sxy[3], double gain[3], double bias[3]) {
    for (int c = 0; c < 3; ++c) {
        gain[c] = 1.0;
        bias[c] = 0.0;
    }
    if (samples < cfg::kMinExposureSamples) return;
    double n = static_cast<double>(samples);
    for (int c = 0; c < 3; ++c) {
        double denominator = n * sxx[c] - sx[c] * sx[c];
        if (abs_of(denominator) > 1e-9) {
            gain[c] = max_of(cfg::kGainMin, min_of(cfg::kGainMax, (n * sxy[c] - sx[c] * sy[c]) / denominator));
            bias[c] = max_of(-cfg::kBiasLimit, min_of(cfg::kBiasLimit, (sy[c] - gain[c] * sx[c]) / n));
        }
    }
}

HD inline double exposure_apply(double gain, double bias, float value) {
    return max_of(0.0, min_of(255.0, gain * value + bias));
}

HD inline float raw_abs_error(float a, float b) {
    return abs_of(a - b);
}

HD inline double corrected_abs_error(float a, float b, double gain, double bias) {
    return abs_of(a - exposure_apply(gain, bias, b));
}

HD inline ExposureDecision exposure_decide(std::size_t exposure_samples, std::size_t error_samples,
                                           double raw_error, double corrected_error,
                                           double gain[3], double bias[3]) {
    ExposureDecision decision;
    decision.compensate = exposure_samples >= cfg::kMinExposureSamples && error_samples > 0 &&
                          corrected_error < raw_error * cfg::kRequiredImprovement;
    if (!decision.compensate) {
        for (int c = 0; c < 3; ++c) {
            gain[c] = 1.0;
            bias[c] = 0.0;
        }
        corrected_error = raw_error;
    }
    decision.overlap_mae = error_samples ? corrected_error / error_samples : 0.0;
    decision.use_seam = error_samples > cfg::kMinSeamSamples && decision.overlap_mae > cfg::kSeamMaeThreshold;
    return decision;
}

HD inline float abs_diff_sum(const float rgb_a[3], const float rgb_b[3]) {
    return abs_of(rgb_a[0] - rgb_b[0]) + abs_of(rgb_a[1] - rgb_b[1]) + abs_of(rgb_a[2] - rgb_b[2]);
}

HD inline double seam_pixel_cost(float abs_sum) {
    return abs_sum / 3.0 + cfg::kSeamPixelCostBias;
}

HD inline double seam_best_previous(const double* previous, int width, int px, int& direction) {
    double best = infinity_of();
    direction = 0;
    for (int d = -cfg::kSeamWindow; d <= cfg::kSeamWindow; ++d) {
        int q = px + d;
        if (q >= 0 && q < width && previous[q] < best) {
            best = previous[q];
            direction = d;
        }
    }
    return best;
}

HD inline int seam_clamp_x(int x, int width) {
    return max_of(0, min_of(width - 1, x));
}

HD inline float feather_weight(float x, float y, int width, int height) {
    return max_of(1.0f, min_of(min_of(x, width - 1.0f - x), min_of(y, height - 1.0f - y)));
}

HD inline float seam_weight_a(int seam_x, int px) {
    return max_of(0.0f, min_of(1.0f, (seam_x + cfg::kSeamBand - px) / (2.0f * cfg::kSeamBand)));
}

HD inline void warp_blend_pixel(const RgbView& image_a, const RgbView& image_b, const double h_inv[9],
                                float offset_x, float offset_y, const double gain[3], const double bias[3],
                                int seam_x, int px, int py, uint8_t out[3]) {
    float ax = px - offset_x;
    float ay = py - offset_y;
    float rgb_a[3] = {0.0f, 0.0f, 0.0f};
    bool inside_a = sample_bilinear(image_a.data, image_a.width, image_a.height, ax, ay, rgb_a);
    float bx, by;
    project_point(h_inv, ax, ay, bx, by);
    float rgb_b[3] = {0.0f, 0.0f, 0.0f};
    bool inside_b = sample_bilinear(image_b.data, image_b.width, image_b.height, bx, by, rgb_b);
    if (inside_b) {
        for (int c = 0; c < 3; ++c) rgb_b[c] = static_cast<float>(exposure_apply(gain[c], bias[c], rgb_b[c]));
    }
    if (inside_a && inside_b) {
        float weight_a, weight_b;
        if (seam_x >= 0) {
            weight_a = seam_weight_a(seam_x, px);
            weight_b = 1.0f - weight_a;
        } else {
            weight_a = feather_weight(ax, ay, image_a.width, image_a.height);
            weight_b = feather_weight(bx, by, image_b.width, image_b.height);
        }
        float inv_sum = 1.0f / (weight_a + weight_b);
        for (int c = 0; c < 3; ++c)
            out[c] = static_cast<uint8_t>((weight_a * rgb_a[c] + weight_b * rgb_b[c]) * inv_sum);
    } else if (inside_a) {
        for (int c = 0; c < 3; ++c) out[c] = static_cast<uint8_t>(rgb_a[c]);
    } else if (inside_b) {
        for (int c = 0; c < 3; ++c) out[c] = static_cast<uint8_t>(rgb_b[c]);
    } else {
        for (int c = 0; c < 3; ++c) out[c] = 0;
    }
}

}
