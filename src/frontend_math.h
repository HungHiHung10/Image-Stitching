#pragma once
#include "pipeline_config.h"
#include <cmath>

struct GaussianWeights {
    float values[2 * cfg::kBlurRadius + 1];
};

inline GaussianWeights gaussian_weights(float sigma) {
    GaussianWeights weights;
    float sum = 0.0f;
    for (int i = -cfg::kBlurRadius; i <= cfg::kBlurRadius; ++i) {
        const float value = std::exp(-(i * i) / (2.0f * sigma * sigma));
        weights.values[i + cfg::kBlurRadius] = value;
        sum += value;
    }
    for (float& value : weights.values) value /= sum;
    return weights;
}
