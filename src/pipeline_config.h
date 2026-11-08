#pragma once
#include <cstddef>

namespace cfg {

constexpr float kBlurSigma = 1.2f;
constexpr int kBlurRadius = 2;
constexpr int kFastThreshold = 20;
constexpr int kDescriptorBorder = 16;
constexpr int kMaxKeypoints = 1500;
constexpr unsigned kBriefSeed = 14120;
constexpr float kBriefSigma = 6.0f;
constexpr int kBriefPatchRadius = 15;

constexpr float kRatioThreshold = 0.75f;

constexpr int kRansacIters = 1500;
constexpr float kInlierThresholdPx = 3.0f;
constexpr unsigned kRansacSeed = 42;
constexpr int kRefitPasses = 2;
constexpr int kMinInliers = 8;
constexpr int kMinInlierRatioDen = 5;

constexpr double kMaxCanvasSide = 32768;
constexpr double kMaxCanvasPixels = 120000000;

constexpr int kExposureStride = 2;
constexpr float kDarkMeanMax = 20.0f;
constexpr float kSaturatedMin = 245.0f;
constexpr std::size_t kMinExposureSamples = 1000;
constexpr double kGainMin = 0.75;
constexpr double kGainMax = 1.33;
constexpr double kBiasLimit = 64.0;
constexpr double kRequiredImprovement = 0.90;

constexpr std::size_t kMinSeamSamples = 3000;
constexpr double kSeamMaeThreshold = 18.0;
constexpr double kSeamPixelCostBias = 1.0;
constexpr int kSeamWindow = 2;
constexpr float kSeamBand = 24.0f;

}
