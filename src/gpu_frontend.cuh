#pragma once
#include "frontend_math.h"
#include "gpu_backend.cuh"

struct GpuFrontendImage {
    GpuImage image() const { return {rgb.data(), width, height}; }
    GpuFeatures features() const { return {keypoints.data(), descriptors.data(), count}; }
    ImageGray download_blurred(cudaStream_t stream) const;

    DeviceBuffer<uint8_t> rgb, blurred;
    DeviceBuffer<Keypoint> keypoints;
    DeviceBuffer<Descriptor256> descriptors;
    int width = 0, height = 0, count = 0;
};

class GpuFrontend {
public:
    void preprocess(GpuFrontendImage& image, cudaStream_t stream);
    void detect(GpuFrontendImage& image, cudaStream_t stream);
    void describe(GpuFrontendImage& image, cudaStream_t stream);

private:
    GaussianWeights weights_ = gaussian_weights(cfg::kBlurSigma);
    DeviceBuffer<uint8_t> gray_;
    DeviceBuffer<float> horizontal_;
    DeviceBuffer<uint16_t> scores_;
    DeviceBuffer<uint64_t> candidates_, sorted_;
    DeviceBuffer<int> candidate_count_;
    DeviceBuffer<uint8_t> sort_scratch_;
};
