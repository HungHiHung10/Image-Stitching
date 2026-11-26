#pragma once
#include "gpu_frontend.cuh"

class V1Pipeline {
public:
    bool run_v1(const ImageRGB& imgA, const ImageRGB& imgB, ImageRGB& panorama,
                StageTimings& timings, Homography3x3* outH = nullptr, PipelineTrace* trace = nullptr,
                bool compact_seam = false);

private:
    struct Stream {
        Stream() { CHECK_CUDA(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
        ~Stream() { cudaStreamDestroy(value); }
        Stream(const Stream&) = delete;
        Stream& operator=(const Stream&) = delete;
        cudaStream_t value{};
    };

    Stream stream_;
    GpuTimer h2d_, preprocess_, keypoints_, descriptors_, matching_, ransac_, blend_, d2h_;
    GpuFrontend frontend_;
    GpuFrontendImage a_, b_;
    GpuBackend backend_;
};
