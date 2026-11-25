#pragma once
#include "common.h"
#include "device_buffer.cuh"
#include "hd_backend.h"
#include <vector>

struct GpuFeatures {
    const Keypoint* keypoints;
    const Descriptor256* descriptors;
    int count;
};

struct GpuImage {
    const uint8_t* data;
    int width;
    int height;
};

struct GpuModel {
    double h[9];
    double h_inv[9];
};

struct GpuWarpSetup {
    hd::CanvasLayout layout;
    double h_inv[9];
    bool invertible;
};

struct GpuExposureState {
    double gain[3];
    double bias[3];
    unsigned long long exposure_samples;
    unsigned long long error_samples;
    double raw_error;
    double corrected_error;
    double overlap_mae;
    int compensate;
    int use_seam;
};

class GpuBackend {
public:
    int match(const GpuFeatures& a, const GpuFeatures& b, cudaStream_t stream);
    bool estimate_homography(const GpuFeatures& a, const GpuFeatures& b, const Match* matches,
                             int match_count, Homography3x3& h, int& inlier_count, cudaStream_t stream);
    void warp_and_blend(const GpuImage& a, const GpuImage& b, const Homography3x3& h,
                        BlendInfo* info, cudaStream_t stream);

    std::vector<Match> download_matches(int count, cudaStream_t stream) const;
    std::vector<int> download_inliers(int count, cudaStream_t stream) const;
    void download_panorama(ImageRGB& panorama, cudaStream_t stream) const;

    const Match* matches() const { return matches_.data(); }
    const uint8_t* panorama() const { return panorama_.data(); }
    hd::CanvasLayout layout() const { return layout_; }

private:
    enum Counter { kMatchCount, kBestIter, kBestScore, kInlierCount, kRefitStatus, kRefinedCount, kCounterSlots };

    int collect_inliers(const GpuModel* model, int match_count, DeviceBuffer<int>& out, Counter slot,
                        cudaStream_t stream);
    int read_counter(Counter slot, cudaStream_t stream) const;

    DeviceBuffer<int> counters_;
    DeviceBuffer<int> forward_index_;
    DeviceBuffer<int> forward_best_;
    DeviceBuffer<int> forward_second_;
    DeviceBuffer<int> reverse_index_;
    DeviceBuffer<int> flags_;
    DeviceBuffer<Match> matches_;

    DeviceBuffer<float> src_points_;
    DeviceBuffer<float> dst_points_;
    DeviceBuffer<GpuModel> models_;
    DeviceBuffer<int> valid_;
    DeviceBuffer<int> scores_;
    DeviceBuffer<GpuModel> current_model_;
    DeviceBuffer<GpuModel> refined_model_;
    DeviceBuffer<int> inliers_;
    DeviceBuffer<int> refined_inliers_;
    DeviceBuffer<float> inlier_src_;
    DeviceBuffer<float> inlier_dst_;

    DeviceBuffer<double> partial_sums_;
    DeviceBuffer<GpuWarpSetup> warp_setup_;
    DeviceBuffer<unsigned long long> partial_counts_;
    DeviceBuffer<GpuExposureState> exposure_;
    DeviceBuffer<float> seam_cost_;
    DeviceBuffer<int8_t> seam_parent_;
    DeviceBuffer<double> seam_rows_;
    DeviceBuffer<int> seam_x_;
    DeviceBuffer<uint8_t> panorama_;
    hd::CanvasLayout layout_{0, 0, 0.0f, 0.0f};
};
