// ============================================================================
// CSC14120 - Parallel Programming | Final Project: Panorama Image Stitching
// Module: v0_stitching.cpp (Sequential Pipeline Orchestrator)
// Author: Nguyen Huu Khanh Hung - 23120271 (Member 3)
// Lead Role: System, Data, Profiling & Notebook Lead
//
// Description:
// Integrates Front-end (Stages 1-3) and Back-end (Stages 4-6) into the V0 pipeline
// with isolated, accurate per-stage CPU timing measurements.
// ============================================================================

#include "versions.h"
#include "logging.h"
#include <algorithm>
#include <iostream>
// Integration validation by Member 3; original code, no external adaptation.

bool run_v0(const ImageRGB& imgA,
            const ImageRGB& imgB,
            ImageRGB& panorama,
            StageTimings& timings,
            Homography3x3* outH,
            PipelineTrace* trace) {
    timings = StageTimings{};
    panorama = ImageRGB{};
    if (trace) *trace = PipelineTrace{};
    for (const auto* image : {&imgA, &imgB}) {
        if (image->width <= 0 || image->height <= 0 || image->channels != 3 ||
            image->data.size() != size_t(image->width) * image->height * 3) {
            log_message(LogLevel::Error) << "Invalid RGB input\n";
            return false;
        }
    }
    CpuTimer timer;
    CpuTimer stage_timer;

    timer.start();

    // ------------------------------------------------------------------------
    // Stage 1: Preprocessing (RGB to Gray + Gaussian Blur)
    // ------------------------------------------------------------------------
    stage_timer.start();
    ImageGray grayA = v0_rgb_to_gray(imgA);
    ImageGray blurA = v0_gaussian_blur(grayA);

    ImageGray grayB = v0_rgb_to_gray(imgB);
    ImageGray blurB = v0_gaussian_blur(grayB);
    timings.stage1_preprocess_ms = stage_timer.stop();

    // ------------------------------------------------------------------------
    // Stage 2: Keypoint Detection (FAST / Harris + NMS + Top-K)
    // ------------------------------------------------------------------------
    stage_timer.start();
    std::vector<Keypoint> kptsA = v0_detect_keypoints(blurA, cfg::kMaxKeypoints);
    std::vector<Keypoint> kptsB = v0_detect_keypoints(blurB, cfg::kMaxKeypoints);
    timings.stage2_keypoint_ms = stage_timer.stop();
    log_message(LogLevel::Process) << "Keypoints: A=" << kptsA.size()
                                   << ", B=" << kptsB.size() << "\n";
    if (trace) {
        trace->kptsA = kptsA;
        trace->kptsB = kptsB;
    }

    // ------------------------------------------------------------------------
    // Stage 3: Feature Description (BRIEF 256-bit Binary Descriptor)
    // ------------------------------------------------------------------------
    stage_timer.start();
    std::vector<Descriptor256> descA = v0_compute_descriptors(blurA, kptsA);
    std::vector<Descriptor256> descB = v0_compute_descriptors(blurB, kptsB);
    timings.stage3_descriptor_ms = stage_timer.stop();
    if (trace) {
        trace->descA = descA;
        trace->descB = descB;
    }

    // ------------------------------------------------------------------------
    // Stage 4: Feature Matching (Hamming Distance + Lowe's Ratio Test)
    // ------------------------------------------------------------------------
    stage_timer.start();
    std::vector<Match> matches = v0_match_descriptors(descA, descB, cfg::kRatioThreshold);
    timings.stage4_matching_ms = stage_timer.stop();
    log_message(LogLevel::Process) << "Mutual ratio matches: " << matches.size() << "\n";
    if (trace) trace->matches = matches;

    if (!check_match_count(matches.size())) return false;

    // ------------------------------------------------------------------------
    // Stage 5: Homography Estimation (RANSAC 4-point DLT)
    // ------------------------------------------------------------------------
    stage_timer.start();
    Homography3x3 H_BtoA;
    std::vector<int> inliers;
    bool ransac_ok = v0_estimate_homography_ransac(kptsA, kptsB, matches, H_BtoA, inliers,
                                                 cfg::kRansacIters, cfg::kInlierThresholdPx);
    timings.stage5_ransac_ms = stage_timer.stop();
    if (trace) {
        trace->inliers = inliers;
        trace->H = H_BtoA;
    }

    if (!check_ransac_consensus(ransac_ok, inliers.size(), matches.size())) return false;
    if (!check_canvas_bounds(H_BtoA, imgA.width, imgA.height, imgB.width, imgB.height)) return false;

    // ------------------------------------------------------------------------
    // Stage 6: Inverse warp + bilinear + exposure compensation + adaptive seam/feather
    // ------------------------------------------------------------------------
    stage_timer.start();
    panorama = v0_warp_and_blend(imgA, imgB, H_BtoA, trace ? &trace->blend : nullptr);
    timings.stage6_warp_blend_ms = stage_timer.stop();

    timings.total_pipeline_ms = timer.stop();
    if (panorama.data.empty()) return false;
    if (outH) *outH = H_BtoA;
    return true;
}

bool check_match_count(size_t match_count) {
    if (match_count >= 4) return true;
    log_message(LogLevel::Error) << "Insufficient feature matches: " << match_count << " (need >= 4)\n";
    return false;
}

bool check_ransac_consensus(bool ransac_ok, size_t inlier_count, size_t match_count) {
    const size_t minimum_inliers = static_cast<size_t>(cfg::kMinInliers);
    const size_t ratio_den = static_cast<size_t>(cfg::kMinInlierRatioDen);
    const bool weak_consensus = match_count == 0 || inlier_count * ratio_den < match_count;
    if (!ransac_ok || inlier_count < minimum_inliers || weak_consensus) {
        log_message(LogLevel::Error) << "RANSAC consensus too weak: " << inlier_count
                                     << "/" << match_count << " inliers (need >= "
                                     << minimum_inliers << " and >= "
                                     << 100 / cfg::kMinInlierRatioDen << "%)\n";
        return false;
    }
    log_message(LogLevel::Success) << "RANSAC inliers: " << inlier_count
                                   << "/" << match_count << "\n";
    return true;
}

bool check_canvas_bounds(const Homography3x3& H_BtoA, int width_a, int height_a,
                         int width_b, int height_b) {
    const double* matrix = H_BtoA.h;
    double determinant = matrix[0]*(matrix[4]*matrix[8]-matrix[5]*matrix[7])
                       - matrix[1]*(matrix[3]*matrix[8]-matrix[5]*matrix[6])
                       + matrix[2]*(matrix[3]*matrix[7]-matrix[4]*matrix[6]);
    if (!std::isfinite(determinant) || std::abs(determinant) < 1e-12) {
        log_message(LogLevel::Error) << "Singular homography\n";
        return false;
    }
    double min_x = 0, min_y = 0, max_x = width_a - 1, max_y = height_a - 1;
    double first_z = 0;
    for (int corner = 0; corner < 4; ++corner) {
        double x = (corner & 1) ? width_b - 1 : 0;
        double y = (corner & 2) ? height_b - 1 : 0;
        const auto& h = H_BtoA.h;
        double z = h[6]*x + h[7]*y + h[8];
        if (corner == 0) first_z = z;
        double px = (h[0]*x + h[1]*y + h[2]) / z;
        double py = (h[3]*x + h[4]*y + h[5]) / z;
        if (!std::isfinite(px) || !std::isfinite(py) || std::abs(z) < 1e-9 || z*first_z <= 0) {
            log_message(LogLevel::Error) << "Unbounded homography canvas\n";
            return false;
        }
        min_x = std::min(min_x, px); max_x = std::max(max_x, px);
        min_y = std::min(min_y, py); max_y = std::max(max_y, py);
    }
    double width = std::ceil(max_x - min_x + 1), height = std::ceil(max_y - min_y + 1);
    if (width > cfg::kMaxCanvasSide || height > cfg::kMaxCanvasSide ||
        width*height > cfg::kMaxCanvasPixels) {
        log_message(LogLevel::Error) << "Implausible canvas; check input overlap and matches\n";
        return false;
    }
    return true;
}
