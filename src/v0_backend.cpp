// ============================================================================
// CSC14120 - Parallel Programming | Final Project: Panorama Image Stitching
// Module: v0_backend.cpp (Sequential Stages 4, 5, 6)
// Author: Tran Nguyen Minh Quan - 23120342 (Member 1)
// Lead Role: Back-end & Algorithmic Core Lead
//
// References & Citations:
// 1. Lowe, D. G. (2004). Distinctive image features from scale-invariant keypoints. IJCV.
// 2. Fischler, M. A., & Bolles, R. C. (1981). RANSAC for model fitting. CACM.
// 3. Hartley, R., & Zisserman, A. (2003). Multiple View Geometry in Computer Vision.
// ============================================================================

#include "versions.h"
#include "logging.h"
#include "hd_backend.h"
#include <cmath>
#include <algorithm>
#include <cstring>
#include <limits>

// ----------------------------------------------------------------------------
// STAGE 4: FEATURE MATCHING (BRUTE-FORCE HAMMING & LOWE'S RATIO TEST)
// ----------------------------------------------------------------------------

std::vector<Match> v0_match_descriptors(const std::vector<Descriptor256>& descA,
                                       const std::vector<Descriptor256>& descB,
                                       float ratio_threshold) {
    std::vector<Match> matches;
    if (descA.empty() || descB.empty()) return matches;

    matches.reserve(descA.size());
    std::vector<int> best_b(descA.size(), -1);
    std::vector<int> best_distance(descA.size(), 256);
    std::vector<int> second_distance(descA.size(), 256);
    for (size_t i = 0; i < descA.size(); ++i) {
        for (size_t j = 0; j < descB.size(); ++j) {
            hd::match_keep_top2(hd::hamming256(descA[i].data, descB[j].data), static_cast<int>(j),
                                best_distance[i], second_distance[i], best_b[i]);
        }
    }

    // Reverse nearest-neighbour lookup. Keeping only mutual pairs rejects many
    // accidental matches on repeated or weakly textured regions.
    std::vector<int> reverse_best_a(descB.size(), -1);
    std::vector<int> reverse_best_distance(descB.size(), 256);
    for (size_t j = 0; j < descB.size(); ++j) {
        for (size_t i = 0; i < descA.size(); ++i) {
            hd::match_keep_best(hd::hamming256(descB[j].data, descA[i].data), static_cast<int>(i),
                                reverse_best_distance[j], reverse_best_a[j]);
        }
    }

    for (size_t i = 0; i < descA.size(); ++i) {
        int j = best_b[i];
        bool passes_ratio = hd::match_passes_ratio(j, best_distance[i], second_distance[i], ratio_threshold);
        if (passes_ratio && reverse_best_a[j] == static_cast<int>(i)) {
            matches.emplace_back(static_cast<int>(i), j, best_distance[i]);
        }
    }

    return matches;
}

// ----------------------------------------------------------------------------
// STAGE 5: HOMOGRAPHY ESTIMATION VIA RANSAC 4-POINT DLT & REFINEMENT
// ----------------------------------------------------------------------------

static bool invert_homography(const Homography3x3& H, Homography3x3& invH) {
    return hd::invert_h(H.h, invH.h);
}

bool v0_estimate_homography_ransac(const std::vector<Keypoint>& kptsA,
                                   const std::vector<Keypoint>& kptsB,
                                   const std::vector<Match>& matches,
                                   Homography3x3& bestH,
                                   std::vector<int>& best_inliers,
                                   int max_iters,
                                   float inlier_thresh) {
    if (matches.size() < 4) return false;

    int max_inliers = 0;
    const float thresh_sq = inlier_thresh * inlier_thresh;

    auto collect_inliers = [&](const Homography3x3& H) {
        std::vector<int> result;
        Homography3x3 H_inv;
        if (!invert_homography(H, H_inv)) return result;
        result.reserve(matches.size());
        for (size_t k = 0; k < matches.size(); ++k) {
            const auto& m = matches[k];
            double symmetric_error = hd::symmetric_error_sq(
                H.h, H_inv.h, kptsA[m.queryIdx].x, kptsA[m.queryIdx].y,
                kptsB[m.trainIdx].x, kptsB[m.trainIdx].y);
            if (hd::is_inlier(symmetric_error, thresh_sq))
                result.push_back(static_cast<int>(k));
        }
        return result;
    };

    for (int iter = 0; iter < max_iters; ++iter) {
        // Sample 4 unique random matches
        int idxs[4];
        if (!hd::ransac_sample(cfg::kRansacSeed, static_cast<uint32_t>(iter),
                               static_cast<uint32_t>(matches.size()), idxs)) continue;

        float src[8], dst[8];
        for (int i = 0; i < 4; ++i) {
            const auto& m = matches[idxs[i]];
            src[2*i] = kptsB[m.trainIdx].x;
            src[2*i+1] = kptsB[m.trainIdx].y;
            dst[2*i] = kptsA[m.queryIdx].x;
            dst[2*i+1] = kptsA[m.queryIdx].y;
        }

        Homography3x3 H_candidate;
        if (!hd::dlt_fit(src, dst, 4, H_candidate.h)) continue;
        std::vector<int> current_inliers = collect_inliers(H_candidate);

        if (static_cast<int>(current_inliers.size()) > max_inliers) {
            max_inliers = static_cast<int>(current_inliers.size());
            best_inliers = current_inliers;
            bestH = H_candidate;
        }
    }
    if (max_inliers < 4) return false;

    // Refit with every inlier instead of retaining the four-point hypothesis.
    // A second collection updates the final consensus after least squares.
    for (int pass=0; pass<cfg::kRefitPasses; ++pass) {
        std::vector<float> src, dst;
        src.reserve(2 * best_inliers.size()); dst.reserve(2 * best_inliers.size());
        for (int index : best_inliers) {
            const auto& m = matches[index];
            src.push_back(kptsB[m.trainIdx].x);
            src.push_back(kptsB[m.trainIdx].y);
            dst.push_back(kptsA[m.queryIdx].x);
            dst.push_back(kptsA[m.queryIdx].y);
        }
        Homography3x3 refined;
        if (!hd::dlt_fit(src.data(), dst.data(), static_cast<int>(best_inliers.size()), refined.h)) break;
        std::vector<int> refined_inliers = collect_inliers(refined);
        if (refined_inliers.size() < 4) break;
        bestH = refined;
        best_inliers.swap(refined_inliers);
    }
    return best_inliers.size() >= 4;
}

// ----------------------------------------------------------------------------
// STAGE 6: WARPING & LINEAR FEATHERING BLENDING
// ----------------------------------------------------------------------------

ImageRGB v0_warp_and_blend(const ImageRGB& imgA,
                           const ImageRGB& imgB,
                           const Homography3x3& H_BtoA,
                           BlendInfo* info) {
    ImageRGB panorama;
    if (info) *info = BlendInfo{};

    // 1. Calculate bounding box of the transformed Image B in Image A coordinates
    const hd::CanvasLayout layout = hd::canvas_layout(H_BtoA.h, imgA.width, imgA.height,
                                                      imgB.width, imgB.height);
    const int pano_w = layout.width;
    const int pano_h = layout.height;
    const float offset_x = layout.offset_x;
    const float offset_y = layout.offset_y;

    panorama.width = pano_w;
    panorama.height = pano_h;
    panorama.channels = 3;
    panorama.data.assign(size_t(pano_w) * pano_h * 3, 0);
    if (info) {
        info->canvas_w = pano_w;
        info->canvas_h = pano_h;
        info->offset_x = offset_x;
        info->offset_y = offset_y;
    }

    // Compute H_inv to project from panorama (Image A coords) back to Image B
    Homography3x3 H_inv;
    if (!invert_homography(H_BtoA, H_inv)) return panorama;
    const hd::RgbView viewA{imgA.data.data(), imgA.width, imgA.height};
    const hd::RgbView viewB{imgB.data.data(), imgB.width, imgB.height};

    // Estimate a conservative per-channel affine exposure transform A ~= gB+b
    // from non-dark, non-saturated overlap pixels.  It is accepted only when
    // it improves overlap MAE by at least 10%, so parallax cannot accidentally
    // trigger a destructive colour correction.
    double sx[3] = {}, sy[3] = {}, sxx[3] = {}, sxy[3] = {};
    size_t exposure_samples = 0;
    for (int py = 0; py < pano_h; py += cfg::kExposureStride) {
        for (int px = 0; px < pano_w; px += cfg::kExposureStride) {
            float ax = px - offset_x, ay = py - offset_y;
            float a[3], b[3];
            if (!hd::overlap_samples(viewA, viewB, H_inv.h, ax, ay, a, b) ||
                !hd::exposure_usable(a, b)) continue;
            ++exposure_samples;
            for (int c=0;c<3;++c) {
                sx[c]+=b[c]; sy[c]+=a[c]; sxx[c]+=b[c]*b[c]; sxy[c]+=b[c]*a[c];
            }
        }
    }
    double gain[3], bias[3];
    hd::exposure_fit(exposure_samples, sx, sy, sxx, sxy, gain, bias);
    double raw_error=0, corrected_error=0;
    size_t error_samples=0;
    for (int py = 0; py < pano_h; py += cfg::kExposureStride) {
        for (int px = 0; px < pano_w; px += cfg::kExposureStride) {
            float ax=px-offset_x, ay=py-offset_y, a[3], b[3];
            if (!hd::overlap_samples(viewA, viewB, H_inv.h, ax, ay, a, b)) continue;
            for (int c=0;c<3;++c) {
                raw_error+=hd::raw_abs_error(a[c], b[c]);
                corrected_error+=hd::corrected_abs_error(a[c], b[c], gain[c], bias[c]);
            }
            error_samples+=3;
        }
    }
    const hd::ExposureDecision decision = hd::exposure_decide(exposure_samples, error_samples,
                                                              raw_error, corrected_error, gain, bias);
    const bool compensate_exposure = decision.compensate;
    const double overlap_mae = decision.overlap_mae;

    // A high residual after exposure compensation indicates parallax or moving
    // content.  In that case find a minimum-error top-to-bottom seam, then use
    // only a narrow feather band. This keeps foliage sharper than blending the
    // complete overlap. Parent directions need one signed byte per canvas pixel.
    const bool use_seam = decision.use_seam;
    log_message(LogLevel::Process) << "Blend: exposure="
        << (compensate_exposure ? "affine" : "unchanged")
        << ", overlap MAE=" << overlap_mae
        << ", seam=" << (use_seam ? "optimal" : "feather") << "\n";
    std::vector<int> seam_x(pano_h,-1);
    if (use_seam) {
        const double inf=std::numeric_limits<double>::infinity();
        std::vector<double> previous(pano_w,inf), current(pano_w,inf);
        std::vector<int8_t> parent(size_t(pano_w)*pano_h,0);
        int first_y=-1,last_y=-1;
        for (int py=0;py<pano_h;++py) {
            bool any=false;
            std::fill(current.begin(),current.end(),inf);
            for (int px=0;px<pano_w;++px) {
                float ax=px-offset_x,ay=py-offset_y,a[3],b[3];
                if (!hd::overlap_samples(viewA, viewB, H_inv.h, ax, ay, a, b)) continue;
                for (int c=0;c<3;++c) b[c]=static_cast<float>(hd::exposure_apply(gain[c], bias[c], b[c]));
                any=true;
                double pixel_cost=hd::seam_pixel_cost(hd::abs_diff_sum(a, b));
                if (first_y<0) current[px]=pixel_cost;
                else {
                    int direction;
                    double best=hd::seam_best_previous(previous.data(), pano_w, px, direction);
                    if (std::isfinite(best)) {
                        current[px]=best+pixel_cost;
                        parent[size_t(py)*pano_w+px]=static_cast<int8_t>(direction);
                    }
                }
            }
            if (any) {
                if (first_y<0) first_y=py;
                last_y=py;
                previous.swap(current);
            }
        }
        if (last_y>=first_y) {
            int x=static_cast<int>(std::min_element(previous.begin(),previous.end())-previous.begin());
            for (int py=last_y;py>=first_y;--py) {
                seam_x[py]=x;
                x+=parent[size_t(py)*pano_w+x];
                x=hd::seam_clamp_x(x, pano_w);
            }
        }
    }

    // Inverse warping & feathering blending
    for (int py = 0; py < pano_h; ++py) {
        const int seam = use_seam ? seam_x[py] : -1;
        for (int px = 0; px < pano_w; ++px) {
            hd::warp_blend_pixel(viewA, viewB, H_inv.h, offset_x, offset_y, gain, bias, seam, px, py,
                                 &panorama.data[(size_t(py) * pano_w + px) * 3]);
        }
    }

    if (info) {
        info->exposure_applied = compensate_exposure;
        info->overlap_mae = overlap_mae;
        info->seam_used = use_seam;
        for (int c = 0; c < 3; ++c) {
            info->gain[c] = gain[c];
            info->bias[c] = bias[c];
        }
        info->seam_x = seam_x;
    }
    return panorama;
}
