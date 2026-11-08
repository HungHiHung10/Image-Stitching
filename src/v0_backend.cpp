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
#include "hd_math.h"
#include <cmath>
#include <algorithm>
#include <random>
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
            int d = hd::hamming256(descA[i].data, descB[j].data);
            if (d < best_distance[i]) {
                second_distance[i] = best_distance[i];
                best_distance[i] = d;
                best_b[i] = static_cast<int>(j);
            } else if (d < second_distance[i]) {
                second_distance[i] = d;
            }
        }
    }

    // Reverse nearest-neighbour lookup. Keeping only mutual pairs rejects many
    // accidental matches on repeated or weakly textured regions.
    std::vector<int> reverse_best_a(descB.size(), -1);
    std::vector<int> reverse_best_distance(descB.size(), 256);
    for (size_t j = 0; j < descB.size(); ++j) {
        for (size_t i = 0; i < descA.size(); ++i) {
            int d = hd::hamming256(descB[j].data, descA[i].data);
            if (d < reverse_best_distance[j]) {
                reverse_best_distance[j] = d;
                reverse_best_a[j] = static_cast<int>(i);
            }
        }
    }

    for (size_t i = 0; i < descA.size(); ++i) {
        int j = best_b[i];
        bool passes_ratio = j >= 0 && second_distance[i] > 0 &&
            (static_cast<float>(best_distance[i]) /
             static_cast<float>(second_distance[i])) < ratio_threshold;
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

static inline Point2f project_point(const Homography3x3& H, float x, float y) {
    float ox, oy;
    hd::project_point(H.h, x, y, ox, oy);
    return Point2f(ox, oy);
}

bool v0_estimate_homography_ransac(const std::vector<Keypoint>& kptsA,
                                   const std::vector<Keypoint>& kptsB,
                                   const std::vector<Match>& matches,
                                   Homography3x3& bestH,
                                   std::vector<int>& best_inliers,
                                   int max_iters,
                                   float inlier_thresh) {
    if (matches.size() < 4) return false;

    std::mt19937 rng(cfg::kRansacSeed); // Deterministic seed
    std::uniform_int_distribution<size_t> dist(0, matches.size() - 1);

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
            if (hd::is_finite(symmetric_error) && symmetric_error < thresh_sq)
                result.push_back(static_cast<int>(k));
        }
        return result;
    };

    for (int iter = 0; iter < max_iters; ++iter) {
        // Sample 4 unique random matches
        size_t idxs[4];
        for (int i = 0; i < 4; ++i) {
            bool unique = false;
            while (!unique) {
                idxs[i] = dist(rng);
                unique = true;
                for (int j = 0; j < i; ++j) {
                    if (idxs[i] == idxs[j]) { unique = false; break; }
                }
            }
        }

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

static inline bool sample_bilinear(const ImageRGB& img, float x, float y, float out_rgb[3]) {
    return hd::sample_bilinear(img.data.data(), img.width, img.height, x, y, out_rgb);
}

ImageRGB v0_warp_and_blend(const ImageRGB& imgA,
                           const ImageRGB& imgB,
                           const Homography3x3& H_BtoA) {
    ImageRGB panorama;

    // 1. Calculate bounding box of the transformed Image B in Image A coordinates
    Point2f cornersB[4] = {
        Point2f(0.0f, 0.0f),
        Point2f(static_cast<float>(imgB.width - 1), 0.0f),
        Point2f(static_cast<float>(imgB.width - 1), static_cast<float>(imgB.height - 1)),
        Point2f(0.0f, static_cast<float>(imgB.height - 1))
    };

    float min_x = 0.0f, max_x = static_cast<float>(imgA.width - 1);
    float min_y = 0.0f, max_y = static_cast<float>(imgA.height - 1);

    for (int i = 0; i < 4; ++i) {
        Point2f p = project_point(H_BtoA, cornersB[i].x, cornersB[i].y);
        min_x = std::min(min_x, p.x);
        max_x = std::max(max_x, p.x);
        min_y = std::min(min_y, p.y);
        max_y = std::max(max_y, p.y);
    }

    int pano_w = static_cast<int>(std::ceil(max_x - min_x + 1));
    int pano_h = static_cast<int>(std::ceil(max_y - min_y + 1));
    float offset_x = -min_x;
    float offset_y = -min_y;

    panorama.width = pano_w;
    panorama.height = pano_h;
    panorama.channels = 3;
    panorama.data.assign(pano_w * pano_h * 3, 0);

    // Compute H_inv to project from panorama (Image A coords) back to Image B
    Homography3x3 H_inv;
    if (!invert_homography(H_BtoA, H_inv)) return panorama;

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
            Point2f pB = project_point(H_inv, ax, ay);
            if (!sample_bilinear(imgA, ax, ay, a) ||
                !sample_bilinear(imgB, pB.x, pB.y, b)) continue;
            float mean_a=(a[0]+a[1]+a[2])/3.0f, mean_b=(b[0]+b[1]+b[2])/3.0f;
            float max_a=std::max(a[0],std::max(a[1],a[2]));
            float max_b=std::max(b[0],std::max(b[1],b[2]));
            if (mean_a <= cfg::kDarkMeanMax || mean_b <= cfg::kDarkMeanMax ||
                max_a >= cfg::kSaturatedMin || max_b >= cfg::kSaturatedMin) continue;
            ++exposure_samples;
            for (int c=0;c<3;++c) {
                sx[c]+=b[c]; sy[c]+=a[c]; sxx[c]+=b[c]*b[c]; sxy[c]+=b[c]*a[c];
            }
        }
    }
    double gain[3]={1,1,1}, bias[3]={0,0,0};
    if (exposure_samples >= cfg::kMinExposureSamples) {
        double n=static_cast<double>(exposure_samples);
        for (int c=0;c<3;++c) {
            double denominator=n*sxx[c]-sx[c]*sx[c];
            if (std::abs(denominator)>1e-9) {
                gain[c]=std::max(cfg::kGainMin,std::min(cfg::kGainMax,(n*sxy[c]-sx[c]*sy[c])/denominator));
                bias[c]=std::max(-cfg::kBiasLimit,std::min(cfg::kBiasLimit,(sy[c]-gain[c]*sx[c])/n));
            }
        }
    }
    double raw_error=0, corrected_error=0;
    size_t error_samples=0;
    for (int py = 0; py < pano_h; py += cfg::kExposureStride) {
        for (int px = 0; px < pano_w; px += cfg::kExposureStride) {
            float ax=px-offset_x, ay=py-offset_y, a[3], b[3];
            Point2f pB=project_point(H_inv,ax,ay);
            if (!sample_bilinear(imgA,ax,ay,a) || !sample_bilinear(imgB,pB.x,pB.y,b)) continue;
            for (int c=0;c<3;++c) {
                raw_error+=std::abs(a[c]-b[c]);
                corrected_error+=std::abs(a[c]-std::max(0.0,std::min(255.0,gain[c]*b[c]+bias[c])));
            }
            error_samples+=3;
        }
    }
    bool compensate_exposure=exposure_samples>=cfg::kMinExposureSamples && error_samples>0 &&
        corrected_error < raw_error*cfg::kRequiredImprovement;
    if (!compensate_exposure) {
        for (int c=0;c<3;++c) { gain[c]=1.0; bias[c]=0.0; }
        corrected_error=raw_error;
    }
    double overlap_mae=error_samples ? corrected_error/error_samples : 0.0;

    auto adjust_b = [&](float rgb[3]) {
        for (int c=0;c<3;++c)
            rgb[c]=static_cast<float>(std::max(0.0,std::min(255.0,gain[c]*rgb[c]+bias[c])));
    };

    // A high residual after exposure compensation indicates parallax or moving
    // content.  In that case find a minimum-error top-to-bottom seam, then use
    // only a narrow feather band. This keeps foliage sharper than blending the
    // complete overlap. Parent directions need one signed byte per canvas pixel.
    const bool use_seam = error_samples>cfg::kMinSeamSamples && overlap_mae>cfg::kSeamMaeThreshold;
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
                Point2f pB=project_point(H_inv,ax,ay);
                if (!sample_bilinear(imgA,ax,ay,a) || !sample_bilinear(imgB,pB.x,pB.y,b)) continue;
                adjust_b(b); any=true;
                double pixel_cost=(std::abs(a[0]-b[0])+std::abs(a[1]-b[1])+std::abs(a[2]-b[2]))/3.0+cfg::kSeamPixelCostBias;
                if (first_y<0) current[px]=pixel_cost;
                else {
                    double best=inf; int direction=0;
                    for (int d=-cfg::kSeamWindow;d<=cfg::kSeamWindow;++d) {
                        int q=px+d;
                        if (q>=0 && q<pano_w && previous[q]<best) { best=previous[q]; direction=d; }
                    }
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
                x=std::max(0,std::min(pano_w-1,x));
            }
        }
    }

    // Inverse warping & feathering blending
    for (int py = 0; py < pano_h; ++py) {
        for (int px = 0; px < pano_w; ++px) {
            float ax = px - offset_x;
            float ay = py - offset_y;

            float rgbA[3] = {0.0f, 0.0f, 0.0f};
            bool insideA = sample_bilinear(imgA, ax, ay, rgbA);

            Point2f pB = project_point(H_inv, ax, ay);
            float rgbB[3] = {0.0f, 0.0f, 0.0f};
            bool insideB = sample_bilinear(imgB, pB.x, pB.y, rgbB);
            if (insideB) adjust_b(rgbB);

            int out_idx = (py * pano_w + px) * 3;
            if (insideA && insideB) {
                // Linear feathering: pixels near an image boundary receive a
                // lower weight. Coordinates are evaluated in each image's own
                // frame, so this also works for a projectively warped image B.
                float wA, wB;
                if (use_seam && seam_x[py]>=0) {
                    const float band=cfg::kSeamBand;
                    float alpha=std::max(0.0f,std::min(1.0f,
                        (seam_x[py]+band-px)/(2.0f*band)));
                    wA=alpha; wB=1.0f-alpha;
                } else {
                    wA = std::max(1.0f, std::min(std::min(ax, imgA.width-1.0f-ax),
                                                 std::min(ay, imgA.height-1.0f-ay)));
                    wB = std::max(1.0f, std::min(std::min(pB.x, imgB.width-1.0f-pB.x),
                                                 std::min(pB.y, imgB.height-1.0f-pB.y)));
                }
                float inv_sum = 1.0f / (wA + wB);
                for (int c = 0; c < 3; ++c) {
                    panorama.data[out_idx + c] = static_cast<uint8_t>(
                        (wA * rgbA[c] + wB * rgbB[c]) * inv_sum);
                }
            } else if (insideA) {
                for (int c = 0; c < 3; ++c) panorama.data[out_idx + c] = static_cast<uint8_t>(rgbA[c]);
            } else if (insideB) {
                for (int c = 0; c < 3; ++c) panorama.data[out_idx + c] = static_cast<uint8_t>(rgbB[c]);
            }
        }
    }

    return panorama;
}
