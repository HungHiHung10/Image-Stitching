#include "gpu_pipeline.cuh"
#include "versions.h"
#include <limits>

bool V1Pipeline::run_v1(const ImageRGB& imgA, const ImageRGB& imgB, ImageRGB& panorama,
                        StageTimings& timings, Homography3x3* outH, PipelineTrace* trace,
                        bool compact_seam) {
    timings = StageTimings{};
    if (trace) *trace = PipelineTrace{};
    panorama.width = panorama.height = 0;
    panorama.data.clear();
    for (const auto* image : {&imgA, &imgB}) {
        if (image->width <= 0 || image->height <= 0 || image->channels != 3 ||
            image->width > std::numeric_limits<int>::max() / 3 / image->height ||
            image->data.size() != size_t(image->width) * image->height * 3) {
            log_message(LogLevel::Error) << "Invalid RGB input\n";
            return false;
        }
    }

    CpuTimer clock;
    clock.start();
    const auto stream = stream_.value;
    a_.width = imgA.width;
    a_.height = imgA.height;
    b_.width = imgB.width;
    b_.height = imgB.height;
    h2d_.start(stream);
    a_.rgb.upload(imgA.data, stream);
    b_.rgb.upload(imgB.data, stream);
    h2d_.stop(stream);

    preprocess_.start(stream);
    frontend_.preprocess(a_, stream);
    frontend_.preprocess(b_, stream);
    preprocess_.stop(stream);
    keypoints_.start(stream);
    frontend_.detect(a_, stream);
    frontend_.detect(b_, stream);
    keypoints_.stop(stream);
    log_message(LogLevel::Process) << "Keypoints: A=" << a_.count << ", B=" << b_.count << '\n';

    descriptors_.start(stream);
    frontend_.describe(a_, stream);
    frontend_.describe(b_, stream);
    descriptors_.stop(stream);

    const auto features_a = a_.features(), features_b = b_.features();
    matching_.start(stream);
    const int count = backend_.match(features_a, features_b, stream);
    matching_.stop(stream);
    log_message(LogLevel::Process) << "Mutual ratio matches: " << count << '\n';
    if (!check_match_count(count)) return false;

    Homography3x3 h;
    int inliers = 0;
    ransac_.start(stream);
    const bool ransac_ok = backend_.estimate_homography(features_a, features_b, backend_.matches(),
                                                        count, h, inliers, stream);
    ransac_.stop(stream);
    if (!check_ransac_consensus(ransac_ok, inliers, count)) return false;
    if (!check_canvas_bounds(h, a_.width, a_.height, b_.width, b_.height)) return false;

    blend_.start(stream);
    backend_.warp_and_blend(a_.image(), b_.image(), h, trace ? &trace->blend : nullptr, stream, compact_seam);
    blend_.stop(stream);
    d2h_.start(stream);
    backend_.download_panorama(panorama, stream);
    d2h_.stop(stream);
    timings.d2h_ms = d2h_.elapsed();
    timings.total_pipeline_ms = clock.stop();
    timings.h2d_ms = h2d_.elapsed();
    timings.stage1_preprocess_ms = preprocess_.elapsed();
    timings.stage2_keypoint_ms = keypoints_.elapsed();
    timings.stage3_descriptor_ms = descriptors_.elapsed();
    timings.stage4_matching_ms = matching_.elapsed();
    timings.stage5_ransac_ms = ransac_.elapsed();
    timings.stage6_warp_blend_ms = blend_.elapsed();
    timings.gpu_stage_sum_ms = timings.stage1_preprocess_ms + timings.stage2_keypoint_ms +
        timings.stage3_descriptor_ms + timings.stage4_matching_ms + timings.stage5_ransac_ms +
        timings.stage6_warp_blend_ms;

    if (trace) {
        trace->blurredA = a_.download_blurred(stream);
        trace->blurredB = b_.download_blurred(stream);
        trace->kptsA = a_.keypoints.to_host(a_.count, stream);
        trace->kptsB = b_.keypoints.to_host(b_.count, stream);
        trace->descA = a_.descriptors.to_host(a_.count, stream);
        trace->descB = b_.descriptors.to_host(b_.count, stream);
        trace->matches = backend_.download_matches(count, stream);
        trace->inliers = backend_.download_inliers(inliers, stream);
        trace->H = h;
    }
    if (outH) *outH = h;
    return true;
}
