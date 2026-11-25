#include "../gpu_backend.cuh"
#include "../versions.h"
#include "../logging.h"
#include "../stb_image.h"
#include "../stb_image_write.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

class JsonWriter {
public:
    JsonWriter() { out_ << std::setprecision(17); }

    JsonWriter& open(const std::string& key = std::string()) {
        prefix(key);
        out_ << '{';
        first_ = true;
        return *this;
    }

    JsonWriter& close() {
        out_ << '}';
        first_ = false;
        return *this;
    }

    template <typename T>
    JsonWriter& field(const std::string& key, const T& value) {
        prefix(key);
        write(value);
        first_ = false;
        return *this;
    }

    std::string str() const { return out_.str(); }

private:
    void prefix(const std::string& key) {
        if (!first_) out_ << ',';
        if (!key.empty()) out_ << '"' << key << "\":";
    }

    void write(bool value) { out_ << (value ? "true" : "false"); }
    void write(int value) { out_ << value; }
    void write(long long value) { out_ << value; }
    void write(size_t value) { out_ << value; }

    void write(double value) {
        if (std::isfinite(value)) out_ << value;
        else out_ << "null";
    }

    void write(const std::string& value) {
        out_ << '"';
        for (char c : value) {
            if (c == '"' || c == '\\') out_ << '\\' << c;
            else if (c == '\n') out_ << "\\n";
            else out_ << c;
        }
        out_ << '"';
    }

    void write(const char*) = delete;

    void write(const std::vector<double>& values) {
        out_ << '[';
        for (size_t i = 0; i < values.size(); ++i) {
            if (i) out_ << ',';
            write(values[i]);
        }
        out_ << ']';
    }

    std::ostringstream out_;
    bool first_ = true;
};

struct Options {
    std::string left;
    std::string right;
    std::string out_dir;
    int runs = 5;
    int warmup = 1;
};

struct PixelDiff {
    bool same_size = false;
    bool identical = false;
    int max_abs = 0;
    size_t differing = 0;
    size_t above_one = 0;
    size_t total = 0;
};

struct StageTimes {
    std::vector<double> h2d, stage4, stage5, stage6, d2h, total;
};

ImageRGB load_rgb(const std::string& path) {
    ImageRGB image;
    int channels = 0;
    std::unique_ptr<stbi_uc, decltype(&stbi_image_free)> pixels(
        stbi_load(path.c_str(), &image.width, &image.height, &channels, 3), stbi_image_free);
    if (!pixels) throw std::runtime_error("Cannot load " + path + ": " + stbi_failure_reason());
    image.data.assign(pixels.get(), pixels.get() + size_t(image.width) * image.height * 3);
    return image;
}

void save_png(const std::string& path, const ImageRGB& image) {
    if (image.data.empty()) return;
    if (!stbi_write_png(path.c_str(), image.width, image.height, 3, image.data.data(), image.width * 3))
        throw std::runtime_error("Cannot write " + path);
}

void save_h(const std::string& path, const Homography3x3& h) {
    std::ofstream file(path);
    file << std::setprecision(17);
    for (int r = 0; r < 3; ++r) file << h.h[3 * r] << ' ' << h.h[3 * r + 1] << ' ' << h.h[3 * r + 2] << '\n';
    if (!file) throw std::runtime_error("Cannot write " + path);
}

Options parse(int argc, char** argv) {
    if (argc < 4) throw std::runtime_error("Usage: test_backend <left> <right> <out_dir> [--runs N] [--warmup N]");
    Options options;
    options.left = argv[1];
    options.right = argv[2];
    options.out_dir = argv[3];
    for (int i = 4; i < argc; i += 2) {
        const std::string option = argv[i];
        if (i + 1 >= argc) throw std::runtime_error("Missing value for " + option);
        const int value = std::atoi(argv[i + 1]);
        if (option == "--runs" && value >= 1) options.runs = value;
        else if (option == "--warmup" && value >= 0) options.warmup = value;
        else throw std::runtime_error("Invalid option: " + option);
    }
    return options;
}

bool same_matches(const std::vector<Match>& a, const std::vector<Match>& b, long long& first_difference) {
    first_difference = -1;
    const size_t n = std::min(a.size(), b.size());
    for (size_t i = 0; i < n; ++i) {
        if (a[i].queryIdx != b[i].queryIdx || a[i].trainIdx != b[i].trainIdx || a[i].distance != b[i].distance) {
            first_difference = static_cast<long long>(i);
            return false;
        }
    }
    if (a.size() != b.size()) first_difference = static_cast<long long>(n);
    return a.size() == b.size();
}

bool same_bits(const Homography3x3& a, const Homography3x3& b) {
    return std::memcmp(a.h, b.h, sizeof(a.h)) == 0;
}

void projection_error(const Homography3x3& a, const Homography3x3& b, int width, int height,
                      double& mean, double& peak) {
    mean = 0.0;
    peak = 0.0;
    int samples = 0;
    for (int j = 0; j < 12; ++j) {
        for (int i = 0; i < 16; ++i) {
            const double x = 20 + (width - 41) * i / 15.0;
            const double y = 20 + (height - 41) * j / 11.0;
            const double wa = a.h[6] * x + a.h[7] * y + a.h[8];
            const double wb = b.h[6] * x + b.h[7] * y + b.h[8];
            const double dx = (a.h[0] * x + a.h[1] * y + a.h[2]) / wa - (b.h[0] * x + b.h[1] * y + b.h[2]) / wb;
            const double dy = (a.h[3] * x + a.h[4] * y + a.h[5]) / wa - (b.h[3] * x + b.h[4] * y + b.h[5]) / wb;
            const double error = std::sqrt(dx * dx + dy * dy);
            mean += error;
            peak = std::max(peak, error);
            ++samples;
        }
    }
    mean /= samples;
}

PixelDiff compare_pixels(const ImageRGB& a, const ImageRGB& b) {
    PixelDiff diff;
    diff.same_size = a.width == b.width && a.height == b.height && a.data.size() == b.data.size();
    if (!diff.same_size) return diff;
    diff.total = static_cast<size_t>(a.width) * a.height;
    for (size_t p = 0; p < diff.total; ++p) {
        int worst = 0;
        for (int c = 0; c < 3; ++c) worst = std::max(worst, std::abs(int(a.data[3 * p + c]) - int(b.data[3 * p + c])));
        diff.max_abs = std::max(diff.max_abs, worst);
        diff.differing += worst > 0;
        diff.above_one += worst > 1;
    }
    diff.identical = diff.differing == 0;
    return diff;
}

void write_pixels(JsonWriter& json, const std::string& key, const PixelDiff& diff) {
    json.open(key)
        .field("same_size", diff.same_size)
        .field("identical", diff.identical)
        .field("max_abs", diff.max_abs)
        .field("differing", diff.differing)
        .field("above_one", diff.above_one)
        .field("total", diff.total)
        .close();
}

void write_h(JsonWriter& json, const std::string& key, const Homography3x3& v0, const Homography3x3& v1,
             int width_b, int height_b) {
    double mean = 0.0, peak = 0.0;
    projection_error(v0, v1, width_b, height_b, mean, peak);
    json.open(key)
        .field("bitwise", same_bits(v0, v1))
        .field("mean_px", mean)
        .field("max_px", peak)
        .close();
}

bool blend_decisions_equal(const BlendInfo& a, const BlendInfo& b) {
    return a.canvas_w == b.canvas_w && a.canvas_h == b.canvas_h && a.offset_x == b.offset_x &&
           a.offset_y == b.offset_y && a.exposure_applied == b.exposure_applied && a.seam_used == b.seam_used;
}

double max_gain_bias_diff(const BlendInfo& a, const BlendInfo& b) {
    double worst = 0.0;
    for (int c = 0; c < 3; ++c) {
        worst = std::max(worst, std::fabs(a.gain[c] - b.gain[c]));
        worst = std::max(worst, std::fabs(a.bias[c] - b.bias[c]));
    }
    return worst;
}

void write_blend(JsonWriter& json, const std::string& key, const BlendInfo& v0, const BlendInfo& v1) {
    json.open(key)
        .field("canvas_equal", v0.canvas_w == v1.canvas_w && v0.canvas_h == v1.canvas_h &&
                                   v0.offset_x == v1.offset_x && v0.offset_y == v1.offset_y)
        .field("canvas_w", v1.canvas_w)
        .field("canvas_h", v1.canvas_h)
        .field("exposure_equal", v0.exposure_applied == v1.exposure_applied)
        .field("exposure_applied", v1.exposure_applied)
        .field("seam_equal", v0.seam_used == v1.seam_used)
        .field("seam_used", v1.seam_used)
        .field("seam_x_equal", v0.seam_x == v1.seam_x)
        .field("overlap_mae_v0", v0.overlap_mae)
        .field("overlap_mae_v1", v1.overlap_mae)
        .field("overlap_mae_abs_diff", std::fabs(v0.overlap_mae - v1.overlap_mae))
        .field("gain_bias_max_abs_diff", max_gain_bias_diff(v0, v1))
        .field("gain_v1", std::vector<double>(v1.gain, v1.gain + 3))
        .field("bias_v1", std::vector<double>(v1.bias, v1.bias + 3))
        .close();
}

double mean_of(const std::vector<double>& values) {
    double sum = 0.0;
    for (double v : values) sum += v;
    return values.empty() ? 0.0 : sum / values.size();
}

double min_of(const std::vector<double>& values) {
    return values.empty() ? 0.0 : *std::min_element(values.begin(), values.end());
}

void write_times(JsonWriter& json, const std::string& key, const std::vector<double>& values) {
    json.open(key).field("mean_ms", mean_of(values)).field("min_ms", min_of(values)).field("runs", values).close();
}

struct DeviceInputs {
    DeviceBuffer<uint8_t> image_a, image_b;
    DeviceBuffer<Keypoint> keypoints_a, keypoints_b;
    DeviceBuffer<Descriptor256> descriptors_a, descriptors_b;

    void upload(const ImageRGB& a, const ImageRGB& b, const PipelineTrace& trace, cudaStream_t stream) {
        image_a.upload(a.data, stream);
        image_b.upload(b.data, stream);
        keypoints_a.upload(trace.kptsA, stream);
        keypoints_b.upload(trace.kptsB, stream);
        descriptors_a.upload(trace.descA, stream);
        descriptors_b.upload(trace.descB, stream);
    }

    GpuFeatures features_a(const PipelineTrace& trace) const {
        return GpuFeatures{keypoints_a.data(), descriptors_a.data(), static_cast<int>(trace.descA.size())};
    }

    GpuFeatures features_b(const PipelineTrace& trace) const {
        return GpuFeatures{keypoints_b.data(), descriptors_b.data(), static_cast<int>(trace.descB.size())};
    }
};

struct ChainResult {
    bool ok = false;
    std::string failed_check;
    std::vector<Match> matches;
    std::vector<int> inliers;
    Homography3x3 h;
    BlendInfo blend;
    ImageRGB panorama;
};

ChainResult run_chain(GpuBackend& backend, DeviceInputs& inputs, const ImageRGB& a, const ImageRGB& b,
                      const PipelineTrace& trace, cudaStream_t stream, StageTimes* times, bool keep) {
    ChainResult result;
    GpuTimer total, h2d, stage4, stage5, stage6, d2h;
    total.start(stream);
    h2d.start(stream);
    inputs.upload(a, b, trace, stream);
    h2d.stop(stream);
    const GpuFeatures fa = inputs.features_a(trace), fb = inputs.features_b(trace);
    stage4.start(stream);
    const int match_count = backend.match(fa, fb, stream);
    stage4.stop(stream);
    log_message(LogLevel::Process) << "Mutual ratio matches: " << match_count << "\n";
    if (keep) result.matches = backend.download_matches(match_count, stream);
    if (!check_match_count(match_count)) {
        result.failed_check = "matches";
        return result;
    }
    int inlier_count = 0;
    stage5.start(stream);
    const bool ransac_ok = backend.estimate_homography(fa, fb, backend.matches(), match_count, result.h,
                                                       inlier_count, stream);
    stage5.stop(stream);
    if (keep) result.inliers = backend.download_inliers(inlier_count, stream);
    if (!check_ransac_consensus(ransac_ok, inlier_count, match_count)) {
        result.failed_check = "consensus";
        return result;
    }
    if (!check_canvas_bounds(result.h, a.width, a.height, b.width, b.height)) {
        result.failed_check = "canvas";
        return result;
    }
    stage6.start(stream);
    backend.warp_and_blend(GpuImage{inputs.image_a.data(), a.width, a.height},
                           GpuImage{inputs.image_b.data(), b.width, b.height}, result.h,
                           keep ? &result.blend : nullptr, stream);
    stage6.stop(stream);
    d2h.start(stream);
    backend.download_panorama(result.panorama, stream);
    d2h.stop(stream);
    total.stop(stream);
    if (times) {
        times->h2d.push_back(h2d.elapsed());
        times->stage4.push_back(stage4.elapsed());
        times->stage5.push_back(stage5.elapsed());
        times->stage6.push_back(stage6.elapsed());
        times->d2h.push_back(d2h.elapsed());
        times->total.push_back(total.elapsed());
    }
    result.ok = true;
    return result;
}

void check_error_paths() {
    GpuBackend backend;
    const GpuFeatures empty{nullptr, nullptr, 0};
    if (backend.match(empty, empty, nullptr) != 0)
        throw std::runtime_error("Empty descriptors must produce zero matches");

    std::vector<Descriptor256> descriptors(4);
    std::vector<Keypoint> points;
    for (int i = 0; i < 4; ++i) {
        descriptors[i].data[0] = 1u << i;
        points.emplace_back(float(i), 0.0f);
    }
    DeviceBuffer<Keypoint> device_points;
    DeviceBuffer<Descriptor256> device_desc;
    device_points.upload(points, nullptr);
    device_desc.upload(descriptors, nullptr);
    GpuFeatures features{device_points.data(), device_desc.data(), 3};
    Homography3x3 h;
    int inliers = -1;
    int matches = backend.match(features, features, nullptr);
    if (matches != 3 || check_match_count(matches) ||
        backend.estimate_homography(features, features, backend.matches(), matches, h, inliers, nullptr))
        throw std::runtime_error("Three matches must fail homography estimation");

    features.count = 4;
    matches = backend.match(features, features, nullptr);
    if (matches != 4 ||
        backend.estimate_homography(features, features, backend.matches(), matches, h, inliers, nullptr))
        throw std::runtime_error("Collinear keypoints must fail homography estimation");

    Homography3x3 singular, oversized;
    singular.h[4] = 0.0;
    oversized.h[2] = cfg::kMaxCanvasSide;
    if (check_canvas_bounds(singular, 64, 64, 64, 64) ||
        check_canvas_bounds(oversized, 64, 64, 64, 64))
        throw std::runtime_error("Singular H and oversized canvases must be rejected");
    log_message(LogLevel::Success) << "Backend error cases passed\n";
}

int run(const Options& options) {
    const ImageRGB a = load_rgb(options.left), b = load_rgb(options.right);
    log_message(LogLevel::Process) << "Backend test: A=" << a.width << 'x' << a.height << ", B=" << b.width
                                   << 'x' << b.height << ", runs=" << options.runs
                                   << ", warmup=" << options.warmup << '\n';
    ImageRGB v0_panorama;
    StageTimings v0_times;
    Homography3x3 v0_h;
    PipelineTrace trace;
    const bool v0_ok = run_v0(a, b, v0_panorama, v0_times, &v0_h, &trace);
    save_png(options.out_dir + "/panorama_v0.png", v0_panorama);
    if (v0_ok) save_h(options.out_dir + "/H_v0.txt", v0_h);

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));
    GpuBackend backend;
    DeviceInputs inputs;
    inputs.upload(a, b, trace, stream);
    const GpuFeatures fa = inputs.features_a(trace), fb = inputs.features_b(trace);
    const GpuImage image_a{inputs.image_a.data(), a.width, a.height};
    const GpuImage image_b{inputs.image_b.data(), b.width, b.height};

    JsonWriter json;
    json.open().field("left", options.left).field("right", options.right).field("v0_ok", v0_ok);
    json.open("counts")
        .field("keypoints_a", static_cast<int>(trace.kptsA.size()))
        .field("keypoints_b", static_cast<int>(trace.kptsB.size()))
        .field("matches_v0", static_cast<int>(trace.matches.size()))
        .field("inliers_v0", static_cast<int>(trace.inliers.size()))
        .close();

    bool core = true;
    bool pixels_exact = true;
    bool pixels_within_one = true;

    const int match_count = backend.match(fa, fb, stream);
    const std::vector<Match> gpu_matches = backend.download_matches(match_count, stream);
    long long first_match_difference = -1;
    const bool matches_equal = same_matches(trace.matches, gpu_matches, first_match_difference);
    core &= matches_equal;
    json.open("stage4")
        .field("matches_v1", match_count)
        .field("matches_equal", matches_equal)
        .field("first_difference", first_match_difference)
        .close();

    const bool has_stage5 = trace.matches.size() >= 4;
    if (has_stage5) {
        DeviceBuffer<Match> v0_matches;
        v0_matches.upload(trace.matches, stream);
        Homography3x3 h;
        int inlier_count = 0;
        const bool ok = backend.estimate_homography(fa, fb, v0_matches.data(), static_cast<int>(trace.matches.size()),
                                                    h, inlier_count, stream);
        const std::vector<int> inliers = backend.download_inliers(inlier_count, stream);
        const bool inliers_equal = inliers == trace.inliers;
        core &= ok && inliers_equal && same_bits(h, trace.H);
        json.open("stage5").field("ransac_ok", ok).field("inliers_v1", inlier_count).field("inliers_equal", inliers_equal);
        write_h(json, "h", trace.H, h, b.width, b.height);
        json.close();
    }

    if (v0_ok) {
        BlendInfo blend;
        backend.warp_and_blend(image_a, image_b, trace.H, &blend, stream);
        ImageRGB panorama;
        backend.download_panorama(panorama, stream);
        const PixelDiff pixels = compare_pixels(v0_panorama, panorama);
        const bool decisions = blend_decisions_equal(trace.blend, blend);
        core &= decisions && trace.blend.seam_x == blend.seam_x && pixels.same_size;
        pixels_exact &= pixels.identical;
        pixels_within_one &= pixels.max_abs <= 1;
        json.open("stage6");
        write_blend(json, "blend", trace.blend, blend);
        write_pixels(json, "pixels", pixels);
        json.close();
    }

    const ChainResult chain = run_chain(backend, inputs, a, b, trace, stream, nullptr, true);
    long long first_chain_difference = -1;
    const bool chain_matches = same_matches(trace.matches, chain.matches, first_chain_difference);
    const bool chain_inliers = chain.inliers == trace.inliers;
    json.open("chain").field("ok", chain.ok).field("failed_check", chain.failed_check)
        .field("same_outcome", chain.ok == v0_ok)
        .field("matches_equal", chain_matches).field("inliers_equal", chain_inliers);
    core &= chain.ok == v0_ok && chain_matches;
    if (chain.ok && v0_ok) {
        const PixelDiff pixels = compare_pixels(v0_panorama, chain.panorama);
        const bool decisions = blend_decisions_equal(trace.blend, chain.blend);
        core &= chain_inliers && same_bits(chain.h, v0_h) && decisions &&
                trace.blend.seam_x == chain.blend.seam_x && pixels.same_size;
        pixels_exact &= pixels.identical;
        pixels_within_one &= pixels.max_abs <= 1;
        write_h(json, "h", v0_h, chain.h, b.width, b.height);
        write_blend(json, "blend", trace.blend, chain.blend);
        write_pixels(json, "pixels", pixels);
        save_png(options.out_dir + "/panorama_v1.png", chain.panorama);
        save_h(options.out_dir + "/H_v1.txt", chain.h);
    }
    json.close();

    StageTimes times;
    if (chain.ok) {
        for (int i = 0; i < options.warmup; ++i) run_chain(backend, inputs, a, b, trace, stream, nullptr, false);
        for (int i = 0; i < options.runs; ++i) run_chain(backend, inputs, a, b, trace, stream, &times, false);
    }
    json.open("timing_v1");
    write_times(json, "h2d", times.h2d);
    write_times(json, "stage4_matching", times.stage4);
    write_times(json, "stage5_ransac", times.stage5);
    write_times(json, "stage6_warp_blend", times.stage6);
    write_times(json, "d2h", times.d2h);
    write_times(json, "backend_total", times.total);
    json.close();
    json.open("timing_v0_single_run")
        .field("stage4_matching_ms", v0_times.stage4_matching_ms)
        .field("stage5_ransac_ms", v0_times.stage5_ransac_ms)
        .field("stage6_warp_blend_ms", v0_times.stage6_warp_blend_ms)
        .field("total_pipeline_ms", v0_times.total_pipeline_ms)
        .close();
    const bool strict = core && pixels_exact;
    const bool tolerant = core && pixels_within_one;
    json.field("core_pass", core).field("strict_pass", strict).field("tolerant_pass", tolerant).close();

    CHECK_CUDA(cudaStreamDestroy(stream));
    std::ofstream file(options.out_dir + "/result.json");
    file << json.str() << '\n';
    if (!file) throw std::runtime_error("Cannot write result.json");
    std::cout << "RESULT_JSON " << json.str() << '\n';
    log_message(strict ? LogLevel::Success : LogLevel::Warn)
        << "core=" << (core ? "pass" : "fail") << ", strict=" << (strict ? "pass" : "fail")
        << ", tolerant=" << (tolerant ? "pass" : "fail") << '\n';
    return tolerant ? 0 : 1;
}

}

int main(int argc, char** argv) {
    init_logging();
    try {
        if (argc == 2 && std::string(argv[1]) == "--check-errors") {
            check_error_paths();
            return 0;
        }
        return run(parse(argc, argv));
    } catch (const std::exception& error) {
        log_message(LogLevel::Error) << error.what() << '\n';
        return 2;
    }
}
