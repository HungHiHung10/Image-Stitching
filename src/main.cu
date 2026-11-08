// Author: Nguyen Huu Khanh Hung - 23120271 (Member 3). Original CLI; stb I/O: MIT/public domain.
#include "versions.h"
#include "logging.h"
#include "stb_image.h"
#include "stb_image_write.h"
#include <algorithm>
#include <cctype>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <stdexcept>

static const char* names[] = {"preprocess_ms", "keypoint_ms", "descriptor_ms",
    "matching_ms", "ransac_ms", "warp_blend_ms", "total_pipeline_ms"};
static const double StageTimings::* fields[] = {
    &StageTimings::stage1_preprocess_ms, &StageTimings::stage2_keypoint_ms,
    &StageTimings::stage3_descriptor_ms, &StageTimings::stage4_matching_ms,
    &StageTimings::stage5_ransac_ms, &StageTimings::stage6_warp_blend_ms,
    &StageTimings::total_pipeline_ms};
static void usage(const char* name) {
    std::cout << "Usage: " << name << " <left> <right> <output.png|jpg> --version 0\n"
        "  [--runs N] [--warmup N] [--save-h FILE] [--timings-csv FILE] [--dump-dir DIR]\n"
        "Runs must be positive; warmup defaults to 0 and is excluded from statistics.\n"
        "--dump-dir writes keypoints, descriptors, matches, inliers, H and blend data\n"
        "  from one extra untimed run; DIR must exist.\n"
        "H maps right (B) to left (A). Set NO_COLOR to disable colors.\n";
}
static int integer(const std::string& text, int minimum) {
    size_t end = 0;
    int value = std::stoi(text, &end);
    if (end != text.size() || value < minimum)
        throw std::runtime_error("Invalid integer: " + text);
    return value;
}
static ImageRGB load(const std::string& path) {
    ImageRGB image;
    int channels = 0;
    std::unique_ptr<stbi_uc, decltype(&stbi_image_free)> pixels(
        stbi_load(path.c_str(), &image.width, &image.height, &channels, 3), stbi_image_free);
    if (!pixels) throw std::runtime_error("Cannot load " + path + ": " + stbi_failure_reason());
    image.data.assign(pixels.get(), pixels.get() + size_t(image.width) * image.height * 3);
    return image;
}
static void check_stream(std::ofstream& file, const std::string& path) {
    file.flush();
    if (!file) throw std::runtime_error("Cannot write " + path);
}
static void write_h(const std::string& path, const Homography3x3& h) {
    std::ofstream file(path);
    file << std::setprecision(17);
    for (int r = 0; r < 3; ++r)
        file << h.h[3*r] << ' ' << h.h[3*r+1] << ' ' << h.h[3*r+2] << '\n';
    check_stream(file, path);
}
static void write_keypoints(const std::string& path, const std::vector<Keypoint>& keypoints) {
    std::ofstream file(path);
    file << "index,x,y,response\n" << std::setprecision(9);
    for (size_t i = 0; i < keypoints.size(); ++i)
        file << i << ',' << keypoints[i].x << ',' << keypoints[i].y << ','
             << keypoints[i].response << '\n';
    check_stream(file, path);
}
static void write_descriptors(const std::string& path, const std::vector<Descriptor256>& descriptors) {
    static const char digits[] = "0123456789abcdef";
    std::ofstream file(path);
    for (const auto& descriptor : descriptors) {
        for (uint8_t byte : descriptor.data) file << digits[byte >> 4] << digits[byte & 15];
        file << '\n';
    }
    check_stream(file, path);
}
static void write_trace(const std::string& dir, const PipelineTrace& trace) {
    const std::string base = dir + "/";
    write_keypoints(base + "keypoints_A.csv", trace.kptsA);
    write_keypoints(base + "keypoints_B.csv", trace.kptsB);
    write_descriptors(base + "descriptors_A.txt", trace.descA);
    write_descriptors(base + "descriptors_B.txt", trace.descB);
    {
        std::ofstream file(base + "matches.csv");
        file << "index,query,train,distance\n";
        for (size_t i = 0; i < trace.matches.size(); ++i)
            file << i << ',' << trace.matches[i].queryIdx << ',' << trace.matches[i].trainIdx
                 << ',' << trace.matches[i].distance << '\n';
        check_stream(file, base + "matches.csv");
    }
    {
        std::ofstream file(base + "inliers.csv");
        file << "match_index\n";
        for (int index : trace.inliers) file << index << '\n';
        check_stream(file, base + "inliers.csv");
    }
    write_h(base + "H.txt", trace.H);
    {
        const BlendInfo& blend = trace.blend;
        std::ofstream file(base + "blend.json");
        file << std::setprecision(17) << std::boolalpha
             << "{\n  \"exposure_applied\": " << blend.exposure_applied
             << ",\n  \"overlap_mae\": " << blend.overlap_mae
             << ",\n  \"seam_used\": " << blend.seam_used
             << ",\n  \"gain\": [" << blend.gain[0] << ", " << blend.gain[1] << ", " << blend.gain[2]
             << "],\n  \"bias\": [" << blend.bias[0] << ", " << blend.bias[1] << ", " << blend.bias[2]
             << "],\n  \"canvas_w\": " << blend.canvas_w
             << ",\n  \"canvas_h\": " << blend.canvas_h
             << ",\n  \"offset_x\": " << blend.offset_x
             << ",\n  \"offset_y\": " << blend.offset_y << "\n}\n";
        check_stream(file, base + "blend.json");
    }
    {
        std::ofstream file(base + "seam_x.csv");
        file << "row,x\n";
        for (size_t row = 0; row < trace.blend.seam_x.size(); ++row)
            file << row << ',' << trace.blend.seam_x[row] << '\n';
        check_stream(file, base + "seam_x.csv");
    }
}
int main(int argc, char** argv) {
    init_logging();
    try {
        if (argc == 2 && std::string(argv[1]) == "--help") { usage(argv[0]); return 0; }
        if (argc < 6) { usage(argv[0]); throw std::runtime_error("Missing required arguments"); }
        int runs = 1, warmup = 0;
        bool has_version = false;
        std::string h_path, csv_path, dump_dir;
        for (int i = 4; i < argc; ++i) {
            const std::string option = argv[i];
            if (option != "--version" && option != "--runs" && option != "--warmup" &&
                option != "--save-h" && option != "--timings-csv" && option != "--dump-dir")
                throw std::runtime_error("Unknown option: " + option);
            if (++i >= argc) throw std::runtime_error("Missing value for " + option);
            const std::string value = argv[i];
            if (option == "--version") {
                if (integer(value, 0) != 0) throw std::runtime_error("Only V0 is implemented");
                has_version = true;
            } else if (option == "--runs") runs = integer(value, 1);
            else if (option == "--warmup") warmup = integer(value, 0);
            else if (option == "--save-h") h_path = value;
            else if (option == "--dump-dir") dump_dir = value;
            else csv_path = value;
        }
        if (!has_version) throw std::runtime_error("--version 0 is required");
        std::string output = argv[3];
        auto dot = output.find_last_of('.');
        std::string extension = dot == std::string::npos ? "" : output.substr(dot);
        std::transform(extension.begin(), extension.end(), extension.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (extension != ".png" && extension != ".jpg" && extension != ".jpeg")
            throw std::runtime_error("Output extension must be PNG or JPEG");
        for (const auto& destination : {output, h_path, csv_path}) {
            if (!destination.empty() && (destination == argv[1] || destination == argv[2]))
                throw std::runtime_error("Output must not overwrite an input");
        }
        if ((!h_path.empty() && h_path == output) || (!csv_path.empty() &&
            (csv_path == output || csv_path == h_path)))
            throw std::runtime_error("Output paths must be distinct");
        const ImageRGB a = load(argv[1]), b = load(argv[2]);
        log_message(LogLevel::Process) << "V0: A=" << a.width << 'x' << a.height
            << ", B=" << b.width << 'x' << b.height << ", runs=" << runs
            << ", warmup=" << warmup << '\n';
        if (runs < 5) log_message(LogLevel::Warn) << "Use --runs 5 or more for the report\n";
        ImageRGB panorama;
        Homography3x3 h;
        std::vector<StageTimings> samples;
        for (int i = 0; i < warmup; ++i) {
            StageTimings ignored;
            if (!run_v0(a, b, panorama, ignored, &h)) throw std::runtime_error("Warmup failed");
        }
        for (int i = 0; i < runs; ++i) {
            log_message(LogLevel::Process) << "Run " << i + 1 << '/' << runs << '\n';
            StageTimings time;
            if (!run_v0(a, b, panorama, time, &h)) throw std::runtime_error("V0 pipeline failed");
            samples.push_back(time);
        }
        if (!dump_dir.empty()) {
            log_message(LogLevel::Process) << "Trace run (untimed) -> " << dump_dir << '\n';
            PipelineTrace trace;
            StageTimings ignored;
            ImageRGB traced;
            if (!run_v0(a, b, traced, ignored, nullptr, &trace))
                throw std::runtime_error("Trace run failed");
            write_trace(dump_dir, trace);
        }
        std::cout << std::fixed << std::setprecision(6)
                  << "Stage                       Mean (ms)      Min (ms)\n";
        for (int f = 0; f < 7; ++f) {
            double sum = 0, minimum = std::numeric_limits<double>::infinity();
            for (const auto& sample : samples) {
                sum += sample.*fields[f]; minimum = std::min(minimum, sample.*fields[f]);
            }
            std::cout << std::left << std::setw(26) << names[f] << std::right
                << std::setw(12) << sum / runs << std::setw(14) << minimum << '\n';
        }
        int saved = extension == ".png"
            ? stbi_write_png(output.c_str(), panorama.width, panorama.height, 3,
                             panorama.data.data(), panorama.width * 3)
            : stbi_write_jpg(output.c_str(), panorama.width, panorama.height, 3,
                             panorama.data.data(), 95);
        if (!saved) throw std::runtime_error("Cannot save panorama: " + output);
        if (!h_path.empty()) write_h(h_path, h);
        if (!csv_path.empty()) {
            std::ofstream file(csv_path);
            file << "run";
            for (auto name : names) file << ',' << name;
            file << '\n' << std::setprecision(17);
            for (size_t i = 0; i < samples.size(); ++i) {
                file << i + 1;
                for (auto field : fields) file << ',' << samples[i].*field;
                file << '\n';
            }
            check_stream(file, csv_path);
        }
        log_message(LogLevel::Success) << "Saved " << output << " ("
            << panorama.width << 'x' << panorama.height << ")\n";
        return 0;
    } catch (const std::exception& error) {
        log_message(LogLevel::Error) << error.what() << '\n';
        return 1;
    }
}
