#include "../src/pipeline_config.h"
#include <algorithm>
#include <cstdio>
#include <random>

int main() {
    std::mt19937 rng(cfg::kBriefSeed);
    std::normal_distribution<float> dist(0.0f, cfg::kBriefSigma);
    const int radius = cfg::kBriefPatchRadius;
    std::printf("#pragma once\n#include <cstdint>\n\n"
                "#ifdef STITCH_DEVICE_BRIEF_PATTERN\n"
                "static __constant__ int8_t kBriefPattern[256][4] = {\n"
                "#else\n"
                "static const int8_t kBriefPattern[256][4] = {\n"
                "#endif\n");
    for (int i = 0; i < 256; ++i) {
        int v[4];
        for (int k = 0; k < 4; ++k)
            v[k] = std::min(std::max(static_cast<int>(dist(rng)), -radius), radius);
        std::printf("    {%d, %d, %d, %d}%s\n", v[0], v[1], v[2], v[3], i == 255 ? "" : ",");
    }
    std::printf("};\n");
    return 0;
}
