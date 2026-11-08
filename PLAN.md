# PLAN — Phần việc TV1 Trần Nguyễn Minh Quân (Finalterm)

> Nhóm 22 · CSC14120 Lập trình song song · Lập ngày 08/11/2026 trên nền `main` @ `39b4764`.
>
> Đây là **kế hoạch**, chưa mục nào được thực hiện hay kiểm chứng. Chỉ tick ô khi đã có mã, output kiểm tra và bằng chứng tương ứng.
>
> Mục có nhãn **[Cần nhóm duyệt]** đụng tới baseline chung hoặc file do thành viên khác phụ trách. Những mục này phải được Phúc/Hưng đồng ý trước khi merge vào `main`.

## 0. Tóm tắt

- **Vai trò:** back-end stages 4–6 (matching, RANSAC, warp/blend); chủ trì ghép frontend ↔ backend và **nghiệm thu V1 Basic CUDA**.
- **Hiện trạng:** V0 chạy được 12/12 cặp và synthetic đạt. Repo **chưa có dòng CUDA nào**.
- **Việc đầu tiên (Phần A):** khóa baseline V0 để V1 có thể cho kết quả giống V0:
  - tách tham số ra một chỗ;
  - CPU/GPU dùng chung một bản mã cho các hàm toán nhỏ;
  - thay RNG của RANSAC;
  - chốt thứ tự keypoint;
  - xuất được dữ liệu trung gian để đối chiếu.
- **Sau đó (Phần B):** viết kernel stage 4 → 5 → 6, kiểm tra độc lập với V0, ghép `run_v1`, nghiệm thu, viết mục V1 trong notebook.

## 1. Căn cứ

- [Đề Final (EN)](<requirement/finalterm/CSC14120_2026_Final Project.pdf>): mục 3 (các phiên bản), 4 (chạy và đo), 5 (sản phẩm nộp), 6 (liêm chính học thuật).
- [assignment.md](assignment.md): kế hoạch nhóm ngày 03/11/2026, mục 4.3, 5, 9, 10, 11.3, 12.
- [team.json](team.json): role 1 = back-end.
- Mã nguồn hiện tại trong [src/](src/), [tools/](tools/), [Group22_Midterm.ipynb](Group22_Midterm.ipynb).

## 2. Việc được giao

| Mốc | Việc của TV1 |
|---|---|
| M0 Chốt baseline | Rà backend, H và seam; khóa tham số; thống nhất giao diện (Phần A) |
| M1 V1 từng module | CUDA cho stage 4 (matching), 5 (RANSAC + refinement), 6 (warp, exposure, seam, blend) |
| M2 V1 tích hợp | Chủ trì ghép frontend ↔ backend, sửa lỗi, nghiệm thu V1 |
| M3 V2 | Review tiled matching của Phúc |
| M4 V3 | Review phụ thuộc stream/event phía backend của Hưng |
| M5 Nộp | Notebook: mục 5 (V1), mục 2 (cùng Phúc), mục 3 phần stage 4–6; góp gallery, kết luận, references; slide và demo backend |

## 3. Hiện trạng so với phần việc

| Hạng mục | Hiện có | Còn thiếu |
|---|---|---|
| V0 backend [v0_backend.cpp](src/v0_backend.cpp) | Đủ stage 4–6. Notebook Midterm ghi 12/12 cặp thành công, sai số chiếu synthetic khoảng 1e-11 px | Tham số viết cứng rải rác; GPU không tái tạo được RNG |
| CUDA stage 4–6 | Chưa có, chưa có `gpu_backend.cu` | Toàn bộ |
| `run_v1` / `gpu_pipeline.cu` | Chỉ có prototype bị comment trong [versions.h](src/versions.h#L62) | Toàn bộ |
| CLI | [main.cu:63](src/main.cu#L63) từ chối mọi version khác 0; `--runs` mặc định 1 | Phần của Hưng, TV1 cần để chạy V1 |
| Build | Target `cuda` trong Makefile chỉ dùng nvcc biên dịch lại nguồn V0, không có `-arch`, không có file `.cu` mới | Phần của Hưng |
| Benchmark | [benchmark.py:58](tools/benchmark.py#L58) cứng `--version 0` | Phần của Hưng. V1 phải in đúng dòng `Blend: ...` mà [benchmark.py:76](tools/benchmark.py#L76) đọc |
| Đối chiếu stage | Chưa có `verify_versions.py` | Hưng viết công cụ; TV1 cung cấp dữ liệu trung gian stage 4–6 (A5) |

Commit `37b72fe` (23/10) bổ sung cho backend: mutual cross-check, DLT chuẩn hóa, sai số đối xứng, refit 2 lượt, bù sáng affine và adaptive seam. Đây là những phần khó nhất phải đưa lên GPU và phải giải thích khi vấn đáp, nên cần đọc kỹ `git show 37b72fe -- src/v0_backend.cpp`.

---

# Phần A — Các thay đổi trên V0

## A.0 Nguyên tắc

- **Không đổi phương pháp hay tham số** (assignment 4.3). V0 chỉ được sửa để:
  1. V0 và V1 cho kết quả xác định và giống nhau;
  2. CPU và GPU dùng chung một bản mã cho các hàm toán nhỏ;
  3. lấy được dữ liệu trung gian để đối chiếu.
- Trước khi sửa: tag mốc Midterm và lưu output V0 hiện tại làm mốc so sánh (bước 0 ở A.8).
- Thay đổi nào làm đổi kết quả (A3, A4) thì phải chạy lại toàn bộ V0, rồi ghi vào notebook là đã đổi gì và vì sao.
- V0 vẫn đơn luồng: không OpenMP, không `std::thread`, không SIMD tự viết.

## A.1 Tổng quan các thay đổi

| # | Thay đổi | File | Đổi kết quả V0? | Duyệt |
|---|---|---|---|---|
| A1 | Gom tham số vào một header cấu hình | `src/pipeline_config.h` (mới), [v0_stitching.cpp](src/v0_stitching.cpp), [v0_backend.cpp](src/v0_backend.cpp) | Không | [Cần nhóm duyệt]: header dùng chung |
| A2 | Tách các hàm toán dùng chung CPU/GPU | `src/hd_math.h` (mới), [v0_backend.cpp](src/v0_backend.cpp) | Không, output phải trùng từng byte | TV1 |
| A3 | Thay RNG của RANSAC | `src/hd_math.h`, [v0_backend.cpp](src/v0_backend.cpp) | **Có**: mẫu, inlier, H | [Cần nhóm duyệt] |
| A4 | Thứ tự keypoint xác định | [v0_frontend.cpp](src/v0_frontend.cpp) (của Phúc) | **Có**: thứ tự keypoint; tập keypoint ở biên Top-K có thể đổi | [Cần nhóm duyệt]; Phúc sửa hoặc duyệt |
| A5 | Xuất dữ liệu trung gian để đối chiếu | [versions.h](src/versions.h), [common.h](src/common.h), [v0_backend.cpp](src/v0_backend.cpp), [main.cu](src/main.cu) | Không | [Cần nhóm duyệt]: giao diện của Hưng |
| A6 | Giữ nguyên các dòng log | — | — | Ràng buộc |

## A1 — Header cấu hình chung

**Vấn đề:** tham số đang nằm ở ba nơi:
- trong lời gọi hàm ở [v0_stitching.cpp:52](src/v0_stitching.cpp#L52), [:70](src/v0_stitching.cpp#L70), [:85](src/v0_stitching.cpp#L85);
- trong giá trị mặc định ở [versions.h](src/versions.h);
- và viết trực tiếp trong thân hàm stage 6.

assignment 4.3 yêu cầu ghi tham số thực dùng vào một cấu hình chung. V0 và V1 phải đọc từ cùng một chỗ.

**Đề xuất** (`src/pipeline_config.h`):

```cpp
// Biến constexpr kiểu scalar ở namespace scope dùng được trực tiếp trong device code.
namespace cfg {
// Stage 1–3 (do Phúc dùng chung; giá trị lấy từ v0_frontend.cpp hiện tại)
constexpr float    kBlurSigma            = 1.2f;
constexpr int      kBlurRadius           = 2;
constexpr int      kFastThreshold        = 20;
constexpr int      kDescriptorBorder     = 16;
constexpr int      kMaxKeypoints         = 1500;
// Stage 4
constexpr float    kRatioThreshold       = 0.75f;
// Stage 5
constexpr int      kRansacIters          = 1500;
constexpr float    kInlierThresholdPx    = 3.0f;
constexpr unsigned kRansacSeed           = 42;
constexpr int      kRefitPasses          = 2;
constexpr int      kMinInliers           = 8;
constexpr int      kMinInlierRatioDen    = 5;      // giữ biểu thức inliers*5 < matches
// Canvas
constexpr double   kMaxCanvasSide        = 32768;
constexpr double   kMaxCanvasPixels      = 120000000;
// Stage 6 — exposure
constexpr int      kExposureStride       = 2;
constexpr float    kDarkMeanMax          = 20.0f;  // bỏ mẫu nếu mean <= 20
constexpr float    kSaturatedMin         = 245.0f; // bỏ mẫu nếu max >= 245
constexpr size_t   kMinExposureSamples   = 1000;
constexpr double   kGainMin = 0.75, kGainMax = 1.33;
constexpr double   kBiasLimit            = 64.0;
constexpr double   kRequiredImprovement  = 0.90;
// Stage 6 — seam
constexpr size_t   kMinSeamSamples       = 3000;
constexpr double   kSeamMaeThreshold     = 18.0;
constexpr double   kSeamPixelCostBias    = 1.0;
constexpr int      kSeamWindow           = 2;
constexpr float    kSeamBand             = 24.0f;
}
```

- Giữ nguyên **kiểu dữ liệu và biểu thức so sánh** như V0 (`<=`, `>=`, `<`, phép chia float). Chỉ thay số viết cứng bằng tên hằng.
- Notebook mục 3 in bảng tham số trực tiếp từ header này, để số trong báo cáo luôn khớp với mã.

**Kiểm tra:** H, panorama và số match/inlier phải trùng từng byte với mốc Midterm (A.8 bước 1).

## A2 — Hàm toán dùng chung CPU/GPU

**Vấn đề:** V1 phải tính DLT, nghịch đảo H, chiếu điểm, sai số đối xứng và nội suy bilinear giống V0 đến từng bit. Nếu viết hai bản riêng thì dễ lệch, và lúc vấn đáp phải giải thích cả hai bản.

**Đề xuất** `src/hd_math.h` (header-only, tất cả là `inline`):

```cpp
#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif

HD int    hamming256(const uint8_t* a, const uint8_t* b);   // host: vòng Kernighan như V0; device: 8 × __popc
HD bool   solve_linear_8(double A[8][8], double b[8], double x[8]);
HD void   multiply_3x3(const double a[9], const double b[9], double out[9]);
HD bool   invert_h(const double h[9], double inv[9]);
HD void   project_point(const double h[9], float x, float y, float& ox, float& oy); // giữ ép float và kẹp wz như V0
HD double symmetric_error_sq(const double h[9], const double hinv[9],
                             float ax, float ay, float bx, float by);
// DLT chuẩn hóa tách thành ba bước để GPU làm reduction được, còn CPU gọi tuần tự đúng thứ tự cũ:
struct DltNorm { double csx, csy, cdx, cdy, ss, sd; };
HD void   dlt_accumulate(const DltNorm& n, float sx, float sy, float dx, float dy,
                         double ata[8][8], double atb[8]);
HD bool   dlt_solve(double ata[8][8], double atb[8], const DltNorm& n, double h[9]);
HD bool   sample_bilinear(const uint8_t* rgb, int w, int h, float x, float y, float out[3]);
```

Quy tắc khi viết:
- Không dùng `std::vector`, `std::max/min`, `std::abs` trong hàm `HD`.
- Dùng helper tự viết `hd_max(a,b) = (a < b) ? b : a` và `hd_min(a,b) = (b < a) ? b : a`, đúng ngữ nghĩa `std::max/min`. Dùng `fabs`, `sqrt`, `hypot`, `isfinite` của `<cmath>` vì đều có trên device.
- **Giữ đúng thứ tự phép tính và các chỗ ép kiểu** như V0, gồm: ép `float` trong `project_point` ([v0_backend.cpp:192-198](src/v0_backend.cpp#L192-L198)) và thứ tự tính tâm → khoảng cách trung bình → cộng dồn ATA ([138-187](src/v0_backend.cpp#L138-L187)).
- `solve_homography_points` trong V0 đổi sang nhận con trỏ + số phần tử. `sample_bilinear` nhận con trỏ dữ liệu ảnh thay vì `ImageRGB`.

**Kiểm tra:** output trùng từng byte với mốc (A.8 bước 2). Không cần thay đổi cờ biên dịch.

## A3 — RNG của RANSAC dùng chung CPU/GPU [Cần nhóm duyệt]

**Vấn đề:** [v0_backend.cpp:209-210](src/v0_backend.cpp#L209-L210) dùng `std::mt19937` + `std::uniform_int_distribution`. Cách distribution biến số ngẫu nhiên thành chỉ số do thư viện chuẩn tự quyết, nên GPU không tái tạo được chuỗi mẫu. Hai thư viện khác nhau dù cùng seed cũng không chắc cho cùng chuỗi (assignment 5.3).

**Đề xuất:** hàm sinh số dạng hash theo bộ đếm (counter-based). Mỗi hypothesis tự sinh mẫu của mình mà không cần trạng thái chung, nên chạy được một thread cho một hypothesis:

```cpp
HD inline uint64_t splitmix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}
// Rút 4 chỉ số khác nhau cho hypothesis `iter`; trả false nếu vượt số lần thử.
HD inline bool ransac_sample(uint32_t seed, uint32_t iter, uint32_t n, int out[4]) {
    uint32_t attempt = 0;
    for (int k = 0; k < 4; ++k) {
        for (;;) {
            if (attempt >= 64) return false;
            uint64_t key = (uint64_t(seed) << 32) ^ (uint64_t(iter) << 8) ^ attempt++;
            int idx = int(splitmix64(key) % n);
            bool dup = false;
            for (int j = 0; j < k; ++j) dup |= (out[j] == idx);
            if (!dup) { out[k] = idx; break; }
        }
    }
    return true;
}
```

- Giới hạn 64 lần thử để vòng lặp trên GPU luôn kết thúc. Với n ≥ 4, trường hợp vượt giới hạn gần như không xảy ra; khi xảy ra, hypothesis bị bỏ qua giống như khi DLT thất bại. CPU và GPU xử lý giống nhau.
- Độ lệch do phép `%` không đáng kể với n ≤ 1500.
- **Trích dẫn** splitmix64 (Steele, Lea, Flood, "Fast Splittable Pseudorandom Number Generators", OOPSLA 2014) ở đầu file và trong mục References của notebook.
- Phương án dự phòng nếu nhóm không muốn sửa V0: sinh sẵn bảng 1500 × 4 chỉ số ở host rồi upload. Khi đó phải ghi rõ phạm vi việc làm trên CPU và tính thời gian vào H2D.

**Kiểm tra:** 12/12 cặp vẫn thành công; synthetic dưới 1 px; ghi bảng số inlier mới/cũ và sai số chiếu giữa H mới và H Midterm (A.8 bước 4).

## A4 — Thứ tự keypoint xác định [Cần nhóm duyệt — file của Phúc]

**Vấn đề:** score FAST là tổng các số nguyên ([v0_frontend.cpp:149-153](src/v0_frontend.cpp#L149-L153)), nên các điểm **bằng response xảy ra thường xuyên**. Thêm hai điều nữa:
- `std::partial_sort` không ổn định ([181-187](src/v0_frontend.cpp#L181-L187)), nên thứ tự giữa các điểm bằng nhau không xác định;
- khi có ≤ 1500 điểm thì không sắp xếp, giữ thứ tự quét raster.

Frontend GPU gom điểm bằng atomics sẽ ra thứ tự khác. Thứ tự keypoint đổi kéo theo chỉ số match đổi, mẫu RANSAC đổi và H khác, dù backend hoàn toàn đúng.

**Đề xuất:** luôn sắp theo khóa toàn phần (response giảm dần, y tăng, x tăng) bằng `std::sort`, rồi cắt lấy K điểm đầu. Nếu Phúc muốn giữ thứ tự raster khi ≤ K thì cũng được, miễn V0 và V1 dùng cùng một quy tắc.

**Đề xuất thêm cho Phúc:** pattern BRIEF hiện sinh bằng `std::mt19937` + `std::normal_distribution` ([200-212](src/v0_frontend.cpp#L200-L212)), nên GPU cũng không tái tạo được. Nên sinh pattern một lần từ V0 rồi ghi thành mảng hằng 256 × 4 dùng chung (assignment 5.2).

TV1 chỉ đề xuất và kiểm tra ảnh hưởng tới stage 4–6; việc sửa file do Phúc quyết.

## A5 — Xuất dữ liệu trung gian để đối chiếu [Cần nhóm duyệt]

**Vấn đề:** để so V0 với V1 theo từng stage (assignment 9.1), cần lấy được danh sách match, tập inlier, H và các quyết định blend. Hiện stage 6 chỉ in log; các giá trị gain, bias và `seam_x` không đi ra khỏi hàm.

**Đề xuất:**

```cpp
struct BlendInfo {
    bool   exposure_applied = false;  double overlap_mae = 0;
    bool   seam_used = false;
    double gain[3] = {1,1,1}, bias[3] = {0,0,0};
    int    canvas_w = 0, canvas_h = 0;  float offset_x = 0, offset_y = 0;
    std::vector<int> seam_x;
};
struct PipelineTrace {                 // run_v0 và run_v1 cùng điền
    std::vector<Keypoint> kptsA, kptsB;
    std::vector<Match>    matches;
    std::vector<int>      inliers;
    Homography3x3         H;
    BlendInfo             blend;
};
ImageRGB v0_warp_and_blend(const ImageRGB&, const ImageRGB&, const Homography3x3&,
                           BlendInfo* info = nullptr);
bool run_v0(..., Homography3x3* outH = nullptr, PipelineTrace* trace = nullptr);
```

- Chỉ điền khi con trỏ khác `nullptr`. Lần chạy dump tách riêng khỏi các lần đo, để không làm sai thời gian.
- Đề xuất CLI có `--dump-dir DIR` để ghi `matches.csv`, `inliers.csv`, `H.txt`, `blend.json`, `seam_x.csv`. `main.cu` do Hưng phụ trách nên gửi đề xuất trước.
- `verify_versions.py` của Hưng đọc các file này cho cả V0 và V1.

## A6 — Các dòng log phải giữ nguyên

`benchmark.py` và notebook đọc các dòng sau, nên V0 giữ nguyên và V1 phải in giống hệt:

- `Keypoints: A=<n>, B=<n>`
- `Mutual ratio matches: <n>`
- `RANSAC inliers: <n>/<m>`
- `Blend: exposure=<affine|unchanged>, overlap MAE=<x>, seam=<optimal|feather>` ([v0_backend.cpp:444-447](src/v0_backend.cpp#L444-L447))
- Các thông báo lỗi trong [v0_stitching.cpp:74-129](src/v0_stitching.cpp#L74-L129)

## A7 — Những gì cố ý KHÔNG sửa trong V0

| Giữ nguyên | Lý do |
|---|---|
| Đếm bit Kernighan trong V0 | Không đổi baseline; GPU dùng `__popc` nhưng ra cùng kết quả |
| `project_point` kẹp `|wz| < 1e-12` thành `1e-12` (mất dấu) | Giữ hành vi gốc; V1 làm y hệt |
| Double trong DLT, RANSAC, exposure, seam | Không giảm độ chính xác riêng ở một phiên bản |
| Mọi ngưỡng, bước lấy mẫu 2, band 24, cửa sổ ±2 | assignment 4.3: không đổi phương pháp/tham số |
| Stage 1–3 ngoài A4 | Phạm vi của Phúc |

## A.8 Thứ tự commit và kiểm tra hồi quy

Làm các thay đổi không đổi kết quả trước. Nhờ vậy nếu output lệch, ta biết ngay commit nào gây ra.

| Bước | Việc | Tiêu chí so với mốc Midterm |
|---|---|---|
| 0 | Nhờ Hưng tag `midterm-v0` trên `main`; chạy benchmark V0 và lưu `outputs_v0/` ra **ngoài repo** làm mốc | — |
| 0b | Viết `tools/compare_runs.py`: so hai thư mục output (H.txt từng byte và chênh lệch lớn nhất, pixel từng ảnh panorama, số match/inlier đọc từ `.log`) | Chạy thử với chính mốc → trùng 100% |
| 1 | A1 header cấu hình | Trùng từng byte: `*.H.txt`, `panorama_*.png`, số match/inlier |
| 2 | A2 hàm `HD` | Như bước 1 |
| 3 | A5 dữ liệu trung gian | Như bước 1, và các file dump đọc được |
| 4 | A3 RNG mới | 12/12 thành công; synthetic dưới 1 px; bảng chênh lệch H mới/cũ; xem lại panorama bằng mắt |
| 5 | A4 thứ tự keypoint (khi Phúc đồng ý) | Như bước 4 |

So sánh "trùng từng byte" chỉ có nghĩa khi hai lần chạy dùng cùng máy và cùng trình biên dịch. Vì vậy chạy cả mốc và bản mới trong **cùng một phiên Colab**.

## A.9 Lệnh chạy V0 trên Colab

```bash
git clone -b <tên-nhánh> https://github.com/HungHiHung10/Image-Stitching.git
cd Image-Stitching
pip install -q -r requirements.txt
python - <<'EOF'
import gdown, re, shutil, tempfile
from pathlib import Path
url = 'https://drive.google.com/drive/folders/1KEfstxQ1an5bEngfaHfXl9wIdbXWo2wF?usp=sharing'
raw, syn = Path('dataset/raw'), Path('dataset/synthetic')
raw.mkdir(parents=True, exist_ok=True); syn.mkdir(parents=True, exist_ok=True)
with tempfile.TemporaryDirectory() as tmp:
    gdown.download_folder(url=url, output=tmp, quiet=True, use_cookies=False)
    for p in Path(tmp).rglob('*'):
        if re.fullmatch(r'(indoor|outdoor|custom|challenging)_(left|right)(\.\w+)?', p.name, re.I):
            shutil.copy2(p, raw / p.name)
        elif re.fullmatch(r'synthetic(_GT)?\.\w+', p.name, re.I):
            shutil.copy2(p, syn / 'synthetic_GT.HEIC')
EOF
python tools/prepare_dataset.py
python tools/build.py                      # trên Colab sẽ dùng nvcc nếu có
python tools/benchmark.py --runs 5 --warmup 1
cp -r outputs_v0 /content/baseline_midterm_v0   # chỉ làm ở bước 0, trên commit đã tag
```

Trên macOS chỉ biên dịch được V0: `cd src && make cpu`. Để chuẩn bị dataset tại máy cần có `pillow-heif` và `gdown` như trong `requirements.txt`.

---

# Phần B — V1 và các mốc

## B.0 Mười điểm V1 phải giữ để khớp V0

1. **RNG RANSAC:** dùng `ransac_sample` chung (A3).
2. **Thứ tự match:** V0 xuất match theo `queryIdx` tăng dần, và mẫu RANSAC là chỉ số vào danh sách match. Không gom match bằng `atomicAdd`; dùng cờ đánh dấu + prefix sum để giữ thứ tự.
3. **Cách chọn khi khoảng cách bằng nhau trong matching** ([52-58](src/v0_backend.cpp#L52-L58), [64-74](src/v0_backend.cpp#L64-L74)):
   - Chiều thuận lấy `j` nhỏ nhất.
   - Khoảng cách bằng best được ghi vào `second`, nên ratio bằng 1 và match bị loại.
   - Chiều ngược lấy `i` nhỏ nhất.
   - Giữ nguyên phép chia float `best/second < 0.75`, đừng đổi thành `best < 0.75*second`.
4. **Chọn hypothesis tốt nhất** ([260](src/v0_backend.cpp#L260)): V0 dùng `>` nghiêm ngặt, nên khi bằng số inlier thì iteration sớm nhất thắng. Reduction trên GPU so theo cặp (số inlier lớn nhất, iter nhỏ nhất).
5. **Ép float giữa chừng** trong `project_point`: dùng hàm chung (A2).
6. **FMA:** nvcc mặc định gộp phép nhân-cộng. Điểm nằm sát ngưỡng 3 px có thể đổi trạng thái inlier, và pixel có thể lệch 1 khi ép `uint8_t`. Đề xuất `--fmad=false` cho mọi bản GPU (mục 6).
7. **Cộng số thực theo thứ tự khác:** thống kê exposure và ATA khi refit đều là phép cộng dồn. Không dùng `atomicAdd` double vì mỗi lần chạy cho kết quả hơi khác. Dùng reduction 2 bước có thứ tự cố định: mỗi block tính một tổng con, rồi một kernel cộng các tổng con theo thứ tự.
8. **Seam DP** ([449-490](src/v0_backend.cpp#L449-L490)):
   - Xét `d = -2..2` theo thứ tự với `<`, nên khi bằng nhau thì `d` nhỏ nhất thắng.
   - Hàng không có pixel hợp lệ bị bỏ qua và `previous` giữ nguyên.
   - Truy vết bắt đầu từ chỉ số đầu tiên mà `min_element` trả về.
9. **Thứ tự keypoint từ frontend:** theo quy tắc chung (A4).
10. **Log và H:** in đúng các dòng ở A6; trả H qua `outH` để `--save-h` hoạt động, vì phần kiểm tra synthetic đọc file này.

## B.1 M0 — Chốt baseline (= Phần A + giao diện)

- [ ] Gửi nhóm đề xuất A1–A5, mục 5 (giao diện) và mục 6 (quyết định).
- [ ] Thực hiện A.8 bước 0 → 5.
- [ ] **Xong khi:** V0 mới chạy 12/12 cặp, synthetic đạt, giao diện mục 5 đã thống nhất bằng văn bản.

## B.2 M1 — Kernel stages 4–6

File mới của TV1: `src/gpu_backend.cuh` (giao diện), `src/gpu_backend.cu` (kernel + hàm host).

**Stage 4 — Matching**

| Kernel | Mỗi thread/block làm gì |
|---|---|
| `match_forward` | 1 thread cho 1 descriptor A, duyệt hết B bằng `hamming256` (8 × `__popc`); giữ best, second, idx theo đúng quy tắc của V0 |
| `match_reverse` | 1 thread cho 1 descriptor B, duyệt A; chỉ giữ best và idx |
| `match_filter` | 1 thread cho 1 A; áp ratio + mutual, ghi cờ 0/1 |
| `compact_ordered` | Prefix sum trên cờ (1 block đủ cho ≤ 1500 phần tử); ghi `d_matches` theo `queryIdx` tăng dần và `d_match_count` |

- Không lưu ma trận khoảng cách đầy đủ.
- Đọc số keypoint từ con trỏ device để khỏi đồng bộ.
- Chỉ copy `match_count` về host để dừng khi có ít hơn 4 match. Ghi rõ đây là bước điều phối.

**Stage 5 — RANSAC + refinement**

| Kernel | Mỗi thread/block làm gì |
|---|---|
| `ransac_hypotheses` | 1 thread cho 1 hypothesis: `ransac_sample`, chuẩn hóa + `dlt_accumulate` + `dlt_solve` cho 4 điểm, `invert_h`; ghi H, H⁻¹, cờ hợp lệ |
| `ransac_score` | 1 block cho 1 hypothesis: các thread chia nhau duyệt match, block reduction đếm inlier |
| `ransac_select` | Argmax theo (số inlier, −iter) |
| `collect_inliers` | 1 thread cho 1 match, sau đó gom có thứ tự |
| `refit_normalize` / `refit_accumulate` | Reduction tâm và khoảng cách trung bình, rồi reduction ATA (64 số) + ATb (8 số) theo thứ tự cố định |
| `refit_solve` | 1 thread chạy `dlt_solve`; lặp 2 lượt như V0 |
| `canvas_bounds` | 1 thread chiếu 4 góc ảnh B; copy vài số về host để cấp phát canvas và kiểm tra như [v0_stitching.cpp:99-129](src/v0_stitching.cpp#L99-L129) |

- Giữ double để khớp V0. GPU T4 chạy double rất chậm (khoảng 1/32 tốc độ float); ghi điều này vào phần phân tích V1, đây là dữ liệu tốt cho V2/V3.
- Không đổi sang sai số một chiều.
- Mẫu suy biến phải bị loại như khi `solve_linear_8` trả `false`.

**Stage 6 — Warp, exposure, seam, blend**

| Kernel | Mỗi thread/block làm gì |
|---|---|
| `exposure_stats` | 1 thread cho 1 pixel được lấy mẫu (bước 2); reduction 13 giá trị (sx, sy, sxx, sxy cho 3 kênh + số mẫu), 2 bước có thứ tự |
| `exposure_fit` | 1 thread tính gain/bias |
| `exposure_error` | Reduction raw/corrected error và số mẫu |
| Quyết định | Copy vài số về host để in dòng `Blend:` và chọn nhánh seam |
| `seam_cost` | 1 thread cho 1 pixel canvas: tính cost và cờ "hàng có pixel hợp lệ" |
| `seam_dp_row` | 1 kernel cho mỗi hàng, 1 thread cho mỗi cột; đọc hàng trước, ghi hàng hiện tại và `parent` (int8) |
| `seam_traceback` | 1 thread, bắt buộc vẫn chạy trên GPU |
| `warp_blend` | 1 thread cho 1 pixel đầu ra: bilinear A, B, chỉnh B, feather hoặc seam alpha, ép `uint8_t` như V0 |

- Không dùng `__syncthreads()` để đồng bộ nhiều block.
- Hàng không hợp lệ: copy cờ hàng về host một lần; host chỉ launch kernel cho các hàng hợp lệ và truyền đúng con trỏ "hàng trước".
- Phương án hợp lệ khác cho seam: kernel **chỉ có 1 block**, lặp qua các hàng và gọi `__syncthreads()` giữa các hàng.

**Giao diện hàm host (bản nháp, chốt cùng Phúc/Hưng):**

```cpp
struct GpuFeatures { const Keypoint* d_kpts; const Descriptor256* d_desc; const int* d_count; int capacity; };
struct GpuMatches  { Match* d_matches; int* d_count; int capacity; };

int  gpu_match(const GpuFeatures& A, const GpuFeatures& B, GpuMatches& out, cudaStream_t s);
bool gpu_ransac(const GpuFeatures& A, const GpuFeatures& B, const GpuMatches& m,
                double H_out[9], int* inlier_count, cudaStream_t s);
bool gpu_warp_blend(const uint8_t* d_rgbA, int wA, int hA,
                    const uint8_t* d_rgbB, int wB, int hB,
                    const double H[9], DeviceImage& pano, BlendInfo* info, cudaStream_t s);
```

## B.3 M1b — Kiểm tra backend độc lập (không chờ frontend của Phúc)

- [ ] Viết `src/tests/test_backend.cu`, hoặc một chế độ trong CLI:
  1. chạy frontend **V0**;
  2. upload keypoint và descriptor của V0 lên GPU;
  3. chạy stage 4–6 trên GPU;
  4. so với `PipelineTrace` của V0.
- [ ] Các tiêu chí so sánh:
  - Danh sách match giống hệt: index, distance, thứ tự.
  - Tập inlier giống hệt.
  - H: sai số chiếu trên lưới điểm so với H của V0 (mean và max px).
  - Canvas cùng kích thước và offset.
  - Cùng quyết định exposure và seam; gain/bias; `seam_x` giống hệt.
  - Pixel: max |diff| và % pixel lệch trên vùng hợp lệ.
- [ ] Thử các ca lỗi (assignment 9.3): ít hơn 4 match, descriptor rỗng, H suy biến, canvas quá lớn, ảnh large 3840×2160.
- [ ] Chạy `compute-sanitizer --tool memcheck` và `--tool racecheck`.
- [ ] Synthetic: sai số chiếu của V1 dưới 1 px.

## B.4 M2 — Tích hợp và nghiệm thu V1 (TV1 chủ trì)

- [ ] Viết `run_v1(...)` trong `src/gpu_pipeline.cu`, cùng chữ ký `run_v0` (có `outH`, `trace`). Luồng: H2D RGB A/B → frontend A, B (Phúc) → stage 4–6 → các kiểm tra như [v0_stitching.cpp:88-129](src/v0_stitching.cpp#L88-L129) với cùng thông báo lỗi → D2H panorama.
- [ ] Đo bằng `GpuTimer` cho từng stage, H2D, D2H và end-to-end. Ghi rõ:
  - mỗi lần copy nhỏ giữa các stage được tính vào stage nào;
  - việc cấp phát nằm trong hay ngoài vùng đo (thống nhất với Hưng).
- [ ] Không dùng `CHECK_KERNEL_ASYNC` khi đo thời gian: macro này thực ra có gọi `cudaDeviceSynchronize` ([cuda_utils.cuh:16](src/cuda_utils.cuh#L16)). Khi đo chỉ gọi `cudaGetLastError`; đồng bộ đầy đủ chỉ dùng ở chế độ debug.
- [ ] Giải phóng bộ nhớ đúng cả khi gặp lỗi (dùng buffer RAII trong `gpu_buffers.cuh` của Hưng).
- [ ] Chạy 4 cặp × 3 kích thước, warmup 1, runs 5. Cặp thất bại phải ghi rõ lỗi, không dùng lại ảnh kết quả cũ.
- [ ] Checklist nghiệm thu (assignment 5.5):
  - [ ] Biên dịch bằng CUDA trên Colab và thực sự launch kernel cho cả sáu stage.
  - [ ] Không chuyển Top-K, RANSAC/refinement hoặc seam về CPU.
  - [ ] Đối chiếu từng stage và kiểm tra hình học đạt tolerance đã chốt.
  - [ ] Chạy đủ 4 cặp × 3 kích thước; thất bại được ghi rõ.
  - [ ] Có timing GPU từng stage, transfer, tổng kernel và end-to-end.
  - [ ] Có code và giải thích mapping thread/block trong notebook.
  - [ ] Có số đo V1 để chọn tối ưu cho V2.

## B.5 M3 / M4 — Review

- [ ] **V2 (Phúc, tiled matching):**
  - popcount, top-2 và cách chọn khi bằng nhau giống V1;
  - thứ tự match đầu ra giữ nguyên;
  - tile cuối thiếu phần tử vẫn đi qua barrier hợp lệ;
  - danh sách match của V2 trùng V1.
- [ ] **V3 (Hưng, streams):**
  - matching chờ đúng event của A và B;
  - buffer/counter của A và B tách riêng;
  - timing không cộng trùng phần chạy chồng lấp;
  - chạy lặp nhiều lần để phát hiện race.

## B.6 M5 — Báo cáo, slide, video, vấn đáp

- [ ] Mục 3 notebook, stage 4–6: phương pháp, tham số (in từ `pipeline_config.h`), lý do về chất lượng và khả năng song song.
- [ ] Mục 5 V1:
  - bảng mapping thread/block cho cả 6 stage (Phúc bổ sung stage 1–3);
  - các đoạn code chính;
  - bảng đối chiếu với V0;
  - timing và speedup theo độ phân giải, tách H2D/D2H/kernel/end-to-end.
  - Mọi số lấy từ output cell trong cùng một phiên Colab.
- [ ] Mục 2 Introduction (cùng Phúc).
- [ ] Gallery: phân tích failure case về mặt hình học (parallax, global H, bóng kép ở site4); xem cả ảnh đầy đủ canvas.
- [ ] References:
  - Lowe 2004; Fischler & Bolles 1981; Hartley & Zisserman 2003;
  - splitmix64 (A3);
  - các tài liệu CUDA đã tham khảo;
  - khai báo các công cụ AI đã dùng và dùng vào việc gì.
- [ ] Ghi chú baseline: notebook phải nêu rõ V0 Final khác V0 Midterm ở A3/A4 và vì sao.
- [ ] Slide phần V1/backend. Video: demo `--version 0` so với `--version 1`, giải thích kernel stage 4–6.
- [ ] Ôn để giải thích được toàn pipeline, kể cả V2 và V3.

---

## 5. Giao diện cần thống nhất bằng văn bản

**Với Phúc (frontend → backend):**
- `Keypoint*` và `Descriptor256*` nằm trên device.
- Số phần tử hợp lệ nằm trong `int*` trên device; dung lượng tối đa `cfg::kMaxKeypoints`.
- Thứ tự keypoint theo A4.
- Descriptor 32 byte: bit `b` nằm ở byte `b/8`, vị trí `b%8`, như [v0_frontend.cpp:239](src/v0_frontend.cpp#L239).
- Thống nhất stream dùng và ai sở hữu buffer.

**Với Hưng:**
- `--version 1` gọi `run_v1`; `--runs` mặc định 5.
- Các trường timing mới (H2D, D2H, tổng kernel, end-to-end) trong `StageTimings` và CSV.
- `gpu_buffers.cuh` (RAII).
- Cờ build: `-arch`, `--fmad=false`; danh sách nguồn `.cu`.
- `--dump-dir` và định dạng file dump để `verify_versions.py` dùng (A5).

## 6. Quyết định cần nhóm chốt trước khi benchmark

| Vấn đề | Đề xuất |
|---|---|
| RNG RANSAC | Hàm hash `__host__ __device__` dùng chung, sửa cả V0 (A3) |
| Thứ tự keypoint, pattern BRIEF | Khóa toàn phần (response, y, x); pattern BRIEF thành mảng hằng (A4) |
| FMA | `--fmad=false` cho mọi bản GPU, ghi trong notebook |
| Tolerance | Match, inlier, canvas, quyết định blend, `seam_x`: khớp tuyệt đối. H: sai số chiếu so với V0 rất nhỏ (ví dụ mean ≤ 0.01 px). Pixel: lệch tối đa ±1 trên vùng hợp lệ. Nhóm chốt con số cụ thể, ghi vào notebook và giữ nguyên cho mọi phiên bản |
| Mapping RANSAC | 1 thread cho mỗi hypothesis để sinh H, 1 block cho mỗi hypothesis để chấm điểm. Cách này khớp cả assignment 5.3 lẫn kế hoạch GPU trong notebook Midterm |
| Cấp phát trong hay ngoài vùng đo | Cùng một chính sách cho V1–V3, ghi rõ là đo steady-state nếu có tái sử dụng buffer |

## 7. Vấn đề ngoài phạm vi TV1 có thể chặn nghiệm thu V1

| Vấn đề | Vị trí | Người phụ trách |
|---|---|---|
| Hàm `synthetic_procedural()` được gọi nhưng chưa định nghĩa → `NameError` khi thiếu HEIC hoặc chạy `--procedural-synthetic` (README dòng 10 hướng dẫn chạy lệnh này) | [prepare_dataset.py:100](tools/prepare_dataset.py#L100) | Hưng |
| README nhắc `tools/verify_cli.py`, `execute_notebook.py`, `package.py` nhưng repo không có | [README.md:12-14](README.md#L12-L14) | Hưng |
| Cell kiểm tra correctness (mục 3.5.1) dùng đường dẫn cứng `Path('A:\\PP\\Midterm')` → Run All trên Colab lỗi | Group22_Midterm.ipynb | Hưng |
| Bảng "Lộ trình phân công" trong notebook Midterm khác phân công ngày 03/11 | Group22_Midterm.ipynb | Hưng (notebook Final) |
| CLI chỉ nhận V0, Makefile/build chưa có nguồn CUDA, benchmark cứng V0 | main.cu, Makefile, build.py, benchmark.py | Hưng |

## 8. Quy ước Git cho phần việc này

- **Nhánh:** tách từ `main` @ `39b4764`. Tên gợi ý theo assignment 12.1: `feat/v1-basic-cuda`.
- **Khuyến nghị:** Phần A làm thay đổi baseline dùng chung, nên mở PR riêng cho Phần A sớm và merge vào `main` trước, để Phúc/Hưng làm trên cùng baseline. Nếu chỉ dùng một nhánh thì đặt Phần A thành các commit đầu tiên, tách riêng theo đúng thứ tự A.8.
- **Commit đầu tiên:** chính file `PLAN.md` này.
- **Mỗi commit/PR** ghi: thay đổi gì, thuộc phiên bản nào, kiểm tra bằng cách nào, kết quả, giới hạn còn lại.
- Không commit binary, dataset, `outputs_*`, ảnh tạm (đã có trong `.gitignore`).
- Không đổi `common.h`, `versions.h`, `main.cu` hoặc schema timing mà chưa báo Hưng (assignment 12.1).
- Mỗi file mới có comment đầu file ghi tác giả và nguồn tham khảo bên ngoài, nếu có.

## 9. Thứ tự bắt đầu

1. Đọc diff `37b72fe` và toàn bộ [v0_backend.cpp](src/v0_backend.cpp) cho tới khi tự giải thích được từng nhánh.
2. Gửi nhóm đề xuất A1–A5 cùng mục 5 và mục 6; nhờ Hưng tag `midterm-v0`.
3. Bước 0 → 3 của A.8 (mốc, `compare_runs.py`, A1, A2, A5). Các bước này không đổi kết quả, nên có thể làm trước khi nhóm duyệt xong A3/A4.
4. A3 (và A4 khi Phúc đồng ý), chạy lại V0 trong cùng phiên Colab.
5. Stage 4 GPU + kiểm tra độc lập trên Colab (máy Mac không có GPU NVIDIA).
6. Stage 5, rồi stage 6; để seam sau cùng vì khó nhất.
7. Ghép `run_v1` khi frontend của Phúc sẵn sàng; nghiệm thu theo B.4.

Kế hoạch không gắn ngày vì assignment.md ghi rõ chưa có deadline Finalterm được xác nhận.
