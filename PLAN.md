# PLAN — Phần việc TV1 Trần Nguyễn Minh Quân (Finalterm)

> Nhóm 22 · CSC14120 Lập trình song song · Lập ngày 08/11/2026 trên nền `main` @ `39b4764`.
>
> Tiến độ thực tế ghi ở các mục **Trạng thái** ngay dưới đây. Từ mục 0 trở đi là kế hoạch gốc; chỗ nào thiết kế đã đổi thì có ghi chú tại chỗ. Chỉ tick ô khi đã có mã, output kiểm tra và bằng chứng tương ứng.
>
> Mục có nhãn **[Cần nhóm duyệt]** đụng tới baseline chung hoặc file do thành viên khác phụ trách. Những mục này phải được Phúc/Hưng đồng ý trước khi merge vào `main`.

## Trạng thái Phần A (cập nhật 25/11/2026, nhánh `refactor/v0-baseline-lock`)

**Kết luận:** Phần A đã xong và đã được kiểm tra qua 2 lượt trên Colab (Tesla T4, nvcc 13.0, g++ 13.3, Python 3.13). Nhánh đã push lên GitHub ngày 08/11/2026 và đã mở PR vào `main` ngày 25/11/2026. Việc duy nhất chặn merge là Phúc duyệt A4. Nên merge bằng *Create a merge commit* (không squash hay rebase), vì mục này và nhánh Phần B tham chiếu mã commit của nhánh. Tag `midterm-v0` đặt vào commit `39b4764`, và vì `main` chưa đổi kể từ khi tách nhánh nên tag có thể đặt trước hay sau khi merge.

| Mục | Commit | Trạng thái |
|---|---|---|
| `PLAN.md` | `6659c03` | Xong |
| A.8 bước 0b `tools/compare_runs.py` | `d00e314` | Xong |
| A1 `pipeline_config.h` | `d547f95` | Xong |
| A2 `hd_math.h` | `acb2e4f`, sửa ở `4fcd8a5` | Xong |
| A5 dữ liệu trung gian + `--dump-dir` | `e3034d2` | Xong |
| A3 RNG splitmix64 | `5803e2f` | Xong |
| A4 thứ tự keypoint + bảng BRIEF | `ce188ab` | Xong, **chờ Phúc duyệt** |
| A6 giữ dòng log | — | Giữ nguyên |
| A.8 bước 0 tag `midterm-v0` | — | **Chưa làm**, cần Hưng tag commit `39b4764` |

Ngày 25/11/2026 lịch sử nhánh được viết lại để sửa commit message; nội dung file của từng commit không đổi. Vì vậy mã commit trong bảng này khác mã ghi trong log kiểm tra ở `verify_partA/results/`.

### Kết quả kiểm tra trên Colab

Bộ kiểm tra nằm ngoài repo, trong thư mục `verify_partA/` cạnh repo. Đó là một notebook chứa sẵn mã nguồn của các mốc commit và chạy nền trên Colab. Log và JSON tổng hợp của 2 lượt lưu trong `verify_partA/results/`.

| Kiểm tra | Lượt | Kết quả |
|---|---|---|
| 8 ảnh raw tải từ Drive, so SHA256 với `dataset/prepared/manifest.json` | 1, 2 | Trùng 8/8 |
| `main` chạy trên Colab so với notebook Midterm | 1 | 12/12 cặp thành công như Midterm; 4 MAE medium trùng (11.951 / 3.824 / 9.051 / 22.330). Notebook Midterm chạy trên Windows và không lưu H hay ảnh từng cặp, nên chỉ so được hai chỉ số này |
| `main → A1 → A2 → A5` trên 12 cặp thật + synthetic | 1 | Trùng từng byte: H, panorama, log |
| Commit sửa `hypot` so với commit trước | 2 | 12/13 trùng từng byte; `small_site3_custom` có H lệch 6e-12 px, log và panorama vẫn trùng |
| Bảng BRIEF so với `std::normal_distribution` của libstdc++ | 1, 2 | Trùng 256/256 hàng |
| Build g++ `-Wall -Wextra` mọi mốc; `tools/build.py` (nvcc), `make cuda`, `make cpu` | 1, 2 | Đều qua, không có cảnh báo từ code của nhóm |
| `--dump-dir` (medium ×4, `small_site4`, synthetic): match tính lại bằng brute-force theo luật V0, inlier tính lại từ H, canvas/seam/blend | 1, 2 | Đạt mọi cặp; inlier lệch 0 |
| Thứ tự keypoint trong dump | 1, 2 | Từ A4 trở đi đã sắp theo (response giảm, y, x) |
| Synthetic | 1, 2 | Sai số < 1e-10 px ở mọi mốc |
| `tools/benchmark.py` của HEAD (binary nvcc, runs=5, warmup=1) | 2 | 12/12 cặp, correctness đạt; H trùng bản build bằng g++ |
| Hàm `hd::` trên GPU so với CPU (8.192 trường hợp, `--fmad=false`) | 2 | Trùng từng bit ở mọi hàm |

### Phát hiện cần nhóm biết

1. **`hypot`:** ở lượt 1, `hd::dlt_fit` trên GPU lệch CPU ở 2.917/8.107 trường hợp dù đã `--fmad=false`, vì `hypot` của CUDA và glibc làm tròn khác nhau (lệch 2.298/8.192 lần gọi). Commit `4fcd8a5` thay bằng `sqrt(dx*dx + dy*dy)`, và lượt 2 lệch 0. Quy tắc rút ra: hàm `HD` chỉ dùng `+ − × ÷` và `sqrt`; không thêm `-march=native` hay `-ffast-math` vào build V0.
2. **FMA:** khi để FMA mặc định, `dlt_fit` lệch ở mọi trường hợp, `sample_bilinear` lệch 3.825/7.025 điểm, `sqrt_sum` lệch 34 lần. Vì vậy mọi bản GPU phải build với `--fmad=false` (mục 6).
3. **A3/A4 làm đổi kết quả ở các cặp ít inlier.** Lượt 2 cho đúng số liệu của lượt 1, nghĩa là kết quả xác định; nó chỉ đổi theo chuỗi mẫu RANSAC và thứ tự keypoint. MAE vùng chồng lấp giảm nhẹ hoặc giữ nguyên, không thấy dấu hiệu chất lượng kém đi. Panorama của V0 Final sẽ khác Midterm ở các cặp có H lệch dưới đây, và notebook Final phải nêu rõ điều này.

   A4 đổi cả danh sách match ở những cặp không bị cắt Top-K: `large_site2_outdoor` (1196/1252 keypoint) từ 526 lên 527 match, `large_site3_custom` (838/236) từ 78 xuống 77. Lượt 1 cho thấy số match chỉ đổi ở commit A4. Nguyên nhân: ở chiều ngược, khi nhiều điểm A cùng khoảng cách nhỏ nhất tới một điểm B, V0 chọn `i` nhỏ nhất, mà `i` phụ thuộc thứ tự keypoint; vì vậy kết quả kiểm tra hai chiều (mutual) đổi theo. Chiều thuận không bị ảnh hưởng, vì khi bằng nhau ở best thì ratio bằng 1 và match bị loại.

   | Cặp | `main`: inlier, MAE, blend | HEAD: inlier, MAE, blend | H lệch (mean / max px) |
   |---|---|---|---|
   | small_site1_indoor | 20/38, 9.76, unchanged/feather | 21/38, 8.98, unchanged/feather | 29.3 / 193 |
   | small_site2_outdoor | 141/142, 3.56, affine/feather | 140/142, 3.55, affine/feather | 2.4 / 11 |
   | small_site3_custom | 11/12, 9.30, affine/feather | 11/12, 9.30, affine/feather | < 1e-10 |
   | small_site4_challenging | 259/273, 18.08, unchanged/optimal | 259/273, 18.08, unchanged/optimal | < 1e-10 |
   | medium_site1_indoor | 26/58, 11.95, unchanged/feather | 25/58, 11.67, unchanged/feather | 12.4 / 72 |
   | medium_site2_outdoor | 215/239, 3.82, affine/feather | 215/239, 3.82, affine/feather | < 1e-10 |
   | medium_site3_custom | 23/28, 9.05, affine/feather | 23/28, 9.05, affine/feather | < 1e-10 |
   | medium_site4_challenging | 173/196, 22.33, unchanged/optimal | 171/195, 22.14, unchanged/optimal | 0.7 / 6 |
   | large_site1_indoor | 22/46, 10.50, unchanged/feather | 21/46, 9.32, unchanged/feather | 39.0 / 271 |
   | large_site2_outdoor | 344/526, 4.80, affine/feather | 341/527, 4.81, affine/feather | 1.7 / 10 |
   | large_site3_custom | 53/78, 9.83, affine/feather | 52/77, 9.61, affine/feather | 9.2 / 51 |
   | large_site4_challenging | 76/120, 28.03, unchanged/optimal | 74/120, 27.18, unchanged/optimal | 52.4 / 388 |
   | synthetic | 1343/1347, 0.00, unchanged/feather | 1343/1347, 0.00, unchanged/feather | < 1e-10 |

4. **Pixel ở mép canvas:** ngay cả khi H chỉ lệch 1e-11 px, khoảng 0,016% số pixel ở mép canvas vẫn bị đổi giữa "trong" và "ngoài" ảnh (lệch tới 255). Tiêu chí so pixel giữa V0 và V1 phải tính đến điều này, trừ khi H trùng từng bit.
5. **Thời gian V0 trên Colab** (benchmark lượt 2, trung bình 4 site): small 738 ms, medium 1.108 ms, large 3.250 ms. Stage chậm nhất là matching ở small/medium và keypoint ở large. Số này chỉ để định hướng; bảng speedup Final phải đo lại trong cùng phiên với V1.

### Khác so với kế hoạch

- A1 sửa cả hằng stage 1–3 trong `v0_frontend.cpp`, giá trị mặc định trong `versions.h` và danh sách `HEADERS` trong Makefile.
- A2: `dlt_fit` dùng `sqrt(dx*dx + dy*dy)` thay `hypot` (commit `4fcd8a5`, phát hiện 1).
- A5: `PipelineTrace` có thêm `descA/descB` để so descriptor từng bit (assignment 9.1); `BlendInfo` ghi cả `canvas_w/h`, `offset_x/y`.
- A3: giới hạn 64 lần rút là hằng `hd::kRansacMaxDrawAttempts`.
- A4: bảng BRIEF được sinh bằng cách mô phỏng `std::normal_distribution<float>` của libstdc++; Colab đã xác nhận trùng 256/256 hàng. Generator `tools/gen_brief_pattern.cpp` sinh lại bảng bằng thư viện thật.
- Theo yêu cầu, code mới không có comment, kể cả comment đầu file ghi tác giả/nguồn.

### Việc còn lại để đóng Phần A

- [x] TV1 push nhánh (08/11/2026).
- [x] TV1 mở PR vào `main`, kèm tóm tắt mục này (25/11/2026).
- [ ] Phúc duyệt A4 (`v0_frontend.cpp`, `brief_pattern.h`).
- [ ] Hưng tag `midterm-v0` vào commit `39b4764`, và duyệt các thay đổi ở `common.h`, `versions.h`, `main.cu`, Makefile.
- [ ] Sau khi merge: Hưng chạy lại benchmark V0 cho notebook Final. Notebook ghi rõ V0 Final khác V0 Midterm ở A3/A4 (bảng ở phát hiện 3) và trích dẫn splitmix64 (Steele, Lea, Flood, OOPSLA 2014) ở hai nơi: mục 3 stage 5, chỗ mô tả RANSAC, và mục References.

## Trạng thái tích hợp trên nhánh `v1`

Đã ghép frontend và backend CUDA qua `--version 1` và bổ sung lệnh đối chiếu
V0/V1 cùng benchmark trong README. Stage 6 tính kích thước canvas và H nghịch
đảo trên GPU; host kiểm tra H và giới hạn kích thước trước khi cấp phát.
Kết quả Colab dưới đây thuộc bài test backend độc lập. Kết quả nghiệm thu V1
tích hợp sẽ được cập nhật sau khi chạy trên GPU.

## Trạng thái Phần B ở mốc backend độc lập (25/11/2026, nhánh `feat/v1-basic-cuda`)

**Kết luận:** M1 (kernel stage 4–6) và chương trình kiểm tra độc lập B.3 đã xong và đạt lượt Colab 1 (Tesla T4, 25/11/2026). Trên 12 cặp thật và synthetic, backend GPU cho match, inlier, H, canvas, quyết định blend, `seam_x` và panorama trùng từng bit với V0, và compute-sanitizer không báo lỗi. Phần còn lại của M1 là các ca lỗi B.3. Chưa làm M2 (`run_v1`), vì phần này cần frontend GPU của Phúc và CLI `--version 1` của Hưng. Nhánh đã push và mở draft PR với base là `refactor/v0-baseline-lock`; khi PR Phần A được merge thì đổi base sang `main`.

| Mục | File | Trạng thái |
|---|---|---|
| Quy tắc từng phần tử dùng chung CPU/GPU (matching, inlier, canvas, exposure, seam, blend) | `src/hd_backend.h` (mới), `src/hd_math.h` | Xong |
| V0 gọi các hàm dùng chung, kết quả không đổi | `src/v0_backend.cpp` | Xong, trùng từng byte với nhánh Phần A trên Colab (12 cặp + synthetic) |
| Tách 3 bước kiểm tra khỏi `run_v0` để V1 dùng lại, log giữ nguyên | `src/v0_stitching.cpp`, `src/versions.h` | Xong, [Cần nhóm duyệt]: file của Hưng |
| Buffer RAII tối thiểu | `src/device_buffer.cuh` (mới) | Xong; Hưng có thể gộp vào `gpu_buffers.cuh` |
| Kernel stage 4–6 + lớp `GpuBackend` | `src/gpu_backend.cuh`, `src/gpu_backend.cu` (mới) | Xong, đạt lượt Colab 1 |
| Kiểm tra độc lập B.3 | `src/tests/test_backend.cu` (mới) | Xong, đạt lượt Colab 1; còn các ca lỗi |
| Build `make test-backend` (`--fmad=false -arch=sm_75`), bỏ binary khỏi Git | `src/Makefile`, `.gitignore` | Xong, [Cần nhóm duyệt]: file của Hưng |

### Mapping thread/block đã cài

| Stage | Kernel | Mapping |
|---|---|---|
| 4 | `match_forward_kernel` / `match_reverse_kernel` | 1 thread cho 1 descriptor A (hoặc B), duyệt toàn bộ ảnh kia bằng `hd::hamming256` (8 × `__popc`) |
| 4 | `match_flag_kernel` → `compact_matches_kernel` | 1 thread cho 1 A ghi cờ; 1 block 1024 thread prefix sum có chia đoạn, ghi match theo `queryIdx` tăng dần |
| 5 | `gather_points_kernel` | 1 thread cho 1 match, gom tọa độ B (nguồn) và A (đích) |
| 5 | `hypothesis_kernel` | 1 thread cho 1 hypothesis: `ransac_sample` + `dlt_fit` 4 điểm + `invert_h` |
| 5 | `score_kernel` | 1 block (256 thread) cho 1 hypothesis, H và H⁻¹ nạp vào shared memory, reduction đếm inlier |
| 5 | `select_kernel` | 1 block: (số inlier lớn nhất, iter nhỏ nhất) |
| 5 | `inlier_flag_kernel` → `compact_indices_kernel` | 1 thread cho 1 match; gom inlier theo thứ tự |
| 5 | `refit_kernel` | 1 thread gọi `hd::dlt_fit` trên inlier theo đúng thứ tự V0; lặp 2 lượt |
| 6 | `exposure_stats_kernel` → `exposure_fit_kernel` | 1 thread cho 1 điểm lấy mẫu (bước 2); reduction 2 bước có thứ tự cố định (12 tổng + số mẫu) |
| 6 | `exposure_error_kernel` → `exposure_decide_kernel` | Như trên cho raw/corrected error; 1 thread quyết định exposure và seam |
| 6 | `seam_cost_kernel` | 1 thread cho 1 pixel canvas (khối 16×16) |
| 6 | `seam_dp_kernel` | 1 block 1024 thread, lặp theo hàng, `__syncthreads_or` cuối mỗi hàng; argmin + truy vết trong cùng kernel |
| 6 | `warp_blend_kernel` | 1 thread cho 1 pixel đầu ra (khối 16×16), gọi `hd::warp_blend_pixel` |

Host chỉ đọc về các số dùng để in log và rẽ nhánh như V0: số match, (iter, số inlier) của hypothesis tốt nhất, trạng thái refit, số inlier mỗi lượt, H, và quyết định exposure/seam. Canvas và H⁻¹ tính trên host bằng đúng hàm `hd::canvas_layout` và `hd::invert_h` mà V0 dùng.

### Kết quả kiểm tra trên Colab, lượt 1 (25/11/2026)

Môi trường: Tesla T4 (`sm_75`), nvcc 13.0, g++ 13.3, compute-sanitizer 2025.3.1. Bộ kiểm tra nằm ngoài repo: notebook `verify_partB/verify_partB.ipynb`, log và JSON lưu ở `verify_partB/results/round1_*`. Notebook build cùng lúc nhánh Phần A và nhánh này trong một phiên, chạy trên 12 cặp thật và synthetic. Cả 9 mục kết luận đều đạt.

| Kiểm tra | Kết quả |
|---|---|
| Build g++ V0 của nhánh Phần A và nhánh này | Qua, không cảnh báo |
| `make test-backend` (`--fmad=false -arch=sm_75`), `tools/build.py`, `make cpu` | Qua; 1 cảnh báo nvcc #177: hàm `JsonWriter::write(const char*)` trong `test_backend.cu` không được dùng |
| V0 của nhánh này so với nhánh Phần A, 12 cặp + synthetic | Trùng từng byte: log, H, panorama |
| `test_backend`, 13 ca | 13/13 đạt mức nghiêm ngặt: match, inlier, H (từng bit), canvas, quyết định exposure/seam, `seam_x` và panorama trùng V0; gain/bias trùng; `overlap MAE` lệch tối đa 6,3e-12. Chạy nối liền 3 stage cũng trùng |
| compute-sanitizer `memcheck`, `racecheck`, `initcheck`, `synccheck` trên `small_site4_challenging` (seam optimal) và `small_site2_outdoor` (exposure affine) | 0 lỗi, 0 hazard |
| ptxas | 19 kernel, không spill. Nặng nhất là `hypothesis_kernel` (232 thanh ghi) và `refit_kernel` (180 thanh ghi), mỗi kernel 576 B stack |

Thời gian (ms). V0 là 1 lần chạy của `run_v0`. V1 là trung bình 5 lần sau 1 lần warmup, đo bằng `GpuTimer`, buffer được giữ lại giữa các lần chạy. H2D gồm 2 ảnh RGB, keypoint và descriptor. D2H là chép panorama về bộ nhớ pageable.

| Cặp | Canvas | V0 s4 / s5 / s6 | V1 H2D | V1 s4 | V1 s5 | V1 s6 | V1 D2H | V1 tổng |
|---|---|---|---|---|---|---|---|---|
| small_site1_indoor | 2405×1298 | 22.0 / 3.0 / 107.9 | 1.2 | 0.36 | 2.25 | 4.2 | 2.0 | 10.0 |
| small_site2_outdoor | 1636×858 | 38.6 / 7.3 / 87.7 | 1.3 | 0.43 | 3.77 | 2.3 | 0.9 | 8.7 |
| small_site3_custom | 2079×1068 | 3.6 / 2.0 / 93.0 | 1.2 | 0.14 | 2.59 | 3.1 | 1.2 | 8.3 |
| small_site4_challenging | 2314×1238 | 1596.8 / 12.9 / 162.3 | 1.2 | 1.80 | 3.88 | 12.2 | 1.7 | 20.9 |
| medium_site1_indoor | 3559×1815 | 65.8 / 3.5 / 230.3 | 2.7 | 0.48 | 1.79 | 7.7 | 3.9 | 16.6 |
| medium_site2_outdoor | 2463×1295 | 140.4 / 10.8 / 188.6 | 2.7 | 0.51 | 3.14 | 3.2 | 1.7 | 11.4 |
| medium_site3_custom | 3062×1546 | 14.9 / 2.6 / 211.3 | 2.7 | 0.34 | 2.66 | 8.1 | 2.9 | 16.8 |
| medium_site4_challenging | 3475×1861 | 1587.9 / 9.1 / 361.1 | 2.8 | 1.81 | 3.08 | 26.4 | 3.9 | 38.0 |
| large_site1_indoor | 7120×3841 | 277.1 / 3.3 / 1016.0 | 12.0 | 1.49 | 2.36 | 36.2 | 45.3 | 97.4 |
| large_site2_outdoor | 4884×2568 | 1081.2 / 21.9 / 733.1 | 10.9 | 2.32 | 8.00 | 20.1 | 12.7 | 54.1 |
| large_site3_custom | 6220×3204 | 138.3 / 4.2 / 1118.0 | 11.1 | 0.85 | 2.43 | 26.8 | 25.6 | 66.9 |
| large_site4_challenging | 7055×3666 | 1756.5 / 8.5 / 2086.7 | 11.6 | 1.38 | 1.64 | 72.4 | 17.6 | 104.6 |
| synthetic | 11210×3916 | 1597.8 / 41.4 / 2000.5 | 36.7 | 1.70 | 14.52 | 34.7 | 76.1 | 163.8 |

Nhận xét để chọn hướng tối ưu cho V2/V3:
- Stage 6 chiếm phần lớn thời gian kernel, nhất là ở các cặp có seam optimal (`large_site4_challenging` 72 ms).
- Ở canvas lớn, D2H chiếm tới gần một nửa tổng thời gian (`large_site1_indoor` 45/97 ms). Tốc độ chép pageable chỉ 1,7–5,5 GB/s và chậm nhất ở canvas lớn, nên pinned memory là hướng đáng thử.
- Stage 5 gần như không đổi theo kích thước ảnh (1,6–14,5 ms), vì chi phí phụ thuộc số match chứ không phụ thuộc số pixel.

Các số này chỉ để định hướng. Bảng speedup Final phải đo lại bằng `run_v1` và `tools/benchmark.py` trong cùng một phiên với V0.

### Kết quả kiểm tra trên máy (25/11/2026)

Bộ kiểm tra nằm ngoài repo, trong `verify_partB/local/run_local.py`. Không có GPU NVIDIA, nên kernel được chạy bằng giả lập: mỗi block chạy bằng `blockDim` thread hệ điều hành, `__syncthreads` là barrier thật (dừng có báo lỗi nếu các thread của một block không cùng tới barrier), bộ nhớ cấp phát được điền rác `0xCD`. Build bằng `-ffp-contract=off` để giống ngữ nghĩa `--fmad=false`. Dữ liệu là 5 cặp tự sinh từ `synthetic_GT.HEIC`, phủ đủ các nhánh: dịch thuần, warp phối cảnh kèm lệch sáng (exposure affine), thị sai nhẹ, vật che khuất (seam optimal, MAE 27,7), ảnh nhỏ xoay 3°.

| Kiểm tra | Kết quả |
|---|---|
| Comment trong dòng code mới | 0 |
| clang `-x cuda -fsyntax-only` phía host và `sm_75`, `-Wall -Wextra` | Không lỗi, không cảnh báo; đã thử cài lỗi (constructor host, `std::isfinite`, `std::vector` trong kernel) và lint đều bắt được |
| V0 sau refactor so với nhánh Phần A | Trùng từng byte trên 5 cặp: H, panorama, log |
| `test_backend` giả lập, 5 cặp | Đạt cả 5 ở mức nghiêm ngặt: match, inlier, H (từng bit), canvas, quyết định blend, `seam_x`, panorama trùng V0; chạy nối liền 3 stage cũng trùng. `overlap MAE` lệch ≤ 6e-14 (thứ tự cộng khác), gain/bias trùng |
| Kiểm tra đột biến (cài lỗi vào bản sao `gpu_backend.cu`, cặp `occluder_seam`) | Bỏ điều kiện mutual: bắt được (match khác). Hỏng `first_y` của seam DP: bắt được (`seam_x` và pixel khác). Đổi tie-break khi chọn hypothesis: không đổi kết quả, vì các hypothesis cùng số inlier có cùng tập inlier và refit chỉ dùng tập inlier |

### Khác so với kế hoạch

- **Refit (B.0 #7, B.2):** dùng 1 thread gọi `hd::dlt_fit` theo thứ tự inlier của V0 thay cho reduction tâm/ATA. Có tối đa 1500 inlier nên chi phí nhỏ, đổi lại H trùng từng bit với V0, tránh lệch pixel ở mép canvas (phát hiện 4 ở Phần A). Reduction song song cho refit có thể là một hướng tối ưu cho V2/V3 nếu đo thấy refit đáng kể.
- **Exposure:** reduction 2 bước có thứ tự cố định nên kết quả xác định, nhưng thứ tự cộng khác V0. Tổng có thể lệch ở bit cuối; trên 5 cặp thử ở máy, chỉ `overlap MAE` lệch cỡ 1e-14; trên 13 ca ở Colab, `overlap MAE` lệch tối đa 6,3e-12, gain/bias và pixel không đổi. Tiêu chí pixel ±1 ở mục 6 vẫn cần cho trường hợp này.
- **Canvas và H⁻¹:** tính trên host bằng hàm dùng chung (4 phép chiếu góc và 1 nghịch đảo 3×3), không dùng kernel `canvas_bounds`. H phải về host để kiểm tra canvas như V0, nên làm thêm 1 kernel cũng không bớt được lần chép nào.
- **Seam DP:** chọn phương án 1 block lặp theo hàng (đã ghi là hợp lệ ở B.2), host không phải điều phối từng hàng.
- **Giao diện:** `GpuFeatures.count` là số trên host thay vì con trỏ device, vì V1 vẫn phải in `Keypoints: A=..., B=...` nên số keypoint luôn phải về host. Mọi buffer làm việc thuộc `GpuBackend` và được giữ lại giữa các lần chạy (đo ở trạng thái ổn định); cấp phát chỉ xảy ra ở lần đầu.
- **Refactor V0:** các quy tắc từng phần tử của stage 4–6 chuyển vào `hd_backend.h`, V0 và V1 gọi cùng hàm. Ba bước kiểm tra `check_match_count`, `check_ransac_consensus`, `check_canvas_bounds` được tách khỏi `run_v0` để `run_v1` dùng lại, log giữ nguyên.

### Việc tiếp theo

- [x] Lượt Colab 1: chạy `verify_partB/verify_partB.ipynb` (T4). Đạt 9/9 mục (25/11/2026).
- [ ] Ca lỗi B.3 (assignment 9.3): ít hơn 4 match, descriptor rỗng, H suy biến, canvas quá lớn.
- [ ] Sửa cảnh báo nvcc #177: đổi `JsonWriter::write(const char*)` thành `= delete`. Nếu xóa hẳn hàm này, một chuỗi literal truyền vào `field()` sẽ âm thầm gọi `write(bool)`. Kiểm lại trên Colab rằng cảnh báo đã hết. Sửa luôn bảng ptxas của bộ kiểm tra (ngoài repo) đang in thiếu tên kernel.
- [ ] Cả nhóm chốt mục 6: tolerance V0/V1 và chính sách cấp phát trong hay ngoài vùng đo.
- [ ] Chốt với Hưng: `--version 1`, các trường timing H2D/D2H trong `StageTimings`, gộp `device_buffer.cuh` vào `gpu_buffers.cuh`, target build cho V1.
- [ ] Chốt với Phúc: frontend GPU trả `Keypoint*`, `Descriptor256*` trên device và số keypoint trên host, theo thứ tự A4; bảng BRIEF cần bản `__constant__`.
- [ ] M2: `run_v1` trong `src/gpu_pipeline.cu`, dùng `GpuBackend` và các hàm `check_*`.

## 0. Tóm tắt

- **Vai trò:** back-end stages 4–6 (matching, RANSAC, warp/blend); chủ trì ghép frontend ↔ backend và **nghiệm thu V1 Basic CUDA**.
- **Hiện trạng lúc lập kế hoạch (08/11/2026):** V0 chạy được 12/12 cặp và synthetic đạt. Repo **chưa có dòng CUDA nào**.
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

## 3. Hiện trạng so với phần việc (lúc lập kế hoạch, 08/11/2026)

| Hạng mục | Hiện có | Còn thiếu |
|---|---|---|
| V0 backend [v0_backend.cpp](src/v0_backend.cpp) | Đủ stage 4–6. Notebook Midterm ghi 12/12 cặp thành công, sai số chiếu synthetic khoảng 1e-11 px | Tham số viết cứng rải rác; GPU không tái tạo được RNG |
| CUDA stage 4–6 | Chưa có, chưa có `gpu_backend.cu` | Toàn bộ |
| `run_v1` / `gpu_pipeline.cu` | Chỉ có prototype bị comment trong [versions.h](src/versions.h#L65) | Toàn bộ |
| CLI | [main.cu:133](src/main.cu#L133) từ chối mọi version khác 0; `--runs` mặc định 1 | Phần của Hưng, TV1 cần để chạy V1 |
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
| A2 | Tách các hàm toán dùng chung CPU/GPU | `src/hd_math.h` (mới), [v0_backend.cpp](src/v0_backend.cpp) | Gần như không: bản tách `acb2e4f` trùng từng byte; bản sửa `hypot` `4fcd8a5` làm H của `small_site3_custom` lệch 6e-12 px, log và panorama vẫn trùng | TV1 |
| A3 | Thay RNG của RANSAC | `src/hd_math.h`, [v0_backend.cpp](src/v0_backend.cpp) | **Có**: mẫu, inlier, H | [Cần nhóm duyệt] |
| A4 | Thứ tự keypoint xác định | [v0_frontend.cpp](src/v0_frontend.cpp) (của Phúc) | **Có**: thứ tự keypoint; tập keypoint ở biên Top-K; danh sách match, kể cả khi không cắt Top-K (phát hiện 3) | [Cần nhóm duyệt]; Phúc sửa hoặc duyệt |
| A5 | Xuất dữ liệu trung gian để đối chiếu | [versions.h](src/versions.h), [common.h](src/common.h), [v0_backend.cpp](src/v0_backend.cpp), [main.cu](src/main.cu) | Không | [Cần nhóm duyệt]: giao diện của Hưng |
| A6 | Giữ nguyên các dòng log | — | — | Ràng buộc |

## A1 — Header cấu hình chung

**Vấn đề:** tham số đang nằm ở ba nơi:
- trong lời gọi hàm ở `v0_stitching.cpp` dòng 52, 70, 85 (bản `main`);
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
- Dùng helper tự viết `hd_max(a,b) = (a < b) ? b : a` và `hd_min(a,b) = (b < a) ? b : a`, đúng ngữ nghĩa `std::max/min`. Dùng `fabs`, `sqrt`, `isfinite` của `<cmath>` vì đều có trên device. Không dùng `hypot`: CUDA và glibc làm tròn khác nhau (xem phát hiện 1 ở mục Trạng thái).
- **Giữ đúng thứ tự phép tính và các chỗ ép kiểu** như V0, gồm: ép `float` trong `project_point` (nay ở [hd_math.h:229-236](src/hd_math.h#L229-L236)) và thứ tự tính tâm → khoảng cách trung bình → cộng dồn ATA (nay ở [hd_math.h:185-207](src/hd_math.h#L185-L207)).
- `solve_homography_points` trong V0 đổi sang nhận con trỏ + số phần tử. `sample_bilinear` nhận con trỏ dữ liệu ảnh thay vì `ImageRGB`.

**Kiểm tra:** output trùng từng byte với mốc (A.8 bước 2). Không cần thay đổi cờ biên dịch. Riêng commit `4fcd8a5` (bỏ `hypot`) cố ý đổi kết quả rất nhỏ; xem phát hiện 1 ở mục Trạng thái.

## A3 — RNG của RANSAC dùng chung CPU/GPU [Cần nhóm duyệt]

**Vấn đề:** `v0_backend.cpp` dòng 209–210 (bản `main`) dùng `std::mt19937` + `std::uniform_int_distribution`. Cách distribution biến số ngẫu nhiên thành chỉ số do thư viện chuẩn tự quyết, nên GPU không tái tạo được chuỗi mẫu. Hai thư viện khác nhau dù cùng seed cũng không chắc cho cùng chuỗi (assignment 5.3).

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
- **Trích dẫn** splitmix64 (Steele, Lea, Flood, "Fast Splittable Pseudorandom Number Generators", OOPSLA 2014) trong notebook ở hai nơi: chỗ mô tả RANSAC (mục 3, stage 5) và mục References, theo mục 6 của đề Final. Code không có comment (mục 8).
- Phương án dự phòng nếu nhóm không muốn sửa V0: sinh sẵn bảng 1500 × 4 chỉ số ở host rồi upload. Khi đó phải ghi rõ phạm vi việc làm trên CPU và tính thời gian vào H2D.

**Kiểm tra:** 12/12 cặp vẫn thành công; synthetic dưới 1 px; ghi bảng số inlier mới/cũ và sai số chiếu giữa H mới và H Midterm (A.8 bước 4).

## A4 — Thứ tự keypoint xác định [Cần nhóm duyệt — file của Phúc]

**Vấn đề:** score FAST là tổng các số nguyên ([v0_frontend.cpp:149-153](src/v0_frontend.cpp#L149-L153)), nên các điểm **bằng response xảy ra thường xuyên**. Thêm hai điều nữa:
- `std::partial_sort` không ổn định (dòng 181–187 bản `main`), nên thứ tự giữa các điểm bằng nhau không xác định;
- khi có ≤ 1500 điểm thì không sắp xếp, giữ thứ tự quét raster.

Frontend GPU gom điểm bằng atomics sẽ ra thứ tự khác. Thứ tự keypoint đổi kéo theo chỉ số match đổi, mẫu RANSAC đổi và H khác, dù backend hoàn toàn đúng.

**Đề xuất:** luôn sắp theo khóa toàn phần (response giảm dần, y tăng, x tăng) bằng `std::sort`, rồi cắt lấy K điểm đầu. Nếu Phúc muốn giữ thứ tự raster khi ≤ K thì cũng được, miễn V0 và V1 dùng cùng một quy tắc.

**Đề xuất thêm cho Phúc:** pattern BRIEF hiện sinh bằng `std::mt19937` + `std::normal_distribution` (dòng 200–212 bản `main`), nên GPU cũng không tái tạo được. Nên sinh pattern một lần từ V0 rồi ghi thành mảng hằng 256 × 4 dùng chung (assignment 5.2).

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
- `Blend: exposure=<affine|unchanged>, overlap MAE=<x>, seam=<optimal|feather>` ([v0_backend.cpp:287-290](src/v0_backend.cpp#L287-L290))
- Các thông báo lỗi trong [v0_stitching.cpp:84-147](src/v0_stitching.cpp#L84-L147)

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
3. **Cách chọn khi khoảng cách bằng nhau trong matching** ([35-46](src/v0_backend.cpp#L35-L46), [52-60](src/v0_backend.cpp#L52-L60)):
   - Chiều thuận lấy `j` nhỏ nhất.
   - Khoảng cách bằng best được ghi vào `second`, nên ratio bằng 1 và match bị loại.
   - Chiều ngược lấy `i` nhỏ nhất.
   - Giữ nguyên phép chia float `best/second < 0.75`, đừng đổi thành `best < 0.75*second`.
4. **Chọn hypothesis tốt nhất** ([136](src/v0_backend.cpp#L136)): V0 dùng `>` nghiêm ngặt, nên khi bằng số inlier thì iteration sớm nhất thắng. Reduction trên GPU so theo cặp (số inlier lớn nhất, iter nhỏ nhất).
5. **Ép float giữa chừng** trong `project_point`: dùng hàm chung (A2).
6. **FMA:** nvcc mặc định gộp phép nhân-cộng. Điểm nằm sát ngưỡng 3 px có thể đổi trạng thái inlier, và pixel có thể lệch 1 khi ép `uint8_t`. Đề xuất `--fmad=false` cho mọi bản GPU (mục 6).
7. **Cộng số thực theo thứ tự khác:** thống kê exposure và ATA khi refit đều là phép cộng dồn. Không dùng `atomicAdd` double vì mỗi lần chạy cho kết quả hơi khác. Refit: 1 thread gọi `hd::dlt_fit` theo đúng thứ tự inlier, nên H trùng từng bit với V0. Exposure: reduction 2 bước có thứ tự cố định, mỗi block tính một tổng con rồi một kernel cộng các tổng con theo thứ tự; kết quả xác định nhưng có thể lệch V0 ở bit cuối.
8. **Seam DP** ([291-333](src/v0_backend.cpp#L291-L333)):
   - Xét `d = -2..2` theo thứ tự với `<`, nên khi bằng nhau thì `d` nhỏ nhất thắng.
   - Hàng không có pixel hợp lệ bị bỏ qua và `previous` giữ nguyên.
   - Truy vết bắt đầu từ chỉ số đầu tiên mà `min_element` trả về.
9. **Thứ tự keypoint từ frontend:** theo quy tắc chung (A4).
10. **Log và H:** in đúng các dòng ở A6; trả H qua `outH` để `--save-h` hoạt động, vì phần kiểm tra synthetic đọc file này.

## B.1 M0 — Chốt baseline (= Phần A + giao diện)

- [x] Gửi nhóm đề xuất A1–A5 qua PR Phần A (25/11/2026).
- [ ] Thống nhất với Phúc và Hưng mục 5 (giao diện) và mục 6 (quyết định).
- [ ] Thực hiện A.8 bước 0 → 5: bước 0b → 5 đã xong (mục Trạng thái Phần A), còn bước 0 (Hưng tag `midterm-v0`).
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
- Số keypoint là `int` trên host (đã đổi so với bản nháp dùng con trỏ device, xem mục 5).
- Chỉ copy `match_count` về host để dừng khi có ít hơn 4 match. Ghi rõ đây là bước điều phối.

**Stage 5 — RANSAC + refinement**

| Kernel | Mỗi thread/block làm gì |
|---|---|
| `ransac_hypotheses` | 1 thread cho 1 hypothesis: `ransac_sample`, chuẩn hóa + `dlt_accumulate` + `dlt_solve` cho 4 điểm, `invert_h`; ghi H, H⁻¹, cờ hợp lệ |
| `ransac_score` | 1 block cho 1 hypothesis: các thread chia nhau duyệt match, block reduction đếm inlier |
| `ransac_select` | Argmax theo (số inlier, −iter) |
| `collect_inliers` | 1 thread cho 1 match, sau đó gom có thứ tự |
| `refit` | 1 thread chạy `hd::dlt_fit` trên các inlier theo thứ tự của V0; lặp 2 lượt như V0 (đã đổi so với bản nháp dùng reduction, xem mục Trạng thái Phần B) |
| Canvas | Host tính bằng `hd::canvas_layout` sau khi H về host để kiểm tra như `check_canvas_bounds` (đã đổi so với bản nháp có kernel `canvas_bounds`) |

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

**Giao diện hàm host (bản nháp ban đầu; bản đã cài là lớp `GpuBackend` trong `src/gpu_backend.cuh`, xem mục Trạng thái Phần B):**

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

- [x] Viết `src/tests/test_backend.cu`, hoặc một chế độ trong CLI:
  1. chạy frontend **V0**;
  2. upload keypoint và descriptor của V0 lên GPU;
  3. chạy stage 4–6 trên GPU;
  4. so với `PipelineTrace` của V0.
- [x] Các tiêu chí so sánh (đã cài trong `test_backend`; đạt 13/13 ca ở lượt Colab 1):
  - Danh sách match giống hệt: index, distance, thứ tự.
  - Tập inlier giống hệt.
  - H: sai số chiếu trên lưới điểm so với H của V0 (mean và max px).
  - Canvas cùng kích thước và offset.
  - Cùng quyết định exposure và seam; gain/bias; `seam_x` giống hệt.
  - Pixel: max |diff| và % pixel lệch trên vùng hợp lệ.
- [ ] Chạy `test_backend --check-errors` trên GPU. Bộ test đã có ca dưới 4 match, descriptor rỗng, keypoint thẳng hàng, H suy biến và canvas quá lớn.
- [x] Chạy `compute-sanitizer` với `memcheck`, `racecheck`, `initcheck`, `synccheck`: lượt Colab 1 không báo lỗi trên `small_site4_challenging` và `small_site2_outdoor`.
- [x] Synthetic: H của V1 trùng từng bit với V0, nên sai số chiếu bằng của V0 (< 1e-10 px).

## B.4 M2 — Tích hợp và nghiệm thu V1 (TV1 chủ trì)

- [x] Tích hợp `V1Pipeline::run_v1` trong `src/gpu_pipeline.cu`. CLI `--version 1` chạy H2D RGB → frontend GPU → backend GPU → D2H panorama, có `--save-h` và `--dump-dir`.
- [x] Frontend GPU dùng một thread mỗi pixel cho grayscale, hai pass Gaussian, FAST và NMS. Top-K sắp khóa response giảm rồi chỉ số pixel tăng bằng CUB radix sort. Mỗi thread ghi một BRIEF 256 bit từ bảng hằng dùng chung.
- [x] Dùng CUDA events cho mỗi stage và hai lần truyền ảnh chính. Dùng đồng hồ host cho end-to-end. Các lần đọc nhỏ thuộc thời gian từng stage; warmup làm nóng các buffer được tái sử dụng. Xem cách chạy `tools/verify_v1.py` và benchmark trong README.
- [ ] Không dùng `CHECK_KERNEL_ASYNC` khi đo thời gian: macro này thực ra có gọi `cudaDeviceSynchronize` ([cuda_utils.cuh:16](src/cuda_utils.cuh#L16)). Khi đo chỉ gọi `cudaGetLastError`; đồng bộ đầy đủ chỉ dùng ở chế độ debug.
- [x] Stream, timer và buffer dùng RAII trong pipeline V1.
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
- Số keypoint hợp lệ là `int` trên host, dung lượng tối đa `cfg::kMaxKeypoints`. Bản nháp dùng `int*` trên device, nhưng V1 phải in dòng `Keypoints: A=..., B=...` nên số này luôn phải về host. Backend nhận qua `GpuFeatures` trong `src/gpu_backend.cuh`.
- Bảng BRIEF cần bản `__constant__` sinh từ `src/brief_pattern.h`.
- Thứ tự keypoint theo A4.
- Descriptor 32 byte: bit `b` nằm ở byte `b/8`, vị trí `b%8`, như [v0_frontend.cpp:223](src/v0_frontend.cpp#L223).
- Stream và buffer (đề xuất theo bản đã cài): mọi hàm của `GpuBackend` nhận `cudaStream_t`; backend sở hữu buffer làm việc của stage 4–6; frontend sở hữu buffer ảnh, keypoint và descriptor, và phải giữ chúng tới hết stage 6.

**Với Hưng:**
- `--version 1` gọi `run_v1`; `--runs` mặc định 5.
- Các trường timing mới (H2D, D2H, tổng kernel, end-to-end) trong `StageTimings` và CSV.
- `gpu_buffers.cuh` (RAII). Hiện đã có `src/device_buffer.cuh` tối thiểu; Hưng gộp hoặc thay.
- Cờ build: `-arch`, `--fmad=false`; danh sách nguồn `.cu`. Target `test-backend` trong `src/Makefile` có thể dùng làm mẫu.
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
| Đã bổ sung fixture tự sinh khi không đọc được HEIC hoặc chạy `--procedural-synthetic` | [prepare_dataset.py](tools/prepare_dataset.py) | Hưng |
| README nhắc `tools/verify_cli.py`, `execute_notebook.py`, `package.py` nhưng repo không có | [README.md:12-14](README.md#L12-L14) | Hưng |
| Cell kiểm tra correctness (mục 3.5.1) dùng đường dẫn cứng `Path('A:\\PP\\Midterm')` → Run All trên Colab lỗi | Group22_Midterm.ipynb | Hưng |
| Bảng "Lộ trình phân công" trong notebook Midterm khác phân công ngày 03/11 | Group22_Midterm.ipynb | Hưng (notebook Final) |
| Đã nối CLI, build và benchmark cho V0/V1. Còn xác nhận trên Colab GPU | main.cu, Makefile, build.py, benchmark.py | Hưng |

## 8. Quy ước Git cho phần việc này

- **Nhánh:** tách từ `main` @ `39b4764`. Tên gợi ý theo assignment 12.1: `feat/v1-basic-cuda`.
- **Khuyến nghị:** Phần A làm thay đổi baseline dùng chung, nên mở PR riêng cho Phần A sớm và merge vào `main` trước, để Phúc/Hưng làm trên cùng baseline. Nếu chỉ dùng một nhánh thì đặt Phần A thành các commit đầu tiên, tách riêng theo đúng thứ tự A.8.
- **Commit đầu tiên:** chính file `PLAN.md` này.
- **Mỗi commit/PR** ghi: thay đổi gì, thuộc phiên bản nào, kiểm tra bằng cách nào, kết quả, giới hạn còn lại.
- Không commit binary, dataset, `outputs_*`, ảnh tạm (đã có trong `.gitignore`).
- Không đổi `common.h`, `versions.h`, `main.cu` hoặc schema timing mà chưa báo Hưng (assignment 12.1).
- Code mới không có comment, kể cả comment đầu file. Tác giả thể hiện qua commit; nguồn bên ngoài được trích dẫn trong notebook, tại chỗ dùng và trong References (đề Final mục 6).

## 9. Thứ tự bắt đầu

1. Đọc diff `37b72fe` và toàn bộ [v0_backend.cpp](src/v0_backend.cpp) cho tới khi tự giải thích được từng nhánh.
2. Gửi nhóm đề xuất A1–A5 cùng mục 5 và mục 6; nhờ Hưng tag `midterm-v0`.
3. Bước 0 → 3 của A.8 (mốc, `compare_runs.py`, A1, A2, A5). Các bước này không đổi kết quả, nên có thể làm trước khi nhóm duyệt xong A3/A4.
4. A3 (và A4 khi Phúc đồng ý), chạy lại V0 trong cùng phiên Colab.
5. Stage 4 GPU + kiểm tra độc lập trên Colab (máy Mac không có GPU NVIDIA).
6. Stage 5, rồi stage 6; để seam sau cùng vì khó nhất.
7. Ghép `run_v1` khi frontend của Phúc sẵn sàng; nghiệm thu theo B.4.

Kế hoạch không gắn ngày vì assignment.md ghi rõ chưa có deadline Finalterm được xác nhận.
