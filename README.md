# Nhóm 22 – phần việc thành viên 3

Bản thực hiện gồm khung chương trình, log màu, dữ liệu chuẩn hóa, kiểm thử tích hợp, profiling và notebook có output thật. `Group22_Midterm.ipynb` là notebook báo cáo duy nhất của nhóm.

## Chạy tại máy

```powershell
python -m pip install -r requirements.txt
python tools/prepare_dataset.py
python tools/prepare_dataset.py --synthetic-only --procedural-synthetic
python tools/build.py
python tools/verify_cli.py
python tools/execute_notebook.py
python tools/package.py
```

Nếu đã có đủ prepared/synthetic thì không cần chuẩn bị lại. Để đọc HEIC, cài `pillow-heif` bằng `requirements.txt`; không thêm thư mục chứa bản sao Pillow vào `sys.path` vì extension nhị phân phụ thuộc môi trường Python. `prepare_dataset.py` giữ nguyên ảnh raw và không ghi đè synthetic đã có. Nếu chỉ có một phần synthetic, script dừng để tránh phá dữ liệu.

CLI trên Windows:

```powershell
.\stitch.exe dataset/prepared/medium/site1_indoor_left.png dataset/prepared/medium/site1_indoor_right.png outputs_v0/panorama_site1_indoor.png --version 0 --runs 5 --warmup 1 --save-h outputs_v0/H.txt --timings-csv outputs_v0/timing.csv
```

Linux/Colab dùng `./stitch`. H luôn ánh xạ **right B → left A**. Mặc định warmup = 0; công cụ benchmark/notebook đặt warmup = 1 và runs = 5. Thư mục đầu ra phải tồn tại. CLI trả mã lỗi khi không ghép được hoặc không ghi được ảnh/H/CSV. `NO_COLOR=1` tắt ANSI; mặc định SUCCESS xanh, ERROR đỏ, WARN vàng, PROCESS trắng.

V0 có thể biên dịch không cần GPU/CUDA:

```bash
g++ -O3 -std=c++14 -x c++ src/main.cu src/stb_impl.cpp src/v0_frontend.cpp src/v0_backend.cpp src/v0_stitching.cpp -o stitch
# Hoặc: cd src && make cpu
# Có CUDA: cd src && make cuda
```

## V1 CUDA trên Colab

Mở [Group22_V0_V1_Colab.ipynb](Group22_V0_V1_Colab.ipynb) để chạy toàn bộ thực
nghiệm V0/V1, xem ảnh ghép và tải kết quả trong một notebook.

V1 chạy cả sáu stage qua `--version 1`. CPU đọc và ghi ảnh, quản lý buffer, kiểm tra
điều kiện chạy và nhận các số đếm phục vụ điều phối. Dùng Colab GPU T4 với CUDA
Toolkit và dữ liệu đã chuẩn bị như trên. Kiến trúc CUDA khác có thể chọn bằng
`--arch` hoặc `CUDA_ARCH` của Makefile.

```bash
python tools/build.py --cuda --arch sm_75
python tools/verify_v1.py --report outputs_v1/verification.json
(cd src && make test-backend && ./test_backend --check-errors)
python tools/benchmark.py --version 0 --warmup 1 --runs 5
python tools/benchmark.py --version 1 --warmup 1 --runs 5
```

Công cụ kiểm tra đối chiếu đầu ra từng stage trên 12 cặp thật và một cặp synthetic.
Nếu không đọc được HEIC, `python tools/prepare_dataset.py --synthetic-only --procedural-synthetic`
tạo cặp ảnh tự sinh với homography tịnh tiến đã biết.

Chạy hai benchmark trong cùng phiên Colab. Với mỗi cặp, lấy `total_pipeline_ms`
của V0 chia cho V1 để tính speedup. Trường này đo từ ảnh RGB trong RAM đến panorama
trong RAM. V1 đo sáu stage, H2D ảnh RGB và D2H panorama bằng CUDA events. Các lần
đọc số đếm nhỏ thuộc thời gian của stage tương ứng. `gpu_stage_sum_ms` là tổng thời
gian sáu stage. Có thể dùng CUDA trace trong Nsight Systems để đo riêng thời gian
kernel. Buffer GPU tăng dung lượng khi cần và được dùng lại sau warmup. Khởi tạo
stream và timer, đọc/ghi file và lần chạy thêm của `--dump-dir` nằm ngoài số đo
benchmark.

## V2 của Vũ Trần Phúc

[Notebook V2 trên Colab](Group22_V2_Colab.ipynb) chứa phần báo cáo, đối chiếu
V2 với V0 và V1, cùng benchmark 12 cấu hình ảnh thật và cặp synthetic.
Với `--version 2`, GPU tính bảng đường nối trên các cột có thể thuộc ảnh A.
Notebook lưu kết quả từng cặp, thời gian stage 6 và thời gian toàn pipeline.

## Dữ liệu và kết quả

- 8 ảnh raw đặt phẳng theo mẫu `dataset/raw/<category>_<left|right>.<ext>`; category là `indoor`, `outdoor`, `custom` hoặc `challenging`.
- 24 ảnh PNG: `dataset/prepared/{small,medium,large}/`; manifest ghi nguồn, SHA256, crop box và kiểm tra upscale.
- `dataset/synthetic/`: fixture tự sinh với H dịch chuyển 320 px, seed cố định; provenance ghi rõ không phải tài nguyên môn học.
- `outputs_v0/`: bốn panorama medium giữ toàn bộ canvas và bốn bản `*_cropped.png` để trình bày, CSV từng lần chạy, bảng tổng hợp, H, log, JSON ghi phần cứng/compiler/hash nguồn và biểu đồ. Crop nằm ngoài timing sáu stage và không thay đổi homography.
- `team.json`: thông tin nhóm, vai trò, địa điểm và URL dataset. TV1: Trần Nguyễn Minh Quân; TV2: Vũ Trần Phúc; TV3: Nguyễn Hữu Khánh Hưng.
- `dist/Group22_Midterm.zip`: notebook có output, src, tools và tài liệu hỗ trợ. `dist/Group22_Dataset.zip`: raw + prepared + synthetic. Dataset nằm ngoài ZIP mã nguồn theo yêu cầu link dataset trong đề.

## Colab

Giải nén hai ZIP vào cùng `/content/Group22_Midterm`, mở notebook và Run all. Cell tải dataset trong notebook sẽ đồng bộ lại tám raw images từ [Google Drive nhóm](https://drive.google.com/drive/folders/1KEfstxQ1an5bEngfaHfXl9wIdbXWo2wF?usp=sharing) trước khi chuẩn bị dữ liệu; cần cài `gdown` (đã thêm vào `requirements.txt`). Hoặc đặt các thư mục/file tương ứng vào `MyDrive/Group22_Midterm`; sửa `DRIVE_PROJECT` trong setup cell rồi Run all để mount và copy project. Notebook biên dịch trực tiếp từ `src/`, không sinh bản sao thuật toán C++ trong cell.

Đã chạy notebook bằng kernel Python trên Windows; chưa có phiên Colab để xác nhận trực tiếp. Đo lại trên Colab sẽ thay đổi số liệu theo CPU/compiler. Phần CUDA utilities chưa được chạy thử với nvcc/GPU tại máy này.

## Các việc còn phụ thuộc nhóm

1. Cung cấp synthetic pair chính thức từ môn học nếu được yêu cầu. Giữ đúng chiều H B→A; nếu GT là A→B thì nghịch đảo trước khi so sánh, cập nhật provenance và chạy lại notebook. Fixture dịch chuyển hiện tại chỉ kiểm tra tích hợp, không chứng minh trường hợp projective tổng quát.
2. Upload `Group22_Dataset.zip` vào [Drive nhóm](https://drive.google.com/drive/folders/1KEfstxQ1an5bEngfaHfXl9wIdbXWo2wF?usp=sharing). Quyền tải công khai của thư mục raw đã được xác minh ngày 2026-10-21.
3. Xác nhận phân loại `site2_outdoor`: ảnh tầng 2 tòa E hướng vào hành lang; `site4_challenging` hướng ra cây xanh. Chưa có chứng cứ việc đăng ký địa điểm với lớp. Không tự đổi nhãn thư mục hoặc khẳng định đã đạt tiêu chí đăng ký.
4. TV1/TV2 duyệt các giới hạn V0: frontend BRIEF chưa bất biến xoay/tỷ lệ và global homography vẫn gây méo phối cảnh. Stage 6 đã có bù phơi sáng affine có điều kiện và adaptive seam, nhưng không thể phục hồi hình học sai do parallax. Ảnh canvas đầy đủ và bản crop trình bày đều được giữ để báo cáo trung thực các giới hạn này.
