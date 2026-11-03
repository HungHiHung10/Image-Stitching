# Kế hoạch Finalterm — Nhóm 22: Panorama Image Stitching

Môn học: CSC14120 — Lập trình song song.

Cập nhật: 03/11/2026.

Mục đích: kế thừa Midterm V0, triển khai CUDA V1–V3 và hoàn thiện bài nộp cuối kỳ.

Tài liệu này thay thế kế hoạch phân công Midterm cũ. Đây là **kế hoạch triển khai**, không phải xác nhận rằng V1, V2, V3 đã được thực hiện hoặc đạt speedup. Không ghi thời hạn hoặc trọng số Finalterm khi chưa có thông báo được xác nhận.

## 1. Đọc nhanh: mỗi người làm gì?

| Thành viên | MSSV | Vai trò Midterm | Phiên bản chịu trách nhiệm Finalterm |
| --- | --- | --- | --- |
| Trần Nguyễn Minh Quân | 23120342 | Back-end, stages 4–6 | **V1 — Basic CUDA**: tích hợp pipeline GPU đầy đủ |
| Vũ Trần Phúc | 23120333 | Front-end, stages 1–3 | **V2 — Optimized CUDA**: dự kiến tối ưu matching bằng shared memory |
| Nguyễn Hữu Khánh Hưng | 23120271 | System, data, profiling, notebook | **V3 — Optimized CUDA**: dự kiến chạy frontend hai ảnh đồng thời bằng CUDA streams |

“Chịu trách nhiệm phiên bản” nghĩa là dẫn dắt thiết kế, tích hợp, kiểm tra, đo đạc và viết phần báo cáo của phiên bản đó. Mỗi người vẫn hỗ trợ các phiên bản khác theo chuyên môn; không chờ người trước hoàn thành rồi mới bắt đầu làm.

Ngay khi bắt đầu: Quân viết CUDA backend; Phúc viết CUDA frontend; Hưng làm khung chạy, đo đạc và kiểm tra. Sau khi tích hợp thành V1, Phúc dẫn dắt V2 rồi Hưng dẫn dắt V3.

## 2. Căn cứ và yêu cầu bắt buộc

Nguồn chính trong repository:

- [Đề Final Project bản tiếng Anh](<requirement/finalterm/CSC14120_2026_Final Project.pdf>): mục 3 về phiên bản, mục 4 về benchmark, mục 5 về sản phẩm nộp và mục 6 về trích dẫn/AI.
- [Đề bản tiếng Việt](<requirement/finalterm/CSC14120_2026_Final Project vi.pdf>): dùng để đọc đối chiếu.
- [Dataset requirements](<requirement/finalterm/Dataset requirements.pdf>): hướng dẫn dữ liệu đi kèm.

Phân biệt hai loại thông tin trong tài liệu này: **yêu cầu của đề** là tiêu chí phải đáp ứng; **đề xuất của nhóm** là lựa chọn kỹ thuật có thể điều chỉnh sau profiling.

### 2.1. Bốn phiên bản tối thiểu

| Phiên bản | Yêu cầu của đề | Cách hiểu dễ tiếp cận |
| --- | --- | --- |
| V0 | C/C++ đơn luồng, tuần tự | Bản chuẩn để đối chiếu kết quả và thời gian |
| V1 | Mọi stage chạy trên GPU bằng CUDA cơ bản | Chuyển cùng công việc của V0 sang GPU, ưu tiên đúng trước |
| V2 | Kế thừa V1, thêm một ý tưởng tối ưu chính | Đo V1, xác định chỗ chậm, tối ưu chỗ đó |
| V3 | Kế thừa V2, thêm một ý tưởng tối ưu chính | Đo lại V2 rồi xử lý bottleneck tiếp theo |

V0–V3 phải dùng cùng phương pháp và tham số để so sánh công bằng. Mỗi phiên bản tối ưu phải có kết quả nhất quán với V0. Nếu bỏ một tối ưu đã có, notebook phải giải thích. Một tối ưu chạy chậm hơn vẫn có thể được báo cáo nếu có số liệu và phân tích trung thực.

Biên dịch bằng `nvcc` chưa chứng minh chương trình chạy trên GPU: phải có kernel CUDA thực hiện các phép tính và bằng chứng đo đạc tương ứng.

### 2.2. Dữ liệu và đầu ra

- Ba cặp ảnh chuẩn ở ba địa điểm: indoor, outdoor và custom; thêm một cặp challenging.
- Tám ảnh gốc, overlap khoảng 30–50%; crop 16:9 và resize thành 24 ảnh ở ba độ phân giải.
- Small: 1280 × 720; medium: 1920 × 1080; large: 3840 × 2160.
- Dùng synthetic pair có ground truth để kiểm tra hình học. Nếu tự tạo, phải ghi rõ nguồn gốc; không gọi đó là synthetic chính thức của môn học.
- Panorama ánh xạ ảnh phải B vào hệ tọa độ ảnh trái A: `H: B → A`.
- Gallery có các cặp được cung cấp nếu có, các cặp nhóm tự chụp và ít nhất một trường hợp thất bại được phân tích.

### 2.3. Những gì đề không bắt buộc

Đề không bắt buộc FAST–BRIEF, SIFT, ORB hoặc một paper duy nhất; nhóm được chọn phương pháp phù hợp và phải giải thích chất lượng cùng hiệu quả song song. Đề không yêu cầu ảnh gốc phải là HEIC, không bắt buộc upload toàn bộ project lên Drive, cũng không bắt buộc mỗi lần Run All tải lại dataset nếu dữ liệu đã sẵn sàng. Yêu cầu chính là notebook chạy từ đầu đến cuối trên phiên Colab GPU mới, không cần thao tác thủ công ngoài upload dataset nếu cần.

## 3. Điểm xuất phát sau Midterm

### 3.1. Những thành phần hiện có

| Thành phần | Hiện trạng để kế thừa |
| --- | --- |
| `src/v0_frontend.cpp` | Grayscale, Gaussian blur, FAST, NMS, Top-K, BRIEF |
| `src/v0_backend.cpp` | Matching, RANSAC/refinement, inverse warp, exposure compensation, seam/feather |
| `src/v0_stitching.cpp` | Kết nối 6 stage, kiểm tra input/H/canvas và đo CPU |
| `src/main.cu` | CLI hiện chỉ cho chạy V0; cần bổ sung lựa chọn V1–V3 |
| `src/common.h`, `src/versions.h` | Kiểu dữ liệu và giao diện cần mở rộng thống nhất |
| `src/cuda_utils.cuh` | Tiện ích CUDA cần kiểm tra và hoàn thiện khi dùng thực tế |
| `tools/build.py`, `src/Makefile` | Hiện xây dựng V0; cần bổ sung nguồn CUDA mới |
| `tools/prepare_dataset.py` | Chuẩn bị ảnh từ raw và tạo synthetic |
| `tools/benchmark.py`, `tools/report.py` | Đo/báo cáo V0; cần hỗ trợ nhiều phiên bản và GPU metrics |
| `Group22_Midterm.ipynb` | Báo cáo Midterm; giữ làm mốc, xây dựng notebook Finalterm riêng |

Đây là trạng thái mã hiện có, không phải kết quả nghiệm thu Finalterm. CUDA V1–V3 và benchmark GPU chưa được chứng minh hoàn thành trong kế hoạch này.

### 3.2. Những hạn chế cần biết

- FAST–BRIEF hiện chưa có đầy đủ khả năng bất biến xoay/tỉ lệ như ORB đa tỉ lệ hoặc SIFT.
- Một global homography không xử lý được mọi thị sai; panorama vẫn có thể bị méo phối cảnh hoặc bóng kép.
- CUDA tăng tốc phép tính, không tự sửa giới hạn hình học của thuật toán.
- Crop chỉ giúp bỏ vùng đen; phải giữ ảnh full canvas để đánh giá xem có mất nội dung hay che lỗi.
- Synthetic chỉ gồm hai crop tịnh tiến từ một ảnh chưa kiểm tra đầy đủ homography phối cảnh tổng quát.
- Cần đối chiếu ảnh thực với nhãn indoor/outdoor/challenging; không coi tên file là bằng chứng đạt tiêu chí cảnh chụp.

### 3.3. Số liệu V0 dùng để định hướng ban đầu

Các giá trị dưới đây lấy từ `outputs_v0/summary.json` đang có khi lập kế hoạch, trên Windows CPU; đơn vị ms, medium 1920 × 1080, làm tròn hai chữ số.

| Cặp ảnh | Matching | Warp/blend | Tổng pipeline |
| --- | ---: | ---: | ---: |
| `site1_indoor` | 51,20 | 160,23 | 280,32 |
| `site2_outdoor` | 102,15 | 122,91 | 303,00 |
| `site3_custom` | 10,73 | 142,19 | 217,16 |
| `site4_challenging` | 1.116,53 | 209,50 | 1.812,66 |

Matching và warp/blend đáng được khảo sát, nhưng bảng CPU này **không thay thế profiling V1/V2 trên GPU**. Không đưa nguyên các số này vào bảng speedup Finalterm; phải đo lại V0–V3 trong phiên Colab dùng để so sánh và sinh bảng từ output notebook.

## 4. Thống nhất pipeline trước khi viết CUDA

### 4.1. Luồng dữ liệu

```text
Ảnh RGB A → grayscale → Gaussian blur → FAST/NMS/Top-K → BRIEF A ─┐
                                                               ├→ matching
Ảnh RGB B → grayscale → Gaussian blur → FAST/NMS/Top-K → BRIEF B ─┘
    → RANSAC + refinement → H: B → A
    → xác định canvas → inverse warp + bilinear
    → exposure compensation + seam/feather → panorama
```

Grayscale và blur là preprocessing bên trong pipeline. Crop/resize ảnh về ba kích thước là bước chuẩn bị dataset trước benchmark. FAST, BRIEF, matching và RANSAC là các thuật toán xử lý chính, không gọi toàn bộ chuỗi là preprocessing.

### 4.2. Đầu vào và đầu ra của từng stage

| Stage | Làm gì? | Đầu vào → đầu ra |
| --- | --- | --- |
| 1. Preprocessing | Đổi ảnh sang mức xám và làm trơn | RGB → grayscale đã blur |
| 2. Detection | Tìm vị trí góc, bỏ điểm cực đại trùng lân cận, giữ điểm mạnh | Grayscale → danh sách keypoint |
| 3. Description | Tạo chữ ký nhị phân cho vùng quanh mỗi điểm | Keypoint + ảnh → descriptor 256 bit |
| 4. Matching | Tìm chữ ký gần nhất và loại cặp mơ hồ | Descriptor A/B → danh sách match |
| 5. Homography | Loại match sai và tìm phép biến đổi | Match + keypoint → H và tập inlier |
| 6. Warp/blend | Biến đổi tọa độ và nối màu hai ảnh | RGB A/B + H → panorama |

### 4.3. Quy tắc khóa baseline

Khuyến nghị giữ phương pháp Midterm cho đợt CUDA đầu tiên. Nếu cần nâng chất lượng, nhóm sửa và kiểm chứng V0 trước, sau đó áp dụng cùng thay đổi cho mọi phiên bản. Lưu mốc Midterm bằng commit/tag trước khi sửa baseline để còn đối chiếu.

Không đổi riêng V3 sang SIFT/APAP rồi dùng speedup V3/V0 để kết luận về tối ưu CUDA. Không giảm độ phân giải, số keypoint, số RANSAC hypothesis hoặc chất lượng blending riêng ở một phiên bản để làm số đo đẹp hơn.

Ghi tham số thực dùng vào một cấu hình chung; không chỉ đọc giá trị mặc định trong header vì nơi gọi hàm có thể truyền giá trị khác. Hiện pipeline gọi tối đa 1.500 keypoint/ảnh, ratio 0,75 và RANSAC tối đa 1.500 hypothesis, ngưỡng 3 pixel. FAST threshold hiện là 20. Đây là tham số của mã hiện có, không phải giá trị bắt buộc của đề.

Chốt thêm sigma/kernel/border blur, seed và pattern BRIEF, NMS/tie-break, cách tính inlier, số lượt refinement, chính sách exposure/seam và giới hạn canvas. Thay đổi cấu hình phải làm phát sinh một lượt kiểm tra và benchmark lại toàn bộ phiên bản liên quan.

## 5. V1 — Basic CUDA: Quân chủ trì

### 5.1. Mục tiêu và phối hợp

V1 thực hiện cùng công việc của V0 trên GPU. Thiết kế cơ bản, dễ giải thích; chưa cần gom nhiều tối ưu. CPU vẫn đọc/ghi ảnh, cấp phát, điều phối và xuất log. Những phép tính của sáu stage, gồm Top-K, refinement, exposure và seam khi được kích hoạt, phải được đưa lên GPU.

- Quân: trực tiếp stages 4–6, kết nối dữ liệu giữa frontend/backend, chịu trách nhiệm nghiệm thu V1.
- Phúc: trực tiếp stages 1–3, bàn giao keypoint/descriptor nằm trên GPU cho Quân.
- Hưng: build/CLI, quản lý buffer dùng chung, CUDA events, công cụ đối chiếu và notebook V1.

### 5.2. Công việc của Phúc trong V1

1. Grayscale: mỗi thread xử lý một pixel, giữ cùng hệ số và cách làm tròn với V0.
2. Gaussian blur: giữ phương pháp tách ngang/dọc như V0, cùng cách xử lý biên và dữ liệu trung gian.
3. FAST: mỗi thread kiểm tra một pixel, ghi response vào buffer.
4. NMS: đọc response đã tính xong; không vừa đọc vừa ghi đè response trong cùng bước.
5. Thu thập keypoint: có thể bắt đầu bằng atomics; bảo đảm đủ dung lượng và không cắt cụt tùy tiện theo thứ tự thread chạy.
6. Top-K: chọn trên GPU, với quy tắc xử lý điểm bằng response được thống nhất với V0.
7. BRIEF: mỗi thread xử lý một keypoint; dùng đúng cùng pattern 256 cặp điểm và cùng quy tắc border.

Không tự sinh pattern BRIEF bằng hai bộ sinh số ngẫu nhiên CPU/GPU rồi cho rằng cùng seed sẽ cho cùng kết quả. Nên dùng một pattern cố định chung cho mọi phiên bản.

### 5.3. Công việc của Quân trong V1

**Matching:** mỗi thread có thể phụ trách một descriptor A và duyệt descriptor B để tìm hai láng giềng gần nhất; thêm bước tìm chiều ngược cho cross-check. Dùng XOR/popcount để tính Hamming; lưu cả khoảng cách và index, quy định tie-break nhất quán. Không cần lưu toàn bộ ma trận khoảng cách nếu chỉ dùng top-2.

**RANSAC:** các hypothesis độc lập được kiểm tra song song. Bản cơ bản có thể dùng một thread/hypothesis để giải bài toán nhỏ và duyệt match; sau đó reduction chọn hypothesis tốt nhất trên GPU. Giữ kiểm tra suy biến, metric sai số đối xứng và refinement như V0. Không thay bằng sai số một chiều chỉ vì dễ viết hơn.

Chuẩn hóa cách lấy mẫu để CPU/GPU kiểm tra cùng danh sách bộ bốn match. Cùng seed ở hai thư viện random khác nhau chưa bảo đảm cùng chuỗi. Nếu chuẩn bị danh sách mẫu ở host, phải công khai phạm vi đó và tính đúng chi phí; các phép ước lượng, chấm điểm, chọn H và refinement vẫn thực hiện trên GPU.

**Warp/blend:** tính phép chiếu/canvas, inverse warp và bilinear; thực hiện thống kê exposure bằng reduction; áp dụng cùng quyết định bật/tắt bù sáng và seam như V0. Mỗi pixel đầu ra do một thread ghi để tránh ghi đè lẫn nhau.

**Seam:** thuật toán hiện tại có phụ thuộc giữa các hàng. Có thể tính các cột của một hàng song song, nhưng hàng sau phải đợi hàng trước. Kernel riêng cho từng hàng là cách khởi đầu dễ hiểu dù có overhead; không dùng `__syncthreads()` để giả định đồng bộ được toàn bộ các block. Giữ bước truy vết seam trên GPU, kể cả khi bản đầu chỉ dùng một thread cho phần này.

### 5.4. Công việc của Hưng trong V1

- Mở rộng CLI để điều phối `--version 0|1|2|3`; phiên bản chưa triển khai phải báo lỗi rõ ràng.
- Sửa build để biên dịch/link nguồn CUDA; V0 vẫn giữ cách chạy đơn luồng để làm chuẩn.
- Thêm buffer GPU có vòng đời rõ ràng, kiểm tra lỗi launch/cấp phát, giới hạn canvas và dọn bộ nhớ trên đường lỗi.
- Mở rộng cấu trúc timing và CSV, ghi GPU/compiler/config/dataset/source hash.
- Viết đối chiếu stage trung gian; tạo notebook chạy V0 và V1 cùng đầu vào.
- Giữ log: `[SUCCESS]:` xanh, `[ERROR]:` đỏ, `[WARN]:` vàng, `[PROCESS]:` trắng; hỗ trợ `NO_COLOR` như hiện có.

### 5.5. Tiêu chí hoàn thành V1

- [ ] Biên dịch bằng CUDA trên môi trường mục tiêu và thực sự launch kernel cho cả sáu stage.
- [ ] Không lén chuyển Top-K, RANSAC/refinement hoặc seam về CPU.
- [ ] Đối chiếu stage và kiểm tra hình học đạt tiêu chí nhóm đã chốt.
- [ ] Chạy đủ 4 cặp × 3 kích thước; thất bại được ghi rõ, không dùng lại ảnh kết quả cũ.
- [ ] Có timing GPU từng stage, transfer, tổng kernel và end-to-end.
- [ ] Có code và giải thích mapping thread/block trong notebook.
- [ ] Có số đo V1 để lựa chọn tối ưu tiếp theo.

## 6. V2 — Tối ưu matching: Phúc chủ trì

### 6.1. Ý tưởng dự kiến và điều kiện chọn

V2 dự kiến bổ sung **shared-memory tiling cho matching**. Ví dụ: nhiều thread cùng cần đọc các descriptor B; block tải một nhóm descriptor B vào bộ nhớ dùng chung, các thread tái sử dụng rồi chuyển sang nhóm tiếp theo.

Chỉ chốt hướng này khi V1 cho thấy matching chiếm thời gian đáng kể và có bằng chứng truy cập bộ nhớ/tái sử dụng dữ liệu là vấn đề. Không suy ra matching bị giới hạn bởi bandwidth chỉ từ việc nó chạy chậm. Nếu matching đã rất nhỏ, chọn một bottleneck thực tế khác và cập nhật kế hoạch trước khi triển khai.

### 6.2. Nhiệm vụ cụ thể

1. Profile V1 trên medium và large, gồm cảnh nhiều keypoint và ít keypoint.
2. Ghi thời gian matching, số descriptor A/B, cấu hình launch và bằng chứng profiler nếu có.
3. Thêm kernel tiled matching, dùng cùng popcount và chính sách top-2/cross-check của V1.
4. Bảo đảm mọi thread trong block đi qua barrier hợp lệ, kể cả tile cuối chưa đủ phần tử.
5. Đo vài kích thước tile/block; ghi cấu hình và lý do chọn, không chỉ báo cấu hình nhanh nhất mà thiếu bối cảnh.
6. Kiểm tra danh sách match và panorama; so sánh V2/V1 và V2/V0.

Không gộp đổi thuật toán, giảm K, thêm streams và kernel fusion vào cùng V2. Giữ một ý tưởng chính để biết thay đổi nào tạo ra tác động. Nếu muốn thử đổi mapping thread sang warp, coi đó là thí nghiệm riêng hoặc giải thích rõ phần nào thuộc thay đổi chính.

### 6.3. Phối hợp và bàn giao

Phúc viết kernel và báo cáo V2. Quân review logic matching, tie-break và tác động tới RANSAC. Hưng tích hợp build/benchmark và lưu output. V2 phải tiếp tục dùng các stage GPU của V1; không dựng lại một pipeline độc lập khó đối chiếu.

- [ ] Có bằng chứng bottleneck của V1 và giả thuyết vì sao tiling có ích.
- [ ] V2 giữ nguyên phương pháp, tham số và kết quả trong tolerance đã chốt.
- [ ] Có thời gian matching và toàn pipeline; không đánh đồng tăng tốc một stage với toàn chương trình.
- [ ] Có phân tích trường hợp tile tăng overhead hoặc occupancy giảm.
- [ ] Có commit, output và phần notebook tái lập được thí nghiệm.

## 7. V3 — Tối ưu concurrency: Hưng chủ trì

### 7.1. Ý tưởng dự kiến và điều kiện chọn

Stages 1–3 của A không cần dữ liệu từ B. V3 dự kiến dùng hai CUDA streams để hai nhánh có cơ hội chạy đồng thời:

```text
Stream A: preprocess A → detect A → describe A → event A ─┐
                                                        ├→ matching → RANSAC → warp/blend
Stream B: preprocess B → detect B → describe B → event B ─┘
                   Matching phải chờ cả event A và B.
```

Streams giống hai hàng công việc độc lập. GPU có chạy đồng thời được hay không còn phụ thuộc tài nguyên và mức sử dụng của từng kernel; hai streams không đồng nghĩa tốc độ gấp đôi.

### 7.2. Nhiệm vụ cụ thể

1. Profile V2 để xem frontend đang nối tiếp bao lâu và có cơ hội overlap hay không.
2. Cho các hàm frontend nhận stream; tách scratch buffer/counter của A và B.
3. Ghi event khi descriptor mỗi ảnh sẵn sàng; stream downstream chờ đúng event.
4. Tránh đồng bộ toàn device sau mỗi kernel nếu không cần; giữ kiểm tra lỗi và đồng bộ tại ranh giới đo/phụ thuộc.
5. Giữ nguyên tiled matching của V2 và cấu hình thuật toán chung.
6. Đo thời gian hai nhánh, thời gian join, tổng kernel và end-to-end; kiểm tra kết quả qua nhiều lần chạy.
7. Dùng timeline hoặc bằng chứng tương đương để xác nhận có overlap thực tế.

Để tách tác động của streams, V1/V2 nên đã giữ buffer trên GPU giữa các stage và tái sử dụng cấp phát. Không đồng thời thêm pinned memory, memory pool và kernel fusion vào V3 rồi gán mọi lợi ích cho streams. Nếu chọn các tối ưu đó, tách thí nghiệm và giải thích phạm vi.

### 7.3. Khi nào đổi hướng V3?

Nếu V2 chủ yếu tốn warp/blend hoặc frontend đã dùng hết tài nguyên GPU, streams có thể không đáng làm. Khi đó Hưng chọn một ý tưởng khác dựa trên profiling, ví dụ tối ưu đọc ảnh cho warping hoặc tối ưu reduction của RANSAC. Giữ thứ tự `V0 → V1 → V2 → V3`, ghi lý do thay đổi trong notebook.

Texture filtering có thể khác manual bilinear về độ chính xác; chỉ chọn khi hiểu và kiểm chứng sai số. Nếu dùng streams nhưng chậm hơn, có thể báo cáo trung thực; không tạo số đo tốt giả hoặc giấu thí nghiệm.

### 7.4. Tiêu chí hoàn thành V3

- [ ] Có bằng chứng bottleneck V2 và mô tả một ý tưởng tối ưu chính.
- [ ] Giữ các tối ưu V2, hoặc giải thích rõ nếu loại bỏ.
- [ ] Có kiểm tra dependency, buffer và tính lặp lại; không race condition.
- [ ] Nếu dùng streams, có bằng chứng overlap hoặc phân tích vì sao không overlap.
- [ ] Timing không cộng nhầm các khoảng chồng lấp thành tổng thời gian thực thi.
- [ ] Có so sánh V3/V2, V3/V0 và phân tích cả trường hợp chậm hơn.

## 8. Giao diện chung và tổ chức mã nguồn đề xuất

Cấu trúc dưới đây là đề xuất, các file mang nhãn mới chưa mặc nhiên tồn tại:

```text
Midterm/                               # Có thể đổi tên thư mục sau, không khóa đường dẫn tuyệt đối
├── assignment.md                      # Kế hoạch này
├── Group22_Midterm.ipynb               # Mốc Midterm
├── Group22_Finalterm.ipynb             # Mới: notebook nộp cuối kỳ
├── README.md
├── team.json
├── requirements.txt
├── requirement/finalterm/
├── src/
│   ├── common.h, versions.h           # Giao diện chung do nhóm thống nhất
│   ├── main.cu                        # Một CLI chọn V0–V3
│   ├── logging.h, cuda_utils.cuh
│   ├── v0_frontend.cpp
│   ├── v0_backend.cpp
│   ├── v0_stitching.cpp
│   ├── gpu_buffers.cuh                # Mới: buffer GPU và context
│   ├── gpu_frontend.cu                # Mới: Phúc, stages 1–3
│   ├── gpu_backend.cu                 # Mới: Quân, stages 4–6
│   ├── gpu_pipeline.cu                # Mới: tích hợp phiên bản/context
│   ├── matching_tiled.cu              # Mới: Phúc, tối ưu V2
│   └── Makefile
├── tools/
│   ├── build.py                       # Mở rộng
│   ├── prepare_dataset.py             # Kế thừa, kiểm tra tương thích dữ liệu
│   ├── benchmark.py                   # Mở rộng V0–V3
│   ├── report.py                      # Sinh bảng/chart từ kết quả thật
│   └── verify_versions.py             # Mới: đối chiếu các phiên bản
├── dataset/
│   ├── raw/                           # Tám ảnh gốc
│   ├── prepared/{small,medium,large}/  # 24 ảnh benchmark và manifest
│   └── synthetic/                     # GT, ảnh kiểm thử và provenance
├── outputs_v0/                        # Giữ mốc cũ; tách với số đo Finalterm
├── outputs_final/                     # Mới: mỗi phiên đo có thư mục riêng
│   └── <session_id>/
│       ├── environment.json
│       ├── timings.csv, summary.json
│       ├── v0/, v1/, v2/, v3/
│       └── profiling/
└── presentation/                      # Mới: slides và thông tin video
```

Giao diện module cần nêu: kích thước buffer, kiểu dữ liệu, host/device ownership, stream, số phần tử hợp lệ, nơi chờ kết quả và cách trả lỗi. Giữ descriptor 32 byte/keypoint hoặc mô tả cách biểu diễn tương đương rõ ràng; không để frontend và backend hiểu khác thứ tự bit/index.

Hưng chủ trì `common.h`, `versions.h`, `main.cu`, schema timing và build; Quân/Phúc gửi đề xuất thay đổi giao diện trước khi dùng. Các phiên bản tái sử dụng stage chung, chỉ thay kernel/chiến lược liên quan thay vì sao chép toàn bộ pipeline.

CLI mục tiêu trên Colab, sau khi đã triển khai:

```bash
mkdir -p outputs_final/manual
./stitch dataset/prepared/medium/site1_indoor_left.png \
         dataset/prepared/medium/site1_indoor_right.png \
         outputs_final/manual/panorama_v1.png \
         --version 1 --runs 5 --warmup 1
```

Lặp với version 0, 2, 3 và đầu ra khác nhau. Đây là giao diện đích; CLI hiện tại vẫn chỉ hỗ trợ V0. Khi sửa CLI, đặt mặc định `--runs` thành 5 để khớp đặc tả Finalterm; mã hiện có đang mặc định 1. Tài liệu chỉ đưa lệnh chạy khi chức năng tương ứng đã có.

## 9. Kiểm tra tính đúng: làm trước khi tin vào speedup

### 9.1. Đối chiếu theo từng stage

| Đối tượng | Cách kiểm tra |
| --- | --- |
| Grayscale/blur | So cùng pixel và báo sai số tối đa/trung bình; giải thích sai số làm tròn |
| Keypoint | So vị trí/response, NMS, Top-K và tie-break; không chỉ so số lượng |
| Descriptor | Với cùng ảnh/keypoint/pattern, so từng bit |
| Matching | So index, khoảng cách, top-2, ratio và mutual cross-check |
| RANSAC | Dùng cùng mẫu kiểm thử, so inlier và sai số chiếu; kiểm tra mẫu suy biến |
| Panorama | So hình học/canvas, pixel trên vùng hợp lệ, seam, exposure và ảnh full canvas |

Chốt tolerance trước khi benchmark, ghi trong notebook và giữ nguyên giữa các phiên bản. Không nới tolerance tùy phiên bản chỉ để kiểm tra vượt qua. Với dữ liệu rời rạc như descriptor/matching, ưu tiên khớp chính xác khi các đầu vào tương ứng giống nhau.

### 9.2. Kiểm tra homography

Chuẩn hóa H nếu cần và đo sai số vị trí điểm sau phép chiếu. Một tiêu chí nội bộ đề xuất là sai số chiếu trung bình trên lưới kiểm tra không vượt 1 pixel với synthetic có GT; đây không phải ngưỡng bắt buộc do đề quy định. Ghi thêm sai số lớn nhất, vùng hợp lệ và điều kiện kiểm tra.

Không chỉ dùng khoảng cách giữa chín phần tử H: hai ma trận khác hệ số có thể biểu diễn cùng phép biến đổi. Tránh so trực tiếp hai panorama khác canvas bằng phép trừ pixel; phải căn đúng hệ tọa độ và vùng hợp lệ.

Ngoài synthetic tịnh tiến hiện có, nên bổ sung fixture có xoay/phối cảnh biết trước, mask vùng hợp lệ và provenance. Nếu môn học cung cấp cặp chính thức, kiểm thử thêm cặp đó. Không tự coi fixture tự tạo là thay thế được mọi kiểm thử chính thức.

### 9.3. Trường hợp lỗi cần thử

- Ảnh không đủ match, ảnh ít texture, pattern lặp lại.
- Input không hợp lệ, H suy biến hoặc canvas quá lớn.
- Nhiều góc vượt dung lượng candidate buffer và tile cuối không đủ phần tử.
- Chạy nhiều lần với streams để phát hiện lỗi không ổn định.
- Large có bộ nhớ đầu ra lớn hơn dự kiến do phép chiếu.

Chạy công cụ kiểm tra lỗi bộ nhớ CUDA nếu môi trường hỗ trợ. Khi chạy thất bại, ghi trạng thái và nguyên nhân, không điền thời gian 0 hoặc tái sử dụng panorama cũ để làm như thành công.

## 10. Benchmark và profiling công bằng

### 10.1. Quy trình đo theo đề

1. Dùng cùng ảnh, tham số, mã nguồn và chính sách cấp phát cho các phiên bản.
2. Trong một phiên Colab GPU, ghi GPU, CPU, CUDA/compiler và session ID; đo lại cả V0.
3. Warm-up một lần trước đo; tối thiểu năm lần chạy có timing cho mỗi cấu hình.
4. V0 dùng `std::chrono::high_resolution_clock`; stage GPU dùng CUDA events.
5. Loại thời gian tải/crop/resize dataset và đọc/ghi file ảnh khỏi timing pipeline.
6. Báo mean; báo thêm minimum khi phương sai lớn. Nên lưu đủ từng sample để kiểm tra độ biến động.
7. So sánh từng phiên bản trên từng cặp ảnh và từng độ phân giải.

Ma trận chính: 4 phiên bản × 4 cặp × 3 kích thước = 48 cấu hình, tối thiểu 240 lần chạy có timing, chưa tính warm-up và synthetic. Cặp thất bại vẫn là một kết quả phải báo cáo, không ép pipeline nhận H sai để đủ số thành công.

### 10.2. Các loại thời gian phải tách rõ

| Đại lượng | Ý nghĩa |
| --- | --- |
| Thời gian stage | Các phép tính thuộc một stage; ghi rõ cách gộp hai ảnh |
| Tổng kernel time | Tổng thời gian kernel, nêu cách thu thập; có thể lấy từ profiler |
| GPU pipeline elapsed | Khoảng thực từ lúc pipeline GPU bắt đầu đến lúc hoàn tất, gồm phụ thuộc/overlap |
| H2D / D2H | Thời gian truyền host → device và device → host |
| End-to-end | Từ input RGB đã ở RAM đến panorama cuối đã ở RAM, gồm truyền và điều phối; loại disk I/O |

CUDA event quanh một stage nhiều kernel đo khoảng elapsed của stage, không mặc nhiên là tổng thời lượng riêng từng kernel. Với streams, tổng các thời lượng có thể lớn hơn thời gian thực trôi qua vì chồng lấp. Cần event trên đúng stream và join đầy đủ trước khi kết thúc phép đo.

Thống nhất việc cấp phát nằm trong hay ngoài vùng đo và công khai. Nếu tái sử dụng buffer, mọi phiên bản GPU áp dụng cùng chính sách, đồng thời ghi rõ đây là phép đo steady-state.

```text
Speedup so V0 = T(V0) / T(Vn)
Speedup so bản trước = T(Vn-1) / T(Vn)
```

So cùng phạm vi thời gian, ví dụ end-to-end với end-to-end. Tách bảng kernel-only nếu cần; không lấy CPU end-to-end chia GPU kernel-only rồi gọi là speedup toàn chương trình. Khi tổng hợp nhiều cảnh, nêu rõ cách lấy trung bình; không trộn mẫu thất bại hoặc số đo khác session vào cùng bảng.

### 10.3. Bằng chứng tối ưu V2/V3

Mỗi phiên bản cần đủ năm phần: **bottleneck → ý tưởng → thay đổi mã → kết quả → phân tích**. Lưu timeline Nsight Systems, metrics Nsight Compute hoặc bằng chứng profiling phù hợp; ảnh chụp và tóm tắt văn bản đều được đề chấp nhận. Nếu Colab hạn chế công cụ, ghi giới hạn thực tế và dùng bằng chứng đo được; không tạo metrics giả hay suy diễn bandwidth/occupancy từ một con số thời gian.

Không hứa trước V2 nhanh gấp 2 hoặc V3 nhanh gấp 10. Dùng profiling để quyết định và báo cả kết quả không đạt kỳ vọng.

## 11. Notebook Finalterm và Colab

### 11.1. Luồng Run All mục tiêu

```text
Kiểm tra môi trường/GPU → xác định ROOT → chuẩn bị dependencies và mã nguồn
→ lấy dataset, kiểm tra đủ file/hash → chuẩn bị 24 ảnh và synthetic
→ biên dịch từ source → kiểm tra correctness
→ benchmark V0–V3 cùng session → sinh bảng/chart/gallery
→ trình bày profiling, hạn chế, kết luận và references
```

Không phụ thuộc `stitch.exe` của Windows trên Colab. Colab phải biên dịch binary Linux `stitch` từ source. `ROOT` phải là thư mục chứa project thực tế; `Path.cwd()` trên Colab không tự cho biết thư mục cha của notebook trong Drive.

Để tái lập dễ hơn, có thể copy mã và dữ liệu cần thiết từ Drive sang thư mục runtime để chạy, sau đó lưu kết quả về Drive. Mount Drive có thể yêu cầu cấp quyền; ghi rõ bước đó. Trước nộp, thử Run All trên phiên GPU mới theo đường cấp dữ liệu đã chọn, tránh phụ thuộc file còn sót từ lần chạy trước.

### 11.2. Dataset và file sinh lại

- `dataset/raw`: giữ ảnh gốc, tránh xóa khi đường tải chưa được xác nhận hoạt động.
- `dataset/prepared`: có thể sinh lại nếu đủ raw và dependencies đọc ảnh.
- `dataset/synthetic/synthetic_GT.HEIC`: là ảnh nguồn nếu nhóm dùng fixture hiện tại; cần giữ hoặc có cách tải lại.
- Synthetic left/right, H và `provenance.json`: sinh đồng bộ từ cùng nguồn; provenance mô tả nguồn thật.
- Output benchmark: có thể chạy lại nhưng số đo sẽ phụ thuộc phần cứng/session; lưu nguyên kết quả đã dùng trong báo cáo.
- Binary: build lại từ source; giữ source/build scripts và dependencies tương ứng.

Nếu dùng Google Drive public download, phải thử quyền tải trong môi trường không đăng nhập. Nếu dùng mount Drive, đường dẫn cấu hình phải được ghi rõ. Không chỉ đổi đuôi JPG thành HEIC; định dạng nội dung cần đọc được. Ảnh raw không bắt buộc mang đuôi HEIC.

### 11.3. Cấu trúc báo cáo theo đề và người viết

| Mục notebook | Nội dung | Người chủ trì |
| --- | --- | --- |
| 1. Team information | Tên, MSSV, bảng đóng góp theo stage và version | Hưng tổng hợp; mỗi người xác nhận phần mình |
| 2. Introduction | Bài toán và sơ đồ pipeline | Phúc + Quân |
| 3. Chosen methods | Phương pháp, tham số, chất lượng và khả năng song song | Phúc stages 1–3; Quân stages 4–6 |
| 4. V0 Sequential | Thiết kế, ảnh kết quả, timing và bottleneck | Hưng tổng hợp từ mã/kết quả thật |
| 5. V1 Basic CUDA | Mapping, code, correctness, timing, speedup | Quân chủ trì; Phúc/Hưng bổ sung phần trực tiếp làm |
| 6a. V2 | Đủ năm phần tối ưu | Phúc |
| 6b. V3 | Đủ năm phần tối ưu | Hưng |
| 7. Final comparison | Bảng/chart tất cả version/resolution, breakdown bản tốt nhất | Hưng |
| 8. Results gallery | Synthetic/cặp được cung cấp và ảnh nhóm, failure case | Cả nhóm |
| 9. Conclusion | Bài học, giới hạn, hướng tiếp | Cả nhóm |
| 10. References | Trích dẫn đúng nơi dùng và khai báo AI | Mỗi người bổ sung nguồn; Hưng kiểm tra |

Mỗi con số trong phần diễn giải phải có output cell làm căn cứ. Lưu output vào notebook nộp và xuất PDF. Không khẳng định đạt hiệu năng/chất lượng chỉ dựa trên kế hoạch này.

## 12. Mốc thực hiện và phối hợp

Các mốc dưới đây là thứ tự nghiệm thu, chưa gắn ngày cụ thể vì chưa có deadline Finalterm được xác nhận.

| Mốc | Quân | Phúc | Hưng | Điều kiện chuyển mốc |
| --- | --- | --- | --- | --- |
| M0 — Chốt baseline | Rà backend/H/seam | Rà frontend/BRIEF/Top-K | Dataset, cấu hình, schema, môi trường | V0 tái lập được, giao diện đã thống nhất |
| M1 — V1 từng module | Matching, RANSAC, warp/blend GPU | Preprocess, FAST/NMS/Top-K, BRIEF GPU | Build/CLI/events/đối chiếu | Module vượt kiểm tra stage |
| M2 — V1 tích hợp | Chủ trì ghép pipeline và sửa lỗi | Sửa frontend, hỗ trợ tích hợp | Chạy ma trận V0/V1 và lưu evidence | Đủ sáu stage GPU, có correctness và timing |
| M3 — V2 | Review backend | Chủ trì profiling và tối ưu V2 | Benchmark/báo cáo so sánh | Có evidence, kết quả và phân tích V2 |
| M4 — V3 | Review dependency/backend | Hỗ trợ frontend/streams | Chủ trì profiling và tối ưu V3 | Có evidence, kết quả và phân tích V3 |
| M5 — Bài nộp | Viết V1, demo backend | Viết V2, demo frontend/matching | Viết V3, tổng hợp Colab/PDF/package | Fresh Run All đạt, đủ hồ sơ |

Trong lúc V1 đang phát triển, Phúc có thể chuẩn bị mô hình tiling và Hưng có thể chuẩn bị timeline/benchmark; chưa chốt kết luận tối ưu khi chưa có V1 chạy được. Ưu tiên hoàn thành V1 đầy đủ trước khi thêm nhiều tính năng nâng chất lượng.

### 12.1. Quy ước Git

- Nhánh gợi ý: `feat/v1-basic-cuda`, `feat/v1-frontend`, `feat/v2-tiled-matching`, `feat/v3-streams`, `chore/final-benchmark`.
- Nhánh V2 lấy từ V1 đã tích hợp; nhánh V3 cập nhật từ V2 đã tích hợp. Merge thay đổi theo phụ thuộc và kiểm tra lại sau merge.
- Mỗi PR/commit lớn nêu thay đổi, phạm vi phiên bản, cách kiểm tra, kết quả và giới hạn còn lại.
- Không đổi schema/giao diện chung âm thầm; không commit dữ liệu nặng/binary/output tạm nếu repository không có chủ trương lưu chúng.
- Người viết module chịu trách nhiệm sửa lỗi module và viết báo cáo phần mình; người tích hợp không mặc nhiên làm thay toàn bộ.

### 12.2. Rủi ro và hướng xử lý

| Rủi ro | Cách xử lý |
| --- | --- |
| Quân quá tải vì V1 | Phúc làm stages 1–3; Hưng làm hệ thống/đo/kiểm tra ngay từ M1 |
| Chưa có GPU laptop | Dùng Colab GPU cho kiểm tra CUDA, lưu số đo và checkpoint; CPU local làm V0/công cụ |
| GPU thay đổi giữa các lần chạy | Đo lại cả bảng trong một session; ghi GPU cho mỗi kết quả |
| Top-K/random không nhất quán | Chốt tie-break, seed/pattern và dữ liệu kiểm thử chung trước tích hợp |
| Seam khó song song | Bản đầu giữ phụ thuộc theo hàng, chấp nhận overhead; tối ưu sau profiling |
| Large hết bộ nhớ | Ước lượng cả canvas/scratch, tránh ma trận matching không cần thiết; báo lỗi rõ ràng |
| Panorama vẫn méo | Phân tích parallax/global H và ảnh chụp; nếu đổi phương pháp phải cập nhật mọi version |
| Streams không tăng tốc | Kiểm tra overlap/tài nguyên và báo kết quả thật; đổi hướng V3 nếu evidence phù hợp |
| Drive download lỗi | Xác nhận quyền hoặc dùng đường upload/mount được mô tả; kiểm tra đủ file trước prepare |

## 13. Hồ sơ cuối kỳ và điều kiện hoàn tất

Theo đề, nhóm cần notebook và PDF, toàn bộ mã nguồn cùng README, slides và video 20–30 phút. Video upload YouTube ở chế độ Unlisted, đặt link trong notebook/README; không upload video lên Moodle.

Phân bổ video đề xuất: tổng quan 4 phút, live demo 7 phút, kết quả 7 phút, tối ưu/bài học 5 phút; tổng khoảng 23 phút. Mỗi người trình bày và giải thích phần mình, nhưng cả ba phải hiểu toàn pipeline.

### Checklist nhóm trước nộp

- [ ] Một CLI chạy được V0–V3 bằng `--version`.
- [ ] V0 đơn luồng; V1 đủ sáu stage GPU; V2/V3 kế thừa bản trước với một ý tưởng tối ưu chính.
- [ ] Cùng dataset/phương pháp/tham số và quy tắc timing giữa các bản.
- [ ] Đủ 24 ảnh chuẩn bị và synthetic có nguồn gốc/GT rõ ràng; nhãn cảnh đã được kiểm tra.
- [ ] Có correctness theo stage và hình học, có phân tích trường hợp thất bại.
- [ ] Có đủ timing, speedup và evidence profiling cho các phiên bản tối ưu.
- [ ] Mọi số liệu/ảnh trong báo cáo sinh từ mã nguồn nộp; lưu đủ sample/config/session.
- [ ] Notebook Run All được trên phiên Colab GPU mới; đã lưu output và xuất PDF.
- [ ] README có dependencies, CUDA version, build/run, yêu cầu phần cứng và output mong đợi.
- [ ] Có slides và link YouTube Unlisted 20–30 phút trong notebook/README.
- [ ] Trích dẫn ý tưởng/mã ngoài nhóm tại nơi dùng và References; khai báo AI đúng phạm vi sử dụng.
- [ ] Bảng contribution phản ánh việc thực sự đã làm; cả ba giải thích và sửa được mã khi bảo vệ.

Việc cập nhật `assignment.md` chỉ hoàn thiện kế hoạch. Chỉ đánh dấu các mục hoàn thành sau khi có mã, output kiểm tra và bằng chứng thực nghiệm tương ứng.
