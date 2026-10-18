# Fuzzy-Adaptive ByteTrack (MOT17 Benchmark)

Hệ thống nghiên cứu và đánh giá so sánh hiệu năng theo dõi đa đối tượng (Multi-Object Tracking - MOT) trên tập dữ liệu chuẩn **MOT17** giữa ba cấu hình:
1. **ByteTrack Mặc định (Baseline)**.
2. **ByteTrack Tối ưu hóa bằng Thuật toán Di truyền (GA)** chỉ trên Development split.
3. **Fuzzy-Adaptive ByteTrack** với ngưỡng `track_high_thresh` được điều chỉnh thích nghi động theo từng frame thông qua bộ điều khiển mờ Mamdani.

---

## 1. Cấu trúc Dự án

```text
Fuzzy-Adaptive ByteTrack/
├── configs/
│   ├── dataset.yaml                # Cấu hình nguồn MOT17, URL, checksum, điều khoản
│   ├── splits.yaml                 # Phân chia Development và Held-out chống rò rỉ dữ liệu
│   ├── experiment.yaml             # Cấu hình tổng thể thí nghiệm (detector, tracker, cache, GA)
│   ├── default_bytetrack.yaml      # Cấu hình ByteTrack baseline
│   ├── fuzzy_bytetrack.yaml        # Cấu hình bộ điều khiển Fuzzy
│   └── ga_bytetrack.yaml           # Kết quả tham số tối ưu tìm bởi GA
├── data/
│   ├── raw/                        # Chứa MOT17.zip (5.46 GB) đã tải
│   ├── MOT17/                      # Dữ liệu giải nén (train/ và test/, 42 sequences)
│   └── cache/detections/           # Detection cache (.npz và _meta.json)
├── scripts/
│   ├── check_environment.py        # Kiểm tra môi trường Conda base, GPU, PyTorch, sao lưu
│   ├── download_mot17.py           # Tải MOT17 tự động, hiển thị tiến độ, chống tràn đĩa
│   ├── validate_mot17.py           # Kiểm tra toàn diện seqinfo, frame liên tục, GT 9 cột
│   └── create_splits.py            # Tạo và xác minh splits chống rò rỉ dữ liệu
├── src/
│   ├── main.py                     # Master CLI (cache_detections, baseline, fuzzy, ga, evaluate)
│   ├── bytetrack_adapter.py        # Subclass AdaptiveByteTracker kế thừa Ultralytics BYTETracker
│   ├── detection_cache.py          # Quản lý trích xuất và đọc detection cache siêu tốc
│   ├── fuzzy_controller.py         # Bộ điều khiển mờ Mamdani (scikit-fuzzy)
│   ├── genetic_optimizer.py        # Tối ưu hóa di truyền (GA) trên validation
│   ├── tracking_logger.py          # Log processing time, predicted lifespan, GT metrics
│   ├── metrics.py                  # Xuất chuẩn MOTChallenge và tích hợp TrackEval
│   └── visualization.py            # Bảng so sánh và biểu đồ trực quan hóa
├── tests/
│   └── test_fuzzy_bytetrack.py     # Unit test cho giới hạn ngưỡng, empty frame, reset state
├── outputs/
│   ├── environment_report.json     # Chi tiết thông số hệ điều hành, Python, PyTorch, CUDA
│   ├── environment_before.yml      # Sao lưu conda env export
│   ├── pip_before.txt              # Sao lưu pip freeze
│   ├── requirements_resolved.txt   # Phiên bản thực tế của tất cả dependencies
│   ├── download_log.json           # Log kiểm toán tải và giải nén MOT17
│   ├── dataset_manifest.csv        # Danh mục 42 sequences MOT17
│   ├── dataset_validation.json     # Kết quả chi tiết kiểm tra tính hợp lệ dữ liệu
│   ├── splits_summary.json         # Tóm tắt phân chia tập dữ liệu
│   └── experiments/                # Kết quả chạy thí nghiệm (baseline, fuzzy, ga)
├── requirements.txt
└── README.md
```

---

## 2. Yêu cầu Môi trường: Conda `base`

Dự án được cấu hình để chạy **trực tiếp trong môi trường Conda `base`**, không tạo môi trường mới hoặc virtualenv.

### 2.1. Kiểm tra môi trường và sao lưu
Trước khi chạy, thực thi script kiểm tra hệ thống:
```bash
conda activate base
python scripts/check_environment.py
```
Script sẽ tự động:
- Xác minh môi trường active là `base`.
- Kiểm tra OS, Python (3.10+), PyTorch, CUDA, thông số GPU (VRAM) và Ultralytics.
- Tự động sao lưu trạng thái môi trường ra `outputs/environment_before.yml` và `outputs/pip_before.txt`.
- Ghi kết quả kiểm toán vào `outputs/environment_report.json`.

### 2.2. Cài đặt Dependencies
Dự án sử dụng các package đã được xác minh tính tương thích:
```bash
python -m pip install -r requirements.txt
```
Danh sách package chính:
- `torch>=2.0.0`, `torchvision>=0.15.0`
- `ultralytics>=8.2.0`
- `opencv-python>=4.8.0`, `numpy>=1.24.0`, `pandas>=2.0.0`, `scipy>=1.10.0`
- `scikit-fuzzy>=0.4.2` (điều khiển mờ Mamdani)
- `lap>=0.5.12` (linear sum assignment cho ByteTrack)
- `PyYAML>=6.0`, `matplotlib>=3.7.0`, `tqdm>=4.65.0`

---

## 3. Dataset MOT17: Tải, Kiểm tra & Phân chia Split

### 3.1. Tải dữ liệu tự động (`scripts/download_mot17.py`)
```bash
python scripts/download_mot17.py --config configs/dataset.yaml
```
- Tải từ nguồn MOTChallenge chính thức (`https://motchallenge.net/data/MOT17.zip` - 5.46 GB).
- Tích hợp bảo vệ dung lượng đĩa: kiểm tra dung lượng trống trước khi tải và giải nén.
- Hỗ trợ resume tải gián đoạn qua HTTP Range header (`.part`).
- Giải nén an toàn 33,960 files (bảo vệ chống lỗ hổng Zip Slip).

### 3.2. Kiểm tra tính toàn vẹn dữ liệu (`scripts/validate_mot17.py`)
```bash
python scripts/validate_mot17.py --config configs/dataset.yaml
```
Kiểm tra toàn bộ 42 sequences:
- Xác thực `seqinfo.ini` (FPS, độ phân giải, số frame).
- Kiểm tra tính liên tục của các frame ảnh trong `img1/`.
- Phân tích `gt/gt.txt` theo protocol MOTChallenge (9 cột: `frame, id, bb_left, bb_top, bb_width, bb_height, conf, class, visibility`).
- Phân loại rõ ràng: Pedestrian (Class 1) là đối tượng đánh giá; các Class 2, 7, 8, 12 là Ignore/Distractor.
- Xuất kết quả ra `outputs/dataset_manifest.csv` và `outputs/dataset_validation.json`.

### 3.3. Phân chia Split chống rò rỉ dữ liệu (`scripts/create_splits.py`)
```bash
python scripts/create_splits.py --config configs/splits.yaml
```
Tuân thủ nguyên tắc **No Data Leakage**:
- **Tách theo Video Scene gốc**: Tuyệt đối không chia ngẫu nhiên frame.
- **Tập Development / Tuning**: `MOT17-02`, `MOT17-05`, `MOT17-09`, `MOT17-10` (4 scenes: góc thấp, đám đông đi bộ, camera cầm tay, đường phố đêm). Dành riêng cho GA và tinh chỉnh luật Fuzzy.
- **Tập Local Held-out Evaluation**: `MOT17-04`, `MOT17-11`, `MOT17-13` (3 scenes: góc cao ban đêm, trung tâm thương mại, ngã tư xe buýt). Được **khóa tuyệt đối** trong quá trình tối ưu và chỉ dùng đánh giá cuối cùng.
- Mọi biến thể detector (DPM, FRCNN, SDP) của cùng một video luôn nằm chung một split duy nhất.

---

## 4. Tối ưu Chi phí Tính toán: Detection Cache

Để tránh lãng phí GPU RTX 3050 khi chạy nhiều vòng lặp GA và Fuzzy:
- Chạy YOLO detector **đúng 1 lần** cho mỗi sequence với `conf=0.01` và `iou=0.7`.
- Ngưỡng 0.01 đảm bảo giữ lại các detection có độ tin cậy thấp phục vụ lượt association thứ 2 của ByteTrack.
- Lưu cache dạng ma trận nén `.npz` và thông số cấu hình hash SHA256 vào `_meta.json`.
- Khi chạy tracking trên cache (`cached_tracking`), tốc độ đạt **~1300 - 2000 FPS** (chỉ tốn ~0.5 - 0.8 ms/frame).

Sinh detection cache cho các sequence:
```bash
python -m src.main --mode cache_detections --config configs/experiment.yaml
```
*(Thêm `--smoke-test` để chạy thử 30 frame kiểm tra)*.

---

## 5. Định nghĩa Chuẩn 3 Đại lượng Đo lường

Theo yêu cầu chuẩn hóa thực nghiệm, hệ thống phân biệt độc lập 3 đại lượng:

### A. Processing Time (Thời gian xử lý)
- Đo bằng đơn vị milliseconds (ms).
- Tách riêng: thời gian detector, tracker, fuzzy controller, và tổng pipeline.
- Báo cáo chi tiết: **mean**, **median**, **p95** và **FPS**.
- Tách 5 frame warm-up đầu tiên khỏi số liệu chính.
- Tách bạch rõ ràng `tracker_only_fps` (trên cache) và `end_to_end_fps`.

### B. Predicted Track Lifespan (Vòng đời ID dự đoán)
Ghi nhận chi tiết cho từng track ID xuất hiện:
- `sequence`, `track_id`, `first_frame`, `last_frame`
- `span_frames = last_frame - first_frame + 1`
- `span_seconds = span_frames / sequence_fps`
- `observed_frames`, `missing_frames_within_span`
- `longest_consecutive_observed_frames`, `longest_consecutive_observed_seconds`
- `mean_confidence`
> [!NOTE]
> `span_seconds` không phải là thời gian tracking đúng. Track tồn tại lâu chưa chắc bám đúng một đối tượng thực tế.

### C. Correct Identity Continuity (Độ liên tục danh tính)
- Phân tích sự liên tục khi khớp track ID dự đoán với ground truth ID.
- Đo lường ID switches (IDSW), track fragmentations, MT (Mostly Tracked), ML (Mostly Lost) theo chuẩn CLEAR MOT.

---

## 6. Tích hợp Fuzzy & Adaptive ByteTracker

### 6.1. Thiết kế Adapter thực sự (`src/bytetrack_adapter.py`)
- Lớp `AdaptiveByteTracker` kế thừa từ `BYTETracker` của Ultralytics.
- Thay vì sửa file YAML trên đĩa, adapter truyền trực tiếp `dynamic_high_thresh` vào `self.args.track_high_thresh` ngay trước khi gọi `super().update()`.
- Không khởi tạo lại tracker mỗi frame -> giữ nguyên vẹn toàn bộ trạng thái Kalman Filter.
- Tránh vòng lặp logic (Circular Dependency): `unmatched_ratio` được tính toán sau association của frame trước và truyền vào Fuzzy controller ở frame tiếp theo.
- Reset sạch trạng thái giữa hai sequence qua phương thức `reset_state()`.

### 6.2. Bộ điều khiển mờ Mamdani (`src/fuzzy_controller.py`)
- **Đầu vào (Antecedents)**:
  - `low_conf_ratio` [0, 1]: Tỷ lệ detection có confidence thấp (< track_high_thresh).
  - `unmatched_ratio` [0, 1]: Tỷ lệ track bị mất liên kết ở frame trước.
  - `confidence_spread` [0, 1]: Độ lệch chuẩn của confidence detections.
- **Đầu ra (Consequent)**:
  - `high_thresh_delta` [-0.10, 0.10]: Độ dịch chuyển của ngưỡng association chính.
- **Clipping an toàn**: Ngưỡng được kiểm soát chặt chẽ trong khoảng `[track_low_thresh, 1.0]`.

### 6.3. Unit Tests (`tests/test_fuzzy_bytetrack.py`)
Chạy kiểm thử toàn diện 5 kịch bản:
```bash
python -m unittest tests/test_fuzzy_bytetrack.py
```
- `test_parameter_bounds`: Kiểm tra clipping ngưỡng động.
- `test_empty_detection_frames`: Kiểm tra chuỗi frame liên tiếp không có detection.
- `test_sequence_reset`: Kiểm tra xóa sạch bộ nhớ Kalman và reset ID khi đổi sequence.
- `test_no_circular_dependency`: Kiểm tra luồng tuần tự của unmatched ratio.
- `test_fuzzy_boundary_values`: Kiểm tra tính toán ổn định tại các giá trị biên (0.0, 1.0).

---

## 7. Hướng dẫn Sử dụng CLI

Toàn bộ pipeline được vận hành thông qua module `src.main`:

```bash
# 1. Kích hoạt môi trường
conda activate base
cd "/mnt/coding/python FPTU/DPL303m/Fuzzy-Adaptive ByteTrack"

# 2. Kiểm tra môi trường & Sao lưu
python scripts/check_environment.py

# 3. Tải và kiểm tra tính toàn vẹn MOT17
python scripts/download_mot17.py --config configs/dataset.yaml
python scripts/validate_mot17.py --config configs/dataset.yaml
python scripts/create_splits.py --config configs/splits.yaml

# 4. Chạy Unit Tests
python -m unittest tests/test_fuzzy_bytetrack.py

# 5. Sinh Detection Cache (Chạy 1 lần duy nhất cho các sequence)
python -m src.main --mode cache_detections --config configs/experiment.yaml

# 6. Chạy Baseline ByteTrack trên tập Development
python -m src.main --mode baseline --config configs/experiment.yaml --split development

# 7. Chạy Fuzzy-Adaptive ByteTrack trên tập Development
python -m src.main --mode fuzzy --config configs/experiment.yaml --split development

# 8. Chạy Genetic Algorithm tối ưu hóa (Chỉ trên Development split)
python -m src.main --mode ga --config configs/experiment.yaml

# 9. Đánh giá so với Ground Truth
python -m src.main --mode evaluate --config configs/experiment.yaml

# 10. Tạo bảng và biểu đồ so sánh tổng hợp
python -m src.main --mode compare --config configs/experiment.yaml
```

### Các cờ tùy chọn hữu ích:
- `--dry-run`: Kiểm tra cấu hình và tài nguyên mà không thực thi tính toán.
- `--smoke-test`: Chạy thử trên 30 frames đầu của sequence để kiểm tra toàn bộ luồng dữ liệu trước khi chạy full sequence.
- `--split`: Chọn tập chạy (`development`, `held_out`, hoặc `all`).

---

## 8. Kết quả Thực nghiệm & Kiểm chứng

Mỗi lần chạy thí nghiệm sẽ tạo thư mục riêng trong `outputs/experiments/<mode>/<sequence_name>/` chứa:
- `tracking_log.csv`: Toàn bộ bounding box, track ID, confidence, và ngưỡng thực tế áp dụng trên từng frame.
- `prediction_mot.txt`: File dự đoán định dạng chuẩn 9 cột MOTChallenge.
- `predicted_track_lifespan.csv`: Thống kê chi tiết vòng đời 11 trường thông tin cho từng ID.
- `fuzzy_adjustments.csv`: Log chi tiết giá trị đầu vào và độ dịch ngưỡng mờ (chỉ có ở mode fuzzy).
- `summary.json`: Báo cáo thời gian xử lý (mean, median, p95, FPS) tách biệt tracker, detector và pipeline.
- `config_snapshot.yaml`: Bản sao cấu hình đã dùng để bảo đảm khả năng tái lập kết quả.
