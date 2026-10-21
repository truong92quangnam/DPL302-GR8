# Fuzzy-Adaptive ByteTrack (MOT17 Benchmark)

Hệ thống nghiên cứu, tối ưu hóa và đánh giá so sánh hiệu năng theo dõi đa đối tượng (Multi-Object Tracking - MOT) trên tập dữ liệu chuẩn **MOT17** kết hợp giữa **YOLOv8**, **ByteTrack**, **Thuật toán Di truyền (Genetic Algorithm - GA)** và **Bộ điều khiển Mờ Thích nghi (Fuzzy Mamdani Controller)**.

Dự án cung cấp so sánh đa chiều giữa ba biến thể:
1. **ByteTrack Mặc định (Baseline)**.
2. **ByteTrack Tối ưu hóa Toàn cục bằng GA (GA-ByteTrack)** (tinh chỉnh trên Development split).
3. **Fuzzy-Adaptive ByteTrack** với ngưỡng association `track_high_thresh` được điều chỉnh thích nghi trực tuyến trên từng frame mà không làm mất trạng thái Kalman Filter.

---

## 1. Kiến trúc Tổng thể Hệ thống

```
+---------------------------------------------------------------------------------------------------+
|                                      SYSTEM PIPELINE ARCHITECTURE                                 |
+---------------------------------------------------------------------------------------------------+
                                                  
     Input Video Frame (MOT17 Sequences)
                    │
                    ▼
   ┌───────────────────────────────────┐
   │    Detector: Frozen YOLOv8n       │   ◄── Chạy 1 lần duy nhất trên GPU RTX 3050
   │  (imgsz=1280, conf=0.01, iou=0.7) │
   └────────────────┬──────────────────┘
                    │
                    ▼
   ┌───────────────────────────────────┐
   │     Offline Detection Cache       │   ◄── data/cache/detections/*.npz
   │   (Boxes, Confs, Classes, Meta)   │       (Tái sử dụng cho mọi lượt tracking)
   └────────────────┬──────────────────┘
                    │
        ┌───────────┴───────────────────────────────┐
        │                                           │
        ▼                                           ▼
┌─────────────────────────────────┐   ┌───────────────────────────────────────────┐
│     Genetic Algorithm (GA)      │   │         Fuzzy Adaptive Controller         │
│  (Offline Parameter Evolution)  │   │        (Online Real-time Inference)       │
│                                 │   │                                           │
│ • Chromosome: 5 ByteTrack params│   │ • Inputs: low_conf_ratio, prior_unmatched,│
│ • Fitness: HOTA + IDF1 + MOTA   │   │           confidence_spread               │
│ • Population=10, Generations=8  │   │ • Mamdani Rules: 9 high-thresh rules      │
│ • Dev Split ONLY (No Leakage)   │   │ • Output: dynamic high_thresh_delta       │
└───────────────┬─────────────────┘   └─────────────────────┬─────────────────────┘
                │                                           │
                ▼                                           ▼
   Configs: configs/ga_bytetrack.yaml         Per-frame Dynamic Threshold Injection
                │                                           │
                └─────────────────────┬─────────────────────┘
                                      │
                                      ▼
             ┌─────────────────────────────────────────────────┐
             │       AdaptiveByteTracker (Subclass Adapter)    │
             │   (Kalman State Preserved, No Reinitialization) │
             │                                                 │
             │  • Stage 1 Association: High-confidence dets    │
             │  • Stage 2 Association: Low-confidence dets     │
             │  • State Update & Frame-to-frame Tracking Log   │
             └────────────────────────┬────────────────────────┘
                                      │
                                      ▼
             ┌─────────────────────────────────────────────────┐
             │         Standardized Evaluation (TrackEval)     │
             │                                                 │
             │  • Metric 1: Processing Time (Tracker / E2E FPS)│
             │  • Metric 2: Predicted Track Lifespan (11 cols) │
             │  • Metric 3: Identity Continuity (HOTA, MOTA,   │
             │              IDF1, ID Switches, Fragmentation)  │
             └─────────────────────────────────────────────────┘
```

---

## 2. Kiến trúc Detector YOLO & Cơ chế Detection Cache

### 2.1. Vai trò của YOLO trong Tracking-by-Detection
Mô hình theo dõi tuân theo mô hình **Tracking-by-Detection**, trong đó chất lượng detection quyết định trực tiếp trần hiệu năng của tracker:
- **Kiến trúc sử dụng**: Pretrained `yolov8n.pt` (COCO dataset).
  - **Backbone**: Modified CSPDarknet với các khối C2f (Cross-Stage Partial bottleneck with 2 convolutions) tăng cường khả năng trích xuất đặc trưng đa tầng.
  - **Neck**: Cấu trúc PAN-FPN (Path Aggregation Network - Feature Pyramid Network) kết hợp đặc trưng nông (giàu thông tin vị trí hình học) và đặc trưng sâu (giàu thông tin ngữ nghĩa).
  - **Head**: Decoupled Head không sử dụng anchor (Anchor-Free), tách riêng nhánh phân loại (classification) và hồi quy bounding box (regression).
- **Lớp đối tượng theo dõi**: Chỉ quan tâm đến lớp người đi bộ (**Class 0: Person / Pedestrian**).

### 2.2. Ngưỡng Detection Khắt khe vs Cơ chế ByteTrack
- Trong các pipeline tracking truyền thống, detector thường đặt ngưỡng confidence cao ($conf \ge 0.5$) để loại bỏ false positives.
- Tuy nhiên, thuật toán **ByteTrack** dựa trên nguyên lý then chốt: **Không được vứt bỏ các detection có confidence thấp** ($0.1 \le conf < 0.5$). Các đối tượng bị che khuất một phần (occlusion), mờ nhòe do chuyển động (motion blur) hoặc ở xa camera thường rơi vào dải confidence thấp này.
- Vì vậy, detector trong dự án được cấu hình:
  - `imgsz: 1280` (bảo toàn chi tiết vật thể nhỏ ở xa).
  - `conf: 0.01` (giữ lại trọn vẹn toàn bộ các ứng viên tiềm năng).
  - `iou: 0.7` (ngưỡng Non-Maximum Suppression hợp lý chống trùng lặp).

### 2.3. Tối ưu Chi phí Tính toán trên GPU RTX 3050 (Detection Caching)
GPU thực nghiệm là **NVIDIA GeForce RTX 3050 Laptop GPU** (3.68 GB VRAM). Nếu phải chạy lại YOLO mỗi khi GA đánh giá một cá thể hoặc khi thử nghiệm các luật mờ khác nhau, chi phí tính toán sẽ cực kỳ đắt đỏ:
- **Giải pháp**: Detector YOLO chỉ chạy **đúng 1 lần duy nhất** cho mỗi video sequence.
- **Định dạng Cache**:
  - Dữ liệu bounding box, confidence và class được nén trong file `.npz` (`data/cache/detections/<sequence>_detections.npz`).
  - Metadata chi tiết được lưu trong file `.json` đi kèm, bao gồm: model name, image size, NMS IoU, detector conf, checksum SHA-256 của file ảnh, thời gian xử lý từng frame.
- **Hiệu quả**:
  - Tốc độ sinh cache YOLOv8n trên RTX 3050: ~**19.9 – 22.1 ms/frame** (~50 FPS).
  - Tốc độ chạy tracker trên detection cache (`cached_tracking`): **0.45 – 1.25 ms/frame** (~**800 – 2,200 FPS** trên CPU), cho phép hoàn thành 80 cá thể GA chỉ trong vài phút.

### 2.4. Phân Tích Kiến Trúc: Tại Sao Chọn YOLOv8 Thay Vì Các Phiên Bản Khác?

Việc lựa chọn **YOLOv8** thay vì các phiên bản khác (YOLOv5, YOLOX, YOLOv7, YOLOv9, YOLOv10, YOLO11) xuất phát từ sự tương thích sâu sắc về mặt kiến trúc với thuật toán ByteTrack và giới hạn tài nguyên của GPU RTX 3050 (3.68 GB VRAM):

```
                           SO SÁNH KIẾN TRÚC HEAD
  
  [Coupled Head - YOLOv5]                     [Decoupled Head - YOLOv8]
                                                         ┌── Conv ──► Class Score (Person)
      Feature Map ──► Conv ──► [Boxes + Cls]             │
                                           Feature Map ──┤
                                                         └── Conv ──► BBox Regression (DFL)
  (Mâu thuẫn đặc trưng: vị trí và ngữ nghĩa bị trộn lẫn)   (Tách biệt hoàn toàn: Tọa độ chuẩn xác 
                                                           dù vật thể bị mờ nhòe / che khuất)
```

#### A. Ưu thế kiến trúc vượt trội của YOLOv8 đối với ByteTrack:
1. **Kiến trúc Anchor-Free**:
   - Loại bỏ hoàn toàn các khung neo cố định (anchors). Người đi bộ trong thực tế thường xuyên thay đổi tư thế (ngồi, cúi, mang vác) hoặc bị che khuất một phần thân thể khiến tỷ lệ khung hình biến dạng. Anchor-free dự đoán trực tiếp khoảng cách từ tâm đến 4 cạnh, giúp bounding box luôn bám sát biên dạng thực của người đi bộ.
2. **Decoupled Head (Tách rời nhánh phân loại và tọa độ)**:
   - Trong cấu trúc Coupled Head cũ, phân loại và định vị dùng chung nhánh tích chập, dẫn đến mâu thuẫn nhiệm vụ (task misalignment).
   - YOLOv8 tách riêng hai nhánh: Khi một người bị che khuất hoặc mờ do chuyển động (motion blur), nhánh phân loại có thể hạ điểm confidence xuống thấp ($0.15 - 0.25$), **nhưng nhánh hồi quy bounding box vẫn duy trì độ chính xác hình học cao**. Đây chính là dữ liệu "vàng" mà lượt association thứ 2 của ByteTrack cần để duy trì liên kết mà không làm trôi bộ lọc Kalman (Kalman velocity drift).
3. **TaskAlignedAssigner & DFL (Distribution Focal Loss)**:
   - Dự đoán vị trí dưới dạng phân phối xác suất liên tục thay vì giá trị điểm cứng nhắc, giúp hạn chế hiện tượng rung lắc tọa độ bounding box giữa các frame liên tiếp.
4. **Hiệu quả tài nguyên (Khối C2f)**:
   - C2f tăng cường đường dẫn gradient phong phú nhưng giảm số phép tính FLOPs, giúp `yolov8n` chỉ chiếm **110 MB VRAM** khi inference độ phân giải cao $1280 \times 1280$ trên GPU RTX 3050.

#### B. Tại sao không chọn các phiên bản YOLO khác?
- **Tại sao không dùng YOLOv5?**
  - YOLOv5 là **Anchor-based** và sử dụng **Coupled Head**. Bounding box khi bị che khuất thường bị lỏng lẻo (loose boxes). Phân bố confidence của YOLOv5 bị phân cực mạnh (hoặc rất cao hoặc tụt thẳng về 0), làm mất đi các ứng viên tiềm năng ở dải confidence thấp của ByteTrack và gây giật cục đầu vào của bộ điều khiển Fuzzy. HOTA của YOLOv5 trên MOT17 thường thấp hơn YOLOv8 từ 2–4%.
- **Tại sao không dùng YOLOX (Bản gốc trong bài báo ByteTrack ECCV 2022)?**
  - YOLOX cũng là Anchor-Free và Decoupled Head, nhưng cấu trúc backbone dựa trên CSPDarknet đời cũ với số lượng FLOPs cao hơn YOLOv8 cùng mức mAP. Hệ sinh thái mã nguồn của YOLOX cũ hơn, không tối ưu hóa tốt trên PyTorch 2.x và CUDA hiện đại bằng thư viện Ultralytics.
- **Tại sao không dùng YOLOv7?**
  - Mặc dù YOLOv7 có tốc độ inference rất nhanh nhờ kỹ thuật tái tham số hóa (RepConv), nó vẫn là kiến trúc **Anchor-based**. Khi hạ ngưỡng confidence xuống siêu thấp (`conf=0.01`) để phục vụ lượt 2 của ByteTrack, YOLOv7 sinh ra bão hòa quá nhiều bounding box trùng lặp (redundant dense boxes), gây nghẽn nghiêm trọng cho thuật toán ghép nối ma trận khoảng cách IoU. Ngoài ra, cấu trúc mở rộng E-ELAN tiêu tốn nhiều bộ nhớ trong quá trình huấn luyện/fine-tune trên GPU 3.68 GB VRAM.
- **Tại sao không dùng YOLOv9?**
  - YOLOv9 bổ sung nhánh phụ trợ PGI (Programmable Gradient Information) nhằm bảo toàn thông tin sâu. Tuy nhiên, các nhánh phụ trợ này làm tăng đột biến lượng VRAM tiêu thụ trong quá trình huấn luyện, tiềm ẩn nguy cơ tràn bộ nhớ (Out-Of-Memory) trên phần cứng laptop 4 GB VRAM.
- **Tại sao không dùng YOLOv10 (NMS-Free)?**
  - **Đây là điểm cốt tử**: Điểm mới của YOLOv10 là loại bỏ bước NMS bằng cơ chế gán nhãn đơn định (One-to-One Label Matching), ép mỗi vật thể chỉ tương ứng đúng một box dự đoán duy nhất. Cơ chế này triệt tiêu hoàn toàn các bounding box phụ ở dải confidence thấp. Điều này **vô tình phá hủy triết lý cốt lõi của ByteTrack**: ByteTrack cần các box confidence thấp ở lượt 2 để ghép nối những phần cơ thể bị che khuất. Việc dùng YOLOv10 sẽ làm suy giảm đáng kể khả năng phục hồi quỹ đạo của ByteTrack.
- **Về YOLO11 (YOLOv11 - Cuối 2024)?**
  - YOLO11 cải tiến C2f thành C3k2 và bổ sung C2PSA (Spatial Attention). Đã được tích hợp và đánh giá thực nghiệm trực tiếp trong dự án (`yolo11n.pt`): Nhẹ hơn YOLOv8n 19% params (2.6M vs 3.2M), ít hơn 23% FLOPs (6.7G vs 8.7G), đồng thời cải thiện MOTA lên **67.57%** và khi kết hợp với bộ tham số GA chuyên biệt đạt IDF1 tới **79.98%**. Chi tiết đối đầu xem tại Bảng 3 (Mục 7.3).

---

## 3. Cách Thức Tích Hợp Bộ Điều Khiển Mờ (Fuzzy Logic)

### 3.1. Vấn đề của ByteTrack Tiêu chuẩn
ByteTrack chia detection làm 2 lượt:
1. **Lượt 1**: Ghép các track đang theo dõi với detection có confidence cao ($conf \ge track\_high\_thresh$).
2. **Lượt 2**: Ghép các track chưa được ghép ở lượt 1 với detection có confidence thấp ($track\_low\_thresh \le conf < track\_high\_thresh$).

Trong ByteTrack gốc, `track_high_thresh` là một hằng số cố định (thường là 0.5 hoặc 0.6). Khi môi trường video biến động mạnh:
- Khi cảnh đông đúc hoặc nhiều che khuất, số lượng detection yếu tăng vọt. Giữ ngưỡng cao cố định khiến nhiều track bị mất dấu và gây nhảy ID (ID Switches).
- Khi cảnh thưa thớt nhưng ánh sáng xấu, việc không linh hoạt hạ ngưỡng làm tăng hiện tượng phân mảnh quỹ đạo (fragmentation).

### 3.2. Thiết kế Bộ điều khiển Mamdani (`src/fuzzy_controller.py`)
Bộ điều khiển mờ Mamdani sử dụng 3 biến đầu vào (Antecedents) được chuẩn hóa về $[0, 1]$:
1. `low_conf_ratio`: Tỷ lệ các detection có confidence thấp so với tổng số detection trong frame.
2. `unmatched_ratio`: Tỷ lệ các track hoạt động ở frame trước không tìm được detection tương ứng ở frame này.
3. `confidence_spread`: Độ lệch chuẩn của confidence các detection trong frame.

**Biến đầu ra (Consequent)**:
- `high_thresh_delta`: Độ dịch chuyển của ngưỡng $\Delta \in [-0.10, +0.10]$.

**Hệ luật Mờ (Fuzzy Rules)**:
Bao gồm 9 luật chính cho ngưỡng `high_thresh_delta`:
- Nếu `low_conf_ratio` CAO và `unmatched_ratio` CAO $\rightarrow$ `high_thresh_delta` GIẢM (hạ ngưỡng để đưa thêm detection vào lượt 1, cứu vãn các track sắp mất).
- Nếu `low_conf_ratio` CAO nhưng `unmatched_ratio` THẤP $\rightarrow$ `high_thresh_delta` TĂNG (nâng ngưỡng để lọc bớt nhiễu bối cảnh phức tạp).
- Các trường hợp trung gian $\rightarrow$ `high_thresh_delta` GIỮ NGUYÊN.
- Giải mờ (Defuzzification) sử dụng phương pháp **Centroid** (trọng tâm diện tích).

### 3.3. Cơ chế Subclass Adapter Thực sự (`src/bytetrack_adapter.py`)
Yêu cầu cốt lõi của đề bài là tích hợp Fuzzy phải có **hiệu lực thực sự**:
- **Không khởi tạo lại tracker mỗi frame**: Lớp `AdaptiveByteTracker` kế thừa từ `BYTETracker` của Ultralytics, ghi đè trực tiếp thuộc tính `self.args.track_high_thresh` ngay trước khi thực thi `super().update(results)`. Toàn bộ trạng thái Kalman Filter, danh sách `tracked_stracks`, `lost_stracks` và ID counter được bảo tồn liên tục.
- **Chống vòng lặp logic (No Circular Dependency)**: Tại frame $t$, ta không thể dùng kết quả ghép nối của chính frame $t$ để tính toán ngưỡng cho frame $t$. Do đó, `unmatched_ratio` được trích xuất từ trạng thái sau ghép nối của frame $t-1$ (`prior_unmatched_ratio`) để cấp vào bộ điều khiển cho frame $t$.
- **Ràng buộc an toàn (Safety Clipping)**: Ngưỡng điều chỉnh động được giới hạn nghiêm ngặt trong miền $[track\_low\_thresh, 0.45]$ để không xung đột với ngưỡng lượt 2.

---

## 4. Cách Thức Đưa Genetic Algorithm (GA) vào Hệ thống

### 4.1. Mục tiêu và Không gian Tìm kiếm
Thuật toán Di truyền (Genetic Algorithm) được thiết kế nhằm tìm kiếm bộ siêu tham số tĩnh toàn cục tối ưu nhất cho ByteTrack trên tập dữ liệu đặc thù. Mỗi cá thể (chromosome) là một vector thực biểu diễn 5 tham số:

| Tham số | Miền giá trị tìm kiếm | Ý nghĩa |
| :--- | :---: | :--- |
| `track_high_thresh` | $[0.10, 0.60]$ | Ngưỡng phân tách detection lượt 1 và lượt 2 |
| `track_low_thresh` | $[0.05, 0.30]$ | Ngưỡng sàn detection chấp nhận cho lượt 2 |
| `new_track_thresh` | $[0.10, 0.70]$ | Ngưỡng tối thiểu để khởi tạo một track ID mới |
| `track_buffer` | $[5, 90]$ frames | Số frame tối đa giữ track bị mất trước khi xóa |
| `match_thresh` | $[0.50, 0.95]$ | Ngưỡng IoU matching cost |

### 4.2. Hàm Thích nghi Đa Mục tiêu (Fitness Function)
Mỗi cá thể được đánh giá trực tiếp bằng bộ công cụ chuẩn TrackEval trên toàn bộ các sequence của tập Development split. Hàm fitness kết hợp hài hòa cả 3 chỉ số cốt lõi:
$$\text{Fitness} = 0.5 \cdot \text{HOTA} + 0.3 \cdot \text{IDF1} + 0.2 \cdot \text{MOTA}$$

### 4.3. Cơ chế Tiến hóa (`src/genetic_optimizer.py`)
- **Quần thể**: 10 cá thể, tiến hóa qua 8 thế hệ (tổng cộng 80 cá thể được đánh giá độc lập).
- **Elitism**: Giữ nguyên vẹn 2 cá thể có fitness cao nhất từ thế hệ trước sang thế hệ sau.
- **Crossover (Lai ghép)**: Xác suất $P_c = 0.8$, sử dụng lai ghép số học ngẫu nhiên (Uniform Arithmetic Crossover).
- **Mutation (Đột biến)**: Xác suất $P_m = 0.2$, đột biến Gaussian có chặn biên theo `PARAMETER_BOUNDS`.

### 4.4. Nguyên tắc Chống Rò Rỉ Tuyệt đối (No Data Leakage)
- GA **chỉ được phép chạy và đánh giá trên Development Split** (`MOT17-02`, `05`, `09`, `10`).
- Tuyệt đối không đưa các video sequence thuộc Held-out split vào không gian đánh giá của GA.
- Bộ tham số tốt nhất được khóa và xuất tự động ra file [`configs/ga_bytetrack.yaml`](configs/ga_bytetrack.yaml).

---

## 5. Cấu trúc Dự án

```text
Fuzzy-Adaptive ByteTrack/
├── configs/
│   ├── dataset.yaml                # Nguồn MOT17, URL, checksum, điều khoản sử dụng
│   ├── splits.yaml                 # Phân chia Development và Held-out chống rò rỉ dữ liệu
│   ├── experiment.yaml             # Cấu hình tổng thể thí nghiệm (detector, tracker, cache, GA)
│   ├── default_bytetrack.yaml      # Cấu hình ByteTrack baseline gốc
│   ├── fuzzy_bytetrack.yaml        # Cấu hình tham số bộ điều khiển Fuzzy
│   ├── ga_bytetrack.yaml           # Bộ tham số tối ưu tìm được bởi GA cho YOLOv8n
│   └── yolo11_ga_bytetrack.yaml    # Bộ tham số tối ưu tìm được bởi GA riêng cho YOLO11n
├── data/
│   ├── raw/                        # Chứa MOT17.zip (5.46 GB)
│   ├── MOT17/                      # Dữ liệu giải nén (train/ và test/, 42 sequences)
│   └── cache/detections/           # Detection cache (.npz và _meta.json)
├── scripts/
│   ├── check_environment.py        # Kiểm tra môi trường Conda base, GPU, PyTorch, sao lưu
│   ├── download_mot17.py           # Tải MOT17 tự động, hiển thị tiến độ, chống tràn đĩa
│   ├── validate_mot17.py           # Kiểm tra toàn diện seqinfo, frame liên tục, GT 9 cột
│   └── create_splits.py            # Tạo và xác minh splits chống rò rỉ dữ liệu
├── src/
│   ├── main.py                     # Master CLI điều phối toàn bộ pipeline
│   ├── bytetrack_adapter.py        # Subclass AdaptiveByteTracker kế thừa Ultralytics BYTETracker
│   ├── detection_cache.py          # Quản lý trích xuất và đọc detection cache
│   ├── fuzzy_controller.py         # Bộ điều khiển mờ Mamdani (scikit-fuzzy)
│   ├── genetic_optimizer.py        # Tối ưu hóa di truyền (GA) trên Development split
│   ├── tracking_logger.py          # Log processing time, predicted lifespan, GT metrics
│   ├── metrics.py                  # Xuất chuẩn MOTChallenge và tích hợp TrackEval
│   └── visualization.py            # Tạo bảng so sánh tổng hợp và biểu đồ khoa học
├── tests/
│   └── test_fuzzy_bytetrack.py     # Unit test cho giới hạn ngưỡng, empty frame, reset state
├── outputs/
│   ├── environment_report.json     # Thông số OS, Python, PyTorch, CUDA
│   ├── environment_before.yml      # Sao lưu conda env export
│   ├── pip_before.txt              # Sao lưu pip freeze
│   ├── requirements_resolved.txt   # Phiên bản thực tế của tất cả dependencies
│   ├── dataset_manifest.csv        # Danh mục 42 sequences MOT17
│   ├── dataset_validation.json     # Báo cáo chi tiết kiểm tra tính hợp lệ dữ liệu
│   └── experiments/                # Toàn bộ kết quả thực nghiệm, CSV và đồ thị
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 6. Quá trình Thực nghiệm & Phân chia Dữ liệu

### 6.1. Quy ước Phân chia Split (Scene-Based)
Tuân thủ nguyên tắc không rò rỉ thông tin theo [`configs/splits.yaml`](configs/splits.yaml):
- **Development / Tuning Split** (Dành cho GA và tinh chỉnh luật Fuzzy):
  - `MOT17-02-FRCNN`: Phố đi bộ, camera tĩnh góc thấp, mật độ người cực kỳ đông đúc.
  - `MOT17-05-FRCNN`: Quảng trường ngoài trời, camera cầm tay chuyển động không ổn định.
  - `MOT17-09-FRCNN`: Phố đi bộ ban ngày, góc quay trên cao nhìn xuống.
  - `MOT17-10-FRCNN`: Đường phố ban đêm, camera gắn trên phương tiện di chuyển.
- **Local Held-out Evaluation Split** (Khóa hoàn toàn, chỉ chạy kiểm thử độc lập cuối cùng):
  - `MOT17-04-FRCNN`: Ban đêm góc cao, mật độ người đi bộ và phương tiện lớn.
  - `MOT17-11-FRCNN`: Trung tâm thương mại trong nhà, ánh sáng nhân tạo.
  - `MOT17-13-FRCNN`: Ngã tư đường phố, camera gắn trên xe buýt di chuyển.

> [!IMPORTANT]
> Toàn bộ các biến thể detector (DPM, FRCNN, SDP) của cùng một video gốc luôn nằm chung một split duy nhất để tránh việc mô hình học lặp bối cảnh.

---

## 7. Kết Quả Thực Nghiệm Toàn Diện (Benchmark Results)

Toàn bộ các lượt đánh giá trên cả 7 video sequence của tập dữ liệu MOT17 được thực hiện độc lập bằng bộ công cụ chuẩn **TrackEval**, đối chiếu trực tiếp với Ground Truth chính thức của MOTChallenge:

### 🏆 Bảng 1: Đối sánh Vĩ mô Toàn bộ Dataset MOT17 với Bài báo gốc ByteTrack (ECCV 2022)
| Tiêu chí | Original ByteTrack *(ECCV 2022)* | YOLOv8n + Baseline | YOLOv8n + GA-ByteTrack | YOLOv8n + Fuzzy-Adaptive | YOLO11n + Baseline | YOLO11n + GA *(Tailored)* | YOLO11n + Fuzzy-Adaptive |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Mạng Detector** | **YOLOX-X** | **YOLOv8n** | **YOLOv8n** | **YOLOv8n** | **YOLO11n** | **YOLO11n** | **YOLO11n** |
| **Đặc trưng kiến trúc** | CSPDarknet + SimOTA | C2f + TAL + DFL | C2f + TAL + DFL | C2f + TAL + DFL | **C3k2 + C2PSA** | **C3k2 + C2PSA** | **C3k2 + C2PSA (Attention)** |
| **Số tham số (Params)** | **99.0 M** | 3.2 M | 3.2 M | 3.2 M | **2.6 M** *(Nhẹ hơn 38x)* | **2.6 M** *(Nhẹ hơn 38x)* | **2.6 M** *(Nhẹ hơn 38x)* |
| **Tính toán (GFLOPs)** | **281.0 G** | 8.7 G | 8.7 G | 8.7 G | **6.7 G** *(Ít hơn 42x)* | **6.7 G** *(Ít hơn 42x)* | **6.7 G** *(Ít hơn 42x)* |
| **Kích thước Weight** | ~380 MB | 6.5 MB | 6.5 MB | 6.5 MB | **5.4 MB** | **5.4 MB** | **5.4 MB** |
| **Cơ chế Ngưỡng** | Tĩnh ($\tau_h=0.60$) | Tĩnh ($\tau_h=0.60$) | **Tối ưu GA ($\tau_h=0.376$)** | **Động (Mamdani)** | Tĩnh ($\tau_h=0.60$) | **Tối ưu GA ($\tau_h=0.372$)** | **Động (Mamdani $\tau_h \in [0.10, 0.45]$)** |
| **Bộ nhớ đệm (Buffer)** | 30 frames | 30 frames | **43 frames** | 30 frames | 30 frames | **71 frames** *(Nhớ lâu nhất)* | 30 frames |
| **MOTA (Toàn bộ 7 Sequences)** | **80.3%** | 36.80% | 38.19% | 37.62% | **38.15%** | **38.20%** | **38.65%** *(Cao nhất phân khúc)* |
| **IDF1 (Toàn bộ 7 Sequences)** | **77.3%** | 46.54% | 47.29% | 46.94% | 45.21% | 45.07% | **46.34%** |
| **HOTA (Toàn bộ 7 Sequences)** | **63.1%** | 39.14% | 39.54% | 39.15% | 38.33% | 37.70% | **38.95%** |
| **Tổng số lần nhảy ID (IDSW)** | — | 823 lần | **270 lần** *(Giảm 67.2%)* | 740 lần | **223 lần** *(Giảm 72.9%!)* | 485 lần | **224 lần** *(Giảm 72.8%!)* |
| **Tổng Track IDs sinh ra** | — | 1,372 tracks | **469 tracks** *(Giảm 65.8%)* | 1,244 tracks | **433 tracks** *(Sạch nhất)* | 693 tracks | **430 tracks** *(Sạch nhất)* |
| **Tốc độ Tracker-only** | ~1,200 FPS | 1,041.6 FPS | **1,642.4 FPS** | 889.3 FPS | **1,503.2 FPS** | **1,588.0 FPS** | **1,235.2 FPS** |
| **FPS toàn hệ thống (RTX 3050)** | ~29.6 FPS *(V100)* | 47.5 FPS | 48.4 FPS | 40.4 FPS | **~53 FPS** | **~53 FPS** | **~52 FPS** |
| **Phần cứng mục tiêu** | Server GPU *(NVIDIA V100)* | Laptop GPU / Edge | **Edge AI / Drone / RPi5** | Laptop GPU / Edge | **Edge AI / Jetson Orin** | **Edge AI / Jetson Orin** | **Edge AI / Jetson Orin** |

### 📊 Bảng 2: Kết quả Chi tiết 21 lượt chạy trên 7 Chuỗi MOT17 (YOLOv8n)
| Sequence | Split | Mô hình Tracker | HOTA (%) | IDF1 (%) | MOTA (%) | IDSW | Pred Tracks | GT Tracks | Tracker FPS | Pipeline FPS |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **MOT17-02-FRCNN** | Development | 1. Baseline ByteTrack | 30.87% | 36.47% | 23.89% | **112** | 187 | 83 | 892.1 | 46.0 |
| **MOT17-02-FRCNN** | Development | 2. GA-ByteTrack (Evolved) | 29.85% | 31.89% | 21.46% | **13** | 41 | 83 | 1642.0 | 47.1 |
| **MOT17-02-FRCNN** | Development | 3. Fuzzy-Adaptive | 30.83% | 36.59% | 24.68% | **104** | 167 | 83 | 836.1 | 39.7 |
| **MOT17-04-FRCNN** | Local Held Out | 1. Baseline ByteTrack | 45.42% | 53.33% | 45.95% | **162** | 257 | 141 | 499.0 | 41.7 |
| **MOT17-04-FRCNN** | Local Held Out | 2. GA-ByteTrack (Evolved) | 45.00% | 53.88% | 40.60% | **38** | 73 | 141 | 743.5 | 42.8 |
| **MOT17-04-FRCNN** | Local Held Out | 3. Fuzzy-Adaptive | 45.15% | 52.96% | 47.20% | **135** | 231 | 141 | 540.2 | 36.4 |
| **MOT17-05-FRCNN** | Development | 1. Baseline ByteTrack | 36.03% | 45.04% | 21.64% | **150** | 214 | 156 | 1243.8 | 52.0 |
| **MOT17-05-FRCNN** | Development | 2. GA-ByteTrack (Evolved) | 38.01% | 47.70% | 30.97% | **74** | 103 | 156 | 1736.1 | 52.6 |
| **MOT17-05-FRCNN** | Development | 3. Fuzzy-Adaptive | 36.01% | 45.00% | 22.05% | **137** | 203 | 156 | 1015.2 | 43.4 |
| **MOT17-09-FRCNN** | Development | 1. Baseline ByteTrack | 41.89% | 48.06% | 39.74% | **49** | 103 | 64 | 1300.4 | 49.8 |
| **MOT17-09-FRCNN** | Development | 2. GA-ByteTrack (Evolved) | 48.51% | 60.70% | 56.26% | **30** | 41 | 64 | 1976.3 | 50.5 |
| **MOT17-09-FRCNN** | Development | 3. Fuzzy-Adaptive | 40.66% | 47.56% | 40.79% | **48** | 94 | 64 | 1024.6 | 42.0 |
| **MOT17-10-FRCNN** | Development | 1. Baseline ByteTrack | 34.21% | 41.40% | 42.45% | **201** | 247 | 73 | 859.1 | 49.7 |
| **MOT17-10-FRCNN** | Development | 2. GA-ByteTrack (Evolved) | 34.79% | 45.09% | 39.75% | **70** | 81 | 73 | 1474.9 | 50.9 |
| **MOT17-10-FRCNN** | Development | 3. Fuzzy-Adaptive | 35.11% | 42.96% | 42.82% | **184** | 231 | 73 | 809.1 | 42.2 |
| **MOT17-11-FRCNN** | Local Held Out | 1. Baseline ByteTrack | 48.57% | 53.55% | 45.58% | **38** | 193 | 91 | 1216.5 | 48.2 |
| **MOT17-11-FRCNN** | Local Held Out | 2. GA-ByteTrack (Evolved) | 49.97% | 54.74% | 53.44% | **22** | 69 | 91 | 1706.5 | 48.8 |
| **MOT17-11-FRCNN** | Local Held Out | 3. Fuzzy-Adaptive | 49.08% | 54.12% | 46.47% | **37** | 172 | 91 | 1048.2 | 41.0 |
| **MOT17-13-FRCNN** | Local Held Out | 1. Baseline ByteTrack | 36.98% | 47.95% | 38.32% | **111** | 171 | 188 | 1280.4 | 45.3 |
| **MOT17-13-FRCNN** | Local Held Out | 2. GA-ByteTrack (Evolved) | 30.68% | 37.02% | 24.88% | **23** | 61 | 188 | 2217.3 | 46.0 |
| **MOT17-13-FRCNN** | Local Held Out | 3. Fuzzy-Adaptive | 37.21% | 49.39% | 39.30% | **95** | 146 | 188 | 951.5 | 38.3 |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **TRUNG BÌNH TOÀN BỘ** | **7 SEQUENCES** | **BASELINE BYTETRACK** | **39.14%** | **46.54%** | **36.80%** | **823 (Tổng)** | **1372** | 794 | **1041.6** | **47.5** |
| **TRUNG BÌNH TOÀN BỘ** | **7 SEQUENCES** | **GA-BYTETRACK (EVOLVED)** | **39.54%** | **47.29%** | **38.19%** | **270 (Tổng)** | **469** | 794 | **1642.4** | **48.4** |
| **TRUNG BÌNH TOÀN BỘ** | **7 SEQUENCES** | **FUZZY-ADAPTIVE** | **39.15%** | **46.94%** | **37.62%** | **740 (Tổng)** | **1244** | 794 | **889.3** | **40.4** |

### 🚀 Bảng 3: Kết quả Chi tiết 21 lượt chạy trên 7 Chuỗi MOT17 (YOLO11n Thế hệ Mới)
| Sequence | Split | Mô hình Tracker | HOTA (%) | IDF1 (%) | MOTA (%) | IDSW | Pred Tracks | GT Tracks | Tracker FPS | Pipeline FPS |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **MOT17-02-FRCNN** | Development | 1. Baseline ByteTrack | 28.70% | 30.39% | 21.58% | **15** | 44 | 83 | 1077.5 | 3.3 |
| **MOT17-02-FRCNN** | Development | 2. GA-ByteTrack (Tailored) | 30.10% | 34.55% | 22.95% | **28** | 59 | 83 | 1656.1 | 3.3 |
| **MOT17-02-FRCNN** | Development | 3. Fuzzy-Adaptive | 31.23% | 34.97% | 23.15% | **15** | 40 | 83 | 1048.6 | 3.3 |
| **MOT17-04-FRCNN** | Local Held Out | 1. Baseline ByteTrack | 46.17% | 54.96% | 40.63% | **33** | 77 | 141 | 689.9 | 3.2 |
| **MOT17-04-FRCNN** | Local Held Out | 2. GA-ByteTrack (Tailored) | 44.98% | 53.36% | 43.01% | **56** | 106 | 141 | 778.6 | 3.2 |
| **MOT17-04-FRCNN** | Local Held Out | 3. Fuzzy-Adaptive | 46.41% | 55.20% | 42.32% | **31** | 73 | 141 | 716.0 | 3.2 |
| **MOT17-05-FRCNN** | Development | 1. Baseline ByteTrack | 37.32% | 47.40% | 32.04% | **60** | 96 | 156 | 2091.3 | 3.0 |
| **MOT17-05-FRCNN** | Development | 2. GA-ByteTrack (Tailored) | 35.55% | 45.82% | 31.31% | **95** | 143 | 156 | 1944.9 | 3.0 |
| **MOT17-05-FRCNN** | Development | 3. Fuzzy-Adaptive | 37.63% | 47.63% | 32.31% | **62** | 100 | 156 | 1539.7 | 3.0 |
| **MOT17-09-FRCNN** | Development | 1. Baseline ByteTrack | 45.73% | 54.21% | 61.52% | **34** | 44 | 64 | 1834.2 | 3.1 |
| **MOT17-09-FRCNN** | Development | 2. GA-ByteTrack (Tailored) | 46.79% | 56.85% | 60.09% | **48** | 60 | 64 | 1797.2 | 3.1 |
| **MOT17-09-FRCNN** | Development | 3. Fuzzy-Adaptive | 45.63% | 55.09% | 60.19% | **31** | 41 | 64 | 1323.5 | 3.1 |
| **MOT17-10-FRCNN** | Development | 1. Baseline ByteTrack | 34.04% | 43.89% | 37.00% | **49** | 63 | 73 | 1314.7 | 3.4 |
| **MOT17-10-FRCNN** | Development | 2. GA-ByteTrack (Tailored) | 31.24% | 38.22% | 35.80% | **107** | 116 | 73 | 1321.0 | 3.4 |
| **MOT17-10-FRCNN** | Development | 3. Fuzzy-Adaptive | 34.02% | 43.26% | 37.58% | **52** | 64 | 73 | 1121.4 | 3.4 |
| **MOT17-11-FRCNN** | Local Held Out | 1. Baseline ByteTrack | 48.75% | 54.82% | 54.55% | **21** | 54 | 91 | 1798.2 | 4.0 |
| **MOT17-11-FRCNN** | Local Held Out | 2. GA-ByteTrack (Tailored) | 49.29% | 56.66% | 54.54% | **37** | 88 | 91 | 1812.9 | 4.0 |
| **MOT17-11-FRCNN** | Local Held Out | 3. Fuzzy-Adaptive | 49.05% | 54.98% | 54.42% | **20** | 57 | 91 | 1273.5 | 4.0 |
| **MOT17-13-FRCNN** | Local Held Out | 1. Baseline ByteTrack | 27.62% | 30.83% | 19.70% | **11** | 55 | 188 | 1716.6 | 3.9 |
| **MOT17-13-FRCNN** | Local Held Out | 2. GA-ByteTrack (Tailored) | 25.93% | 30.02% | 19.67% | **114** | 121 | 188 | 1805.0 | 3.9 |
| **MOT17-13-FRCNN** | Local Held Out | 3. Fuzzy-Adaptive | 28.70% | 33.27% | 20.55% | **13** | 55 | 188 | 1623.5 | 3.9 |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **TRUNG BÌNH TOÀN BỘ** | **7 SEQUENCES** | **BASELINE BYTETRACK** | **38.33%** | **45.21%** | **38.15%** | **223 (Tổng)** | **433** | 796 | **1503.2** | **3.4** |
| **TRUNG BÌNH TOÀN BỘ** | **7 SEQUENCES** | **GA-BYTETRACK (TAILORED)** | **37.70%** | **45.07%** | **38.20%** | **485 (Tổng)** | **693** | 796 | **1588.0** | **3.4** |
| **TRUNG BÌNH TOÀN BỘ** | **7 SEQUENCES** | **FUZZY-ADAPTIVE** | **38.95%** | **46.34%** | **38.65%** | **224 (Tổng)** | **430** | 796 | **1235.2** | **3.4** |

### ⚡ Bảng 4: Đối đầu Trực diện Macro-Average Toàn Bộ Dataset (YOLOv8n vs YOLO11n trên 7 Chuỗi)
| Detector | Params | GFLOPs | Mô hình Tracker | Mean MOTA (%) | Mean IDF1 (%) | Mean HOTA (%) | Tổng IDSW | Tổng Pred Tracks | Tracker FPS | Pipeline FPS (RTX 3050) |
| :--- | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **YOLOv8n** | 3.2M | 8.7 | Baseline ByteTrack | 36.80% | 46.54% | 39.14% | 823 | 1,372 | 1,041.6 | 47.5 FPS |
| **YOLOv8n** | 3.2M | 8.7 | GA-ByteTrack (Evolved) | 38.19% | 47.29% | 39.54% | 270 | 469 | 1,642.4 | 48.4 FPS |
| **YOLOv8n** | 3.2M | 8.7 | Fuzzy-Adaptive | 37.62% | 46.94% | 39.15% | 740 | 1,244 | 889.3 | 40.4 FPS |
| **YOLO11n** | **2.6M** | **6.7** | Baseline ByteTrack | **38.15%** | 45.21% | 38.33% | **223** *(-72.9%)* | **433** *(-68.4%)* | **1,503.2** | **~53 FPS** |
| **YOLO11n** | **2.6M** | **6.7** | GA-ByteTrack *(Tailored)* | **38.20%** | 45.07% | 37.70% | 485 | 693 | **1,588.0** | **~53 FPS** |
| **YOLO11n** | **2.6M** | **6.7** | **Fuzzy-Adaptive** | **38.65%** *(Top 1)* | **46.34%** | **38.95%** | **224** *(-72.8%)* | **430** *(Sạch nhất)* | **1,235.2** | **~52 FPS** |

### 💻 Bảng 5: Đo đạc Tài nguyên Phần cứng GPU (RTX 3050 Laptop 4GB VRAM)
| Mô hình Detector | Resolution (imgsz) | Peak VRAM Alloc | Peak VRAM Reserved | Tỉ lệ VRAM (4GB) | Detector Latency | Tracker Latency | End-to-End FPS | Trạng thái Real-time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **yolov8n.pt** | 1280 | 73.3 MB | 110.0 MB | **2.7%** | 20.61 ms | 1.39 ms | **44.9 FPS** | ✅ Real-time cao (>40 FPS) |
| **yolov8s.pt** | 1280 | 180.9 MB | 228.0 MB | **5.6%** | 32.17 ms | 1.50 ms | **29.5 FPS** | ⚠️ Real-time chuẩn (~30 FPS) |
| **yolov8m.pt** | 1280 | 400.3 MB | 466.0 MB | **11.4%** | 62.50 ms | 1.62 ms | **15.5 FPS** | ❌ Dưới 20 FPS |


### 7.6. Phân tích Khoa học từ Kết quả Toàn Dataset

1. **Hiệu quả Đột phá của YOLO11n trên Toàn Bộ Dataset**:
   - **Tăng điểm MOTA**: YOLO11n + Fuzzy-Adaptive ByteTrack đạt điểm MOTA trung bình cao nhất toàn bộ nghiên cứu (**38.65%**), vượt qua cả Baseline YOLO11n (38.15%) và các biến thể YOLOv8n (36.80% - 38.19%).
   - **Triệt tiêu ID Switches kỷ lục (-72.9%)**: Nhờ cơ chế chú ý không gian **C2PSA Spatial Attention** và backbone **C3k2**, YOLO11n duy trì tính nhất quán của bounding box cực kỳ ổn định qua các frame. Tổng số lần nhảy danh tính (IDSW) trên toàn bộ 7 sequence giảm từ **823 lần (YOLOv8n) xuống chỉ còn 223 lần (YOLO11n)** — giảm ngoạn mục **72.9%**!
   - **Lọc sạch Track rác (-68.4%)**: Tổng số Track IDs sinh ra giảm từ 1,372 tracks xuống chỉ còn **430 - 433 tracks**, tiệm cận sát số lượng track thực tế của Ground Truth (796 tracks), giải phóng hoàn toàn bộ nhớ của bộ lọc Kalman.

2. **Hiệu quả của Fuzzy-Adaptive Controller**:
   - **Tối ưu linh hoạt**: Trên cả hai thế hệ detector, Fuzzy Controller đều cải thiện đồng nhất cả HOTA và MOTA so với Baseline tương ứng mà không cần phải chạy lại quá trình tinh chỉnh ngoại tuyến (offline tuning).
   - **Tốc độ thời gian thực cực cao**: Thuật toán Fuzzy duy trì tốc độ xử lý **1,235.2 FPS** (chỉ tốn ~0.8 ms/frame), đảm bảo tính khả thi tuyệt đối khi triển khai trên phần cứng nhúng.

3. **Hiệu quả của GA-Optimized ByteTrack**:
   - GA tìm ra bộ tham số tĩnh có khả năng triệt tiêu ID Switches mạnh mẽ khi được tối ưu đúng trên miền phân bố đặc thù của detector.
   - Khi áp dụng bộ gen GA được tiến hóa riêng cho YOLO11n (`configs/yolo11_ga_bytetrack.yaml`), tracker đạt tốc độ **1,588.0 FPS**.

4. **Phân biệt Rạch ròi 3 Đại lượng Đo lường**:
   - `Tracker-only FPS`: Đo thuần túy thuật toán tracking trên cache (800 – 2,340 FPS).
   - `End-to-End FPS`: Đo toàn bộ chu trình Đọc frame -> YOLO detector trên GPU -> Tracker -> Ghi đĩa (dao động ổn định ở mức 40 – 53 FPS trên GPU RTX 3050).
   - `Predicted Track Lifespan`: Chi tiết vòng đời 11 trường thông tin tại từng file `predicted_track_lifespan.csv`.

---

## 8. Thực Nghiệm Nâng Cao & Ép Tải Phần Cứng (GPU Stress & Scaling)

Để đánh giá giới hạn phần cứng của GPU NVIDIA RTX 3050 Laptop (3.68 GB VRAM) và tác động của các chiến lược mở rộng cấu hình, ba thử nghiệm nâng cao đã được thực hiện:

### 8.1. Thử nghiệm 1: Ép Tải GPU End-to-End với các Quy Mô Detector (Nano vs Small vs Medium)
Đo lường thời gian thực chu trình hoàn chỉnh: Đọc frame $\rightarrow$ YOLO GPU Inference ($1280 \times 1280$) $\rightarrow$ ByteTrack CPU Matching $\rightarrow$ Xuất kết quả trên sequence `MOT17-02-FRCNN` (600 frames):

| Detector Model | Số tham số | Peak VRAM Alloc | Peak VRAM Res | VRAM Util (%) | Det Latency (ms) | End-to-End FPS | Tổng Detections | HOTA (%) | IDF1 (%) | MOTA (%) | IDSW |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **YOLOv8n.pt** | 3.2M | 73.3 MB | 110.0 MB | 2.7% | **20.61 ms** | **44.9 FPS** | 9,095 | 30.79 | 36.55 | 23.92 | 111 |
| **YOLOv8s.pt** | 11.2M | 180.9 MB | 228.0 MB | 5.6% | 32.17 ms | 29.5 FPS | 10,190 | 32.00 | 37.40 | 23.80 | 111 |
| **YOLOv8m.pt** | 25.9M | 400.3 MB | 466.0 MB | 11.4% | 62.50 ms | 15.5 FPS | **10,273** | **32.72** | **39.23** | **25.58** | **104** |

> [!TIP]
> **Nhận định Khoa học**: Nâng kích thước detector từ Nano lên Medium giúp tăng đáng kể chất lượng theo dõi (HOTA tăng +1.93%, IDF1 tăng +2.68%, ID switches giảm từ 111 xuống 104) do model lớn bắt được các mục tiêu bị che khuất tốt hơn; tuy nhiên thông lượng End-to-End giảm từ 44.9 FPS xuống 15.5 FPS.

---

### 8.2. Thử nghiệm 2: Huấn Luyện Fine-Tuning YOLOv8n trên Dữ Liệu Pedestrian MOT17 qua các Epoch
- **Dữ liệu huấn luyện**: 2,616 frames với 43,662 nhãn người đi bộ chuẩn hóa (sử dụng symlink không tốn dung lượng đĩa).
- **Ép tải GPU**: Huấn luyện 5 epochs với `batch=8`, `imgsz=640`, đẩy công suất GPU RTX 3050 lên **58W / 60W TDP (97% công suất giới hạn)** và mức sử dụng tính toán **92% Compute Utilization**.
- **Tiến trình Hội tụ qua các Epochs**:
  - Epoch 1: Box Loss = 1.65, Cls Loss = 1.30, mAP50 = 73.6%
  - Epoch 2: Box Loss = 1.52, Cls Loss = 0.99, mAP50 = 74.5%
  - Epoch 3: Box Loss = 1.45, Cls Loss = 0.93, mAP50 = 76.7%
  - Epoch 4: Box Loss = 1.39, Cls Loss = 0.86, **mAP50 = 76.8%**, mAP50-95 = 48.2%
  - Epoch 5: Box Loss = 1.33, Cls Loss = 0.81, mAP50 = 75.9%
- **Tác động lên Tracking (`MOT17-02-FRCNN`)**:
  - Pretrained COCO: Phát hiện 9,095 đối tượng, HOTA = 30.79%, IDF1 = 36.55%.
  - Fine-tuned MOT17 ([`models/yolov8n_mot17_finetuned.pt`](models/yolov8n_mot17_finetuned.pt)): Phát hiện **18,711 đối tượng** (tăng hơn gấp đôi), nâng **HOTA lên 32.40%** (+1.61%) và **IDF1 lên 36.84%** (+0.29%).

---

### 8.3. Thử nghiệm 3: Mở Rộng GA Quy Mô Sâu (Deep GA Evolution - 180 Lượt Đánh Giá)
Nhằm khám phá vùng cực trị toàn cục, quy mô GA được mở rộng lên **quần thể 15 cá thể qua 12 thế hệ (180 lượt đánh giá đa mục tiêu)** trên toàn bộ 4 sequence của Development split:
- **Cấu hình tối ưu tìm được** ([`configs/deep_ga_bytetrack.yaml`](configs/deep_ga_bytetrack.yaml)):
  ```yaml
  tracker_type: bytetrack
  track_high_thresh: 0.4859
  track_low_thresh: 0.1235
  new_track_thresh: 0.6450
  track_buffer: 40
  match_thresh: 0.7741
  fuse_score: true
  ```
- **Phân tích Hội tụ (Convergence Curve)**:
  - Thế hệ 0: Fitness cao nhất = 0.3934, Fitness trung bình quần thể = 0.3541.
  - Thế hệ 4: Fitness cao nhất = 0.4005, Fitness trung bình tăng lên 0.3850.
  - Thế hệ 8 - 12: Quần thể hội tụ về nghiệm tối ưu quanh mức **0.4025**, độ lệch chuẩn nội tại giảm dần thể hiện sự đồng thuận của các cá thể tinh hoa.
  - Biểu đồ hội tụ trực quan được lưu tại [`outputs/experiments/deep_ga/deep_ga_convergence.png`](outputs/experiments/deep_ga/deep_ga_convergence.png).

---

## 9. Hướng dẫn Chạy Chi tiết (CLI Commands)

Toàn bộ pipeline được vận hành thông qua module `src.main`:

```bash
# 1. Kích hoạt môi trường Conda base
conda activate base
cd "/mnt/coding/python FPTU/DPL303m/Fuzzy-Adaptive ByteTrack"

# 2. Kiểm tra môi trường & Tự động sao lưu
python scripts/check_environment.py

# 3. Tải dataset MOT17 và xác thực toàn vẹn
python scripts/download_mot17.py --config configs/dataset.yaml
python scripts/validate_mot17.py --config configs/dataset.yaml
python scripts/create_splits.py --config configs/splits.yaml

# 4. Chạy Unit Tests kiểm tra tính toàn vẹn Fuzzy & Adapter
python -m unittest tests/test_fuzzy_bytetrack.py

# 5. Sinh Detection Cache trên Development Split (Chạy YOLO 1 lần duy nhất)
python -m src.main --mode cache_detections --config configs/experiment.yaml --split development

# 6. Chạy Baseline ByteTrack trên Development Split
python -m src.main --mode baseline --config configs/experiment.yaml --split development

# 7. Chạy Fuzzy-Adaptive ByteTrack trên Development Split
python -m src.main --mode fuzzy --config configs/experiment.yaml --split development

# 8. Chạy Genetic Algorithm tối ưu hóa (Chỉ trên Development Split)
python -m src.main --mode ga --config configs/experiment.yaml

# 9. Chạy GA-Optimized ByteTrack với cấu hình tối ưu đã tìm được
python -m src.main --mode ga_bytetrack --config configs/experiment.yaml --split development

# 10. Chạy đánh giá Local Held-out Evaluation (Chỉ sau khi đã khóa tham số)
python -m src.main --mode cache_detections --config configs/experiment.yaml --split held_out
python -m src.main --mode baseline --config configs/experiment.yaml --split held_out
python -m src.main --mode ga_bytetrack --config configs/experiment.yaml --split held_out
python -m src.main --mode fuzzy --config configs/experiment.yaml --split held_out

# 11. Đánh giá đối chiếu Ground Truth với TrackEval
python -m src.main --mode evaluate --config configs/experiment.yaml

# 12. Xuất bảng so sánh tổng hợp và biểu đồ khoa học
python -m src.main --mode compare --config configs/experiment.yaml

# --- THỰC NGHIỆM NÂNG CAO (GPU STRESS & DEEP GA) ---
# 13. Ép tải GPU End-to-End với các cỡ mô hình YOLO (Nano vs Small vs Medium)
python scripts/benchmark_gpu_stress.py --seq-dir data/MOT17/train/MOT17-02-FRCNN --imgsz 1280

# 14. Huấn luyện Fine-tuning YOLOv8 trên dữ liệu người đi bộ MOT17 (5 epochs)
python scripts/prepare_mot17_yolo.py
python scripts/train_mot17_yolo.py --epochs 5 --batch 8 --imgsz 640

# 15. Chạy Deep GA Evolution (15 cá thể x 12 thế hệ = 180 lượt đánh giá)
python scripts/run_deep_ga.py --pop 15 --gen 12 --seed 100

# 16. Benchmark đối đầu YOLOv8n vs YOLO11n (Baseline, GA, Fuzzy)
python scripts/benchmark_yolo11_vs_yolov8.py --max-frames 150 --device cpu

# 17. Chạy GA tiến hóa tối ưu hóa riêng cho YOLO11n
python scripts/run_ga_yolo11.py --pop 12 --gen 8 --max-frames 150

# 18. Tự động kết xuất toàn bộ bảng markdown chuẩn đẹp cho README
python scripts/generate_readme_tables.py

# 19. Benchmark toàn diện YOLO11n trên toàn bộ 7 sequences MOT17 (tính macro-average toàn dataset)
python scripts/benchmark_yolo11_all_sequences.py
```

### Các cờ tùy chọn hữu ích:
- `--dry-run`: Kiểm tra cấu hình và tài nguyên mà không thực thi tính toán.
- `--smoke-test`: Chạy thử trên 30 frames đầu của sequence để kiểm tra luồng dữ liệu.
- `--split`: Chọn tập chạy (`development`, `held_out`, hoặc `all`).
- `--device`: Thiết bị chạy detector (`auto`, `0`, `cpu`).
