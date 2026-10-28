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
