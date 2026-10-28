"""Tự động tổng hợp toàn bộ kết quả thực nghiệm thành các bảng Markdown chuẩn đẹp cho README.md.

Các bảng được xuất:
1. Bảng đối sánh vĩ mô toàn bộ dataset MOT17: Original ByteTrack (ECCV 2022) vs YOLOv8n vs YOLO11n (Trung bình Macro 7 Sequences)
2. Bảng chi tiết 21 lượt chạy trên 7 Chuỗi MOT17 của YOLOv8n + Hàng tổng kết trung bình
3. Bảng chi tiết 21 lượt chạy trên 7 Chuỗi MOT17 của YOLO11n (Mới) + Hàng tổng kết trung bình
4. Bảng đối đầu trực diện Macro-Average Toàn Dataset: YOLOv8n vs YOLO11n
5. Bảng đo đạc tài nguyên phần cứng GPU (VRAM, Latency, FPS trên RTX 3050)
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd


def generate_tables() -> str:
    md_output = []

    # =========================================================================
    # BẢNG 1: ĐỐI SÁNH VĨ MÔ TOÀN BỘ DATASET VỚI BÀI BÁO GỐC (ECCV 2022)
    # =========================================================================
    md_output.append("### 🏆 Bảng 1: Đối sánh Vĩ mô Toàn bộ Dataset MOT17 với Bài báo gốc ByteTrack (ECCV 2022)\n")
    md_output.append(
        "| Tiêu chí | Original ByteTrack *(ECCV 2022)* | YOLOv8n + Baseline | YOLOv8n + GA-ByteTrack | YOLOv8n + Fuzzy-Adaptive | YOLO11n + Baseline | YOLO11n + GA *(Tailored)* | YOLO11n + Fuzzy-Adaptive |\n"
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n"
        "| **Mạng Detector** | **YOLOX-X** | **YOLOv8n** | **YOLOv8n** | **YOLOv8n** | **YOLO11n** | **YOLO11n** | **YOLO11n** |\n"
        "| **Đặc trưng kiến trúc** | CSPDarknet + SimOTA | C2f + TAL + DFL | C2f + TAL + DFL | C2f + TAL + DFL | **C3k2 + C2PSA** | **C3k2 + C2PSA** | **C3k2 + C2PSA (Attention)** |\n"
        "| **Số tham số (Params)** | **99.0 M** | 3.2 M | 3.2 M | 3.2 M | **2.6 M** *(Nhẹ hơn 38x)* | **2.6 M** *(Nhẹ hơn 38x)* | **2.6 M** *(Nhẹ hơn 38x)* |\n"
        "| **Tính toán (GFLOPs)** | **281.0 G** | 8.7 G | 8.7 G | 8.7 G | **6.7 G** *(Ít hơn 42x)* | **6.7 G** *(Ít hơn 42x)* | **6.7 G** *(Ít hơn 42x)* |\n"
        "| **Kích thước Weight** | ~380 MB | 6.5 MB | 6.5 MB | 6.5 MB | **5.4 MB** | **5.4 MB** | **5.4 MB** |\n"
        r"| **Cơ chế Ngưỡng** | Tĩnh ($\tau_h=0.60$) | Tĩnh ($\tau_h=0.60$) | **Tối ưu GA ($\tau_h=0.376$)** | **Động (Mamdani)** | Tĩnh ($\tau_h=0.60$) | **Tối ưu GA ($\tau_h=0.372$)** | **Động (Mamdani $\tau_h \in [0.10, 0.45]$)** |" "\n"
        "| **Bộ nhớ đệm (Buffer)** | 30 frames | 30 frames | **43 frames** | 30 frames | 30 frames | **71 frames** *(Nhớ lâu nhất)* | 30 frames |\n"
        "| **MOTA (Toàn bộ 7 Sequences)** | **80.3%** | 36.80% | 38.19% | 37.62% | **38.15%** | **38.20%** | **38.65%** *(Cao nhất phân khúc)* |\n"
        "| **IDF1 (Toàn bộ 7 Sequences)** | **77.3%** | 46.54% | 47.29% | 46.94% | 45.21% | 45.07% | **46.34%** |\n"
        "| **HOTA (Toàn bộ 7 Sequences)** | **63.1%** | 39.14% | 39.54% | 39.15% | 38.33% | 37.70% | **38.95%** |\n"
        "| **Tổng số lần nhảy ID (IDSW)** | — | 823 lần | **270 lần** *(Giảm 67.2%)* | 740 lần | **223 lần** *(Giảm 72.9%!)* | 485 lần | **224 lần** *(Giảm 72.8%!)* |\n"
        "| **Tổng Track IDs sinh ra** | — | 1,372 tracks | **469 tracks** *(Giảm 65.8%)* | 1,244 tracks | **433 tracks** *(Sạch nhất)* | 693 tracks | **430 tracks** *(Sạch nhất)* |\n"
        "| **Tốc độ Tracker-only** | ~1,200 FPS | 1,041.6 FPS | **1,642.4 FPS** | 889.3 FPS | **1,503.2 FPS** | **1,588.0 FPS** | **1,235.2 FPS** |\n"
        "| **FPS toàn hệ thống (RTX 3050)** | ~29.6 FPS *(V100)* | 47.5 FPS | 48.4 FPS | 40.4 FPS | **~53 FPS** | **~53 FPS** | **~52 FPS** |\n"
        "| **Phần cứng mục tiêu** | Server GPU *(NVIDIA V100)* | Laptop GPU / Edge | **Edge AI / Drone / RPi5** | Laptop GPU / Edge | **Edge AI / Jetson Orin** | **Edge AI / Jetson Orin** | **Edge AI / Jetson Orin** |\n"
    )

    # =========================================================================
    # BẢNG 2: CHI TIẾT 7 CHUỖI MOT17 (YOLOv8n Baseline vs GA vs Fuzzy)
    # =========================================================================
    cmp_csv = Path("outputs/experiments/comparison_table.csv")
    if cmp_csv.is_file():
        df_cmp = pd.read_csv(cmp_csv)
        md_output.append("\n### 📊 Bảng 2: Kết quả Chi tiết 21 lượt chạy trên 7 Chuỗi MOT17 (YOLOv8n)\n")
        md_output.append(
            "| Sequence | Split | Mô hình Tracker | HOTA (%) | IDF1 (%) | MOTA (%) | IDSW | Pred Tracks | GT Tracks | Tracker FPS | Pipeline FPS |\n"
            "| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n"
        )
        seq_list = sorted(df_cmp["sequence"].unique())
        mode_order = {"baseline": "1. Baseline ByteTrack", "ga_bytetrack": "2. GA-ByteTrack (Evolved)", "fuzzy": "3. Fuzzy-Adaptive"}
        for seq in seq_list:
            seq_df = df_cmp[df_cmp["sequence"] == seq]
            split_name = seq_df["split"].iloc[0].replace("_", " ").title()
            for mode_key in ["baseline", "ga_bytetrack", "fuzzy"]:
                row = seq_df[seq_df["mode"] == mode_key]
                if not row.empty:
                    r = row.iloc[0]
                    mode_label = mode_order.get(mode_key, mode_key)
                    md_output.append(
                        f"| **{seq}** | {split_name} | {mode_label} | {r['HOTA']:.2f}% | {r['IDF1']:.2f}% | {r['MOTA']:.2f}% | **{int(r['IDSW'])}** | {int(r['pred_tracks'])} | {int(r['gt_tracks'])} | {r['tracker_only_fps']:.1f} | {r['end_to_end_fps']:.1f} |\n"
                    )

        md_output.append("| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for mode_key in ["baseline", "ga_bytetrack", "fuzzy"]:
            sub = df_cmp[df_cmp["mode"] == mode_key]
            lbl = mode_order[mode_key].split(". ")[1].upper()
            md_output.append(
                f"| **TRUNG BÌNH TOÀN BỘ** | **7 SEQUENCES** | **{lbl}** | **{sub['HOTA'].mean():.2f}%** | **{sub['IDF1'].mean():.2f}%** | **{sub['MOTA'].mean():.2f}%** | **{int(sub['IDSW'].sum())} (Tổng)** | **{int(sub['pred_tracks'].sum())}** | 794 | **{sub['tracker_only_fps'].mean():.1f}** | **{sub['end_to_end_fps'].mean():.1f}** |\n"
            )

    # =========================================================================
    # BẢNG 3: CHI TIẾT 7 CHUỖI MOT17 (YOLO11n Baseline vs GA Tailored vs Fuzzy)
    # =========================================================================
    yolo11_detail_csv = Path("outputs/experiments/yolo11_full_benchmark/comparison_table_yolo11n.csv")
    if yolo11_detail_csv.is_file():
        df_11 = pd.read_csv(yolo11_detail_csv)
        md_output.append("\n### 🚀 Bảng 3: Kết quả Chi tiết 21 lượt chạy trên 7 Chuỗi MOT17 (YOLO11n Thế hệ Mới)\n")
        md_output.append(
            "| Sequence | Split | Mô hình Tracker | HOTA (%) | IDF1 (%) | MOTA (%) | IDSW | Pred Tracks | GT Tracks | Tracker FPS | Pipeline FPS |\n"
            "| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n"
        )
        seq_list_11 = sorted(df_11["sequence"].unique())
        mode_order_11 = {"baseline": "1. Baseline ByteTrack", "ga_bytetrack": "2. GA-ByteTrack (Tailored)", "fuzzy": "3. Fuzzy-Adaptive"}
        for seq in seq_list_11:
            seq_df = df_11[df_11["sequence"] == seq]
            split_name = seq_df["split"].iloc[0].replace("_", " ").title()
            for mode_key in ["baseline", "ga_bytetrack", "fuzzy"]:
                row = seq_df[seq_df["mode"] == mode_key]
                if not row.empty:
                    r = row.iloc[0]
                    mode_label = mode_order_11.get(mode_key, mode_key)
                    md_output.append(
                        f"| **{seq}** | {split_name} | {mode_label} | {r['HOTA']:.2f}% | {r['IDF1']:.2f}% | {r['MOTA']:.2f}% | **{int(r['IDSW'])}** | {int(r['pred_tracks'])} | {int(r['gt_tracks'])} | {r['tracker_only_fps']:.1f} | {r['end_to_end_fps']:.1f} |\n"
                    )

        md_output.append("| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for mode_key in ["baseline", "ga_bytetrack", "fuzzy"]:
            sub = df_11[df_11["mode"] == mode_key]
            lbl = mode_order_11[mode_key].split(". ")[1].upper()
            md_output.append(
                f"| **TRUNG BÌNH TOÀN BỘ** | **7 SEQUENCES** | **{lbl}** | **{sub['HOTA'].mean():.2f}%** | **{sub['IDF1'].mean():.2f}%** | **{sub['MOTA'].mean():.2f}%** | **{int(sub['IDSW'].sum())} (Tổng)** | **{int(sub['pred_tracks'].sum())}** | 796 | **{sub['tracker_only_fps'].mean():.1f}** | **{sub['end_to_end_fps'].mean():.1f}** |\n"
            )

    # =========================================================================
    # BẢNG 4: ĐỐI ĐẦU TRỰC DIỆN MACRO-AVERAGE TOÀN DATASET (YOLOv8n vs YOLO11n)
    # =========================================================================
    md_output.append("\n### ⚡ Bảng 4: Đối đầu Trực diện Macro-Average Toàn Bộ Dataset (YOLOv8n vs YOLO11n trên 7 Chuỗi)\n")
    md_output.append(
        "| Detector | Params | GFLOPs | Mô hình Tracker | Mean MOTA (%) | Mean IDF1 (%) | Mean HOTA (%) | Tổng IDSW | Tổng Pred Tracks | Tracker FPS | Pipeline FPS (RTX 3050) |\n"
        "| :--- | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n"
        "| **YOLOv8n** | 3.2M | 8.7 | Baseline ByteTrack | 36.80% | 46.54% | 39.14% | 823 | 1,372 | 1,041.6 | 47.5 FPS |\n"
        "| **YOLOv8n** | 3.2M | 8.7 | GA-ByteTrack (Evolved) | 38.19% | 47.29% | 39.54% | 270 | 469 | 1,642.4 | 48.4 FPS |\n"
        "| **YOLOv8n** | 3.2M | 8.7 | Fuzzy-Adaptive | 37.62% | 46.94% | 39.15% | 740 | 1,244 | 889.3 | 40.4 FPS |\n"
        "| **YOLO11n** | **2.6M** | **6.7** | Baseline ByteTrack | **38.15%** | 45.21% | 38.33% | **223** *(-72.9%)* | **433** *(-68.4%)* | **1,503.2** | **~53 FPS** |\n"
        "| **YOLO11n** | **2.6M** | **6.7** | GA-ByteTrack *(Tailored)* | **38.20%** | 45.07% | 37.70% | 485 | 693 | **1,588.0** | **~53 FPS** |\n"
        "| **YOLO11n** | **2.6M** | **6.7** | **Fuzzy-Adaptive** | **38.65%** *(Top 1)* | **46.34%** | **38.95%** | **224** *(-72.8%)* | **430** *(Sạch nhất)* | **1,235.2** | **~52 FPS** |\n"
    )

    # =========================================================================
    # BẢNG 5: ĐO ĐẠC TÀI NGUYÊN PHẦN CỨNG (RTX 3050 LAPTOP GPU)
    # =========================================================================
    gpu_csv = Path("outputs/experiments/gpu_stress/gpu_stress_benchmark.csv")
    if gpu_csv.is_file():
        df_gpu = pd.read_csv(gpu_csv)
        md_output.append("\n### 💻 Bảng 5: Đo đạc Tài nguyên Phần cứng GPU (RTX 3050 Laptop 4GB VRAM)\n")
        md_output.append(
            "| Mô hình Detector | Resolution (imgsz) | Peak VRAM Alloc | Peak VRAM Reserved | Tỉ lệ VRAM (4GB) | Detector Latency | Tracker Latency | End-to-End FPS | Trạng thái Real-time |\n"
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n"
        )
        for _, r in df_gpu.iterrows():
            rt_status = "✅ Real-time cao (>40 FPS)" if r["end_to_end_fps"] >= 40 else ("⚠️ Real-time chuẩn (~30 FPS)" if r["end_to_end_fps"] >= 25 else "❌ Dưới 20 FPS")
            md_output.append(
                f"| **{r['model']}** | {int(r['imgsz'])} | {r['peak_vram_allocated_mb']:.1f} MB | {r['peak_vram_reserved_mb']:.1f} MB | **{r['vram_utilization_pct']:.1f}%** | {r['detector_mean_ms']:.2f} ms | {r['tracker_mean_ms']:.2f} ms | **{r['end_to_end_fps']:.1f} FPS** | {rt_status} |\n"
            )

    full_markdown = "".join(md_output)
    readme_tables_file = Path("outputs/README_BENCHMARK_TABLES.md")
    readme_tables_file.write_text(full_markdown, encoding="utf-8")
    print(f"[OK] Đã xuất toàn bộ 5 bảng markdown tại: {readme_tables_file}")
    return full_markdown


if __name__ == "__main__":
    generate_tables()
