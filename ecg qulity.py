import os
import glob
import numpy as np
import pandas as pd
import neurokit2 as nk
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox


# =========================
# ECG 分析與畫圖
# =========================
def process_and_plot_ecg(
    file_path,
    ecg_col="CH1",
    fs=250,
    start_sec=20,
    duration_sec=20
):
    file_name = os.path.basename(file_path)
    print(f"\n正在處理檔案: {file_name}")

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"讀取失敗: {e}")
        return

    if ecg_col not in df.columns:
        print(f"找不到欄位: {ecg_col}，跳過此檔案。")
        return

    # 轉成數值
    ecg_signal = pd.to_numeric(
        df[ecg_col],
        errors="coerce"
    ).dropna().values

    if len(ecg_signal) < fs * 5:
        print("資料太短，不適合檢查")
        return

    # =========================
    # ECG clean
    # =========================
    ecg_clean = nk.ecg_clean(
        ecg_signal,
        sampling_rate=fs,
        method="pantompkins1985"
    )

    # =========================
    # 找 R peaks
    # =========================
    _, info = nk.ecg_peaks(
        ecg_clean,
        sampling_rate=fs
    )

    rpeaks = np.asarray(
        info["ECG_R_Peaks"],
        dtype=int
    )

    # =========================
    # 全段數據計算 HR
    # =========================
    hr_mean = np.nan
    total_peaks = len(rpeaks)

    if total_peaks >= 2:

        rr_intervals = np.diff(rpeaks) / fs

        # RR interval 保留 0.3～2.0 秒
        rr_intervals = rr_intervals[
            (rr_intervals >= 0.3) &
            (rr_intervals <= 2.0)
        ]

        if len(rr_intervals) > 0:

            hr_mean = np.mean(
                60 / rr_intervals
            )

            print(
                f"平均 HR = {hr_mean:.2f} bpm | "
                f"總 R peaks 數 = {total_peaks}"
            )

        else:
            print("RR 過濾後沒有可用資料")

    else:
        print("R peaks 太少，無法算 HR")

    # =========================
    # 畫圖區段
    # =========================
    start_idx = int(start_sec * fs)
    end_idx = int(
        (start_sec + duration_sec) * fs
    )

    # 開始時間超過資料長度
    if start_idx >= len(ecg_clean):
        print(
            f"設定的開始時間 {start_sec} 秒 "
            f"已超過檔案長度，跳過。"
        )
        return

    # 如果結束時間超過資料長度
    if end_idx > len(ecg_clean):
        end_idx = len(ecg_clean)

    segment = ecg_clean[
        start_idx:end_idx
    ]

    raw_segment = ecg_signal[
        start_idx:end_idx
    ]

    time_axis = np.arange(
        start_idx,
        end_idx
    ) / fs

    # 只取目前顯示範圍內的 R peaks
    seg_rpeaks = rpeaks[
        (rpeaks >= start_idx) &
        (rpeaks < end_idx)
    ]

    # =========================
    # 畫圖
    # =========================
    plt.figure(figsize=(14, 5))

    # Raw ECG
    plt.plot(
        time_axis,
        raw_segment,
        label="Raw ECG",
        color="lightgray",
        alpha=0.7
    )

    # Clean ECG
    plt.plot(
        time_axis,
        segment,
        label="Clean ECG",
        color="tab:blue"
    )

    # R peaks
    plt.scatter(
        seg_rpeaks / fs,
        ecg_clean[seg_rpeaks],
        marker="o",
        color="red",
        label="R peaks",
        zorder=5
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")

    if np.isnan(hr_mean):
        hr_text = "N/A"
    else:
        hr_text = f"{hr_mean:.2f}"

    plt.title(
        f"ECG with detected R peaks\n"
        f"{file_name} | HR: {hr_text} bpm"
    )

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # 關閉圖片後才處理下一檔
    plt.show()


# =========================
# 選擇資料夾
# =========================
def browse_folder():
    folder = filedialog.askdirectory()

    if folder:
        folder_entry.delete(0, tk.END)
        folder_entry.insert(0, folder)


# =========================
# 開始分析
# =========================
def start_analysis():

    folder_path = folder_entry.get().strip()
    ecg_col = ecg_col_entry.get().strip()

    # -------------------------
    # 檢查資料夾
    # -------------------------
    if not os.path.isdir(folder_path):
        messagebox.showerror(
            "錯誤",
            "請選擇有效的資料夾。"
        )
        return

    # -------------------------
    # 檢查 ECG 欄位
    # -------------------------
    if not ecg_col:
        messagebox.showerror(
            "錯誤",
            "請輸入 ECG 欄位名稱。"
        )
        return

    # -------------------------
    # 讀取數字設定
    # -------------------------
    try:
        fs = int(fs_entry.get())
        start_sec = float(start_entry.get())
        duration_sec = float(duration_entry.get())

        if fs <= 0:
            raise ValueError

        if start_sec < 0:
            raise ValueError

        if duration_sec <= 0:
            raise ValueError

    except ValueError:
        messagebox.showerror(
            "錯誤",
            "FS、開始秒數、顯示秒數設定有誤。"
        )
        return

    # =========================
    # 找 CSV
    # =========================
    csv_files = glob.glob(
        os.path.join(
            folder_path,
            "*.csv"
        )
    )

    if not csv_files:
        messagebox.showwarning(
            "找不到檔案",
            f"資料夾內找不到 CSV：\n{folder_path}"
        )
        return

    # 排序
    csv_files.sort()

    print("\n=========================")
    print("開始 ECG 分析")
    print("=========================")
    print(f"資料夾: {folder_path}")
    print(f"ECG 欄位: {ecg_col}")
    print(f"FS: {fs} Hz")
    print(f"開始時間: {start_sec} 秒")
    print(f"顯示長度: {duration_sec} 秒")
    print(f"CSV 數量: {len(csv_files)}")
    print("=========================")

    # =========================
    # 批次分析
    # =========================
    for file_path in csv_files:

        process_and_plot_ecg(
            file_path,
            ecg_col=ecg_col,
            fs=fs,
            start_sec=start_sec,
            duration_sec=duration_sec
        )

    messagebox.showinfo(
        "完成",
        f"全部處理完成！\n共 {len(csv_files)} 個 CSV 檔案。"
    )


# =========================
# GUI
# =========================
root = tk.Tk()
root.title("ECG R-Peak 檢查工具")
root.geometry("620x330")
root.resizable(False, False)


# -------------------------
# 資料夾
# -------------------------
tk.Label(
    root,
    text="CSV 資料夾："
).grid(
    row=0,
    column=0,
    padx=10,
    pady=15,
    sticky="e"
)

folder_entry = tk.Entry(
    root,
    width=50
)

folder_entry.grid(
    row=0,
    column=1,
    padx=5,
    pady=15
)

# 預設值
folder_entry.insert(
    0,
    r"D:\115-2 data\LB0427-30"
)

tk.Button(
    root,
    text="選擇資料夾",
    command=browse_folder
).grid(
    row=0,
    column=2,
    padx=5
)


# -------------------------
# ECG 欄位
# -------------------------
tk.Label(
    root,
    text="ECG 欄位："
).grid(
    row=1,
    column=0,
    padx=10,
    pady=8,
    sticky="e"
)

ecg_col_entry = tk.Entry(
    root,
    width=20
)

ecg_col_entry.grid(
    row=1,
    column=1,
    sticky="w",
    padx=5
)

ecg_col_entry.insert(
    0,
    "CH1"
)


# -------------------------
# FS
# -------------------------
tk.Label(
    root,
    text="取樣率 FS (Hz)："
).grid(
    row=2,
    column=0,
    padx=10,
    pady=8,
    sticky="e"
)

fs_entry = tk.Entry(
    root,
    width=20
)

fs_entry.grid(
    row=2,
    column=1,
    sticky="w",
    padx=5
)

fs_entry.insert(
    0,
    "250"
)


# -------------------------
# 開始秒數
# -------------------------
tk.Label(
    root,
    text="從第幾秒開始："
).grid(
    row=3,
    column=0,
    padx=10,
    pady=8,
    sticky="e"
)

start_entry = tk.Entry(
    root,
    width=20
)

start_entry.grid(
    row=3,
    column=1,
    sticky="w",
    padx=5
)

start_entry.insert(
    0,
    "20"
)


# -------------------------
# 顯示秒數
# -------------------------
tk.Label(
    root,
    text="顯示幾秒："
).grid(
    row=4,
    column=0,
    padx=10,
    pady=8,
    sticky="e"
)

duration_entry = tk.Entry(
    root,
    width=20
)

duration_entry.grid(
    row=4,
    column=1,
    sticky="w",
    padx=5
)

duration_entry.insert(
    0,
    "20"
)


# =========================
# 開始按鈕
# =========================
start_button = tk.Button(
    root,
    text="開始分析",
    command=start_analysis,
    width=20,
    height=2
)

start_button.grid(
    row=5,
    column=0,
    columnspan=3,
    pady=25
)


root.mainloop()
