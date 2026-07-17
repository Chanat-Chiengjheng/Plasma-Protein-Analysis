import math
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

VERSION = "2.0"
BUILD_TAG = "wave-compare-v2-dual-mode"

V_TO_KV = 1e-3
A_TO_MA = 1e3
C_TO_NC = 1e9

MAX_RECOMMENDED_WAVES = 8
SMOOTHING_WINDOW_PERIOD_FRACTION = 1 / 20
MINIMUM_GAP_PERIOD_FRACTION = 0.5
FIXED_INTERVAL_MARK_SPACING_SECONDS = 0.001


def to_kilovolts(value):
    return value * V_TO_KV


def to_milliamps(value):
    return value * A_TO_MA


def to_nanocoulombs(value):
    return value * C_TO_NC


def _select_csv_paths():
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    paths = filedialog.askopenfilenames(
        title="Select one or more oscilloscope CSV files",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )
    root.destroy()

    if not paths:
        raise RuntimeError("No files were selected.")
    return paths


def _parse_files(paths):
    files = []
    for path in paths:
        try:
            with open(path, "r") as f:
                header_line = f.readline()
            t0, t_inc = _parse_header(header_line)

            df = pd.read_csv(
                path, skiprows=1, header=None,
                names=["CH1V", "CH2A", "_col3", "_col4"],
                usecols=["CH1V", "CH2A"],
            )
            n_samples = len(df)
            if n_samples == 0:
                raise RuntimeError(f"File '{path}' has no data rows.")

            start_time = t0
            end_time = t0 + t_inc * (n_samples - 1)
            df["time"] = t0 + t_inc * np.arange(n_samples)

            files.append({
                "path": path, "df": df, "t0": t0, "tInc": t_inc,
                "n": n_samples, "start": start_time, "end": end_time,
            })
        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(f"Could not load '{path}': {type(e).__name__}: {e}")

    files.sort(key=lambda f: f["start"])
    return files


def load_scope_files():
    paths = _select_csv_paths()
    files = _parse_files(paths)

    overlaps = _find_overlapping_files(files)
    if overlaps:
        return _handle_overlapping_files(files, overlaps)

    gaps = []
    for prev_file, next_file in zip(files, files[1:]):
        tolerance = 1.5 * abs(prev_file["tInc"])
        if next_file["start"] - prev_file["end"] > tolerance:
            gaps.append((prev_file["end"], next_file["start"]))

    combined = pd.concat([f["df"][["time", "CH1V", "CH2A"]] for f in files], ignore_index=True)
    combined.sort_values("time", inplace=True)
    combined.reset_index(drop=True, inplace=True)

    valid_min = files[0]["start"]
    valid_max = files[-1]["end"]

    print(f"Loaded {len(files)} file(s) as sequential segments. Valid time range: "
          f"{valid_min:.3f}s to {valid_max:.3f}s.")
    for gap_start, gap_end in gaps:
        print(f"Note: gap detected between {gap_start:.3f}s and {gap_end:.3f}s")

    return [{"label": None, "data": combined, "valid_min": valid_min, "valid_max": valid_max, "gaps": gaps}]


def _find_overlapping_files(files):
    overlaps = []
    for prev_file, next_file in zip(files, files[1:]):
        if next_file["start"] < prev_file["end"]:
            overlaps.append((prev_file, next_file))
    return overlaps


def load_scope_files_per_file():
    paths = _select_csv_paths()
    files = _parse_files(paths)
    overlaps = _find_overlapping_files(files)

    datasets = []
    for f in files:
        label = os.path.basename(f["path"])
        df = f["df"][["time", "CH1V", "CH2A"]].copy()
        datasets.append({
            "label": label, "data": df, "path": f["path"],
            "valid_min": f["start"], "valid_max": f["end"], "gaps": [],
        })

    print(f"Loaded {len(datasets)} file(s) as independent snaps.")
    return datasets, overlaps


def _warn_about_possible_duplicate_snaps(overlaps):
    if not overlaps:
        return
    print("\nWARNING: some selected files have OVERLAPPING time ranges - they may be")
    print("duplicate/overlapping captures rather than genuinely different snaps:")
    for prev_file, next_file in overlaps:
        print(f"  '{os.path.basename(prev_file['path'])}' covers "
              f"{prev_file['start']:.6f}s to {prev_file['end']:.6f}s")
        print(f"  '{os.path.basename(next_file['path'])}' covers "
              f"{next_file['start']:.6f}s to {next_file['end']:.6f}s")
    print("  Proceeding anyway, treating each file as its own snap.\n")


def _handle_overlapping_files(files, overlaps):
    print("\nThe selected files have OVERLAPPING time ranges - they look like separate")
    print("captures (e.g. separate scope triggers), not sequential segments of one experiment:")
    for prev_file, next_file in overlaps:
        print(f"  '{os.path.basename(prev_file['path'])}' covers "
              f"{prev_file['start']:.6f}s to {prev_file['end']:.6f}s")
        print(f"  '{os.path.basename(next_file['path'])}' covers "
              f"{next_file['start']:.6f}s to {next_file['end']:.6f}s")
        print(f"    -> these overlap, so they can't be joined end-to-end.\n")

    while True:
        answer = input(
            "Analyze each file separately instead? "
            "(y = analyze separately, n = cancel and re-select files): "
        ).strip().lower()
        if answer in ("y", "n"):
            break
        print("  Please enter 'y' or 'n'.")

    if answer == "n":
        raise RuntimeError("Cancelled - please re-select files with non-overlapping time ranges.")

    datasets = []
    for f in files:
        label = os.path.basename(f["path"])
        df = f["df"][["time", "CH1V", "CH2A"]].copy()
        datasets.append({
            "label": label,
            "data": df,
            "path": f["path"],
            "valid_min": f["start"],
            "valid_max": f["end"],
            "gaps": [],
        })
        print(f"  Loaded '{label}' as an independent dataset: "
              f"{f['start']:.3f}s to {f['end']:.3f}s")

    return datasets


def _parse_header(header_line):
    import re
    match = re.search(r"t0\s*=\s*([-\d.eE+]+)\s*,\s*tInc\s*=\s*([-\d.eE+]+)", header_line)
    if not match:
        raise RuntimeError(
            f"Header line doesn't match the expected format "
            f"'CH1V,CH2A,t0 =<value>, tInc = <value>,':\n  '{header_line.strip()}'"
        )
    return float(match.group(1)), float(match.group(2))


def get_valid_time_range(loaded_data):
    return loaded_data["valid_min"], loaded_data["valid_max"], loaded_data["gaps"]


def _time_is_valid(t, valid_min, valid_max, gaps):
    if t < valid_min or t > valid_max:
        return False
    for gap_start, gap_end in gaps:
        if gap_start <= t <= gap_end:
            return False
    return True


def select_comparison_waves_gui(valid_min, valid_max, gaps):
    import tkinter as tk

    result = {}
    time_vars = []

    root = tk.Tk()
    root.title("Wave Compare - Select Target Times")
    root.geometry("480x520")

    info_text = f"Valid time range: {valid_min:.3f}s to {valid_max:.3f}s"
    for gap_start, gap_end in gaps:
        info_text += f"\nGap detected: {gap_start:.3f}s to {gap_end:.3f}s"
    tk.Label(root, text=info_text, justify='left').pack(anchor='w', padx=16, pady=(12, 6))

    count_frame = tk.Frame(root)
    count_frame.pack(fill='x', padx=16)
    count_var = tk.StringVar()
    tk.Label(count_frame, text="Number of waves to compare:").pack(side='left')
    tk.Entry(count_frame, textvariable=count_var, width=6).pack(side='left', padx=(6, 6))
    tk.Button(count_frame, text="Set", command=lambda: build_entries()).pack(side='left')

    canvas_frame = tk.Frame(root)
    canvas_frame.pack(fill='both', expand=True, padx=16, pady=(10, 4))
    canvas = tk.Canvas(canvas_frame, borderwidth=0, highlightthickness=0)
    scrollbar = tk.Scrollbar(canvas_frame, orient='vertical', command=canvas.yview)
    entries_frame = tk.Frame(canvas)
    entries_frame.bind(
        "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )
    canvas.create_window((0, 0), window=entries_frame, anchor='nw')
    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.pack(side='left', fill='both', expand=True)
    scrollbar.pack(side='right', fill='y')

    error_var = tk.StringVar()
    tk.Label(root, textvariable=error_var, fg='red', wraplength=440, justify='left').pack(
        anchor='w', padx=16, pady=(2, 2))

    def build_entries():
        time_vars.clear()
        for widget in entries_frame.winfo_children():
            widget.destroy()
        try:
            count = int(count_var.get().strip())
            if count <= 0:
                raise ValueError
        except ValueError:
            error_var.set("Enter a positive whole number of waves, then click Set.")
            return

        error_var.set(
            f"Warning: {count} waves is more than the recommended max of "
            f"{MAX_RECOMMENDED_WAVES} - the overlay plot may get cluttered."
            if count > MAX_RECOMMENDED_WAVES else ""
        )

        for i in range(count):
            var = tk.StringVar()
            tk.Label(entries_frame, text=f"Target time #{i + 1} (s):").grid(
                row=i, column=0, sticky='w', pady=3)
            tk.Entry(entries_frame, textvariable=var, width=16).grid(
                row=i, column=1, sticky='ew', pady=3, padx=(6, 0))
            time_vars.append(var)

    def on_run():
        if not time_vars:
            error_var.set("Set the number of waves and enter target times first.")
            return
        times = []
        for i, var in enumerate(time_vars):
            text = var.get().strip()
            try:
                t = float(text)
            except ValueError:
                error_var.set(f"Target time #{i + 1} is not a valid number.")
                return
            if not _time_is_valid(t, valid_min, valid_max, gaps):
                error_var.set(f"Target time #{i + 1} ({t}s) is out of range or inside a gap.")
                return
            times.append(t)
        result['times'] = times
        root.destroy()

    tk.Button(root, text="Run Comparison", command=on_run).pack(pady=10)

    root.mainloop()

    if 'times' not in result:
        print("\nSelection window closed without running - exiting.")
        sys.exit(0)

    return result['times']


def select_comparison_waves_terminal(valid_min, valid_max, gaps):
    print("Popup window unavailable - using terminal input instead.\n")

    while True:
        count_input = input("How many waves do you want to compare? ").strip()
        try:
            count = int(count_input)
            if count > 0:
                break
        except ValueError:
            pass
        print("  Please enter a positive whole number.")

    if count > MAX_RECOMMENDED_WAVES:
        print(f"  Warning: {count} waves is more than the recommended max of "
              f"{MAX_RECOMMENDED_WAVES} - the overlay plot may get cluttered.")

    target_times = []
    for i in range(count):
        while True:
            t_input = input(
                f"Target time #{i + 1} in seconds (valid range "
                f"{valid_min:.3f}s to {valid_max:.3f}s): "
            ).strip()
            try:
                t = float(t_input)
            except ValueError:
                print("  Please enter a number.")
                continue
            if not _time_is_valid(t, valid_min, valid_max, gaps):
                print(f"  {t}s is out of range or inside a known gap - please re-enter.")
                continue
            target_times.append(t)
            break

    return target_times


def select_comparison_waves(valid_min, valid_max, gaps):
    try:
        return select_comparison_waves_gui(valid_min, valid_max, gaps)
    except Exception as e:
        print(f"Popup window unavailable ({type(e).__name__}: {e})")
        return select_comparison_waves_terminal(valid_min, valid_max, gaps)


MIN_PLAUSIBLE_PERIOD_SAMPLES = 5


def estimate_expected_period(combined_df):
    time = combined_df["time"].to_numpy()
    voltage = combined_df["CH1V"].to_numpy()
    sample_spacing = float(np.median(np.diff(time)))

    demeaned = voltage - np.mean(voltage)
    magnitude = np.abs(np.fft.rfft(demeaned))
    freqs = np.fft.rfftfreq(len(demeaned), d=sample_spacing)

    peak_freq = 0.0
    if len(magnitude) > 1:
        magnitude_no_dc = magnitude.copy()
        magnitude_no_dc[0] = -np.inf
        peak_idx = int(np.argmax(magnitude_no_dc))
        peak_freq = freqs[peak_idx]

    if peak_freq <= 0:
        print("Auto-estimated period looks implausibly small, likely noise-dominated signal "
              "-- falling back to manual entry")
        return _prompt_manual_period()

    expected_period = 1.0 / peak_freq
    if expected_period < MIN_PLAUSIBLE_PERIOD_SAMPLES * sample_spacing:
        print("Auto-estimated period looks implausibly small, likely noise-dominated signal "
              "-- falling back to manual entry")
        expected_period = _prompt_manual_period()

    return expected_period


def _prompt_manual_period():
    while True:
        text = input("Enter an approximate period in seconds (e.g. 9e-6 for 9 microseconds): ").strip()
        try:
            value = float(text)
            if value > 0:
                return value
        except ValueError:
            pass
        print("  Please enter a positive number.")


def _compute_smoothing_window(expected_period, sample_spacing):
    safe_sample_spacing = sample_spacing if sample_spacing > 0 else 1e-12
    period_samples = max(expected_period, 0) / safe_sample_spacing
    window = max(3, int(round(period_samples * SMOOTHING_WINDOW_PERIOD_FRACTION)))
    if window % 2 == 0:
        window += 1
    return window


def _format_duration(seconds):
    abs_s = abs(seconds)
    if abs_s < 1e-6:
        return f"{seconds * 1e9:.1f}ns"
    if abs_s < 1e-3:
        return f"{seconds * 1e6:.2f}us"
    if abs_s < 1:
        return f"{seconds * 1e3:.2f}ms"
    return f"{seconds:.3f}s"


def extract_one_cycle(combined_df, target_time, expected_period=None, minimum_gap_seconds=None):
    if expected_period is None:
        expected_period = estimate_expected_period(combined_df)

    time = combined_df["time"].to_numpy()
    raw_voltage = combined_df["CH1V"].to_numpy()
    raw_current = combined_df["CH2A"].to_numpy()
    sample_spacing = float(np.median(np.diff(time)))

    if minimum_gap_seconds is None:
        minimum_gap_seconds = expected_period * MINIMUM_GAP_PERIOD_FRACTION

    smoothing_window = _compute_smoothing_window(expected_period, sample_spacing)
    smoothed = combined_df["CH1V"].rolling(
        window=smoothing_window, center=True, min_periods=1
    ).mean().to_numpy()
    smoothed = smoothed - np.mean(smoothed)

    sign = np.sign(smoothed)
    sign[sign == 0] = 1
    candidates = np.where((sign[:-1] < 0) & (sign[1:] > 0))[0] + 1

    if len(candidates) == 0:
        return None

    accepted = []
    last_accepted_time = None
    for idx in candidates:
        t = time[idx]
        if last_accepted_time is None or (t - last_accepted_time) >= minimum_gap_seconds:
            accepted.append(idx)
            last_accepted_time = t
    rising = np.array(accepted)

    if len(rising) == 0:
        return None

    crossing_times = time[rising]
    start_candidates = rising[crossing_times >= target_time]
    if len(start_candidates) == 0:
        return None
    cycle_start_idx = start_candidates[0]

    end_candidates = rising[rising > cycle_start_idx]
    if len(end_candidates) == 0:
        return None
    cycle_end_idx = end_candidates[0]

    segment_time = time[cycle_start_idx:cycle_end_idx + 1]
    segment_voltage = raw_voltage[cycle_start_idx:cycle_end_idx + 1]
    segment_current = raw_current[cycle_start_idx:cycle_end_idx + 1]
    period = segment_time[-1] - segment_time[0]
    if period <= 0:
        return None

    return {
        "time": segment_time - segment_time[0],
        "voltage": segment_voltage,
        "current": segment_current,
        "frequency": 1.0 / period,
        "period": period,
        "target_time": target_time,
    }


def run_wave_comparison():
    try:
        datasets = load_scope_files()
    except RuntimeError as e:
        print(f"ERROR loading files: {e}")
        return

    for dataset in datasets:
        label = dataset["label"]
        if label:
            print(f"\n=== Dataset: {label} ===")

        valid_min, valid_max, gaps = get_valid_time_range(dataset)
        target_times = select_comparison_waves(valid_min, valid_max, gaps)

        expected_period = estimate_expected_period(dataset["data"])
        print(f"Auto-estimated expected period: {_format_duration(expected_period)} "
              f"(used to size smoothing/debounce for cycle detection)")

        waves = []
        for target_time in target_times:
            try:
                wave = extract_one_cycle(dataset["data"], target_time, expected_period=expected_period)
            except Exception as e:
                print(f"WARNING: failed to extract a cycle near {target_time:.3f}s "
                      f"({type(e).__name__}: {e}) - skipping.")
                continue
            if wave is None:
                print(f"WARNING: no valid cycle could be found near {target_time:.3f}s - skipping.")
                continue
            waves.append(wave)

        if not waves:
            print("No waves were successfully extracted - nothing to plot.")
            continue

        fig, ax = plt.subplots(figsize=(10, 6))
        for wave in waves:
            ax.plot(
                wave["time"], to_kilovolts(wave["voltage"]),
                label=(f"t={wave['target_time']:.3f}s (f={wave['frequency']:.1f}Hz, "
                        f"period={_format_duration(wave['period'])})")
            )
        ax.set_xlabel("Time since cycle start (s)")
        ax.set_ylabel("Voltage (kV)")
        title = f"Wave Compare v{VERSION} [{BUILD_TAG}] - {len(waves)} cycle(s)"
        if label:
            title += f" - {label}"
        ax.set_title(title)
        ax.legend()
        fig.tight_layout()

        plt.show()


def select_analysis_mode_gui(datasets):
    import tkinter as tk

    result = {}
    selected_n_vars = {}
    analysis_cache = {}
    task_g_built = {'done': False}

    root = tk.Tk()
    root.title("Wave Compare - Select Analysis Mode")
    root.geometry("620x560")
    root.resizable(True, True)

    tk.Label(root, text="Choose an analysis mode:", pady=10).pack()

    mode_var = tk.StringVar(value="A")
    options = [
        ("Task A: Compare cycles at fixed 1ms intervals within snap", "A"),
        ("Task B: Compare effective (RMS) V/I across snaps", "B"),
        ("Task C: Per-cycle charge calculation with statistics", "C"),
        ("Task D: Charge histogram (frequency distribution)", "D"),
        ("Task G: Classify cycle behavior by peak pattern", "G"),
        ("All", "ALL"),
    ]

    task_g_frame = tk.Frame(root)

    for text, value in options:
        tk.Radiobutton(root, text=text, variable=mode_var, value=value).pack(anchor='w', padx=24, pady=2)
        if value == "G":
            task_g_frame.pack(fill='both', expand=True, padx=40, pady=(0, 6))
            task_g_frame.pack_forget()

    def build_task_g_panel():
        for widget in task_g_frame.winfo_children():
            widget.destroy()

        analyzing_label = tk.Label(task_g_frame, text="Analyzing...", fg='blue')
        analyzing_label.pack(anchor='w', pady=4)
        root.update()

        selected_n_vars.clear()
        analysis_cache.clear()
        for dataset in datasets:
            label = dataset["label"]
            try:
                analysis = _compute_peak_types_for_dataset(dataset)
            except Exception as e:
                analysis = None
                print(f"WARNING [{label}]: Task G pre-analysis failed ({type(e).__name__}: {e})")
            analysis_cache[label] = analysis

        analyzing_label.destroy()

        for dataset in datasets:
            label = dataset["label"]
            analysis = analysis_cache.get(label)
            row_frame = tk.Frame(task_g_frame)

            if len(datasets) > 1:
                tk.Label(row_frame, text=label, font=('TkDefaultFont', 9, 'bold')).pack(anchor='w')

            if not analysis or not analysis["type_list"]:
                tk.Label(row_frame, text="  Not enough cycles detected for Type analysis.",
                         fg='red').pack(anchor='w')
                row_frame.pack(anchor='w', fill='x', pady=(4, 8))
                continue

            type_list = analysis["type_list"]
            n_cycles = analysis["n_cycles"]
            cum_pct = _compute_cumulative_coverage(type_list, n_cycles)
            total_types = len(type_list)
            candidates = _generate_odd_candidate_ns(total_types)
            default_n = next((n for n in candidates if cum_pct[n - 1] >= 90.0), candidates[-1])

            n_var = tk.IntVar(value=default_n)
            selected_n_vars[label] = n_var

            btn_row = tk.Frame(row_frame)
            btn_row.pack(anchor='w')
            for n in candidates:
                pct = cum_pct[n - 1]
                tk.Radiobutton(btn_row, text=f"N={n} ({pct:.1f}%)", variable=n_var, value=n).pack(
                    side='left', padx=4)

            row_frame.pack(anchor='w', fill='x', pady=(4, 8))

        task_g_built['done'] = True

    def on_mode_change(*_args):
        if mode_var.get() == "G":
            task_g_frame.pack(fill='both', expand=True, padx=40, pady=(0, 6))
            if not task_g_built['done']:
                build_task_g_panel()
        else:
            task_g_frame.pack_forget()

    mode_var.trace_add('write', on_mode_change)

    def on_run():
        result['mode'] = mode_var.get()
        result['type_g_n'] = {label: var.get() for label, var in selected_n_vars.items()}
        result['type_g_cache'] = dict(analysis_cache)
        root.destroy()

    tk.Button(root, text="Run", command=on_run).pack(pady=15)

    root.mainloop()

    if 'mode' not in result:
        print("\nMode selection window closed without choosing - exiting.")
        sys.exit(0)

    return result['mode'], result['type_g_n'], result['type_g_cache']


def select_analysis_mode_terminal():
    print("Popup window unavailable - using terminal input instead.\n")
    print("Choose an analysis mode:")
    print("  a) Task A: Compare cycles at fixed 1ms intervals within snap")
    print("  b) Task B: Compare effective (RMS) V/I across snaps")
    print("  c) Task C: Per-cycle charge calculation with statistics")
    print("  d) Task D: Charge histogram (frequency distribution)")
    print("  e) Task G: Classify cycle behavior by peak pattern")
    print("  f) All")
    while True:
        choice = input("Enter a/b/c/d/e/f: ").strip().lower()
        if choice in ("a", "b", "c", "d", "e", "f"):
            return {"a": "A", "b": "B", "c": "C", "d": "D", "e": "G", "f": "ALL"}[choice]
        print("  Please enter 'a', 'b', 'c', 'd', 'e', or 'f'.")


def select_analysis_mode(datasets):
    try:
        return select_analysis_mode_gui(datasets)
    except Exception as e:
        print(f"Popup window unavailable ({type(e).__name__}: {e})")
        return select_analysis_mode_terminal(), {}, {}


def _generate_fixed_interval_marks(valid_min, valid_max):
    marks = []
    t = math.ceil(valid_min * 1000) / 1000
    while t <= valid_max:
        marks.append(t)
        t += FIXED_INTERVAL_MARK_SPACING_SECONDS
    return marks


def run_fixed_interval_cycle_comparison(dataset):
    label = dataset["label"] or "combined dataset"
    valid_min, valid_max, gaps = get_valid_time_range(dataset)

    expected_period = estimate_expected_period(dataset["data"])
    print(f"[{label}] Auto-estimated expected period: {_format_duration(expected_period)}")

    marks = _generate_fixed_interval_marks(valid_min, valid_max)

    waves = []
    for mark in marks:
        mark_ms = mark * 1000
        if not _time_is_valid(mark, valid_min, valid_max, gaps):
            print(f"WARNING [{label}]: {mark_ms:.0f}ms mark falls in a gap or out of range - skipping.")
            continue
        try:
            wave = extract_one_cycle(dataset["data"], mark, expected_period=expected_period)
        except Exception as e:
            print(f"WARNING [{label}]: failed to extract cycle at {mark_ms:.0f}ms mark "
                  f"({type(e).__name__}: {e}) - skipping.")
            continue
        if wave is None:
            print(f"WARNING [{label}]: no valid cycle found at {mark_ms:.0f}ms mark - skipping.")
            continue
        wave["mark_ms"] = mark_ms
        waves.append(wave)

    if not waves:
        print(f"[{label}] No cycles could be extracted - skipping this file.")
        return

    fig, (ax_v, ax_i) = plt.subplots(1, 2, figsize=(16, 6))
    for wave in waves:
        ax_v.plot(wave["time"], to_kilovolts(wave["voltage"]), label=f"t={wave['mark_ms']:.0f}ms")
    ax_v.set_xlabel("Time since cycle start (s)")
    ax_v.set_ylabel("Voltage (kV)")
    ax_v.set_title(f"{label} - Voltage - cycles at 1ms intervals")
    ax_v.legend()

    for wave in waves:
        ax_i.plot(wave["time"], to_milliamps(wave["current"]), label=f"t={wave['mark_ms']:.0f}ms")
    ax_i.set_xlabel("Time since cycle start (s)")
    ax_i.set_ylabel("Current (mA)")
    ax_i.set_title(f"{label} - Current - cycles at 1ms intervals")
    ax_i.legend()

    fig.tight_layout()
    plt.show(block=False)


def compute_rms_per_file(dataset):
    voltage = dataset["data"]["CH1V"].to_numpy()
    current = dataset["data"]["CH2A"].to_numpy()
    vrms = float(np.sqrt(np.mean(voltage ** 2)))
    irms = float(np.sqrt(np.mean(current ** 2)))
    return vrms, irms


CYCLE_START_GAP_STEP_FRACTION = 0.75


def detect_all_cycle_starts(v, dt, expected_period):
    """Port of the VBA DetectZeroCrossingsNearIdeal function.

    Finds every negative-to-positive zero crossing in v, interpolated to
    sub-sample precision, and returns the crossing times (seconds, relative
    to v[0] at t=0). Crossings less than 75% of the expected period apart
    are treated as noise and dropped, matching the VBA threshold.
    """
    step_size = expected_period / dt

    signs = np.sign(v)
    signs[signs == 0] = 1
    candidates = np.where((signs[:-1] < 0) & (signs[1:] > 0))[0]

    # TEMP DEBUG (Issue 3, cycle-count investigation) -- remove once resolved.
    print(f"    [DEBUG detect_all_cycle_starts] raw rising crossings BEFORE debounce: {len(candidates)}")
    print(f"    [DEBUG detect_all_cycle_starts] expected_period={expected_period!r}, "
          f"dt={dt!r}, step_size={step_size!r} samples, "
          f"debounce_threshold={step_size * CYCLE_START_GAP_STEP_FRACTION!r} samples")

    crossing_positions = []
    last_position = None
    for i in candidates:
        frac = v[i] / (v[i] - v[i + 1])
        position = i + frac
        if last_position is None or (position - last_position) >= step_size * CYCLE_START_GAP_STEP_FRACTION:
            crossing_positions.append(position)
            last_position = position

    # TEMP DEBUG (Issue 3, cycle-count investigation) -- remove once resolved.
    print(f"    [DEBUG detect_all_cycle_starts] crossings AFTER debounce: {len(crossing_positions)}")

    return np.array(crossing_positions) * dt


def calculate_cycle_charges(i_data, dt, cycle_starts):
    """Port of CalculateCycleChargesImproved.

    For each pair of consecutive cycle boundaries, sums i_data * dt
    separately over samples with current >= 0 and current < 0.
    """
    positions = cycle_starts / dt

    pos_charges = []
    neg_charges = []
    total_charges = []
    for k in range(len(positions) - 1):
        i_start = int(round(positions[k]))
        i_end = int(round(positions[k + 1]))
        segment = i_data[i_start:i_end]

        pos_charge = float(np.sum(segment[segment >= 0]) * dt)
        neg_charge = float(np.sum(segment[segment < 0]) * dt)

        pos_charges.append(pos_charge)
        neg_charges.append(neg_charge)
        total_charges.append(pos_charge + abs(neg_charge))

    return np.array(pos_charges), np.array(neg_charges), np.array(total_charges)


def compute_charge_statistics(pos_charges, neg_charges, total_charges):
    # ddof=1 -> sample standard deviation, matching Excel's WorksheetFunction.StDev
    return {
        "pos_mean": float(np.mean(pos_charges)),
        "pos_std": float(np.std(pos_charges, ddof=1)),
        "pos_median": float(np.median(pos_charges)),
        "pos_max": float(np.max(pos_charges)),
        "neg_mean": float(np.mean(neg_charges)),
        "neg_std": float(np.std(neg_charges, ddof=1)),
        "neg_median": float(np.median(neg_charges)),
        "neg_max": float(np.max(neg_charges)),
        "total_mean": float(np.mean(total_charges)),
        "total_std": float(np.std(total_charges, ddof=1)),
        "total_median": float(np.median(total_charges)),
        "total_max": float(np.max(total_charges)),
    }


def _detect_dataset_cycle_charges(dataset, task_label):
    label = dataset["label"] or "combined dataset"
    time = dataset["data"]["time"].to_numpy()
    voltage = dataset["data"]["CH1V"].to_numpy()
    current = dataset["data"]["CH2A"].to_numpy()
    dt = float(np.median(np.diff(time)))

    expected_period = estimate_expected_period(dataset["data"])
    print(f"\n[{label}] Auto-estimated expected period: {_format_duration(expected_period)}")

    cycle_starts = detect_all_cycle_starts(voltage, dt, expected_period)
    if len(cycle_starts) < 2:
        print(f"[{label}] Fewer than 2 cycle boundaries were detected - skipping {task_label}.")
        return None

    pos_charges, neg_charges, total_charges = calculate_cycle_charges(current, dt, cycle_starts)
    n_cycles = len(total_charges)
    if n_cycles == 0:
        print(f"[{label}] No complete cycles were found - skipping {task_label}.")
        return None

    return {
        "label": label, "n_cycles": n_cycles,
        "pos_charges": pos_charges, "neg_charges": neg_charges, "total_charges": total_charges,
    }


def _print_charge_stats_table(label, n_cycles, stats):
    print(f"[{label}] Total cycles found: {n_cycles}")
    print(f"  {'':<12}{'Mean (nC)':>14}{'StDev (nC)':>14}{'Median (nC)':>14}{'Max (nC)':>14}")
    for name, key in (("Positive", "pos"), ("Negative", "neg"), ("Total", "total")):
        values = "".join(
            f"{to_nanocoulombs(stats[f'{key}_{stat}']):>14.4f}"
            for stat in ("mean", "std", "median", "max")
        )
        print(f"  {name:<12}{values}")


def _draw_charge_stats_table(ax, stats, n_cycles):
    ax.axis("off")
    ax.set_title(f"Total cycles found: {n_cycles}", loc='left', fontsize=10)
    col_labels = ["", "Mean (nC)", "StDev (nC)", "Median (nC)", "Max (nC)"]
    row_names = ["Positive", "Negative", "Total"]
    row_keys = ["pos", "neg", "total"]
    cell_text = [
        [name] + [f"{to_nanocoulombs(stats[f'{key}_{stat}']):.4f}" for stat in ("mean", "std", "median", "max")]
        for name, key in zip(row_names, row_keys)
    ]
    table = ax.table(cellText=cell_text, colLabels=col_labels, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)


def _draw_single_charge_stats_table(ax, stats, key, n_cycles):
    ax.axis("off")
    ax.set_title(f"n = {n_cycles} cycles", loc='center', fontsize=9)
    col_labels = ["Mean (nC)", "StDev (nC)", "Median (nC)", "Max (nC)"]
    cell_text = [[f"{to_nanocoulombs(stats[f'{key}_{stat}']):.4f}" for stat in ("mean", "std", "median", "max")]]
    table = ax.table(cellText=cell_text, colLabels=col_labels, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)


def run_charge_per_cycle_analysis(dataset):
    result = _detect_dataset_cycle_charges(dataset, "charge analysis")
    if result is None:
        return
    label, n_cycles = result["label"], result["n_cycles"]
    pos_charges, neg_charges, total_charges = result["pos_charges"], result["neg_charges"], result["total_charges"]

    stats = compute_charge_statistics(pos_charges, neg_charges, total_charges)

    print(f"[{label}] Task C: charge-per-cycle analysis")
    _print_charge_stats_table(label, n_cycles, stats)

    cycle_indices = np.arange(1, n_cycles + 1)

    fig, (ax_plot, ax_table) = plt.subplots(
        2, 1, figsize=(10, 8), gridspec_kw={"height_ratios": [3, 1]}
    )

    ax_plot.plot(cycle_indices, to_nanocoulombs(total_charges), marker='o', markersize=3, linestyle='-')
    ax_plot.set_xlabel("Cycle number")
    ax_plot.set_ylabel("Total charge (nC)")
    ax_plot.set_title(f"{label} - Charge per cycle ({n_cycles} cycles)")

    _draw_charge_stats_table(ax_table, stats, n_cycles)

    fig.tight_layout()
    plt.show(block=False)


HISTOGRAM_BIN_COUNT = 30


def run_charge_histogram_analysis(dataset):
    result = _detect_dataset_cycle_charges(dataset, "charge histogram analysis")
    if result is None:
        return
    label, n_cycles = result["label"], result["n_cycles"]
    pos_charges, neg_charges, total_charges = result["pos_charges"], result["neg_charges"], result["total_charges"]

    stats = compute_charge_statistics(pos_charges, neg_charges, total_charges)

    print(f"[{label}] Task D: charge distribution histograms")
    _print_charge_stats_table(label, n_cycles, stats)

    # Total charge first/primary (matches the professor's sketch); positive and
    # negative are supplementary views of the same underlying per-cycle data.
    histogram_specs = [
        ("Total Charge", total_charges, "total"),
        ("Positive Charge", pos_charges, "pos"),
        ("Negative Charge", neg_charges, "neg"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 8), gridspec_kw={"height_ratios": [3, 1]})
    for col, (title_word, charges_c, key) in enumerate(histogram_specs):
        charges_nc = to_nanocoulombs(charges_c)

        ax_hist = axes[0, col]
        ax_hist.hist(charges_nc, bins=HISTOGRAM_BIN_COUNT, edgecolor='black')
        ax_hist.set_xlabel("Charge (nC)")
        ax_hist.set_ylabel("Frequency (cycle count)")
        ax_hist.set_title(title_word)

        _draw_single_charge_stats_table(axes[1, col], stats, key, n_cycles)

    fig.suptitle(f"{label} - Charge Distributions ({n_cycles} cycles)")
    fig.tight_layout()
    plt.show(block=False)


# Window-tiling helper: positions figure windows into a 2x2 grid on screen so
# multiple per-file popups don't stack directly on top of each other.
WINDOW_TILE_MARGIN_PX = 40
WINDOW_TILE_GAP_PX = 10


def _get_screen_size():
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        width = root.winfo_screenwidth()
        height = root.winfo_screenheight()
        root.destroy()
        return width, height
    except Exception:
        return 1920, 1080


def _tile_figure_window(fig, slot_index):
    """Position a matplotlib figure window into one quadrant of a 2x2 screen grid.

    slot_index: 0=top-left, 1=top-right, 2=bottom-left, 3=bottom-right.
    Silently does nothing if the current backend doesn't support window placement.
    """
    screen_w, screen_h = _get_screen_size()

    usable_w = screen_w - 2 * WINDOW_TILE_MARGIN_PX
    usable_h = screen_h - 2 * WINDOW_TILE_MARGIN_PX
    quadrant_w = int(round((usable_w - WINDOW_TILE_GAP_PX) / 2))
    quadrant_h = int(round((usable_h - WINDOW_TILE_GAP_PX) / 2))

    col = slot_index % 2
    row = (slot_index // 2) % 2
    x = int(round(WINDOW_TILE_MARGIN_PX + col * (quadrant_w + WINDOW_TILE_GAP_PX)))
    y = int(round(WINDOW_TILE_MARGIN_PX + row * (quadrant_h + WINDOW_TILE_GAP_PX)))

    backend = plt.get_backend().lower()
    try:
        manager = fig.canvas.manager
        if "tk" in backend:
            geometry_str = f"{quadrant_w}x{quadrant_h}+{x}+{y}"
            print(f"[tile] attempting: {geometry_str}")
            manager.window.geometry(geometry_str)
        elif "qt" in backend:
            print(f"[tile] attempting: move({x}, {y}), resize({quadrant_w}, {quadrant_h})")
            manager.window.move(x, y)
            manager.window.resize(quadrant_w, quadrant_h)
        # Any other backend (e.g. Agg, MacOSX, WebAgg): skip positioning silently.
    except Exception as e:
        print(f"[tile] window positioning failed: {type(e).__name__}: {e}")


# Task G: current-peak behavior classification.
# Max gap (as a fraction of one cycle) allowed within a single cluster of peak
# phase positions before a new canonical position is started.
PHASE_CLUSTER_TOLERANCE = 0.02
# Minimum |current|, in mA, for a local peak to be counted at all.
PEAK_MAGNITUDE_THRESHOLD_MA = 15.0


PEAK_MIN_DISTANCE_PERIOD_FRACTION = 0.05


def _find_cycle_peaks(current_ma, cycle_start_idx, cycle_end_idx, min_peak_distance_samples=1):
    span = cycle_end_idx - cycle_start_idx
    if span <= 0:
        return []
    segment = current_ma[cycle_start_idx:cycle_end_idx + 1]

    peaks = []
    pos_idx, _ = find_peaks(segment, height=PEAK_MAGNITUDE_THRESHOLD_MA, distance=min_peak_distance_samples)
    for i in pos_idx:
        peaks.append({"phase": i / span, "sign": "positive", "magnitude": float(abs(segment[i]))})

    neg_idx, _ = find_peaks(-segment, height=PEAK_MAGNITUDE_THRESHOLD_MA, distance=min_peak_distance_samples)
    for i in neg_idx:
        peaks.append({"phase": i / span, "sign": "negative", "magnitude": float(abs(segment[i]))})

    return peaks


def _cluster_phase_positions(phases, tolerance=PHASE_CLUSTER_TOLERANCE):
    if not phases:
        return []
    sorted_phases = sorted(phases)
    clusters = [[sorted_phases[0]]]
    for p in sorted_phases[1:]:
        if p - clusters[-1][-1] > tolerance:
            clusters.append([p])
        else:
            clusters[-1].append(p)
    return [float(np.mean(cluster)) for cluster in clusters]


def _assign_canonical_label(phase, canonical_positions, prefix):
    closest = int(np.argmin([abs(phase - c) for c in canonical_positions]))
    return f"{prefix}{closest + 1}"


def _fingerprint_sort_key(label):
    return (0 if label[0] == "P" else 1, int(label[1:]))


def _label_phase(label, pos_canonical, neg_canonical):
    canonical = pos_canonical if label[0] == "P" else neg_canonical
    return canonical[int(label[1:]) - 1]


def _phase_order_fingerprint(fp, pos_canonical, neg_canonical):
    """Order a Type's canonical labels by actual phase position (ascending),
    not by label name. Used only for gap-lookup "consecutive position" logic
    -- the fingerprint STRING shown in tables/legends keeps its own (label-name)
    order for display."""
    return sorted(fp, key=lambda lbl: _label_phase(lbl, pos_canonical, neg_canonical))


def _compute_cumulative_coverage(type_list, n_cycles):
    """Cumulative % of all cycles covered by the top N Types, for N = 1..total."""
    cumulative = 0
    cum_pct = []
    for _, cycle_idxs in type_list:
        cumulative += len(cycle_idxs)
        cum_pct.append(100.0 * cumulative / n_cycles)
    return cum_pct


def _print_type_coverage_table(cum_pct):
    print("\n  Type coverage by count:")
    total = len(cum_pct)
    last_bucket = -1
    for n, pct in enumerate(cum_pct, start=1):
        bucket = int(pct // 10)
        is_last = (n == total)
        if bucket > last_bucket or is_last:
            suffix = " (all Types)" if is_last else ""
            print(f"    Top {n} Types -> {pct:.1f}% of cycles{suffix}")
            last_bucket = bucket


def _auto_select_n(cum_pct):
    """Smallest N (out of all N=1..total) whose cumulative coverage is >=90%."""
    total_types = len(cum_pct)
    return next((i + 1 for i, pct in enumerate(cum_pct) if pct >= 90.0), total_types)


def _generate_odd_candidate_ns(total_types):
    """Candidate Type-count options at odd spacing (1, 3, 5, ...), always
    ending on the true total Type count even if it isn't itself odd."""
    if total_types <= 0:
        return []
    candidates = list(range(1, total_types + 1, 2))
    if candidates[-1] != total_types:
        candidates.append(total_types)
    return candidates


def _compute_cycle_peak_gaps(cycle_idx, peaks_in_cycle, cycle_duration_seconds):
    sorted_peaks = sorted(peaks_in_cycle, key=lambda p: p["phase"])
    gaps = []
    for a, b in zip(sorted_peaks, sorted_peaks[1:]):
        gap_phase = b["phase"] - a["phase"]
        gaps.append({
            "cycle_idx": cycle_idx,
            "gap_time_us": gap_phase * cycle_duration_seconds * 1e6,
            "label_a": a["label"], "label_b": b["label"],
            "sign_a": a["sign"], "sign_b": b["sign"],
            "phase_a": a["phase"], "phase_b": b["phase"],
        })
    return gaps


def _compute_peak_types_for_dataset(dataset):
    """Cycle-detection + Type (fingerprint) grouping for one dataset.

    Shared by the Task G GUI sub-panel (to compute coverage options) and by
    run_peak_classification_analysis (to avoid doing the detection twice).
    Returns None if fewer than 2 cycle boundaries were detected.
    """
    label = dataset["label"] or "combined dataset"
    time = dataset["data"]["time"].to_numpy()
    voltage = dataset["data"]["CH1V"].to_numpy()
    current = dataset["data"]["CH2A"].to_numpy()
    dt = float(np.median(np.diff(time)))

    expected_period = estimate_expected_period(dataset["data"])

    cycle_starts_t = detect_all_cycle_starts(voltage, dt, expected_period)
    if len(cycle_starts_t) < 2:
        return None

    cycle_starts_idx = np.round(cycle_starts_t / dt).astype(int)
    current_ma = to_milliamps(current)
    n_cycles = len(cycle_starts_idx) - 1

    min_peak_distance_samples = max(1, round((expected_period * PEAK_MIN_DISTANCE_PERIOD_FRACTION) / dt))

    all_peaks = []
    for cycle_idx in range(n_cycles):
        start_idx = cycle_starts_idx[cycle_idx]
        end_idx = cycle_starts_idx[cycle_idx + 1]
        for peak in _find_cycle_peaks(current_ma, start_idx, end_idx, min_peak_distance_samples):
            peak["cycle_idx"] = cycle_idx
            all_peaks.append(peak)

    pos_canonical = _cluster_phase_positions([p["phase"] for p in all_peaks if p["sign"] == "positive"])
    neg_canonical = _cluster_phase_positions([p["phase"] for p in all_peaks if p["sign"] == "negative"])

    for p in all_peaks:
        canonical = pos_canonical if p["sign"] == "positive" else neg_canonical
        prefix = "P" if p["sign"] == "positive" else "N"
        p["label"] = _assign_canonical_label(p["phase"], canonical, prefix)

    cycle_labels = {i: set() for i in range(n_cycles)}
    cycle_peak_magnitudes = {i: [] for i in range(n_cycles)}
    for p in all_peaks:
        cycle_labels[p["cycle_idx"]].add(p["label"])
        cycle_peak_magnitudes[p["cycle_idx"]].append((p["label"], p["magnitude"]))

    fingerprints = {
        i: tuple(sorted(cycle_labels[i], key=_fingerprint_sort_key)) for i in range(n_cycles)
    }

    groups = {}
    for i in range(n_cycles):
        groups.setdefault(fingerprints[i], []).append(i)
    type_list = sorted(groups.items(), key=lambda kv: -len(kv[1]))

    return {
        "label": label,
        "expected_period": expected_period,
        "dt": dt,
        "voltage": voltage,
        "current_ma": current_ma,
        "cycle_starts_idx": cycle_starts_idx,
        "n_cycles": n_cycles,
        "all_peaks": all_peaks,
        "pos_canonical": pos_canonical,
        "neg_canonical": neg_canonical,
        "cycle_peak_magnitudes": cycle_peak_magnitudes,
        "type_list": type_list,
    }


def run_peak_classification_analysis(dataset, n_selected=None, slot_index=None, precomputed=None):
    label = dataset["label"] or "combined dataset"

    analysis = precomputed if precomputed is not None else _compute_peak_types_for_dataset(dataset)
    if analysis is None:
        print(f"[{label}] Fewer than 2 cycle boundaries were detected - skipping Task G.")
        return

    expected_period = analysis["expected_period"]
    dt = analysis["dt"]
    voltage = analysis["voltage"]
    current_ma = analysis["current_ma"]
    cycle_starts_idx = analysis["cycle_starts_idx"]
    n_cycles = analysis["n_cycles"]
    pos_canonical = analysis["pos_canonical"]
    neg_canonical = analysis["neg_canonical"]
    cycle_peak_magnitudes = analysis["cycle_peak_magnitudes"]
    all_peaks = analysis["all_peaks"]
    type_list = analysis["type_list"]

    print(f"\n[{label}] Auto-estimated expected period: {_format_duration(expected_period)}")
    print(f"\n[{label}] Task G: current-peak behavior classification")
    print("  Canonical positive peak positions (% of cycle): " +
          (", ".join(f"P{i + 1}={c * 100:.1f}%" for i, c in enumerate(pos_canonical)) or "(none)"))
    print("  Canonical negative peak positions (% of cycle): " +
          (", ".join(f"N{i + 1}={c * 100:.1f}%" for i, c in enumerate(neg_canonical)) or "(none)"))

    total_types = len(type_list)
    singleton_count = sum(1 for _, v in type_list if len(v) == 1)

    print(f"\n  Total distinct Types found: {total_types}")
    print(f"  Types occurring only once: {singleton_count} "
          f"({singleton_count} cycles, {100.0 * singleton_count / n_cycles:.1f}% of total)")

    cum_pct = _compute_cumulative_coverage(type_list, n_cycles)
    _print_type_coverage_table(cum_pct)
    if n_selected is None:
        n_selected = _auto_select_n(cum_pct)
        print(f"  Auto-selected top {n_selected} Types ({cum_pct[n_selected - 1]:.1f}% coverage)")
    else:
        n_selected = min(n_selected, len(type_list))
        print(f"  Using GUI-selected top {n_selected} Types ({cum_pct[n_selected - 1]:.1f}% coverage)")

    print(f"\n  {'Type':<10}{'Fingerprint':<30}{'Cycles':>10}{'% of total':>12}")
    shown_types = type_list[:n_selected]
    for rank, (fp, cycle_idxs) in enumerate(shown_types, start=1):
        fp_str = ", ".join(fp) if fp else "(no peaks)"
        pct = 100.0 * len(cycle_idxs) / n_cycles
        print(f"  {'Type ' + str(rank):<10}{fp_str:<30}{len(cycle_idxs):>10}{pct:>11.1f}%")
    if len(type_list) > n_selected:
        remaining_cycles = sum(len(v) for _, v in type_list[n_selected:])
        print(f"  +{len(type_list) - n_selected} more types ({remaining_cycles} cycles total)")

    print("\n  Magnitude stats per Type (mean, range in mA):")
    for rank, (fp, cycle_idxs) in enumerate(shown_types, start=1):
        if not fp:
            continue
        print(f"    Type {rank} ({', '.join(fp)}):")
        for lbl in fp:
            mags = [m for i in cycle_idxs for (l, m) in cycle_peak_magnitudes[i] if l == lbl]
            if mags:
                print(f"      {lbl}: mean={np.mean(mags):.2f}mA, "
                      f"range=[{min(mags):.2f}, {max(mags):.2f}]mA")

    # Additive: inter-peak time gaps (real per-cycle peak sequence, not canonical positions).
    peaks_by_cycle = {i: [] for i in range(n_cycles)}
    for p in all_peaks:
        peaks_by_cycle[p["cycle_idx"]].append(p)

    all_gaps = []
    for cycle_idx in range(n_cycles):
        start_idx = cycle_starts_idx[cycle_idx]
        end_idx = cycle_starts_idx[cycle_idx + 1]
        cycle_duration_seconds = (end_idx - start_idx) * dt
        all_gaps.extend(
            _compute_cycle_peak_gaps(cycle_idx, peaks_by_cycle[cycle_idx], cycle_duration_seconds)
        )

    selected_for_gaps = type_list[:n_selected]
    print(f"\n  Inter-peak time gaps for top {n_selected} Types (mean +/- stdev, us):")
    for rank, (fp, cycle_idxs) in enumerate(selected_for_gaps, start=1):
        pct = 100.0 * len(cycle_idxs) / n_cycles
        header = f"Type {rank} ({len(cycle_idxs)} cycles, {pct:.1f}%)"
        if len(fp) < 2:
            print(f"    {header}: (fewer than 2 peak positions - no gaps)")
            continue
        cycle_idx_set = set(cycle_idxs)
        phase_ordered = _phase_order_fingerprint(fp, pos_canonical, neg_canonical)
        segments = []
        for label_a, label_b in zip(phase_ordered, phase_ordered[1:]):
            matching = [
                g["gap_time_us"] for g in all_gaps
                if g["cycle_idx"] in cycle_idx_set
                and g["label_a"] == label_a and g["label_b"] == label_b
            ]
            if matching:
                std_us = float(np.std(matching, ddof=1)) if len(matching) > 1 else 0.0
                segments.append(f"{label_a} -> {label_b}: {np.mean(matching):.1f} +/- {std_us:.1f} us")
            else:
                segments.append(f"{label_a} -> {label_b}: (no matching gaps)")
        print(f"    {header}: " + " | ".join(segments))

    all_gap_values = [g["gap_time_us"] for g in all_gaps]
    print("\n  Overall inter-peak time gap statistics (all cycles, us):")
    if all_gap_values:
        gap_arr = np.array(all_gap_values)
        overall_std = float(np.std(gap_arr, ddof=1)) if len(gap_arr) > 1 else 0.0
        print(f"    mean={np.mean(gap_arr):.2f}us, stdev={overall_std:.2f}us, "
              f"median={np.median(gap_arr):.2f}us, min={np.min(gap_arr):.2f}us, "
              f"max={np.max(gap_arr):.2f}us  (n={len(gap_arr)} peak-pairs)")
    else:
        print("    No consecutive peak pairs were found - no gap statistics available.")

    fig, (ax_top, ax_bottom, ax_wave) = plt.subplots(3, 1, figsize=(12, 13), sharex=True)

    mean_cycle_duration_us = float(np.mean(
        [(cycle_starts_idx[i + 1] - cycle_starts_idx[i]) * dt for i in range(n_cycles)]
    )) * 1e6

    def phase_to_us(phase):
        return phase * mean_cycle_duration_us

    def us_to_phase(us):
        return us / mean_cycle_duration_us

    rep_idx = 0
    rep_start = cycle_starts_idx[rep_idx]
    rep_end = cycle_starts_idx[rep_idx + 1]
    rep_phase = np.linspace(0, 1, rep_end - rep_start)
    ax_top.plot(rep_phase, to_kilovolts(voltage[rep_start:rep_end]))
    ax_top.set_xlabel("Phase (fraction of cycle)")
    ax_top.set_ylabel("Voltage (kV)")
    ax_top.set_title(f"{label} - Representative voltage cycle (cycle {rep_idx + 1})")

    selected_types = type_list[:n_selected]
    markers = ['o', 's', '^', 'D', 'v']
    colors = plt.cm.tab10.colors
    for rank, (fp, cycle_idxs) in enumerate(selected_types):
        pct = 100.0 * len(cycle_idxs) / n_cycles
        legend_label = f"Type {rank + 1} ({len(cycle_idxs)} cycles, {pct:.1f}%)"
        marker = markers[rank % len(markers)]
        color = colors[rank % len(colors)]
        if not fp:
            ax_bottom.plot([], [], marker=marker, color=color, linestyle='none',
                            label=legend_label + " - no peaks")
            continue
        phases = []
        for lbl in fp:
            canonical = pos_canonical if lbl[0] == "P" else neg_canonical
            phases.append(canonical[int(lbl[1:]) - 1])
        size = 40 + 10 * len(cycle_idxs)
        ax_bottom.scatter(phases, [rank] * len(phases), marker=marker, color=color,
                           s=size, label=legend_label)
    ax_bottom.set_xlabel("Phase (fraction of cycle)")
    ax_bottom.set_yticks([])
    ax_bottom.set_title(f"{label} - Top {n_selected} Type peak positions")
    ax_bottom.legend(markerscale=0.5, handletextpad=0.5, labelspacing=0.4, fontsize=8)
    ax_bottom_secondary = ax_bottom.secondary_xaxis('top', functions=(phase_to_us, us_to_phase))
    ax_bottom_secondary.set_xlabel("Time (us)")

    for rank, (fp, cycle_idxs) in enumerate(selected_types):
        pct = 100.0 * len(cycle_idxs) / n_cycles
        legend_label = f"Type {rank + 1} ({len(cycle_idxs)} cycles, {pct:.1f}%)"
        marker = markers[rank % len(markers)]
        color = colors[rank % len(colors)]
        rep_cycle_idx = cycle_idxs[0]
        wave_start = cycle_starts_idx[rep_cycle_idx]
        wave_end = cycle_starts_idx[rep_cycle_idx + 1]
        wave_phase = np.linspace(0, 1, wave_end - wave_start)
        ax_wave.plot(wave_phase, current_ma[wave_start:wave_end], color=color,
                     marker=marker, markevery=max(1, (wave_end - wave_start) // 15),
                     markersize=5, label=legend_label)
    ax_wave.set_xlabel("Phase (fraction of cycle)")
    ax_wave.set_ylabel("Current (mA)")
    ax_wave.set_title(f"{label} - Representative current waveform per Type")
    ax_wave.legend(markerscale=0.5, handletextpad=0.5, labelspacing=0.4, fontsize=8)
    ax_wave_secondary = ax_wave.secondary_xaxis('top', functions=(phase_to_us, us_to_phase))
    ax_wave_secondary.set_xlabel("Time (us)")

    fig.tight_layout()
    plt.show(block=False)
    if slot_index is not None:
        _tile_figure_window(fig, slot_index)


def run_effective_value_comparison(datasets, overlaps):
    _warn_about_possible_duplicate_snaps(overlaps)

    labels = []
    vrms_list = []
    irms_list = []
    for dataset in datasets:
        vrms, irms = compute_rms_per_file(dataset)
        labels.append(dataset["label"] or "combined")
        vrms_list.append(vrms)
        irms_list.append(irms)

    print("\nRMS Voltage/Current per file:")
    print(f"  {'File':<30}{'Vrms (kV)':>14}{'Irms (mA)':>14}")
    for label, vrms, irms in zip(labels, vrms_list, irms_list):
        print(f"  {label:<30}{to_kilovolts(vrms):>14.4f}{to_milliamps(irms):>14.4f}")

    x = np.arange(len(labels))

    fig, (ax_v, ax_i) = plt.subplots(1, 2, figsize=(16, 6))

    vrms_kv = to_kilovolts(np.array(vrms_list))
    ax_v.bar(x, vrms_kv)
    for xi, val in zip(x, vrms_kv):
        ax_v.text(xi, val, f"{val:.3f}", ha='center', va='bottom')
    ax_v.set_xticks(x)
    ax_v.set_xticklabels(labels, rotation=30, ha='right')
    ax_v.set_ylabel("Vrms (kV)")
    ax_v.set_title("RMS Voltage per file")

    irms_ma = to_milliamps(np.array(irms_list))
    ax_i.bar(x, irms_ma)
    for xi, val in zip(x, irms_ma):
        ax_i.text(xi, val, f"{val:.3f}", ha='center', va='bottom')
    ax_i.set_xticks(x)
    ax_i.set_xticklabels(labels, rotation=30, ha='right')
    ax_i.set_ylabel("Irms (mA)")
    ax_i.set_title("RMS Current per file")

    fig.tight_layout()
    plt.show()


def run_dual_mode_comparison():
    try:
        datasets, overlaps = load_scope_files_per_file()
    except RuntimeError as e:
        print(f"ERROR loading files: {e}")
        return

    mode, type_g_n, type_g_cache = select_analysis_mode(datasets)

    if mode in ("A", "ALL"):
        for dataset in datasets:
            print(f"\n=== Task A: fixed 1ms-interval cycle comparison - {dataset['label']} ===")
            run_fixed_interval_cycle_comparison(dataset)
        plt.show()

    if mode in ("B", "ALL"):
        print("\n=== Task B: RMS comparison across snaps ===")
        run_effective_value_comparison(datasets, overlaps)

    if mode in ("C", "ALL"):
        for dataset in datasets:
            print(f"\n=== Task C: charge-per-cycle analysis - {dataset['label']} ===")
            run_charge_per_cycle_analysis(dataset)
        plt.show()

    if mode in ("D", "ALL"):
        for dataset in datasets:
            print(f"\n=== Task D: charge histogram - {dataset['label']} ===")
            run_charge_histogram_analysis(dataset)
        plt.show()

    if mode in ("G", "ALL"):
        for i, dataset in enumerate(datasets):
            print(f"\n=== Task G: peak-behavior classification - {dataset['label']} ===")
            # Tile each file's main Task G figure into a screen quadrant so multiple
            # snaps don't stack directly on top of each other. Only slots 0-2 are
            # used for now (top-left, top-right, bottom-left); slot 3 (bottom-right)
            # is intentionally left unused, reserved for a future addition (e.g. the
            # peak-gap histogram). Cycling through 3 slots is a simple wraparound
            # fallback for many files and could be improved later if needed.
            slot_index = i % 3
            n_selected = type_g_n.get(dataset["label"])
            precomputed = type_g_cache.get(dataset["label"])
            run_peak_classification_analysis(
                dataset, n_selected=n_selected, slot_index=slot_index, precomputed=precomputed
            )
        plt.show()


if __name__ == "__main__":
    print("=" * 60)
    print(f"WAVE COMPARE  v{VERSION}  [build: {BUILD_TAG}]")
    print("=" * 60)

    try:
        run_dual_mode_comparison()
    except Exception as e:
        print(f"ERROR: unexpected failure ({type(e).__name__}: {e})")
        sys.exit(1)

    print("\nDone.")
