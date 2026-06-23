import sys
import os
import math
import statistics
from pathlib import Path
from datetime import datetime

def parse_file(path: str) -> tuple[dict, list[tuple[float, float, float]]]:
    header = {}
    samples = []
    in_data = False
    header_skips = 0

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()

            if not in_data:
                parts = line.split("\t")
                if len(parts) >= 2 and parts[0] and parts[1]:
                    header[parts[0]] = parts[1]

            if line.startswith("Setpoint"):
                in_data = True
                header_skips = 2
                continue

            if in_data:
                if header_skips > 0:
                    header_skips -= 1
                    continue
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) < 3:
                    continue
                try:
                    s, a, c = float(parts[0]), float(parts[1]), float(parts[2])
                    if s == 0.0 and a == 0.0 and c == 0.0:
                        continue
                    samples.append((s, a, c))
                except ValueError:
                    pass

    return header, samples


def auto_threshold(
    all_samples: list[list[tuple[float, float, float]]],
    search_window: int = 30,
) -> float:

    first_temp = all_samples[0][0][1]
    last_temp  = all_samples[0][-1][1]
    falling    = last_temp < first_temp

    anchors = []
    for samples in all_samples:
        window = samples[: min(search_window, len(samples))]
        if falling:

            anchor = max(s[1] for s in window)
        else:

            anchor = min(s[1] for s in window)
        anchors.append(anchor)

    if falling:
        threshold = min(anchors) - 0.05
    else:
        threshold = max(anchors) + 0.05

    return threshold, anchors

def find_ingress(
    samples: list[tuple[float, float, float]],
    threshold: float,
    run_label: str = "",
) -> float:
    temp_min = min(s[1] for s in samples)
    temp_max = max(s[1] for s in samples)

    if threshold > temp_max:
        raise ValueError(
            f"{run_label}: threshold {threshold:.4f} °C is ABOVE the peak "
            f"temperature {temp_max:.4f} °C.\n"
            f"  → Lower --threshold or check your data."
        )
    if threshold < temp_min:
        raise ValueError(
            f"{run_label}: threshold {threshold:.4f} °C is BELOW the minimum "
            f"temperature {temp_min:.4f} °C.\n"
            f"  → Raise --threshold or check your data."
        )


    first_temp = samples[0][1]
    last_temp  = samples[-1][1]
    falling    = last_temp < first_temp

    for i in range(1, len(samples)):
        v_prev = samples[i - 1][1]
        v_curr = samples[i    ][1]

        if falling:

            if v_curr < threshold <= v_prev:
                frac = (threshold - v_prev) / (v_curr - v_prev)
                return (i - 1) + frac
        else:

            if v_curr > threshold >= v_prev:
                frac = (threshold - v_prev) / (v_curr - v_prev)
                return (i - 1) + frac

    raise ValueError(
        f"{run_label}: temperature never crosses {threshold:.4f} °C on a "
        f"{'falling' if falling else 'rising'} edge.\n"
        f"  Temperature range: {temp_min:.4f} – {temp_max:.4f} °C.\n"
        f"  → Adjust --threshold or check the run."
    )

def interp_sample(
    samples: list[tuple[float, float, float]],
    idx: float,
) -> tuple[float, float, float]:
    i = int(idx)
    f = idx - i
    if i + 1 >= len(samples):
        return samples[-1]
    s0, s1 = samples[i], samples[i + 1]
    return (
        s0[0] + f * (s1[0] - s0[0]),
        s0[1] + f * (s1[1] - s0[1]),
        s0[2] + f * (s1[2] - s0[2]),
    )

def phase_fold_and_average(
    all_samples: list[list[tuple[float, float, float]]],
    ingress_indices: list[float],
    interval: float,
) -> tuple[list[tuple[float, float, float]], list[float], list[float], list[int]]:
    windows = [
        int(len(samples) - ingress)
        for samples, ingress in zip(all_samples, ingress_indices)
    ]
    n_pts = max(windows)

    averaged  = []
    std_temp  = []
    sem_temp  = []
    n_contrib = []

    for phase_step in range(n_pts):

        vals = [
            interp_sample(samples, ingress + phase_step)
            for samples, ingress, window in zip(all_samples, ingress_indices, windows)
            if phase_step < window
        ]
        n = len(vals)
        mean_sp  = statistics.mean(v[0] for v in vals)
        mean_act = statistics.mean(v[1] for v in vals)
        mean_cur = statistics.mean(v[2] for v in vals)
        averaged.append((mean_sp, mean_act, mean_cur))
        n_contrib.append(n)

        sd = statistics.stdev(v[1] for v in vals) if n > 1 else 0.0
        std_temp.append(sd)
        sem_temp.append(sd / math.sqrt(n))

    return averaged, std_temp, sem_temp, n_contrib

def write_tec_output(
    path: str,
    header: dict,
    averaged: list[tuple[float, float, float]],
    source_files: list[str],
    ingress_indices: list[float],
    threshold: float,
    interval: float,
) -> None:
    now = datetime.now()
    lines = []

    lines.append(f"Date\t{now.strftime('%d-%b-%y')}\t")
    lines.append(f"Time\t{now.strftime('%I:%M %p')}\t")
    lines.append("\t\t")
    lines.append("Generated by\ttec_noise_cancel.py (phase-fold method)\t")
    lines.append(f"Source files\t{len(source_files)} runs merged\t")
    lines.append(f"Phase anchor\t{threshold:.4f} degC falling-edge crossing\t")
    for i, (sf, ig) in enumerate(zip(source_files, ingress_indices)):
        lines.append(
            f"  Run {i}\t{Path(sf).name}  "
            f"(ingress @ sample {ig:.4f}, t={ig * interval:.3f}s)\t"
        )
    lines.append("\t\t")

    preserve = [
        "Current Limit", "Voltage Limit",
        "Low Temp. Limit", "High Temp. Limit",
        "Kp", "Ki", "Kd", "Int",
        "Sample Interval", "Temp. Display Offset",
    ]
    unit_map = {
        "Current Limit":      "\tAmps",
        "Voltage Limit":      "\tVolts",
        "Low Temp. Limit":    "\tdegC",
        "High Temp. Limit":   "\tdegC",
        "Sample Interval":    "\tseconds",
        "Temp. Display Offset": "\tdegC",
    }
    for key in preserve:
        if key in header:
            lines.append(f"{key}\t{header[key]}{unit_map.get(key, '')}")

    lines.append(f"Number of Samples\t{len(averaged)}\t")
    lines.append("\t\t")
    lines.append("Setpoint\tActual\tCurrent")
    lines.append("x1\tx1\tx1")
    lines.append("degC\tdegC\tAmps")

    for sp, act, cur in averaged:
        lines.append(f"{sp:.6f}\t{act:.6f}\t{cur:.6f}")

    with open(path, "w", encoding="utf-8", newline="\r\n") as f:
        f.write("\n".join(lines) + "\n")

    print(f"  ✓  TEC format output  →  {path}")


def bin_averaged(
    averaged: list[tuple[float, float, float]],
    std_temp: list[float],
    n_contrib: list[int],
    interval: float,
    bins: int,
) -> list[tuple[float, float, float, float, int]]:
    n_pts = len(averaged)
    bin_size = n_pts / bins
    result = []
    for b in range(bins):
        lo = int(b * bin_size)
        hi = int((b + 1) * bin_size)
        if lo >= n_pts:
            break
        hi = min(hi, n_pts)
        t_centre = ((lo + hi - 1) / 2) * interval
        temps  = [averaged[i][1] for i in range(lo, hi)]
        stds   = [std_temp[i]    for i in range(lo, hi)]
        ns     = [n_contrib[i]   for i in range(lo, hi)]
        mean_t = statistics.mean(temps)

        pooled_std = math.sqrt(statistics.mean(s**2 for s in stds)) if stds else 0.0
        min_n      = min(ns)
        pooled_sem = pooled_std / math.sqrt(min_n) if min_n > 0 else 0.0
        result.append((t_centre, mean_t, pooled_std, pooled_sem, min_n))
    return result


def write_txt(
    path: str,
    averaged: list[tuple[float, float, float]],
    std_temp: list[float],
    sem_temp: list[float],
    n_contrib: list[int],
    interval: float,
    bins: int,
) -> tuple[str, str]:
    stem        = Path(path).stem
    out_dir     = Path(path).parent
    binned_path = out_dir / f"noise_reduced_{stem}_binned{bins}.txt"

    header = "time_s\tmean_temp_C\tstd_C\tsem_C\tn_runs_contributing"


    binned = bin_averaged(averaged, std_temp, n_contrib, interval, bins)
    blines = [header]
    for t, mt, sd, se, n in binned:
        blines.append(f"{t:.3f}\t{mt:.6f}\t{sd:.6f}\t{se:.6f}\t{n}")
    with open(str(binned_path), "w", encoding="utf-8", newline="\r\n") as f:
        f.write("\n".join(blines) + "\n")
    print(f"  ✓  Binned txt ({bins} pts)   →  {binned_path}")

    return str(binned_path)

def plot_stacked(
    averaged: list[tuple[float, float, float]],
    std_temp: list[float],
    n_contrib: list[int],
    all_samples: list[list[tuple[float, float, float]]],
    ingress_indices: list[float],
    source_files: list[str],
    interval: float,
    bins: int,
    out_stem: str,
    out_dir: str,
) -> None:
    try:
        import matplotlib
    except ImportError:
        print("  ⚠  matplotlib not found — skipping plot.")
        return


    import matplotlib.pyplot as plt
    for backend in ["MacOSX", "TkAgg", "Qt5Agg", "Agg"]:
        try:
            matplotlib.use(backend)
            import matplotlib.pyplot as plt
            plt.figure()
            plt.close()
            break
        except Exception:
            continue

    time_full = [i * interval for i in range(len(averaged))]
    mean_temp = [s[1] for s in averaged]
    upper     = [m + s for m, s in zip(mean_temp, std_temp)]
    lower     = [m - s for m, s in zip(mean_temp, std_temp)]

    binned    = bin_averaged(averaged, std_temp, n_contrib, interval, bins)
    bin_t     = [b[0] for b in binned]
    bin_m     = [b[1] for b in binned]
    bin_se    = [b[3] for b in binned]

    n_files   = len(source_files)
    colors    = ["#378ADD", "#1D9E75", "#D85A30", "#BA7517",
                 "#D4537E", "#7F77DD", "#639922", "#E24B4A"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        f"TEC Noise Reduction — Phase-Fold Stack  ({n_files} runs)",
        fontsize=13, fontweight="bold"
    )


    ax = axes[0]
    for i, (samples, ig) in enumerate(zip(all_samples, ingress_indices)):
        t_raw  = [(j - ig) * interval for j in range(len(samples))]
        t_plot = [t for t in t_raw if t >= 0]
        v_plot = [samples[j][1] for j in range(len(samples)) if t_raw[j] >= 0]
        ax.plot(t_plot, v_plot, lw=0.8, alpha=0.55,
                color=colors[i % len(colors)],
                label=Path(source_files[i]).name)
    ax.set_title("Raw Runs", fontsize=11)
    ax.set_xlabel("Time since ingress (s)")
    ax.set_ylabel("Temperature (°C)")
    ax.legend(fontsize=7, loc="upper right")


    ax = axes[1]
    ax.fill_between(time_full, lower, upper,
                    alpha=0.2, color="#378ADD", label="±1σ")
    ax.plot(time_full, mean_temp, color="#378ADD", lw=1.2, label="Stacked mean")
    ax.set_title("Stacked (Phase-Folded)", fontsize=11)
    ax.set_xlabel("Time since ingress (s)")
    ax.set_ylabel("Temperature (°C)")
    ax.legend(fontsize=8)


    ax = axes[2]
    ax.errorbar(bin_t, bin_m, yerr=bin_se,
                fmt="o-", color="#1D9E75", markersize=3,
                lw=1.2, elinewidth=0.8, capsize=3,
                label=f"Binned mean ±SEM ({bins} bins)")
    ax.set_title(f"Binned ({bins} bins)", fontsize=11)
    ax.set_xlabel("Time since ingress (s)")
    ax.set_ylabel("Temperature (°C)")
    ax.legend(fontsize=8)

    plt.tight_layout()

    plot_path = Path(out_dir) / f"noise_reduced_{out_stem}.png"
    plt.savefig(str(plot_path), dpi=150, bbox_inches="tight")
    print(f"  ✓  Plot saved           →  {plot_path}")


    import subprocess, platform
    try:
        if platform.system() == "Darwin":
            subprocess.Popen(["open", str(plot_path)])
        elif platform.system() == "Windows":
            os.startfile(str(plot_path))
        else:
            subprocess.Popen(["xdg-open", str(plot_path)])
    except Exception:
        pass

    try:
        plt.show(block=True)
    except Exception:
        pass
    plt.close()

def write_report(
    path: str,
    source_files: list[str],
    all_samples: list[list[tuple[float, float, float]]],
    averaged: list[tuple[float, float, float]],
    std_temp: list[float],
    n_contrib: list[int],
    ingress_indices: list[float],
    threshold: float,
    interval: float,
    detected_peaks: list[float],
) -> None:
    n = len(source_files)
    theoretical_gain = 1 / math.sqrt(n)

    lines = []
    lines.append("=" * 64)
    lines.append("  TEC Noise Cancellation Report  —  Phase-Fold Method")
    lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 64)
    lines.append(f"\nRuns merged  : {n}")
    lines.append(f"Phase anchor : {threshold:.4f} °C falling-edge crossing")
    lines.append(f"\nPer-run summary:")
    for i, (sf, ig, pk) in enumerate(
        zip(source_files, ingress_indices, detected_peaks)
    ):
        lines.append(
            f"  [{i}] {Path(sf).name:<48}\n"
            f"       peak={pk:.4f} °C   "
            f"ingress @ sample {ig:.4f}  (t={ig * interval:.3f} s)"
        )

    offsets = [ig - ingress_indices[0] for ig in ingress_indices]
    lines.append(f"\nTiming offsets relative to Run 0:")
    for i, off in enumerate(offsets):
        lines.append(f"  Run {i}: {off:+.4f} samples  ({off * interval:+.4f} s)")

    min_contrib = min(n_contrib)
    full_cover  = sum(1 for n in n_contrib if n == len(source_files))
    lines.append(
        f"\nTotal output samples    : {len(averaged)}  ({len(averaged) * interval:.1f} s)"
    )
    lines.append(
        f"Fully covered samples   : {full_cover}  ({full_cover * interval:.1f} s)  "
        f"[all {len(source_files)} runs contributing]"
    )
    lines.append(
        f"Minimum run coverage    : {min_contrib} run(s) at tail of output"
    )


    lines.append("\n── Raw noise per run (post-ingress window) ──────────────")
    per_run_temp_stds = []
    for i, (samples, ig) in enumerate(zip(all_samples, ingress_indices)):
        folded_act = [
            interp_sample(samples, ig + j)[1] for j in range(len(averaged))
        ]
        sd = statistics.stdev(folded_act) if len(folded_act) > 1 else 0.0
        pk = max(folded_act) - min(folded_act)
        per_run_temp_stds.append(sd)
        lines.append(
            f"  Run {i}  ({Path(source_files[i]).name})\n"
            f"    actual temp:  std={sd:.6f} °C   peak-to-peak={pk:.6f} °C"
        )

    lines.append("\n── Averaged output noise ─────────────────────────────────")
    avg_std = statistics.mean(std_temp) if std_temp else 0.0
    avg_act_vals = [s[1] for s in averaged]
    overall_std  = statistics.stdev(avg_act_vals) if len(avg_act_vals) > 1 else 0.0
    pk_avg = max(avg_act_vals) - min(avg_act_vals)
    lines.append(f"  Mean per-point std  : {avg_std:.6f} °C")
    lines.append(f"  Overall signal std  : {overall_std:.6f} °C")
    lines.append(f"  Peak-to-peak range  : {pk_avg:.6f} °C")

    mean_raw_std = statistics.mean(per_run_temp_stds) if per_run_temp_stds else 1.0
    actual_gain  = avg_std / mean_raw_std if mean_raw_std > 0 else 0.0

    lines.append("\n── Noise reduction summary ───────────────────────────────")
    lines.append(f"  Runs averaged              : {n}")
    lines.append(
        f"  Theoretical gain (1/√{n})   : {theoretical_gain:.4f}  "
        f"({(1 - theoretical_gain) * 100:.1f}% noise reduction)"
    )
    lines.append(f"  Mean raw temp std          : {mean_raw_std:.6f} °C")
    lines.append(f"  Mean averaged point std    : {avg_std:.6f} °C")
    lines.append(
        f"  Actual gain                : {actual_gain:.4f}  "
        f"({(1 - actual_gain) * 100:.1f}% noise reduction)"
    )
    lines.append(f"\n  To improve further:")
    for target_n in [4, 9, 16, 25]:
        if target_n > n:
            red = (1 - 1 / math.sqrt(target_n)) * 100
            lines.append(
                f"    {target_n:>2} runs  →  {red:.0f}% theoretical noise reduction"
            )
    lines.append("\n" + "=" * 64)

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"  ✓  Noise report       →  {path}")

def parse_args(argv):
    files         = []
    threshold     = None
    out_name      = "averaged_output.txt"
    search_window = 30
    bins          = 100
    i = 0
    while i < len(argv):
        if argv[i] == "--threshold" and i + 1 < len(argv):
            threshold = float(argv[i + 1]); i += 2
        elif argv[i] == "--out" and i + 1 < len(argv):
            out_name = argv[i + 1]; i += 2
        elif argv[i] == "--search" and i + 1 < len(argv):
            search_window = int(argv[i + 1]); i += 2
        elif argv[i] == "--bins" and i + 1 < len(argv):
            bins = int(argv[i + 1]); i += 2
        else:
            files.append(argv[i]); i += 1
    return files, threshold, out_name, search_window, bins


def main(argv):
    file_paths, threshold_arg, out_name, search_window, bins = parse_args(argv)

    if len(file_paths) < 2:
        print(__doc__)
        print("ERROR: supply at least 2 input files.")
        sys.exit(1)

    for fp in file_paths:
        if not os.path.isfile(fp):
            print(f"ERROR: file not found — {fp}")
            sys.exit(1)


    print(f"\nReading {len(file_paths)} file(s)...")
    all_headers = []
    all_samples = []
    for fp in file_paths:
        hdr, smp = parse_file(fp)
        all_headers.append(hdr)
        all_samples.append(smp)
        print(f"  {Path(fp).name:<52} {len(smp):>4} valid samples")


    try:
        interval = float(all_headers[0].get("Sample Interval", "0.5"))
    except ValueError:
        interval = 0.5


    detected_peaks: list[float]
    if threshold_arg is None:
        threshold, detected_peaks = auto_threshold(all_samples, search_window)
        print(f"\nAuto-detected threshold: {threshold:.4f} °C")
        print(f"  (min peak across runs = {min(detected_peaks):.4f} °C  minus 0.05 °C margin)")
        for i, pk in enumerate(detected_peaks):
            print(f"  Run {i} peak: {pk:.4f} °C")
    else:
        threshold = threshold_arg

        _, detected_peaks = auto_threshold(all_samples, search_window)
        print(f"\nUsing manual threshold: {threshold:.4f} °C")


    print(f"\nLocating ingress in each run...")
    ingress_indices = []
    for fp, samples in zip(file_paths, all_samples):
        label = Path(fp).name
        try:
            ig = find_ingress(samples, threshold, run_label=label)
            ingress_indices.append(ig)
            print(f"  {label:<52} ingress @ sample {ig:.4f}  "
                  f"(t = {ig * interval:.3f} s)")
        except ValueError as e:
            print(f"\nERROR: {e}")
            sys.exit(1)

    offsets = [ig - ingress_indices[0] for ig in ingress_indices]
    print(f"\n  Timing offsets relative to Run 0:")
    for i, off in enumerate(offsets):
        print(f"    Run {i}: {off:+.4f} samples  ({off * interval:+.4f} s)")


    print("\nPhase-folding and averaging...")
    averaged, std_temp, sem_temp, n_contrib = phase_fold_and_average(
        all_samples, ingress_indices, interval
    )
    full_cover = sum(1 for n in n_contrib if n == len(file_paths))
    print(f"  Total output   : {len(averaged)} samples  ({len(averaged) * interval:.1f} s)")
    print(f"  Full coverage  : {full_cover} samples  ({full_cover * interval:.1f} s)  [all runs present]")

    out_path   = Path(out_name).resolve()
    out_dir    = out_path.parent
    out_stem   = out_path.stem
    out_data   = out_dir / f"noise_reduced_{out_stem}.txt"
    out_report = out_dir / f"noise_reduced_{out_stem}_report.txt"

    print()
    write_txt(
        str(out_dir / f"{out_stem}_data.txt"),
        averaged, std_temp, sem_temp, n_contrib, interval, bins,
    )
    write_report(
        str(out_report), file_paths, all_samples,
        averaged, std_temp, n_contrib, ingress_indices,
        threshold, interval, detected_peaks,
    )
    plot_stacked(
        averaged, std_temp, n_contrib,
        all_samples, ingress_indices, file_paths,
        interval, bins, out_stem, str(out_dir),
    )


    n = len(file_paths)
    print(
        f"\nNoise reduction (theoretical):  1/√{n} = {1/math.sqrt(n):.4f}  "
        f"— {(1 - 1/math.sqrt(n))*100:.1f}% reduction"
    )
    if std_temp:
        print(f"Mean per-point temp std : {statistics.mean(std_temp):.6f} °C")
    print()

def pick_files_gui() -> list[str]:
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        paths = filedialog.askopenfilenames(
            title="Select Newport 3700 TEC run files (2 or more)",
            filetypes=[("TEC data files", "*.txt"), ("All files", "*.*")],
        )
        root.destroy()
        return sorted(paths)
    except Exception as e:
        print(f"GUI picker unavailable ({e}). "
              "Pass files as command-line arguments instead.")
        return []


if __name__ == "__main__":
    print("Opening file picker — hold Ctrl to select multiple files...")
    selected = pick_files_gui()
    if len(selected) < 2:
        print("Please select at least 2 files. Exiting.")
        sys.exit(1)
    print(f"Selected {len(selected)} file(s):")
    for p in selected:
        print(f"  {p}")


    data_folder = str(Path(selected[0]).parent)
    out = os.path.join(data_folder, "averaged_output.txt")
    print(f"\nOutputs will be saved to: {data_folder}")

    bins_str = input("Number of bins (press Enter for 100): ").strip()
    bins_val = int(bins_str) if bins_str.isdigit() else 100
    main(list(selected) + ["--out", out, "--bins", str(bins_val)])
