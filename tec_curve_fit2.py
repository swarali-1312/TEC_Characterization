import sys, os, math
import numpy as np
from scipy.optimize import curve_fit
from scipy.ndimage import uniform_filter1d
from pathlib import Path
from datetime import datetime

MIN_SEG1_S   = 30.0
MAX_SEG1_S   = None
KINK_SNR_THR = 3.0
KINK_SC_THR  = 2

def pick_file():
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk(); root.withdraw(); root.attributes("-topmost", True)
        path = filedialog.askopenfilename(
            title="Select TEC data file (Newport raw OR noise-cancelled)",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        root.destroy()
        if not path:
            print("No file selected. Exiting."); sys.exit(0)
        return path
    except Exception as exc:
        print("GUI picker unavailable ({}).".format(exc))
        print("Pass the file path as a command-line argument."); sys.exit(1)

def detect_file_type(path):
    with open(path, "r", errors="replace") as fh:
        for i, line in enumerate(fh):
            if i > 30: break
            if line.strip().startswith("time_s"):        return "noise_cancelled"
            if "Setpoint" in line and "Actual" in line:  return "newport"
            if "Sample Interval" in line:                return "newport"
    return "newport"

def load_newport(path):
    sample_interval = 0.5
    temps, data_started, skip_units = [], False, 2
    with open(path, "r", errors="replace") as fh:
        for line in fh:
            s = line.strip()
            if "Sample Interval" in s:
                try: sample_interval = float(s.split("\t")[1])
                except: pass
            if s.startswith("Setpoint"):
                data_started = True; continue
            if not data_started: continue
            if skip_units > 0:   skip_units -= 1; continue
            if not s: continue
            parts = s.split("\t")
            if len(parts) < 2: continue
            try:
                temp = float(parts[1])
                if temp == 0.0 and len(temps) > 10: break
                temps.append(temp)
            except ValueError: continue
    if not temps:
        print("ERROR: No temperature data found: {}".format(path)); sys.exit(1)
    times = np.array([i * sample_interval for i in range(len(temps))])
    return times, np.array(temps), np.full(len(temps), 0.1)


def load_noise_cancelled(path):
    rows = []
    with open(path, "r", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("time"): continue
            parts = line.split("\t")
            if len(parts) < 4: continue
            try: rows.append([float(p) for p in parts[:4]])
            except ValueError: continue
    if not rows:
        print("ERROR: No data loaded: {}".format(path)); sys.exit(1)
    arr = np.array(rows)
    sems = np.where(arr[:, 3] < 1e-6, 1e-4, arr[:, 3])
    return arr[:, 0], arr[:, 1], sems


def load_data(path):
    ft = detect_file_type(path)
    if ft == "newport": t, T, s = load_newport(path)
    else:               t, T, s = load_noise_cancelled(path)
    return t, T, s, ft

def detect_direction(temps):
    n = len(temps)
    early = float(np.mean(temps[:max(1, n // 10)]))
    late  = float(np.mean(temps[max(1, 9 * n // 10):]))
    return "heating" if late > early else "cooling"

def single_exp_model(t, T_ss, tau, T0):
    return T_ss + (T0 - T_ss) * np.exp(-t / tau)


def fit_single(times, temps, sems, direction):
    T0  = float(temps[0])
    t   = times - times[0]
    dur = float(t[-1])

    tail_n  = max(10, len(temps) // 5)
    T_ss_g  = float(np.median(temps[-tail_n:]))
    tau_g   = dur / 4.0

    if direction == "cooling":
        lo = [-np.inf, 1.0];  hi = [T0 + 2.0, 2000.0]
    else:
        lo = [T0 - 2.0, 1.0]; hi = [np.inf, 2000.0]

    T_ss_g = float(np.clip(T_ss_g,
                            lo[0] if not np.isinf(lo[0]) else T_ss_g - 20,
                            hi[0] if not np.isinf(hi[0]) else T_ss_g + 20))

    def model(t_arr, T_ss, tau):
        return single_exp_model(t_arr, T_ss, tau, T0)

    try:
        popt, pcov = curve_fit(model, t, temps, p0=[T_ss_g, tau_g],
                               sigma=sems, absolute_sigma=True,
                               bounds=(lo, hi), maxfev=50000)
    except RuntimeError as exc:
        print("  WARNING: single-exp fit failed: {}".format(exc))
        return None, None, None, None

    T_ss_f, tau_f = popt
    errs   = np.sqrt(np.diag(pcov))
    T_pred = model(t, *popt)

    ss_res = float(np.sum((temps - T_pred) ** 2))
    ss_tot = float(np.sum((temps - temps.mean()) ** 2))
    r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    chi2   = float(np.sum(((temps - T_pred) / sems) ** 2)) / max(1, len(temps) - 2)

    params = {
        "model":    "single",
        "direction": direction,
        "T0":       T0,
        "tau":      tau_f,      "tau_err":  errs[1],
        "T_ss":     T_ss_f,     "T_ss_err": errs[0],
    }
    return params, r2, chi2, T_pred

def detect_kink(times, temps, sems, single_T_pred):
    residuals = temps - single_T_pred
    noise     = float(np.median(sems))

    win      = max(5, len(residuals) // 7)
    smoothed = uniform_filter1d(residuals, size=win)

    signs        = np.sign(smoothed)
    sign_changes = int(np.sum(np.diff(signs) != 0))
    peak_amp     = float(np.max(np.abs(smoothed)))
    snr          = peak_amp / noise if noise > 0 else 0.0

    has_kink = (snr > KINK_SNR_THR) and (sign_changes >= KINK_SC_THR)
    return has_kink, snr, sign_changes

def two_seg_model(t, t_kink, tau1, T_ss1, tau2, T_ss2, T0):
    T_k  = T_ss1 + (T0 - T_ss1) * np.exp(-t_kink / tau1)
    seg1 = T_ss1 + (T0 - T_ss1) * np.exp(-t / tau1)
    seg2 = T_ss2 + (T_k - T_ss2) * np.exp(-(t - t_kink) / tau2)
    return np.where(t <= t_kink, seg1, seg2)


def fit_two_segment(times, temps, sems, direction):
    T0  = float(temps[0])
    t   = times - times[0]
    dur = float(t[-1])

    tail_n  = max(10, len(temps) // 5)
    T_ss2_g = float(np.median(temps[-tail_n:]))

    t_kink_lo = max(dur * 0.02, MIN_SEG1_S)
    t_kink_hi = min(dur * 0.80, float(MAX_SEG1_S)) if MAX_SEG1_S else dur * 0.80

    if t_kink_lo >= t_kink_hi:
        print("  WARNING: MIN/MAX_SEG1_S conflict. Relaxing to 10%/80%.")
        t_kink_lo, t_kink_hi = dur * 0.10, dur * 0.80

    if direction == "heating":
        lo = [1.0,  T0 - 2.0,          3.0, T0 - 2.0,          t_kink_lo]
        hi = [800., T_ss2_g + 15.0, 2000., T_ss2_g + 15.0, t_kink_hi]
    else:
        data_ceil = max(T0 + 2.0, float(np.max(temps)) + 1.0)
        lo = [1.0, -np.inf, 3.0, -np.inf, t_kink_lo]
        hi = [800., data_ceil, 2000., data_ceil, t_kink_hi]

    def model_fit(t_arr, tau1, T_ss1, tau2, T_ss2, t_kink):
        return two_seg_model(t_arr, t_kink, tau1, T_ss1, tau2, T_ss2, T0)

    best_sse, best_popt, best_pcov = np.inf, None, None

    for frac in [0.04, 0.08, 0.13, 0.19, 0.27, 0.37, 0.48]:
        t_kg = dur * frac
        if t_kg < MIN_SEG1_S:
            continue
        if MAX_SEG1_S and t_kg > float(MAX_SEG1_S):
            continue
        t_kg = max(t_kink_lo + 0.1, min(t_kink_hi - 0.1, t_kg))

        T_k_g  = float(np.interp(t_kg, t, temps))
        tau1_g = float(np.clip(max(2.0, t_kg / 2.0), lo[0], hi[0]))
        tau2_g = float(np.clip(max(10.0, (dur - t_kg) / 2.0), lo[2], hi[2]))
        T_ss1_g = float(np.clip(T_k_g,
                                 lo[1] if not np.isinf(lo[1]) else T_k_g - 20, hi[1]))
        T_ss2_c = float(np.clip(T_ss2_g,
                                 lo[3] if not np.isinf(lo[3]) else T_ss2_g - 20, hi[3]))
        p0 = [tau1_g, T_ss1_g, tau2_g, T_ss2_c, t_kg]

        try:
            popt, pcov = curve_fit(model_fit, t, temps, p0=p0,
                                   sigma=sems, absolute_sigma=True,
                                   maxfev=50000, bounds=(lo, hi))
            sse = float(np.sum((temps - model_fit(t, *popt)) ** 2))
            if sse < best_sse:
                best_sse, best_popt, best_pcov = sse, popt, pcov
        except Exception:
            pass

    if best_popt is None:
        print("  WARNING: all grid candidates failed. Falling back to midpoint.")
        t_kg = max(MIN_SEG1_S + 1.0, dur * 0.30)
        T_k_g = float(np.interp(t_kg, t, temps))
        p0 = [max(2., t_kg/2.), T_k_g, max(10., (dur-t_kg)/2.), T_ss2_g, t_kg]
        try:
            best_popt, best_pcov = curve_fit(model_fit, t, temps, p0=p0,
                                             sigma=sems, absolute_sigma=True,
                                             maxfev=200000, bounds=(lo, hi))
        except RuntimeError as exc:
            print("  FATAL: two-segment fit failed: {}".format(exc)); sys.exit(1)

    tau1_f, T_ss1_f, tau2_f, T_ss2_f, t_kink_f = best_popt
    errs  = np.sqrt(np.diag(best_pcov))
    T_k_f = T_ss1_f + (T0 - T_ss1_f) * np.exp(-t_kink_f / tau1_f)
    T_pred = model_fit(t, *best_popt)

    ss_res = float(np.sum((temps - T_pred) ** 2))
    ss_tot = float(np.sum((temps - temps.mean()) ** 2))
    r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    chi2   = float(np.sum(((temps - T_pred) / sems) ** 2)) / max(1, len(temps) - 5)

    print("  Two-seg: best t_kink = {:.1f} s  (SSE = {:.4f})".format(
        t_kink_f + times[0], best_sse))

    params = {
        "model":      "two_segment",
        "direction":   direction,
        "T0":          T0,
        "T_k":         T_k_f,
        "tau1":        tau1_f,    "tau1_err":   errs[0],
        "T_ss1":       T_ss1_f,   "T_ss1_err":  errs[1],
        "tau2":        tau2_f,    "tau2_err":   errs[2],
        "T_ss2":       T_ss2_f,   "T_ss2_err":  errs[3],
        "t_kink":      t_kink_f + times[0],
        "t_kink_err":  errs[4],
    }
    return params, r2, chi2, T_pred, (t <= t_kink_f), ~(t <= t_kink_f)

def fit_data(times, temps, sems, direction):
    print("  [1/2] Fitting single exponential...")
    s_params, s_r2, s_chi2, s_Tpred = fit_single(times, temps, sems, direction)

    if s_params is None:
        print("  Single-exp failed. Routing to two-segment.")
        return (*fit_two_segment(times, temps, sems, direction),)

    has_kink, snr, sc = detect_kink(times, temps, sems, s_Tpred)

    print("  Kink detector: SNR={:.2f} (thr={})  sign_changes={} (thr={})  ->  {}".format(
        snr, KINK_SNR_THR, sc, KINK_SC_THR,
        "KINK DETECTED" if has_kink else "NO KINK"))

    if not has_kink:
        print("  [2/2] Single-exponential selected.")
        s_params["kink_snr"] = snr
        s_params["kink_sc"]  = sc
        return s_params, s_r2, s_chi2, s_Tpred, None, None

    print("  [2/2] Fitting two-segment model...")
    params, r2, chi2, T_pred, m1, m2 = fit_two_segment(times, temps, sems, direction)
    if params is None:
        print("  Two-segment failed. Keeping single-exp result.")
        s_params["kink_snr"] = snr
        s_params["kink_sc"]  = sc
        return s_params, s_r2, s_chi2, s_Tpred, None, None
    params["kink_snr"] = snr
    params["kink_sc"]  = sc
    return params, r2, chi2, T_pred, m1, m2

def save_report(out_path, filename, params, r2, chi2,
                times, temps, T_pred, seg1_mask, seg2_mask, file_type):
    model     = params["model"]
    direction = params["direction"]
    sep  = "=" * 66
    lines = []

    lines += [sep,
              "  TEC {} FIT REPORT".format(direction.upper()),
              "  Model     : {}".format(
                  "Single Exponential"
                  if model == "single"
                  else "Two-Segment Exponential  [kink detected]"),
              "  Generated : {}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
              "  Source    : {}".format(Path(filename).name),
              "  Data Type : {}".format(file_type.replace("_", " ")), sep]

    lines += ["  KINK DETECTION",
              "  Residual SNR     : {:.2f}  (threshold {})".format(
                  params.get("kink_snr", 0), KINK_SNR_THR),
              "  Sign changes     : {}  (threshold {})".format(
                  params.get("kink_sc", 0), KINK_SC_THR),
              "  Decision         : {}".format(
                  "Kink confirmed -- two-segment model used"
                  if model == "two_segment"
                  else "No kink -- single exponential used"), sep]

    if model == "single":
        T0, tau, T_ss = params["T0"], params["tau"], params["T_ss"]
        lines += ["  MODEL: T(t) = T_ss + (T0 - T_ss)*exp(-t/tau)", sep,
                  "  PARAMETERS",
                  "  T0               : {:.4f} degC".format(T0),
                  "  tau              : {:.3f} +/- {:.3f} s  ({:.3f} min)".format(
                      tau, params["tau_err"], tau / 60.0),
                  "  T_ss (final)     : {:.4f} +/- {:.4f} degC".format(
                      T_ss, params["T_ss_err"]), "",
                  "  R squared        : {:.6f}".format(r2),
                  "  Reduced chi2     : {:.3f}".format(chi2), sep,
                  "  MILESTONES",
                  "  {:<8} {:<14} {:<14} {}".format("n*tau","Time (s)","% done","Temp (degC)"),
                  "  " + "-" * 50]
        for n in [1, 2, 3, 5]:
            t_n = n * tau
            T_n = T_ss + (T0 - T_ss) * math.exp(-n)
            lines.append("  {:<8} {:<14.1f} {:<14.1f} {:.4f}".format(
                "{}*tau".format(n), t_n, (1 - math.exp(-n)) * 100, T_n))
    else:
        tau1, tau2   = params["tau1"], params["tau2"]
        T_ss1, T_ss2 = params["T_ss1"], params["T_ss2"]
        t_kink, T_k  = params["t_kink"], params["T_k"]
        T0           = params["T0"]
        lines += ["  MODEL",
                  "    Seg 1 [t0->t_kink]:  T(t) = T_ss1 + (T0-T_ss1)*exp(-t/tau1)",
                  "    Seg 2 [t_kink->end]: T(t) = T_ss2 + (T_k-T_ss2)*exp(-(t-t_kink)/tau2)",
                  "    Continuity:          T_k = Seg1(t_kink)  [not free]", sep,
                  "  PARAMETERS",
                  "  T0               : {:.4f} degC".format(T0),
                  "  t_kink           : {:.2f} +/- {:.2f} s".format(
                      t_kink, params["t_kink_err"]),
                  "  T_k (continuity) : {:.4f} degC".format(T_k), "",
                  "  --- Segment 1 ---",
                  "  tau1             : {:.3f} +/- {:.3f} s  ({:.3f} min)".format(
                      tau1, params["tau1_err"], tau1 / 60.0),
                  "  T_ss1 (intermed) : {:.4f} +/- {:.4f} degC".format(
                      T_ss1, params["T_ss1_err"]), "",
                  "  --- Segment 2 ---",
                  "  tau2             : {:.3f} +/- {:.3f} s  ({:.3f} min)".format(
                      tau2, params["tau2_err"], tau2 / 60.0),
                  "  T_ss2 (final)    : {:.4f} +/- {:.4f} degC".format(
                      T_ss2, params["T_ss2_err"]), "",
                  "  R squared        : {:.6f}".format(r2),
                  "  Reduced chi2     : {:.3f}".format(chi2), sep,
                  "  MILESTONES  (Segment 2 -- post-kink)",
                  "  {:<9} {:<14} {:<14} {}".format(
                      "n*tau2","Time (s)","% done","Temp (degC)"),
                  "  " + "-" * 52]
        for n in [1, 2, 3, 5]:
            t_n = t_kink + n * tau2
            T_n = T_ss2 + (T_k - T_ss2) * math.exp(-n)
            lines.append("  {:<9} {:<14.1f} {:<14.1f} {:.4f}".format(
                "{}*tau2".format(n), t_n, (1 - math.exp(-n)) * 100, T_n))

    lines += [sep, "DATA TABLE", sep]
    if model == "single":
        lines.append("{:<12}{:<16}{:<16}{}".format(
            "time_s","measured_C","fitted_C","residual_C"))
        lines.append("-" * 56)
        for i in range(len(times)):
            lines.append("{:<12.3f}{:<16.6f}{:<16.6f}{:.6f}".format(
                times[i], temps[i], T_pred[i], temps[i] - T_pred[i]))
    else:
        lines.append("{:<12}{:<16}{:<16}{:<10}{}".format(
            "time_s","measured_C","fitted_C","segment","residual_C"))
        lines.append("-" * 66)
        for i in range(len(times)):
            seg = "1" if seg1_mask[i] else "2"
            lines.append("{:<12.3f}{:<16.6f}{:<16.6f}{:<10}{:.6f}".format(
                times[i], temps[i], T_pred[i], seg, temps[i] - T_pred[i]))

    text = "\n".join(lines)
    print(text)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    print("\n  OK  Report -> {}".format(out_path))

def save_tau_summary(out_path, filename, params, r2, chi2):
    model     = params["model"]
    direction = params["direction"]
    dir_label = "FALLING" if direction == "cooling" else "RISING"
    tec_state = "OFF"     if direction == "cooling" else "ON"

    with open(out_path, "a", encoding="utf-8") as fh:
        fh.write("\n" + "=" * 52 + "\n")
        fh.write("  Source      : {}\n".format(Path(filename).name))
        fh.write("=" * 52 + "\n")
        fh.write("  Direction   : {}  (TEC {})\n".format(dir_label, tec_state))
        fh.write("  Kink SNR    : {:.2f}  (thr {})\n".format(
            params.get("kink_snr", 0), KINK_SNR_THR))

        if model == "single":
            fh.write("  Model       : Single Exponential\n")
            fh.write("  T0          : {:.4f} C\n".format(params["T0"]))
            fh.write("  tau         : {:.4f} s  ({:.4f} min)\n".format(
                params["tau"], params["tau"] / 60.0))
            fh.write("  tau_err     : +/- {:.4f} s\n".format(params["tau_err"]))
            fh.write("  T_ss        : {:.4f} +/- {:.4f} C\n".format(
                params["T_ss"], params["T_ss_err"]))
            fh.write("  R2          : {:.6f}\n".format(r2))
            fh.write("  chi2_red    : {:.4f}\n".format(chi2))
            fh.write("-" * 52 + "\n")
            fh.write("  MILESTONES\n")
            fh.write("  {:<10}{:<14}{:<10}{}\n".format(
                "n*tau","Time (s)","% done","Temp (C)"))
            T0, tau, T_ss = params["T0"], params["tau"], params["T_ss"]
            for n in [1, 2, 3, 5]:
                fh.write("  {:<10}{:<14.2f}{:<10.1f}{:.4f}\n".format(
                    "{}tau".format(n), n * tau,
                    (1 - math.exp(-n)) * 100,
                    T_ss + (T0 - T_ss) * math.exp(-n)))
        else:
            tau1, tau2   = params["tau1"], params["tau2"]
            T_ss1, T_ss2 = params["T_ss1"], params["T_ss2"]
            t_kink, T_k  = params["t_kink"], params["T_k"]
            T0           = params["T0"]
            fh.write("  Model       : Two-Segment Exponential\n")
            fh.write("  T0          : {:.4f} C\n".format(T0))
            fh.write("  t_kink      : {:.2f} +/- {:.2f} s\n".format(
                t_kink, params["t_kink_err"]))
            fh.write("  T_k         : {:.4f} C\n".format(T_k))
            fh.write("  tau1        : {:.4f} s  ({:.4f} min)\n".format(tau1, tau1/60.))
            fh.write("  tau1_err    : +/- {:.4f} s\n".format(params["tau1_err"]))
            fh.write("  T_ss1       : {:.4f} +/- {:.4f} C\n".format(
                T_ss1, params["T_ss1_err"]))
            fh.write("  tau2        : {:.4f} s  ({:.4f} min)\n".format(tau2, tau2/60.))
            fh.write("  tau2_err    : +/- {:.4f} s\n".format(params["tau2_err"]))
            fh.write("  T_ss2       : {:.4f} +/- {:.4f} C\n".format(
                T_ss2, params["T_ss2_err"]))
            fh.write("  R2          : {:.6f}\n".format(r2))
            fh.write("  chi2_red    : {:.4f}\n".format(chi2))
            fh.write("-" * 52 + "\n")
            fh.write("  MILESTONES  (Seg 2 post-kink)\n")
            fh.write("  {:<10}{:<14}{:<10}{}\n".format(
                "n*tau2","Time (s)","% done","Temp (C)"))
            for n in [1, 2, 3, 5]:
                fh.write("  {:<10}{:<14.2f}{:<10.1f}{:.4f}\n".format(
                    "{}tau2".format(n), t_kink + n * tau2,
                    (1 - math.exp(-n)) * 100,
                    T_ss2 + (T_k - T_ss2) * math.exp(-n)))
        fh.write("=" * 52 + "\n")
    print("  OK  Summary -> {}".format(out_path))

DARK, GRID_C = "#0f1117", "#1e2130"
TEXT, SEG1_C, SEG2_C = "#c8cdd8", "#4c9be8", "#f5a623"
FIT_C, KINK_C = "#e85c5c", "#b07fec"

def _style(ax, title=""):
    ax.set_facecolor(DARK)
    ax.tick_params(colors=TEXT, labelsize=9)
    for sp in ax.spines.values(): sp.set_color(GRID_C)
    ax.yaxis.label.set_color(TEXT); ax.xaxis.label.set_color(TEXT)
    ax.grid(True, color=GRID_C, linewidth=0.6, linestyle="--")
    if title: ax.set_title(title, color=TEXT, fontsize=10.5, pad=5)

def _open(path):
    try:
        import subprocess, platform
        if platform.system() == "Darwin":    subprocess.Popen(["open", path])
        elif platform.system() == "Windows": os.startfile(path)
        else:                                subprocess.Popen(["xdg-open", path])
    except Exception: pass

def _init_mpl():
    import matplotlib
    for b in ["MacOSX", "TkAgg", "Qt5Agg", "Agg"]:
        try:
            matplotlib.use(b)
            import matplotlib.pyplot as plt
            plt.figure(); plt.close(); return plt
        except Exception: continue
    import matplotlib.pyplot as plt
    return plt

def plot_single(times, temps, sems, T_pred, params, r2, chi2,
                filename, out_path, file_type):
    plt = _init_mpl()
    import matplotlib.gridspec as gridspec

    direction = params["direction"]
    T0, tau, T_ss = params["T0"], params["tau"], params["T_ss"]
    DATA_C = SEG2_C if direction == "heating" else SEG1_C
    residuals = temps - T_pred

    fig = plt.figure(figsize=(11, 10))
    fig.patch.set_facecolor(DARK)
    gs = gridspec.GridSpec(3, 1, figure=fig,
                           height_ratios=[2.5, 0.9, 1.0], hspace=0.28)
    ax_main = fig.add_subplot(gs[0])
    ax_rate = fig.add_subplot(gs[1], sharex=ax_main)
    ax_res  = fig.add_subplot(gs[2], sharex=ax_main)

    _style(ax_main, "1. Measured Data & Single Exponential Fit")

    if file_type == "noise_cancelled":
        ax_main.fill_between(times, temps - sems, temps + sems,
                             alpha=0.15, color=DATA_C, label=r"$\pm$1 SEM")
    ax_main.scatter(times, temps, s=10, color=DATA_C, alpha=0.70,
                    label="Measured", zorder=3)
    ax_main.plot(times, T_pred, color=FIT_C, lw=2.2,
                 label="Single exp fit", zorder=4)
    ax_main.axhline(T_ss, color="#5dd8a0", lw=1.1, linestyle="--", alpha=0.7,
                    label="$T_{{ss}}={:.2f}°C$".format(T_ss))
    ax_main.axhline(0.0, color=TEXT, lw=0.7, linestyle=":", alpha=0.35,
                    label="0°C")
    for n, col in [(1,"#5dd8a0"),(2,"#a8e6cf"),(3,"#6bcbaa")]:
        t_n = n * tau + times[0]
        if t_n < times[-1]:
            ax_main.axvline(t_n, color=col, lw=0.8, linestyle="--", alpha=0.45)
            ax_main.text(t_n + 1.5, T0 + (T_ss - T0) * 0.05,
                         "{}$\\tau$".format(n), color=col, fontsize=7.5)

    box = ["tau    = {:7.2f} +/- {:.2f} s".format(tau, params["tau_err"]),
           "       = {:7.3f} +/- {:.3f} min".format(
               tau/60., params["tau_err"]/60.),
           "T_ss   = {:7.3f} +/- {:.4f} degC".format(T_ss, params["T_ss_err"]),
           "Kink SNR = {:.2f}  (< {} = no kink)".format(
               params.get("kink_snr",0), KINK_SNR_THR),
           "R2     = {:.5f}".format(r2),
           "chi2_r = {:.3f}".format(chi2)]
    ax_main.text(0.01, 0.98, "\n".join(box), transform=ax_main.transAxes,
                 fontsize=7.8, verticalalignment="top", family="monospace",
                 color=TEXT, bbox=dict(boxstyle="round,pad=0.4",
                 facecolor=GRID_C, edgecolor="#5dd8a0", alpha=0.85))

    loc = "lower right" if direction == "heating" else "upper right"
    ax_main.set_ylabel("Temperature (°C)", color=TEXT)
    ax_main.legend(fontsize=8, loc=loc, framealpha=0.4,
                   facecolor=DARK, edgecolor=GRID_C, labelcolor=TEXT, ncol=2)

    _style(ax_rate, "2. Heating/Cooling Rate  dT/dt  (°C/s)")
    if len(times) > 1:
        rate = np.diff(temps) / np.diff(times)
        t_mid = 0.5 * (times[:-1] + times[1:])
        ax_rate.plot(t_mid, rate, color=DATA_C, lw=1.2, alpha=0.75)
        ax_rate.axhline(0, color=TEXT, lw=0.7, alpha=0.35)
    ax_rate.set_ylabel("dT/dt (°C/s)", color=TEXT)

    _style(ax_res, "3. Fit Residuals (Measured − Fitted)")
    bar_w = float(np.mean(np.diff(times))) * 0.75
    ax_res.bar(times, residuals, width=bar_w,
               color=["#5dd8a0" if r >= 0 else FIT_C for r in residuals], alpha=0.65)
    ax_res.axhline(0, color=TEXT, lw=0.8)
    std_res = float(np.std(residuals))
    ax_res.axhline( std_res, color=DATA_C, lw=0.8, linestyle=":", alpha=0.6)
    ax_res.axhline(-std_res, color=DATA_C, lw=0.8, linestyle=":", alpha=0.6,
                   label="$\\pm1\\sigma_{{res}}=\\pm{:.4f}°C$".format(std_res))
    ax_res.set_ylabel("Residual (°C)", color=TEXT)
    ax_res.set_xlabel("Time (seconds)", color=TEXT)
    ax_res.legend(fontsize=8, loc="upper right", framealpha=0.4,
                  facecolor=DARK, edgecolor=GRID_C, labelcolor=TEXT)

    fig.suptitle(
        "TEC {}  --  Single Exponential Fit  [Kink SNR={:.1f} < {}]\n"
        "tau = {:.2f} +/- {:.2f} s  ({:.3f} min)  |  R2 = {:.5f}\n"
        "Source: {}".format(direction.upper(),
            params.get("kink_snr",0), KINK_SNR_THR,
            tau, params["tau_err"], tau/60., r2,
            Path(filename).name),
        color=TEXT, fontsize=10.5, y=0.99)

    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=DARK)
    print("  OK  Plot -> {}".format(out_path))
    _open(out_path)
    try: plt.show(block=True)
    except Exception: pass
    plt.close()

def plot_two_segment(times, temps, sems, T_pred, seg1_mask, seg2_mask,
                     params, r2, chi2, filename, out_path, file_type):
    plt = _init_mpl()
    import matplotlib.gridspec as gridspec

    direction = params["direction"]
    tau1, tau2   = params["tau1"], params["tau2"]
    T_ss1, T_ss2 = params["T_ss1"], params["T_ss2"]
    t_kink, T_k  = params["t_kink"], params["T_k"]
    T0           = params["T0"]
    DATA_C       = SEG2_C if direction == "heating" else SEG1_C
    residuals    = temps - T_pred

    fig = plt.figure(figsize=(12, 13))
    fig.patch.set_facecolor(DARK)
    gs = gridspec.GridSpec(4, 1, figure=fig,
                           height_ratios=[2.8, 0.9, 0.9, 1.0], hspace=0.26)
    ax_main = fig.add_subplot(gs[0])
    ax_seg  = fig.add_subplot(gs[1], sharex=ax_main)
    ax_rate = fig.add_subplot(gs[2], sharex=ax_main)
    ax_res  = fig.add_subplot(gs[3], sharex=ax_main)

    _style(ax_main, "1. Measured Data & Two-Segment Exponential Fit")
    if file_type == "noise_cancelled":
        ax_main.fill_between(times, temps-sems, temps+sems,
                             alpha=0.15, color=DATA_C, label=r"$\pm$1 SEM")
    ax_main.scatter(times, temps, s=10, color=DATA_C, alpha=0.70,
                    label="Measured", zorder=3)
    ax_main.plot(times[seg1_mask], T_pred[seg1_mask], color=SEG1_C,
                 lw=2.0, alpha=0.55, linestyle="--",
                 label="Seg 1  $\\tau_1={:.1f}$ s".format(tau1))
    ax_main.plot(times[seg2_mask], T_pred[seg2_mask], color=SEG2_C,
                 lw=2.0, alpha=0.55, linestyle="--",
                 label="Seg 2  $\\tau_2={:.1f}$ s".format(tau2))
    ax_main.plot(times, T_pred, color=FIT_C, lw=2.0,
                 label="Combined fit", zorder=4)
    ax_main.axhline(T_ss2, color=SEG2_C, lw=1.1, linestyle="--", alpha=0.7,
                    label="$T_{{ss2}}={:.2f}°C$".format(T_ss2))
    ax_main.axhline(T_ss1, color=SEG1_C, lw=0.9, linestyle=":", alpha=0.5,
                    label="$T_{{ss1}}={:.2f}°C$".format(T_ss1))
    ax_main.axhline(0.0, color=TEXT, lw=0.7, linestyle=":", alpha=0.35,
                    label="0°C")
    ax_main.axvline(t_kink, color=KINK_C, lw=1.8, linestyle=":",
                    label="$t_{{kink}}={:.1f}$ s".format(t_kink))
    ax_main.scatter([t_kink], [T_k], color=KINK_C, s=65, zorder=6,
                    label="$T_k={:.2f}°C$".format(T_k))
    for n, col in [(1,"#5dd8a0"),(2,"#a8e6cf"),(3,"#6bcbaa")]:
        t_n = t_kink + n * tau2
        if t_n < times[-1]:
            ax_main.axvline(t_n, color=col, lw=0.8, linestyle="--", alpha=0.45)
            ax_main.text(t_n+1.5, T0+(T_ss2-T0)*0.05,
                         "{}$\\tau_2$".format(n), color=col, fontsize=7.5)

    box = ["tau1   = {:7.2f} +/- {:.2f} s".format(tau1, params["tau1_err"]),
           "tau2   = {:7.2f} +/- {:.2f} s".format(tau2, params["tau2_err"]),
           "       = {:7.3f} +/- {:.3f} min".format(tau2/60.,params["tau2_err"]/60.),
           "T_ss1  = {:7.3f} +/- {:.4f} degC".format(T_ss1,params["T_ss1_err"]),
           "T_ss2  = {:7.3f} +/- {:.4f} degC".format(T_ss2,params["T_ss2_err"]),
           "t_kink = {:7.1f} +/- {:.2f} s".format(t_kink,params["t_kink_err"]),
           "T_k    = {:7.3f} degC  [continuity]".format(T_k),
           "Kink SNR = {:.2f}  (>= {} confirmed)".format(
               params.get("kink_snr",0), KINK_SNR_THR),
           "R2     = {:.5f}".format(r2),
           "chi2_r = {:.3f}".format(chi2)]
    ax_main.text(0.01, 0.98, "\n".join(box), transform=ax_main.transAxes,
                 fontsize=7.8, verticalalignment="top", family="monospace",
                 color=TEXT, bbox=dict(boxstyle="round,pad=0.4",
                 facecolor=GRID_C, edgecolor=KINK_C, alpha=0.85))

    loc = "lower right" if direction == "heating" else "upper right"
    ax_main.set_ylabel("Temperature (°C)", color=TEXT)
    ax_main.legend(fontsize=7.5, loc=loc, framealpha=0.4,
                   facecolor=DARK, edgecolor=GRID_C, labelcolor=TEXT, ncol=2)

    _style(ax_seg, "2. Segment Assignment  [Blue=Seg1 | Amber=Seg2 | Purple=t_kink]")
    ones = np.ones(len(times))
    ax_seg.fill_between(times, 0, ones, where=seg1_mask,
                        color=SEG1_C, alpha=0.40, label="Segment 1")
    ax_seg.fill_between(times, 0, ones, where=seg2_mask,
                        color=SEG2_C, alpha=0.40, label="Segment 2")
    ax_seg.axvline(t_kink, color=KINK_C, lw=1.5, linestyle=":")
    ax_seg.set_ylim(-0.05, 1.25); ax_seg.set_yticks([])
    ax_seg.legend(fontsize=8, loc="center right", framealpha=0.4,
                  facecolor=DARK, edgecolor=GRID_C, labelcolor=TEXT)

    _style(ax_rate, "3. Heating/Cooling Rate  dT/dt  (°C/s)")
    if len(times) > 1:
        rate = np.diff(temps) / np.diff(times)
        t_mid = 0.5 * (times[:-1] + times[1:])
        ax_rate.plot(t_mid, rate, color=DATA_C, lw=1.2, alpha=0.75)
        ax_rate.axhline(0, color=TEXT, lw=0.7, alpha=0.35)
        ax_rate.axvline(t_kink, color=KINK_C, lw=1.5, linestyle=":")
    ax_rate.set_ylabel("dT/dt (°C/s)", color=TEXT)

    _style(ax_res, "4. Fit Residuals (Measured − Fitted)")
    bar_w = float(np.mean(np.diff(times))) * 0.75
    ax_res.bar(times, residuals, width=bar_w,
               color=["#5dd8a0" if r >= 0 else FIT_C for r in residuals], alpha=0.65)
    ax_res.axhline(0, color=TEXT, lw=0.8)
    std_res = float(np.std(residuals))
    ax_res.axhline( std_res, color=DATA_C, lw=0.8, linestyle=":", alpha=0.6)
    ax_res.axhline(-std_res, color=DATA_C, lw=0.8, linestyle=":", alpha=0.6,
                   label="$\\pm1\\sigma_{{res}}=\\pm{:.4f}°C$".format(std_res))
    ax_res.axvline(t_kink, color=KINK_C, lw=1.2, linestyle=":")
    ax_res.set_ylabel("Residual (°C)", color=TEXT)
    ax_res.set_xlabel("Time (seconds)", color=TEXT)
    ax_res.legend(fontsize=8, loc="upper right", framealpha=0.4,
                  facecolor=DARK, edgecolor=GRID_C, labelcolor=TEXT)

    fig.suptitle(
        "TEC {}  --  Two-Segment Exponential Fit  [Kink SNR={:.1f}]\n"
        "tau1={:.2f}+/-{:.2f}s  |  tau2={:.2f}+/-{:.2f}s ({:.3f}min)  |  R2={:.5f}\n"
        "Source: {}".format(direction.upper(), params.get("kink_snr",0),
            tau1, params["tau1_err"],
            tau2, params["tau2_err"], tau2/60., r2,
            Path(filename).name),
        color=TEXT, fontsize=10.5, y=0.988)

    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=DARK)
    print("  OK  Plot -> {}".format(out_path))
    _open(out_path)
    try: plt.show(block=True)
    except Exception: pass
    plt.close()

def main(argv):
    if len(argv) >= 1 and os.path.isfile(argv[0]):
        filename = argv[0]
    else:
        print("Opening file picker..."); filename = pick_file()

    print("\nProcessing: {}".format(filename))
    times, temps, sems, file_type = load_data(filename)
    direction = detect_direction(temps)
    print("Direction: {}".format(direction))

    params, r2, chi2, T_pred, seg1_mask, seg2_mask = fit_data(
        times, temps, sems, direction)

    stem    = Path(filename).stem
    out_dir = Path(__file__).resolve().parent

    save_report(str(out_dir / "fit_results_{}.txt".format(stem)),
                filename, params, r2, chi2,
                times, temps, T_pred, seg1_mask, seg2_mask, file_type)
    save_tau_summary(str(out_dir / "tau_summary_{}.txt".format(stem)),
                     filename, params, r2, chi2)

    plot_path = str(out_dir / "fit_plot_{}.png".format(stem))
    if params["model"] == "single":
        plot_single(times, temps, sems, T_pred, params, r2, chi2,
                    filename, plot_path, file_type)
    else:
        plot_two_segment(times, temps, sems, T_pred, seg1_mask, seg2_mask,
                         params, r2, chi2, filename, plot_path, file_type)


if __name__ == "__main__":
    main(sys.argv[1:])
