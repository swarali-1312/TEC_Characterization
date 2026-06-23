# TEC Thermal Characterization Pipeline

**Collaborative project at ISRO · Python · SciPy · NumPy · Matplotlib**  
**Instrument:** Newport 3700 Series Temperature Controller

---

## Overview

This pipeline processes temperature–time data from thermoelectric cooler (TEC) characterization experiments. It automates two tasks that are otherwise done by hand: noise reduction across repeated measurement runs, and curve fitting to extract thermal time constants.

The two scripts are designed to run in sequence — noise cancellation first, curve fitting second — though each can also be used independently on raw data.

---

## Scripts

### `tec_noise_cancel.py` — Phase-Fold Ensemble Averaging

Reduces measurement noise by stacking multiple repeated cooling or heating runs, inspired by the phase-folding technique used in exoplanet transit photometry.

**Method:**
1. Detect the *ingress point* in each run — the moment temperature crosses a threshold on the falling (or rising) edge, analogous to a planet crossing the stellar limb.
2. Phase-fold all runs onto a common time axis anchored at that crossing, using sub-sample linear interpolation to avoid timing smear.
3. Compute a point-by-point ensemble average across all aligned runs. Random noise cancels; the coherent thermal curve survives.

For N runs, the theoretical noise reduction is 1/√N.

**Outputs:**
- `noise_reduced_*_binned.txt` — binned averaged curve (time, mean temp, std, SEM) ready for curve fitting
- `noise_reduced_*_report.txt` — per-run noise statistics and noise reduction summary
- `noise_reduced_*.png` — 3-panel figure: raw runs, stacked mean ± 1σ, binned curve

**Usage:**
```bash
python tec_noise_cancel.py run0.txt run1.txt run2.txt --bins 100
```
Or double-click to use the GUI file picker.

---

### `tec_curve_fit2.py` — Auto-Routing Exponential Curve Fit

Fits temperature–time data to either a single exponential or a piecewise two-segment exponential model, selected automatically by analysing the residuals of an initial single-exp fit.

**Model selection logic:**
1. Fit a single exponential to the full dataset.
2. Smooth the residuals with a uniform moving average (window = 1/7 of dataset length).
3. Compute two diagnostics on the smoothed residual:
   - **SNR** = peak amplitude / noise floor
   - **Sign changes** = number of zero-crossings in the smoothed residual
4. If SNR > 3 **and** sign changes ≥ 2 → genuine two-stage response → fit two-segment model. Otherwise → single exponential is adequate.

**Single-segment model** (no kink):

$$T(t) = T_{ss} + (T_0 - T_{ss})\, e^{-t/\tau}$$

Free parameters: τ, T_ss (T₀ fixed to first data point).

**Two-segment model** (kink detected):

$$\text{Seg 1:}\quad T(t) = T_{ss1} + (T_0 - T_{ss1})\, e^{-t/\tau_1}$$

$$\text{Seg 2:}\quad T(t) = T_{ss2} + (T_k - T_{ss2})\, e^{-(t - t_{kink})/\tau_2}$$

Continuity is enforced at the kink: T_k = Seg1(t_kink) is not a free parameter.

Free parameters: τ₁, T_ss1, τ₂, T_ss2, t_kink (5 total). Grid-searched over 7 kink candidates to avoid local minima.

**Outputs:**
- `fit_results_*.txt` — full parameter table with uncertainties, goodness-of-fit metrics (R², χ²_r), and thermal milestone table (1τ, 2τ, 3τ, 5τ)
- `tau_summary_*.txt` — compact summary for logging across multiple runs
- `fit_plot_*.png` — multi-panel figure: data + fit, dT/dt rate, residuals (+ segment assignment panel for two-segment fits)

**Usage:**
```bash
python tec_curve_fit2.py data.txt
```
Or run without arguments to use the GUI file picker. Accepts both raw Newport 3700 files and noise-cancelled outputs.

---

## Pipeline

```
Newport 3700 raw files (.txt)
        │
        ▼
tec_noise_cancel.py      ← stack N repeated runs, reduce noise by 1/√N
        │
        ▼
noise_reduced_*_binned.txt
        │
        ▼
tec_curve_fit2.py        ← auto-detect model, fit, extract τ and T_ss
        │
        ▼
fit_results_*.txt  ·  fit_plot_*.png
```

---

## Dependencies

```
pip install numpy scipy matplotlib
```

Python 3.8+ required. No other packages needed for `tec_noise_cancel.py`.

---

## Notes

This pipeline was developed as part of a collaborative TEC characterization project at ISRO. The phase-folding approach in `tec_noise_cancel.py` was adapted from the transit stacking method used in exoplanet photometry (as implemented in the `lightkurve` library), applied here to the analogous problem of aligning repeated thermal transient measurements.
