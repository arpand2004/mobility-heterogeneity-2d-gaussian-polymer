from pathlib import Path
import numpy as np

# Data

data_file = Path("data/processed/block_com_msd_data.npz")
diffusion_csv_file = Path(
    "data/processed/block_com_diffusion_coefficients.csv"
)

rho_list = [1.0, 2.0, 4.0]

short_window = (1.0e-4, 0.1)

late_windows = {
    1.0: (1.0, 10.0),
    2.0: (3.0, 10.0),
    4.0: (5.0, 10.0),
}

local_half_width = 5
slope_tolerance = 0.10


def rho_key(rho):
    return str(float(rho)).replace(".", "p")


def sem(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) <= 1:
        return 0.0

    return np.std(values, ddof=1) / np.sqrt(len(values))


# Local log-log slope

def local_loglog_slope(t, msd, half_width=5):
    t = np.asarray(t, dtype=float)
    msd = np.asarray(msd, dtype=float)

    log_t = np.log(t)
    log_msd = np.log(msd)

    slopes = np.full(len(t), np.nan)

    for i in range(len(t)):
        lo = max(0, i - half_width)
        hi = min(len(t), i + half_width + 1)

        if hi - lo < 3:
            continue

        slope, _ = np.polyfit(
            log_t[lo:hi],
            log_msd[lo:hi],
            1,
        )

        slopes[i] = slope

    return slopes


def print_slope_window(label, t, slopes, window):
    tmin, tmax = window

    mask = (
        np.isfinite(slopes)
        & (t >= tmin)
        & (t <= tmax)
    )

    if np.count_nonzero(mask) == 0:
        print(f"  {label}: no points")
        return

    vals = slopes[mask]

    print(
        f"  {label:24s} "
        f"median slope = {np.median(vals):.3f}, "
        f"mean = {np.mean(vals):.3f}, "
        f"range = [{np.min(vals):.3f}, {np.max(vals):.3f}]"
    )


def print_diffusive_intervals(t, slopes, tolerance=0.10):
    good = (
        np.isfinite(slopes)
        & (np.abs(slopes - 1.0) <= tolerance)
    )

    intervals = []
    start = None

    for i, is_good in enumerate(good):
        if is_good and start is None:
            start = i

        if start is not None and (
            not is_good or i == len(good) - 1
        ):
            end = i if is_good else i - 1

            if end - start + 1 >= 3:
                intervals.append((t[start], t[end]))

            start = None

    if intervals:
        for lo, hi in intervals:
            print(f"    {lo:.4g} to {hi:.4g}")
    else:
        print("    none found")


# Diffusion coefficient extraction

def fit_D_per_run(t_sweep, t_scaled, msd_all, window):
    tmin, tmax = window

    mask = (
        np.isfinite(t_sweep)
        & np.isfinite(t_scaled)
        & (t_scaled >= tmin)
        & (t_scaled <= tmax)
    )

    if np.count_nonzero(mask) < 2:
        raise ValueError(f"Too few points in window {window}")

    D_values = []

    for msd in msd_all:
        valid = mask & np.isfinite(msd)

        if np.count_nonzero(valid) < 2:
            continue

        slope, _ = np.polyfit(
            t_sweep[valid],
            msd[valid],
            1,
        )

        D_values.append(slope / 4.0)

    return np.asarray(D_values)


def print_D_result(name, values):
    print(
        f"  {name:5s} = "
        f"{np.mean(values):.8e} +/- {sem(values):.2e}"
    )


# Main analysis

data = np.load(data_file)

print(f"\nLoaded: {data_file}")
print("=" * 72)

diffusion_rows = []

for rho in rho_list:
    key = rho_key(rho)

    t_sweep = data[f"t_sweep_rho_{key}"]
    t_scaled = data[f"t_scaled_rho_{key}"]

    A_mean = data[f"A_com_mean_rho_{key}"]
    B_mean = data[f"B_com_mean_rho_{key}"]
    full_mean = data[f"full_com_mean_rho_{key}"]

    A_all = data[f"A_com_all_rho_{key}"]
    B_all = data[f"B_com_all_rho_{key}"]
    full_all = data[f"full_com_all_rho_{key}"]

    print(f"\n\nrho = {rho:g}")
    print("=" * 72)

    slopes_A = local_loglog_slope(
        t_scaled,
        A_mean,
        local_half_width,
    )

    slopes_B = local_loglog_slope(
        t_scaled,
        B_mean,
        local_half_width,
    )

    slopes_full = local_loglog_slope(
        t_scaled,
        full_mean,
        local_half_width,
    )

    print("\nLOCAL LOG-LOG SLOPE ANALYSIS")

    for name, slopes in [
        ("block A", slopes_A),
        ("block B", slopes_B),
        ("full chain", slopes_full),
    ]:
        print(f"\n{name}")

        print_slope_window(
            "short [1e-4, 0.1]",
            t_scaled,
            slopes,
            short_window,
        )

        print_slope_window(
            "intermediate [0.1, 1]",
            t_scaled,
            slopes,
            (0.1, 1.0),
        )

        print_slope_window(
            f"late {late_windows[rho]}",
            t_scaled,
            slopes,
            late_windows[rho],
        )

        print(
            f"  intervals with local slope within "
            f"{1 - slope_tolerance:.2f}-"
            f"{1 + slope_tolerance:.2f}:"
        )

        print_diffusive_intervals(
            t_scaled,
            slopes,
            slope_tolerance,
        )

    # Short-time diffusion coefficients

    print(
        "\nSHORT-TIME DIFFUSION COEFFICIENTS "
        "[1e-4, 0.1]"
    )

    D_A_short = fit_D_per_run(
        t_sweep,
        t_scaled,
        A_all,
        short_window,
    )

    D_B_short = fit_D_per_run(
        t_sweep,
        t_scaled,
        B_all,
        short_window,
    )

    D_cm_short = fit_D_per_run(
        t_sweep,
        t_scaled,
        full_all,
        short_window,
    )

    print_D_result("D_A", D_A_short)
    print_D_result("D_B", D_B_short)
    print_D_result("D_cm", D_cm_short)

    diffusion_rows.append(
        [
            rho,
            short_window[0],
            short_window[1],
            np.mean(D_A_short),
            sem(D_A_short),
            np.mean(D_B_short),
            sem(D_B_short),
            np.mean(D_cm_short),
            sem(D_cm_short),
        ]
    )

    # Long-time diffusion coefficients

    late_window = late_windows[rho]

    print(
        f"\nLONG-TIME DIFFUSION COEFFICIENTS "
        f"[{late_window[0]}, {late_window[1]}]"
    )

    D_A_late = fit_D_per_run(
        t_sweep,
        t_scaled,
        A_all,
        late_window,
    )

    D_B_late = fit_D_per_run(
        t_sweep,
        t_scaled,
        B_all,
        late_window,
    )

    D_cm_late = fit_D_per_run(
        t_sweep,
        t_scaled,
        full_all,
        late_window,
    )

    print_D_result("D_A", D_A_late)
    print_D_result("D_B", D_B_late)
    print_D_result("D_cm", D_cm_late)

    diffusion_rows.append(
        [
            rho,
            late_window[0],
            late_window[1],
            np.mean(D_A_late),
            sem(D_A_late),
            np.mean(D_B_late),
            sem(D_B_late),
            np.mean(D_cm_late),
            sem(D_cm_late),
        ]
    )


# Save results

np.savetxt(
    diffusion_csv_file,
    np.asarray(diffusion_rows, dtype=np.float64),
    delimiter=",",
    header=(
        "rho,fit_scaled_min,fit_scaled_max,"
        "D_A_mean,D_A_sem,"
        "D_B_mean,D_B_sem,"
        "D_cm_mean,D_cm_sem"
    ),
    comments="",
)

print("\n" + "=" * 72)
print(f"Saved: {diffusion_csv_file}")
print("Done.")