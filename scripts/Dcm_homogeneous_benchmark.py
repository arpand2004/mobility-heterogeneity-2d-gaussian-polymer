import os
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count

L = 500
N_list = [20, 40, 60, 80, 100]
n_runs = 100
max_scaled_time = 20.0
n_sample_points = 700

fit_scaled_min = 0.1
fit_scaled_max = 10.0

base_seed = 12345

moves = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]], dtype=np.int32)

colors = {
    20: "#1f77b4",
    40: "#ff7f0e",
    60: "#2ca02c",
    80: "#d62728",
    100: "#9467bd",
}

def build_three_monomer_dictionary():
    dictionary = {}

    for v_left in moves:
        for v_right in moves:
            v_left = tuple(v_left)
            v_right = tuple(v_right)

            left = np.array(v_left, dtype=np.int32)
            right = np.array(v_right, dtype=np.int32)
            current = np.array([0, 0], dtype=np.int32)

            allowed = []

            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    candidate = np.array([dx, dy], dtype=np.int32)

                    if np.array_equal(candidate, current):
                        continue

                    dist_left = abs(candidate[0] - left[0]) + abs(candidate[1] - left[1])
                    dist_right = abs(candidate[0] - right[0]) + abs(candidate[1] - right[1])

                    if dist_left == 1 and dist_right == 1:
                        allowed.append((dx, dy))

            dictionary[(v_left[0], v_left[1], v_right[0], v_right[1])] = allowed

    return dictionary


THREE_MONOMER_DICT = build_three_monomer_dictionary()

def init_chain(L, N, rng):
    chain = np.zeros((N, 2), dtype=np.int32)
    chain[0] = [L // 2, L // 2]

    for i in range(1, N):
        for dx, dy in moves[rng.permutation(len(moves))]:
            p = chain[i - 1] + np.array([dx, dy], dtype=np.int32)

            if 0 <= p[0] < L and 0 <= p[1] < L:
                chain[i] = p
                break
        else:
            return init_chain(L, N, rng)

    return chain

def legal_positions_gaussian_dictionary(chain, i, N, L):
    curr_x, curr_y = chain[i]

    if i == 0 or i == N - 1:
        j = 1 if i == 0 else N - 2
        nx, ny = chain[j]

        opts = []
        for dx, dy in moves:
            px, py = nx + dx, ny + dy

            if px == curr_x and py == curr_y:
                continue
            if 0 <= px < L and 0 <= py < L:
                opts.append((px, py))

        return opts

    left = chain[i - 1]
    right = chain[i + 1]
    curr = chain[i]

    v_left = left - curr
    v_right = right - curr

    key = (int(v_left[0]), int(v_left[1]), int(v_right[0]), int(v_right[1]))

    allowed_displacements = THREE_MONOMER_DICT.get(key, [])

    opts = []
    for dx, dy in allowed_displacements:
        px = curr_x + dx
        py = curr_y + dy

        if 0 <= px < L and 0 <= py < L:
            opts.append((px, py))

    return opts


def mc_attempt(chain, N, L, rng):
    i = rng.integers(N)
    opts = legal_positions_gaussian_dictionary(chain, i, N, L)

    if opts:
        chain[i] = opts[rng.integers(len(opts))]
        return True

    return False

def autocorrelation_fft(x):
    x = np.asarray(x, dtype=np.float64)
    T = len(x)

    n_fft = 1
    while n_fft < 2 * T:
        n_fft *= 2

    f = np.fft.fft(x, n=n_fft)
    ac = np.fft.ifft(f * np.conjugate(f)).real[:T]

    return ac


def time_origin_com_msd(Rcm, lag_sweeps):
    Rcm = np.asarray(Rcm, dtype=np.float64)
    T = len(Rcm)

    x = Rcm[:, 0]
    y = Rcm[:, 1]

    x2y2 = x**2 + y**2
    prefix = np.concatenate([[0.0], np.cumsum(x2y2)])

    ac_x = autocorrelation_fft(x)
    ac_y = autocorrelation_fft(y)
    ac = ac_x + ac_y

    msd = np.empty(len(lag_sweeps), dtype=np.float64)

    for idx, lag in enumerate(lag_sweeps):
        lag = int(lag)

        if lag <= 0 or lag >= T:
            raise ValueError("Lag must satisfy 1 <= lag < trajectory length.")

        n_terms = T - lag

        sum_first = prefix[T - lag] - prefix[0]

        sum_second = prefix[T] - prefix[lag]

        sum_cross = ac[lag]

        msd[idx] = (sum_first + sum_second - 2.0 * sum_cross) / n_terms

    return msd

def run_com_msd_single(args):
    L, N, max_scaled_time, n_sample_points, seed = args

    rng = np.random.default_rng(seed)
    chain = init_chain(L, N, rng)

    n_sweeps = int(np.ceil(max_scaled_time * N**2))

    Rcm_traj = np.empty((n_sweeps + 1, 2), dtype=np.float64)
    Rcm_traj[0] = chain.mean(axis=0)

    for sweep in range(1, n_sweeps + 1):

        for _ in range(N):
            mc_attempt(chain, N, L, rng)

        Rcm_traj[sweep] = chain.mean(axis=0)

    lag_sweeps = np.unique(
        np.round(
            np.logspace(0, np.log10(n_sweeps), n_sample_points)
        ).astype(int)
    )
    lag_sweeps = lag_sweeps[(lag_sweeps >= 1) & (lag_sweeps < len(Rcm_traj))]

    com_msd = time_origin_com_msd(Rcm_traj, lag_sweeps)

    t_sweep = lag_sweeps.astype(np.float64)
    t_scaled = t_sweep / (N**2)

    return N, t_sweep, t_scaled, com_msd

def extract_Dcm(t_sweep, t_scaled, com_msd, fit_scaled_min, fit_scaled_max):
    fit_mask = (t_scaled >= fit_scaled_min) & (t_scaled <= fit_scaled_max)

    if fit_mask.sum() < 5:
        raise ValueError("Not enough points inside the D_cm fitting window.")

    slope, intercept = np.polyfit(t_sweep[fit_mask], com_msd[fit_mask], 1)

    D_cm = slope / 4.0

    return D_cm, slope, intercept, fit_mask

if __name__ == "__main__":

    os.makedirs("figures", exist_ok=True)

    all_results = {}
    D_values = []
    D_errors = []

    n_processes = min(max(1, cpu_count() - 1), n_runs)
    print(f"Using {n_processes} processes")

    for N in N_list:
        print(f"\n=== Running center-of-mass diffusion benchmark: N = {N} ===")

        seeds = [base_seed + 1000 * N + run for run in range(n_runs)]
        args_list = [
            (L, N, max_scaled_time, n_sample_points, seed)
            for seed in seeds
        ]

        curves = []

        with Pool(processes=n_processes) as pool:
            for run_idx, result in enumerate(pool.imap_unordered(run_com_msd_single, args_list), start=1):
                _, t_sweep, t_scaled, com_msd = result
                curves.append(com_msd)

                if run_idx % 10 == 0 or run_idx == n_runs:
                    print(f"  Finished {run_idx}/{n_runs} runs")

        com_msd_all = np.vstack(curves)
        com_msd_avg = com_msd_all.mean(axis=0)
        com_msd_sem = com_msd_all.std(axis=0, ddof=1) / np.sqrt(n_runs)

        D_cm, slope, intercept, fit_mask = extract_Dcm(
            t_sweep,
            t_scaled,
            com_msd_avg,
            fit_scaled_min,
            fit_scaled_max,
        )

        D_run_values = []
        for run_curve in com_msd_all:
            D_run, _, _, _ = extract_Dcm(
                t_sweep,
                t_scaled,
                run_curve,
                fit_scaled_min,
                fit_scaled_max,
            )
            D_run_values.append(D_run)

        D_run_values = np.array(D_run_values, dtype=np.float64)
        D_err = D_run_values.std(ddof=1) / np.sqrt(n_runs)

        D_values.append(D_cm)
        D_errors.append(D_err)

        all_results[N] = {
            "t_sweep": t_sweep,
            "t_scaled": t_scaled,
            "com_msd_avg": com_msd_avg,
            "com_msd_sem": com_msd_sem,
            "D_cm": D_cm,
            "D_err": D_err,
            "fit_slope": slope,
            "fit_intercept": intercept,
            "fit_mask": fit_mask,
        }

        print(f"  D_cm(N={N}) = {D_cm:.6e} ± {D_err:.2e}")

    D_values = np.array(D_values, dtype=np.float64)
    D_errors = np.array(D_errors, dtype=np.float64)
    N_array = np.array(N_list, dtype=np.float64)

    logN = np.log(N_array)
    logD = np.log(D_values)
    slope_loglog, intercept_loglog = np.polyfit(logN, logD, 1)
    gamma = -slope_loglog

    print("\n=== Scaling estimate ===")
    print(f"D_cm ~ N^(-gamma)")
    print(f"gamma = {gamma:.3f}")

    fig, ax = plt.subplots(figsize=(6.4, 4.8))

    for idx, N in enumerate(N_list):
        ax.errorbar(
            N,
            D_values[idx],
            yerr=D_errors[idx],
            fmt="o",
            markersize=8.0,
            capsize=4,
            color=colors[N],
            markeredgecolor="black",
            markeredgewidth=0.7,
            linewidth=1.5,
            label=fr"$N={N}$",
            zorder=3,
        )

    N_guide = np.linspace(min(N_list), max(N_list), 300)

    N_anchor = 40
    D_anchor = D_values[N_list.index(N_anchor)]
    D_guide = D_anchor * (N_guide / N_anchor)**(-1)

    ax.loglog(
        N_guide,
        D_guide,
        linestyle="--",
        linewidth=2.0,
        color="0.55",
        label=r"slope $-1$",
        zorder=1,
    )

    D_fit = np.exp(intercept_loglog) * N_guide**slope_loglog

    ax.loglog(
        N_guide,
        D_fit,
        linestyle="-",
        linewidth=1.8,
        color="0.25",
        label=fr"fit slope $-{gamma:.2f}$",
        zorder=2,
    )

    ax.set_xlabel(r"Chain length, $N$", fontsize=13)
    ax.set_ylabel(r"$D_{\mathrm{cm}}$", fontsize=14)

    ax.tick_params(axis="both", which="major", labelsize=11)
    ax.tick_params(axis="both", which="minor", labelsize=9)

    ax.legend(
        fontsize=10,
        frameon=False,
        loc="upper right",
    )

    ax.grid(False)

    fig.tight_layout()

    fig.savefig("figures/Fig4.png", dpi=800, bbox_inches="tight")

    plt.show()
