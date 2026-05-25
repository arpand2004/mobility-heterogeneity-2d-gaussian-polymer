import os
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count

L = 500
N_list = [20, 40, 60, 80, 100]
rho_list = [1.0, 2.0, 4.0]

n_runs = 100
max_scaled_time = 20.0
n_sample_points = 700

base_seed = 12345

data_file = "data/processed/dcm_N_data.npz"
summary_file = "data/processed/dcm_N_summary.csv"
check_figure_file = "figures/COM_MSD_check.png"

moves = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]], dtype=np.int32)

colors_N = {
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


def init_gaussian_chain(L, N, rng):
    chain = np.zeros((N, 2), dtype=np.int32)
    chain[0] = [L // 2, L // 2]

    for i in range(1, N):
        k = rng.integers(len(moves))
        chain[i] = chain[i - 1] + moves[k]

    return chain


def in_bounds(p, L):
    return 0 <= p[0] < L and 0 <= p[1] < L


def legal_positions_dictionary_gaussian(chain, i, N, L):
    curr_x, curr_y = chain[i]

    if i == 0 or i == N - 1:
        j = 1 if i == 0 else N - 2
        neighbor = chain[j]

        opts = []

        for dx, dy in moves:
            p = neighbor + np.array([dx, dy], dtype=np.int32)

            if p[0] == curr_x and p[1] == curr_y:
                continue

            if not in_bounds(p, L):
                continue

            opts.append(p.copy())

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
        p = curr + np.array([dx, dy], dtype=np.int32)

        if not in_bounds(p, L):
            continue

        opts.append(p.copy())

    return opts


def pick_index_by_rate(rng, N, rho):
    half = N // 2

    weight_A = rho * half
    weight_B = N - half

    p_A = weight_A / (weight_A + weight_B)

    if rng.random() < p_A:
        return int(rng.integers(0, half))

    return int(rng.integers(half, N))


def mc_attempt(chain, N, L, rho, rng):
    i = pick_index_by_rate(rng, N, rho)
    opts = legal_positions_dictionary_gaussian(chain, i, N, L)

    if opts:
        new_pos_array = opts[rng.integers(len(opts))]
        chain[i] = new_pos_array
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


def time_averaged_com_msd(Rcm, lag_sweeps):
    Rcm = np.asarray(Rcm, dtype=np.float64)
    T = len(Rcm)

    x = Rcm[:, 0]
    y = Rcm[:, 1]

    r2 = x**2 + y**2
    prefix = np.concatenate([[0.0], np.cumsum(r2)])

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
    L, N, rho, max_scaled_time, n_sample_points, seed = args

    rng = np.random.default_rng(seed)
    chain = init_gaussian_chain(L, N, rng)

    n_sweeps = int(np.ceil(max_scaled_time * N**2))

    Rcm_traj = np.empty((n_sweeps + 1, 2), dtype=np.float64)
    Rcm_traj[0] = chain.mean(axis=0)

    for sweep in range(1, n_sweeps + 1):
        for _ in range(N):
            mc_attempt(chain, N, L, rho, rng)

        Rcm_traj[sweep] = chain.mean(axis=0)

    lag_sweeps = np.unique(
        np.round(
            np.logspace(0, np.log10(n_sweeps), n_sample_points)
        ).astype(int)
    )

    lag_sweeps = lag_sweeps[(lag_sweeps >= 1) & (lag_sweeps < len(Rcm_traj))]

    com_msd = time_averaged_com_msd(Rcm_traj, lag_sweeps)

    t_sweep = lag_sweeps.astype(np.float64)
    t_scaled = t_sweep / (N**2)

    return N, rho, t_sweep, t_scaled, com_msd


def save_com_results(results_by_rho_N):
    os.makedirs("data/processed", exist_ok=True)

    npz_data = {
        "L": np.array(L),
        "N_list": np.array(N_list),
        "rho_list": np.array(rho_list),
        "n_runs": np.array(n_runs),
        "max_scaled_time": np.array(max_scaled_time),
        "n_sample_points": np.array(n_sample_points),
    }

    csv_rows = []

    for rho in rho_list:
        rho_key = str(rho).replace(".", "p")

        for N in N_list:
            data = results_by_rho_N[(rho, N)]
            N_key = str(N)

            npz_data[f"t_scaled_rho_{rho_key}_N_{N_key}"] = data["t_scaled"]
            npz_data[f"t_sweep_rho_{rho_key}_N_{N_key}"] = data["t_sweep"]
            npz_data[f"com_msd_mean_rho_{rho_key}_N_{N_key}"] = data["mean"]
            npz_data[f"com_msd_sem_rho_{rho_key}_N_{N_key}"] = data["sem"]
            npz_data[f"com_msd_all_rho_{rho_key}_N_{N_key}"] = data["all_runs"]

            for j in range(len(data["t_scaled"])):
                csv_rows.append([
                    rho,
                    N,
                    data["t_sweep"][j],
                    data["t_scaled"][j],
                    data["mean"][j],
                    data["sem"][j],
                ])

    np.savez(data_file, **npz_data)

    np.savetxt(
        summary_file,
        np.array(csv_rows, dtype=np.float64),
        delimiter=",",
        header="rho,N,t_sweep,t_scaled,COM_MSD_mean,COM_MSD_sem",
        comments="",
    )


def add_slope_guides(ax, x_data, y_data):
    x_positive = np.asarray(x_data)
    y_positive = np.asarray(y_data)

    mask = (x_positive > 0) & (y_positive > 0)

    if mask.sum() < 5:
        return

    x_positive = x_positive[mask]
    y_positive = y_positive[mask]

    x_min = np.nanmin(x_positive)
    x_max = np.nanmax(x_positive)

    if x_min <= 0 or x_max <= x_min:
        return

    xg = np.logspace(np.log10(x_min), np.log10(x_max), 300)

    y_anchor = np.percentile(y_positive, 30)
    x_anchor = np.sqrt(x_min * x_max)

    y_half = y_anchor * (xg / x_anchor) ** 0.5
    y_one = y_anchor * 2.8 * (xg / x_anchor) ** 1.0

    ax.loglog(
        xg,
        y_half,
        linestyle="--",
        linewidth=1.8,
        color="0.50",
        label=r"slope $1/2$",
        zorder=1,
    )

    ax.loglog(
        xg,
        y_one,
        linestyle="-",
        linewidth=1.8,
        color="0.50",
        label=r"slope $1$",
        zorder=1,
    )


def plot_com_msd(results_by_rho_N):
    os.makedirs("figures", exist_ok=True)

    fig, axs = plt.subplots(1, 3, figsize=(13.6, 5.4), sharey=True)

    for ax, rho in zip(axs, rho_list):
        all_x = []
        all_y = []

        for N in N_list:
            data = results_by_rho_N[(rho, N)]
            t = data["t_scaled"]
            y = data["mean"]
            err = data["sem"]

            all_x.append(t)
            all_y.append(y)

            lower = np.maximum(y - err, 1.0e-14)
            upper = y + err

            ax.loglog(
                t,
                y,
                color=colors_N[N],
                linewidth=2.1,
                label=rf"$N={N}$",
                zorder=3,
            )

            ax.fill_between(
                t,
                lower,
                upper,
                color=colors_N[N],
                alpha=0.14,
                linewidth=0,
                zorder=2,
            )

        all_x = np.concatenate(all_x)
        all_y = np.concatenate(all_y)

        add_slope_guides(ax, all_x, all_y)

        ax.set_title(rf"$\rho={rho:g}$", fontsize=13)
        ax.set_xlabel(r"Scaled time, $t_{\mathrm{sweep}}/N^2$", fontsize=12)
        ax.set_xlim(None, max_scaled_time)
        ax.tick_params(axis="both", which="major", labelsize=10)
        ax.tick_params(axis="both", which="minor", labelsize=8)
        ax.grid(False)

    axs[0].set_ylabel(
        r"COM MSD" + "\n" + r"$\langle |R_{\mathrm{cm}}(t)-R_{\mathrm{cm}}(0)|^2\rangle$",
        fontsize=12,
    )

    handles, labels = axs[0].get_legend_handles_labels()

    N_handles = []
    N_labels = []
    guide_handles = []
    guide_labels = []

    for h, lab in zip(handles, labels):
        if lab.startswith(r"$N="):
            N_handles.append(h)
            N_labels.append(lab)
        elif "slope" in lab:
            guide_handles.append(h)
            guide_labels.append(lab)

    axs[0].legend(
        N_handles + guide_handles,
        N_labels + guide_labels,
        fontsize=8.7,
        frameon=False,
        loc="upper left",
        handlelength=2.0,
        labelspacing=0.32,
    )

    fig.suptitle(
        r"Center-of-mass MSD for rate-only Gaussian block polymer",
        fontsize=14,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(check_figure_file, dpi=800, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    results_by_rho_N = {}

    n_processes = min(max(1, cpu_count() - 1), n_runs)
    print(f"Using {n_processes} processes")

    for rho in rho_list:
        print("\n==============================")
        print(f"Running rho = {rho:g}")
        print("==============================")

        for N in N_list:
            print(f"\n--- N = {N}, rho = {rho:g} ---")

            seeds = [
                base_seed + int(100000 * rho) + 1000 * N + run
                for run in range(n_runs)
            ]

            args_list = [
                (L, N, rho, max_scaled_time, n_sample_points, seed)
                for seed in seeds
            ]

            curves = []
            t_sweep_ref = None
            t_scaled_ref = None

            with Pool(processes=n_processes) as pool:
                for run_idx, result in enumerate(
                    pool.imap_unordered(run_com_msd_single, args_list),
                    start=1,
                ):
                    _, _, t_sweep, t_scaled, com_msd = result

                    curves.append(com_msd)
                    t_sweep_ref = t_sweep
                    t_scaled_ref = t_scaled

                    if run_idx % 5 == 0 or run_idx == n_runs:
                        print(f"  Finished {run_idx}/{n_runs} runs")

            curves = np.vstack(curves)

            results_by_rho_N[(rho, N)] = {
                "t_sweep": t_sweep_ref,
                "t_scaled": t_scaled_ref,
                "mean": curves.mean(axis=0),
                "sem": curves.std(axis=0, ddof=1) / np.sqrt(n_runs),
                "all_runs": curves,
            }

    save_com_results(results_by_rho_N)
    plot_com_msd(results_by_rho_N)

    print("\nSaved data:")
    print(data_file)
    print(summary_file)

    print("\nSaved check figure:")
    print(check_figure_file)

    print("Done.")
