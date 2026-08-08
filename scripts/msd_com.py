# Fig. 9: COM MSDs for block A, block B and full chain

from multiprocessing import Pool, cpu_count
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


L = 500
N = 100
rho_list = [1.0, 2.0, 4.0]

n_runs = 100
max_scaled_time = 20.0
n_sample_points = 700
base_seed = 12345

data_dir = Path("data/processed")
figure_dir = Path("figures")

npz_file = data_dir / "block_com_msd_data.npz"
summary_csv_file = data_dir / "block_com_msd_summary.csv"
png_file = figure_dir / "Fig9.png"

color_A = "#ff7f0e"
color_B = "#1f77b4"
color_full = "#2ca02c"

moves = np.array(
    [[1, 0], [-1, 0], [0, 1], [0, -1]],
    dtype=np.int32,
)


half = N // 2
A_indices = np.arange(half)
B_indices = np.arange(half, N)


def build_three_monomer_dictionary():
    dictionary = {}

    for v_left in moves:
        for v_right in moves:
            left = np.asarray(v_left, dtype=np.int32)
            right = np.asarray(v_right, dtype=np.int32)
            allowed = []

            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    candidate = np.array([dx, dy], dtype=np.int32)

                    if dx == 0 and dy == 0:
                        continue

                    dist_left = np.abs(candidate - left).sum()
                    dist_right = np.abs(candidate - right).sum()

                    if dist_left == 1 and dist_right == 1:
                        allowed.append((dx, dy))

            key = (
                int(v_left[0]),
                int(v_left[1]),
                int(v_right[0]),
                int(v_right[1]),
            )
            dictionary[key] = allowed

    return dictionary


THREE_MONOMER_DICT = build_three_monomer_dictionary()


def rho_key(rho):
    return str(float(rho)).replace(".", "p")


def init_gaussian_chain(rng):
    chain = np.zeros((N, 2), dtype=np.int32)
    chain[0] = [L // 2, L // 2]

    for i in range(1, N):
        chain[i] = chain[i - 1] + moves[rng.integers(len(moves))]

    return chain


def in_bounds(position):
    return 0 <= position[0] < L and 0 <= position[1] < L


def legal_positions(chain, i):
    current = chain[i]

    if i == 0 or i == N - 1:
        neighbor = chain[1 if i == 0 else N - 2]
        options = []

        for move in moves:
            candidate = neighbor + move

            if np.array_equal(candidate, current):
                continue

            if in_bounds(candidate):
                options.append(candidate.copy())

        return options

    left_vector = chain[i - 1] - current
    right_vector = chain[i + 1] - current

    key = (
        int(left_vector[0]),
        int(left_vector[1]),
        int(right_vector[0]),
        int(right_vector[1]),
    )

    options = []

    for dx, dy in THREE_MONOMER_DICT.get(key, []):
        candidate = current + np.array([dx, dy], dtype=np.int32)

        if in_bounds(candidate):
            options.append(candidate)

    return options


def pick_index_by_rate(rng, rho):
    probability_A = rho * len(A_indices)
    probability_A /= probability_A + len(B_indices)

    if rng.random() < probability_A:
        return int(A_indices[rng.integers(len(A_indices))])

    return int(B_indices[rng.integers(len(B_indices))])


def mc_attempt(chain, rho, rng):
    i = pick_index_by_rate(rng, rho)
    options = legal_positions(chain, i)

    if not options:
        return

    chain[i] = options[rng.integers(len(options))]


def autocorrelation_fft(values):
    values = np.asarray(values, dtype=np.float64)
    length = len(values)

    n_fft = 1
    while n_fft < 2 * length:
        n_fft *= 2

    transformed = np.fft.fft(values, n=n_fft)
    return np.fft.ifft(
        transformed * np.conjugate(transformed)
    ).real[:length]


def time_averaged_com_msd(trajectory, lag_sweeps):
    trajectory = np.asarray(trajectory, dtype=np.float64)
    length = len(trajectory)

    x = trajectory[:, 0]
    y = trajectory[:, 1]

    squared_position = x**2 + y**2
    cumulative = np.concatenate([[0.0], np.cumsum(squared_position)])

    autocorrelation = autocorrelation_fft(x) + autocorrelation_fft(y)
    msd = np.empty(len(lag_sweeps), dtype=np.float64)

    for index, lag in enumerate(lag_sweeps):
        lag = int(lag)

        if lag <= 0 or lag >= length:
            raise ValueError("Lag must satisfy 1 <= lag < trajectory length.")

        n_terms = length - lag
        first_sum = cumulative[length - lag]
        second_sum = cumulative[length] - cumulative[lag]
        cross_sum = autocorrelation[lag]

        msd[index] = (
            first_sum + second_sum - 2.0 * cross_sum
        ) / n_terms

    return msd


def simulate_one_run(arguments):
    rho, seed = arguments

    rng = np.random.default_rng(seed)
    chain = init_gaussian_chain(rng)

    n_sweeps = int(np.ceil(max_scaled_time * N**2))

    sample_sweeps = np.unique(
        np.round(
            np.logspace(0, np.log10(n_sweeps), n_sample_points)
        ).astype(np.int64)
    )
    sample_sweeps = sample_sweeps[
        (sample_sweeps >= 1) & (sample_sweeps <= n_sweeps)
    ]

    A_trajectory = np.empty((n_sweeps + 1, 2), dtype=np.float64)
    B_trajectory = np.empty((n_sweeps + 1, 2), dtype=np.float64)
    full_trajectory = np.empty((n_sweeps + 1, 2), dtype=np.float64)

    A_trajectory[0] = chain[:half].mean(axis=0)
    B_trajectory[0] = chain[half:].mean(axis=0)
    full_trajectory[0] = chain.mean(axis=0)

    for sweep in range(1, n_sweeps + 1):
        for _ in range(N):
            mc_attempt(chain, rho, rng)

        A_trajectory[sweep] = chain[:half].mean(axis=0)
        B_trajectory[sweep] = chain[half:].mean(axis=0)
        full_trajectory[sweep] = chain.mean(axis=0)

    t_sweep = sample_sweeps.astype(np.float64)
    t_scaled = t_sweep / N**2

    return {
        "t_sweep": t_sweep,
        "t_scaled": t_scaled,
        "A_com_msd": time_averaged_com_msd(
            A_trajectory,
            sample_sweeps,
        ),
        "B_com_msd": time_averaged_com_msd(
            B_trajectory,
            sample_sweeps,
        ),
        "full_com_msd": time_averaged_com_msd(
            full_trajectory,
            sample_sweeps,
        ),
    }


def run_simulations():
    all_results = {}
    n_processes = min(max(1, cpu_count() - 1), n_runs)

    print(f"Using {n_processes} processes")

    for rho in rho_list:
        print(f"\nRunning rho = {rho:g}")

        seeds = [
            base_seed + int(100000 * rho) + run
            for run in range(n_runs)
        ]
        arguments = [(rho, seed) for seed in seeds]

        A_curves = []
        B_curves = []
        full_curves = []
        t_sweep = None
        t_scaled = None

        with Pool(processes=n_processes) as pool:
            for run_index, result in enumerate(
                pool.imap(simulate_one_run, arguments),
                start=1,
            ):
                A_curves.append(result["A_com_msd"])
                B_curves.append(result["B_com_msd"])
                full_curves.append(result["full_com_msd"])
                t_sweep = result["t_sweep"]
                t_scaled = result["t_scaled"]

                if run_index % 5 == 0 or run_index == n_runs:
                    print(f"Finished {run_index}/{n_runs} runs")

        A_curves = np.vstack(A_curves)
        B_curves = np.vstack(B_curves)
        full_curves = np.vstack(full_curves)

        all_results[rho] = {
            "t_sweep": t_sweep,
            "t_scaled": t_scaled,
            "A_all": A_curves,
            "B_all": B_curves,
            "full_all": full_curves,
            "A_mean": A_curves.mean(axis=0),
            "A_sem": A_curves.std(axis=0, ddof=1) / np.sqrt(n_runs),
            "B_mean": B_curves.mean(axis=0),
            "B_sem": B_curves.std(axis=0, ddof=1) / np.sqrt(n_runs),
            "full_mean": full_curves.mean(axis=0),
            "full_sem": full_curves.std(axis=0, ddof=1) / np.sqrt(n_runs),
        }

    return all_results


def save_results(all_results):
    data_dir.mkdir(parents=True, exist_ok=True)

    npz_data = {
        "L": np.array(L),
        "N": np.array(N),
        "rho_list": np.array(rho_list),
        "n_runs": np.array(n_runs),
        "max_scaled_time": np.array(max_scaled_time),
        "n_sample_points": np.array(n_sample_points),
    }

    summary_rows = []

    for rho in rho_list:
        key = rho_key(rho)
        data = all_results[rho]

        npz_data[f"t_sweep_rho_{key}"] = data["t_sweep"]
        npz_data[f"t_scaled_rho_{key}"] = data["t_scaled"]
        npz_data[f"A_com_mean_rho_{key}"] = data["A_mean"]
        npz_data[f"A_com_sem_rho_{key}"] = data["A_sem"]
        npz_data[f"B_com_mean_rho_{key}"] = data["B_mean"]
        npz_data[f"B_com_sem_rho_{key}"] = data["B_sem"]
        npz_data[f"full_com_mean_rho_{key}"] = data["full_mean"]
        npz_data[f"full_com_sem_rho_{key}"] = data["full_sem"]
        npz_data[f"A_com_all_rho_{key}"] = data["A_all"]
        npz_data[f"B_com_all_rho_{key}"] = data["B_all"]
        npz_data[f"full_com_all_rho_{key}"] = data["full_all"]

        for index in range(len(data["t_scaled"])):
            summary_rows.append(
                [
                    rho,
                    data["t_sweep"][index],
                    data["t_scaled"][index],
                    data["A_mean"][index],
                    data["A_sem"][index],
                    data["B_mean"][index],
                    data["B_sem"][index],
                    data["full_mean"][index],
                    data["full_sem"][index],
                ]
            )

    np.savez(npz_file, **npz_data)

    np.savetxt(
        summary_csv_file,
        np.asarray(summary_rows, dtype=np.float64),
        delimiter=",",
        header=(
            "rho,t_sweep,t_scaled,"
            "A_COM_MSD_mean,A_COM_MSD_sem,"
            "B_COM_MSD_mean,B_COM_MSD_sem,"
            "full_COM_MSD_mean,full_COM_MSD_sem"
        ),
        comments="",
    )


# Plotting

def add_slope_one_guide(ax, x_data, y_data, show_label=True):
    x_data = np.asarray(x_data, dtype=np.float64)
    y_data = np.asarray(y_data, dtype=np.float64)

    mask = (
        np.isfinite(x_data)
        & np.isfinite(y_data)
        & (x_data > 0)
        & (y_data > 0)
    )

    if np.count_nonzero(mask) < 5:
        return

    x_data = x_data[mask]
    y_data = y_data[mask]

    x_min = np.nanmin(x_data)
    x_max = np.nanmax(x_data)

    if x_min <= 0 or x_max <= x_min:
        return

    x_guide = np.logspace(np.log10(x_min), np.log10(x_max), 300)
    x_anchor = np.sqrt(x_min * x_max)
    y_anchor = np.percentile(y_data, 18)
    y_guide = y_anchor * 0.55 * (x_guide / x_anchor)

    ax.loglog(
        x_guide,
        y_guide,
        linestyle="-",
        linewidth=1.8,
        color="0.50",
        label=r"slope $1$" if show_label else "_nolegend_",
        zorder=1,
    )


def plot_figure(all_results):
    figure_dir.mkdir(parents=True, exist_ok=True)

    all_x = []
    all_y = []

    for rho in rho_list:
        data = all_results[rho]

        for curve in (
            data["A_mean"],
            data["B_mean"],
            data["full_mean"],
        ):
            mask = (
                np.isfinite(data["t_scaled"])
                & np.isfinite(curve)
                & (data["t_scaled"] > 0)
                & (curve > 0)
            )

            if np.any(mask):
                all_x.append(data["t_scaled"][mask])
                all_y.append(curve[mask])

    if not all_x or not all_y:
        raise ValueError("No finite positive COM MSD data were found.")

    all_x = np.concatenate(all_x)
    all_y = np.concatenate(all_y)

    x_min = np.nanmin(all_x)
    x_max = np.nanmax(all_x)
    y_min = 10 ** np.floor(np.log10(np.nanmin(all_y)))
    y_max = 10 ** (
        np.ceil(np.log10(np.nanmax(all_y))) + 0.25
    )

    fig, axes = plt.subplots(
        1,
        len(rho_list),
        figsize=(13.6, 4.65),
        sharey=True,
    )
    axes = np.atleast_1d(axes)

    for panel_index, (ax, rho) in enumerate(zip(axes, rho_list)):
        data = all_results[rho]
        t_scaled = data["t_scaled"]

        lower_A = np.maximum(
            data["A_mean"] - data["A_sem"],
            1.0e-14,
        )
        upper_A = data["A_mean"] + data["A_sem"]

        lower_B = np.maximum(
            data["B_mean"] - data["B_sem"],
            1.0e-14,
        )
        upper_B = data["B_mean"] + data["B_sem"]

        lower_full = np.maximum(
            data["full_mean"] - data["full_sem"],
            1.0e-14,
        )
        upper_full = data["full_mean"] + data["full_sem"]

        ax.loglog(
            t_scaled,
            data["A_mean"],
            color=color_A,
            linewidth=2.3,
            label="block A COM",
            zorder=4,
        )
        ax.fill_between(
            t_scaled,
            lower_A,
            upper_A,
            color=color_A,
            alpha=0.16,
            linewidth=0,
            zorder=3,
        )

        ax.loglog(
            t_scaled,
            data["B_mean"],
            color=color_B,
            linewidth=2.3,
            label="block B COM",
            zorder=4,
        )
        ax.fill_between(
            t_scaled,
            lower_B,
            upper_B,
            color=color_B,
            alpha=0.16,
            linewidth=0,
            zorder=3,
        )

        ax.loglog(
            t_scaled,
            data["full_mean"],
            color=color_full,
            linewidth=2.3,
            label="full-chain COM",
            zorder=4,
        )
        ax.fill_between(
            t_scaled,
            lower_full,
            upper_full,
            color=color_full,
            alpha=0.16,
            linewidth=0,
            zorder=3,
        )

        add_slope_one_guide(
            ax,
            np.concatenate([t_scaled, t_scaled, t_scaled]),
            np.concatenate(
                [
                    data["A_mean"],
                    data["B_mean"],
                    data["full_mean"],
                ]
            ),
            show_label=(panel_index == 0),
        )

        ax.set_title(rf"$\rho={rho:g}$", fontsize=13)
        ax.set_xlabel(
            r"Scaled lag time, $\Delta t_{\mathrm{sweep}}/N^2$",
            fontsize=12,
        )
        ax.set_xlim(x_min * 0.85, x_max * 1.05)
        ax.set_ylim(y_min, y_max)
        ax.tick_params(axis="both", which="major", labelsize=10)
        ax.tick_params(axis="both", which="minor", labelsize=8)
        ax.grid(False)

    axes[0].set_ylabel(
        r"Center-of-mass MSD, "
        r"$\langle |\mathbf{R}(t+\Delta t)-\mathbf{R}(t)|^2\rangle$",
        fontsize=12,
    )

    handles, labels = axes[0].get_legend_handles_labels()

    main_handles = []
    main_labels = []
    guide_handles = []
    guide_labels = []

    for handle, label in zip(handles, labels):
        if label in (
            "block A COM",
            "block B COM",
            "full-chain COM",
        ):
            main_handles.append(handle)
            main_labels.append(label)
        elif "slope" in label:
            guide_handles.append(handle)
            guide_labels.append(label)

    axes[0].legend(
        main_handles + guide_handles,
        main_labels + guide_labels,
        fontsize=8.7,
        frameon=False,
        loc="upper left",
        handlelength=2.0,
        labelspacing=0.32,
    )

    fig.tight_layout(w_pad=1.2)
    fig.savefig(png_file, dpi=800, bbox_inches="tight")
    plt.close(fig)


def main():
    data_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    all_results = run_simulations()
    save_results(all_results)
    plot_figure(all_results)

    print("\nSaved:")
    print(npz_file)
    print(summary_csv_file)
    print(png_file)
    print("\nDone.")


if __name__ == "__main__":
    main()