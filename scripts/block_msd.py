import os
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count

L = 500
N = 100

rate_list = [1.0, 2.0, 4.0]

n_runs = 100
max_scaled_time = 20.0
n_sample_points = 700

base_seed = 12345

data_file = "data/processed/block_msd_data.npz"
summary_file = "data/processed/block_msd_summary.csv"

fig6_file = "figures/Fig6_block_MSD.png"
fig7_file = "figures/Fig7_AMSD.png"

color_A = "#ff7f0e"
color_B = "#1f77b4"
color_interface = "black"

rho_colors = {
    1.0: "#9bbfc1",
    2.0: "#b8a99a",
    4.0: "#c7a2b6",
}

moves = np.array(
    [[1, 0], [-1, 0], [0, 1], [0, -1]],
    dtype=np.int32,
)

half = N // 2
A_indices = np.arange(0, half)
B_indices = np.arange(half, N)
interface_index = half - 1


#3-monomer dictionary

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

    key = (
        int(v_left[0]),
        int(v_left[1]),
        int(v_right[0]),
        int(v_right[1]),
    )

    allowed_displacements = THREE_MONOMER_DICT.get(key, [])

    opts = []

    for dx, dy in allowed_displacements:
        p = curr + np.array([dx, dy], dtype=np.int32)

        if not in_bounds(p, L):
            continue

        opts.append(p.copy())

    return opts


def pick_index_by_rate(rng, rho):
    weight_A = rho * len(A_indices)
    weight_B = len(B_indices)

    p_A = weight_A / (weight_A + weight_B)

    if rng.random() < p_A:
        return int(A_indices[rng.integers(len(A_indices))])

    return int(B_indices[rng.integers(len(B_indices))])


def mc_attempt(chain, rho, rng):
    i = pick_index_by_rate(rng, rho)
    opts = legal_positions_dictionary_gaussian(chain, i, N, L)

    if opts:
        new_pos = opts[rng.integers(len(opts))]
        chain[i] = new_pos
        return True

    return False


def simulate_one_run(args):
    rho, seed = args

    rng = np.random.default_rng(seed)
    chain = init_gaussian_chain(L, N, rng)

    r0 = chain.copy()
    r0_interface = chain[interface_index].copy()

    n_sweeps = int(np.ceil(max_scaled_time * N**2))

    sample_sweeps = np.unique(
        np.round(
            np.logspace(0, np.log10(n_sweeps), n_sample_points)
        ).astype(int)
    )

    sample_sweeps = sample_sweeps[sample_sweeps >= 1]

    msd_A = np.empty(len(sample_sweeps), dtype=np.float64)
    msd_B = np.empty(len(sample_sweeps), dtype=np.float64)
    msd_interface = np.empty(len(sample_sweeps), dtype=np.float64)

    sample_idx = 0

    for sweep in range(1, n_sweeps + 1):
        for _ in range(N):
            mc_attempt(chain, rho, rng)

        if sample_idx < len(sample_sweeps) and sweep == sample_sweeps[sample_idx]:
            disp = chain - r0
            sq = disp[:, 0]**2 + disp[:, 1]**2

            msd_A[sample_idx] = sq[:half].mean()
            msd_B[sample_idx] = sq[half:].mean()

            d_int = chain[interface_index] - r0_interface
            msd_interface[sample_idx] = d_int[0]**2 + d_int[1]**2

            sample_idx += 1

    t_scaled = sample_sweeps.astype(np.float64) / (N**2)

    denom = msd_A + msd_B
    A_msd = np.zeros_like(denom)

    mask = denom > 0
    A_msd[mask] = (msd_A[mask] - msd_B[mask]) / denom[mask]

    return rho, t_scaled, msd_A, msd_B, msd_interface, A_msd

def save_results(all_results):
    os.makedirs("data/processed", exist_ok=True)

    npz_data = {
        "L": np.array(L),
        "N": np.array(N),
        "rate_list": np.array(rate_list),
        "n_runs": np.array(n_runs),
        "max_scaled_time": np.array(max_scaled_time),
        "n_sample_points": np.array(n_sample_points),
    }

    summary_rows = []

    for rho in rate_list:
        key = str(rho).replace(".", "p")
        data = all_results[rho]

        npz_data[f"t_scaled_rho_{key}"] = data["t_scaled"]

        npz_data[f"A_mean_rho_{key}"] = data["A_mean"]
        npz_data[f"A_sem_rho_{key}"] = data["A_sem"]

        npz_data[f"B_mean_rho_{key}"] = data["B_mean"]
        npz_data[f"B_sem_rho_{key}"] = data["B_sem"]

        npz_data[f"I_mean_rho_{key}"] = data["I_mean"]
        npz_data[f"I_sem_rho_{key}"] = data["I_sem"]

        npz_data[f"AMSD_mean_rho_{key}"] = data["AMSD_mean"]
        npz_data[f"AMSD_sem_rho_{key}"] = data["AMSD_sem"]

        npz_data[f"A_all_rho_{key}"] = data["A_all"]
        npz_data[f"B_all_rho_{key}"] = data["B_all"]
        npz_data[f"I_all_rho_{key}"] = data["I_all"]
        npz_data[f"AMSD_all_rho_{key}"] = data["AMSD_all"]

        for j in range(len(data["t_scaled"])):
            summary_rows.append(
                [
                    rho,
                    data["t_scaled"][j],
                    data["A_mean"][j],
                    data["A_sem"][j],
                    data["B_mean"][j],
                    data["B_sem"][j],
                    data["I_mean"][j],
                    data["I_sem"][j],
                    data["AMSD_mean"][j],
                    data["AMSD_sem"][j],
                ]
            )

    np.savez(data_file, **npz_data)

    np.savetxt(
        summary_file,
        np.array(summary_rows, dtype=np.float64),
        delimiter=",",
        header=(
            "rho,t_scaled,"
            "BlockA_mean,BlockA_sem,"
            "BlockB_mean,BlockB_sem,"
            "Junction_mean,Junction_sem,"
            "A_MSD_mean,A_MSD_sem"
        ),
        comments="",
    )

def rho_key(rho):
    return str(rho).replace(".", "p")


def add_slope_guides(ax, x_data, y_data):
    x_positive = np.asarray(x_data, dtype=np.float64)
    y_positive = np.asarray(y_data, dtype=np.float64)

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

    y_anchor = np.percentile(y_positive, 18)
    x_anchor = np.sqrt(x_min * x_max)

    y_half = y_anchor * 1.8 * (xg / x_anchor) ** 0.5
    y_one = y_anchor * 0.55 * (xg / x_anchor) ** 1.0

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

#Figure 6

def plot_fig6_from_saved_data():
    os.makedirs("figures", exist_ok=True)

    data = np.load(data_file)

    N_loaded = int(data["N"])

    all_positive_x = []
    all_positive_y = []

    for rho in rate_list:
        key = rho_key(rho)

        t = data[f"t_scaled_rho_{key}"]
        A_mean = data[f"A_mean_rho_{key}"]
        B_mean = data[f"B_mean_rho_{key}"]
        I_mean = data[f"I_mean_rho_{key}"]

        for y in [A_mean, B_mean, I_mean]:
            mask = (t > 0) & (y > 0)
            all_positive_x.append(t[mask])
            all_positive_y.append(y[mask])

    all_positive_x = np.concatenate(all_positive_x)
    all_positive_y = np.concatenate(all_positive_y)

    x_min_global = np.nanmin(all_positive_x)
    x_max_global = np.nanmax(all_positive_x)

    y_min_global = np.nanmin(all_positive_y)
    y_max_global = np.nanmax(all_positive_y)

    y_min_plot = 10 ** (np.floor(np.log10(y_min_global)))
    y_max_plot = 10 ** (np.ceil(np.log10(y_max_global)) + 0.25)

    fig, axs = plt.subplots(1, 3, figsize=(13.6, 4.65), sharey=True)

    for ax, rho in zip(axs, rate_list):
        key = rho_key(rho)

        t = data[f"t_scaled_rho_{key}"]

        A_mean = data[f"A_mean_rho_{key}"]
        A_sem = data[f"A_sem_rho_{key}"]

        B_mean = data[f"B_mean_rho_{key}"]
        B_sem = data[f"B_sem_rho_{key}"]

        I_mean = data[f"I_mean_rho_{key}"]

        lower_A = np.maximum(A_mean - A_sem, 1.0e-14)
        upper_A = A_mean + A_sem

        lower_B = np.maximum(B_mean - B_sem, 1.0e-14)
        upper_B = B_mean + B_sem

        ax.loglog(
            t,
            A_mean,
            color=color_A,
            linewidth=2.3,
            label="block A",
            zorder=4,
        )

        ax.fill_between(
            t,
            lower_A,
            upper_A,
            color=color_A,
            alpha=0.16,
            linewidth=0,
            zorder=3,
        )

        ax.loglog(
            t,
            B_mean,
            color=color_B,
            linewidth=2.3,
            label="block B",
            zorder=4,
        )

        ax.fill_between(
            t,
            lower_B,
            upper_B,
            color=color_B,
            alpha=0.16,
            linewidth=0,
            zorder=3,
        )

        ax.loglog(
            t,
            I_mean,
            color=color_interface,
            linewidth=1.7,
            label="junction bead",
            zorder=5,
        )

        guide_x = np.concatenate([t, t, t])
        guide_y = np.concatenate([A_mean, B_mean, I_mean])
        add_slope_guides(ax, guide_x, guide_y)

        ax.set_title(rf"$\rho={rho:g}$", fontsize=13)
        ax.set_xlabel(r"Scaled time, $t_{\mathrm{sweep}}/N^2$", fontsize=12)
        ax.set_xlim(x_min_global * 0.85, x_max_global * 1.05)
        ax.set_ylim(y_min_plot, y_max_plot)
        ax.tick_params(axis="both", which="major", labelsize=10)
        ax.tick_params(axis="both", which="minor", labelsize=8)
        ax.grid(False)

    axs[0].set_ylabel(
        r"Block MSD, $\langle |r_i(t)-r_i(0)|^2\rangle$",
        fontsize=12,
    )

    handles, labels = axs[0].get_legend_handles_labels()

    main_handles = []
    main_labels = []
    guide_handles = []
    guide_labels = []

    for h, lab in zip(handles, labels):
        if lab in ["block A", "block B", "junction bead"]:
            main_handles.append(h)
            main_labels.append(lab)
        elif "slope" in lab:
            guide_handles.append(h)
            guide_labels.append(lab)

    axs[0].legend(
        main_handles + guide_handles,
        main_labels + guide_labels,
        fontsize=8.7,
        frameon=False,
        loc="upper left",
        handlelength=2.0,
        labelspacing=0.32,
    )

    fig.suptitle(
        rf"Rate-only Gaussian block polymer, $N={N_loaded}$",
        fontsize=14,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(fig6_file, dpi=800, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {fig6_file}")

#Figure 7

def plot_fig7_from_saved_data():
    os.makedirs("figures", exist_ok=True)

    data = np.load(data_file)

    fig, ax = plt.subplots(figsize=(7.6, 4.9))

    ax.axhline(
        0,
        color="0.55",
        linewidth=1.4,
        linestyle="--",
        zorder=1,
    )

    for rho in rate_list:
        key = rho_key(rho)

        t = data[f"t_scaled_rho_{key}"]

        A_curve = data[f"AMSD_mean_rho_{key}"]
        A_err = data[f"AMSD_sem_rho_{key}"]

        ax.semilogx(
            t,
            A_curve,
            color=rho_colors[rho],
            linewidth=2.1,
            label=rf"$\rho={rho:g}$",
            zorder=3,
        )

        ax.fill_between(
            t,
            A_curve - A_err,
            A_curve + A_err,
            color=rho_colors[rho],
            alpha=0.18,
            linewidth=0,
            zorder=2,
        )

    ax.set_xlabel(r"Scaled time, $t_{\mathrm{sweep}}/N^2$", fontsize=14)
    ax.set_ylabel(
        r"MSD asymmetry, $\mathcal{A}_{\mathrm{MSD}}(t)$",
        fontsize=14,
    )

    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.tick_params(axis="both", which="minor", labelsize=10)
    ax.grid(False)

    ax.legend(
        fontsize=10,
        frameon=False,
        loc="upper right",
    )

    fig.tight_layout()
    fig.savefig(fig7_file, dpi=800, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {fig7_file}")

if __name__ == "__main__":
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    print("Running Gaussian rate-heterogeneous two-block simulations...")

    all_results = {}

    n_processes = min(max(1, cpu_count() - 1), n_runs)
    print(f"Using {n_processes} processes")

    for rho in rate_list:
        print(f"\n=== Running rho = {rho:g} ===")

        seeds = [
            base_seed + int(10000 * rho) + run
            for run in range(n_runs)
        ]

        args_list = [(rho, seed) for seed in seeds]

        curves_A = []
        curves_B = []
        curves_I = []
        curves_AMSD = []

        with Pool(processes=n_processes) as pool:
            for run_idx, result in enumerate(
                pool.imap_unordered(simulate_one_run, args_list),
                start=1,
            ):
                _, t_scaled, msd_A, msd_B, msd_interface, A_msd = result

                curves_A.append(msd_A)
                curves_B.append(msd_B)
                curves_I.append(msd_interface)
                curves_AMSD.append(A_msd)

                if run_idx % 10 == 0 or run_idx == n_runs:
                    print(f"  Finished {run_idx}/{n_runs} runs")

        curves_A = np.vstack(curves_A)
        curves_B = np.vstack(curves_B)
        curves_I = np.vstack(curves_I)
        curves_AMSD = np.vstack(curves_AMSD)

        all_results[rho] = {
            "t_scaled": t_scaled,
            "A_mean": curves_A.mean(axis=0),
            "B_mean": curves_B.mean(axis=0),
            "I_mean": curves_I.mean(axis=0),
            "AMSD_mean": curves_AMSD.mean(axis=0),
            "A_sem": curves_A.std(axis=0, ddof=1) / np.sqrt(n_runs),
            "B_sem": curves_B.std(axis=0, ddof=1) / np.sqrt(n_runs),
            "I_sem": curves_I.std(axis=0, ddof=1) / np.sqrt(n_runs),
            "AMSD_sem": curves_AMSD.std(axis=0, ddof=1) / np.sqrt(n_runs),
            "A_all": curves_A,
            "B_all": curves_B,
            "I_all": curves_I,
            "AMSD_all": curves_AMSD,
        }

    save_results(all_results)

    plot_fig6_from_saved_data()
    plot_fig7_from_saved_data()

    print("\nSaved data:")
    print(data_file)
    print(summary_file)

    print("\nSaved figures:")
    print(fig6_file)
    print(fig7_file)

    print("\nDone.")