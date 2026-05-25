import os
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count

L = 500
N_list = [20, 40, 60, 80, 100]
n_runs = 100 
max_scaled_time = 20.0
n_sample_points = 700

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


def run_msd_gaussian_single(args):
    L, N, max_scaled_time, n_sample_points, seed = args

    rng = np.random.default_rng(seed)
    chain = init_chain(L, N, rng)
    r0 = chain.copy()

    n_sweeps = int(np.ceil(max_scaled_time * N**2))

    sample_sweeps = np.unique(
        np.round(
            np.logspace(0, np.log10(n_sweeps), n_sample_points)
        ).astype(int)
    )
    sample_sweeps = sample_sweeps[sample_sweeps >= 1]

    msd = np.empty(len(sample_sweeps), dtype=np.float64)

    sample_idx = 0
    for sweep in range(1, n_sweeps + 1):

        for _ in range(N):
            mc_attempt(chain, N, L, rng)

        if sample_idx < len(sample_sweeps) and sweep == sample_sweeps[sample_idx]:
            disp = chain - r0
            sq = disp[:, 0]**2 + disp[:, 1]**2
            msd[sample_idx] = sq.mean()
            sample_idx += 1

    t_scaled = sample_sweeps / (N**2)

    return N, t_scaled, msd


def interp_loglog(x, y, x0):
    mask = (x > 0) & (y > 0)
    x = x[mask]
    y = y[mask]

    if x0 <= x.min():
        return y[0]
    if x0 >= x.max():
        return y[-1]

    return np.exp(np.interp(np.log(x0), np.log(x), np.log(y)))


if __name__ == "__main__":

    os.makedirs("figures", exist_ok=True)

    all_results = {}

    n_processes = min(max(1, cpu_count() - 1), n_runs)
    print(f"Using {n_processes} processes")

    print("\nThree-monomer dictionary:")
    for key, value in THREE_MONOMER_DICT.items():
        print(f"  {key}: {value}")

    for N in N_list:
        print(f"\n=== Dictionary visual test: N = {N} ===")

        seeds = [base_seed + 1000 * N + run for run in range(n_runs)]
        args_list = [
            (L, N, max_scaled_time, n_sample_points, seed)
            for seed in seeds
        ]

        curves = []

        with Pool(processes=n_processes) as pool:
            results = list(pool.map(run_msd_gaussian_single, args_list))

        for _, t_scaled, msd in results:
            curves.append(msd)

        msd_all = np.vstack(curves)
        msd_avg = msd_all.mean(axis=0)

        all_results[N] = {
            "t_scaled": t_scaled,
            "msd_avg": msd_avg,
        }

    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    for N in N_list:
        ax.loglog(
            all_results[N]["t_scaled"],
            all_results[N]["msd_avg"],
            color=colors[N],
            linewidth=2.2,
            label=fr"$N={N}$",
            zorder=3,
        )

    N_ref = 60
    x_ref = all_results[N_ref]["t_scaled"]
    y_ref = all_results[N_ref]["msd_avg"]

    all_x = np.concatenate([all_results[N]["t_scaled"] for N in N_list])
    x_min = all_x[all_x > 0].min()
    x_max = all_x.max()

    xguide = np.logspace(
        np.log10(x_min * 1.3),
        np.log10(x_max / 1.15),
        250
    )

    x_anchor_half = 2.0e-3
    y_anchor_half = 0.75 * interp_loglog(x_ref, y_ref, x_anchor_half)
    A_half = y_anchor_half / (x_anchor_half**0.5)
    yguide_half = A_half * xguide**0.5

    ax.loglog(
        xguide,
        yguide_half,
        linestyle="--",
        linewidth=2.0,
        color="0.55",
        label=r"slope $1/2$",
        zorder=2,
    )

    x_anchor_lin = 3.0e-2
    y_anchor_lin = 0.28 * interp_loglog(x_ref, y_ref, x_anchor_lin)
    A_lin = y_anchor_lin / x_anchor_lin
    yguide_lin = A_lin * xguide

    ax.loglog(
        xguide,
        yguide_lin,
        linestyle="-",
        linewidth=2.2,
        color="0.55",
        label=r"slope $1$",
        zorder=1,
    )

    ax.set_xlabel(r"Scaled time, $t_{\mathrm{sweep}}/N^2$", fontsize=13)
    ax.set_ylabel(
        r"Monomer MSD, $\langle |\mathbf{r}_i(t)-\mathbf{r}_i(0)|^2\rangle$",
        fontsize=13,
    )

    ax.tick_params(axis="both", which="major", labelsize=11)
    ax.tick_params(axis="both", which="minor", labelsize=9)

    ax.legend(
        fontsize=10,
        frameon=False,
        loc="lower right",
    )

    ax.grid(False)

    fig.tight_layout()

    fig.savefig("figures/Fig3.png", dpi=800, bbox_inches="tight")

    plt.show()
