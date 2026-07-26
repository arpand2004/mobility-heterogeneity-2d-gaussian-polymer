import os
import numpy as np
import matplotlib.pyplot as plt

data_file = "data/processed/block_msd_data.npz"

fig6_file = "figures/Fig6.png"
fig8_file = "figures/Fig8.png"

rate_list = [1.0, 2.0, 4.0]

color_A = "#ff7f0e"
color_B = "#1f77b4"
color_interface = "black"

rho_colors = {
    1.0: "#9bbfc1",
    2.0: "#b8a99a",
    4.0: "#c7a2b6",
}


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

# Fig. 6: block MSDs

def plot_fig6(data):
    N = int(data["N"])

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

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(fig6_file, dpi=800, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    print(f"Saved: {fig6_file}")

# Fig. 8: A_MSD(t)

def plot_fig8(data):
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

        if f"AMSD_mean_rho_{key}" in data.files:
            A_curve = data[f"AMSD_mean_rho_{key}"]
        else:
            A_mean = data[f"A_mean_rho_{key}"]
            B_mean = data[f"B_mean_rho_{key}"]
            denom = A_mean + B_mean
            A_curve = np.zeros_like(denom)
            mask = denom > 0
            A_curve[mask] = (A_mean[mask] - B_mean[mask]) / denom[mask]

        if f"AMSD_sem_rho_{key}" in data.files:
            A_err = data[f"AMSD_sem_rho_{key}"]
        else:
            A_err = np.zeros_like(A_curve)

        ax.semilogx(
            t,
            A_curve,
            color=rho_colors[rho],
            linewidth=2.1,
            label=rf"$\rho={rho:g}$",
            zorder=3,
        )

        if np.any(A_err > 0):
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
    fig.savefig(fig8_file, dpi=800, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    print(f"Saved: {fig8_file}")

if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)

    data = np.load(data_file)

    plot_fig6(data)
    plot_fig8(data)

    print("Done.")
