from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


data_file = Path("data/processed/block_msd_data.npz")

figure_dir = Path("figures")
png_file = figure_dir / "Fig7.png"

rate_list = [1.0, 2.0, 4.0]

rho_colors = {
    1.0: "#9bbfc1",
    2.0: "#b8a99a",
    4.0: "#c7a2b6",
}


def rho_key(rho):
    return str(rho).replace(".", "p")


def validate_saved_data(data):
    required_keys = ["N"]

    for rho in rate_list:
        key = rho_key(rho)
        required_keys.extend(
            [
                f"t_scaled_rho_{key}",
                f"A_mean_rho_{key}",
                f"A_sem_rho_{key}",
                f"B_mean_rho_{key}",
                f"B_sem_rho_{key}",
            ]
        )

    missing = [key for key in required_keys if key not in data.files]

    if missing:
        raise KeyError(
            "Missing required arrays:\n  " + "\n  ".join(missing)
        )


def add_slope_guides(ax, x_data, y_data, show_labels=True):
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

    y_anchor = np.percentile(y_data, 18)
    x_anchor = np.sqrt(x_min * x_max)

    y_half = y_anchor * 1.8 * (x_guide / x_anchor) ** 0.5
    y_one = y_anchor * 0.55 * (x_guide / x_anchor)

    ax.loglog(
        x_guide,
        y_half,
        linestyle="--",
        linewidth=1.4,
        color="0.60",
        label=r"slope $1/2$" if show_labels else "_nolegend_",
        zorder=1,
    )

    ax.loglog(
        x_guide,
        y_one,
        linestyle="-",
        linewidth=1.4,
        color="0.60",
        label=r"slope $1$" if show_labels else "_nolegend_",
        zorder=1,
    )


def get_global_plot_limits(data):
    all_x = []
    all_y = []

    for rho in rate_list:
        key = rho_key(rho)

        t = np.asarray(data[f"t_scaled_rho_{key}"], dtype=np.float64)
        A_mean = np.asarray(data[f"A_mean_rho_{key}"], dtype=np.float64)
        B_mean = np.asarray(data[f"B_mean_rho_{key}"], dtype=np.float64)

        for mean in (A_mean, B_mean):
            mask = (
                np.isfinite(t)
                & np.isfinite(mean)
                & (t > 0)
                & (mean > 0)
            )

            if np.any(mask):
                all_x.append(t[mask])
                all_y.append(mean[mask])

    if not all_x or not all_y:
        raise ValueError("No finite positive MSD data were found.")

    all_x = np.concatenate(all_x)
    all_y = np.concatenate(all_y)

    x_min = np.nanmin(all_x)
    x_max = np.nanmax(all_x)
    y_min = 0.65 * np.nanmin(all_y)
    y_max = 1.35 * np.nanmax(all_y)

    return x_min, x_max, y_min, y_max


def plot_block_msd(data):
    x_min, x_max, y_min, y_max = get_global_plot_limits(data)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(9.4, 4.65),
        sharex=True,
        sharey=True,
    )

    panels = [
        ("A", "block A"),
        ("B", "block B"),
    ]

    for panel_index, (ax, (prefix, title)) in enumerate(zip(axes, panels)):
        guide_x = []
        guide_y = []

        for rho in rate_list:
            key = rho_key(rho)

            t = np.asarray(
                data[f"t_scaled_rho_{key}"],
                dtype=np.float64,
            )
            mean = np.asarray(
                data[f"{prefix}_mean_rho_{key}"],
                dtype=np.float64,
            )
            sem = np.asarray(
                data[f"{prefix}_sem_rho_{key}"],
                dtype=np.float64,
            )

            mask = (
                np.isfinite(t)
                & np.isfinite(mean)
                & np.isfinite(sem)
                & (t > 0)
                & (mean > 0)
            )

            if not np.any(mask):
                continue

            t_plot = t[mask]
            mean_plot = mean[mask]
            sem_plot = sem[mask]

            lower = np.maximum(mean_plot - sem_plot, 1.0e-14)
            upper = mean_plot + sem_plot

            ax.loglog(
                t_plot,
                mean_plot,
                color=rho_colors[rho],
                linewidth=2.4,
                label=rf"$\rho={rho:g}$",
                zorder=4,
            )

            ax.fill_between(
                t_plot,
                lower,
                upper,
                color=rho_colors[rho],
                alpha=0.15,
                linewidth=0,
                zorder=3,
            )

            guide_x.append(t_plot)
            guide_y.append(mean_plot)

        if guide_x and guide_y:
            add_slope_guides(
                ax,
                np.concatenate(guide_x),
                np.concatenate(guide_y),
                show_labels=(panel_index == 0),
            )

        ax.set_title(title, fontsize=13)
        ax.set_xlabel(
            r"Scaled time, $t_{\mathrm{sweep}}/N^2$",
            fontsize=12,
        )
        ax.set_xlim(x_min * 0.85, x_max * 1.05)
        ax.set_ylim(y_min, y_max)
        ax.tick_params(axis="both", which="major", labelsize=10)
        ax.tick_params(axis="both", which="minor", labelsize=8)
        ax.grid(False)

    axes[0].set_ylabel(
        r"Block MSD, $\langle |\mathbf{r}_i(t)-\mathbf{r}_i(0)|^2\rangle$",
        fontsize=12,
    )

    handles, labels = axes[0].get_legend_handles_labels()

    axes[0].legend(
        handles,
        labels,
        fontsize=8.7,
        frameon=False,
        loc="upper left",
        handlelength=2.0,
        labelspacing=0.32,
    )

    fig.tight_layout(w_pad=1.6)

    figure_dir.mkdir(parents=True, exist_ok=True)

    fig.savefig(png_file, dpi=800, bbox_inches="tight")

    plt.close(fig)


def main():
    if not data_file.exists():
        raise FileNotFoundError(
            f"Could not find {data_file}. Run the script from the repository root."
        )

    with np.load(data_file, allow_pickle=False) as data:
        validate_saved_data(data)
        plot_block_msd(data)

    print(f"Saved {png_file}")

if __name__ == "__main__":
    main()
