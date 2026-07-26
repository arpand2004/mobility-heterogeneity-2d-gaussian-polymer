import os
import numpy as np
import matplotlib.pyplot as plt

data_file = "data/processed/dcm_N_data.npz"

fig10_file = "figures/Fig10.png"

dcm_output_file = "data/processed/Fig10_Dcm_values.npz"
dcm_summary_file = "data/processed/Fig10_Dcm_values_summary.csv"


N_list = [20, 40, 60, 80, 100]
rho_list = [1.0, 2.0, 4.0]

fit_scaled_min = 0.1
fit_scaled_max = 10.0

colors_N = {
    20: "#1f77b4",
    40: "#ff7f0e",
    60: "#2ca02c",
    80: "#d62728",
    100: "#9467bd",
}

rho_line_colors = {
    1.0: "#9bbfc1",
    2.0: "#b8a99a",
    4.0: "#c7a2b6",
}

rho_line_widths = {
    1.0: 2.8,
    2.0: 2.8,
    4.0: 2.8,
}

def rho_key(rho):
    return str(rho).replace(".", "p")


def extract_Dcm(t_sweep, t_scaled, com_msd, fit_scaled_min, fit_scaled_max):
    fit_mask = (t_scaled >= fit_scaled_min) & (t_scaled <= fit_scaled_max)

    if fit_mask.sum() < 5:
        raise ValueError("Not enough points inside the D_cm fitting window.")

    slope, intercept = np.polyfit(t_sweep[fit_mask], com_msd[fit_mask], 1)

    D_cm = slope / 4.0

    return D_cm, slope, intercept, fit_mask

if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    data = np.load(data_file)

    results_by_rho = {}
    summary_rows = []

    for rho in rho_list:
        rk = rho_key(rho)

        D_values = []
        D_errors = []
        D_all_runs_by_N = []

        for N in N_list:
            nk = str(N)

            t_scaled = data[f"t_scaled_rho_{rk}_N_{nk}"]
            t_sweep = data[f"t_sweep_rho_{rk}_N_{nk}"]
            com_msd_all = data[f"com_msd_all_rho_{rk}_N_{nk}"]

            com_msd_mean = com_msd_all.mean(axis=0)

            D_cm, slope, intercept, fit_mask = extract_Dcm(
                t_sweep,
                t_scaled,
                com_msd_mean,
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
            D_err = D_run_values.std(ddof=1) / np.sqrt(len(D_run_values))

            D_values.append(D_cm)
            D_errors.append(D_err)
            D_all_runs_by_N.append(D_run_values)

            summary_rows.append(
                [
                    rho,
                    N,
                    D_cm,
                    D_err,
                    fit_scaled_min,
                    fit_scaled_max,
                ]
            )

            print(f"rho={rho:g}, N={N}: D_cm = {D_cm:.6e} ± {D_err:.2e}")

        D_values = np.array(D_values, dtype=np.float64)
        D_errors = np.array(D_errors, dtype=np.float64)
        D_all_runs_by_N = np.vstack(D_all_runs_by_N)

        logN = np.log(np.array(N_list, dtype=np.float64))
        logD = np.log(D_values)

        slope_loglog, intercept_loglog = np.polyfit(logN, logD, 1)
        gamma = -slope_loglog

        results_by_rho[rho] = {
            "D_values": D_values,
            "D_errors": D_errors,
            "D_all_runs": D_all_runs_by_N,
            "fit_slope": slope_loglog,
            "fit_intercept": intercept_loglog,
            "gamma": gamma,
        }

        print(f"Scaling for rho={rho:g}: D_cm ~ N^(-{gamma:.3f})")

    npz_data = {
        "N_list": np.array(N_list),
        "rho_list": np.array(rho_list),
        "fit_scaled_min": np.array(fit_scaled_min),
        "fit_scaled_max": np.array(fit_scaled_max),
    }

    for rho in rho_list:
        rk = rho_key(rho)

        npz_data[f"D_values_rho_{rk}"] = results_by_rho[rho]["D_values"]
        npz_data[f"D_errors_rho_{rk}"] = results_by_rho[rho]["D_errors"]
        npz_data[f"D_all_runs_rho_{rk}"] = results_by_rho[rho]["D_all_runs"]
        npz_data[f"fit_slope_rho_{rk}"] = np.array(results_by_rho[rho]["fit_slope"])
        npz_data[f"fit_intercept_rho_{rk}"] = np.array(results_by_rho[rho]["fit_intercept"])
        npz_data[f"gamma_rho_{rk}"] = np.array(results_by_rho[rho]["gamma"])

    np.savez(dcm_output_file, **npz_data)

    np.savetxt(
        dcm_summary_file,
        np.array(summary_rows, dtype=np.float64),
        delimiter=",",
        header="rho,N,D_cm,D_cm_sem,fit_scaled_min,fit_scaled_max",
        comments="",
    )

    # Plot Fig. 10
    fig, ax = plt.subplots(figsize=(6.8, 5.0))

    N_guide = np.linspace(min(N_list), max(N_list), 300)

    for rho in rho_list:
        data_rho = results_by_rho[rho]

        D_values = data_rho["D_values"]
        D_errors = data_rho["D_errors"]

        D_fit = np.exp(data_rho["fit_intercept"]) * N_guide ** data_rho["fit_slope"]

        ax.loglog(
            N_guide,
            D_fit,
            linestyle="-",
            linewidth=rho_line_widths[rho],
            color=rho_line_colors[rho],
            label=fr"$\rho$ = {rho:g}, fit slope -{data_rho['gamma']:.2f}",
            zorder=2,
        )

        for idx, N in enumerate(N_list):
            ax.errorbar(
                N,
                D_values[idx],
                yerr=D_errors[idx],
                fmt="o",
                markersize=8.0,
                capsize=4,
                color=colors_N[N],
                markeredgecolor="black",
                markeredgewidth=0.7,
                linewidth=1.4,
                zorder=3,
            )

    rho_anchor = 1.0
    N_anchor = 40

    D_anchor = results_by_rho[rho_anchor]["D_values"][N_list.index(N_anchor)]
    D_guide = 0.75 * D_anchor * (N_guide / N_anchor) ** (-1)

    ax.loglog(
        N_guide,
        D_guide,
        linestyle="--",
        linewidth=2.0,
        color="gray",
        label=r"slope -1",
        zorder=1,
    )

    ax.set_xlabel(r"Chain length, $N$", fontsize=13)
    ax.set_ylabel(r"$D_{\mathrm{cm}}$", fontsize=14)

    ax.tick_params(axis="both", which="major", labelsize=11)
    ax.tick_params(axis="both", which="minor", labelsize=9)

    ax.set_xlim(18, 110)

    ax.legend(
        fontsize=9,
        frameon=False,
        loc="lower left",
    )

    ax.grid(False)

    fig.tight_layout()
    fig.savefig(fig10_file, dpi=800, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    print("\nSaved figure:")
    print(fig10_file)

    print("\nSaved extracted Dcm data:")
    print(dcm_output_file)
    print(dcm_summary_file)

    print("\nDone.")
