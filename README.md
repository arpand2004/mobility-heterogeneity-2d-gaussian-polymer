# Mobility Heterogeneity in a 2D Gaussian Lattice Polymer

This repository contains simulation code, processed data, and generated figures associated with the article:

**Mobility Heterogeneity in a 2D Gaussian Lattice Polymer: A Dynamic Monte Carlo Study**

arXiv: https://arxiv.org/abs/2606.04002

The study uses dynamic Monte Carlo simulations of 2D Gaussian lattice polymers with block-dependent update rates.

## Repository structure

```text
.
├── data/
│   └── processed/
│       ├── block_msd_data.npz                    # processed data used to generate Figs. 6, 7, and 8
│       ├── dcm_N_data.npz                        # processed COM-MSD data used to generate Fig. 10
│       ├── block_com_msd_data.npz                # processed block-COM MSD data used to generate Fig. 9
│       ├── block_com_msd_summary.csv             # summary table for the block-COM MSD data
│       ├── block_com_diffusion_coefficients.csv  # extracted D_A, D_B, and D_cm values for Fig. 9
│       ├── Fig10_Dcm_values.npz                  # extracted D_cm values used in Fig. 10
│       └── Fig10_Dcm_values_summary.csv          # summary table for the extracted D_cm values
│
├── figures/
│   ├── Fig3.png                                  # homogeneous Gaussian monomer MSD benchmark
│   ├── Fig4.png                                  # homogeneous Gaussian D_cm vs N benchmark
│   ├── Fig6.png                                  # block-resolved MSDs for the Gaussian two-block model
│   ├── Fig7.png                                  # block-resolved MSDs of blocks A and B across rho = 1, 2, 4
│   ├── Fig8.png                                  # normalized MSD asymmetry time series A_MSD(t)
│   ├── Fig9.png                                  # block-A, block-B, and full-chain COM MSDs
│   └── Fig10.png                                 # D_cm vs N for the Gaussian two-block model
│
├── scripts/
│   ├── msd_homogeneous_benchmark.py              # full simulation script generating Fig. 3
│   ├── Dcm_homogeneous_benchmark.py              # full simulation script generating Fig. 4
│   ├── block_msd.py                              # full simulation script generating data for Figs. 6 and 8
│   ├── block_msd_2.py                            # plotting script generating Fig. 7 from block_msd_data.npz
│   ├── Dcm_N.py                                  # full simulation script generating data for Fig. 10
│   ├── msd_com.py                                # full simulation, diffusion extraction, and plotting script for Fig. 9
│   ├── plot_fig6_fig8.py                         # plotting script for Figs. 6 and 8 from block_msd_data.npz
│   └── plot_fig10.py                             # plotting and D_cm extraction script for Fig. 10 from dcm_N_data.npz
│
├── README.md                                     # repository description
└── requirements.txt                              # Python package requirements