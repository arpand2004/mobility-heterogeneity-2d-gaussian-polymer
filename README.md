# Mobility Heterogeneity in a 2D Gaussian Lattice Polymer

This repository contains simulation code, processed data, and generated figures for the preprint:

**Mobility Heterogeneity in a 2D Gaussian Lattice Polymer: A Dynamic Monte Carlo Study**

arXiv preprint: https://arxiv.org/abs/2606.04002 

The study uses dynamic Monte Carlo simulations of 2D Gaussian lattice polymers with block-dependent update rates.

## Repository structure

```text
.
├── data/
│   └── processed/
│       ├── block_msd_data.npz        # processed data used to generate Figs. 6 and 7
│       └── dcm_N_data.npz            # processed COM-MSD data used to extract and generate Fig. 8
│
├── figures/
│   ├── Fig3.png                      # homogeneous Gaussian monomer MSD benchmark
│   ├── Fig4.png                      # homogeneous Gaussian D_cm vs N benchmark
│   ├── Fig6.png                      # block-resolved MSDs for the Gaussian two-block model
│   ├── Fig7.png                      # normalized MSD asymmetry time series A_MSD(t)
│   └── Fig8.png                      # D_cm vs N for the Gaussian two-block model
│
├── scripts/
│   ├── msd_homogeneous_benchmark.py  # full simulation script generating Fig. 3
│   ├── Dcm_homogeneous_benchmark.py  # full simulation script generating Fig. 4
│   ├── block_msd.py                  # full simulation script generating data for Figs. 6 and 7
│   ├── Dcm_N.py                      # full simulation script generating data for Fig. 8
│   ├── plot_fig6_fig7.py             # plotting script for Figs. 6 and 7 from block_msd_data.npz
│   └── plot_fig8.py                  # plotting script for Fig. 8 from dcm_N_data.npz
│
├── README.md                         # repository description
└── requirements.txt                  # Python package requirements
