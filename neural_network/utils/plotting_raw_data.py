import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


def plot_outputs_vs_progvar(
    df,
    output_cols,
    mask=None,
    progvar_col="ProgVar",
    color_col="TotalEnthalpy [J/kg]",
    save_name="Outputs_vs_ProgVar_centered",
    n_points=None
):
    # --- Local style parameters ---
    figsize = (18, 8)
    label_fs =10
    tick_fs = 7
    cbar_label_fs = 9
    marker_size = 20
    alpha = 1.0
    grid_alpha = 0.3
    cmap = plt.cm.magma
    dpi = 150

    # ==============================================================
    # Data preparation
    # ==============================================================
    cols_for_plot = [progvar_col] + output_cols + [color_col]

    if mask is not None:
        df_plot = df.loc[mask, cols_for_plot].reset_index(drop=True)
    else:
        df_plot = df[cols_for_plot].reset_index(drop=True)

    if n_points is not None and n_points < len(df_plot):
        idx = np.random.choice(len(df_plot), n_points, replace=False)
        df_plot = df_plot.iloc[idx]

    # ==============================================================
    # Labels
    # ==============================================================
    axis_labels_y = {
        "ProdRateProgVar [kg/m^3s]": r"$\dot{\omega}_c$ [kg/(m$^3$s)]",
        "temperature [K]": r"$T$ [K]",
        "Y-CO": r"$Y_{\mathrm{CO}}$ [-]",
        "density": r"$\rho$ [kg/m$^3$]",
        "mu [kg/ms]": r"$\mu$ [kg/(m\,s)]",
        "cp [J/kgK]": r"$c_p$ [J/(kg\,K)]",
        "Diff [kg/ms]": r"$D$ [kg/(m\,s)]",
    }

    x_label = r"Progress variable $c$ [-]"

    # ==============================================================
    # Figure (GridSpec layout: 2 rows x (4 plot cols + 1 colorbar col))
    # ==============================================================
    fig = plt.figure(figsize=figsize, dpi=dpi)

    # 5th column is reserved ONLY for the colorbar (prevents overlap)
    gs = fig.add_gridspec(
        2, 5,
        width_ratios=[1, 1, 1, 1, 0.06],
        wspace=0.35,
        hspace=0.35
    )

    # Create the 2x4 plot axes
    axes = []
    for r in range(2):
        for c in range(4):
            axes.append(fig.add_subplot(gs[r, c]))

    # Create a dedicated axis for the colorbar (full height)
    cax = fig.add_subplot(gs[:, 4])

    # ==============================================================
    # Colormap normalization (shared across all subplots)
    # ==============================================================
    cmin, cmax = df_plot[color_col].min(), df_plot[color_col].max()
    norm = mpl.colors.Normalize(vmin=cmin, vmax=cmax)

    # ==============================================================
    # Plot
    # ==============================================================
    sc = None  # will store the last scatter handle for the colorbar

    for i, y_name in enumerate(output_cols):
        ax = axes[i]

        sc = ax.scatter(
            df_plot[progvar_col],
            df_plot[y_name],
            c=df_plot[color_col],
            cmap=cmap,
            norm=norm,
            s=marker_size,
            alpha=alpha,
            edgecolors="none",
            rasterized=True,
        )

        ax.set_xlabel(x_label, fontsize=label_fs)
        ax.set_ylabel(axis_labels_y.get(y_name, y_name), fontsize=label_fs)
        ax.tick_params(labelsize=tick_fs)
        ax.grid(True, alpha=grid_alpha)

    # Hide any unused subplot(s) (e.g., 8th slot if you have 7 outputs)
    for k in range(len(output_cols), len(axes)):
        axes[k].set_visible(False)

    # ==============================================================
    # Colorbar (drawn in its own axis -> no overlap)
    # ==============================================================
    cbar = fig.colorbar(sc, cax=cax)
    cbar.set_label(r"$h$ [J/kg]", fontsize=cbar_label_fs)
    cbar.ax.tick_params(labelsize=tick_fs)

    # ==============================================================
    # Save
    # ==============================================================
    fig.savefig(f"{save_name}.pdf", bbox_inches="tight")
    fig.savefig(f"{save_name}.png", dpi=300, bbox_inches="tight")
    plt.show()
