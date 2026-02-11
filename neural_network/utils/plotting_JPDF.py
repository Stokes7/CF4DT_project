import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatter
from sklearn.metrics import r2_score

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatter
from matplotlib.colors import LogNorm
from sklearn.metrics import r2_score

def plot_JPDF(df_true, df_pred, bins, x_label, y_label):
    """
    Plot Joint PDF (jPDF) using a log-scaled colorbar.
    Robust to empty bins, zeros, NaNs and degenerate vmin/vmax.
    """

    # Convert to 1D arrays
    df_true = np.asarray(df_true).ravel()
    df_pred = np.asarray(df_pred).ravel()

    # Remove NaN / Inf values
    mask = np.isfinite(df_true) & np.isfinite(df_pred)
    df_true = df_true[mask]
    df_pred = df_pred[mask]

    # 2D histogram
    hist, x_edges, y_edges = np.histogram2d(df_true, df_pred, density=True, bins=bins)

    # Use only positive bins for LogNorm
    positive_bins = hist[hist > 0]

    if positive_bins.size == 0:
        raise ValueError("All histogram bins are zero; log scale cannot be applied.")

    # Robust limits for log scale
    vmin = np.percentile(positive_bins, 5)
    vmax = np.percentile(positive_bins, 99)

    # Ensure valid bounds
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        raise ValueError(f"Non-finite vmin/vmax: vmin={vmin}, vmax={vmax}")

    if vmax <= vmin:
        vmax = max(np.max(positive_bins), vmin * 10.0)

    # Plot
    fig, ax = plt.subplots(figsize=(6, 5))

    im = ax.imshow(
        hist.T,
        origin="lower",
        extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
        aspect="auto",
        cmap="magma",
        norm=LogNorm(vmin=vmin, vmax=vmax),
        interpolation="nearest",
    )

    # Metrics
    denom = np.sum(df_true ** 2)
    nmse = np.sum((df_true - df_pred) ** 2) / denom if denom > 0 else np.nan
    r2 = r2_score(df_true, df_pred)

    print(f"R2: {r2:.4f}, NMSE: {nmse}")

    # Labels
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    # Colorbar
    #formatter = LogFormatter(labelOnlyBase=False)
    cbar = fig.colorbar(im, ax=ax)#, format=formatter)
    cbar.set_label("jPDF (log scale)")

    plt.tight_layout()
    plt.show()

