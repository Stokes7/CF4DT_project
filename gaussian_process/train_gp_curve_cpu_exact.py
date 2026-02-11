# train_gp_curve_cpu_exact.py
# ------------------------------------------------------------
# Exact Gaussian Process Regression (RBF) for 2D -> 7D.
# Uses fixed split indices (same as NN) and variable N_train.
# Saves the same metrics/timing/peak-memory into one JSON per run.
#
# IMPORTANT: Exact GP scales poorly (O(N^3) time, O(N^2) memory).
# Use only for small N_train unless you have huge RAM.
# ------------------------------------------------------------

import os
import json
import time
import glob as gb
import tempfile
import resource

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler


# =========================
# Transforms (same as NN)
# =========================
def signed_log1p(y: np.ndarray) -> np.ndarray:
    return np.sign(y) * np.log1p(np.abs(y))

def signed_expm1(y: np.ndarray) -> np.ndarray:
    return np.sign(y) * np.expm1(np.abs(y))


# =========================
# Memory helpers (CPU)
# =========================
def cpu_peak_rss_bytes() -> int:
    # Linux: ru_maxrss in KiB
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)

def bytes_to_mb(b: int) -> float:
    return float(b) / (1024**2)


# =========================
# Metrics (same structure as NN)
# =========================
def compute_metrics(Y_true: np.ndarray, Y_pred: np.ndarray, eps: float = 1e-12) -> dict:
    err = Y_pred - Y_true

    rmse = np.sqrt(np.mean(err**2, axis=0))
    mae  = np.mean(np.abs(err), axis=0)

    std_true = Y_true.std(axis=0) + eps
    rrmse = rmse / std_true

    ss_res = np.sum((Y_true - Y_pred)**2, axis=0)
    ss_tot = np.sum((Y_true - Y_true.mean(axis=0))**2, axis=0) + eps
    r2 = 1.0 - ss_res / ss_tot

    p95_abs = np.percentile(np.abs(err), 95, axis=0)
    max_abs = np.max(np.abs(err), axis=0)

    return {
        "rmse": rmse.tolist(),
        "rrmse": rrmse.tolist(),
        "mae": mae.tolist(),
        "r2": r2.tolist(),
        "p95_abs": p95_abs.tolist(),
        "max_abs": max_abs.tolist(),
        "rmse_mean": float(rmse.mean()),
        "rrmse_mean": float(rrmse.mean()),
        "mae_mean": float(mae.mean()),
        "r2_mean": float(r2.mean()),
        "p95_abs_mean": float(p95_abs.mean()),
        "max_abs_mean": float(max_abs.mean()),
    }


# =========================
# Data loading (same as NN)
# =========================
def load_data(data_glob: str, referenceEnthalpy: float = 276240.0):
    files = sorted(gb.glob(data_glob))
    if len(files) == 0:
        raise FileNotFoundError(f"No files matched: {data_glob}")

    with open(files[0], "r") as f:
        lines = f.readlines()
        column_names = lines[1].strip().split("\t")

    data = []
    for fn in files:
        df_tmp = pd.DataFrame(
            np.loadtxt(fn, skiprows=2, dtype=np.float64),
            columns=column_names
        )
        data.append(df_tmp)

    df = pd.concat(data, ignore_index=True)

    # Derived diffusivity
    df["Diff [kg/ms]"] = df["lambda [W/mK]"] / (df["cp [J/kgK]"] + 1e-30)

    # Shift enthalpy
    df["TotalEnthalpy [J/kg]"] = df["TotalEnthalpy [J/kg]"] - referenceEnthalpy

    input_data = ["ProgVar", "TotalEnthalpy [J/kg]"]
    output_data = [
        "ProdRateProgVar [kg/m^3s]",
        "temperature [K]",
        "Y-CO",
        "density",
        "mu [kg/ms]",
        "cp [J/kgK]",
        "Diff [kg/ms]"
    ]

    X_all = df[input_data].to_numpy(dtype=np.float32)
    Y_all_raw = df[output_data].to_numpy(dtype=np.float32)

    return X_all, Y_all_raw, input_data, output_data


# =========================
# Exact GP building blocks
# =========================
def rbf_kernel_matrix(Xa: np.ndarray, Xb: np.ndarray, sigma_f: float, l: float) -> np.ndarray:
    """
    Vectorized RBF kernel matrix.
    Xa: (Na, d), Xb: (Nb, d)
    Returns K: (Na, Nb) float64
    """
    Xa = np.asarray(Xa, dtype=np.float64)
    Xb = np.asarray(Xb, dtype=np.float64)

    # squared distances: ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a·b
    a2 = np.sum(Xa**2, axis=1, keepdims=True)          # (Na, 1)
    b2 = np.sum(Xb**2, axis=1, keepdims=True).T        # (1, Nb)
    sqdist = a2 + b2 - 2.0 * (Xa @ Xb.T)                # (Na, Nb)

    K = (sigma_f**2) * np.exp(-0.5 * sqdist / (l**2))
    return K

def gp_fit_exact(X_train, Y_train, sigma_f, l, sigma_n):
    """
    Training phase:
      K = K(X_train, X_train) + sigma_n^2 I
      L = chol(K)
      alpha = K^{-1} Y via triangular solves
    Returns:
      L (N,N), alpha (N,d), and optionally K_bytes for model-size
    """
    N = X_train.shape[0]

    K = rbf_kernel_matrix(X_train, X_train, sigma_f=sigma_f, l=l)  # (N,N)
    K[np.diag_indices(N)] += (sigma_n**2)

    L = np.linalg.cholesky(K)                # (N,N)
    v = np.linalg.solve(L, Y_train)          # (N,d)
    alpha = np.linalg.solve(L.T, v)          # (N,d)

    return L, alpha, K  # keep K only if you want model size; otherwise return K.nbytes


def gp_predict_mean(X_test, X_train, alpha, sigma_f, l):
    """
    Inference phase:
      mean = K_*(X_test, X_train) @ alpha
    """
    K_star = rbf_kernel_matrix(X_test, X_train, sigma_f=sigma_f, l=l)  # (N*,N)
    return K_star @ alpha


# def gp_posterior_mean_exact(X_train, Y_train, X_test, sigma_f, l, sigma_n):
#     """
#     Exact GP posterior mean for multi-output Y_train (N,7).
#     Returns mean_test (N*,7).

#     Uses Cholesky for stability:
#       L L^T = K + sigma_n^2 I
#       alpha = (K + sigma_n^2 I)^-1 Y via triangular solves
#       mean  = K_* alpha
#     """
#     N = X_train.shape[0]

#     # Build K and K_*
#     K = rbf_kernel_matrix(X_train, X_train, sigma_f=sigma_f, l=l)   # (N, N)
#     K[np.diag_indices(N)] += (sigma_n**2)                            # add noise/jitter

#     # Cholesky
#     L = np.linalg.cholesky(K)  # (N,N)

#     # Solve for alpha: (K)^-1 Y = L^-T L^-1 Y
#     # First solve L v = Y
#     v = np.linalg.solve(L, Y_train)          # (N,7)
#     # Then solve L^T alpha = v
#     alpha = np.linalg.solve(L.T, v)          # (N,7)

#     # Cross-covariance
#     K_star = rbf_kernel_matrix(X_test, X_train, sigma_f=sigma_f, l=l)  # (N*, N)

#     mean_star = K_star @ alpha  # (N*, 7)
#     return mean_star, K  # return K so we can estimate model "size" if desired


# =========================
# Main experiment
# =========================
def main():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_glob", type=str, default="data/chemtable_FVV_2D_Enthalpy/*.kg")
    parser.add_argument("--splits_dir", type=str, default="./splits_uq")
    parser.add_argument("--out_dir", type=str, default="./gp_results")

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_train", type=int, required=True)

    # GP hyperparameters (fixed for now)
    parser.add_argument("--l", type=float, default=0.2)
    parser.add_argument("--sigma_f", type=float, default=1.0)
    parser.add_argument("--sigma_n", type=float, default=1e-3)

    # Safety: prevent accidental huge exact GP
    parser.add_argument("--max_exact_n", type=int, default=90000)

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Reproducibility
    np_rng = np.random.default_rng(args.seed)

    # -------------------------
    # Load full dataset
    # -------------------------
    t0 = time.time()
    X_all, Y_all_raw, input_data, output_data = load_data(args.data_glob)
    t_load = time.time() - t0

    # Same transform as NN (so metrics are comparable in same space)
    Y_all = signed_log1p(Y_all_raw).astype(np.float64)

    # -------------------------
    # Load fixed splits + subset indices (same as NN)
    # -------------------------
    test_idx = np.load(os.path.join(args.splits_dir, "test_idx.npy"))
    subset_path = os.path.join(args.splits_dir, f"train_subsets_seed{args.seed}.npz")
    train_subsets = np.load(subset_path)

    key = f"train_{args.n_train}"
    if key not in train_subsets.files:
        raise KeyError(f"{key} not found in {subset_path}. Available: {train_subsets.files}")

    train_idx = train_subsets[key]

    if len(train_idx) > args.max_exact_n:
        raise RuntimeError(
            f"Exact GP is too expensive for N_train={len(train_idx)}. "
            f"Set --max_exact_n higher ONLY if you know your RAM/time can handle it. "
            f"Consider Sparse GP for large N."
        )

    # Fixed test
    X_test = X_all[test_idx].astype(np.float64)
    Y_test = Y_all[test_idx].astype(np.float64)

    X_train = X_all[train_idx].astype(np.float64)
    Y_train = Y_all[train_idx].astype(np.float64)

    # -------------------------
    # Scale based on TRAIN only (like NN)
    # -------------------------
    x_scaler = StandardScaler().fit(X_train)
    y_scaler = StandardScaler().fit(Y_train)

    X_train_s = x_scaler.transform(X_train)
    X_test_s  = x_scaler.transform(X_test)

    Y_train_s = y_scaler.transform(Y_train)  # GP trained in scaled-Y space
    Y_test_s  = y_scaler.transform(Y_test)

    # Peak memory baseline
    _ = cpu_peak_rss_bytes()

    # -------------------------
    # "Training" timing = build K + Cholesky + alpha (exact GP solve)
    # -------------------------
    t1 = time.time()
    L, alpha, K_train = gp_fit_exact(
        X_train_s, Y_train_s,
        sigma_f=args.sigma_f, l=args.l, sigma_n=args.sigma_n
    )
    t_train = time.time() - t1

    t2 = time.time()
    mean_test_s = gp_predict_mean(
        X_test_s, X_train_s, alpha,
        sigma_f=args.sigma_f, l=args.l
    )
    t_infer = time.time() - t2

    cpu_rss_peak_bytes = cpu_peak_rss_bytes()

    # -------------------------
    # Inference timing
    # For exact GP, mean computation already includes K_* @ alpha.
    # We'll measure "infer" separately by recomputing only mean using precomputed alpha
    # would require refactor; to keep it simple, we define infer = 0 here and use t_train
    # as "train+infer". But for apples-to-apples with NN, let's approximate infer time:
    # build K_* and multiply (dominant part for inference).
    # -------------------------
    # Recompute inference time only (K_* and matmul), using stored Cholesky factors is possible
    # but we didn't keep them. We'll do a lightweight re-run that isolates K_*@alpha:
    # To do it properly we need alpha; easiest: compute alpha again but ignore its time is hard.
    # So we instead report:
    #   train_total = build+factor+solve+mean
    #   infer_total = 0.0 (and note definition)

    # Unscale prediction back to transformed (signed_log1p) Y space
    Y_pred_t = y_scaler.inverse_transform(mean_test_s)

    # Convert BOTH prediction and ground-truth back to PHYSICAL space
    Y_pred_phys = signed_expm1(Y_pred_t)   # back to physical units
    Y_test_phys = signed_expm1(Y_test)     # Y_test is in signed_log1p space

    # Metrics computed in PHYSICAL space
    metrics = compute_metrics(Y_test_phys, Y_pred_phys)
    


    # -------------------------
    # "Model size" proxies for GP
    # For exact GP, the heavy object is K_train (N x N).
    # -------------------------
    N = len(train_idx)
    K_bytes = int(K_train.nbytes)  # float64 array size in bytes

    result = {
        "model": "gp_exact_rbf_cpu",
        "seed": int(args.seed),
        "n_train": int(N),
        "n_test": int(len(test_idx)),
        "device": "cpu",
        "hyperparams": {
            "kernel": "RBF",
            "l": float(args.l),
            "sigma_f": float(args.sigma_f),
            "sigma_n": float(args.sigma_n),
            "x_scaler": "StandardScaler",
            "y_scaler": "StandardScaler",
            "transform": "signed_log1p",
        },
        "timing_sec": {
            "data_load": float(t_load),
            "train": float(t_train),
            "infer": float(t_infer),
            "note": "For exact GP, 'train' includes building K, Cholesky/solve, and computing mean on test.",
        },
        "model_size": {
            "K_train_bytes": int(K_bytes),
            "K_train_mb": bytes_to_mb(K_bytes),
            "note": "Exact GP stores a dense N x N kernel matrix (dominant memory).",
        },
        "peak_memory": {
            "cpu_rss_peak_bytes": int(cpu_rss_peak_bytes),
            "cpu_rss_peak_mb": bytes_to_mb(cpu_rss_peak_bytes),
        },
        "metrics": metrics,
        "outputs": output_data,
    }

    out_path = os.path.join(args.out_dir, f"gp_exact_seed{args.seed}_N{N}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Saved: {out_path}")
    print(f"rrmse_mean={result['metrics']['rrmse_mean']:.6f} | r2_mean={result['metrics']['r2_mean']:.6f}")
    print(f"K_train_mb={result['model_size']['K_train_mb']:.1f} | cpu_rss_peak_mb={result['peak_memory']['cpu_rss_peak_mb']:.1f}")


if __name__ == "__main__":
    main()
