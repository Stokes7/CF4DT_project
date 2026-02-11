# train_nn_curve_cpu.py
# ------------------------------------------------------------
# Train an MLP surrogate (2D -> 7D) using fixed split indices
# and variable training set sizes. Saves metrics + timing +
# peak CPU memory + model size into one JSON per run.
#
# CPU-only version (no CUDA).
# ------------------------------------------------------------

import os
import json
import time
import glob as gb
import tempfile
import resource

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.preprocessing import StandardScaler


# =========================
# Transforms
# =========================
def signed_log1p(y: np.ndarray) -> np.ndarray:
    return np.sign(y) * np.log1p(np.abs(y))

def signed_expm1(y: np.ndarray) -> np.ndarray:
    return np.sign(y) * np.expm1(np.abs(y))


# =========================
# Memory + model size (CPU)
# =========================
def cpu_peak_rss_bytes() -> int:
    """
    Peak RSS for this process (Linux): ru_maxrss is in KiB.
    """
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)

def model_num_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

def model_param_bytes(model: nn.Module) -> int:
    return sum(p.numel() * p.element_size() for p in model.parameters())

def model_checkpoint_size_bytes(model: nn.Module) -> int:
    """
    Size on disk of the state_dict checkpoint (bytes).
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp:
        path = tmp.name
    torch.save(model.state_dict(), path)
    size = os.path.getsize(path)
    os.remove(path)
    return int(size)

def bytes_to_mb(b: int) -> float:
    return float(b) / (1024**2)


# =========================
# Metrics
# =========================
def compute_metrics(Y_true: np.ndarray, Y_pred: np.ndarray, eps: float = 1e-12) -> dict:
    """
    Y_true, Y_pred: (N, d)
    Returns per-output metrics and aggregated summaries.
    """
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

    out = {
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
    return out


# =========================
# Model
# =========================
class MLP(nn.Module):
    def __init__(self, n_inputs=2, n_outputs=7, width=64, depth=3):
        super().__init__()
        layers = [nn.Linear(n_inputs, width), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.SiLU()]
        layers += [nn.Linear(width, n_outputs)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def run_epoch(model, loader, loss_fn, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total = 0.0
    n = 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for xb, yb in loader:
            if is_train:
                optimizer.zero_grad()

            pred = model(xb)
            loss = loss_fn(pred, yb)

            if is_train:
                loss.backward()
                optimizer.step()

            bs = xb.size(0)
            total += loss.item() * bs
            n += bs

    return total / max(n, 1)


@torch.no_grad()
def predict(model, X_t, batch_size=8192):
    model.eval()
    preds = []
    N = X_t.shape[0]
    for i in range(0, N, batch_size):
        xb = X_t[i:i+batch_size]
        preds.append(model(xb).cpu().numpy())
    return np.vstack(preds)


# =========================
# Data loading
# =========================
def load_data(data_glob: str, referenceEnthalpy: float = 276240.0):
    """
    Loads FlameMaster tables, creates Diff, shifts enthalpy, returns X and Y.

    Returns:
      X_all: (N,2)
      Y_all_raw: (N,7) in physical units
      input_data, output_data: column names
    """
    files = sorted(gb.glob(data_glob))  # IMPORTANT: sorted for reproducibility
    if len(files) == 0:
        raise FileNotFoundError(f"No files matched: {data_glob}")

    # Column names from 2nd line
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

    # Derived diffusivity (avoid divide by zero)
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
# Main
# =========================
def main():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_glob", type=str, default="data/chemtable_FVV_2D_Enthalpy/*.kg")
    parser.add_argument("--splits_dir", type=str, default="./splits_uq")

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_train", type=int, required=True)

    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=30)

    parser.add_argument("--val_frac", type=float, default=0.1)   # from the chosen train subset
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--depth", type=int, default=3)

    parser.add_argument("--out_dir", type=str, default="./nn_results")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Force CPU
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    device = "cpu"

    # Reproducibility
    np_rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    # -------------------------
    # Load full dataset
    # -------------------------
    t0 = time.time()
    X_all, Y_all_raw, input_data, output_data = load_data(args.data_glob)
    t_load = time.time() - t0

    # Apply your output transform (signed log1p)
    Y_all = signed_log1p(Y_all_raw)

    # -------------------------
    # Load fixed splits + subset indices
    # -------------------------
    test_idx = np.load(os.path.join(args.splits_dir, "test_idx.npy"))
    subset_path = os.path.join(args.splits_dir, f"train_subsets_seed{args.seed}.npz")
    train_subsets = np.load(subset_path)

    key = f"train_{args.n_train}"
    if key not in train_subsets.files:
        raise KeyError(f"{key} not found in {subset_path}. Available: {train_subsets.files}")

    train_idx_full = train_subsets[key]  # indices into the ORIGINAL dataset

    # Train/val split from train_idx_full (reproducible for this seed)
    n_val = int(args.val_frac * len(train_idx_full))
    perm_local = np_rng.permutation(len(train_idx_full))
    val_local = perm_local[:n_val]
    tr_local  = perm_local[n_val:]

    train_idx = train_idx_full[tr_local]
    val_idx   = train_idx_full[val_local]

    # Fixed test
    X_test = X_all[test_idx]
    Y_test = Y_all[test_idx]

    X_train = X_all[train_idx]
    Y_train = Y_all[train_idx]

    X_val = X_all[val_idx]
    Y_val = Y_all[val_idx]

    # -------------------------
    # Scale based on TRAIN only
    # -------------------------
    x_scaler = StandardScaler().fit(X_train)
    y_scaler = StandardScaler().fit(Y_train)

    X_train_s = x_scaler.transform(X_train)
    X_val_s   = x_scaler.transform(X_val)
    X_test_s  = x_scaler.transform(X_test)

    Y_train_s = y_scaler.transform(Y_train)
    Y_val_s   = y_scaler.transform(Y_val)
    Y_test_s  = y_scaler.transform(Y_test)

    # Torch tensors (CPU)
    X_train_t = torch.from_numpy(X_train_s).float()
    Y_train_t = torch.from_numpy(Y_train_s).float()
    X_val_t   = torch.from_numpy(X_val_s).float()
    Y_val_t   = torch.from_numpy(Y_val_s).float()
    X_test_t  = torch.from_numpy(X_test_s).float()
    Y_test_t  = torch.from_numpy(Y_test_s).float()

    train_loader = DataLoader(
        TensorDataset(X_train_t, Y_train_t),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False
    )
    val_loader = DataLoader(
        TensorDataset(X_val_t, Y_val_t),
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False
    )

    # -------------------------
    # Model + optimizer
    # -------------------------
    model = MLP(n_inputs=2, n_outputs=7, width=args.width, depth=args.depth).to(device)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Record model size info
    n_params = model_num_params(model)
    param_bytes = model_param_bytes(model)
    ckpt_bytes = model_checkpoint_size_bytes(model)

    # Peak memory baseline (CPU)
    _ = cpu_peak_rss_bytes()  # touch once

    # -------------------------
    # Train with early stopping
    # -------------------------
    best_val = float("inf")
    best_state = None
    bad_epochs = 0

    t1 = time.time()
    train_losses, val_losses = [], []

    for epoch in range(1, args.epochs + 1):
        tr_loss = run_epoch(model, train_loader, loss_fn, optimizer=optimizer)
        va_loss = run_epoch(model, val_loader,   loss_fn, optimizer=None)

        train_losses.append(tr_loss)
        val_losses.append(va_loss)

        if va_loss < best_val - 1e-10:
            best_val = va_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= args.patience:
            break

    t_train = time.time() - t1
    n_epochs_ran = len(train_losses)

    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)

    # Peak memory after training (CPU)
    cpu_rss_peak_bytes = cpu_peak_rss_bytes()

    # -------------------------
    # Inference timing + predictions
    # -------------------------
    t2 = time.time()
    Y_pred_s = predict(model, X_test_t, batch_size=8192)  # scaled space
    t_infer = time.time() - t2

    # Inverse scaling back to transformed Y space
    Y_pred = y_scaler.inverse_transform(Y_pred_s)

    # Compute metrics in transformed Y space
    metrics = compute_metrics(Y_test, Y_pred)

    # -------------------------
    # Save results JSON
    # -------------------------
    result = {
        "model": "nn_mlp_cpu",
        "seed": int(args.seed),
        "n_train": int(args.n_train),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "device": "cpu",
        "hyperparams": {
            "epochs_max": int(args.epochs),
            "epochs_ran": int(n_epochs_ran),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "patience": int(args.patience),
            "width": int(args.width),
            "depth": int(args.depth),
            "val_frac": float(args.val_frac),
            "transform": "signed_log1p",
            "x_scaler": "StandardScaler",
            "y_scaler": "StandardScaler",
        },
        "timing_sec": {
            "data_load": float(t_load),
            "train": float(t_train),
            "infer": float(t_infer),
            "infer_per_sample_ms": float(1000.0 * t_infer / max(len(test_idx), 1)),
        },
        "model_size": {
            "num_params": int(n_params),
            "param_bytes": int(param_bytes),
            "checkpoint_bytes": int(ckpt_bytes),
            "param_mb": bytes_to_mb(param_bytes),
            "checkpoint_mb": bytes_to_mb(ckpt_bytes),
        },
        "peak_memory": {
            "cpu_rss_peak_bytes": int(cpu_rss_peak_bytes),
            "cpu_rss_peak_mb": bytes_to_mb(cpu_rss_peak_bytes),
        },
        "metrics": metrics,
        "outputs": output_data,
    }

    out_path = os.path.join(args.out_dir, f"nn_cpu_seed{args.seed}_N{args.n_train}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Saved: {out_path}")
    print(f"rrmse_mean={result['metrics']['rrmse_mean']:.6f} | r2_mean={result['metrics']['r2_mean']:.6f}")
    print(f"cpu_rss_peak_mb={result['peak_memory']['cpu_rss_peak_mb']:.2f} | ckpt_mb={result['model_size']['checkpoint_mb']:.2f}")


if __name__ == "__main__":
    main()
