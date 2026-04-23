from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


DEFAULT_FEATURE_COLS = ["month", "hour", "wind speed (m/s)", "wind direction (°)"]
DEFAULT_TARGET_COL = "lv activepower (kw)"


@dataclass
class LSTMDataBundle:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    ws_train: np.ndarray
    ws_test: np.ndarray
    th_train: np.ndarray
    th_test: np.ndarray
    X_train_s: np.ndarray
    X_test_s: np.ndarray
    y_train_s: np.ndarray
    y_test_s: np.ndarray
    x_scaler: StandardScaler
    y_scaler: StandardScaler
    train_loader: DataLoader
    test_loader: DataLoader
    X_train_t: torch.Tensor
    X_test_t: torch.Tensor
    y_train_t: torch.Tensor
    y_test_t: torch.Tensor


class WindLSTMRegressor(nn.Module):
    """
    Input shape: (batch, seq_len, input_dim)

    Architecture:
    LSTM(input_dim, 64)
    LSTM(64, 128)
    Take last time-step output
    Linear(128, 64)
    ReLU
    Linear(64, 1)
    """

    def __init__(
        self,
        input_dim: int = 4,
        hidden_dim1: int = 64,
        hidden_dim2: int = 128,
        fc_dim: int = 64,
    ) -> None:
        super().__init__()

        self.lstm1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim1,
            num_layers=1,
            batch_first=True,
        )
        self.lstm2 = nn.LSTM(
            input_size=hidden_dim1,
            hidden_size=hidden_dim2,
            num_layers=1,
            batch_first=True,
        )

        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim2, fc_dim),
            nn.ReLU(),
            nn.Linear(fc_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out1, _ = self.lstm1(x)
        out2, _ = self.lstm2(out1)
        last_out = out2[:, -1, :]
        yhat = self.regressor(last_out)
        return yhat


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_sequences(
    X: np.ndarray,
    y: np.ndarray,
    wind_speed_all: np.ndarray,
    theoretical_all: np.ndarray,
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_seq = []
    y_seq = []
    ws_seq = []
    th_seq = []

    for i in range(seq_len - 1, len(X)):
        X_seq.append(X[i - seq_len + 1 : i + 1])
        y_seq.append(y[i])
        ws_seq.append(wind_speed_all[i])
        th_seq.append(theoretical_all[i])

    return (
        np.array(X_seq, dtype=np.float32),
        np.array(y_seq, dtype=np.float32),
        np.array(ws_seq, dtype=np.float32),
        np.array(th_seq, dtype=np.float32),
    )


def load_and_preprocess_lstm_data(
    data_path: str | Path,
    feature_cols: list[str] | None = None,
    target_col: str = DEFAULT_TARGET_COL,
    seq_len: int = 12,
    test_size: float = 0.2,
    random_state: int = 42,
    batch_size: int = 256,
) -> LSTMDataBundle:
    feature_cols = feature_cols or DEFAULT_FEATURE_COLS

    df = pd.read_csv(data_path)
    df.columns = [c.lower() for c in df.columns]

    if "date/time" not in df.columns:
        raise ValueError("Expected 'date/time' column not found.")

    dt = pd.to_datetime(df["date/time"], format="%d %m %Y %H:%M", errors="coerce")
    if dt.isna().all():
        dt = pd.to_datetime(df["date/time"], errors="coerce")

    df["datetime"] = dt
    df = df.sort_values("datetime").reset_index(drop=True)

    df["month"] = df["datetime"].dt.month
    df["hour"] = df["datetime"].dt.hour

    needed_cols = [
        "wind speed (m/s)",
        "wind direction (°)",
        "theoretical_power_curve (kwh)",
        target_col,
        "month",
        "hour",
    ]
    for col in needed_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df = df[needed_cols].dropna().copy()

    mask_bad_zero = (
        (df[target_col] == 0)
        & (df["theoretical_power_curve (kwh)"] != 0)
        & (df["wind speed (m/s)"] > 3)
    )
    df = df.loc[~mask_bad_zero].copy()

    df.loc[df["wind speed (m/s)"] > 19.447, "wind speed (m/s)"] = 19.0
    df = df.reset_index(drop=True)

    X = df[feature_cols].values.astype(np.float32)
    y = df[target_col].values.astype(np.float32).reshape(-1, 1)
    wind_speed_all = df["wind speed (m/s)"].values.astype(np.float32).reshape(-1, 1)
    theoretical_all = df["theoretical_power_curve (kwh)"].values.astype(np.float32).reshape(-1, 1)

    X_seq, y_seq, ws_seq, th_seq = _build_sequences(
        X=X,
        y=y,
        wind_speed_all=wind_speed_all,
        theoretical_all=theoretical_all,
        seq_len=seq_len,
    )

    X_train, X_test, y_train, y_test, ws_train, ws_test, th_train, th_test = train_test_split(
        X_seq,
        y_seq,
        ws_seq,
        th_seq,
        test_size=test_size,
        random_state=random_state,
        shuffle=False,
    )

    x_scaler = StandardScaler()
    n_train, train_seq_len, n_feat = X_train.shape
    n_test = X_test.shape[0]

    X_train_2d = X_train.reshape(-1, n_feat)
    X_test_2d = X_test.reshape(-1, n_feat)

    X_train_s = x_scaler.fit_transform(X_train_2d).reshape(n_train, train_seq_len, n_feat).astype(np.float32)
    X_test_s = x_scaler.transform(X_test_2d).reshape(n_test, train_seq_len, n_feat).astype(np.float32)

    y_scaler = StandardScaler()
    y_train_s = y_scaler.fit_transform(y_train).astype(np.float32)
    y_test_s = y_scaler.transform(y_test).astype(np.float32)

    X_train_t = torch.tensor(X_train_s, dtype=torch.float32)
    X_test_t = torch.tensor(X_test_s, dtype=torch.float32)
    y_train_t = torch.tensor(y_train_s, dtype=torch.float32)
    y_test_t = torch.tensor(y_test_s, dtype=torch.float32)

    train_ds = TensorDataset(X_train_t, y_train_t)
    test_ds = TensorDataset(X_test_t, y_test_t)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return LSTMDataBundle(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        ws_train=ws_train,
        ws_test=ws_test,
        th_train=th_train,
        th_test=th_test,
        X_train_s=X_train_s,
        X_test_s=X_test_s,
        y_train_s=y_train_s,
        y_test_s=y_test_s,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        train_loader=train_loader,
        test_loader=test_loader,
        X_train_t=X_train_t,
        X_test_t=X_test_t,
        y_train_t=y_train_t,
        y_test_t=y_test_t,
    )


def regression_accuracy_percent(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    tol: float = 0.10,
    eps: float = 1e-8,
) -> float:
    rel_err = np.abs(y_true - y_pred) / np.maximum(np.abs(y_true), eps)
    return float((rel_err <= tol).mean() * 100.0)


def evaluate_lstm_model(
    model: nn.Module,
    X_tensor: torch.Tensor,
    y_scaled_true: np.ndarray,
    y_true_original: np.ndarray,
    y_scaler: StandardScaler,
    device: torch.device,
    tol: float = 0.10,
) -> dict[str, Any]:
    model.eval()
    with torch.no_grad():
        y_scaled_pred = model(X_tensor.to(device)).cpu().numpy()

    y_pred_original = y_scaler.inverse_transform(y_scaled_pred)

    mse_scaled = mean_squared_error(y_scaled_true, y_scaled_pred)
    r2 = r2_score(y_true_original, y_pred_original)
    mae = mean_absolute_error(y_true_original, y_pred_original)
    rmse = np.sqrt(mean_squared_error(y_true_original, y_pred_original))
    acc = regression_accuracy_percent(y_true_original, y_pred_original, tol=tol)

    return {
        "scaled_mse": mse_scaled,
        "r2": r2,
        "mae": mae,
        "rmse": rmse,
        "acc": acc,
        "y_pred_original": y_pred_original,
    }


def train_lstm_model(
    model: nn.Module,
    train_loader: DataLoader,
    X_train_t: torch.Tensor,
    X_test_t: torch.Tensor,
    y_train_s: np.ndarray,
    y_test_s: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    y_scaler: StandardScaler,
    device: torch.device,
    num_epochs: int = 10000,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    tol_acc: float = 0.10,
    print_every: int = 10,
) -> tuple[nn.Module, dict[str, list[float]]]:
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history: dict[str, list[float]] = {
        "train_losses": [],
        "test_losses": [],
        "train_accuracies": [],
        "test_accuracies": [],
    }

    for epoch in range(num_epochs):
        model.train()
        train_loss_sum = 0.0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * xb.size(0)

        epoch_train_loss = train_loss_sum / len(train_loader.dataset)
        history["train_losses"].append(epoch_train_loss)

        train_eval = evaluate_lstm_model(
            model=model,
            X_tensor=X_train_t,
            y_scaled_true=y_train_s,
            y_true_original=y_train,
            y_scaler=y_scaler,
            device=device,
            tol=tol_acc,
        )

        test_eval = evaluate_lstm_model(
            model=model,
            X_tensor=X_test_t,
            y_scaled_true=y_test_s,
            y_true_original=y_test,
            y_scaler=y_scaler,
            device=device,
            tol=tol_acc,
        )

        epoch_test_loss = test_eval["scaled_mse"]
        history["test_losses"].append(epoch_test_loss)
        history["train_accuracies"].append(train_eval["acc"])
        history["test_accuracies"].append(test_eval["acc"])

        if epoch == 0 or (epoch + 1) % print_every == 0:
            print(
                f"Epoch [{epoch + 1:05d}/{num_epochs}] | "
                f"Train Loss: {epoch_train_loss:.6f} | "
                f"Test Loss: {epoch_test_loss:.6f} | "
                f"Train Acc(@{int(tol_acc * 100)}%): {train_eval['acc']:.2f}% | "
                f"Test Acc(@{int(tol_acc * 100)}%): {test_eval['acc']:.2f}%"
            )

    return model, history


def export_lstm_onnx_model(
    model: nn.Module,
    onnx_path: str | Path,
    seq_len: int,
    input_dim: int,
    device: torch.device,
    opset_version: int = 13,
) -> None:
    dummy_input = torch.randn(1, seq_len, input_dim, device=device)
    model.eval()

    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input_features"],
        output_names=["predicted_power"],
        dynamic_axes={
            "input_features": {0: "batch_size"},
            "predicted_power": {0: "batch_size"},
        },
    )


def check_onnx_model(onnx_path: str | Path) -> bool:
    try:
        import onnx

        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)
        return True
    except Exception:
        return False


def save_pickle(obj: Any, file_path: str | Path) -> None:
    with open(file_path, "wb") as f:
        pickle.dump(obj, f)

