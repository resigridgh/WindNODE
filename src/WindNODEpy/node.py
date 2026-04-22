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
class DataBundle:
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


class ODEFunc(nn.Module):
    def __init__(self, hidden_dim: int = 64, augment_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, augment_dim),
            nn.ReLU(),
            nn.Linear(augment_dim, hidden_dim),
        )

    def forward(self, t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


class RK4Block(nn.Module):
    def __init__(self, odefunc: nn.Module, step_size: float = 0.25, num_steps: int = 4) -> None:
        super().__init__()
        self.odefunc = odefunc
        self.step_size = step_size
        self.num_steps = num_steps

    def forward(self, h0: torch.Tensor) -> torch.Tensor:
        h = h0
        t = torch.zeros(1, device=h0.device, dtype=h0.dtype)

        for _ in range(self.num_steps):
            dt = self.step_size

            k1 = self.odefunc(t, h)
            k2 = self.odefunc(t + dt / 2.0, h + dt * k1 / 2.0)
            k3 = self.odefunc(t + dt / 2.0, h + dt * k2 / 2.0)
            k4 = self.odefunc(t + dt, h + dt * k3)

            h = h + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            t = t + dt

        return h


class NeuralODERegressor(nn.Module):
    def __init__(
        self,
        input_dim: int = 4,
        hidden_dim: int = 64,
        augment_dim: int = 128,
        num_steps: int = 4,
        step_size: float = 0.25,
    ) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )

        self.odefunc = ODEFunc(hidden_dim=hidden_dim, augment_dim=augment_dim)
        self.rk4 = RK4Block(self.odefunc, step_size=step_size, num_steps=num_steps)

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h0 = self.encoder(x)
        hT = self.rk4(h0)
        yhat = self.decoder(hT)
        return yhat


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_and_preprocess_data(
    data_path: str | Path,
    feature_cols: list[str] | None = None,
    target_col: str = DEFAULT_TARGET_COL,
    test_size: float = 0.2,
    random_state: int = 42,
    batch_size: int = 256,
) -> DataBundle:
    feature_cols = feature_cols or DEFAULT_FEATURE_COLS

    df = pd.read_csv(data_path)
    df.columns = [c.lower() for c in df.columns]

    if "date/time" not in df.columns:
        raise ValueError("Expected 'date/time' column not found.")

    dt = pd.to_datetime(df["date/time"], format="%d %m %Y %H:%M", errors="coerce")
    if dt.isna().all():
        dt = pd.to_datetime(df["date/time"], errors="coerce")

    df["month"] = dt.dt.month
    df["hour"] = dt.dt.hour

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

    X = df[feature_cols].values.astype(np.float32)
    y = df[target_col].values.astype(np.float32).reshape(-1, 1)
    wind_speed_all = df["wind speed (m/s)"].values.astype(np.float32).reshape(-1, 1)
    theoretical_all = df["theoretical_power_curve (kwh)"].values.astype(np.float32).reshape(-1, 1)

    X_train, X_test, y_train, y_test, ws_train, ws_test, th_train, th_test = train_test_split(
        X,
        y,
        wind_speed_all,
        theoretical_all,
        test_size=test_size,
        random_state=random_state,
    )

    x_scaler = StandardScaler()
    X_train_s = x_scaler.fit_transform(X_train).astype(np.float32)
    X_test_s = x_scaler.transform(X_test).astype(np.float32)

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

    return DataBundle(
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


def evaluate_model(
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


def train_model(
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

        train_eval = evaluate_model(
            model=model,
            X_tensor=X_train_t,
            y_scaled_true=y_train_s,
            y_true_original=y_train,
            y_scaler=y_scaler,
            device=device,
            tol=tol_acc,
        )

        test_eval = evaluate_model(
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


def export_onnx_model(
    model: nn.Module,
    onnx_path: str | Path,
    input_dim: int,
    device: torch.device,
    opset_version: int = 13,
) -> None:
    dummy_input = torch.randn(1, input_dim, device=device)
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
