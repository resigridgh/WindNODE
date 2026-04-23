from __future__ import annotations

import argparse
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from WindNODEpy import (
    DEFAULT_FEATURE_COLS,
    WindCNNRegressor,
    check_onnx_model,
    evaluate_cnn_model,
    export_cnn_onnx_model,
    get_device,
    load_and_preprocess_cnn_data,
    save_pickle,
    set_seed,
    train_cnn_model,
)

warnings.filterwarnings("ignore")


def matlab_like_axes(ax, xlabel_text=r"Time", ylabel_text=None, title_text=None):
    ax.tick_params(axis="both", colors="black", direction="out")
    for spine in ax.spines.values():
        spine.set_color("black")
    ax.set_xlabel(xlabel_text)
    if ylabel_text is not None:
        ax.set_ylabel(ylabel_text)
    if title_text is not None:
        ax.set_title(title_text)


def make_output_dir(output_dir: str | Path) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    return out


def parse_args():
    parser = argparse.ArgumentParser(description="Train CNN for wind turbine power prediction.")
    parser.add_argument("--data-path", type=str, default="data/T1.csv", help="Path to T1.csv")
    parser.add_argument("--output-dir", type=str, default="outputs/cnn", help="Directory to save outputs")
    parser.add_argument("--epochs", type=int, default=10000, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Adam weight decay")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--tol-acc", type=float, default=0.10, help="Relative error tolerance for regression accuracy")
    parser.add_argument("--print-every", type=int, default=10, help="Print frequency in epochs")
    parser.add_argument("--use-latex", action="store_true", help="Enable LaTeX text rendering in matplotlib")
    return parser.parse_args()


def main():
    args = parse_args()

    set_seed(args.seed)
    device = get_device()
    print("Using device:", device)

    ts = datetime.now().strftime("%d-%m-%Y_%I-%M%p")
    output_dir = make_output_dir(args.output_dir)

    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.edgecolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
            "text.usetex": args.use_latex,
        }
    )

    data = load_and_preprocess_cnn_data(
        data_path=args.data_path,
        feature_cols=DEFAULT_FEATURE_COLS,
        test_size=0.2,
        random_state=args.seed,
        batch_size=args.batch_size,
    )

    model = WindCNNRegressor().to(device)
    print(model)

    model, history = train_cnn_model(
        model=model,
        train_loader=data.train_loader,
        X_train_t=data.X_train_t,
        X_test_t=data.X_test_t,
        y_train_s=data.y_train_s,
        y_test_s=data.y_test_s,
        y_train=data.y_train,
        y_test=data.y_test,
        y_scaler=data.y_scaler,
        device=device,
        num_epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        tol_acc=args.tol_acc,
        print_every=args.print_every,
    )

    final_train_eval = evaluate_cnn_model(
        model=model,
        X_tensor=data.X_train_t,
        y_scaled_true=data.y_train_s,
        y_true_original=data.y_train,
        y_scaler=data.y_scaler,
        device=device,
        tol=args.tol_acc,
    )

    final_test_eval = evaluate_cnn_model(
        model=model,
        X_tensor=data.X_test_t,
        y_scaled_true=data.y_test_s,
        y_true_original=data.y_test,
        y_scaler=data.y_scaler,
        device=device,
        tol=args.tol_acc,
    )

    y_pred_test = final_test_eval["y_pred_original"]
    y_true_test = data.y_test

    print("\nFinal Metrics")
    print(f"Training Accuracy (@{int(args.tol_acc * 100)}% relative error): {final_train_eval['acc']:.2f}%")
    print(f"Test Accuracy     (@{int(args.tol_acc * 100)}% relative error): {final_test_eval['acc']:.2f}%")
    print(f"Train R2   : {final_train_eval['r2']:.6f}")
    print(f"Test  R2   : {final_test_eval['r2']:.6f}")
    print(f"Train MAE  : {final_train_eval['mae']:.6f}")
    print(f"Test  MAE  : {final_test_eval['mae']:.6f}")
    print(f"Train RMSE : {final_train_eval['rmse']:.6f}")
    print(f"Test  RMSE : {final_test_eval['rmse']:.6f}")

    loss_pdf = output_dir / f"cnn_loss_curve_{ts}.pdf"
    fig, ax = plt.subplots(figsize=(5, 2.5))
    ax.plot(np.arange(1, args.epochs + 1), history["train_losses"], label="Training Loss")
    ax.plot(np.arange(1, args.epochs + 1), history["test_losses"], label="Testing Loss")
    matlab_like_axes(ax, xlabel_text=r"Epochs", ylabel_text=r"Loss", title_text="CNN Training and Testing Loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(loss_pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)

    acc_pdf = output_dir / f"cnn_accuracy_curve_{ts}.pdf"
    fig, ax = plt.subplots(figsize=(5, 2.5))
    ax.plot(np.arange(1, args.epochs + 1), history["train_accuracies"], label="Training Accuracy")
    ax.plot(np.arange(1, args.epochs + 1), history["test_accuracies"], label="Testing Accuracy")
    matlab_like_axes(
        ax,
        xlabel_text=r"Epochs",
        ylabel_text=rf"Accuracy (%)",
        title_text="CNN Training and Testing Accuracy",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(acc_pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)

    avp_pdf = output_dir / f"cnn_actual_vs_predicted_{ts}.pdf"
    fig, ax = plt.subplots(figsize=(5, 2.5))
    ax.scatter(y_true_test, y_pred_test, alpha=0.5, label="Predicted vs Actual")
    mn = min(y_true_test.min(), y_pred_test.min())
    mx = max(y_true_test.max(), y_pred_test.max())
    ax.plot([mn, mx], [mn, mx], color="red", linewidth=1.3, label="Ideal Line")
    matlab_like_axes(
        ax,
        xlabel_text=r"Actual Power (kW)",
        ylabel_text=r"Predicted Power (kW)",
        title_text="CNN: Actual vs Predicted",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(avp_pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)

    plot_df = pd.DataFrame(
        {
            "wind speed (m/s)": data.ws_test.flatten(),
            "theoretical_power_curve (kwh)": data.th_test.flatten(),
            "actual_power": y_true_test.flatten(),
            "predicted_power": y_pred_test.flatten(),
        }
    ).sort_values("wind speed (m/s)").reset_index(drop=True)

    comp_pdf = output_dir / f"cnn_power_prediction_comparison_{ts}.pdf"
    fig, ax = plt.subplots(figsize=(5, 2.5))
    ax.scatter(plot_df["wind speed (m/s)"], plot_df["actual_power"], alpha=0.35, label="Real Power")
    ax.scatter(plot_df["wind speed (m/s)"], plot_df["predicted_power"], alpha=0.35, label="Predicted Power")
    ax.plot(plot_df["wind speed (m/s)"], plot_df["theoretical_power_curve (kwh)"], label="Theoretical Power")
    matlab_like_axes(
        ax,
        xlabel_text=r"Wind Speed (m/s)",
        ylabel_text=r"Power Production (kW)",
        title_text="CNN Wind Power Prediction",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(comp_pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)

    pth_file = output_dir / f"cnn_wind_power_{ts}.pth"
    torch.save(model.state_dict(), pth_file)
    print(f"\nSaved PyTorch weights: {pth_file}")

    onnx_file = output_dir / f"cnn_wind_power_{ts}.onnx"
    export_cnn_onnx_model(
        model=model,
        onnx_path=onnx_file,
        input_length=len(DEFAULT_FEATURE_COLS),
        device=device,
    )
    print(f"Saved ONNX model: {onnx_file}")

    if check_onnx_model(onnx_file):
        print("ONNX model check: PASSED")
    else:
        print("ONNX model check skipped or failed.")

    x_scaler_file = output_dir / f"cnn_x_scaler_{ts}.pkl"
    y_scaler_file = output_dir / f"cnn_y_scaler_{ts}.pkl"
    save_pickle(data.x_scaler, x_scaler_file)
    save_pickle(data.y_scaler, y_scaler_file)

    print(f"Saved input scaler: {x_scaler_file}")
    print(f"Saved target scaler: {y_scaler_file}")

    print("\nSaved files:")
    print(f" - {pth_file}")
    print(f" - {onnx_file}")
    print(f" - {x_scaler_file}")
    print(f" - {y_scaler_file}")
    print(f" - {loss_pdf}")
    print(f" - {acc_pdf}")
    print(f" - {avp_pdf}")
    print(f" - {comp_pdf}")


if __name__ == "__main__":
    main()

