# ============================================================
# Wind Turbine SCADA Data Pattern Visualization
# ============================================================

from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Find repo root automatically
# parents[2] = WindNODE/
# ------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = REPO_ROOT / "data" / "T1.csv"
OUTPUT_DIR = REPO_ROOT / "outputs" / "data_viz"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TS = datetime.now().strftime("%d-%m-%Y_%I-%M%p")

print(f"Repo root : {REPO_ROOT}")
print(f"Data path : {DATA_PATH}")
print(f"Output dir: {OUTPUT_DIR}")

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

# ------------------------------------------------------------
# Load dataset
# ------------------------------------------------------------
df = pd.read_csv(DATA_PATH)
df.columns = [c.lower() for c in df.columns]

print("Columns:", df.columns.tolist())
print("Original shape:", df.shape)

# ------------------------------------------------------------
# Parse datetime and create month/hour
# ------------------------------------------------------------
if "date/time" not in df.columns:
    raise ValueError("Expected 'date/time' column not found.")

dt = pd.to_datetime(df["date/time"], format="%d %m %Y %H:%M", errors="coerce")
if dt.isna().all():
    dt = pd.to_datetime(df["date/time"], errors="coerce")

df["datetime"] = dt
df["month"] = df["datetime"].dt.month
df["hour"] = df["datetime"].dt.hour

# ------------------------------------------------------------
# Keep required columns
# ------------------------------------------------------------
cols = [
    "datetime",
    "month",
    "hour",
    "wind speed (m/s)",
    "wind direction (°)",
    "theoretical_power_curve (kwh)",
    "lv activepower (kw)",
]

for c in cols:
    if c not in df.columns:
        raise ValueError(f"Missing column: {c}")

df = df[cols].dropna().copy()

# ------------------------------------------------------------
# Basic cleaning
# ------------------------------------------------------------
mask_bad_zero = (
    (df["lv activepower (kw)"] == 0)
    & (df["theoretical_power_curve (kwh)"] != 0)
    & (df["wind speed (m/s)"] > 3)
)

df = df.loc[~mask_bad_zero].copy()
df.loc[df["wind speed (m/s)"] > 19.447, "wind speed (m/s)"] = 19.0

print("Cleaned shape:", df.shape)
print(df.describe())

# ------------------------------------------------------------
# MATLAB-like plot style
# ------------------------------------------------------------
plt.rcParams.update({
    "font.size": 12,
    "axes.edgecolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "text.usetex": False,
})

def style_axes(ax, xlabel_text=None, ylabel_text=None, title_text=None):
    ax.tick_params(axis="both", colors="black", direction="out")
    for spine in ax.spines.values():
        spine.set_color("black")
    if xlabel_text:
        ax.set_xlabel(xlabel_text)
    if ylabel_text:
        ax.set_ylabel(ylabel_text)
    if title_text:
        ax.set_title(title_text)

# ------------------------------------------------------------
# Pattern 1: Actual power vs wind speed
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5, 2.5))

ax.scatter(
    df["wind speed (m/s)"],
    df["lv activepower (kw)"],
    alpha=0.35,
    s=8,
    label="Measured Power"
)

df_sorted = df.sort_values("wind speed (m/s)")
ax.plot(
    df_sorted["wind speed (m/s)"],
    df_sorted["theoretical_power_curve (kwh)"],
    linewidth=2,
    label="Theoretical Power Curve"
)

style_axes(
    ax,
    xlabel_text=r"Wind Speed (m/s)",
    ylabel_text=r"Power (kW)",
    title_text="Wind Speed--Power Pattern"
)

ax.legend()
fig.tight_layout()
file1 = OUTPUT_DIR / f"data_pattern_wind_speed_power_{TS}.pdf"
fig.savefig(file1, format="pdf", bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------------------
# Pattern 2: Average power by month
# ------------------------------------------------------------
monthly_power = df.groupby("month")["lv activepower (kw)"].mean()

fig, ax = plt.subplots(figsize=(5, 2.5))
ax.bar(monthly_power.index, monthly_power.values)

style_axes(
    ax,
    xlabel_text=r"Month",
    ylabel_text=r"Average Power (kW)",
    title_text="Monthly Average Power Pattern"
)

fig.tight_layout()
file2 = OUTPUT_DIR / f"data_pattern_monthly_power_{TS}.pdf"
fig.savefig(file2, format="pdf", bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------------------
# Pattern 3: Average power by hour
# ------------------------------------------------------------
hourly_power = df.groupby("hour")["lv activepower (kw)"].mean()

fig, ax = plt.subplots(figsize=(5, 2.5))
ax.bar(hourly_power.index, hourly_power.values)

style_axes(
    ax,
    xlabel_text=r"Hour",
    ylabel_text=r"Average Power (kW)",
    title_text="Hourly Average Power Pattern"
)

fig.tight_layout()
file3 = OUTPUT_DIR / f"data_pattern_hourly_power_{TS}.pdf"
fig.savefig(file3, format="pdf", bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------------------
# Pattern 4: Wind speed distribution
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5, 2.5))
ax.hist(df["wind speed (m/s)"], bins=30)

style_axes(
    ax,
    xlabel_text=r"Wind Speed (m/s)",
    ylabel_text=r"Count",
    title_text="Wind Speed Distribution"
)

fig.tight_layout()
file4 = OUTPUT_DIR / f"data_pattern_wind_speed_distribution_{TS}.pdf"
fig.savefig(file4, format="pdf", bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------------------
# Pattern 5: Power time pattern
# ------------------------------------------------------------
df_time = df.sort_values("datetime").reset_index(drop=True)
n_plot = min(1000, len(df_time))

fig, ax = plt.subplots(figsize=(5, 2.5))
ax.plot(df_time["lv activepower (kw)"].iloc[:n_plot], label="Measured Power")

style_axes(
    ax,
    xlabel_text=r"Time Index",
    ylabel_text=r"Power (kW)",
    title_text="Measured Power Time Pattern"
)

ax.legend()
fig.tight_layout()
file5 = OUTPUT_DIR / f"data_pattern_power_time_{TS}.pdf"
fig.savefig(file5, format="pdf", bbox_inches="tight")
plt.close(fig)

print("\nSaved PDF files:")
for f in [file1, file2, file3, file4, file5]:
    print(f" - {f}")


