from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "jd0.xlsx"
OUTPUT_DIR = BASE_DIR / "generated_docs"
OUTPUT_DIR.mkdir(exist_ok=True)

YEAR_SHEETS = [str(y) for y in range(2011, 2024)]
WINDOW_SIZE = 3
FORECAST_STEP = 1

SELECTED_PROVINCES = [
    ("江苏省", "Jiangsu"),
    ("内蒙古自治区", "Inner Mongolia"),
    ("天津市", "Tianjin"),
    ("青海省", "Qinghai"),
    ("新疆维吾尔自治区", "Xinjiang"),
    ("四川省", "Sichuan"),
]

REQUIRED_COLUMNS = ["EE", "DE", "GDP", "IS", "EDU", "URBAN", "DENSITY", "OPEN"]


def load_feature_matrices():
    first_df = pd.read_excel(DATA_PATH, sheet_name=YEAR_SHEETS[0])
    provinces = first_df["P"].tolist()

    feature_matrices = []
    for year in YEAR_SHEETS:
        df = pd.read_excel(DATA_PATH, sheet_name=year).copy()
        df = df.set_index("P").reindex(provinces)
        target = df["EE"].to_numpy(dtype=np.float32)
        feature_cols = [col for col in REQUIRED_COLUMNS if col != "EE" and col in df.columns]
        features = df[feature_cols].to_numpy(dtype=np.float32)
        feature_matrices.append(
            {
                "features": features,
                "target": target,
                "feature_names": feature_cols,
            }
        )
    return provinces, feature_matrices


def build_window_dataset(feature_matrices):
    features = [fm["features"] for fm in feature_matrices]
    targets = [fm["target"] for fm in feature_matrices]
    feature_names = feature_matrices[0]["feature_names"]

    all_features = np.concatenate(features, axis=0)
    feature_scaler = StandardScaler()
    scaled_features_all = feature_scaler.fit_transform(all_features)

    scaled_features = []
    start_idx = 0
    for feat in features:
        end_idx = start_idx + feat.shape[0]
        scaled_features.append(scaled_features_all[start_idx:end_idx])
        start_idx = end_idx

    all_targets = np.concatenate(targets, axis=0).reshape(-1, 1)
    target_scaler = StandardScaler()
    target_scaler.fit(all_targets)

    samples = []
    total_steps = len(feature_matrices)
    for t in range(total_steps - WINDOW_SIZE - FORECAST_STEP + 1):
        x_features = []
        for i in range(t, t + WINDOW_SIZE):
            x_features.append(scaled_features[i])
        y = targets[t + WINDOW_SIZE + FORECAST_STEP - 1]
        y_scaled = target_scaler.transform(y.reshape(-1, 1)).flatten()
        samples.append(
            {
                "x_features": np.stack(x_features, axis=0),
                "y": y_scaled,
            }
        )

    return {
        "samples": samples,
        "feature_names": feature_names,
        "feature_scaler": feature_scaler,
        "target_scaler": target_scaler,
    }


def extract_province_series(dataset, provinces, province_name):
    province_idx = provinces.index(province_name)
    feature_names = dataset["feature_names"]
    de_idx = next(i for i, name in enumerate(feature_names) if name.upper() == "DE")

    de_values = []
    ee_values = []
    for sample in dataset["samples"]:
        de_values.append(sample["x_features"][-1, province_idx, de_idx])
        ee_values.append(sample["y"][province_idx])

    de_values = np.asarray(de_values, dtype=np.float32)
    ee_values = np.asarray(ee_values, dtype=np.float32)

    dummy = np.zeros((len(de_values), len(feature_names)), dtype=np.float32)
    dummy[:, de_idx] = de_values
    de_original = dataset["feature_scaler"].inverse_transform(dummy)[:, de_idx]
    ee_original = dataset["target_scaler"].inverse_transform(ee_values.reshape(-1, 1)).flatten()
    return de_original, ee_original


def fit_quadratic(x: np.ndarray, y: np.ndarray):
    coeffs = np.polyfit(x, y, 2)
    y_hat = np.polyval(coeffs, x)
    return coeffs, float(r2_score(y, y_hat))


def format_equation(a: float, b: float, c: float) -> str:
    return f"y = {a:.3f}x² {b:+.3f}x {c:+.3f}"


def draw_panel_figure(dataset, provinces, output_path: Path) -> None:
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["mathtext.fontset"] = "stix"

    fig, axes = plt.subplots(2, 3, figsize=(14.5, 7.8), dpi=220)
    axes = axes.flatten()

    for ax, (province_cn, province_en) in zip(axes, SELECTED_PROVINCES):
        x, y = extract_province_series(dataset, provinces, province_cn)
        coeffs, r2 = fit_quadratic(x, y)

        x_line = np.linspace(x.min(), x.max(), 300)
        y_line = np.polyval(coeffs, x_line)

        ax.scatter(x, y, alpha=0.72, s=38, edgecolor="white", linewidth=0.5, color="#4f63c1")
        ax.plot(x_line, y_line, color="#b22222", linewidth=1.8)

        ax.set_xlabel("Digital Economy (DE)", fontsize=8.5)
        ax.set_ylabel("Energy Efficiency (EE)", fontsize=8.5)
        ax.set_title(f"{province_en}\nR² = {r2:.3f}", fontsize=10.5, fontweight="bold", pad=8)
        ax.grid(True, alpha=0.22, linestyle="--")
        ax.tick_params(labelsize=7.8)

        a, b, c = coeffs
        eq_text = format_equation(a, b, c)
        stat_text = f"N = {len(x)}\nDE mean: {x.mean():.3f}\nEE mean: {y.mean():.3f}"

        ax.text(
            0.05,
            0.95,
            eq_text,
            transform=ax.transAxes,
            fontsize=7.4,
            va="top",
            bbox=dict(boxstyle="round,pad=0.24", facecolor="#f7e7c6", edgecolor="#caa66a", alpha=0.96),
        )
        ax.text(
            0.05,
            0.06,
            stat_text,
            transform=ax.transAxes,
            fontsize=7.1,
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.26", facecolor="white", edgecolor="#9a9a9a", alpha=0.92),
        )

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    provinces, feature_matrices = load_feature_matrices()
    dataset = build_window_dataset(feature_matrices)
    output_path = OUTPUT_DIR / "nonlinear_panel_english_reproduced.png"
    draw_panel_figure(dataset, provinces, output_path)
    print(output_path)


if __name__ == "__main__":
    main()
