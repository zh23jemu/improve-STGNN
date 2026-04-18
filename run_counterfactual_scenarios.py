import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from repeated_stgnn_stability import build_default_config, train_single_run
from train_stgnn_excel import PROVINCE_ORDER, read_matrix_book, row_normalize


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "counterfactual_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

TARGET_PROVINCES = ["江苏", "广东"]
SHOCKS = [0.25, 0.50, 0.75]
SPLIT_SEED = 2026
MODEL_SEED = 132


def build_equal_weight_config():
    config = build_default_config(MODEL_SEED)
    config.geo_weight = 1 / 3
    config.econ_weight = 1 / 3
    config.digital_weight = 1 / 3
    return config


def load_composite_geo_neighbors() -> Dict[str, np.ndarray]:
    geo = read_matrix_book(BASE_DIR / "1.xlsx")
    econ = read_matrix_book(BASE_DIR / "2.xlsx")
    digital = read_matrix_book(BASE_DIR / "3.xlsx")
    composite = {}
    for year in geo:
        composite[year] = row_normalize((geo[year] + econ[year] + digital[year]) / 3.0)
    return composite


def load_geographic_neighbor_mask() -> np.ndarray:
    # 省份类型标记沿用地理邻近关系，模型输入图则采用 1/3 综合权重。
    geo = read_matrix_book(BASE_DIR / "1.xlsx")
    avg_geo = np.mean([row_normalize(mat) for mat in geo.values()], axis=0)
    mask = avg_geo > 0
    np.fill_diagonal(mask, False)
    return mask


def predict_sample(model, sample, x_mean, x_std, device):
    x_scaled = ((sample["x"] - x_mean) / x_std).astype(np.float32)
    x_tensor = torch.tensor(x_scaled[None, ...], dtype=torch.float32, device=device)
    a_tensor = torch.tensor(sample["a"][None, ...], dtype=torch.float32, device=device)
    with torch.no_grad():
        pred_scaled = model(x_tensor, a_tensor).cpu().numpy()[0, :, 0]
    return pred_scaled


def make_counterfactual_sample(sample, target_idx: int, shock_sd: float, feature_names: List[str], x_std: np.ndarray):
    cf = {
        "input_years": list(sample["input_years"]),
        "target_year": sample["target_year"],
        "x": sample["x"].copy(),
        "a": sample["a"].copy(),
        "y": sample["y"].copy(),
    }
    de_idx = feature_names.index("DE")
    de_shock_raw = float(shock_sd * x_std[0, de_idx])

    # 与 1.18.py 口径一致：对输入窗口内目标省份 DE 同步提高若干标准差。
    cf["x"][:, target_idx, de_idx] += de_shock_raw

    # 若模型输入包含 W_DE，则同步更新综合空间矩阵下的空间滞后 DE。
    if "W_DE" in feature_names:
        w_de_idx = feature_names.index("W_DE")
        for t in range(cf["x"].shape[0]):
            de_vector = cf["x"][t, :, de_idx].reshape(-1, 1)
            cf["x"][t, :, w_de_idx] = (cf["a"][t] @ de_vector)[:, 0]

    return cf


def run_counterfactual_for_target(run_result: Dict, target_province: str, shock_sd: float, neighbor_mask: np.ndarray) -> Dict:
    model = run_result["model"]
    model.eval()

    config = run_result["config"]
    raw_samples = run_result["train_raw"] + run_result["valid_raw"] + run_result["test_raw"]
    x_mean = run_result["x_mean"]
    x_std = run_result["x_std"]
    y_mean = float(run_result["y_mean"][0, 0])
    y_std = float(run_result["y_std"][0, 0])
    feature_names = run_result["feature_names"]
    target_idx = PROVINCE_ORDER.index(target_province)

    original_preds = []
    cf_preds = []
    for sample in raw_samples:
        original_scaled = predict_sample(model, sample, x_mean, x_std, config.device)
        cf_sample = make_counterfactual_sample(sample, target_idx, shock_sd, feature_names, x_std)
        cf_scaled = predict_sample(model, cf_sample, x_mean, x_std, config.device)
        original_preds.append(original_scaled * y_std + y_mean)
        cf_preds.append(cf_scaled * y_std + y_mean)

    original_preds = np.asarray(original_preds)
    cf_preds = np.asarray(cf_preds)
    changes = cf_preds - original_preds
    mean_changes = changes.mean(axis=0)
    std_changes = changes.std(axis=0)
    original_mean = original_preds.mean(axis=0)
    rel_change = np.where(np.abs(original_mean) > 1e-12, mean_changes / original_mean * 100.0, 0.0)

    neighbor_indices = np.where(neighbor_mask[target_idx])[0]
    rows = []
    for idx, province in enumerate(PROVINCE_ORDER):
        if idx == target_idx:
            province_type = "Target"
        elif idx in set(neighbor_indices.tolist()):
            province_type = "Neighbor"
        else:
            province_type = "Other"
        rows.append(
            {
                "Province": province,
                "Province_Type": province_type,
                "EE_Change": mean_changes[idx],
                "EE_Change_Std": std_changes[idx],
                "Relative_Change_Percent": rel_change[idx],
            }
        )

    detail_df = pd.DataFrame(rows)
    detail_df["Abs_Change"] = detail_df["EE_Change"].abs()
    detail_df = detail_df.sort_values("Abs_Change", ascending=False).drop(columns=["Abs_Change"]).reset_index(drop=True)
    detail_df["Rank"] = np.arange(1, len(detail_df) + 1)

    target_change = float(mean_changes[target_idx])
    neighbor_changes = mean_changes[neighbor_indices] if len(neighbor_indices) else np.asarray([])
    summary = {
        "target_province": target_province,
        "shock_sd": shock_sd,
        "target_change": target_change,
        "target_relative_change_percent": float(rel_change[target_idx]),
        "neighbor_count": int(len(neighbor_indices)),
        "avg_neighbor_change": float(neighbor_changes.mean()) if len(neighbor_changes) else 0.0,
        "max_neighbor_change": float(neighbor_changes.max()) if len(neighbor_changes) else 0.0,
        "min_neighbor_change": float(neighbor_changes.min()) if len(neighbor_changes) else 0.0,
        "spillover_ratio": float(neighbor_changes.mean() / target_change) if len(neighbor_changes) and abs(target_change) > 1e-12 else 0.0,
    }
    return {"summary": summary, "detail_df": detail_df}


def save_bar_chart(detail_df: pd.DataFrame, target_province: str, shock_sd: float, out_path: Path) -> None:
    top = detail_df.head(10).copy()
    colors = top["Province_Type"].map({"Target": "#d95f02", "Neighbor": "#1b9e77", "Other": "#7570b3"}).fillna("#7570b3")

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=220)
    ax.bar(top["Province"], top["EE_Change"], color=colors)
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_title(f"{target_province}: Counterfactual EE Changes (+{shock_sd:.2f} SD DE)")
    ax.set_ylabel("Mean EE Change")
    ax.set_xlabel("Province")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    print("Training STGNN with 1/3 composite matrix from 1.xlsx, 2.xlsx and 3.xlsx...")
    config = build_equal_weight_config()
    run_result = train_single_run(config=config, split_seed=SPLIT_SEED)
    neighbor_mask = load_geographic_neighbor_mask()

    all_summaries = []
    for target in TARGET_PROVINCES:
        for shock in SHOCKS:
            print(f"Running counterfactual: target={target}, shock=+{shock:.2f} SD")
            result = run_counterfactual_for_target(run_result, target, shock, neighbor_mask)
            summary = result["summary"]
            all_summaries.append(summary)

            tag = f"{target}_plus_{str(shock).replace('.', '')}sd"
            detail_path = OUTPUT_DIR / f"counterfactual_{tag}_details.csv"
            chart_path = OUTPUT_DIR / f"counterfactual_{tag}_top10.png"
            result["detail_df"].to_csv(detail_path, index=False, encoding="utf-8-sig")
            save_bar_chart(result["detail_df"], target, shock, chart_path)

    summary_df = pd.DataFrame(all_summaries)
    summary_path = OUTPUT_DIR / "counterfactual_scenario_summary.csv"
    summary_json = OUTPUT_DIR / "counterfactual_scenario_summary.json"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    summary_json.write_text(json.dumps(all_summaries, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nCounterfactual summary:")
    print(summary_df.to_string(index=False))
    print(f"\nSaved: {summary_path}")
    print(f"Saved figures/details in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
