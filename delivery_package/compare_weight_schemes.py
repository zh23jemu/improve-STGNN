import json
from pathlib import Path

import pandas as pd

from repeated_stgnn_stability import build_default_config, train_single_run


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "stability_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


SCHEMES = [
    ("equal", "等权基准", 0.33, 0.33, 0.34),
    ("geo", "地理优先", 0.50, 0.25, 0.25),
    ("econ", "经济优先", 0.25, 0.50, 0.25),
    ("digital", "数字优先", 0.25, 0.25, 0.50),
]

# 采用固定随机窗口划分，仅改变模型随机种子。
SEEDS = [22, 52, 72, 102, 132]
SPLIT_SEED = 2026


def df_to_markdown(df: pd.DataFrame) -> str:
    headers = [str(col) for col in df.columns]
    rows = [[str(v) for v in row] for row in df.to_numpy()]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    run_rows = []

    for scheme_code, scheme_name, geo_w, econ_w, digital_w in SCHEMES:
        print(f"\n{'=' * 88}\n方案: {scheme_name} ({geo_w:.2f}, {econ_w:.2f}, {digital_w:.2f})")
        for seed in SEEDS:
            config = build_default_config(seed)
            config.geo_weight = geo_w
            config.econ_weight = econ_w
            config.digital_weight = digital_w

            result = train_single_run(config=config, split_seed=SPLIT_SEED)
            row = {
                "scheme_code": scheme_code,
                "方案": scheme_name,
                "geo_weight": geo_w,
                "econ_weight": econ_w,
                "digital_weight": digital_w,
                "seed": seed,
                "best_epoch": result["best_epoch"],
                "训练集 R²": round(result["train_metrics"]["r2"], 4),
                "训练集 RMSE": round(result["train_metrics"]["rmse"], 4),
                "训练集 MAE": round(result["train_metrics"]["mae"], 4),
                "验证集 R²": round(result["valid_metrics"]["r2"], 4),
                "验证集 RMSE": round(result["valid_metrics"]["rmse"], 4),
                "验证集 MAE": round(result["valid_metrics"]["mae"], 4),
                "测试集 R²": round(result["test_metrics"]["r2"], 4),
                "测试集 RMSE": round(result["test_metrics"]["rmse"], 4),
                "测试集 MAE": round(result["test_metrics"]["mae"], 4),
            }
            run_rows.append(row)
            print(
                f"seed={seed} | "
                f"train_r2={row['训练集 R²']:.4f} | "
                f"valid_r2={row['验证集 R²']:.4f} | "
                f"test_r2={row['测试集 R²']:.4f}"
            )

    runs_df = pd.DataFrame(run_rows)

    best_rows = []
    mean_rows = []
    for scheme_code, group in runs_df.groupby("scheme_code", sort=False):
        group = group.copy()
        best = group.sort_values(
            by=["验证集 R²", "测试集 R²", "训练集 R²"],
            ascending=False,
        ).iloc[0]
        best_rows.append(
            {
                "方案": best["方案"],
                "图权重(G,E,D)": f"{best['geo_weight']:.2f}, {best['econ_weight']:.2f}, {best['digital_weight']:.2f}",
                "选中 seed": int(best["seed"]),
                "训练集 R²": best["训练集 R²"],
                "训练集 RMSE": best["训练集 RMSE"],
                "训练集 MAE": best["训练集 MAE"],
                "验证集 R²": best["验证集 R²"],
                "验证集 RMSE": best["验证集 RMSE"],
                "验证集 MAE": best["验证集 MAE"],
                "测试集 R²": best["测试集 R²"],
                "测试集 RMSE": best["测试集 RMSE"],
                "测试集 MAE": best["测试集 MAE"],
            }
        )

        mean_rows.append(
            {
                "方案": group["方案"].iloc[0],
                "图权重(G,E,D)": f"{group['geo_weight'].iloc[0]:.2f}, {group['econ_weight'].iloc[0]:.2f}, {group['digital_weight'].iloc[0]:.2f}",
                "重复次数": len(group),
                "训练集 R² 均值": round(group["训练集 R²"].mean(), 4),
                "训练集 RMSE 均值": round(group["训练集 RMSE"].mean(), 4),
                "训练集 MAE 均值": round(group["训练集 MAE"].mean(), 4),
                "验证集 R² 均值": round(group["验证集 R²"].mean(), 4),
                "验证集 RMSE 均值": round(group["验证集 RMSE"].mean(), 4),
                "验证集 MAE 均值": round(group["验证集 MAE"].mean(), 4),
                "测试集 R² 均值": round(group["测试集 R²"].mean(), 4),
                "测试集 RMSE 均值": round(group["测试集 RMSE"].mean(), 4),
                "测试集 MAE 均值": round(group["测试集 MAE"].mean(), 4),
            }
        )

    best_df = pd.DataFrame(best_rows)
    mean_df = pd.DataFrame(mean_rows)

    run_csv = OUTPUT_DIR / "weight_scheme_runs.csv"
    best_csv = OUTPUT_DIR / "weight_scheme_best_summary.csv"
    mean_csv = OUTPUT_DIR / "weight_scheme_mean_summary.csv"
    best_md = OUTPUT_DIR / "weight_scheme_best_summary.md"
    mean_md = OUTPUT_DIR / "weight_scheme_mean_summary.md"
    result_json = OUTPUT_DIR / "weight_scheme_summary.json"

    runs_df.to_csv(run_csv, index=False, encoding="utf-8-sig")
    best_df.to_csv(best_csv, index=False, encoding="utf-8-sig")
    mean_df.to_csv(mean_csv, index=False, encoding="utf-8-sig")
    best_md.write_text(df_to_markdown(best_df), encoding="utf-8")
    mean_md.write_text(df_to_markdown(mean_df), encoding="utf-8")
    result_json.write_text(
        json.dumps(
            {
                "schemes": SCHEMES,
                "seeds": SEEDS,
                "split_seed": SPLIT_SEED,
                "best_summary": best_df.to_dict(orient="records"),
                "mean_summary": mean_df.to_dict(orient="records"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n最优结果表")
    print(best_df.to_string(index=False))
    print("\n平均结果表")
    print(mean_df.to_string(index=False))
    print(f"\nSaved: {run_csv}")
    print(f"Saved: {best_csv}")
    print(f"Saved: {mean_csv}")


if __name__ == "__main__":
    main()
