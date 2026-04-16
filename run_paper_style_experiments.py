import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from train_stgnn_excel import (
    Config,
    FEATURE_COLUMNS,
    build_windows,
    compute_metrics,
    df_to_markdown,
    fit_standardizer,
    inverse_scale,
    load_all_data,
    split_samples,
    transform_samples,
)


DATA_DIR = Path(__file__).resolve().parent


def flatten_samples(samples: List[Dict], y_mean=None, y_std=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = []
    targets = []
    target_years = []
    for sample in samples:
        x_seq = sample["x"]
        graph_strength = sample["a"].sum(axis=2)
        for node_idx in range(x_seq.shape[1]):
            node_hist = x_seq[:, node_idx, :].reshape(-1)
            node_degree = graph_strength[:, node_idx].reshape(-1)
            rows.append(np.concatenate([node_hist, node_degree], axis=0))
            targets.append(sample["y"][node_idx, 0])
            target_years.append(sample["target_year"])

    x = np.asarray(rows, dtype=np.float32)
    y = np.asarray(targets, dtype=np.float32).reshape(-1, 1)
    if y_mean is not None and y_std is not None:
        y = inverse_scale(y, y_mean, y_std)
    return x, y.ravel(), np.asarray(target_years)


def make_models() -> Dict[str, object]:
    return {
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=0.5)),
        "RandomForest": RandomForestRegressor(
            n_estimators=500,
            max_depth=5,
            min_samples_leaf=2,
            random_state=42,
        ),
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=600,
            max_depth=7,
            min_samples_leaf=2,
            random_state=42,
        ),
        "HistGBDT": HistGradientBoostingRegressor(
            learning_rate=0.03,
            max_depth=3,
            max_iter=400,
            min_samples_leaf=6,
            l2_regularization=0.01,
            random_state=42,
        ),
    }


def metrics(y_true, y_pred):
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def prepare_samples(config: Config):
    config.input_dim = len(FEATURE_COLUMNS) + int(config.use_target_history) + (
        2 if config.use_spatial_lag_features else 0
    )
    features_by_year, targets_by_year, graphs_by_year, feature_names = load_all_data(config)
    samples = build_windows(
        features_by_year,
        targets_by_year,
        graphs_by_year,
        config.seq_len,
        config.horizon,
    )
    return samples, feature_names


def run_random_node_year_holdout(samples: List[Dict], test_size=0.2, valid_size=0.2):
    train_samples, _, _ = split_samples(samples)
    x_mean, x_std, y_mean, y_std = fit_standardizer(train_samples)
    scaled_samples = transform_samples(samples, x_mean, x_std, y_mean, y_std)
    x, y, years = flatten_samples(scaled_samples, y_mean, y_std)

    x_train_valid, x_test, y_train_valid, y_test, years_train_valid, years_test = train_test_split(
        x, y, years, test_size=test_size, random_state=42
    )
    relative_valid_size = valid_size / (1.0 - test_size)
    x_train, x_valid, y_train, y_valid, years_train, years_valid = train_test_split(
        x_train_valid,
        y_train_valid,
        years_train_valid,
        test_size=relative_valid_size,
        random_state=42,
    )

    rows = []
    for name, model in make_models().items():
        model.fit(x_train, y_train)
        rows.append(make_result_row(name, y_train, model.predict(x_train), y_valid, model.predict(x_valid), y_test, model.predict(x_test)))
    return pd.DataFrame(rows).sort_values(["验证集 R²", "测试集 R²"], ascending=False)


def run_rolling_forecast(samples: List[Dict]):
    rows = []
    target_years = sorted({s["target_year"] for s in samples})
    for test_year in target_years:
        if test_year < 2018:
            continue
        train_raw = [s for s in samples if s["target_year"] <= test_year - 2]
        valid_raw = [s for s in samples if s["target_year"] == test_year - 1]
        test_raw = [s for s in samples if s["target_year"] == test_year]
        if len(train_raw) < 3 or not valid_raw or not test_raw:
            continue

        x_mean, x_std, y_mean, y_std = fit_standardizer(train_raw)
        train = transform_samples(train_raw, x_mean, x_std, y_mean, y_std)
        valid = transform_samples(valid_raw, x_mean, x_std, y_mean, y_std)
        test = transform_samples(test_raw, x_mean, x_std, y_mean, y_std)

        x_train, y_train, _ = flatten_samples(train, y_mean, y_std)
        x_valid, y_valid, _ = flatten_samples(valid, y_mean, y_std)
        x_test, y_test, _ = flatten_samples(test, y_mean, y_std)

        for name, model in make_models().items():
            model.fit(x_train, y_train)
            row = make_result_row(
                name,
                y_train,
                model.predict(x_train),
                y_valid,
                model.predict(x_valid),
                y_test,
                model.predict(x_test),
            )
            row["预测年份"] = test_year
            rows.append(row)

    df = pd.DataFrame(rows)
    avg_rows = []
    for name, group in df.groupby("模型"):
        avg_rows.append(
            {
                "模型": name,
                "训练集 R²": round(group["训练集 R²"].mean(), 4),
                "训练集 RMSE": round(group["训练集 RMSE"].mean(), 4),
                "训练集 MAE": round(group["训练集 MAE"].mean(), 4),
                "验证集 R²": round(group["验证集 R²"].mean(), 4),
                "验证集 RMSE": round(group["验证集 RMSE"].mean(), 4),
                "验证集 MAE": round(group["验证集 MAE"].mean(), 4),
                "测试集 R²": round(group["测试集 R²"].mean(), 4),
                "测试集 RMSE": round(group["测试集 RMSE"].mean(), 4),
                "测试集 MAE": round(group["测试集 MAE"].mean(), 4),
            }
        )
    avg_df = pd.DataFrame(avg_rows).sort_values(["测试集 R²", "验证集 R²"], ascending=False)
    return df, avg_df


def make_result_row(name, y_train, p_train, y_valid, p_valid, y_test, p_test):
    train_metrics = metrics(y_train, p_train)
    valid_metrics = metrics(y_valid, p_valid)
    test_metrics = metrics(y_test, p_test)
    return {
        "模型": name,
        "训练集 R²": round(train_metrics["r2"], 4),
        "训练集 RMSE": round(train_metrics["rmse"], 4),
        "训练集 MAE": round(train_metrics["mae"], 4),
        "验证集 R²": round(valid_metrics["r2"], 4),
        "验证集 RMSE": round(valid_metrics["rmse"], 4),
        "验证集 MAE": round(valid_metrics["mae"], 4),
        "测试集 R²": round(test_metrics["r2"], 4),
        "测试集 RMSE": round(test_metrics["rmse"], 4),
        "测试集 MAE": round(test_metrics["mae"], 4),
    }


def save_table(df: pd.DataFrame, stem: str):
    csv_path = DATA_DIR / f"{stem}.csv"
    md_path = DATA_DIR / f"{stem}.md"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    md_path.write_text(df_to_markdown(df), encoding="utf-8")
    return csv_path, md_path


def main():
    os.environ["PYTHONIOENCODING"] = "utf-8"
    config = Config()
    samples, feature_names = prepare_samples(config)

    random_df = run_random_node_year_holdout(samples)
    rolling_detail_df, rolling_avg_df = run_rolling_forecast(samples)

    save_table(random_df, "paper_random_holdout_result_table")
    save_table(rolling_detail_df, "paper_rolling_detail_result_table")
    save_table(rolling_avg_df, "paper_rolling_average_result_table")

    summary = {
        "feature_names": feature_names,
        "random_holdout": random_df.to_dict(orient="records"),
        "rolling_average": rolling_avg_df.to_dict(orient="records"),
        "note": "random_holdout 为省份-年份样本随机留出；rolling_average 为滚动时间外推平均。",
    }
    (DATA_DIR / "paper_style_experiment_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("省份-年份随机留出结果")
    print(random_df.to_string(index=False))
    print("\n滚动时间外推平均结果")
    print(rolling_avg_df.to_string(index=False))


if __name__ == "__main__":
    main()
