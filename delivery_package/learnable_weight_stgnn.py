import copy
import json
import math
import random
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from train_stgnn_excel import (
    Config,
    FEATURE_COLUMNS,
    PROVINCE_ORDER,
    STGNNRegressor,
    build_windows,
    compute_metrics,
    fit_standardizer,
    inverse_scale,
    read_matrix_book,
    read_node_book,
    row_normalize,
    transform_samples,
)


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "stability_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
YEARS = [str(y) for y in range(2011, 2024)]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def random_split_samples(samples: List[Dict], seed: int) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    idx = list(range(len(samples)))
    random.Random(seed).shuffle(idx)
    train_idx = idx[:6]
    valid_idx = idx[6:8]
    test_idx = idx[8:10]
    return [samples[i] for i in train_idx], [samples[i] for i in valid_idx], [samples[i] for i in test_idx]


def finalize_config(config: Config) -> Config:
    config.input_dim = len(FEATURE_COLUMNS) + int(config.use_target_history)
    if config.use_spatial_lag_features:
        config.input_dim += 2
    if config.use_second_order_lag_features:
        config.input_dim += 4
    if config.use_change_features:
        config.input_dim += 2
    return config


def build_default_config(seed: int) -> Config:
    return finalize_config(
        Config(
            seed=seed,
            geo_weight=0.33,
            econ_weight=0.33,
            digital_weight=0.34,
            adaptive_weight=0.02,
            use_target_history=True,
            use_spatial_lag_features=True,
            use_second_order_lag_features=False,
            use_change_features=False,
            encoder_hidden=16,
            hidden_dim=32,
            gru_layers=1,
            dropout=0.05,
            seq_len=2,
            horizon=1,
            loss_name="mse",
            learning_rate=5e-4,
            weight_decay=1e-6,
            max_epochs=320,
            batch_size=2,
            grad_clip=3.0,
            scheduler_patience=14,
            scheduler_factor=0.5,
            scheduler_min_lr=1e-5,
            early_stop_patience=40,
        )
    )


def load_all_data_with_three_graphs(config: Config):
    geo_raw = read_matrix_book(BASE_DIR / "1.xlsx")
    econ_raw = read_matrix_book(BASE_DIR / "2.xlsx")
    digital_raw = read_matrix_book(BASE_DIR / "3.xlsx")
    nodes = read_node_book(BASE_DIR / "jd0.xlsx")

    base_features_by_year: Dict[str, np.ndarray] = {}
    targets_by_year: Dict[str, np.ndarray] = {}
    graph_components_by_year: Dict[str, Dict[str, np.ndarray]] = {}
    feature_names = list(FEATURE_COLUMNS)

    for year in YEARS:
        df = nodes[year].copy()
        df["GDP"] = np.log1p(df["GDP"].astype(np.float32))
        df["DENSITY"] = np.log1p(df["DENSITY"].astype(np.float32))
        df["OPEN"] = np.log1p(df["OPEN"].astype(np.float32))

        base_features_by_year[year] = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        targets_by_year[year] = df[["EE"]].to_numpy(dtype=np.float32)
        graph_components_by_year[year] = {
            "geo": row_normalize(geo_raw[year]),
            "econ": row_normalize(econ_raw[year]),
            "digital": row_normalize(digital_raw[year]),
        }

    if config.use_target_history:
        feature_names = ["EE"] + feature_names
    if config.use_spatial_lag_features:
        feature_names = feature_names + ["W_EE", "W_DE"]
    if config.use_second_order_lag_features:
        feature_names = feature_names + ["EE_lag1", "EE_lag2", "DE_lag1", "DE_lag2"]
    if config.use_change_features:
        feature_names = feature_names + ["dEE", "dDE"]

    year_index = {year: idx for idx, year in enumerate(YEARS)}
    features_by_year: Dict[str, np.ndarray] = {}
    for year in YEARS:
        parts = []
        if config.use_target_history:
            parts.append(targets_by_year[year])
        parts.append(base_features_by_year[year])
        if config.use_spatial_lag_features:
            eq_adj = (
                graph_components_by_year[year]["geo"]
                + graph_components_by_year[year]["econ"]
                + graph_components_by_year[year]["digital"]
            ) / 3.0
            spatial_ee = eq_adj @ targets_by_year[year]
            de_idx = FEATURE_COLUMNS.index("DE")
            spatial_de = eq_adj @ base_features_by_year[year][:, [de_idx]]
            parts.extend([spatial_ee, spatial_de])
        if config.use_second_order_lag_features or config.use_change_features:
            idx = year_index[year]
            prev1 = YEARS[idx - 1] if idx - 1 >= 0 else year
            prev2 = YEARS[idx - 2] if idx - 2 >= 0 else prev1
            prev1_ee = targets_by_year[prev1]
            prev2_ee = targets_by_year[prev2]
            de_idx = FEATURE_COLUMNS.index("DE")
            prev1_de = base_features_by_year[prev1][:, [de_idx]]
            prev2_de = base_features_by_year[prev2][:, [de_idx]]
            if config.use_second_order_lag_features:
                parts.extend([prev1_ee, prev2_ee, prev1_de, prev2_de])
            if config.use_change_features:
                parts.extend([targets_by_year[year] - prev1_ee, base_features_by_year[year][:, [de_idx]] - prev1_de])
        features_by_year[year] = np.concatenate(parts, axis=1).astype(np.float32)

    return features_by_year, targets_by_year, graph_components_by_year, feature_names


def build_windows_with_components(
    features_by_year: Dict[str, np.ndarray],
    targets_by_year: Dict[str, np.ndarray],
    graph_components_by_year: Dict[str, Dict[str, np.ndarray]],
    seq_len: int,
    horizon: int,
) -> List[Dict]:
    years_int = [int(y) for y in YEARS]
    samples = []
    max_start = len(years_int) - seq_len - horizon + 1
    for start_idx in range(max_start):
        input_years = years_int[start_idx : start_idx + seq_len]
        target_year = years_int[start_idx + seq_len + horizon - 1]
        geo_seq = np.stack([graph_components_by_year[str(y)]["geo"] for y in input_years], axis=0).astype(np.float32)
        econ_seq = np.stack([graph_components_by_year[str(y)]["econ"] for y in input_years], axis=0).astype(np.float32)
        digital_seq = np.stack([graph_components_by_year[str(y)]["digital"] for y in input_years], axis=0).astype(np.float32)
        eq_adj = (geo_seq + econ_seq + digital_seq) / 3.0
        samples.append(
            {
                "input_years": input_years,
                "target_year": target_year,
                "x": np.stack([features_by_year[str(y)] for y in input_years], axis=0).astype(np.float32),
                "a": eq_adj,
                "geo_a": geo_seq,
                "econ_a": econ_seq,
                "digital_a": digital_seq,
                "y": targets_by_year[str(target_year)].astype(np.float32),
            }
        )
    return samples


class WindowDatasetWithComponents(Dataset):
    def __init__(self, samples: List[Dict]):
        self.samples = list(samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return (
            torch.tensor(sample["x"], dtype=torch.float32),
            torch.tensor(sample["geo_a"], dtype=torch.float32),
            torch.tensor(sample["econ_a"], dtype=torch.float32),
            torch.tensor(sample["digital_a"], dtype=torch.float32),
            torch.tensor(sample["y"], dtype=torch.float32),
            sample["target_year"],
        )


class LearnableWeightSTGNN(STGNNRegressor):
    def __init__(self, config: Config, num_nodes: int):
        super().__init__(config, num_nodes)
        self.graph_weight_logits = nn.Parameter(torch.zeros(3))

    def get_graph_weights(self) -> torch.Tensor:
        return torch.softmax(self.graph_weight_logits, dim=0)

    def forward(self, x: torch.Tensor, geo_adj: torch.Tensor, econ_adj: torch.Tensor, digital_adj: torch.Tensor) -> torch.Tensor:
        weights = self.get_graph_weights()
        combined = weights[0] * geo_adj + weights[1] * econ_adj + weights[2] * digital_adj
        return super().forward(x, combined)


def evaluate_model(model, loader, device: str, y_mean: np.ndarray, y_std: np.ndarray):
    model.eval()
    preds = []
    targets = []
    years = []
    with torch.no_grad():
        for x, geo_a, econ_a, digital_a, y, target_year in loader:
            x = x.to(device)
            geo_a = geo_a.to(device)
            econ_a = econ_a.to(device)
            digital_a = digital_a.to(device)
            pred = model(x, geo_a, econ_a, digital_a).cpu().numpy()
            preds.append(pred)
            targets.append(y.numpy())
            years.extend(target_year.tolist())

    pred_arr = np.concatenate(preds, axis=0).reshape(-1, 1)
    true_arr = np.concatenate(targets, axis=0).reshape(-1, 1)
    pred_arr = inverse_scale(pred_arr, y_mean, y_std)
    true_arr = inverse_scale(true_arr, y_mean, y_std)
    metrics = compute_metrics(true_arr.ravel(), pred_arr.ravel())
    return metrics, true_arr, pred_arr, years


def train_single_run(config: Config, split_seed: int = 2026) -> Dict:
    set_seed(config.seed)
    features_by_year, targets_by_year, graph_components_by_year, feature_names = load_all_data_with_three_graphs(config)
    samples = build_windows_with_components(features_by_year, targets_by_year, graph_components_by_year, config.seq_len, config.horizon)
    train_raw, valid_raw, test_raw = random_split_samples(samples, split_seed)

    x_mean, x_std, y_mean, y_std = fit_standardizer(train_raw)
    train = transform_samples(train_raw, x_mean, x_std, y_mean, y_std)
    valid = transform_samples(valid_raw, x_mean, x_std, y_mean, y_std)
    test = transform_samples(test_raw, x_mean, x_std, y_mean, y_std)

    train_loader = DataLoader(WindowDatasetWithComponents(train), batch_size=config.batch_size, shuffle=True)
    valid_loader = DataLoader(WindowDatasetWithComponents(valid), batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(WindowDatasetWithComponents(test), batch_size=config.batch_size, shuffle=False)

    model = LearnableWeightSTGNN(config, num_nodes=len(PROVINCE_ORDER)).to(config.device)
    criterion = nn.MSELoss() if config.loss_name.lower() == "mse" else nn.HuberLoss(delta=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.scheduler_factor,
        patience=config.scheduler_patience,
        min_lr=config.scheduler_min_lr,
    )

    best_state = None
    best_valid_r2 = -float("inf")
    best_epoch = -1
    wait = 0
    history = []

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        losses = []
        for x, geo_a, econ_a, digital_a, y, _ in train_loader:
            x = x.to(config.device)
            geo_a = geo_a.to(config.device)
            econ_a = econ_a.to(config.device)
            digital_a = digital_a.to(config.device)
            y = y.to(config.device)

            optimizer.zero_grad()
            pred = model(x, geo_a, econ_a, digital_a)
            loss = criterion(pred, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            losses.append(loss.item())

        train_metrics, _, _, _ = evaluate_model(model, train_loader, config.device, y_mean, y_std)
        valid_metrics, _, _, _ = evaluate_model(model, valid_loader, config.device, y_mean, y_std)
        scheduler.step(valid_metrics["rmse"])

        weights = model.get_graph_weights().detach().cpu().numpy().tolist()
        history.append(
            {
                "epoch": epoch,
                "loss": float(np.mean(losses)),
                "train_r2": train_metrics["r2"],
                "valid_r2": valid_metrics["r2"],
                "valid_rmse": valid_metrics["rmse"],
                "weight_geo": weights[0],
                "weight_econ": weights[1],
                "weight_digital": weights[2],
            }
        )

        if valid_metrics["r2"] > best_valid_r2:
            best_valid_r2 = valid_metrics["r2"]
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            wait = 0
        else:
            wait += 1

        if epoch == 1 or epoch % 25 == 0:
            print(
                f"seed={config.seed} epoch={epoch:03d} "
                f"train_r2={train_metrics['r2']:.4f} valid_r2={valid_metrics['r2']:.4f} "
                f"weights={weights[0]:.4f}/{weights[1]:.4f}/{weights[2]:.4f}"
            )

        if wait >= config.early_stop_patience:
            print(f"seed={config.seed} early stop at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    final_weights = model.get_graph_weights().detach().cpu().numpy()
    train_metrics, _, _, _ = evaluate_model(model, train_loader, config.device, y_mean, y_std)
    valid_metrics, _, _, _ = evaluate_model(model, valid_loader, config.device, y_mean, y_std)
    test_metrics, _, _, _ = evaluate_model(model, test_loader, config.device, y_mean, y_std)

    return {
        "config": asdict(config),
        "feature_names": feature_names,
        "best_epoch": best_epoch,
        "weight_geo": float(final_weights[0]),
        "weight_econ": float(final_weights[1]),
        "weight_digital": float(final_weights[2]),
        "train_metrics": train_metrics,
        "valid_metrics": valid_metrics,
        "test_metrics": test_metrics,
        "history": history,
    }


def main() -> None:
    seeds = [22, 52, 72, 102, 132]
    split_seed = 2026
    rows = []
    all_results = []

    for seed in seeds:
        print(f"\n{'=' * 88}\nlearnable-weight seed={seed}")
        config = build_default_config(seed)
        result = train_single_run(config=config, split_seed=split_seed)
        row = {
            "seed": seed,
            "best_epoch": result["best_epoch"],
            "alpha_geo": round(result["weight_geo"], 4),
            "beta_econ": round(result["weight_econ"], 4),
            "gamma_digital": round(result["weight_digital"], 4),
            "train_r2": round(result["train_metrics"]["r2"], 4),
            "valid_r2": round(result["valid_metrics"]["r2"], 4),
            "test_r2": round(result["test_metrics"]["r2"], 4),
            "train_rmse": round(result["train_metrics"]["rmse"], 4),
            "valid_rmse": round(result["valid_metrics"]["rmse"], 4),
            "test_rmse": round(result["test_metrics"]["rmse"], 4),
            "train_mae": round(result["train_metrics"]["mae"], 4),
            "valid_mae": round(result["valid_metrics"]["mae"], 4),
            "test_mae": round(result["test_metrics"]["mae"], 4),
        }
        rows.append(row)
        all_results.append(result)
        print(row)

    df = pd.DataFrame(rows).sort_values(["valid_r2", "test_r2"], ascending=False)
    best_row = df.iloc[0]
    mean_row = df.mean(numeric_only=True)

    output = {
        "split_seed": split_seed,
        "seeds": seeds,
        "runs": rows,
        "best_by_valid": best_row.to_dict(),
        "mean_weights": {
            "alpha_geo": float(mean_row["alpha_geo"]),
            "beta_econ": float(mean_row["beta_econ"]),
            "gamma_digital": float(mean_row["gamma_digital"]),
        },
        "mean_metrics": {
            "train_r2": float(mean_row["train_r2"]),
            "valid_r2": float(mean_row["valid_r2"]),
            "test_r2": float(mean_row["test_r2"]),
            "train_rmse": float(mean_row["train_rmse"]),
            "valid_rmse": float(mean_row["valid_rmse"]),
            "test_rmse": float(mean_row["test_rmse"]),
            "train_mae": float(mean_row["train_mae"]),
            "valid_mae": float(mean_row["valid_mae"]),
            "test_mae": float(mean_row["test_mae"]),
        },
    }

    csv_path = OUTPUT_DIR / "learnable_weight_stgnn_runs.csv"
    json_path = OUTPUT_DIR / "learnable_weight_stgnn_summary.json"
    md_path = OUTPUT_DIR / "learnable_weight_stgnn_runs.md"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    json_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    md_lines = [
        "| seed | alpha_geo | beta_econ | gamma_digital | train_r2 | valid_r2 | test_r2 | train_rmse | valid_rmse | test_rmse | train_mae | valid_mae | test_mae |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in df.to_dict(orient="records"):
        md_lines.append(
            f"| {int(row['seed'])} | {row['alpha_geo']} | {row['beta_econ']} | {row['gamma_digital']} | {row['train_r2']} | {row['valid_r2']} | {row['test_r2']} | {row['train_rmse']} | {row['valid_rmse']} | {row['test_rmse']} | {row['train_mae']} | {row['valid_mae']} | {row['test_mae']} |"
        )
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print("\n最优可学习权重结果")
    print(df.to_string(index=False))
    print(f"\nSaved: {csv_path}")
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
