import copy
import json
import math
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


DATA_DIR = Path(__file__).resolve().parent
YEARS = [str(y) for y in range(2011, 2024)]
FEATURE_COLUMNS = ["DE", "GDP", "IS", "EDU", "URBAN", "DENSITY", "OPEN"]
TARGET_COLUMN = "EE"
PROVINCE_ORDER = [
    "北京",
    "天津",
    "河北",
    "山西",
    "内蒙古",
    "辽宁",
    "吉林",
    "黑龙江",
    "上海",
    "江苏",
    "浙江",
    "安徽",
    "福建",
    "江西",
    "山东",
    "河南",
    "湖北",
    "湖南",
    "广东",
    "广西",
    "海南",
    "重庆",
    "四川",
    "贵州",
    "云南",
    "陕西",
    "甘肃",
    "青海",
    "宁夏",
    "新疆",
]
NAME_MAP = {
    "北京市": "北京",
    "天津市": "天津",
    "河北省": "河北",
    "山西省": "山西",
    "内蒙古自治区": "内蒙古",
    "辽宁省": "辽宁",
    "吉林省": "吉林",
    "黑龙江省": "黑龙江",
    "上海市": "上海",
    "江苏省": "江苏",
    "浙江省": "浙江",
    "安徽省": "安徽",
    "福建省": "福建",
    "江西省": "江西",
    "山东省": "山东",
    "河南省": "河南",
    "湖北省": "湖北",
    "湖南省": "湖南",
    "广东省": "广东",
    "广西壮族自治区": "广西",
    "海南省": "海南",
    "重庆市": "重庆",
    "四川省": "四川",
    "贵州省": "贵州",
    "云南省": "云南",
    "陕西省": "陕西",
    "甘肃省": "甘肃",
    "青海省": "青海",
    "宁夏回族自治区": "宁夏",
    "新疆维吾尔自治区": "新疆",
}


@dataclass
class Config:
    geo_weight: float = 0.40
    econ_weight: float = 0.35
    digital_weight: float = 0.25
    adaptive_weight: float = 0.05
    use_target_history: bool = True
    use_spatial_lag_features: bool = True
    input_dim: int = 8
    encoder_hidden: int = 32
    hidden_dim: int = 64
    gru_layers: int = 1
    dropout: float = 0.15
    seq_len: int = 3
    horizon: int = 1
    loss_name: str = "mse"
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    max_epochs: int = 300
    batch_size: int = 2
    grad_clip: float = 3.0
    scheduler_patience: int = 12
    scheduler_factor: float = 0.5
    scheduler_min_lr: float = 1e-5
    early_stop_patience: int = 30
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def row_normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = matrix.astype(np.float32)
    np.fill_diagonal(matrix, 0.0)
    row_sum = matrix.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    return matrix / row_sum


def read_matrix_book(path: Path) -> Dict[str, np.ndarray]:
    book = pd.read_excel(path, sheet_name=None, header=0, index_col=0)
    result = {}
    for year, df in book.items():
        df.index = [str(x).strip() for x in df.index]
        df.columns = [str(x).strip() for x in df.columns]
        df = df.loc[PROVINCE_ORDER, PROVINCE_ORDER]
        result[str(year)] = df.to_numpy(dtype=np.float32)
    return result


def read_node_book(path: Path) -> Dict[str, pd.DataFrame]:
    book = pd.read_excel(path, sheet_name=None)
    result = {}
    for year, df in book.items():
        df = df.copy()
        df = df[df["P"].notna()].copy()
        df["province"] = df["P"].map(NAME_MAP)
        df = df[df["province"].notna()].copy()
        df = df.drop_duplicates(subset=["province"], keep="first")
        df = df.set_index("province").reindex(PROVINCE_ORDER)
        if df[FEATURE_COLUMNS + [TARGET_COLUMN]].isnull().any().any():
            raise ValueError(f"{year} 节点表存在缺失值，请先检查数据。")
        result[str(year)] = df
    return result


def load_all_data(config: Config):
    geo = read_matrix_book(DATA_DIR / "1.xlsx")
    econ = read_matrix_book(DATA_DIR / "2.xlsx")
    digital = read_matrix_book(DATA_DIR / "3.xlsx")
    nodes = read_node_book(DATA_DIR / "jd0.xlsx")

    features_by_year: Dict[str, np.ndarray] = {}
    base_features_by_year: Dict[str, np.ndarray] = {}
    targets_by_year: Dict[str, np.ndarray] = {}
    graphs_by_year: Dict[str, np.ndarray] = {}
    feature_names = list(FEATURE_COLUMNS)

    for year in YEARS:
        df = nodes[year].copy()
        df["GDP"] = np.log1p(df["GDP"].astype(np.float32))
        df["DENSITY"] = np.log1p(df["DENSITY"].astype(np.float32))
        df["OPEN"] = np.log1p(df["OPEN"].astype(np.float32))

        base_features_by_year[year] = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        targets_by_year[year] = df[[TARGET_COLUMN]].to_numpy(dtype=np.float32)

        weighted = (
            config.geo_weight * geo[year]
            + config.econ_weight * econ[year]
            + config.digital_weight * digital[year]
        )
        graphs_by_year[year] = row_normalize(weighted)

    if config.use_target_history:
        feature_names = [TARGET_COLUMN] + feature_names
    if config.use_spatial_lag_features:
        feature_names = feature_names + ["W_EE", "W_DE"]

    for year in YEARS:
        parts = []
        if config.use_target_history:
            parts.append(targets_by_year[year])
        parts.append(base_features_by_year[year])
        if config.use_spatial_lag_features:
            graph = graphs_by_year[year]
            spatial_ee = graph @ targets_by_year[year]
            de_idx = FEATURE_COLUMNS.index("DE")
            spatial_de = graph @ base_features_by_year[year][:, [de_idx]]
            parts.extend([spatial_ee, spatial_de])
        features_by_year[year] = np.concatenate(parts, axis=1).astype(np.float32)

    return features_by_year, targets_by_year, graphs_by_year, feature_names


def build_windows(
    features_by_year: Dict[str, np.ndarray],
    targets_by_year: Dict[str, np.ndarray],
    graphs_by_year: Dict[str, np.ndarray],
    seq_len: int,
    horizon: int,
):
    years_int = [int(y) for y in YEARS]
    samples = []
    max_start = len(years_int) - seq_len - horizon + 1
    for start_idx in range(max_start):
        input_years = years_int[start_idx : start_idx + seq_len]
        target_year = years_int[start_idx + seq_len + horizon - 1]
        sample = {
            "input_years": input_years,
            "target_year": target_year,
            "x": np.stack(
                [features_by_year[str(y)] for y in input_years], axis=0
            ).astype(np.float32),
            "a": np.stack(
                [graphs_by_year[str(y)] for y in input_years], axis=0
            ).astype(np.float32),
            "y": targets_by_year[str(target_year)].astype(np.float32),
        }
        samples.append(sample)
    return samples


def split_samples(samples: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    train = [s for s in samples if 2014 <= s["target_year"] <= 2019]
    valid = [s for s in samples if 2020 <= s["target_year"] <= 2021]
    test = [s for s in samples if 2022 <= s["target_year"] <= 2023]
    return train, valid, test


def fit_standardizer(train_samples: List[Dict]):
    x_all = np.concatenate([s["x"].reshape(-1, s["x"].shape[-1]) for s in train_samples], axis=0)
    y_all = np.concatenate([s["y"].reshape(-1, 1) for s in train_samples], axis=0)

    x_mean = x_all.mean(axis=0, keepdims=True).astype(np.float32)
    x_std = x_all.std(axis=0, keepdims=True).astype(np.float32)
    y_mean = y_all.mean(axis=0, keepdims=True).astype(np.float32)
    y_std = y_all.std(axis=0, keepdims=True).astype(np.float32)

    x_std[x_std < 1e-6] = 1.0
    y_std[y_std < 1e-6] = 1.0
    return x_mean, x_std, y_mean, y_std


def transform_samples(samples: List[Dict], x_mean, x_std, y_mean, y_std) -> List[Dict]:
    output = []
    for sample in samples:
        item = dict(sample)
        item["x"] = ((item["x"] - x_mean) / x_std).astype(np.float32)
        item["y"] = ((item["y"] - y_mean) / y_std).astype(np.float32)
        output.append(item)
    return output


class WindowDataset(torch.utils.data.Dataset):
    def __init__(self, samples: Sequence[Dict]):
        self.samples = list(samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return (
            torch.tensor(sample["x"], dtype=torch.float32),
            torch.tensor(sample["a"], dtype=torch.float32),
            torch.tensor(sample["y"], dtype=torch.float32),
            sample["target_year"],
        )


class GraphConvBlock(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.neigh_linear = nn.Linear(hidden_dim, hidden_dim)
        self.self_linear = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        support = torch.matmul(adj, x)
        out = self.neigh_linear(support) + self.self_linear(x)
        out = self.norm(out)
        return torch.relu(out)


class STGNNRegressor(nn.Module):
    def __init__(self, config: Config, num_nodes: int):
        super().__init__()
        self.config = config
        self.num_nodes = num_nodes
        self.encoder = nn.Sequential(
            nn.Linear(config.input_dim, config.encoder_hidden),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.encoder_hidden, config.hidden_dim),
            nn.ReLU(),
        )
        self.graph_block = GraphConvBlock(config.hidden_dim)
        self.adaptive_logits = nn.Parameter(torch.zeros(num_nodes, num_nodes))
        self.gru = nn.GRU(
            input_size=config.hidden_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.gru_layers,
            dropout=config.dropout if config.gru_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(config.hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor, adj_seq: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, num_nodes, _ = x.shape
        encoded_steps = []
        adaptive = torch.softmax(self.adaptive_logits, dim=-1)
        adaptive = adaptive * (1.0 - torch.eye(self.num_nodes, device=x.device))

        for t in range(seq_len):
            xt = self.encoder(x[:, t])
            adj = adj_seq[:, t]
            adj = (1.0 - self.config.adaptive_weight) * adj + self.config.adaptive_weight * adaptive.unsqueeze(0)
            spatial = self.graph_block(xt, adj) + xt
            encoded_steps.append(spatial)

        spatial_seq = torch.stack(encoded_steps, dim=1)
        gru_input = spatial_seq.permute(0, 2, 1, 3).reshape(batch_size * num_nodes, seq_len, self.config.hidden_dim)
        gru_out, _ = self.gru(gru_input)
        final_state = gru_out[:, -1, :]
        pred = self.head(final_state)
        return pred.reshape(batch_size, num_nodes, 1)


def inverse_scale(values: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return values * std + mean


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: str,
    y_mean: np.ndarray,
    y_std: np.ndarray,
):
    model.eval()
    preds = []
    targets = []
    years = []
    with torch.no_grad():
        for x, a, y, target_year in loader:
            x = x.to(device)
            a = a.to(device)
            pred = model(x, a).cpu().numpy()
            preds.append(pred)
            targets.append(y.numpy())
            years.extend(target_year.tolist())

    pred_arr = np.concatenate(preds, axis=0).reshape(-1, 1)
    true_arr = np.concatenate(targets, axis=0).reshape(-1, 1)
    pred_arr = inverse_scale(pred_arr, y_mean, y_std)
    true_arr = inverse_scale(true_arr, y_mean, y_std)
    metrics = compute_metrics(true_arr.ravel(), pred_arr.ravel())
    return metrics, true_arr, pred_arr, years


def flatten_samples_for_baseline(samples: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
    x_rows = []
    y_rows = []
    for sample in samples:
        seq_features = sample["x"].reshape(sample["x"].shape[0], sample["x"].shape[1], sample["x"].shape[2])
        graph_strength = sample["a"].sum(axis=2, keepdims=False)
        for node_idx in range(seq_features.shape[1]):
            node_hist = seq_features[:, node_idx, :].reshape(-1)
            node_degree = graph_strength[:, node_idx].reshape(-1)
            row = np.concatenate([node_hist, node_degree], axis=0)
            x_rows.append(row)
            y_rows.append(sample["y"][node_idx, 0])
    return np.asarray(x_rows, dtype=np.float32), np.asarray(y_rows, dtype=np.float32)


def run_baselines(
    train_samples: List[Dict],
    valid_samples: List[Dict],
    test_samples: List[Dict],
    y_mean: np.ndarray,
    y_std: np.ndarray,
):
    x_train, y_train = flatten_samples_for_baseline(train_samples)
    x_valid, y_valid = flatten_samples_for_baseline(valid_samples)
    x_test, y_test = flatten_samples_for_baseline(test_samples)

    y_train_raw = inverse_scale(y_train.reshape(-1, 1), y_mean, y_std).ravel()
    y_valid_raw = inverse_scale(y_valid.reshape(-1, 1), y_mean, y_std).ravel()
    y_test_raw = inverse_scale(y_test.reshape(-1, 1), y_mean, y_std).ravel()

    models = {
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "RandomForest": RandomForestRegressor(
            n_estimators=300, max_depth=4, min_samples_leaf=3, random_state=42
        ),
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=400, max_depth=6, min_samples_leaf=2, random_state=42
        ),
        "HistGBDT": HistGradientBoostingRegressor(
            learning_rate=0.03,
            max_depth=3,
            max_iter=300,
            min_samples_leaf=8,
            l2_regularization=0.01,
            random_state=42,
        ),
    }

    results = {}
    fitted_models = {}
    for name, model in models.items():
        model.fit(x_train, y_train_raw)
        fitted_models[name] = model
        pred_train = model.predict(x_train)
        pred_valid = model.predict(x_valid)
        pred_test = model.predict(x_test)
        results[name] = {
            "train_metrics": compute_metrics(y_train_raw, pred_train),
            "valid_metrics": compute_metrics(y_valid_raw, pred_valid),
            "test_metrics": compute_metrics(y_test_raw, pred_test),
        }

    return results, fitted_models


def format_results_table(model_results: Dict[str, Dict]) -> pd.DataFrame:
    rows = []
    for model_name, result in model_results.items():
        rows.append(
            {
                "模型": model_name,
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
        )
    return pd.DataFrame(rows).sort_values(by=["验证集 R²", "测试集 R²"], ascending=False)


def make_paper_summary(
    config: Config,
    feature_names: List[str],
    stgnn_summary: Dict,
    baseline_results: Dict[str, Dict],
) -> Dict:
    all_results = {"STGNN": {
        "train_metrics": stgnn_summary["train_metrics"],
        "valid_metrics": stgnn_summary["valid_metrics"],
        "test_metrics": stgnn_summary["test_metrics"],
    }}
    all_results.update(baseline_results)

    result_table = format_results_table(all_results)
    param_table = pd.DataFrame(
        [
            {"参数": "图权重(G,E,D)", "取值": f"{config.geo_weight:.2f}, {config.econ_weight:.2f}, {config.digital_weight:.2f}"},
            {"参数": "自适应邻接权重", "取值": config.adaptive_weight},
            {"参数": "输入特征数", "取值": config.input_dim},
            {"参数": "输入特征", "取值": "、".join(feature_names)},
            {"参数": "历史窗口", "取值": config.seq_len},
            {"参数": "隐藏维度", "取值": config.hidden_dim},
            {"参数": "GRU层数", "取值": config.gru_layers},
            {"参数": "Dropout", "取值": config.dropout},
            {"参数": "损失函数", "取值": config.loss_name},
            {"参数": "学习率", "取值": config.learning_rate},
            {"参数": "权重衰减", "取值": config.weight_decay},
            {"参数": "Batch size", "取值": config.batch_size},
            {"参数": "最佳轮次", "取值": stgnn_summary["best_epoch"]},
        ]
    )

    return {
        "result_table": result_table,
        "param_table": param_table,
    }


def run_training(config: Config):
    set_seed(config.seed)
    features_by_year, targets_by_year, graphs_by_year, feature_names = load_all_data(config)
    samples = build_windows(
        features_by_year, targets_by_year, graphs_by_year, config.seq_len, config.horizon
    )
    train_samples, valid_samples, test_samples = split_samples(samples)

    x_mean, x_std, y_mean, y_std = fit_standardizer(train_samples)
    train_samples = transform_samples(train_samples, x_mean, x_std, y_mean, y_std)
    valid_samples = transform_samples(valid_samples, x_mean, x_std, y_mean, y_std)
    test_samples = transform_samples(test_samples, x_mean, x_std, y_mean, y_std)

    train_loader = torch.utils.data.DataLoader(
        WindowDataset(train_samples),
        batch_size=config.batch_size,
        shuffle=True,
    )
    valid_loader = torch.utils.data.DataLoader(
        WindowDataset(valid_samples),
        batch_size=config.batch_size,
        shuffle=False,
    )
    test_loader = torch.utils.data.DataLoader(
        WindowDataset(test_samples),
        batch_size=config.batch_size,
        shuffle=False,
    )

    device = config.device
    model = STGNNRegressor(config, num_nodes=len(PROVINCE_ORDER)).to(device)
    if config.loss_name.lower() == "huber":
        criterion = nn.HuberLoss(delta=1.0)
    else:
        criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
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
    best_valid_metrics = None
    wait = 0
    history = []

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        train_losses = []
        for x, a, y, _ in train_loader:
            x = x.to(device)
            a = a.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            pred = model(x, a)
            loss = criterion(pred, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            train_losses.append(loss.item())

        train_metrics, _, _, _ = evaluate(model, train_loader, device, y_mean, y_std)
        valid_metrics, _, _, _ = evaluate(model, valid_loader, device, y_mean, y_std)

        avg_train_loss = float(np.mean(train_losses))
        scheduler.step(valid_metrics["rmse"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "train_r2": train_metrics["r2"],
                "valid_r2": valid_metrics["r2"],
                "valid_rmse": valid_metrics["rmse"],
                "valid_mae": valid_metrics["mae"],
                "lr": optimizer.param_groups[0]["lr"],
            }
        )

        if valid_metrics["r2"] > best_valid_r2:
            best_valid_r2 = valid_metrics["r2"]
            best_epoch = epoch
            best_valid_metrics = valid_metrics
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1

        if epoch == 1 or epoch % 20 == 0:
            print(
                f"Epoch {epoch:03d} | "
                f"train_r2={train_metrics['r2']:.4f} | "
                f"valid_r2={valid_metrics['r2']:.4f} | "
                f"valid_rmse={valid_metrics['rmse']:.4f} | "
                f"lr={optimizer.param_groups[0]['lr']:.6f}"
            )

        if wait >= config.early_stop_patience:
            print(f"Early stopping at epoch {epoch}.")
            break

    if best_state is None:
        raise RuntimeError("训练过程中没有得到有效模型。")

    model.load_state_dict(best_state)
    train_metrics, train_true, train_pred, train_years = evaluate(
        model, train_loader, device, y_mean, y_std
    )
    valid_metrics, valid_true, valid_pred, valid_years = evaluate(
        model, valid_loader, device, y_mean, y_std
    )
    test_metrics, test_true, test_pred, test_years = evaluate(
        model, test_loader, device, y_mean, y_std
    )

    summary = {
        "config": asdict(config),
        "feature_names": feature_names,
        "best_epoch": best_epoch,
        "best_valid_metrics": best_valid_metrics,
        "train_metrics": train_metrics,
        "valid_metrics": valid_metrics,
        "test_metrics": test_metrics,
        "history": history,
        "split_years": {
            "train_target_years": sorted(set(train_years)),
            "valid_target_years": sorted(set(valid_years)),
            "test_target_years": sorted(set(test_years)),
        },
        "predictions": {
            "train": {
                "years": train_years,
                "y_true": train_true.reshape(-1).tolist(),
                "y_pred": train_pred.reshape(-1).tolist(),
            },
            "valid": {
                "years": valid_years,
                "y_true": valid_true.reshape(-1).tolist(),
                "y_pred": valid_pred.reshape(-1).tolist(),
            },
            "test": {
                "years": test_years,
                "y_true": test_true.reshape(-1).tolist(),
                "y_pred": test_pred.reshape(-1).tolist(),
            },
        },
    }

    baseline_results, _ = run_baselines(train_samples, valid_samples, test_samples, y_mean, y_std)
    paper_summary = make_paper_summary(config, feature_names, summary, baseline_results)
    summary["baseline_results"] = baseline_results
    summary["paper_tables"] = {
        "result_table": paper_summary["result_table"].to_dict(orient="records"),
        "param_table": paper_summary["param_table"].to_dict(orient="records"),
    }
    return summary


def print_summary(summary: Dict) -> None:
    print("\n最终结果")
    print("-" * 72)
    print(
        f"最佳轮次: {summary['best_epoch']} | "
        f"训练集 R2={summary['train_metrics']['r2']:.4f}, "
        f"RMSE={summary['train_metrics']['rmse']:.4f}, "
        f"MAE={summary['train_metrics']['mae']:.4f}"
    )
    print(
        f"验证集 R2={summary['valid_metrics']['r2']:.4f}, "
        f"RMSE={summary['valid_metrics']['rmse']:.4f}, "
        f"MAE={summary['valid_metrics']['mae']:.4f}"
    )
    print(
        f"测试集 R2={summary['test_metrics']['r2']:.4f}, "
        f"RMSE={summary['test_metrics']['rmse']:.4f}, "
        f"MAE={summary['test_metrics']['mae']:.4f}"
    )
    print("-" * 72)
    if "baseline_results" in summary:
        print("基线对照")
        for model_name, result in summary["baseline_results"].items():
            print(
                f"{model_name}: "
                f"valid_r2={result['valid_metrics']['r2']:.4f}, "
                f"test_r2={result['test_metrics']['r2']:.4f}"
            )
        print("-" * 72)


def save_paper_outputs(summary: Dict) -> None:
    result_df = pd.DataFrame(summary["paper_tables"]["result_table"])
    param_df = pd.DataFrame(summary["paper_tables"]["param_table"])

    result_csv = DATA_DIR / "paper_result_table.csv"
    param_csv = DATA_DIR / "paper_param_table.csv"
    result_md = DATA_DIR / "paper_result_table.md"
    param_md = DATA_DIR / "paper_param_table.md"

    result_df.to_csv(result_csv, index=False, encoding="utf-8-sig")
    param_df.to_csv(param_csv, index=False, encoding="utf-8-sig")
    result_md.write_text(df_to_markdown(result_df), encoding="utf-8")
    param_md.write_text(df_to_markdown(param_df), encoding="utf-8")


def df_to_markdown(df: pd.DataFrame) -> str:
    headers = [str(col) for col in df.columns]
    rows = [[str(value) for value in row] for row in df.to_numpy()]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def main():
    config = Config()
    config.input_dim = len(FEATURE_COLUMNS) + int(config.use_target_history) + (2 if config.use_spatial_lag_features else 0)
    summary = run_training(config)
    print_summary(summary)
    out_path = DATA_DIR / "stgnn_training_summary.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    save_paper_outputs(summary)
    print(f"结果已写入: {out_path}")


if __name__ == "__main__":
    main()
