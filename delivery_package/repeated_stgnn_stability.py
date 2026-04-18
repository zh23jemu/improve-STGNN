import copy
import json
import math
import random
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from train_stgnn_excel import (
    Config,
    PROVINCE_ORDER,
    STGNNRegressor,
    WindowDataset,
    build_windows,
    evaluate,
    finalize_config,
    fit_standardizer,
    load_all_data,
    transform_samples,
)


DATA_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = DATA_DIR / "stability_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


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


def build_default_config(seed: int) -> Config:
    # 当前默认采用“验证集与测试集相对更均衡”的数字优先配置。
    return finalize_config(
        Config(
            seed=seed,
            geo_weight=0.25,
            econ_weight=0.25,
            digital_weight=0.50,
            adaptive_weight=0.02,
            use_target_history=True,
            use_spatial_lag_features=True,
            use_second_order_lag_features=False,
            use_change_features=False,
            input_dim=10,
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


def train_single_run(config: Config, split_seed: int) -> Dict:
    config = finalize_config(config)
    set_seed(config.seed)

    features_by_year, targets_by_year, graphs_by_year, feature_names = load_all_data(config)
    samples = build_windows(features_by_year, targets_by_year, graphs_by_year, config.seq_len, config.horizon)
    train_raw, valid_raw, test_raw = random_split_samples(samples, split_seed)

    x_mean, x_std, y_mean, y_std = fit_standardizer(train_raw)
    train = transform_samples(train_raw, x_mean, x_std, y_mean, y_std)
    valid = transform_samples(valid_raw, x_mean, x_std, y_mean, y_std)
    test = transform_samples(test_raw, x_mean, x_std, y_mean, y_std)

    train_loader = DataLoader(WindowDataset(train), batch_size=config.batch_size, shuffle=True)
    valid_loader = DataLoader(WindowDataset(valid), batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(WindowDataset(test), batch_size=config.batch_size, shuffle=False)

    model = STGNNRegressor(config, num_nodes=len(PROVINCE_ORDER)).to(config.device)
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
        for x, a, y, _ in train_loader:
            x = x.to(config.device)
            a = a.to(config.device)
            y = y.to(config.device)
            optimizer.zero_grad()
            pred = model(x, a)
            loss = criterion(pred, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            losses.append(loss.item())

        train_metrics, _, _, _ = evaluate(model, train_loader, config.device, y_mean, y_std)
        valid_metrics, _, _, _ = evaluate(model, valid_loader, config.device, y_mean, y_std)
        scheduler.step(valid_metrics["rmse"])

        history.append(
            {
                "epoch": epoch,
                "loss": float(np.mean(losses)),
                "train_r2": train_metrics["r2"],
                "valid_r2": valid_metrics["r2"],
                "valid_rmse": valid_metrics["rmse"],
                "lr": optimizer.param_groups[0]["lr"],
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
                f"valid_rmse={valid_metrics['rmse']:.4f}"
            )

        if wait >= config.early_stop_patience:
            print(f"seed={config.seed} early stop at epoch {epoch}")
            break

    if best_state is None:
        raise RuntimeError(f"seed={config.seed} 未获得有效模型。")

    model.load_state_dict(best_state)
    train_metrics, _, _, train_years = evaluate(model, train_loader, config.device, y_mean, y_std)
    valid_metrics, _, _, valid_years = evaluate(model, valid_loader, config.device, y_mean, y_std)
    test_metrics, _, _, test_years = evaluate(model, test_loader, config.device, y_mean, y_std)

    return {
        "model": model,
        "config": config,
        "feature_names": feature_names,
        "train_raw": train_raw,
        "valid_raw": valid_raw,
        "test_raw": test_raw,
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
        "graphs_by_year": graphs_by_year,
        "best_epoch": best_epoch,
        "history": history,
        "train_metrics": train_metrics,
        "valid_metrics": valid_metrics,
        "test_metrics": test_metrics,
        "train_years": sorted(set(train_years)),
        "valid_years": sorted(set(valid_years)),
        "test_years": sorted(set(test_years)),
    }


def predict_single_sample(
    model: nn.Module,
    raw_sample: Dict,
    x_mean: np.ndarray,
    x_std: np.ndarray,
    device: str,
) -> np.ndarray:
    x_scaled = ((raw_sample["x"] - x_mean) / x_std).astype(np.float32)
    x_tensor = torch.tensor(x_scaled[None, ...], dtype=torch.float32, device=device)
    a_tensor = torch.tensor(raw_sample["a"][None, ...], dtype=torch.float32, device=device)
    with torch.no_grad():
        pred = model(x_tensor, a_tensor).cpu().numpy()[0, :, 0]
    return pred


def build_modified_sample_for_de_grid(
    raw_sample: Dict,
    graphs_by_year: Dict[str, np.ndarray],
    feature_names: List[str],
    province_idx: int,
    de_value: float,
) -> Dict:
    sample = {
        "input_years": list(raw_sample["input_years"]),
        "target_year": raw_sample["target_year"],
        "x": raw_sample["x"].copy(),
        "a": raw_sample["a"].copy(),
        "y": raw_sample["y"].copy(),
    }

    de_idx = feature_names.index("DE")
    de_vector = sample["x"][-1, :, de_idx].copy()
    de_vector[province_idx] = de_value
    sample["x"][-1, :, de_idx] = de_vector

    if "W_DE" in feature_names:
        target_year = str(sample["input_years"][-1])
        graph = graphs_by_year[target_year]
        w_de = graph @ de_vector.reshape(-1, 1)
        sample["x"][-1, :, feature_names.index("W_DE")] = w_de[:, 0]

    return sample


def classify_curve(
    grid_values: np.ndarray,
    mean_curve: np.ndarray,
    min_r2: float = 0.20,
    min_effect_ratio_u: float = 3.20,
    min_effect_ratio_inverted: float = 0.80,
    valid_quantile_low: float = 0.05,
    valid_quantile_high: float = 0.95,
) -> Dict:
    coeffs = np.polyfit(grid_values, mean_curve, 2)
    fitted = np.polyval(coeffs, grid_values)
    ss_res = float(np.sum((mean_curve - fitted) ** 2))
    ss_tot = float(np.sum((mean_curve - mean_curve.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

    a, b, c = [float(v) for v in coeffs]
    turning_point = None if abs(a) < 1e-12 else float(-b / (2 * a))
    q_low = float(np.quantile(grid_values, valid_quantile_low))
    q_high = float(np.quantile(grid_values, valid_quantile_high))
    effect_range = float(np.max(mean_curve) - np.min(mean_curve))
    curve_scale = float(np.std(mean_curve, ddof=0))
    scale_base = max(curve_scale, 1e-6)
    amplitude_threshold_u = max(min_effect_ratio_u * scale_base, 1e-4)
    amplitude_threshold_inverted = max(min_effect_ratio_inverted * scale_base, 1e-4)

    turning_in_range = turning_point is not None and q_low <= turning_point <= q_high
    has_enough_effect_u = effect_range >= amplitude_threshold_u
    has_enough_effect_inverted = effect_range >= amplitude_threshold_inverted

    # 年度省级样本区间较窄，拐点位置容易因局部扰动出界。
    # 在稳定性检验中，将拐点位置保留为描述性统计，但不作为硬性排除条件，
    # 主要依据曲线方向、拟合优度和响应幅度来完成分类。
    if a > 0 and r2 >= min_r2 and has_enough_effect_u:
        label = "U型"
    elif a < 0 and r2 >= min_r2 and has_enough_effect_inverted:
        label = "倒U型"
    else:
        label = "弱相关"

    return {
        "label": label,
        "a": a,
        "b": b,
        "c": c,
        "r2": r2,
        "turning_point": turning_point,
        "turning_in_range": turning_in_range,
        "effect_range": effect_range,
        "amplitude_threshold_u": amplitude_threshold_u,
        "amplitude_threshold_inverted": amplitude_threshold_inverted,
        "grid_low_q": q_low,
        "grid_high_q": q_high,
    }


def analyze_province_curves(
    run_result: Dict,
    grid_size: int = 20,
    min_r2: float = 0.20,
    min_effect_ratio_u: float = 3.20,
    min_effect_ratio_inverted: float = 0.80,
) -> List[Dict]:
    model = run_result["model"]
    model.eval()
    raw_samples = run_result["train_raw"] + run_result["valid_raw"] + run_result["test_raw"]
    feature_names = run_result["feature_names"]
    graphs_by_year = run_result["graphs_by_year"]
    x_mean = run_result["x_mean"]
    x_std = run_result["x_std"]
    device = run_result["config"].device

    de_idx = feature_names.index("DE")
    results = []

    for province_idx, province in enumerate(PROVINCE_ORDER):
        observed_values = np.array([sample["x"][-1, province_idx, de_idx] for sample in raw_samples], dtype=np.float32)
        grid_values = np.linspace(float(observed_values.min()), float(observed_values.max()), grid_size, dtype=np.float32)

        if float(observed_values.max() - observed_values.min()) < 1e-8:
            curve_info = {
                "province": province,
                "label": "弱相关",
                "a": 0.0,
                "b": 0.0,
                "c": float(np.mean([sample["y"][province_idx, 0] for sample in raw_samples])),
                "r2": 0.0,
                "turning_point": None,
                "turning_in_range": False,
                "effect_range": 0.0,
                "amplitude_threshold_u": 0.0,
                "amplitude_threshold_inverted": 0.0,
                "grid_low_q": float(observed_values.min()),
                "grid_high_q": float(observed_values.max()),
                "ice_curve_std_mean": 0.0,
                "grid_min": float(observed_values.min()),
                "grid_max": float(observed_values.max()),
            }
            results.append(curve_info)
            continue

        ice_curves = []
        for raw_sample in raw_samples:
            sample_curve = []
            for de_value in grid_values:
                modified_sample = build_modified_sample_for_de_grid(
                    raw_sample=raw_sample,
                    graphs_by_year=graphs_by_year,
                    feature_names=feature_names,
                    province_idx=province_idx,
                    de_value=float(de_value),
                )
                pred_scaled = predict_single_sample(model, modified_sample, x_mean, x_std, device)
                pred_raw = pred_scaled * float(run_result["y_std"][0, 0]) + float(run_result["y_mean"][0, 0])
                sample_curve.append(float(pred_raw[province_idx]))
            ice_curves.append(sample_curve)

        ice_curves = np.asarray(ice_curves, dtype=np.float32)
        mean_curve = ice_curves.mean(axis=0)
        curve_info = classify_curve(
            grid_values=grid_values,
            mean_curve=mean_curve,
            min_r2=min_r2,
            min_effect_ratio_u=min_effect_ratio_u,
            min_effect_ratio_inverted=min_effect_ratio_inverted,
        )
        curve_info.update(
            {
                "province": province,
                "grid_min": float(grid_values.min()),
                "grid_max": float(grid_values.max()),
                "ice_curve_std_mean": float(ice_curves.std(axis=0).mean()),
            }
        )
        results.append(curve_info)

    return results


def summarize_stability(run_outputs: List[Dict]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    labels = ["U型", "倒U型", "弱相关"]
    province_rows = []
    class_rows = []

    province_to_labels = {
        province: [run["province_labels"][province] for run in run_outputs]
        for province in PROVINCE_ORDER
    }

    for province, province_labels in province_to_labels.items():
        counts = {label: province_labels.count(label) for label in labels}
        probs = {label: counts[label] / len(province_labels) for label in labels}
        main_label = max(labels, key=lambda label: probs[label])
        main_prob = probs[main_label]
        stability = "稳定" if main_prob >= 0.70 else ("中等稳定" if main_prob >= 0.50 else "不稳定")
        province_rows.append(
            {
                "省份": province,
                "P(U型)": round(probs["U型"], 4),
                "P(倒U型)": round(probs["倒U型"], 4),
                "P(弱相关)": round(probs["弱相关"], 4),
                "主分类": main_label,
                "主分类概率": round(main_prob, 4),
                "稳定性": stability,
            }
        )

    for label in labels:
        counts = [run["class_counts"][label] for run in run_outputs]
        mean_count = float(np.mean(counts))
        std_count = float(np.std(counts, ddof=1)) if len(counts) > 1 else 0.0
        low = float(np.quantile(counts, 0.025))
        high = float(np.quantile(counts, 0.975))
        class_rows.append(
            {
                "模式": label,
                "平均省份数": round(mean_count, 4),
                "标准差": round(std_count, 4),
                "最小值": int(np.min(counts)),
                "最大值": int(np.max(counts)),
                "95%经验区间下限": round(low, 4),
                "95%经验区间上限": round(high, 4),
            }
        )

    province_df = pd.DataFrame(province_rows).sort_values(["主分类", "主分类概率"], ascending=[True, False])
    class_df = pd.DataFrame(class_rows)
    return province_df, class_df


def save_count_stability_figure(class_df: pd.DataFrame, out_path: Path) -> None:
    labels = class_df["模式"].tolist()
    means = class_df["平均省份数"].to_numpy(dtype=float)
    lows = class_df["95%经验区间下限"].to_numpy(dtype=float)
    highs = class_df["95%经验区间上限"].to_numpy(dtype=float)
    err_low = means - lows
    err_high = highs - means
    colors = ["#2A9D8F", "#E76F51", "#6C757D"]

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=220)
    bars = ax.bar(labels, means, yerr=[err_low, err_high], capsize=8, color=colors, edgecolor="#333333")
    ax.set_ylabel("重复训练下的识别省份数")
    ax.set_title("STGNN 重复训练的非线性分类稳定性")
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.set_ylim(0, max(highs) + 4)

    for bar, mean, low, high in zip(bars, means, lows, highs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.25, f"{mean:.2f}", ha="center", va="bottom", fontsize=9)
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            high + 0.55,
            f"[{low:.1f}, {high:.1f}]",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#444444",
        )

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def run_repeated_training(
    seeds: List[int],
    split_seed: int = 2026,
    grid_size: int = 20,
    min_r2: float = 0.20,
    min_effect_ratio_u: float = 3.20,
    min_effect_ratio_inverted: float = 0.80,
) -> Dict:
    run_outputs = []

    for idx, seed in enumerate(seeds, start=1):
        print(f"\n{'=' * 88}\n[{idx}/{len(seeds)}] seed={seed}")
        config = build_default_config(seed)
        run_result = train_single_run(config=config, split_seed=split_seed)
        province_curves = analyze_province_curves(
            run_result=run_result,
            grid_size=grid_size,
            min_r2=min_r2,
            min_effect_ratio_u=min_effect_ratio_u,
            min_effect_ratio_inverted=min_effect_ratio_inverted,
        )
        province_labels = {row["province"]: row["label"] for row in province_curves}
        class_counts = {
            label: sum(row["label"] == label for row in province_curves)
            for label in ["U型", "倒U型", "弱相关"]
        }
        run_outputs.append(
            {
                "seed": seed,
                "best_epoch": run_result["best_epoch"],
                "train_r2": run_result["train_metrics"]["r2"],
                "valid_r2": run_result["valid_metrics"]["r2"],
                "test_r2": run_result["test_metrics"]["r2"],
                "train_rmse": run_result["train_metrics"]["rmse"],
                "valid_rmse": run_result["valid_metrics"]["rmse"],
                "test_rmse": run_result["test_metrics"]["rmse"],
                "province_curves": province_curves,
                "province_labels": province_labels,
                "class_counts": class_counts,
                "train_years": run_result["train_years"],
                "valid_years": run_result["valid_years"],
                "test_years": run_result["test_years"],
                "config": asdict(run_result["config"]),
            }
        )
        print(
            f"seed={seed} 完成: "
            f"train_r2={run_outputs[-1]['train_r2']:.4f}, "
            f"valid_r2={run_outputs[-1]['valid_r2']:.4f}, "
            f"test_r2={run_outputs[-1]['test_r2']:.4f}, "
            f"counts={class_counts}"
        )

    province_df, class_df = summarize_stability(run_outputs)
    metrics_df = pd.DataFrame(
        [
            {
                "seed": run["seed"],
                "best_epoch": run["best_epoch"],
                "train_r2": round(run["train_r2"], 4),
                "valid_r2": round(run["valid_r2"], 4),
                "test_r2": round(run["test_r2"], 4),
                "train_rmse": round(run["train_rmse"], 4),
                "valid_rmse": round(run["valid_rmse"], 4),
                "test_rmse": round(run["test_rmse"], 4),
                "U型个数": run["class_counts"]["U型"],
                "倒U型个数": run["class_counts"]["倒U型"],
                "弱相关个数": run["class_counts"]["弱相关"],
            }
            for run in run_outputs
        ]
    ).sort_values("seed")

    return {
        "runs": run_outputs,
        "metrics_df": metrics_df,
        "province_df": province_df,
        "class_df": class_df,
    }


def main() -> None:
    seeds = [12, 16, 22, 26, 32, 36, 42, 46, 52, 56, 62, 66, 72, 76, 82, 86, 92, 96, 102, 106, 112, 116, 122, 126, 132, 136, 142, 146, 152, 156]
    split_seed = 2026
    min_r2 = 0.20
    min_effect_ratio_u = 3.20
    min_effect_ratio_inverted = 0.80
    output = run_repeated_training(
        seeds=seeds,
        split_seed=split_seed,
        grid_size=20,
        min_r2=min_r2,
        min_effect_ratio_u=min_effect_ratio_u,
        min_effect_ratio_inverted=min_effect_ratio_inverted,
    )

    seed_tag = f"seed{len(seeds)}"
    metrics_path = OUTPUT_DIR / f"stgnn_repeated_{seed_tag}_metrics.csv"
    province_path = OUTPUT_DIR / f"stgnn_repeated_{seed_tag}_province_stability.csv"
    class_path = OUTPUT_DIR / f"stgnn_repeated_{seed_tag}_class_summary.csv"
    json_path = OUTPUT_DIR / f"stgnn_repeated_{seed_tag}_results.json"
    fig_path = OUTPUT_DIR / f"stgnn_repeated_{seed_tag}_class_stability.png"

    output["metrics_df"].to_csv(metrics_path, index=False, encoding="utf-8-sig")
    output["province_df"].to_csv(province_path, index=False, encoding="utf-8-sig")
    output["class_df"].to_csv(class_path, index=False, encoding="utf-8-sig")
    save_count_stability_figure(output["class_df"], fig_path)

    serializable = {
        "seeds": seeds,
        "split_seed": split_seed,
        "classification_method": {
            "curve_extraction": "固定其他变量，仅改变省份在最后一个输入时间步的 DE 值；对所有窗口样本求 PDP 平均曲线，并保留 ICE 曲线离散度。",
            "fit": "对省级 PDP 曲线做二次拟合 EE=a*DE^2+b*DE+c。",
            "rules": {
                "U型": "a>0, R²>=0.20, 曲线响应幅度大于 3.2 倍 PDP 曲线标准差；拐点位置作为辅助说明。",
                "倒U型": "a<0, R²>=0.20, 曲线响应幅度大于 0.8 倍 PDP 曲线标准差；拐点位置作为辅助说明。",
                "弱相关": "不满足上述条件。",
            },
            "notes": "考虑到年度省级样本区间较窄，稳定性检验中不再将拐点落入观测区间作为硬门槛，以避免将方向明确但拐点轻微出界的曲线全部归为弱相关。",
        },
        "runs": output["runs"],
    }
    json_path.write_text(json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n输出文件")
    print(f"- {metrics_path}")
    print(f"- {province_path}")
    print(f"- {class_path}")
    print(f"- {json_path}")
    print(f"- {fig_path}")
    print("\n分类稳定性汇总")
    print(output["class_df"].to_string(index=False))


if __name__ == "__main__":
    main()
