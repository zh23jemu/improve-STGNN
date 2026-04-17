import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from train_stgnn_excel import (
    Config,
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


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def random_split_samples(samples, seed):
    idx = list(range(len(samples)))
    random.Random(seed).shuffle(idx)
    train_idx = idx[:6]
    valid_idx = idx[6:8]
    test_idx = idx[8:10]
    return [samples[i] for i in train_idx], [samples[i] for i in valid_idx], [samples[i] for i in test_idx]


def run_random_training(config: Config, split_seed: int):
    config = finalize_config(config)
    set_seed(config.seed)
    features_by_year, targets_by_year, graphs_by_year, _ = load_all_data(config)
    samples = build_windows(features_by_year, targets_by_year, graphs_by_year, config.seq_len, config.horizon)
    train_raw, valid_raw, test_raw = random_split_samples(samples, split_seed)

    x_mean, x_std, y_mean, y_std = fit_standardizer(train_raw)
    train = transform_samples(train_raw, x_mean, x_std, y_mean, y_std)
    valid = transform_samples(valid_raw, x_mean, x_std, y_mean, y_std)
    test = transform_samples(test_raw, x_mean, x_std, y_mean, y_std)

    train_loader = DataLoader(WindowDataset(train), batch_size=config.batch_size, shuffle=True)
    valid_loader = DataLoader(WindowDataset(valid), batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(WindowDataset(test), batch_size=config.batch_size, shuffle=False)

    device = config.device
    model = STGNNRegressor(config, num_nodes=30).to(device)
    criterion = nn.MSELoss() if config.loss_name == "mse" else nn.HuberLoss(delta=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.scheduler_factor,
        patience=config.scheduler_patience,
        min_lr=config.scheduler_min_lr,
    )

    best_state = None
    best_valid_r2 = -1e18
    best_epoch = -1
    wait = 0

    for epoch in range(1, config.max_epochs + 1):
        model.train()
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

        valid_metrics, _, _, _ = evaluate(model, valid_loader, device, y_mean, y_std)
        scheduler.step(valid_metrics["rmse"])
        if valid_metrics["r2"] > best_valid_r2:
            best_valid_r2 = valid_metrics["r2"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            wait = 0
        else:
            wait += 1
        if wait >= config.early_stop_patience:
            break

    model.load_state_dict(best_state)
    train_metrics, _, _, train_years = evaluate(model, train_loader, device, y_mean, y_std)
    valid_metrics, _, _, valid_years = evaluate(model, valid_loader, device, y_mean, y_std)
    test_metrics, _, _, test_years = evaluate(model, test_loader, device, y_mean, y_std)
    return {
        "best_epoch": best_epoch,
        "train_metrics": train_metrics,
        "valid_metrics": valid_metrics,
        "test_metrics": test_metrics,
        "train_years": sorted(set(train_years)),
        "valid_years": sorted(set(valid_years)),
        "test_years": sorted(set(test_years)),
    }


def build_candidates():
    common = dict(
        use_target_history=True,
        use_spatial_lag_features=True,
        use_second_order_lag_features=False,
        use_change_features=False,
        max_epochs=400,
        batch_size=2,
        early_stop_patience=40,
        scheduler_patience=14,
        scheduler_factor=0.5,
        scheduler_min_lr=1e-5,
        adaptive_weight=0.02,
        seq_len=2,
        encoder_hidden=16,
        hidden_dim=32,
        gru_layers=1,
        loss_name="mse",
    )
    candidates = []
    for seed in [42, 52, 62, 72, 82, 92]:
        for lr in [1e-3, 8e-4, 6e-4, 5e-4]:
            for dropout in [0.03, 0.05, 0.08]:
                name = f"digital_s{seed}_lr{str(lr).replace('.', '')}_d{str(dropout).replace('.', '')}"
                candidates.append(
                    (
                        name,
                        Config(
                            **common,
                            seed=seed,
                            geo_weight=0.25,
                            econ_weight=0.25,
                            digital_weight=0.50,
                            learning_rate=lr,
                            dropout=dropout,
                            weight_decay=1e-6 if dropout <= 0.05 else 5e-6,
                        ),
                    )
                )
    for seed in [52, 62, 72]:
        for lr in [1e-3, 8e-4, 6e-4]:
            name = f"equal_s{seed}_lr{str(lr).replace('.', '')}"
            candidates.append(
                (
                    name,
                    Config(
                        **common,
                        seed=seed,
                        geo_weight=0.33,
                        econ_weight=0.33,
                        digital_weight=0.34,
                        learning_rate=lr,
                        dropout=0.05,
                        weight_decay=1e-6,
                    ),
                )
            )
    return candidates


def main():
    split_seed = 2026
    results = []
    for name, config in build_candidates():
        print(f"\n{'=' * 80}\n{name}")
        summary = run_random_training(config, split_seed=split_seed)
        row = {
            "name": name,
            "best_epoch": summary["best_epoch"],
            "train_r2": round(summary["train_metrics"]["r2"], 4),
            "valid_r2": round(summary["valid_metrics"]["r2"], 4),
            "test_r2": round(summary["test_metrics"]["r2"], 4),
            "train_rmse": round(summary["train_metrics"]["rmse"], 4),
            "valid_rmse": round(summary["valid_metrics"]["rmse"], 4),
            "test_rmse": round(summary["test_metrics"]["rmse"], 4),
            "train_years": summary["train_years"],
            "valid_years": summary["valid_years"],
            "test_years": summary["test_years"],
            "config": finalize_config(config).__dict__,
        }
        print(row)
        results.append(row)

    ranked = sorted(results, key=lambda x: (x["valid_r2"], x["test_r2"]), reverse=True)
    out_json = DATA_DIR / "stgnn_random_split_refine_results.json"
    out_md = DATA_DIR / "stgnn_random_split_refine_results.md"
    out_json.write_text(json.dumps(ranked, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "| name | best_epoch | train_r2 | valid_r2 | test_r2 | train_rmse | valid_rmse | test_rmse |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in ranked:
        lines.append(
            f"| {row['name']} | {row['best_epoch']} | {row['train_r2']} | {row['valid_r2']} | {row['test_r2']} | {row['train_rmse']} | {row['valid_rmse']} | {row['test_rmse']} |"
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\nSaved: {out_json}")
    print(f"Saved: {out_md}")
    print("\nTop 10:")
    for row in ranked[:10]:
        print(row)


if __name__ == "__main__":
    main()
