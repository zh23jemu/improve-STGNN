import json
from pathlib import Path

from train_stgnn_excel import Config, run_training


DATA_DIR = Path(__file__).resolve().parent


def build_candidates():
    base = dict(
        use_target_history=True,
        use_spatial_lag_features=True,
        max_epochs=350,
        batch_size=2,
        early_stop_patience=40,
        scheduler_patience=15,
        scheduler_factor=0.5,
        scheduler_min_lr=1e-5,
    )
    candidates = [
        ("baseline_v2", Config(**base, geo_weight=0.40, econ_weight=0.35, digital_weight=0.25, adaptive_weight=0.05, seq_len=3, encoder_hidden=32, hidden_dim=64, gru_layers=1, dropout=0.10, loss_name="mse", learning_rate=5e-4, weight_decay=1e-6, seed=42)),
        ("low_reg_mse", Config(**base, geo_weight=0.40, econ_weight=0.35, digital_weight=0.25, adaptive_weight=0.02, seq_len=3, encoder_hidden=32, hidden_dim=64, gru_layers=1, dropout=0.05, loss_name="mse", learning_rate=8e-4, weight_decay=1e-6, seed=42)),
        ("equal_weight_mse", Config(**base, geo_weight=0.33, econ_weight=0.33, digital_weight=0.34, adaptive_weight=0.02, seq_len=3, encoder_hidden=32, hidden_dim=64, gru_layers=1, dropout=0.05, loss_name="mse", learning_rate=8e-4, weight_decay=1e-6, seed=42)),
        ("digital_high", Config(**base, geo_weight=0.25, econ_weight=0.25, digital_weight=0.50, adaptive_weight=0.02, seq_len=3, encoder_hidden=32, hidden_dim=64, gru_layers=1, dropout=0.05, loss_name="mse", learning_rate=8e-4, weight_decay=1e-6, seed=42)),
        ("geo_high", Config(**base, geo_weight=0.50, econ_weight=0.30, digital_weight=0.20, adaptive_weight=0.02, seq_len=3, encoder_hidden=32, hidden_dim=64, gru_layers=1, dropout=0.05, loss_name="mse", learning_rate=8e-4, weight_decay=1e-6, seed=42)),
        ("seq2_compact", Config(**base, geo_weight=0.40, econ_weight=0.35, digital_weight=0.25, adaptive_weight=0.02, seq_len=2, encoder_hidden=32, hidden_dim=64, gru_layers=1, dropout=0.05, loss_name="mse", learning_rate=8e-4, weight_decay=1e-6, seed=42)),
        ("seq4_compact", Config(**base, geo_weight=0.40, econ_weight=0.35, digital_weight=0.25, adaptive_weight=0.02, seq_len=4, encoder_hidden=32, hidden_dim=64, gru_layers=1, dropout=0.05, loss_name="mse", learning_rate=8e-4, weight_decay=1e-6, seed=42)),
        ("hidden96", Config(**base, geo_weight=0.40, econ_weight=0.35, digital_weight=0.25, adaptive_weight=0.02, seq_len=3, encoder_hidden=48, hidden_dim=96, gru_layers=1, dropout=0.05, loss_name="mse", learning_rate=5e-4, weight_decay=1e-6, seed=42)),
        ("gru2_mild", Config(**base, geo_weight=0.40, econ_weight=0.35, digital_weight=0.25, adaptive_weight=0.02, seq_len=3, encoder_hidden=32, hidden_dim=64, gru_layers=2, dropout=0.10, loss_name="mse", learning_rate=5e-4, weight_decay=1e-6, seed=42)),
        ("huber_seq3", Config(**base, geo_weight=0.40, econ_weight=0.35, digital_weight=0.25, adaptive_weight=0.02, seq_len=3, encoder_hidden=32, hidden_dim=64, gru_layers=1, dropout=0.05, loss_name="huber", learning_rate=8e-4, weight_decay=1e-6, seed=42)),
        ("no_adaptive", Config(**base, geo_weight=0.40, econ_weight=0.35, digital_weight=0.25, adaptive_weight=0.00, seq_len=3, encoder_hidden=32, hidden_dim=64, gru_layers=1, dropout=0.05, loss_name="mse", learning_rate=8e-4, weight_decay=1e-6, seed=42)),
        ("seed52_bestlike", Config(**base, geo_weight=0.40, econ_weight=0.35, digital_weight=0.25, adaptive_weight=0.02, seq_len=3, encoder_hidden=32, hidden_dim=64, gru_layers=1, dropout=0.05, loss_name="mse", learning_rate=8e-4, weight_decay=1e-6, seed=52)),
        ("seed62_bestlike", Config(**base, geo_weight=0.40, econ_weight=0.35, digital_weight=0.25, adaptive_weight=0.02, seq_len=3, encoder_hidden=32, hidden_dim=64, gru_layers=1, dropout=0.05, loss_name="mse", learning_rate=8e-4, weight_decay=1e-6, seed=62)),
        ("seed72_bestlike", Config(**base, geo_weight=0.40, econ_weight=0.35, digital_weight=0.25, adaptive_weight=0.02, seq_len=3, encoder_hidden=32, hidden_dim=64, gru_layers=1, dropout=0.05, loss_name="mse", learning_rate=8e-4, weight_decay=1e-6, seed=72)),
    ]
    return candidates


def main():
    results = []
    for name, config in build_candidates():
        print(f"\n{'=' * 80}\n{name}")
        summary = run_training(config)
        row = {
            "name": name,
            "best_epoch": summary["best_epoch"],
            "train_r2": round(summary["train_metrics"]["r2"], 4),
            "valid_r2": round(summary["valid_metrics"]["r2"], 4),
            "test_r2": round(summary["test_metrics"]["r2"], 4),
            "train_rmse": round(summary["train_metrics"]["rmse"], 4),
            "valid_rmse": round(summary["valid_metrics"]["rmse"], 4),
            "test_rmse": round(summary["test_metrics"]["rmse"], 4),
            "config": summary["config"],
        }
        print(row)
        results.append(row)

    ranked = sorted(results, key=lambda x: (x["valid_r2"], x["test_r2"]), reverse=True)
    out_json = DATA_DIR / "stgnn_tuning_results.json"
    out_md = DATA_DIR / "stgnn_tuning_results.md"
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
    print("\nTop 5:")
    for row in ranked[:5]:
        print(row)


if __name__ == "__main__":
    main()
