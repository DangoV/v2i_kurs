from __future__ import annotations

import argparse
import json
from pathlib import Path

from v2i.generator import GenerationConfig, default_priority_params, generate_synthetic_data
from v2i.io_utils import load_from_csv_dir, save_all_formats
from v2i.ml_models import build_training_table, train_sklearn_delay_model, train_torch_delay_model
from v2i.optimization import optimize_priority_params
from v2i.simulation import compare_scenarios


def cmd_generate(args):
    data = generate_synthetic_data(GenerationConfig(seed=args.seed))
    save_all_formats(data, args.out_dir)
    print(f"Saved synthetic data to: {args.out_dir}")


def cmd_simulate(args):
    data = load_from_csv_dir(args.in_dir)
    summary = compare_scenarios(data, baseline_params=default_priority_params(), priority_params=default_priority_params())
    print(summary.to_string(index=False))


def cmd_optimize(args):
    data = load_from_csv_dir(args.in_dir)
    best = optimize_priority_params(data)
    print("Optimized priority params:")
    print(json.dumps(best, indent=2, ensure_ascii=False))


def cmd_train_ml(args):
    data = load_from_csv_dir(args.in_dir)
    train_df = build_training_table(data)

    sk_metrics = train_sklearn_delay_model(train_df)
    print("scikit-learn metrics:")
    print(json.dumps(sk_metrics, indent=2, ensure_ascii=False))

    torch_metrics = train_torch_delay_model(train_df)
    if torch_metrics is None:
        print("PyTorch not installed: skipped.")
    else:
        print("PyTorch metrics:")
        print(json.dumps(torch_metrics, indent=2, ensure_ascii=False))


def cmd_run(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = generate_synthetic_data(GenerationConfig(seed=args.seed))
    save_all_formats(data, out_dir)

    default_params = default_priority_params()
    baseline_vs_default = compare_scenarios(data, baseline_params=default_params, priority_params=default_params)

    best = optimize_priority_params(data)
    baseline_vs_opt = compare_scenarios(data, baseline_params=default_params, priority_params=best)

    train_df = build_training_table(data, seed=args.seed)
    sk_metrics = train_sklearn_delay_model(train_df)
    torch_metrics = train_torch_delay_model(train_df)

    print("=== Baseline vs default V2I ===")
    print(baseline_vs_default.to_string(index=False))
    print("\n=== Baseline vs optimized V2I ===")
    print(baseline_vs_opt.to_string(index=False))
    print("\n=== Optimized parameters ===")
    print(json.dumps(best, indent=2, ensure_ascii=False))
    print("\n=== ML metrics (scikit-learn) ===")
    print(json.dumps(sk_metrics, indent=2, ensure_ascii=False))
    print("\n=== ML metrics (PyTorch, optional) ===")
    print(json.dumps(torch_metrics, indent=2, ensure_ascii=False) if torch_metrics else "Skipped: torch unavailable")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="V2I tram priority coursework toolkit")
    sub = parser.add_subparsers(required=True)

    p_gen = sub.add_parser("generate", help="Generate synthetic dataset")
    p_gen.add_argument("--out-dir", default="data_out")
    p_gen.add_argument("--seed", type=int, default=42)
    p_gen.set_defaults(func=cmd_generate)

    p_sim = sub.add_parser("simulate", help="Run baseline vs V2I simulation")
    p_sim.add_argument("--in-dir", default="data_out")
    p_sim.set_defaults(func=cmd_simulate)

    p_opt = sub.add_parser("optimize", help="Optimize V2I priority parameters")
    p_opt.add_argument("--in-dir", default="data_out")
    p_opt.set_defaults(func=cmd_optimize)

    p_ml = sub.add_parser("train-ml", help="Train delay prediction models")
    p_ml.add_argument("--in-dir", default="data_out")
    p_ml.set_defaults(func=cmd_train_ml)

    p_run = sub.add_parser("run", help="Execute full pipeline")
    p_run.add_argument("--out-dir", default="data_out")
    p_run.add_argument("--seed", type=int, default=42)
    p_run.set_defaults(func=cmd_run)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
