from __future__ import annotations

import argparse
import json
from pathlib import Path

from v2i.generator import GenerationConfig, default_priority_params, generate_synthetic_data
from v2i.io_utils import load_from_csv_dir, save_all_formats
from v2i.ml_models import build_training_table, train_sklearn_delay_model, train_torch_delay_model
from v2i.optimization import optimize_priority_params
from v2i.simulation import compare_scenarios, sensitivity_analysis


def cmd_generate(args):
    data = generate_synthetic_data(GenerationConfig(seed=args.seed, headway_sec=args.headway_sec))
    save_all_formats(data, args.out_dir)
    print(f"Данные сохранены в {args.out_dir}")


def cmd_simulate(args):
    data = load_from_csv_dir(args.in_dir)
    summary = compare_scenarios(data, default_priority_params())
    print(summary.to_string(index=False))


def cmd_optimize(args):
    data = load_from_csv_dir(args.in_dir)
    best = optimize_priority_params(data)
    print(json.dumps(best, indent=2, ensure_ascii=False))


def cmd_sensitivity(args):
    data = load_from_csv_dir(args.in_dir)
    table = sensitivity_analysis(data, default_priority_params())
    print(table.to_string(index=False))


def cmd_train_ml(args):
    data = load_from_csv_dir(args.in_dir)
    df = build_training_table(data)
    print("sklearn:")
    print(json.dumps(train_sklearn_delay_model(df), indent=2, ensure_ascii=False))
    torch_metrics = train_torch_delay_model(df)
    print("torch:")
    print(json.dumps(torch_metrics, indent=2, ensure_ascii=False) if torch_metrics else "Skipped")


def cmd_run(args):
    out_dir = Path(args.out_dir)
    data = generate_synthetic_data(GenerationConfig(seed=args.seed, headway_sec=args.headway_sec))
    save_all_formats(data, out_dir)

    default_params = default_priority_params()
    baseline = compare_scenarios(data, default_params)
    best = optimize_priority_params(data)
    optimized = compare_scenarios(data, best)
    sens = sensitivity_analysis(data, best)

    train_df = build_training_table(data, seed=args.seed)
    sk = train_sklearn_delay_model(train_df)
    torch_m = train_torch_delay_model(train_df)

    print("=== Сценарии (default) ===")
    print(baseline.to_string(index=False))
    print("\n=== Сценарии (optimized) ===")
    print(optimized.to_string(index=False))
    print("\n=== Оптимальные параметры ===")
    print(json.dumps(best, ensure_ascii=False, indent=2))
    print("\n=== Анализ чувствительности ===")
    print(sens.to_string(index=False))
    print("\n=== ML метрики ===")
    print(json.dumps(sk, ensure_ascii=False, indent=2))
    print(json.dumps(torch_m, ensure_ascii=False, indent=2) if torch_m else "Torch недоступен")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="V2I coursework toolkit")
    sub = p.add_subparsers(required=True)

    g = sub.add_parser("generate")
    g.add_argument("--out-dir", default="data_out")
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--headway-sec", type=int, default=600)
    g.set_defaults(func=cmd_generate)

    s = sub.add_parser("simulate")
    s.add_argument("--in-dir", default="data_out")
    s.set_defaults(func=cmd_simulate)

    o = sub.add_parser("optimize")
    o.add_argument("--in-dir", default="data_out")
    o.set_defaults(func=cmd_optimize)

    sn = sub.add_parser("sensitivity")
    sn.add_argument("--in-dir", default="data_out")
    sn.set_defaults(func=cmd_sensitivity)

    ml = sub.add_parser("train-ml")
    ml.add_argument("--in-dir", default="data_out")
    ml.set_defaults(func=cmd_train_ml)

    r = sub.add_parser("run")
    r.add_argument("--out-dir", default="data_out")
    r.add_argument("--seed", type=int, default=42)
    r.add_argument("--headway-sec", type=int, default=600)
    r.set_defaults(func=cmd_run)
    return p


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
