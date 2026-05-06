#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def mean_f1(path: Path) -> float:
    df = pd.read_csv(path)
    return float(df["f1"].mean()) if "f1" in df and len(df) else float("nan")


def latex_table(df: pd.DataFrame, caption: str) -> str:
    return df.to_latex(index=True, float_format="%.3f", escape=False, caption=caption)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    args = ap.parse_args()
    root = Path(args.results_dir)
    tables = root / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    runs = ["of_v2", "sofake_v2", "cf_pretrained", "genimage_swinv2", "semitruths_swinv2"]
    matrix = {}
    for run in runs:
        row = {}
        for csv in (root / run).glob("*.csv"):
            row[csv.stem] = mean_f1(csv)
        matrix[run] = row
    cross = pd.DataFrame.from_dict(matrix, orient="index").sort_index(axis=1)
    (tables / "cross_eval_matrix.tex").write_text(latex_table(cross, "Cross-dataset F1 matrix."))

    pergen_files = list((root / "of_v2").glob("of_test*.csv")) + list((root / "of_v2").glob("reddit.csv"))
    pergen = pd.concat([pd.read_csv(p).assign(test_set=p.stem) for p in pergen_files], ignore_index=True) if pergen_files else pd.DataFrame()
    if not pergen.empty:
        (tables / "per_generator_of_v2.tex").write_text(latex_table(pergen.pivot_table(index=["test_set", "model"], values="f1"), "OpenFake v2 per-generator F1."))
    else:
        (tables / "per_generator_of_v2.tex").write_text("% No per-generator results found yet.\n")

    comp = list((root / "compression").glob("*.csv"))
    if comp:
        cdf = pd.concat([pd.read_csv(p).assign(source=p.stem) for p in comp], ignore_index=True)
        (tables / "compression_sweep.tex").write_text(latex_table(cdf.pivot_table(index=["source", "quality", "model"], values="f1"), "Compression robustness F1."))
    else:
        (tables / "compression_sweep.tex").write_text("% No compression results found yet.\n")

    (root / "SUMMARY.md").write_text(f"# OpenFake Results Summary\n\nGenerated tables in `{tables}`.\n\n{cross.to_markdown()}\n")


if __name__ == "__main__":
    main()
