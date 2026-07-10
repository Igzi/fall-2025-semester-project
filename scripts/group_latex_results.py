#!/usr/bin/env python3
"""Group LaTeX table columns by common prefix and summarize 3 runs with mean ± std."""

from __future__ import annotations

import argparse
import re
import statistics
from pathlib import Path
from typing import Dict, List, Tuple


def parse_latex_table(path: Path) -> Tuple[List[str], List[List[str]]]:
    text = path.read_text(encoding="utf-8")
    match = re.search(r"\\begin\{tabular\}\{.*?\}(.*?)\\end\{tabular\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"Could not find a LaTeX tabular block in {path}")

    rows: List[List[str]] = []
    for raw_line in match.group(1).splitlines():
        line = raw_line.strip()
        if not line or line.startswith("\\toprule") or line.startswith("\\midrule") or line.startswith("\\bottomrule"):
            continue
        if line.endswith("\\\\"):
            line = line[:-2].rstrip()
        if not line:
            continue
        parts = [p.strip() for p in line.split("&")]
        if parts:
            rows.append(parts)

    if not rows:
        raise ValueError("No rows found in the LaTeX table")

    header = [clean_cell(cell) for cell in rows[0]]
    body = [[clean_cell(cell) for cell in row] for row in rows[1:]]
    return header, body


def clean_cell(cell: str) -> str:
    cell = cell.strip()
    cell = cell.replace("\\texttt{", "").replace("}", "")
    cell = cell.replace("\\_", "_")
    return cell


def parse_number(cell: str) -> float:
    cell = cell.strip()
    cell = re.sub(r"[^0-9.+\-eE]", "", cell)
    if not cell:
        raise ValueError(f"Could not parse numeric value from '{cell}'")
    return float(cell)


def group_columns(header: List[str]) -> List[str]:
    groups: List[str] = []
    seen = set()
    for col in header[1:]:
        prefix = re.sub(r"_(\d+)\.json$", "", col)
        if prefix not in seen:
            groups.append(prefix)
            seen.add(prefix)
    return groups


def summarize_rows(header: List[str], body: List[List[str]]) -> Tuple[List[str], List[List[str]]]:
    groups = group_columns(header)
    data_by_group: Dict[str, Dict[int, List[float]]] = {group: {} for group in groups}

    for col_name in header[1:]:
        prefix = re.sub(r"_(\d+)\.json$", "", col_name)
        match = re.search(r"_(\d+)\.json$", col_name)
        if not match:
            continue
        run_idx = int(match.group(1))
        for row_idx, row in enumerate(body):
            if len(row) <= header.index(col_name):
                continue
            value = parse_number(row[header.index(col_name)])
            data_by_group[prefix].setdefault(run_idx, []).append(value)

    # Rebuild as row-wise values for each group
    grouped_rows: List[List[str]] = []
    for row in body:
        row_out = [row[0]]
        for group in groups:
            values = []
            for run_idx in sorted(data_by_group[group]):
                values.append(data_by_group[group][run_idx][body.index(row)])
            if not values:
                row_out.append("-")
                continue
            mean_value = statistics.mean(values)
            std_value = statistics.stdev(values) if len(values) > 1 else 0.0
            row_out.append(f"{mean_value:.2f} $\\pm$ {std_value:.2f}")
        grouped_rows.append(row_out)

    return ["Domain-Metric", *groups], grouped_rows


def write_latex_table(output_path: Path, header: List[str], rows: List[List[str]]) -> None:
    col_spec = "l" + "r" * (len(header) - 1)
    lines = [f"\\begin{{tabular}}{{{col_spec}}}", "\\toprule"]
    lines.append(" & ".join(escape_header(col) for col in header) + " \\")
    lines.append("\\midrule")
    for row in rows:
        line = " & ".join(cell for cell in row) + " \\\\"
        lines.append(line)
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def escape_header(cell: str) -> str:
    if cell == "Domain-Metric":
        return cell
    return cell.replace("_", "\\_")


def main() -> None:
    parser = argparse.ArgumentParser(description="Group LaTeX table columns by common prefix and summarize 3 runs")
    parser.add_argument("--input", default="summary_table.tex", help="Path to the input LaTeX table")
    parser.add_argument("--output", default="summary_table_grouped.tex", help="Path to the output LaTeX table")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    header, body = parse_latex_table(input_path)
    new_header, grouped_rows = summarize_rows(header, body)
    write_latex_table(output_path, new_header, grouped_rows)
    print(f"Wrote grouped table to {output_path}")


if __name__ == "__main__":
    main()
