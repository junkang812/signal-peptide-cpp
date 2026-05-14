#!/usr/bin/env python3
"""
Preprocess SignalP 6.0 three-line FASTA for CPP-based signal peptide analysis.

This script only parses the raw FASTA and creates pre-CS / post-CS regions.
Experimental filtering, train/test split, and balancing should be done in notebooks/scripts.

Input format per record:
    >UniProtID|KINGDOM|SP_TYPE|INDEX
    AMINO_ACID_SEQUENCE
    ANNOTATION_STRING

Output:
    data/processed/dataset_parsed.csv
    data/processed/preprocessing_summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

SP_ANNOTATION_CHARS = {"S", "T", "L", "P"}
DEFAULT_NONSP_PRE_LEN = 24
DEFAULT_POST_LEN = 15


def parse_header(header: str) -> dict[str, str]:
    header = header.lstrip(">").strip()
    parts = header.split("|")

    return {
        "entry": parts[0] if len(parts) > 0 else header,
        "kingdom": parts[1] if len(parts) > 1 else "UNKNOWN",
        "sp_type": parts[2] if len(parts) > 2 else "NO_SP",
        "record_index": parts[3] if len(parts) > 3 else "",
    }


def read_signalp_fasta(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input FASTA not found: {path.resolve()}")

    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]

    if len(lines) % 3 != 0:
        raise ValueError(
            f"Expected SignalP 6.0 three-line FASTA format, but got {len(lines)} non-empty lines."
        )

    records = []
    for i in range(0, len(lines), 3):
        header = lines[i]
        sequence = lines[i + 1]
        annotation = lines[i + 2]

        if not header.startswith(">"):
            raise ValueError(f"Invalid FASTA header at line {i + 1}: {header}")

        if len(sequence) != len(annotation):
            raise ValueError(
                f"Sequence/annotation length mismatch for {header}: "
                f"{len(sequence)} vs {len(annotation)}"
            )

        meta = parse_header(header)
        records.append({**meta, "sequence": sequence, "annotation": annotation})

    df = pd.DataFrame(records)
    return df.loc[:, ~df.columns.duplicated()].copy()


def find_cs_position(annotation: str) -> int | None:
    """
    Return 1-based cleavage-site position for SP records.
    Defined as the last residue annotated with S/T/L/P.
    Return None for non-SP records.
    """
    idx = max(
        (i for i, ch in enumerate(annotation) if ch in SP_ANNOTATION_CHARS),
        default=-1,
    )
    return idx + 1 if idx >= 0 else None


def split_record(row: pd.Series, nonsp_pre_len: int, post_len: int) -> pd.Series:
    sequence = row["sequence"]
    cs_position = find_cs_position(row["annotation"])

    if cs_position is None:
        label_binary = 0
        cs_position = nonsp_pre_len
        pre_cs = sequence[:nonsp_pre_len]
        post_cs = sequence[nonsp_pre_len : nonsp_pre_len + post_len]
        sp_type = "NO_SP"
    else:
        label_binary = 1
        pre_cs = sequence[:cs_position]
        post_cs = sequence[cs_position : cs_position + post_len]
        sp_type = row["sp_type"]

    return pd.Series(
        {
            "label_binary": label_binary,
            "sp_type": sp_type,
            "cs_position": int(cs_position),
            "pre_cs": pre_cs,
            "post_cs": post_cs,
            "pre_cs_len": len(pre_cs),
            "post_cs_len": len(post_cs),
        }
    )


def preprocess(
    input_fasta: Path,
    output_dir: Path,
    nonsp_pre_len: int = DEFAULT_NONSP_PRE_LEN,
    post_len: int = DEFAULT_POST_LEN,
) -> tuple[pd.DataFrame, dict]:
    output_dir.mkdir(parents=True, exist_ok=True)

    df = read_signalp_fasta(input_fasta)

    original_sp_type = df["sp_type"].copy()

    split_cols = df.apply(
        split_record,
        axis=1,
        nonsp_pre_len=nonsp_pre_len,
        post_len=post_len,
    )

    # Avoid duplicate sp_type column by replacing it with the cleaned one from split_cols.
    df = df.drop(columns=["sp_type"])
    df = pd.concat([df, split_cols], axis=1)
    df = df.loc[:, ~df.columns.duplicated()].copy()

    ordered_cols = [
        "entry",
        "kingdom",
        "sp_type",
        "record_index",
        "sequence",
        "annotation",
        "label_binary",
        "cs_position",
        "pre_cs",
        "post_cs",
        "pre_cs_len",
        "post_cs_len",
    ]

    df = df[ordered_cols].copy()

    output_path = output_dir / "dataset_parsed.csv"
    summary_path = output_dir / "preprocessing_summary.json"

    df.to_csv(output_path, index=False)

    summary = {
        "input_fasta": str(input_fasta),
        "output_dataset": str(output_path),
        "nonsp_pre_len": nonsp_pre_len,
        "post_len": post_len,
        "total_records": int(len(df)),
        "sp_records": int((df["label_binary"] == 1).sum()),
        "nonsp_records": int((df["label_binary"] == 0).sum()),
        "sp_type_counts": df["sp_type"].value_counts().to_dict(),
        "kingdom_counts": df["kingdom"].value_counts().to_dict(),
        "pre_cs_len_min": int(df["pre_cs_len"].min()),
        "pre_cs_len_mean": float(df["pre_cs_len"].mean()),
        "pre_cs_len_max": int(df["pre_cs_len"].max()),
        "post_cs_len_min": int(df["post_cs_len"].min()),
        "post_cs_len_mean": float(df["post_cs_len"].mean()),
        "post_cs_len_max": int(df["post_cs_len"].max()),
        "note": (
            "No experimental filtering or balancing was applied here. "
            "Filtering and train/test balancing should be performed in downstream notebooks/scripts."
        ),
    }

    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return df, summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Parse SignalP 6.0 dataset into a CPP-ready tabular dataset."
    )
    parser.add_argument("--input", type=Path, default=Path("data/raw/train_set.fasta"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--nonsp-pre-len", type=int, default=DEFAULT_NONSP_PRE_LEN)
    parser.add_argument("--post-len", type=int, default=DEFAULT_POST_LEN)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    _, summary = preprocess(
        input_fasta=args.input,
        output_dir=args.output_dir,
        nonsp_pre_len=args.nonsp_pre_len,
        post_len=args.post_len,
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
