#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import pickle
from typing import Optional
import pandas as pd
import numpy as np

# Normalization helpers
SPACE_COLLAPSE_RE = re.compile(r"\s+")
NBSP_RE = re.compile(r"[\u00A0\u202F]")
ZERO_WIDTH_RE = re.compile(r"[\u200B-\u200F\u202A-\u202E\u2060\ufeff]")

def normalize_name(name: str) -> str:
    """Normalize column name."""
    if not isinstance(name, str):
        name = str(name)
    # Replace NBSP with space
    name = NBSP_RE.sub(" ", name)
    # Remove zero-width chars
    name = ZERO_WIDTH_RE.sub("", name)
    # Collapse spaces
    name = SPACE_COLLAPSE_RE.sub(" ", name).strip()
    return name

def pick_header_name(l0, l1, l2) -> str:
    """Pick name: prefer l2, else l1, else l0."""
    for level in [l2, l1, l0]:
        if 'Unnamed:' in str(level):
            continue
        norm = normalize_name(level)
        if norm:
            return norm
    return ""

def build_hints_from_analysis(path: str) -> dict:
    """Build hints dict from column_analysis.txt."""
    hints_categorical_by_name = set()
    hints_numerical_by_name = set()
    if not os.path.exists(path):
        return {"categorical": hints_categorical_by_name, "numerical": hints_numerical_by_name}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Extract name from "Column \d+: Name='name', ..."
            match = re.search(r"Name='([^']*)'", line)
            if not match:
                continue
            name = match.group(1)
            norm_name = normalize_name(name).casefold()
            # Check for categorical patterns
            cat_patterns = [
                r"\(.*?\b1\s*-\s*.*?\)",
                r"\bда\s*-\s*1\b",
                r"\bгруппа\b",
                r"\bтип\b",
                r"\bкатегория\b",
                r"\bстепень\b",
                r"\bстадия\b"
            ]
            if any(re.search(p, norm_name, re.IGNORECASE) for p in cat_patterns):
                hints_categorical_by_name.add(norm_name)
                continue
            # Check for numerical patterns
            num_patterns = [
                r"\bкг\b",
                r"\bмм\s*рт\.?\s*ст\.?\b",
                r"\bмг\s*/\s*дл\b",
                r"\bг\s*/\s*л\b",
                r"\bсад\b",
                r"\bдад\b",
                r"\bкреатинин\b",
                r"\bобщий\s+белок\b",
                r"\bwbc[-\s]*лейкоцит"
            ]
            if any(re.search(p, norm_name, re.IGNORECASE) for p in num_patterns):
                hints_numerical_by_name.add(norm_name)
    return {"categorical": hints_categorical_by_name, "numerical": hints_numerical_by_name}

def classify_features(X: pd.DataFrame, hints: dict, target_name: str, identifier_name: Optional[str]) -> tuple[list[str], list[str], list[str]]:
    """Classify features."""
    categorical_features = []
    numerical_features = []
    heuristic_warnings = []
    for col in X.columns:
        if col == target_name or col == identifier_name:
            continue
        col_key = normalize_name(col).casefold()
        # Hints
        if col_key in hints["categorical"]:
            categorical_features.append(col)
            continue
        if col_key in hints["numerical"]:
            numerical_features.append(col)
            continue
        # Name cues
        name_cues = ["(да-1", "(1 -", "группа", "тип", "категория", "степень", "стадия"]
        if any(cue in col_key for cue in name_cues):
            categorical_features.append(col)
            continue
        # Heuristics
        s = X[col]
        try:
            s_num = pd.to_numeric(s, errors="coerce")
            if s_num.notna().mean() >= 0.9:
                nuniq = s_num.dropna().nunique()
                if nuniq <= 10:
                    nearest_int = np.rint(s_num.dropna())
                    if np.all(np.abs(s_num.dropna() - nearest_int) <= 1e-9):
                        categorical_features.append(col)
                        if nuniq > 5:
                            heuristic_warnings.append(f"{col}: borderline unique count {nuniq}")
                        continue
                numerical_features.append(col)
                continue
        except:
            pass
        # Object-like
        non_null = s.dropna()
        nuniq = non_null.nunique()
        if nuniq <= 10:
            categorical_features.append(col)
            if nuniq > 5:
                heuristic_warnings.append(f"{col}: borderline unique count {nuniq}")
        else:
            categorical_features.append(col)
            heuristic_warnings.append(f"{col}: high cardinality {nuniq}, treated as categorical")
    return categorical_features, numerical_features, heuristic_warnings

# Main
def main():
    excel_path = "result_no_text_filled_more_than_80_filled.xlsx"
    analysis_path = "column_analysis.txt"

    # Read Excel
    df = pd.read_excel(excel_path, header=[0,1,2], dtype=str)

    # Derive column names
    column_names = []
    for col in df.columns:
        l0, l1, l2 = col
        name = pick_header_name(l0, l1, l2)
        column_names.append(name)

    # Normalize column names
    column_names = [normalize_name(n) for n in column_names]

    # Handle duplicates
    seen = {}
    final_names = []
    for name in column_names:
        if name in seen:
            seen[name] += 1
            final_names.append(f"{name}_dup{seen[name]}")
        else:
            seen[name] = 0
            final_names.append(name)
    if any(seen[n] > 0 for n in seen):
        print("WARNING: Duplicate normalized column names disambiguated.")

    df.columns = final_names

    # Remove identifier
    identifier_raw = '№ пациента (пробирки) '
    identifier_key = normalize_name(identifier_raw).replace(" ", "").casefold()
    identifier_col = None
    for col in df.columns:
        if normalize_name(col).replace(" ", "").casefold() == identifier_key:
            identifier_col = col
            break
    if identifier_col:
        df = df.drop(columns=[identifier_col])
        print(f"Identifier column removed: {identifier_col}")
    else:
        print("WARNING: Identifier column not found.")

    # Find y
    y_prefix = "Группа наблюдения"
    y_key = normalize_name(y_prefix).casefold()
    y_col = None
    for col in df.columns:
        if normalize_name(col).casefold().startswith(y_key):
            y_col = col
            break
    if not y_col:
        raise RuntimeError("Target column not found.")
    y = df[y_col]
    X = df.drop(columns=[y_col])

    # Load hints
    hints = build_hints_from_analysis(analysis_path)

    # Classify
    categorical_features, numerical_features, warnings = classify_features(X, hints, y_col, identifier_col)

    # Save pickle
    data = {
        "X": X,
        "y": y,
        "categorical_features": categorical_features,
        "numerical_features": numerical_features
    }
    with open("prepared_data.pkl", "wb") as f:
        pickle.dump(data, f)

    # Print summary
    print(f"Rows: {len(X)}")
    print(f"Features in X: {X.shape[1]}")
    print(f"categorical_features: {len(categorical_features)}")
    print(f"numerical_features: {len(numerical_features)}")
    print(f"Target column name: {y.name}")
    if warnings:
        print("Warnings about heuristic classifications:")
        for w in warnings:
            print(f" - {w}")

if __name__ == "__main__":
    main()