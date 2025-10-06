#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script 1 — 01_prepare_data.py

Purpose:
- Prepare data by deriving column names from the first three rows (primarily the 3rd row),
  normalizing headers, separating target y based on a normalized prefix, and classifying features.
- No model building. No transformations of values (beyond header normalization).
- Save audit and output files (X/y in CSV and Parquet; preview and columns JSON).
- Additionally:
  - Auto-drop the known identifier column by normalized match, regardless of CLI args.
  - Read column_analysis.txt to build hints for feature classification.
  - Classify features into categorical_features and numerical_features using hints, name cues, and data heuristics.
  - Save prepared_data.pkl containing X, y, categorical_features, numerical_features in the working directory.
  - Handle duplicate normalized column names by appending _dup1, _dup2, … to subsequent duplicates.

Key rules implemented:
- Read Excel with header=None and dtype=str to preserve text and the first three rows exactly.
- Column name per column: prefer row index 2 (3rd row), else 1, else 0, then normalize.
- Normalization for matching and final names:
  - Strip whitespace, replace non-breaking spaces with spaces, collapse spaces.
  - Remove zero-width/invisible characters.
  - Standardize quotes to regular double quotes (").
  - Normalize dash variants to a hyphen ("-").
- Target detection: startswith normalized prefix (default "Группа наблюдения"),
  robust to extended descriptions and spacing/dash/quote variants.
  If multiple matches within a scope, pick the first after sorting by normalized name and warn.
  If not found in derived names, also scan all three header rows.
- Identifier removal:
  - Optional identifier column via --id-column (matched after normalization). Warn if multiple, drop first.
  - Always attempt to find and drop a specific identifier column by normalized equality (ignoring spaces/NBSP):
    raw canonical string '№ пациента (пробирки) ' (note trailing space in raw). Warn if not found.
- Data values start at 4th row (row index 3). Keep rows as-is. Reset index.
- Outputs:
  - prepared/columns_normalized.json (list of final column names, UTF-8)
  - prepared/header_rows_preview.csv (first three rows after normalization, CSV)
  - prepared/X.csv, prepared/y.csv (UTF-8)
  - prepared/X.parquet, prepared/y.parquet (if parquet engine available; otherwise warn)
  - prepared_data.pkl (in working directory) with raw X, y and feature lists (no OHE/scaling/imputation)

CLI:
  python3 01_prepare_data.py
    --input result_no_text_filled_more_than_80_filled.xlsx
    --output-dir prepared
    --id-column "№ пациента (пробирки)"
    --target-prefix "Группа наблюдения"
"""

import argparse
import json
import os
import re
import sys
import pickle
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np  # noqa: F401  (allowed dependency; not required explicitly)
import pandas as pd


# ----------------------------
# Normalization helpers
# ----------------------------

# Regexes for normalization
SPACE_COLLAPSE_RE = re.compile(r"\s+")
NBSP_RE = re.compile(r"[\u00A0\u202F]")  # NBSP and narrow NBSP
ZERO_WIDTH_RE = re.compile(r"[\u200B-\u200F\u202A-\u202E\u2060\ufeff]")  # zero-widths and directionals

# Translation maps for quotes and dashes
QUOTES_TRANSLATION = {
    ord("\u201C"): '"',  # left double
    ord("\u201D"): '"',  # right double
    ord("\u201E"): '"',  # double low-9
    ord("\u201F"): '"',  # double high-reversed-9
    ord("\u00AB"): '"',  # left guillemet
    ord("\u00BB"): '"',  # right guillemet
    ord("\u2018"): '"',  # left single
    ord("\u2019"): '"',  # right single
    ord("\u201A"): '"',  # single low-9
    ord("\u201B"): '"',  # single high-reversed-9
    ord("\u2039"): '"',  # single left-pointing angle
    ord("\u203A"): '"',  # single right-pointing angle
    ord("'"): '"',       # ASCII single -> standardize to double
}
DASHES_TRANSLATION = {
    ord("\u2010"): "-",  # hyphen
    ord("\u2011"): "-",  # non-breaking hyphen
    ord("\u2012"): "-",  # figure dash
    ord("\u2013"): "-",  # en dash
    ord("\u2014"): "-",  # em dash
    ord("\u2015"): "-",  # horizontal bar
    ord("\u2212"): "-",  # minus sign
    ord("\u2043"): "-",  # hyphen bullet
    ord("–"): "-",       # en dash literal
    ord("—"): "-",       # em dash literal
}


def to_str_or_empty(val: object) -> str:
    """Convert cell value to string for normalization; NaN/None -> empty string."""
    if val is None:
        return ""
    try:
        if pd.isna(val):  # type: ignore[arg-type]
            return ""
    except Exception:
        pass
    return str(val)


def normalize_text(s: object) -> str:
    """
    Robust normalization for header matching and final names:
    - Replace typographic quotes with standard double quote (")
    - Unify all dash variants to "-"
    - Replace NBSPs with normal spaces
    - Remove zero-width/invisible control characters
    - Collapse whitespace runs to a single space
    - Strip
    Keep original Unicode letters; do not transliterate; preserve case.
    """
    txt = to_str_or_empty(s)

    # Standardize quotes and dashes first
    txt = txt.translate(QUOTES_TRANSLATION)
    txt = txt.translate(DASHES_TRANSLATION)

    # Replace non-breaking spaces with regular spaces
    txt = NBSP_RE.sub(" ", txt)

    # Remove zero-width/invisible chars
    txt = ZERO_WIDTH_RE.sub("", txt)

    # Normalize whitespace
    txt = SPACE_COLLAPSE_RE.sub(" ", txt).strip()

    return txt


def normalize_key_for_match(s: object) -> str:
    """Normalization used for case-insensitive matching and prefix checks."""
    return normalize_text(s).casefold()


def normalize_key_for_id_match(s: object) -> str:
    """
    Stricter key for ID equality matching:
    - Apply normalize_key_for_match
    - Remove all spaces (and NBSP already converted to spaces)
    This implements 'ignoring spaces and NBSP' semantics.
    """
    return normalize_key_for_match(s).replace(" ", "")


# ----------------------------
# Header utilities
# ----------------------------

def derive_column_names_from_three_rows(header_rows_norm: pd.DataFrame) -> Tuple[List[str], List[int]]:
    """
    For each column:
      name = header_rows_norm.iloc[2, col] if not empty else header_rows_norm.iloc[1, col] else header_rows_norm.iloc[0, col]
    Returns:
      - list of chosen names (already normalized)
      - list of source row indices used per column (2, 1, or 0)
    """
    n_cols = header_rows_norm.shape[1]
    names: List[str] = []
    used_rows: List[int] = []
    for c in range(n_cols):
        candidates = [
            (2, to_str_or_empty(header_rows_norm.iat[2, c])),
            (1, to_str_or_empty(header_rows_norm.iat[1, c])),
            (0, to_str_or_empty(header_rows_norm.iat[0, c])),
        ]
        chosen = ""
        chosen_row = 2
        for row_idx, raw in candidates:
            norm = normalize_text(raw)
            if norm != "":
                chosen = norm
                chosen_row = row_idx
                break
        names.append(chosen)
        used_rows.append(chosen_row)
    return names, used_rows


def disambiguate_duplicate_names(names: List[str]) -> Tuple[List[str], Dict[str, List[str]]]:
    """
    Detect duplicates in normalized names. Keep first occurrence as-is.
    For subsequent duplicates, append _dup1, _dup2, ... deterministically.
    Returns:
      - new_names list
      - duplicates_info: base_name -> list of assigned duplicate names (in order)
    """
    seen_count: Dict[str, int] = {}
    duplicates_info: Dict[str, List[str]] = {}
    new_names: List[str] = []
    for name in names:
        count = seen_count.get(name, 0)
        if count == 0:
            new_names.append(name)
            seen_count[name] = 1
        else:
            new_name = f"{name}_{count + 1}"
            new_names.append(new_name)
            seen_count[name] = count + 1
            duplicates_info.setdefault(name, []).append(new_name)
    return new_names, duplicates_info


def find_target_index_by_prefix(
    column_names: List[str],
    header_rows_norm: pd.DataFrame,
    target_prefix: str,
) -> Tuple[Optional[int], str, List[int]]:
    """
    Try to find target column by normalized startswith across:
      1) Derived column names
         - If multiple matches, pick the first after sorting by normalized name; return all matched indices.
      2) If not found, scan all header rows (0..2) for any cell that matches and use its column index.
         - If multiple matches, pick the first after sorting by normalized name; return all matched indices.
    Returns:
      (index or None, mode, matched_indices)
      mode is 'derived' if found in names, 'scanned_row_k' if found in header row k, or '' if not found.
    """
    prefix_key = normalize_key_for_match(target_prefix)

    # 1) Search in derived names
    derived_matches = [idx for idx, name in enumerate(column_names) if normalize_key_for_match(name).startswith(prefix_key)]
    if derived_matches:
        # Sort matches by normalized name
        derived_matches_sorted = sorted(derived_matches, key=lambda i: normalize_key_for_match(column_names[i]))
        return derived_matches_sorted[0], "derived", derived_matches_sorted

    # 2) Scan all three header rows
    matched_cols: Set[int] = set()
    for row_idx in range(min(3, header_rows_norm.shape[0])):
        for col_idx in range(header_rows_norm.shape[1]):
            cell = to_str_or_empty(header_rows_norm.iat[row_idx, col_idx])
            if normalize_key_for_match(cell).startswith(prefix_key):
                matched_cols.add(col_idx)
        if matched_cols:
            # Use names (column_names) to sort deterministically
            matched_list = sorted(matched_cols, key=lambda i: normalize_key_for_match(column_names[i]) if i < len(column_names) else "")
            return matched_list[0], f"scanned_row_{row_idx}", matched_list

    return None, "", []


# ----------------------------
# Hints and classification helpers
# ----------------------------

class Hints:
    def __init__(self) -> None:
        self.categorical_by_name: Set[str] = set()
        self.numerical_by_name: Set[str] = set()
        # General patterns for names/descriptions
        self.categorical_patterns: List[re.Pattern] = []
        self.numerical_patterns: List[re.Pattern] = []


def default_categorical_patterns() -> List[re.Pattern]:
    pats: List[str] = [
        r"\(.*?\b1\s*[-–]\s*.*?\)",  # enumeration like "(1 - ...)"
        r"\bда\s*[-=]?\s*1\b",       # "да-1"
        r"\bнет\s*[-=]?\s*2\b",      # "нет-2"
        r"\bгруппа\b",
        r"\bтип\b",
        r"\bкатегор",                # категория/категориальный
        r"\bстепень\b",
        r"\bстадия\b",
    ]
    return [re.compile(p, flags=re.IGNORECASE) for p in pats]


def default_numerical_patterns() -> List[re.Pattern]:
    pats: List[str] = [
        r"\bкг\b",
        r"\bмм\s*рт\.?\s*ст\.?\b",
        r"\bмг\s*/\s*дл\b",
        r"\bг\s*/\s*л\b",
        r"\bсад\b",
        r"\bдад\b",
        r"\bкреатинин\b",
        r"\bобщий\s+белок\b",
        r"\bwbc[-\s]*лейкоцит",  # WBC-лейкоциты
    ]
    return [re.compile(p, flags=re.IGNORECASE) for p in pats]


def load_column_analysis_hints(path: str) -> Hints:
    """
    Parse column_analysis.txt lines as "ColumnName: description".
    Build hints:
      - categorical_by_name / numerical_by_name: normalized names considered definitive.
      - categorical_patterns / numerical_patterns: compiled regexes for general patterns.
    Robust to formatting differences; ignore lines we cannot parse.
    """
    hints = Hints()
    hints.categorical_patterns = default_categorical_patterns()
    hints.numerical_patterns = default_numerical_patterns()

    if not os.path.exists(path):
        print(f'INFO: column_analysis hints file not found at "{path}". Proceeding without direct hints.', flush=True)
        return hints

    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception as e:
        print(f'WARNING: Failed to read "{path}": {e}. Proceeding without direct hints.', flush=True)
        return hints

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        # Attempt to split "name: description"
        name_part = None
        desc_part = ""
        if ":" in line:
            name_part, desc_part = line.split(":", 1)
        else:
            # If no colon, treat whole line as description; skip definitive mapping
            desc_part = line

        name_norm = normalize_text(name_part) if name_part is not None else ""

        desc_norm = normalize_text(desc_part)
        desc_key = desc_norm.casefold()

        # If we have a name, try to infer from description using patterns
        if name_norm:
            cat_hit = any(p.search(desc_key) for p in hints.categorical_patterns)
            num_hit = any(p.search(desc_key) for p in hints.numerical_patterns)
            if cat_hit and not num_hit:
                hints.categorical_by_name.add(name_norm.casefold())
            elif num_hit and not cat_hit:
                hints.numerical_by_name.add(name_norm.casefold())
            # If both/no decisive, leave undecided (heuristics or name patterns will handle)
    return hints


def classify_features(
    X: pd.DataFrame,
    hints: Hints,
) -> Tuple[List[str], List[str], Dict[str, str], Dict[str, int], List[str]]:
    """
    Classify X columns into categorical_features and numerical_features with priority:
      a) Direct hints from column_analysis.txt sets
      b) Name-based cues (keywords/patterns)
      c) Data-based heuristics
    Returns:
      - categorical_features (ordered as in X)
      - numerical_features (ordered as in X)
      - source_map: col -> 'hints' | 'name' | 'heuristic'
      - source_counts: dict with counts per source
      - high_card_cat: list of columns categorized as categorical by heuristics with high cardinality warning
    """
    categorical_features: List[str] = []
    numerical_features: List[str] = []
    source_map: Dict[str, str] = {}
    source_counts: Dict[str, int] = {"hints": 0, "name": 0, "heuristic": 0}
    high_card_cat: List[str] = []

    # Name-based cues (lowercased, normalized)
    name_categorical_cues = [
        "(да-1", "(1 -", "группа", "тип", "категория", "степень", "стадия",
    ]

    for col in X.columns:
        col_key = normalize_key_for_match(col)  # casefolded normalized
        decided: Optional[str] = None
        source: Optional[str] = None

        # a) Direct hints by normalized name
        if col_key in hints.categorical_by_name:
            decided = "categorical"
            source = "hints"
        elif col_key in hints.numerical_by_name:
            decided = "numerical"
            source = "hints"

        # b) Name-based cues/patterns
        if decided is None:
            # Patterns from hints apply to name as well
            name_hit_cat = any(p.search(col_key) for p in hints.categorical_patterns)
            name_hit_num = any(p.search(col_key) for p in hints.numerical_patterns)
            if name_hit_cat and not name_hit_num:
                decided = "categorical"
                source = "name"
            elif name_hit_num and not name_hit_cat:
                decided = "numerical"
                source = "name"
            else:
                # Keyword cues
                if any(k in col_key for k in name_categorical_cues):
                    decided = "categorical"
                    source = "name"

        # c) Data-based heuristics
        if decided is None:
            s = X[col]
            # Try coercion to numeric
            s_num = pd.to_numeric(s, errors="coerce")
            ratio_numeric = float(s_num.notna().mean())
            if ratio_numeric >= 0.90:
                # Use numeric heuristics
                non_null = s_num.dropna()
                nuniq = int(non_null.nunique(dropna=True))
                if nuniq == 0:
                    # Degenerate; treat as categorical with a note
                    decided = "categorical"
                    source = "heuristic"
                else:
                    # Integer-like check within tolerance
                    nearest_int = np.rint(non_null.to_numpy())
                    int_like = np.all(np.isfinite(non_null.to_numpy()) & (np.abs(non_null.to_numpy() - nearest_int) <= 1e-9))
                    if int_like and nuniq <= 10:
                        decided = "categorical"
                        source = "heuristic"
                    else:
                        decided = "numerical"
                        source = "heuristic"
            else:
                # Object-like/text with low numeric coercion success
                non_null = s.dropna()
                nuniq = int(non_null.nunique(dropna=True))
                if nuniq <= 10:
                    decided = "categorical"
                    source = "heuristic"
                else:
                    # Prefer categorical but warn about high cardinality (likely free-form)
                    decided = "categorical"
                    source = "heuristic"
                    high_card_cat.append(col)

        # Record classification preserving order
        if decided == "categorical":
            categorical_features.append(col)
        else:
            numerical_features.append(col)
        # Track source
        source_map[col] = source or "heuristic"
        source_counts[source_map[col]] = source_counts.get(source_map[col], 0) + 1

    return categorical_features, numerical_features, source_map, source_counts, high_card_cat


# ----------------------------
# I/O helpers
# ----------------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_parquet_safe(df_or_series: object, path: str) -> bool:
    """
    Try to save as parquet. Return True if succeeded, False if parquet engine missing or error.
    """
    try:
        if isinstance(df_or_series, pd.Series):
            df_or_series.to_frame().to_parquet(path, index=False)
        elif isinstance(df_or_series, pd.DataFrame):
            df_or_series.to_parquet(path, index=False)
        else:
            raise TypeError("Unsupported type for parquet save")
        return True
    except Exception as e:
        print(f'WARNING: Failed to save parquet "{path}": {e}', flush=True)
        return False


# ----------------------------
# Main
# ----------------------------

AUTO_ID_CANONICAL_RAW = "№ пациента (пробирки) "  # note trailing space in raw


def run(args: argparse.Namespace) -> None:
    input_path: str = args.input
    output_dir: str = args.output_dir
    id_column_arg: Optional[str] = args.id_column
    target_prefix: str = args.target_prefix

    if not os.path.exists(input_path):
        raise FileNotFoundError(f'Input Excel file not found: "{input_path}"')

    # Read entire sheet with header=None and dtype=str to preserve the first three rows as-is.
    # Engine left to pandas default (openpyxl typically for .xlsx).
    df_all = pd.read_excel(input_path, header=None, dtype=str)

    if df_all.shape[0] < 4:
        raise RuntimeError(
            f"Expected at least 4 rows (3 header rows + data), but got {df_all.shape[0]} rows."
        )

    # Build normalized header rows DataFrame (first three rows only)
    header_rows_raw = df_all.iloc[0:3, :].copy()

    # Normalize each cell in these first three rows
    header_rows_norm = header_rows_raw.map(normalize_text)

    # Derive column names primarily from 3rd row with fallback to 2nd then 1st
    column_names_base, used_rows = derive_column_names_from_three_rows(header_rows_norm)

    # Handle duplicate normalized column names deterministically
    column_names, duplicates_info = disambiguate_duplicate_names(column_names_base)
    if duplicates_info:
        details = []
        for base, dups in duplicates_info.items():
            details.append(f'base "{base}" -> duplicates renamed: {", ".join([f"{n}" for n in dups])}')
        print(
            "WARNING: Duplicate normalized column names detected and disambiguated:\n - "
            + "\n - ".join(details),
            flush=True,
        )

    # Construct the data frame from 4th row onward and reset index
    df_data = df_all.iloc[3:, :].copy()
    df_data.columns = column_names
    df_data.reset_index(drop=True, inplace=True)

    # Find target column by prefix on normalized names; if not found, scan all header rows
    target_idx, mode, match_indices = find_target_index_by_prefix(column_names, header_rows_norm, target_prefix)
    if target_idx is None:
        # Helpful diagnostic error
        normalized_name_list = [normalize_text(n) for n in column_names]
        preview_rows = header_rows_norm.copy()
        # Build a compact text preview of first three header rows
        preview_csv = preview_rows.to_csv(index=False, header=False)
        raise RuntimeError(
            "Target column not found using prefix. Diagnostics:\n"
            f' - Target prefix (normalized): "{normalize_key_for_match(target_prefix)}"\n'
            " - Primary naming row used: 2 (with per-column fallback to rows 1 and 0)\n"
            f" - First three header rows after normalization (CSV preview):\n{preview_csv}\n"
            f" - Normalized derived column names ({len(normalized_name_list)}):\n{normalized_name_list}"
        )

    # Warn if multiple matches
    if len(match_indices) > 1:
        # Deterministic chosen name:
        chosen = column_names[target_idx]
        others = [column_names[i] for i in match_indices if i != target_idx]
        print(
            f'WARNING: Multiple columns matched target prefix "{target_prefix}". '
            f'Chosen (by sorted normalized name): "{chosen}". Other matches: {others}',
            flush=True,
        )

    target_name = df_data.columns[target_idx]

    # Identifier column removal
    drop_id_names: List[str] = []
    id_col_matched_name_cli: Optional[str] = None
    id_col_matched_name_auto: Optional[str] = None

    # 1) Auto identifier removal by canonical normalized name ignoring spaces
    auto_id_key = normalize_key_for_id_match(AUTO_ID_CANONICAL_RAW)
    auto_matches = [c for c in df_data.columns if normalize_key_for_id_match(c) == auto_id_key]
    if auto_matches:
        id_col_matched_name_auto = auto_matches[0]
        drop_id_names.append(id_col_matched_name_auto)
        if len(auto_matches) > 1:
            print(
                f'WARNING: Multiple columns matched the auto ID canonical name "{AUTO_ID_CANONICAL_RAW}". '
                f'Only the first matched column will be dropped: "{id_col_matched_name_auto}".',
                flush=True,
            )
    else:
        print(
            f'INFO: Auto ID column not found by canonical match "{AUTO_ID_CANONICAL_RAW}". Proceeding without auto ID drop.',
            flush=True,
        )

    # 2) Optional identifier column removal if --id-column provided
    if id_column_arg:
        id_key_cli = normalize_key_for_match(id_column_arg)
        matches_cli = [c for c in df_data.columns if normalize_key_for_match(c) == id_key_cli]
        if matches_cli:
            id_col_matched_name_cli = matches_cli[0]
            drop_id_names.append(id_col_matched_name_cli)
            if len(matches_cli) > 1:
                print(
                    f'WARNING: Multiple columns match the provided --id-column "{id_column_arg}". '
                    f'Only the first matched column will be dropped: "{id_col_matched_name_cli}".',
                    flush=True,
                )
        else:
            print(
                f'INFO: --id-column provided ("{id_column_arg}") but no matching column found after normalization. '
                "Proceeding without removing an ID column from --id-column.",
                flush=True,
            )

    # De-duplicate drop list and ensure target not included
    drop_cols: List[str] = [target_name]
    for nm in drop_id_names:
        if nm and nm not in drop_cols:
            drop_cols.append(nm)

    # Build X and y
    X = df_data.drop(columns=drop_cols, errors="ignore")
    y = df_data[target_name].copy()

    # Load hints from column_analysis.txt (if available)
    hints_path = "column_analysis.txt"
    hints = load_column_analysis_hints(hints_path)

    # Classify features
    categorical_features, numerical_features, source_map, source_counts, high_card_cat = classify_features(X, hints)

    # Save prepared_data.pkl in working directory
    prepared_pickle_path = os.path.join(".", "prepared_data.pkl")
    try:
        with open(prepared_pickle_path, "wb") as pf:
            pickle.dump(
                {
                    "X": X,
                    "y": y,
                    "categorical_features": categorical_features,
                    "numerical_features": numerical_features,
                },
                pf,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        pickle_ok = True
    except Exception as e:
        print(f'WARNING: Failed to write prepared_data.pkl: {e}', flush=True)
        pickle_ok = False

    # Prepare output directory
    ensure_dir(output_dir)

    # Save audit files
    columns_json_path = os.path.join(output_dir, "columns_normalized.json")
    with open(columns_json_path, "w", encoding="utf-8") as jf:
        json.dump(list(df_data.columns), jf, ensure_ascii=False, indent=2)

    header_preview_path = os.path.join(output_dir, "header_rows_preview.csv")
    header_rows_norm.to_csv(header_preview_path, index=False, header=False, encoding="utf-8")

    # Save X and y as CSV (UTF-8)
    x_csv_path = os.path.join(output_dir, "X.csv")
    y_csv_path = os.path.join(output_dir, "y.csv")
    X.to_csv(x_csv_path, index=False, encoding="utf-8")
    y.to_frame(name=target_name).to_csv(y_csv_path, index=False, encoding="utf-8")

    # Save X and y as Parquet (if parquet engine available)
    x_parquet_path = os.path.join(output_dir, "X.parquet")
    y_parquet_path = os.path.join(output_dir, "y.parquet")
    x_parquet_ok = save_parquet_safe(X, x_parquet_path)
    y_parquet_ok = save_parquet_safe(y, y_parquet_path)

    # Diagnostics printing
    n_rows_total = df_all.shape[0]
    n_cols_total = df_all.shape[1]
    n_rows_data = df_data.shape[0]
    n_cols_data = df_data.shape[1]
    used_counts = {0: 0, 1: 0, 2: 0}
    for r in used_rows:
        used_counts[r] = used_counts.get(r, 0) + 1

    print("=== Data preparation summary ===", flush=True)
    print(f'Input file: "{input_path}"', flush=True)
    print(f"Total rows (including 3 header rows): {n_rows_total}", flush=True)
    print(f"Total columns in source: {n_cols_total}", flush=True)
    print("Header naming source: primary row index 2 with per-column fallback to 1 then 0", flush=True)
    print(
        f" - Chosen name counts by header row: row2={used_counts.get(2,0)}, row1={used_counts.get(1,0)}, row0={used_counts.get(0,0)}",
        flush=True,
    )
    print(f"Data rows (after removing 3 header rows): {n_rows_data}", flush=True)
    print(f"Columns before separation: {n_cols_data}", flush=True)
    print(f'Exact matched target column name: "{target_name}" (found via: {mode})', flush=True)
    if id_col_matched_name_auto:
        print(f'ID column dropped (auto): "{id_col_matched_name_auto}"', flush=True)
    else:
        print("ID column dropped (auto): no", flush=True)
    if id_col_matched_name_cli:
        print(f'ID column dropped (--id-column): "{id_col_matched_name_cli}"', flush=True)
    else:
        print("ID column dropped (--id-column): no", flush=True)

    # Feature classification summary
    print(f"Rows: {len(X)}", flush=True)
    print(f"Features in X: {X.shape[1]}", flush=True)
    print(f"Target column: {y.name}", flush=True)
    print(f"categorical_features: {len(categorical_features)}", flush=True)
    print(f"numerical_features: {len(numerical_features)}", flush=True)
    print(
        f"Classification sources -> hints: {source_counts.get('hints',0)}, "
        f"name-patterns: {source_counts.get('name',0)}, heuristics: {source_counts.get('heuristic',0)}",
        flush=True,
    )
    if high_card_cat:
        print(
            f'WARNING: {len(high_card_cat)} categorical features have high cardinality (likely free-form): {high_card_cat[:10]}'
            + (" ..." if len(high_card_cat) > 10 else ""),
            flush=True,
        )

    if duplicates_info:
        affected = ", ".join(sorted(duplicates_info.keys()))
        print(f"Note: duplicate normalized column names resolved for base names: {affected}", flush=True)

    print("Output files:", flush=True)
    print(f' - columns_normalized.json: "{columns_json_path}"', flush=True)
    print(f' - header_rows_preview.csv: "{header_preview_path}"', flush=True)
    print(f' - X.csv: "{x_csv_path}"', flush=True)
    print(f' - y.csv: "{y_csv_path}"', flush=True)
    if x_parquet_ok:
        print(f' - X.parquet: "{x_parquet_path}"', flush=True)
    else:
        print(' - X.parquet: not written (see warning above)', flush=True)
    if y_parquet_ok:
        print(f' - y.parquet: "{y_parquet_path}"', flush=True)
    else:
        print(' - y.parquet: not written (see warning above)', flush=True)
    if pickle_ok:
        print(f' - prepared_data.pkl: "{prepared_pickle_path}"', flush=True)
    else:
        print(' - prepared_data.pkl: not written (see warning above)', flush=True)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Prepare data by deriving and normalizing headers, splitting X/y, and classifying features.")
    p.add_argument(
        "--input",
        type=str,
        default="result_no_text_filled_more_than_80_filled.xlsx",
        help="Path to the Excel file (.xlsx).",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="prepared",
        help='Directory to save outputs (default: "prepared").',
    )
    p.add_argument(
        "--id-column",
        type=str,
        default=None,
        help="Optional identifier column name to drop (matched after normalization).",
    )
    p.add_argument(
        "--target-prefix",
        type=str,
        default="Группа наблюдения",
        help='Prefix to detect target column by startswith after normalization (default: "Группа наблюдения").',
    )
    return p


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    try:
        run(args)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr, flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()