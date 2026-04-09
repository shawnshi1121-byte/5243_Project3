from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyreadr


def build_sales_demo() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    months = pd.date_range("2025-01-01", periods=36, freq="W")
    regions = ["North", "South", "East", "West"]
    channels = ["Online", "Retail", "Partner"]
    products = ["Analytics", "Cloud", "Consulting", "Support"]

    rows: list[dict[str, Any]] = []
    for i, date in enumerate(months):
        for region in regions:
            revenue = rng.normal(12000, 2500)
            cost = revenue * rng.uniform(0.45, 0.8)
            units = max(10, int(rng.normal(220, 50)))
            rows.append(
                {
                    "week": date,
                    "region": region,
                    "channel": channels[(i + len(region)) % len(channels)],
                    "product": products[(i + len(region) * 2) % len(products)],
                    "revenue": round(revenue, 2),
                    "cost": round(cost, 2),
                    "units": units,
                    "satisfaction": round(rng.uniform(3.1, 4.9), 2),
                    "campaign_spend": round(rng.normal(2500, 550), 2),
                }
            )

    df = pd.DataFrame(rows)
    df.loc[3, "revenue"] = np.nan
    df.loc[7, "channel"] = None
    df = pd.concat([df, df.iloc[[4]]], ignore_index=True)
    return df


def build_student_demo() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    majors = ["Statistics", "Economics", "Computer Science", "Psychology"]
    study_modes = ["Full-time", "Part-time"]
    cohorts = ["2023", "2024", "2025"]

    rows: list[dict[str, Any]] = []
    for idx in range(160):
        attendance = np.clip(rng.normal(0.86, 0.09), 0.45, 1.0)
        homework = np.clip(rng.normal(82, 12), 45, 100)
        exam = np.clip(45 + attendance * 42 + rng.normal(0, 8), 40, 100)
        participation = np.clip(rng.normal(7.2, 1.4), 2, 10)
        rows.append(
            {
                "student_id": f"S{1000 + idx}",
                "cohort": cohorts[idx % len(cohorts)],
                "major": majors[idx % len(majors)],
                "study_mode": study_modes[idx % len(study_modes)],
                "attendance_rate": round(attendance, 3),
                "homework_score": round(homework, 1),
                "exam_score": round(exam, 1),
                "participation": round(participation, 1),
                "internship": "Yes" if rng.random() > 0.35 else "No",
            }
        )

    df = pd.DataFrame(rows)
    df.loc[5, "exam_score"] = np.nan
    df.loc[9, "major"] = None
    df = pd.concat([df, df.iloc[[11]]], ignore_index=True)
    return df


SAMPLE_DATASETS = {
    "Sales Performance Demo": build_sales_demo(),
    "Student Success Demo": build_student_demo(),
}

BASE_DIR = Path(__file__).parent


@dataclass
class FeatureSpec:
    operation: str
    col_a: str
    col_b: str | None
    new_name: str
    bins: int


def parse_uploaded_file(path: str | Path, name: str) -> pd.DataFrame:
    path = str(path)
    suffix = name.lower().rsplit(".", 1)[-1] if "." in name else ""

    if suffix == "csv":
        return pd.read_csv(path, sep=None, engine="python")
    if suffix in {"xlsx", "xls"}:
        return pd.read_excel(path)
    if suffix == "json":
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, (list, dict)):
            return pd.json_normalize(payload)
        raise ValueError("JSON file format is not tabular.")
    if suffix == "rds":
        result = pyreadr.read_r(path)
        if not result:
            raise ValueError("The RDS file could not be parsed.")
        return result[next(iter(result.keys()))]

    raise ValueError("Unsupported file type. Please upload CSV, Excel, JSON, or RDS.")


def standardize_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.select_dtypes(include=["object", "string"]).columns:
        out[col] = (
            out[col]
            .astype("string")
            .str.strip()
            .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
        )
    return out


def handle_missing(df: pd.DataFrame, strategy: str) -> pd.DataFrame:
    out = df.copy()
    if strategy == "none":
        return out
    if strategy == "drop":
        return out.dropna()

    numeric_cols = out.select_dtypes(include=[np.number]).columns
    categorical_cols = out.select_dtypes(exclude=[np.number]).columns
    for col in numeric_cols:
        if strategy == "mean":
            out[col] = out[col].fillna(out[col].mean())
        elif strategy == "median":
            out[col] = out[col].fillna(out[col].median())
        elif strategy == "mode":
            mode = out[col].mode(dropna=True)
            out[col] = out[col].fillna(mode.iloc[0] if not mode.empty else 0)
    for col in categorical_cols:
        mode = out[col].mode(dropna=True)
        fill_value = mode.iloc[0] if not mode.empty else "Missing"
        if strategy in {"mean", "median", "mode"}:
            out[col] = out[col].fillna(fill_value)
    return out


def handle_outliers(df: pd.DataFrame, strategy: str) -> pd.DataFrame:
    if strategy == "none":
        return df.copy()
    out = df.copy()
    for col in out.select_dtypes(include=[np.number]).columns:
        series = out[col]
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            continue
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        if strategy == "cap":
            out[col] = series.clip(lower, upper)
        elif strategy == "remove":
            out = out[(out[col].isna()) | ((out[col] >= lower) & (out[col] <= upper))]
    return out


def scale_numeric(df: pd.DataFrame, strategy: str) -> pd.DataFrame:
    if strategy == "none":
        return df.copy()
    out = df.copy()
    for col in out.select_dtypes(include=[np.number]).columns:
        series = out[col]
        if series.isna().all():
            continue
        if strategy == "zscore":
            std = series.std()
            if pd.notna(std) and std > 0:
                out[col] = (series - series.mean()) / std
        elif strategy == "minmax":
            minimum = series.min()
            maximum = series.max()
            if pd.notna(minimum) and pd.notna(maximum) and maximum > minimum:
                out[col] = (series - minimum) / (maximum - minimum)
    return out


def apply_feature_engineering(df: pd.DataFrame, specs: list[FeatureSpec]) -> pd.DataFrame:
    out = df.copy()
    for spec in specs:
        if spec.col_a not in out.columns:
            continue
        if spec.operation == "log":
            numeric = pd.to_numeric(out[spec.col_a], errors="coerce").clip(lower=0)
            out[spec.new_name] = np.log1p(numeric)
            continue
        if spec.operation == "bin":
            numeric = pd.to_numeric(out[spec.col_a], errors="coerce")
            out[spec.new_name] = pd.cut(numeric, bins=max(2, spec.bins), include_lowest=True)
            continue
        if spec.col_b is None or spec.col_b not in out.columns:
            continue
        left = pd.to_numeric(out[spec.col_a], errors="coerce")
        right = pd.to_numeric(out[spec.col_b], errors="coerce")
        if spec.operation == "sum":
            out[spec.new_name] = left + right
        elif spec.operation == "difference":
            out[spec.new_name] = left - right
        elif spec.operation == "product":
            out[spec.new_name] = left * right
        elif spec.operation == "ratio":
            out[spec.new_name] = left / right.replace({0: np.nan})
    return out


def build_profile(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "metric": [
                "Rows",
                "Columns",
                "Missing cells",
                "Duplicate rows",
                "Numeric columns",
                "Categorical columns",
            ],
            "value": [
                len(df),
                len(df.columns),
                int(df.isna().sum().sum()),
                int(df.duplicated().sum()),
                len(df.select_dtypes(include=[np.number]).columns),
                len(df.select_dtypes(exclude=[np.number]).columns),
            ],
        }
    )


def build_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df.select_dtypes(include=[np.number])
    if numeric.empty:
        return pd.DataFrame({"message": ["No numeric columns available."]})
    return numeric.describe().transpose().reset_index(names="variable").round(3)


def pick_categorical_default(df: pd.DataFrame, max_unique: int = 25) -> str | None:
    candidates: list[tuple[int, str]] = []
    for col in df.select_dtypes(exclude=[np.number]).columns:
        try:
            unique_count = int(df[col].nunique(dropna=True))
        except Exception:
            continue
        if 1 < unique_count <= max_unique:
            candidates.append((unique_count, col))
    if candidates:
        candidates.sort()
        return candidates[0][1]
    return None


def pick_numeric_default(df: pd.DataFrame, rank: int = 0) -> str | None:
    numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
    if not numeric_cols:
        return None
    scored: list[tuple[float, str]] = []
    row_count = max(len(df), 1)
    for col in numeric_cols:
        unique_ratio = float(df[col].nunique(dropna=True)) / row_count
        is_identifier_like = unique_ratio > 0.9 or col.lower().endswith("_id") or col.lower() == "id"
        score = unique_ratio + (2 if is_identifier_like else 0)
        scored.append((score, col))
    scored.sort()
    return scored[min(rank, len(scored) - 1)][1]


def pick_dimension_default(df: pd.DataFrame) -> str | None:
    categorical = pick_categorical_default(df, max_unique=40)
    if categorical:
        return categorical
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            return col
    return df.columns[0] if len(df.columns) else None


def low_cardinality_columns(df: pd.DataFrame, max_unique: int = 25) -> list[str]:
    columns: list[str] = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
        try:
            unique_count = int(df[col].nunique(dropna=True))
        except Exception:
            continue
        if 1 < unique_count <= max_unique:
            columns.append(col)
    return columns
