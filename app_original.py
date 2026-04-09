from __future__ import annotations

import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import pyreadr
from shiny import App, reactive, render, req, ui
from shinywidgets import output_widget, render_widget


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
            row = {
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
            rows.append(row)

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
        row = {
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
        rows.append(row)

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


def parse_uploaded_file(file_info: dict[str, Any]) -> pd.DataFrame:
    path = file_info["datapath"]
    name = file_info["name"]
    suffix = name.lower().rsplit(".", 1)[-1] if "." in name else ""

    if suffix == "csv":
        return pd.read_csv(path, sep=None, engine="python")
    if suffix in {"xlsx", "xls"}:
        return pd.read_excel(path)
    if suffix == "json":
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, list):
            return pd.json_normalize(payload)
        if isinstance(payload, dict):
            return pd.json_normalize(payload)
        raise ValueError("JSON file format is not tabular.")
    if suffix == "rds":
        result = pyreadr.read_r(path)
        if not result:
            raise ValueError("The RDS file could not be parsed.")
        first_key = next(iter(result.keys()))
        return result[first_key]

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

    if strategy == "mean":
        for col in numeric_cols:
            out[col] = out[col].fillna(out[col].mean())
        for col in categorical_cols:
            mode = out[col].mode(dropna=True)
            fill_value = mode.iloc[0] if not mode.empty else "Missing"
            out[col] = out[col].fillna(fill_value)
    elif strategy == "median":
        for col in numeric_cols:
            out[col] = out[col].fillna(out[col].median())
        for col in categorical_cols:
            mode = out[col].mode(dropna=True)
            fill_value = mode.iloc[0] if not mode.empty else "Missing"
            out[col] = out[col].fillna(fill_value)
    elif strategy == "mode":
        for col in out.columns:
            mode = out[col].mode(dropna=True)
            fill_value = mode.iloc[0] if not mode.empty else "Missing"
            out[col] = out[col].fillna(fill_value)

    return out


def handle_outliers(df: pd.DataFrame, strategy: str) -> pd.DataFrame:
    if strategy == "none":
        return df.copy()

    out = df.copy()
    numeric_cols = out.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        return out

    for col in numeric_cols:
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
    numeric_cols = out.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
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
            safe_right = right.replace({0: np.nan})
            out[spec.new_name] = left / safe_right

    return out


def build_profile(df: pd.DataFrame) -> pd.DataFrame:
    summary = pd.DataFrame(
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
    return summary


def build_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df.select_dtypes(include=[np.number])
    if numeric.empty:
        return pd.DataFrame({"message": ["No numeric columns available."]})
    return numeric.describe().transpose().reset_index(names="variable").round(3)


def pick_filter_default(df: pd.DataFrame) -> str | None:
    if df.empty:
        return None

    numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
    if numeric_cols:
        return numeric_cols[0]

    categorical_cols = list(df.select_dtypes(exclude=[np.number]).columns)
    if not categorical_cols:
        return df.columns[0]

    cardinalities = []
    for col in categorical_cols:
        try:
            cardinalities.append((df[col].nunique(dropna=True), col))
        except Exception:
            cardinalities.append((10**9, col))
    cardinalities.sort()
    return cardinalities[0][1]


def pick_categorical_default(df: pd.DataFrame, max_unique: int = 25) -> str | None:
    categorical_cols = list(df.select_dtypes(exclude=[np.number]).columns)
    if not categorical_cols:
        return None

    candidates: list[tuple[int, str]] = []
    for col in categorical_cols:
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


def pick_dimension_default(df: pd.DataFrame) -> str | None:
    categorical = pick_categorical_default(df, max_unique=40)
    if categorical:
        return categorical

    all_cols = list(df.columns)
    if not all_cols:
        return None

    for col in all_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            return col
    return all_cols[0]


def pick_numeric_default(df: pd.DataFrame, rank: int = 0) -> str | None:
    numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
    if not numeric_cols:
        return None

    scored: list[tuple[float, str]] = []
    row_count = max(len(df), 1)
    for col in numeric_cols:
        series = df[col]
        unique_ratio = float(series.nunique(dropna=True)) / row_count
        is_identifier_like = unique_ratio > 0.9 or col.lower().endswith("_id") or col.lower() == "id"
        score = unique_ratio + (2 if is_identifier_like else 0)
        scored.append((score, col))
    scored.sort()
    index = min(rank, len(scored) - 1)
    return scored[index][1]


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


app_ui = ui.page_navbar(
    ui.nav_panel(
        "Guide",
        ui.div(
            {"class": "hero-panel"},
            ui.div(
                {"class": "hero-copy"},
                ui.h1("DataCanvas Studio"),
                ui.p(
                    "A polished Shiny dashboard for uploading, cleaning, transforming, "
                    "and exploring tabular datasets with minimal friction."
                ),
                ui.div(
                    {"class": "hero-badges"},
                    ui.span("CSV / Excel / JSON / RDS"),
                    ui.span("Interactive cleaning"),
                    ui.span("Feature engineering"),
                    ui.span("Plotly EDA"),
                ),
                ui.div(
                    {"style": "margin-top: 1.25rem;"},
                    ui.tags.button(
                        "Import your dataset",
                        type="button",
                        class_="btn btn-primary",
                        onclick="window.openDataSourceUpload();",
                    ),
                ),
            ),
            ui.div(
                {"class": "hero-card"},
                ui.h3("Recommended workflow"),
                ui.tags.ol(
                    ui.tags.li("Start with a built-in demo or upload your own dataset."),
                    ui.tags.li("Choose cleaning rules for missing values, duplicates, scaling, and outliers."),
                    ui.tags.li("Create new features and inspect the transformed preview."),
                    ui.tags.li("Use the EDA tab for filters, summary statistics, and interactive charts."),
                    ui.tags.li("Download the processed dataset or cite the deployment link in your report."),
                ),
            ),
        ),
        ui.layout_columns(
            ui.card(
                ui.card_header("What the app supports"),
                ui.tags.ul(
                    ui.tags.li("Built-in demo datasets for fast exploration"),
                    ui.tags.li("Upload flow for CSV, Excel, JSON, and RDS"),
                    ui.tags.li("Configurable preprocessing pipeline with real-time updates"),
                    ui.tags.li("Feature generation via arithmetic, log transform, and binning"),
                    ui.tags.li("Interactive charts, filters, and correlation view"),
                ),
            ),
            ui.card(
                ui.card_header("Assignment alignment"),
                ui.tags.ul(
                    ui.tags.li("Dataset loading with multiple formats"),
                    ui.tags.li("Data cleaning and preprocessing controls"),
                    ui.tags.li("Feature engineering with visible impact"),
                    ui.tags.li("Exploratory data analysis and summary statistics"),
                    ui.tags.li("Documented workflow and export-ready outputs"),
                ),
            ),
            col_widths=[6, 6],
        ),
    ),
    ui.nav_panel(
        "Data Source",
        ui.layout_sidebar(
            ui.sidebar(
                ui.input_radio_buttons(
                    "source_mode",
                    "Data source",
                    {
                        "demo": "Use a built-in demo dataset",
                        "upload": "Upload my own file",
                    },
                    selected="demo",
                ),
                ui.panel_conditional(
                    "input.source_mode === 'demo'",
                    ui.input_select(
                        "demo_dataset",
                        "Built-in dataset",
                        choices=list(SAMPLE_DATASETS.keys()),
                    ),
                ),
                ui.panel_conditional(
                    "input.source_mode === 'upload'",
                    ui.input_file(
                        "file_upload",
                        "Upload a dataset",
                        accept=[".csv", ".xlsx", ".xls", ".json", ".rds"],
                    ),
                ),
                ui.input_text("dataset_label", "Dataset label", "Project 2 demo"),
                width=320,
            ),
            ui.layout_columns(
                ui.card(
                    ui.card_header("Dataset profile"),
                    ui.output_text_verbatim("source_status"),
                    ui.output_data_frame("profile_table"),
                ),
                ui.card(
                    ui.card_header("Raw preview"),
                    ui.output_data_frame("raw_preview"),
                ),
                col_widths=[4, 8],
            ),
        ),
    ),
    ui.nav_panel(
        "Cleaning",
        ui.layout_sidebar(
            ui.sidebar(
                ui.input_select(
                    "missing_strategy",
                    "Missing values",
                    {
                        "none": "Keep missing values",
                        "drop": "Drop rows with missing values",
                        "mean": "Impute numeric mean + categorical mode",
                        "median": "Impute numeric median + categorical mode",
                        "mode": "Impute mode for every column",
                    },
                    selected="mean",
                ),
                ui.input_switch("drop_duplicates", "Remove duplicate rows", True),
                ui.input_select(
                    "outlier_strategy",
                    "Outlier handling",
                    {
                        "none": "No outlier treatment",
                        "cap": "Cap using IQR fences",
                        "remove": "Remove rows outside IQR fences",
                    },
                    selected="cap",
                ),
                ui.input_select(
                    "scaling_strategy",
                    "Numeric scaling",
                    {
                        "none": "No scaling",
                        "zscore": "Z-score standardization",
                        "minmax": "Min-max scaling",
                    },
                    selected="none",
                ),
                ui.input_switch("encode_categories", "One-hot encode categorical columns", False),
                width=320,
            ),
            ui.layout_columns(
                ui.card(
                    ui.card_header("Cleaning impact"),
                    ui.output_data_frame("cleaning_profile"),
                ),
                ui.card(
                    ui.card_header("Processed preview"),
                    ui.output_data_frame("processed_preview"),
                ),
                col_widths=[4, 8],
            ),
        ),
    ),
    ui.nav_panel(
        "Feature Engineering",
        ui.layout_sidebar(
            ui.sidebar(
                ui.input_select(
                    "feature_operation",
                    "Feature operation",
                    {
                        "sum": "A + B",
                        "difference": "A - B",
                        "product": "A * B",
                        "ratio": "A / B",
                        "log": "log(1 + A)",
                        "bin": "Bin numeric column",
                    },
                    selected="ratio",
                ),
                ui.output_ui("feature_column_ui"),
                ui.input_text("new_feature_name", "New feature name", "engineered_feature"),
                ui.input_slider("bin_count", "Bins", min=2, max=8, value=4),
                ui.input_action_button("apply_feature", "Add feature", class_="btn-primary"),
                ui.input_action_button("reset_features", "Reset features"),
                width=320,
            ),
            ui.layout_columns(
                ui.card(
                    ui.card_header("Active feature recipes"),
                    ui.output_data_frame("feature_table"),
                ),
                ui.card(
                    ui.card_header("Feature-enhanced preview"),
                    ui.output_data_frame("featured_preview"),
                ),
                col_widths=[4, 8],
            ),
        ),
    ),
    ui.nav_panel(
        "EDA",
        ui.layout_sidebar(
            ui.sidebar(
                ui.div(
                    {"class": "editor-block", "id": "main-editor"},
                    ui.div({"class": "text-muted"}, "Main chart"),
                    ui.h4("Interactive chart"),
                    ui.p({"class": "text-muted"}, "Use these controls to change the primary analysis chart."),
                    ui.input_select(
                        "eda_plot_type",
                        "Chart type",
                        {
                            "bar": "Bar: grouped summary",
                            "line": "Line: trend summary",
                            "histogram": "Histogram / distribution",
                            "scatter": "Scatter",
                            "box": "Box plot",
                        },
                        selected="bar",
                    ),
                    ui.input_select("eda_x", "X variable", choices=[""], selected=""),
                    ui.input_select("eda_y", "Y variable", choices=[""], selected=""),
                    ui.input_select("eda_metric", "Metric", choices={"count": "Row count"}, selected="count"),
                    ui.input_select(
                        "eda_agg",
                        "Aggregation",
                        {
                            "count": "Count rows",
                            "mean": "Mean",
                            "median": "Median",
                            "sum": "Sum",
                            "max": "Max",
                            "min": "Min",
                        },
                        selected="count",
                    ),
                    ui.input_select("eda_color", "Color / group", choices=["None"], selected="None"),
                    ui.input_slider("eda_top_n", "Top categories to show", min=5, max=30, value=12),
                    ui.input_slider("eda_bins", "Histogram bins", min=10, max=60, value=25),
                ),
                ui.div(
                    {"class": "editor-block", "id": "corr-editor"},
                    ui.div({"class": "text-muted"}, "Correlation panel"),
                    ui.h4("Correlation heatmap"),
                    ui.p({"class": "text-muted"}, "Tune how the correlation matrix is computed and displayed."),
                    ui.input_select(
                        "corr_chart_type",
                        "Chart type",
                        {
                            "heatmap": "Correlation heatmap",
                            "bars": "Top correlations with one variable",
                        },
                        selected="heatmap",
                    ),
                    ui.input_select(
                        "corr_method",
                        "Correlation method",
                        {"pearson": "Pearson", "spearman": "Spearman"},
                        selected="pearson",
                    ),
                    ui.input_select("corr_focus_var", "Focus variable", choices=["Auto"], selected="Auto"),
                    ui.input_slider("corr_top_n", "Max numeric columns", min=3, max=12, value=8),
                ),
                ui.div(
                    {"class": "editor-block", "id": "summary-editor"},
                    ui.div({"class": "text-muted"}, "Summary panel"),
                    ui.h4("Summary statistics"),
                    ui.p(
                        {"class": "text-muted"},
                        "Summary statistics follow the current dataset filter. To change the rows included here, adjust the shared filter below.",
                    ),
                    ui.input_select(
                        "summary_chart_type",
                        "Chart type",
                        {"table": "Summary table", "bar": "Mean bar chart"},
                        selected="table",
                    ),
                ),
                ui.div(
                    {"class": "editor-block", "id": "missing-editor"},
                    ui.div({"class": "text-muted"}, "Missingness panel"),
                    ui.h4("Missingness by column"),
                    ui.p(
                        {"class": "text-muted"},
                        "Inspect missingness on raw data by default, or switch to a later pipeline stage when needed.",
                    ),
                    ui.input_select(
                        "missing_data_source",
                        "Dataset source",
                        {
                            "raw": "Raw data",
                            "processed": "After cleaning",
                            "featured": "After feature engineering",
                            "filtered": "Current EDA filtered view",
                        },
                        selected="raw",
                    ),
                    ui.input_select(
                        "missing_chart_type",
                        "Chart type",
                        {"bar": "Bar chart", "donut": "Donut chart"},
                        selected="bar",
                    ),
                    ui.input_slider("missing_top_n", "Columns to display", min=5, max=20, value=12),
                    ui.input_select(
                        "missing_sort",
                        "Sort by",
                        {"desc": "Most missing first", "asc": "Least missing first"},
                        selected="desc",
                    ),
                ),
                ui.div(
                    {"class": "editor-block", "id": "shared-filter-editor"},
                    ui.div({"class": "text-muted"}, "Shared filter"),
                    ui.h4("Filter the dataset used by all EDA panels"),
                    ui.input_select("filter_col", "Filter column", choices=[""], selected=""),
                    ui.output_ui("filter_value_ui"),
                ),
                width=330,
            ),
            ui.div(
                {"class": "eda-canvas", "id": "eda-canvas"},
                ui.card(
                    {
                        "class": "eda-card eda-card-main is-active",
                        "data-panel": "main",
                        "onclick": "window.focusEdaEditor('main-editor', 'main')",
                    },
                    ui.card_header(
                        ui.div(
                            {"class": "eda-card-head"},
                            ui.span("Interactive chart"),
                            ui.div(
                                {"class": "eda-card-actions"},
                                ui.tags.button(
                                    "Edit chart",
                                    type="button",
                                    class_="eda-edit-link",
                                    onclick="event.stopPropagation(); window.focusEdaEditor('main-editor', 'main');",
                                ),
                                ui.tags.button(
                                    "Export PNG",
                                    type="button",
                                    class_="eda-export-link",
                                    onclick="event.stopPropagation(); window.exportCardPng('main');",
                                ),
                            ),
                        )
                    ),
                    output_widget("main_plot"),
                ),
                ui.div(
                    {
                        "class": "eda-divider eda-divider-vertical eda-divider-vertical-top",
                        "id": "eda-divider-vertical-top",
                        "onmousedown": "window.startEdaResize(event, 'vertical')",
                    }
                ),
                ui.div(
                    {
                        "class": "eda-divider eda-divider-vertical eda-divider-vertical-bottom",
                        "id": "eda-divider-vertical-bottom",
                        "onmousedown": "window.startEdaResize(event, 'vertical')",
                    }
                ),
                ui.div(
                    {
                        "class": "eda-divider eda-divider-horizontal",
                        "id": "eda-divider-horizontal",
                        "onmousedown": "window.startEdaResize(event, 'horizontal')",
                    }
                ),
                ui.card(
                    {
                        "class": "eda-card eda-card-corr",
                        "data-panel": "corr",
                        "onclick": "window.focusEdaEditor('corr-editor', 'corr')",
                    },
                    ui.card_header(
                        ui.div(
                            {"class": "eda-card-head"},
                            ui.span("Correlation heatmap"),
                            ui.div(
                                {"class": "eda-card-actions"},
                                ui.tags.button(
                                    "Edit heatmap",
                                    type="button",
                                    class_="eda-edit-link",
                                    onclick="event.stopPropagation(); window.focusEdaEditor('corr-editor', 'corr');",
                                ),
                                ui.tags.button(
                                    "Export PNG",
                                    type="button",
                                    class_="eda-export-link",
                                    onclick="event.stopPropagation(); window.exportCardPng('corr');",
                                ),
                            ),
                        )
                    ),
                    output_widget("corr_plot"),
                ),
                ui.card(
                    {
                        "class": "eda-card eda-card-summary",
                        "data-panel": "summary",
                        "onclick": "window.focusEdaEditor('summary-editor', 'summary')",
                    },
                    ui.card_header(
                        ui.div(
                            {"class": "eda-card-head"},
                            ui.span("Summary statistics"),
                            ui.div(
                                {"class": "eda-card-actions"},
                                ui.tags.button(
                                    "Edit summary",
                                    type="button",
                                    class_="eda-edit-link",
                                    onclick="event.stopPropagation(); window.focusEdaEditor('summary-editor', 'summary');",
                                ),
                                ui.tags.button(
                                    "Export PNG",
                                    type="button",
                                    class_="eda-export-link",
                                    onclick="event.stopPropagation(); window.exportCardPng('summary');",
                                ),
                            ),
                        )
                    ),
                    ui.output_ui("summary_panel_ui"),
                ),
                ui.card(
                    {
                        "class": "eda-card eda-card-missing",
                        "data-panel": "missing",
                        "onclick": "window.focusEdaEditor('missing-editor', 'missing')",
                    },
                    ui.card_header(
                        ui.div(
                            {"class": "eda-card-head"},
                            ui.span("Missingness by column"),
                            ui.div(
                                {"class": "eda-card-actions"},
                                ui.tags.button(
                                    "Edit missingness",
                                    type="button",
                                    class_="eda-edit-link",
                                    onclick="event.stopPropagation(); window.focusEdaEditor('missing-editor', 'missing');",
                                ),
                                ui.tags.button(
                                    "Export PNG",
                                    type="button",
                                    class_="eda-export-link",
                                    onclick="event.stopPropagation(); window.exportCardPng('missing');",
                                ),
                            ),
                        )
                    ),
                    output_widget("missing_plot"),
                ),
            ),
        ),
    ),
    ui.nav_panel(
        "Export",
        ui.layout_columns(
            ui.card(
                ui.card_header("Project summary"),
                ui.output_text_verbatim("export_summary"),
                ui.download_button("download_processed", "Download processed CSV"),
            ),
            ui.card(
                ui.card_header("How to cite this demo"),
                ui.p(
                    "Include the deployment URL, a short description of the supported workflow, "
                    "and each team member's contribution in the report."
                ),
                ui.tags.ul(
                    ui.tags.li("State the accepted upload formats and built-in demos."),
                    ui.tags.li("Describe the preprocessing options and feature engineering controls."),
                    ui.tags.li("Explain what users can inspect in the EDA tab."),
                ),
            ),
            col_widths=[6, 6],
        ),
    ),
    title=ui.div({"class": "brand-lockup"}, ui.span("STAT 5243"), ui.strong("Project 2 Demo")),
    id="nav",
    fillable=True,
    bg="#14324a",
    inverse=True,
    header=ui.tags.head(
        ui.tags.link(
            rel="icon",
            href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'%3E%3Cdefs%3E%3ClinearGradient id='g' x1='0%25' x2='100%25' y1='0%25' y2='100%25'%3E%3Cstop offset='0%25' stop-color='%2314324a'/%3E%3Cstop offset='100%25' stop-color='%23d85f3c'/%3E%3C/linearGradient%3E%3C/defs%3E%3Crect width='64' height='64' rx='16' fill='%23f4efe6'/%3E%3Crect x='8' y='8' width='48' height='48' rx='12' fill='url(%23g)'/%3E%3Cpath d='M20 40 L28 24 L36 34 L44 18' fill='none' stroke='%23f4efe6' stroke-linecap='round' stroke-linejoin='round' stroke-width='5'/%3E%3C/svg%3E",
        ),
        ui.tags.script(src="https://cdn.jsdelivr.net/npm/html2canvas@1.4.1/dist/html2canvas.min.js"),
        ui.tags.script(
            """
            window.markActiveCard = function(panel) {
              document.querySelectorAll('.eda-card').forEach((card) => {
                card.classList.toggle('is-active', card.dataset.panel === panel);
              });
            };
            window.openDataSourceUpload = function() {
              const dataSourceTab = Array.from(document.querySelectorAll('[role="tab"]')).find((el) => el.textContent.trim() === 'Data Source');
              if (dataSourceTab) dataSourceTab.click();
              setTimeout(() => {
                const uploadRadio = document.querySelector('input[name="source_mode"][value="upload"]');
                if (uploadRadio) {
                  uploadRadio.click();
                  uploadRadio.dispatchEvent(new Event('change', { bubbles: true }));
                }
                setTimeout(() => {
                  const browseButton = document.querySelector('#file_upload') || document.querySelector('input.shiny-input-file');
                  if (browseButton) browseButton.click();
                }, 250);
              }, 250);
            };
            window.focusEdaEditor = function(editorId, panel) {
              window.markActiveCard(panel);
              const editor = document.getElementById(editorId);
              if (editor) {
                editor.scrollIntoView({behavior: 'smooth', block: 'start'});
                editor.classList.add('is-focused');
                setTimeout(() => editor.classList.remove('is-focused'), 1200);
              }
            };
            window.exportCardPng = async function(panel) {
              const card = document.querySelector(`.eda-card[data-panel="${panel}"]`);
              if (!card || !window.html2canvas) return;
              const canvas = await window.html2canvas(card, {
                backgroundColor: null,
                scale: 2,
                useCORS: true,
              });
              const link = document.createElement('a');
              link.download = `${panel}-panel.png`;
              link.href = canvas.toDataURL('image/png');
              link.click();
            };
            window.startEdaResize = function(event, direction) {
              event.preventDefault();
              const canvas = document.getElementById('eda-canvas');
              if (!canvas) return;
              const rect = canvas.getBoundingClientRect();
              const onMove = (moveEvent) => {
                if (direction === 'vertical') {
                  const pct = ((moveEvent.clientX - rect.left) / rect.width) * 100;
                  const clamped = Math.max(30, Math.min(70, pct));
                  canvas.style.setProperty('--eda-left', `${clamped}%`);
                } else {
                  const pct = ((moveEvent.clientY - rect.top) / rect.height) * 100;
                  const clamped = Math.max(28, Math.min(72, pct));
                  canvas.style.setProperty('--eda-top', `${clamped}%`);
                }
                canvas.querySelectorAll('.js-plotly-plot').forEach((plot) => {
                  if (window.Plotly && window.Plotly.Plots) {
                    window.Plotly.Plots.resize(plot);
                  }
                });
              };
              const onUp = () => {
                window.removeEventListener('mousemove', onMove);
                window.removeEventListener('mouseup', onUp);
              };
              window.addEventListener('mousemove', onMove);
              window.addEventListener('mouseup', onUp);
            };
            window.installEdaResizeObserver = function() {
              if (!window.ResizeObserver) return;
              const observer = new ResizeObserver((entries) => {
                entries.forEach((entry) => {
                  entry.target.querySelectorAll('.js-plotly-plot').forEach((plot) => {
                    if (window.Plotly && window.Plotly.Plots) {
                      window.Plotly.Plots.resize(plot);
                    }
                  });
                });
              });
              document.querySelectorAll('.eda-card').forEach((card) => observer.observe(card));
            };
            window.addEventListener('load', () => {
              window.markActiveCard('main');
              window.installEdaResizeObserver();
            });
            """
        ),
        ui.tags.style(
            """
        :root {
          --bg: #f4efe6;
          --panel: rgba(255, 252, 246, 0.88);
          --ink: #163047;
          --muted: #617284;
          --line: rgba(20, 50, 74, 0.12);
          --accent: #d85f3c;
          --accent-soft: #f6d7c5;
          --deep: #14324a;
          --gold: #d6a94f;
        }
        body {
          background:
            radial-gradient(circle at top left, rgba(214, 169, 79, 0.20), transparent 28%),
            radial-gradient(circle at bottom right, rgba(216, 95, 60, 0.14), transparent 22%),
            linear-gradient(135deg, #f4efe6 0%, #f8f5ef 52%, #efe6d7 100%);
          color: var(--ink);
          font-family: "Avenir Next", "Segoe UI", sans-serif;
        }
        h1, h2, h3, .navbar-brand strong {
          font-family: "Palatino", "Georgia", serif;
          letter-spacing: 0.02em;
        }
        .navbar {
          box-shadow: 0 12px 36px rgba(20, 50, 74, 0.18);
        }
        .brand-lockup {
          display: flex;
          gap: 0.8rem;
          align-items: center;
          font-size: 1rem;
          letter-spacing: 0.12em;
          text-transform: uppercase;
        }
        .brand-lockup span {
          color: rgba(255, 255, 255, 0.74);
          font-size: 0.78rem;
        }
        .tab-content {
          padding-top: 1rem;
        }
        .card, .sidebar, .well {
          background: var(--panel);
          border: 1px solid var(--line);
          border-radius: 22px;
          box-shadow: 0 20px 44px rgba(20, 50, 74, 0.08);
          backdrop-filter: blur(8px);
        }
        .card-header {
          border-bottom: 1px solid var(--line);
          font-weight: 700;
          color: var(--deep);
          background: transparent;
        }
        .eda-card-head {
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 1rem;
        }
        .eda-card-actions {
          display: flex;
          align-items: center;
          gap: 0.6rem;
          flex-wrap: wrap;
        }
        .eda-edit-link {
          background: none;
          border: none;
          padding: 0;
          color: var(--accent);
          font-weight: 700;
          text-decoration: none;
          white-space: nowrap;
          cursor: pointer;
        }
        .eda-export-link {
          background: rgba(20, 50, 74, 0.08);
          border: 1px solid rgba(20, 50, 74, 0.12);
          color: var(--deep);
          font-weight: 700;
          border-radius: 999px;
          padding: 0.35rem 0.7rem;
          cursor: pointer;
          white-space: nowrap;
        }
        .eda-canvas {
          --eda-left: 62%;
          --eda-top: 56%;
          display: grid;
          grid-template-columns: minmax(0, calc(var(--eda-left) - 0.5rem)) 1rem minmax(280px, calc(100% - var(--eda-left) - 0.5rem));
          grid-template-rows: minmax(320px, calc(var(--eda-top) - 0.5rem)) 1rem minmax(260px, calc(100% - var(--eda-top) - 0.5rem));
          grid-template-areas:
            "main divider-v-top corr"
            "divider-h divider-h divider-h"
            "summary divider-v-bottom missing";
          gap: 1rem;
          align-items: start;
          min-width: 0;
          min-height: 860px;
          position: relative;
        }
        .eda-card {
          min-width: 0;
          min-height: 0;
          overflow: auto;
          position: relative;
          transition: transform 120ms ease, box-shadow 120ms ease, border-color 120ms ease;
        }
        .eda-card.is-active {
          border-color: rgba(216, 95, 60, 0.55);
          box-shadow: 0 20px 44px rgba(216, 95, 60, 0.12);
          transform: translateY(-1px);
        }
        .eda-card-main { grid-area: main; }
        .eda-card-corr { grid-area: corr; }
        .eda-card-summary { grid-area: summary; min-height: 0; }
        .eda-card-missing { grid-area: missing; min-height: 0; }
        .eda-divider {
          background: linear-gradient(180deg, rgba(214, 169, 79, 0.2), rgba(216, 95, 60, 0.45), rgba(20, 50, 74, 0.2));
          border-radius: 999px;
          cursor: col-resize;
          align-self: stretch;
          justify-self: stretch;
          opacity: 0.8;
        }
        .eda-divider-vertical {
          width: 0.5rem;
          justify-self: center;
          height: 100%;
        }
        .eda-divider-vertical-top { grid-area: divider-v-top; }
        .eda-divider-vertical-bottom { grid-area: divider-v-bottom; }
        .eda-divider-horizontal {
          grid-area: divider-h;
          height: 0.5rem;
          cursor: row-resize;
          align-self: center;
          background: linear-gradient(90deg, rgba(214, 169, 79, 0.2), rgba(216, 95, 60, 0.45), rgba(20, 50, 74, 0.2));
        }
        .editor-block {
          padding-bottom: 1rem;
          margin-bottom: 1rem;
          border-bottom: 1px dashed rgba(20, 50, 74, 0.12);
        }
        .editor-block.is-focused {
          background: rgba(216, 95, 60, 0.08);
          border-radius: 18px;
          padding: 0.75rem;
          margin: -0.75rem 0 1rem;
        }
        .hero-panel {
          display: grid;
          grid-template-columns: 1.5fr 1fr;
          gap: 1.2rem;
          margin-bottom: 1.25rem;
          animation: rise 0.65s ease-out;
        }
        .hero-copy, .hero-card {
          border-radius: 28px;
          padding: 2rem;
          background: linear-gradient(145deg, rgba(255,255,255,0.86), rgba(246,231,215,0.94));
          border: 1px solid rgba(20, 50, 74, 0.10);
          box-shadow: 0 22px 48px rgba(20, 50, 74, 0.10);
        }
        .hero-copy h1 {
          font-size: clamp(2.2rem, 4vw, 3.5rem);
          margin-bottom: 0.75rem;
        }
        .hero-copy p {
          font-size: 1.06rem;
          color: var(--muted);
          max-width: 48rem;
        }
        .hero-badges {
          display: flex;
          flex-wrap: wrap;
          gap: 0.65rem;
          margin-top: 1rem;
        }
        .hero-badges span {
          background: rgba(20, 50, 74, 0.08);
          border: 1px solid rgba(20, 50, 74, 0.10);
          color: var(--deep);
          padding: 0.55rem 0.9rem;
          border-radius: 999px;
          font-size: 0.9rem;
        }
        .btn-primary, .btn-default {
          border-radius: 999px;
          padding: 0.55rem 1rem;
        }
        .btn-primary {
          background: var(--accent);
          border-color: var(--accent);
        }
        .form-control, .selectize-input {
          border-radius: 14px;
          border-color: rgba(20, 50, 74, 0.14);
        }
        .shiny-file-input-progress {
          display: none !important;
        }
        .shiny-data-grid {
          border-radius: 14px;
          overflow: hidden;
        }
        @keyframes rise {
          from { opacity: 0; transform: translateY(10px); }
          to { opacity: 1; transform: translateY(0); }
        }
        @media (max-width: 900px) {
          .hero-panel {
            grid-template-columns: 1fr;
          }
          .eda-canvas {
            grid-template-columns: 1fr;
            grid-template-areas:
              "main"
              "corr"
              "summary"
              "missing";
            grid-template-rows: auto;
            min-height: auto;
          }
          .eda-divider { display: none; }
        }
        """
        ),
    ),
)


def server(input, output, session):
    feature_specs = reactive.Value([])

    @reactive.calc
    def raw_data() -> pd.DataFrame:
        mode = input.source_mode()
        if mode == "demo":
            return SAMPLE_DATASETS[input.demo_dataset()].copy()

        uploaded = input.file_upload()
        req(uploaded)
        try:
            return parse_uploaded_file(uploaded[0])
        except Exception as err:
            return pd.DataFrame({"Upload error": [str(err)]})

    @reactive.calc
    def processed_data() -> pd.DataFrame:
        df = raw_data().copy()
        if "Upload error" in df.columns:
            return df

        df = standardize_text_columns(df)
        if input.drop_duplicates():
            df = df.drop_duplicates()
        df = handle_missing(df, input.missing_strategy())
        df = handle_outliers(df, input.outlier_strategy())
        df = scale_numeric(df, input.scaling_strategy())
        if input.encode_categories():
            text_cols = list(df.select_dtypes(exclude=[np.number, "datetime"]).columns)
            if text_cols:
                df = pd.get_dummies(df, columns=text_cols, dummy_na=True)
        return df

    @reactive.calc
    def featured_data() -> pd.DataFrame:
        df = processed_data().copy()
        if "Upload error" in df.columns:
            return df
        return apply_feature_engineering(df, feature_specs())

    @reactive.effect
    @reactive.event(input.apply_feature)
    def _add_feature():
        df = processed_data()
        if "Upload error" in df.columns:
            return

        op = input.feature_operation()
        col_a = input.feature_col_a()
        col_b = input.feature_col_b() if op in {"sum", "difference", "product", "ratio"} else None
        name = input.new_feature_name().strip() or "engineered_feature"
        current = list(feature_specs())
        current.append(FeatureSpec(op, col_a, col_b, name, input.bin_count()))
        feature_specs.set(current)

    @reactive.effect
    @reactive.event(input.reset_features)
    def _reset_features():
        feature_specs.set([])

    @render.text
    def source_status():
        df = raw_data()
        label = input.dataset_label().strip() or "Dataset"
        if "Upload error" in df.columns:
            return f"{label}\nStatus: upload parsing issue"
        return (
            f"{label}\n"
            f"Rows: {len(df):,}\n"
            f"Columns: {len(df.columns):,}\n"
            f"Source mode: {input.source_mode()}"
        )

    @render.data_frame
    def profile_table():
        return render.DataGrid(build_profile(raw_data()), width="100%")

    @render.data_frame
    def raw_preview():
        return render.DataGrid(raw_data().head(18), filters=True, width="100%")

    @render.data_frame
    def cleaning_profile():
        before = build_profile(raw_data()).rename(columns={"value": "before"})
        after = build_profile(processed_data()).rename(columns={"value": "after"})
        merged = before.merge(after, on="metric", how="left")
        return render.DataGrid(merged, width="100%")

    @render.data_frame
    def processed_preview():
        return render.DataGrid(processed_data().head(18), filters=True, width="100%")

    @render.ui
    def feature_column_ui():
        df = processed_data()
        numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
        all_cols = list(df.columns)
        if not all_cols:
            return ui.p("No columns available.")

        op = input.feature_operation()
        choices_a = numeric_cols if op in {"sum", "difference", "product", "ratio", "log", "bin"} else all_cols
        col_a_default = choices_a[0] if choices_a else None
        col_b_default = numeric_cols[1] if len(numeric_cols) > 1 else (numeric_cols[0] if numeric_cols else None)

        controls = [
            ui.input_select("feature_col_a", "Column A", choices=choices_a, selected=col_a_default),
        ]
        if op in {"sum", "difference", "product", "ratio"}:
            controls.append(
                ui.input_select("feature_col_b", "Column B", choices=numeric_cols, selected=col_b_default)
            )
        else:
            controls.append(ui.p({"class": "text-muted"}, "Column B is not needed for this operation."))
        return ui.TagList(*controls)

    @render.data_frame
    def feature_table():
        specs = feature_specs()
        if not specs:
            return render.DataGrid(pd.DataFrame({"status": ["No engineered features yet."]}), width="100%")
        table = pd.DataFrame(
            [
                {
                    "operation": spec.operation,
                    "column_a": spec.col_a,
                    "column_b": spec.col_b or "-",
                    "new_name": spec.new_name,
                    "bins": spec.bins,
                }
                for spec in specs
            ]
        )
        return render.DataGrid(table, width="100%")

    @render.data_frame
    def featured_preview():
        return render.DataGrid(featured_data().head(18), filters=True, width="100%")

    @reactive.effect
    def _update_eda_inputs():
        df = featured_data()
        if df.empty:
            return

        numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
        all_cols = list(df.columns)
        categorical_cols = low_cardinality_columns(df, max_unique=25)

        x_default = pick_dimension_default(df) or pick_numeric_default(df, 0) or all_cols[0]
        y_default = pick_numeric_default(df, 1) or pick_numeric_default(df, 0) or all_cols[0]
        metric_default = pick_numeric_default(df, 0) or "count"
        color_default = pick_categorical_default(df) or "None"
        filter_default = pick_filter_default(df) or all_cols[0]

        current_x = input.eda_x() if input.eda_x() in all_cols else x_default
        current_y = input.eda_y() if input.eda_y() in (numeric_cols or all_cols) else y_default
        metric_choices = {"count": "Row count", **{col: col for col in numeric_cols}}
        current_metric = input.eda_metric() if input.eda_metric() in metric_choices else metric_default
        color_choices = ["None"] + categorical_cols
        current_color = input.eda_color() if input.eda_color() in color_choices else color_default
        corr_focus_choices = ["Auto"] + numeric_cols
        current_corr_focus = input.corr_focus_var() if input.corr_focus_var() in corr_focus_choices else "Auto"
        current_filter = input.filter_col() if input.filter_col() in all_cols else filter_default

        ui.update_select("eda_x", choices={col: col for col in all_cols}, selected=current_x, session=session)
        ui.update_select(
            "eda_y",
            choices={col: col for col in (numeric_cols or all_cols)},
            selected=current_y,
            session=session,
        )
        ui.update_select("eda_metric", choices=metric_choices, selected=current_metric, session=session)
        ui.update_select(
            "eda_color",
            choices={choice: choice for choice in color_choices},
            selected=current_color,
            session=session,
        )
        ui.update_select(
            "corr_focus_var",
            choices={choice: choice for choice in corr_focus_choices},
            selected=current_corr_focus,
            session=session,
        )
        ui.update_select("filter_col", choices={col: col for col in all_cols}, selected=current_filter, session=session)

    @render.ui
    def filter_value_ui():
        df = featured_data()
        col = input.filter_col()
        if col is None or col not in df.columns:
            return ui.p("Select a filter column.")

        if pd.api.types.is_numeric_dtype(df[col]):
            values = df[col].dropna()
            if values.empty:
                return ui.p("No numeric values available for filtering.")
            lower = float(values.min())
            upper = float(values.max())
            return ui.input_slider("filter_range", "Numeric range", min=lower, max=upper, value=(lower, upper))

        choices = [str(v) for v in sorted(df[col].dropna().astype(str).unique().tolist())]
        if len(choices) > 200:
            return ui.TagList(
                ui.p(
                    {"class": "text-muted"},
                    f'"{col}" has {len(choices):,} unique values. Use a lower-cardinality column for dropdown filtering.',
                ),
                ui.input_text("filter_text", "Text contains", ""),
            )
        selected = choices[: min(4, len(choices))]
        return ui.input_selectize(
            "filter_values",
            "Allowed values",
            choices=choices,
            selected=selected,
            multiple=True,
        )

    @reactive.calc
    def filtered_eda_data() -> pd.DataFrame:
        df = featured_data().copy()
        col = input.filter_col()
        if col is None or col not in df.columns:
            return df

        if pd.api.types.is_numeric_dtype(df[col]):
            selected = input.filter_range()
            if selected is None:
                return df
            low, high = selected
            return df[(df[col].isna()) | ((df[col] >= low) & (df[col] <= high))]

        selected_values = input.filter_values()
        text_query = input.filter_text()
        if text_query:
            return df[df[col].astype(str).str.contains(text_query, case=False, na=False)]
        if not selected_values:
            return df
        return df[df[col].astype(str).isin(selected_values)]

    @reactive.calc
    def missingness_data() -> pd.DataFrame:
        source = input.missing_data_source()
        if source == "processed":
            return processed_data()
        if source == "featured":
            return featured_data()
        if source == "filtered":
            return filtered_eda_data()
        return raw_data()

    @render.data_frame
    def summary_stats():
        if input.summary_chart_type() != "table":
            return render.DataGrid(pd.DataFrame({"info": ["Switch summary panel to table mode to view the grid."]}), width="100%")
        return render.DataGrid(build_summary_stats(filtered_eda_data()), width="100%")

    @render.ui
    def summary_panel_ui():
        if input.summary_chart_type() == "bar":
            return output_widget("summary_plot")
        return ui.output_data_frame("summary_stats")

    @render_widget
    def summary_plot():
        stats = build_summary_stats(filtered_eda_data())
        if "variable" not in stats.columns:
            return px.bar(title="No numeric summary available.")
        chart = stats[["variable", "mean"]].sort_values("mean", ascending=False).head(12)
        fig = px.bar(
            chart,
            x="variable",
            y="mean",
            color="mean",
            color_continuous_scale=["#f6d7c5", "#d85f3c"],
            title="Average value by numeric feature",
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=20, r=20, t=40, b=20),
        )
        return fig

    @render_widget
    def missing_plot():
        df = missingness_data()
        missing = (
            df.isna()
            .sum()
            .reset_index()
            .rename(columns={"index": "column", 0: "missing"})
        )
        ascending = input.missing_sort() == "asc"
        missing = missing.sort_values("missing", ascending=ascending).head(input.missing_top_n())
        if input.missing_chart_type() == "donut":
            fig = px.pie(
                missing,
                names="column",
                values="missing",
                hole=0.55,
                title="Missing-value share by column",
            )
        else:
            fig = px.bar(
                missing,
                x="column",
                y="missing",
                color="missing",
                color_continuous_scale=["#f6d7c5", "#d85f3c"],
                title="Missing values per column",
            )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=20, r=20, t=50, b=20),
        )
        return fig

    @render_widget
    def main_plot():
        df = filtered_eda_data()
        if df.empty:
            return px.scatter(title="No data available after filtering.")

        plot_type = input.eda_plot_type()
        x = input.eda_x()
        y = input.eda_y()
        metric = input.eda_metric()
        agg = input.eda_agg()
        color = None if input.eda_color() in {None, "None"} else input.eda_color()
        top_n = input.eda_top_n()
        bins = input.eda_bins()

        if color and color not in df.columns:
            color = None

        def aggregated_frame() -> pd.DataFrame:
            group_cols = [x] + ([color] if color and color != x else [])
            if metric == "count" or agg == "count":
                out = df.groupby(group_cols, dropna=False).size().reset_index(name="value")
            else:
                out = (
                    df.groupby(group_cols, dropna=False)[metric]
                    .agg(agg)
                    .reset_index(name="value")
                )
            if x in out.columns:
                order = (
                    out.groupby(x, dropna=False)["value"]
                    .sum()
                    .sort_values(ascending=False)
                    .head(top_n)
                    .index
                )
                out = out[out[x].isin(order)]
                out[x] = pd.Categorical(out[x], categories=list(order), ordered=True)
                out = out.sort_values([x] + ([color] if color and color != x else []))
            return out

        if plot_type == "scatter":
            scatter_df = df
            if len(scatter_df) > 4000:
                scatter_df = scatter_df.sample(4000, random_state=42)
            fig = px.scatter(
                scatter_df,
                x=x,
                y=y,
                color=color,
                hover_data=list(scatter_df.columns[:6]),
                title=f"{y} vs {x}",
            )
        elif plot_type == "box":
            box_x = x
            if pd.api.types.is_numeric_dtype(df[x]) and color:
                box_x = color
            fig = px.box(
                df,
                x=box_x,
                y=y,
                color=color if color and color != box_x else None,
                points="outliers",
                title=f"Distribution of {y} by {box_x}",
            )
        elif plot_type == "line":
            grouped = aggregated_frame()
            fig = px.line(
                grouped,
                x=x,
                y="value",
                color=color if color and color != x else None,
                markers=True,
                title=f"{agg.title()} of {metric} by {x}",
            )
        elif plot_type == "bar":
            grouped = aggregated_frame()
            fig = px.bar(
                grouped,
                x=x,
                y="value",
                color=color if color and color != x else None,
                barmode="group",
                title=f"{agg.title()} of {metric} by {x}",
            )
        else:
            if pd.api.types.is_numeric_dtype(df[x]):
                fig = px.histogram(
                    df,
                    x=x,
                    color=color,
                    marginal="box",
                    nbins=bins,
                    title=f"Distribution of {x}",
                )
            else:
                grouped = df.groupby(x, dropna=False).size().reset_index(name="value")
                grouped = grouped.sort_values("value", ascending=False).head(top_n)
                fig = px.bar(
                    grouped,
                    x=x,
                    y="value",
                    title=f"Top {top_n} categories in {x}",
                )

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=20, r=20, t=40, b=20),
        )
        return fig

    @render_widget
    def corr_plot():
        numeric = filtered_eda_data().select_dtypes(include=[np.number])
        if numeric.shape[1] < 2:
            return px.imshow(pd.DataFrame([[1.0]], columns=["N/A"], index=["N/A"]), text_auto=True)
        top_n = min(input.corr_top_n(), numeric.shape[1])
        selected_cols = (
            numeric.var(numeric_only=True)
            .sort_values(ascending=False)
            .head(top_n)
            .index
            .tolist()
        )
        corr = numeric[selected_cols].corr(method=input.corr_method(), numeric_only=True).round(2)
        if input.corr_chart_type() == "bars":
            focus = input.corr_focus_var()
            focus_var = focus if focus in corr.columns else corr.columns[0]
            bars = (
                corr[focus_var]
                .drop(labels=[focus_var], errors="ignore")
                .abs()
                .sort_values(ascending=False)
                .head(max(1, top_n - 1))
                .reset_index()
                .rename(columns={"index": "variable", focus_var: "abs_corr"})
            )
            fig = px.bar(
                bars,
                x="variable",
                y="abs_corr",
                color="abs_corr",
                color_continuous_scale=["#f6d7c5", "#d85f3c"],
                title=f"Strongest correlations with {focus_var}",
            )
        else:
            fig = px.imshow(
                corr,
                text_auto=True,
                aspect="auto",
                color_continuous_scale=["#14324a", "#f4efe6", "#d85f3c"],
            )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=20, r=20, t=40, b=20),
        )
        return fig

    @render.text
    def export_summary():
        df = featured_data()
        profile = build_profile(df)
        lines = [f"{row.metric}: {row.value}" for row in profile.itertuples()]
        lines.append(f"Engineered features: {len(feature_specs())}")
        lines.append(f"Dataset label: {input.dataset_label().strip() or 'Project 2 demo'}")
        return "\n".join(lines)

    @render.download(filename="processed_dataset.csv")
    def download_processed():
        df = featured_data()
        buffer = io.StringIO()
        df.to_csv(buffer, index=False)
        yield buffer.getvalue()


app = App(app_ui, server, static_assets={"/favicon.ico": BASE_DIR / "assets" / "favicon.ico"})
