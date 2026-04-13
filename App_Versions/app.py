from __future__ import annotations

import csv
import io
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
from shiny import App, reactive, render, req, ui
from shinywidgets import output_widget, render_widget

from core import (
    SAMPLE_DATASETS,
    FeatureSpec,
    apply_feature_engineering,
    build_profile,
    build_summary_stats,
    handle_missing,
    handle_outliers,
    low_cardinality_columns,
    parse_uploaded_file,
    pick_categorical_default,
    pick_dimension_default,
    pick_numeric_default,
    scale_numeric,
    standardize_text_columns,
)

APP_VERSION = "dashboard_b_internal"
BASE_DIR = Path(__file__).parent
LOG_DIR = BASE_DIR / "logs"
USAGE_LOG = LOG_DIR / "usage_log.csv"
FEEDBACK_LOG = LOG_DIR / "feedback_log.csv"
LOG_DIR.mkdir(parents=True, exist_ok=True)


# ---------- logging helpers ----------
def now_utc() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def append_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ---------- UI ----------
app_ui = ui.page_navbar(
    ui.nav_panel(
        "Overview",
        ui.div(
            {"class": "hero-panel"},
            ui.div(
                {"class": "hero-copy"},
                ui.h1("DataCanvas Studio"),
                ui.p(
                   "An interactive dashboard for data upload, cleaning, feature engineering, and exploratory analysis."
                ),
                ui.div(
                    {"class": "hero-badges"},
                    ui.span("Interactive workflow"),
                    ui.span("Data cleaning"),
                    ui.span("Feature engineering"),
                    ui.span("Exploratory analysis"),
                ),
            ),
            ui.div(
                {"class": "hero-card"},
                ui.h3("Quick workflow"),
                ui.tags.ol(
                    ui.tags.li("Choose a demo dataset or upload your own file."),
                    ui.tags.li("Apply a few cleaning settings and optional features."),
                    ui.tags.li("Inspect charts, download the result, and submit feedback."),
                ),
                ui.hr(),
                ui.p({"class": "text-muted"}, "Use this dashboard to prepare, explore, and export your data."),
            ),
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
                        "demo": "Built-in demo dataset",
                        "upload": "Upload my file",
                    },
                    selected="demo",
                ),
                ui.panel_conditional(
                    "input.source_mode === 'demo'",
                    ui.input_select(
                        "demo_dataset",
                        "Demo dataset",
                        choices=list(SAMPLE_DATASETS.keys()),
                    ),
                ),
                ui.panel_conditional(
                    "input.source_mode === 'upload'",
                    ui.input_file(
                        "file_upload",
                        "Upload dataset",
                        accept=[".csv", ".xlsx", ".xls", ".json", ".rds"],
                    ),
                ),
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
                        "none": "Keep as-is",
                        "drop": "Drop rows",
                        "mean": "Impute mean/mode",
                        "median": "Impute median/mode",
                    },
                    selected="mean",
                ),
                ui.input_switch("drop_duplicates", "Remove duplicates", True),
                ui.input_select(
                    "outlier_strategy",
                    "Outliers",
                    {
                        "none": "No treatment",
                        "cap": "Cap using IQR",
                        "remove": "Remove extreme rows",
                    },
                    selected="cap",
                ),
                width=320,
            ),
            ui.layout_columns(
                ui.card(
                    ui.card_header("Before vs after"),
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
                    "Operation",
                    {
                        "sum": "A + B",
                        "ratio": "A / B",
                        "log": "log(1 + A)",
                        "bin": "Bin A",
                    },
                    selected="sum",
                ),
                ui.output_ui("feature_column_ui"),
                ui.input_text("new_feature_name", "New feature name", "engineered_feature"),
                ui.input_numeric("bin_count", "Number of bins", 4, min=2, max=12),
                ui.input_action_button("apply_feature", "Add feature", class_="btn-primary"),
                ui.input_action_button("reset_features", "Reset features"),
                width=320,
            ),
            ui.layout_columns(
                ui.card(
                    ui.card_header("Feature list"),
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
                ui.input_select(
                    "eda_plot_type",
                    "Chart type",
                    {
                        "bar": "Bar",
                        "line": "Line",
                        "histogram": "Histogram",
                        "scatter": "Scatter",
                        "box": "Box",
                    },
                    selected="bar",
                ),
                ui.input_select("eda_x", "X variable", choices=[""], selected=""),
                ui.input_select("eda_y", "Y variable", choices=[""], selected=""),
                ui.input_select("eda_metric", "Metric", choices={"count": "Row count"}, selected="count"),
                ui.input_select("eda_color", "Color group", choices={"None": "None"}, selected="None"),
                ui.input_select("filter_col", "Filter column", choices=[""], selected=""),
                ui.output_ui("filter_value_ui"),
                ui.input_numeric("eda_top_n", "Top categories", 10, min=3, max=30),
                ui.input_numeric("eda_bins", "Histogram bins", 20, min=5, max=60),
                width=340,
            ),
            ui.div(
                {"class": "simple-eda-grid"},
                ui.card(
                    {"class": "simple-card"},
                    ui.card_header("Main chart"),
                    output_widget("main_plot"),
                ),
                ui.card(
                    {"class": "simple-card"},
                    ui.card_header("Summary statistics"),
                    ui.output_data_frame("summary_stats"),
                ),
                ui.card(
                    {"class": "simple-card"},
                    ui.card_header("Correlation heatmap"),
                    output_widget("corr_plot"),
                ),
                ui.card(
                    {"class": "simple-card"},
                    ui.card_header("Missing values"),
                    output_widget("missing_plot"),
                ),
            ),
        ),
    ),
    ui.nav_panel(
        "Feedback",
        ui.layout_columns(
            ui.card(
                ui.card_header("User feedback"),
                ui.input_slider("ease_score", "Ease of use", min=1, max=7, value=5),
                ui.input_slider("clarity_score", "Clarity of layout", min=1, max=7, value=5),
                ui.input_text_area("feedback_text", "Comments", rows=5, placeholder="What felt easier or harder?"),
                ui.input_action_button("submit_feedback", "Submit feedback", class_="btn-primary"),
                ui.output_text_verbatim("feedback_status"),
            ),
            ui.card(
                ui.card_header("Session info"),
                ui.output_text_verbatim("session_info"),
            ),
            col_widths=[7, 5],
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
                ui.card_header("Usage notes"),
                ui.tags.ul(
                    ui.tags.li("Review the cleaned data before export."),
                    ui.tags.li("Download the processed dataset for later analysis."),
                    ui.tags.li("Use the feedback section to record your experience."),
                ),
            ),
            col_widths=[6, 6],
        ),
    ),
    title=ui.div({"class": "brand-lockup"}, ui.span("STAT 5243"), ui.strong("DataCanvas Studio")),
    id="nav",
    fillable=True,
    bg="#14324a",
    inverse=True,
    header=ui.tags.head(
        ui.tags.style(
            """
            :root {
              --bg: #fbfaf8;
              --panel: rgba(255, 255, 255, 0.97);
              --ink: #163047;
              --muted: #617284;
              --line: rgba(20, 50, 74, 0.12);
              --accent: #d85f3c;
              --accent-soft: #f6d7c5;
              --deep: #14324a;
              --gold: #d6a94f;
            }
            body {
              background: linear-gradient(135deg, #fbfaf8 0%, #fdfcfb 55%, #f7f5ef 100%);
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
            .hero-panel {
              display: grid;
              grid-template-columns: 1.4fr 1fr;
              gap: 1.2rem;
              margin-bottom: 1.25rem;
            }
            .hero-copy, .hero-card {
              border-radius: 28px;
              padding: 2rem;
              background: linear-gradient(145deg, rgba(255,255,255,0.98), rgba(250,246,240,0.98));
              border: 1px solid rgba(20, 50, 74, 0.10);
              box-shadow: 0 22px 48px rgba(20, 50, 74, 0.10);
            }
            .hero-copy h1 {
              font-size: clamp(2.2rem, 4vw, 3.4rem);
              margin-bottom: 0.75rem;
            }
            .hero-copy p {
              font-size: 1.03rem;
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
            .simple-eda-grid {
              display: grid;
              grid-template-columns: 1.2fr 0.9fr;
              gap: 1rem;
              align-items: start;
            }
            .simple-card {
              min-width: 0;
              overflow: auto;
            }
            @media (max-width: 960px) {
              .hero-panel,
              .simple-eda-grid {
                grid-template-columns: 1fr;
              }
            }
            """
        )
    ),
)


# ---------- server ----------
def server(input, output, session):
    feature_specs = reactive.Value([])
    session_id = uuid.uuid4().hex[:10]
    started_at = datetime.utcnow()
    feedback_message = reactive.Value("No feedback submitted yet.")

    append_row(
        USAGE_LOG,
        {
            "timestamp_utc": now_utc(),
            "session_id": session_id,
            "event": "session_start",
            "app_version": APP_VERSION,
        },
    )

    @reactive.calc
    def raw_data() -> pd.DataFrame:
        mode = input.source_mode()
        if mode == "demo":
            return SAMPLE_DATASETS[input.demo_dataset()].copy()

        uploaded = input.file_upload()
        req(uploaded)
        try:
            file_info = uploaded[0]
            return parse_uploaded_file(file_info["datapath"], file_info["name"])
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
        if "Upload error" in df.columns or df.empty:
            return

        op = input.feature_operation()
        col_a = input.feature_col_a()
        col_b = input.feature_col_b() if op in {"sum", "ratio"} else None
        name = input.new_feature_name().strip() or "engineered_feature"
        current = list(feature_specs())
        current.append(FeatureSpec(op, col_a, col_b, name, int(input.bin_count() or 4)))
        feature_specs.set(current)

        append_row(
            USAGE_LOG,
            {
                "timestamp_utc": now_utc(),
                "session_id": session_id,
                "event": "feature_added",
                "app_version": APP_VERSION,
                "operation": op,
                "new_name": name,
            },
        )

    @reactive.effect
    @reactive.event(input.reset_features)
    def _reset_features():
        feature_specs.set([])
        append_row(
            USAGE_LOG,
            {
                "timestamp_utc": now_utc(),
                "session_id": session_id,
                "event": "features_reset",
                "app_version": APP_VERSION,
            },
        )

    @render.text
    def source_status():
        df = raw_data()
        label = "Dataset"
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
        if not numeric_cols:
            return ui.p("No numeric columns available.")

        op = input.feature_operation()
        controls: list[Any] = [
            ui.input_select("feature_col_a", "Column A", choices=numeric_cols, selected=numeric_cols[0])
        ]
        if op in {"sum", "ratio"}:
            col_b_default = numeric_cols[1] if len(numeric_cols) > 1 else numeric_cols[0]
            controls.append(ui.input_select("feature_col_b", "Column B", choices=numeric_cols, selected=col_b_default))
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
        filter_default = pick_dimension_default(df) or all_cols[0]

        ui.update_select("eda_x", choices={col: col for col in all_cols}, selected=x_default, session=session)
        ui.update_select(
            "eda_y",
            choices={col: col for col in (numeric_cols or all_cols)},
            selected=y_default,
            session=session,
        )
        ui.update_select(
            "eda_metric",
            choices={"count": "Row count", **{col: col for col in numeric_cols}},
            selected=metric_default,
            session=session,
        )
        ui.update_select(
            "eda_color",
            choices={choice: choice for choice in (["None"] + categorical_cols)},
            selected=color_default,
            session=session,
        )
        ui.update_select(
            "filter_col",
            choices={col: col for col in all_cols},
            selected=filter_default,
            session=session,
        )

    @render.ui
    def filter_value_ui():
        df = featured_data()
        col = input.filter_col()
        if not col or col not in df.columns:
            return ui.p("Choose a filter column.")

        if pd.api.types.is_numeric_dtype(df[col]):
            values = df[col].dropna()
            if values.empty:
                return ui.p("No numeric values available.")
            lower = float(values.min())
            upper = float(values.max())
            return ui.input_slider("filter_range", "Range", min=lower, max=upper, value=(lower, upper))

        choices = [str(v) for v in sorted(df[col].dropna().astype(str).unique().tolist())]
        if len(choices) > 60:
            return ui.input_text("filter_text", "Text contains", "")
        return ui.input_selectize(
            "filter_values",
            "Allowed values",
            choices=choices,
            selected=choices[: min(4, len(choices))],
            multiple=True,
        )

    @reactive.calc
    def filtered_eda_data() -> pd.DataFrame:
        df = featured_data().copy()
        col = input.filter_col()
        if not col or col not in df.columns:
            return df

        if pd.api.types.is_numeric_dtype(df[col]):
            try:
                selected = input.filter_range()
            except Exception:
                return df
            if selected is None:
                return df
            low, high = selected
            return df[(df[col].isna()) | ((df[col] >= low) & (df[col] <= high))]

        try:
            text_query = input.filter_text()
        except Exception:
            text_query = None

        if text_query:
            return df[df[col].astype(str).str.contains(text_query, case=False, na=False)]

        try:
            selected_values = input.filter_values()
        except Exception:
            selected_values = None

        if not selected_values:
            return df
        return df[df[col].astype(str).isin(selected_values)]

    @render.data_frame
    def summary_stats():
        return render.DataGrid(build_summary_stats(filtered_eda_data()), width="100%")

    @render_widget
    def main_plot():
        df = filtered_eda_data()
        if df.empty:
            return px.scatter(title="No data available after filtering.")

        plot_type = input.eda_plot_type()
        x = input.eda_x()
        y = input.eda_y()
        metric = input.eda_metric()
        color = None if input.eda_color() in {None, "None"} else input.eda_color()
        top_n = int(input.eda_top_n() or 10)
        bins = int(input.eda_bins() or 20)

        if color and color not in df.columns:
            color = None

        def aggregated_frame() -> pd.DataFrame:
            group_cols = [x] + ([color] if color and color != x else [])
            if metric == "count":
                out = df.groupby(group_cols, dropna=False).size().reset_index(name="value")
            else:
                out = df.groupby(group_cols, dropna=False)[metric].mean().reset_index(name="value")
            if x in out.columns:
                order = out.groupby(x, dropna=False)["value"].sum().sort_values(ascending=False).head(top_n).index
                out = out[out[x].isin(order)]
            return out

        if plot_type == "scatter":
            scatter_df = df if len(df) <= 4000 else df.sample(4000, random_state=42)
            fig = px.scatter(scatter_df, x=x, y=y, color=color, title=f"{y} vs {x}")
        elif plot_type == "box":
            fig = px.box(df, x=x, y=y, color=color if color and color != x else None, points="outliers", title=f"{y} by {x}")
        elif plot_type == "line":
            grouped = aggregated_frame()
            fig = px.line(grouped, x=x, y="value", color=color if color and color != x else None, markers=True, title=f"Trend by {x}")
        elif plot_type == "bar":
            grouped = aggregated_frame()
            fig = px.bar(grouped, x=x, y="value", color=color if color and color != x else None, barmode="group", title=f"Comparison by {x}")
        else:
            if pd.api.types.is_numeric_dtype(df[x]):
                fig = px.histogram(df, x=x, color=color, nbins=bins, marginal="box", title=f"Distribution of {x}")
            else:
                grouped = df.groupby(x, dropna=False).size().reset_index(name="value").sort_values("value", ascending=False).head(top_n)
                fig = px.bar(grouped, x=x, y="value", title=f"Top {top_n} categories in {x}")

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
        selected_cols = numeric.var(numeric_only=True).sort_values(ascending=False).head(min(8, numeric.shape[1])).index.tolist()
        corr = numeric[selected_cols].corr(numeric_only=True).round(2)
        fig = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale=["#14324a", "#f4efe6", "#d85f3c"])
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=20, r=20, t=30, b=20),
        )
        return fig

    @render_widget
    def missing_plot():
        df = filtered_eda_data()
        missing = df.isna().sum().reset_index().rename(columns={"index": "column", 0: "missing"})
        missing = missing.sort_values("missing", ascending=False).head(12)
        fig = px.bar(missing, x="column", y="missing", color="missing", color_continuous_scale=["#f6d7c5", "#d85f3c"], title="Missing values per column")
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=20, r=20, t=40, b=20),
        )
        return fig

    @render.text
    def session_info():
        elapsed_seconds = int((datetime.utcnow() - started_at).total_seconds())
        return (
            f"Session ID: {session_id}\n"
            f"Elapsed seconds: {elapsed_seconds}\n"
            f"Current features: {len(feature_specs())}"
        )

    @reactive.effect
    @reactive.event(input.submit_feedback)
    def _submit_feedback():
        elapsed_seconds = int((datetime.utcnow() - started_at).total_seconds())
        row = {
            "timestamp_utc": now_utc(),
            "session_id": session_id,
            "app_version": APP_VERSION,
            "ease_score": int(input.ease_score()),
            "clarity_score": int(input.clarity_score()),
            "elapsed_seconds": elapsed_seconds,
            "source_mode": input.source_mode(),
            "rows_after_cleaning": len(processed_data()),
            "engineered_features": len(feature_specs()),
            "comments": input.feedback_text().strip(),
        }
        append_row(FEEDBACK_LOG, row)
        append_row(
            USAGE_LOG,
            {
                "timestamp_utc": now_utc(),
                "session_id": session_id,
                "event": "feedback_submitted",
                "app_version": APP_VERSION,
                "elapsed_seconds": elapsed_seconds,
            },
        )
        feedback_message.set("Feedback saved. Thank you.")

    @render.text
    def feedback_status():
        return feedback_message()

    @render.text
    def export_summary():
        df = featured_data()
        profile = build_profile(df)
        lines = [f"{row.metric}: {row.value}" for row in profile.itertuples()]
        lines.append(f"Engineered features: {len(feature_specs())}")
        return "\n".join(lines)

    @render.download(filename="processed_dataset.csv")
    def download_processed():
        df = featured_data()
        append_row(
            USAGE_LOG,
            {
                "timestamp_utc": now_utc(),
                "session_id": session_id,
                "event": "download_processed",
                "app_version": APP_VERSION,
                "rows": len(df),
                "columns": len(df.columns),
            },
        )
        buffer = io.StringIO()
        df.to_csv(buffer, index=False)
        yield buffer.getvalue()


app = App(app_ui, server)
