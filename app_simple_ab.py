from __future__ import annotations

import csv
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
from shiny import App, reactive, render, ui
from shinywidgets import output_widget, render_widget

APP_VERSION = "dashboard_b_internal"
BASE_DIR = Path(__file__).parent
LOG_DIR = BASE_DIR / "logs"
USAGE_LOG = LOG_DIR / "usage_log.csv"
FEEDBACK_LOG = LOG_DIR / "feedback_log.csv"
LOG_DIR.mkdir(parents=True, exist_ok=True)


# ---------- helpers ----------
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


def log_usage(session_id: str, event_name: str, **kwargs: Any) -> None:
    payload = {
        "timestamp_utc": now_utc(),
        "session_id": session_id,
        "app_version": APP_VERSION,
        "event_name": event_name,
    }
    payload.update(kwargs)
    append_row(USAGE_LOG, payload)



def build_demo_df() -> pd.DataFrame:
    df = px.data.tips().copy()
    df.loc[3, "tip"] = None
    df.loc[5, "sex"] = None
    df = pd.concat([df, df.iloc[[2]]], ignore_index=True)
    return df



def parse_uploaded_file(file_info: dict[str, Any]) -> pd.DataFrame:
    name = file_info["name"]
    path = file_info["datapath"]
    suffix = Path(name).suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, list):
            return pd.json_normalize(payload)
        if isinstance(payload, dict):
            return pd.json_normalize(payload)
        raise ValueError("JSON file is not tabular.")

    raise ValueError("Only CSV, Excel, and JSON files are supported in the simple version.")



def mode_fill(series: pd.Series) -> Any:
    modes = series.mode(dropna=True)
    if len(modes) == 0:
        return None
    return modes.iloc[0]



def clean_dataframe(df: pd.DataFrame, na_method: str, remove_duplicates: bool) -> pd.DataFrame:
    out = df.copy()

    if remove_duplicates:
        out = out.drop_duplicates().reset_index(drop=True)

    if na_method == "drop":
        out = out.dropna().reset_index(drop=True)
    elif na_method == "median":
        for col in out.columns:
            if out[col].isna().any():
                if pd.api.types.is_numeric_dtype(out[col]):
                    out[col] = out[col].fillna(out[col].median())
                else:
                    out[col] = out[col].fillna(mode_fill(out[col]))
    elif na_method == "mode":
        for col in out.columns:
            if out[col].isna().any():
                out[col] = out[col].fillna(mode_fill(out[col]))

    return out


# ---------- UI ----------
app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.h4("Simple UI version"),
        ui.p(
            "This version keeps only the core tasks: load data, basic cleaning, quick check, one chart, and feedback.",
            class_="text-muted",
        ),
        ui.input_radio_buttons(
            "data_source",
            "Dataset source",
            {
                "demo": "Use built-in demo dataset",
                "upload": "Upload my own dataset",
            },
            selected="demo",
        ),
        ui.panel_conditional(
            "input.data_source === 'upload'",
            ui.input_file(
                "file1",
                "Upload CSV / Excel / JSON",
                accept=[".csv", ".xlsx", ".xls", ".json"],
            ),
        ),
        ui.hr(),
        ui.input_select(
            "na_method",
            "Missing values",
            {
                "median": "Fill numeric with median, others with mode",
                "mode": "Fill all missing values with mode",
                "drop": "Drop rows with missing values",
            },
            selected="median",
        ),
        ui.input_switch("remove_duplicates", "Remove duplicate rows", value=True),
        ui.hr(),
        ui.input_slider("ease_of_use", "Ease of use", min=1, max=5, value=4),
        ui.input_slider("visual_clarity", "Visual clarity", min=1, max=5, value=4),
        ui.input_text_area(
            "feedback_text",
            "Comments",
            placeholder="What felt easier or harder in this version?",
            rows=4,
        ),
        ui.input_action_button("submit_feedback", "Submit feedback", class_="btn-success"),
        ui.output_text("feedback_status"),
        width=320,
    ),
    ui.tags.style(
        """
        .app-header {margin-bottom: 1rem;}
        .version-badge {
            display: inline-block;
            padding: 0.25rem 0.6rem;
            border-radius: 999px;
            background: #eef2ff;
            color: #3730a3;
            font-size: 0.85rem;
            margin-left: 0.5rem;
        }
        .metric-box {
            padding: 0.9rem 1rem;
            border: 1px solid #e5e7eb;
            border-radius: 14px;
            background: #fafafa;
        }
        .metric-label {font-size: 0.85rem; color: #6b7280;}
        .metric-value {font-size: 1.4rem; font-weight: 700;}
        .shiny-input-container {margin-bottom: 0.9rem;}
        """
    ),
    ui.div(
        ui.h2("Data Cleaning App", class_="app-header"),
        ui.span(APP_VERSION, class_="version-badge"),
    ),
    ui.p(
        "Designed for Project 3 A/B testing: fewer controls, shorter path, and explicit user feedback collection.",
        class_="text-muted",
    ),
    ui.layout_columns(
        ui.card(
            ui.card_header("Dataset summary"),
            ui.output_ui("summary_cards"),
        ),
        ui.card(
            ui.card_header("Current data source"),
            ui.output_ui("dataset_note"),
        ),
        col_widths=[7, 5],
    ),
    ui.layout_columns(
        ui.card(
            ui.card_header("Preview"),
            ui.output_table("preview_table"),
        ),
        ui.card(
            ui.card_header("Quick chart"),
            ui.output_ui("plot_controls"),
            output_widget("quick_plot"),
        ),
        col_widths=[6, 6],
    ),
    ui.card(
        ui.card_header("Export cleaned data"),
        ui.p("Download the cleaned dataset used in this simple version."),
        ui.download_button("download_cleaned", "Download cleaned CSV", class_="btn-primary"),
    ),
    title="Simple UI - Project 3",
)


# ---------- server ----------
def server(input, output, session):
    session_id = str(uuid.uuid4())
    session_started_at = datetime.utcnow()
    feedback_status = reactive.value("")

    log_usage(session_id, "session_start")

    @reactive.calc
    def dataset_bundle() -> tuple[pd.DataFrame, str]:
        if input.data_source() == "demo":
            return build_demo_df(), "Using built-in demo dataset (tips)."

        files = input.file1()
        if not files:
            return build_demo_df(), "No file uploaded yet. Showing built-in demo dataset."

        try:
            df = parse_uploaded_file(files[0])
            return df, f"Uploaded file: {files[0]['name']}"
        except Exception as e:  # pragma: no cover - safe fallback for app runtime
            return build_demo_df(), f"Upload failed ({e}). Showing built-in demo dataset instead."

    @reactive.calc
    def raw_df() -> pd.DataFrame:
        return dataset_bundle()[0]

    @reactive.calc
    def cleaned_df() -> pd.DataFrame:
        return clean_dataframe(
            raw_df(),
            na_method=input.na_method(),
            remove_duplicates=input.remove_duplicates(),
        )

    @render.ui
    def dataset_note():
        return ui.div(
            ui.p(dataset_bundle()[1]),
            ui.tags.ul(
                ui.tags.li("Simple version intentionally removes advanced feature engineering and multi-tab navigation."),
                ui.tags.li("This makes it easier to compare completion rate and subjective usability against the original app."),
            ),
        )

    @render.ui
    def summary_cards():
        raw = raw_df()
        clean = cleaned_df()
        missing_cells = int(raw.isna().sum().sum())
        duplicate_rows = int(raw.duplicated().sum())

        return ui.layout_columns(
            ui.div(
                ui.div("Rows after cleaning", class_="metric-label"),
                ui.div(str(clean.shape[0]), class_="metric-value"),
                class_="metric-box",
            ),
            ui.div(
                ui.div("Columns", class_="metric-label"),
                ui.div(str(clean.shape[1]), class_="metric-value"),
                class_="metric-box",
            ),
            ui.div(
                ui.div("Missing cells (raw)", class_="metric-label"),
                ui.div(str(missing_cells), class_="metric-value"),
                class_="metric-box",
            ),
            ui.div(
                ui.div("Duplicate rows (raw)", class_="metric-label"),
                ui.div(str(duplicate_rows), class_="metric-value"),
                class_="metric-box",
            ),
            col_widths=[3, 3, 3, 3],
        )

    @render.table
    def preview_table():
        return cleaned_df().head(10)

    @render.ui
    def plot_controls():
        df = cleaned_df()
        if df.empty:
            return ui.p("No data available for plotting.")

        all_cols = df.columns.tolist()
        non_numeric = ["(None)"] + df.select_dtypes(exclude="number").columns.tolist()
        default_focus = all_cols[0] if all_cols else None

        return ui.TagList(
            ui.input_select("focus_var", "Focus variable", choices=all_cols, selected=default_focus),
            ui.input_select("group_var", "Group by (optional)", choices=non_numeric, selected="(None)"),
        )

    @render_widget
    def quick_plot():
        df = cleaned_df()
        if df.empty or input.focus_var() is None:
            return px.scatter(title="No data to plot")

        focus_var = input.focus_var()
        group_var = input.group_var()
        color = None if group_var in (None, "(None)") else group_var

        if pd.api.types.is_numeric_dtype(df[focus_var]):
            fig = px.histogram(
                df,
                x=focus_var,
                color=color,
                marginal="box",
                title=f"Distribution of {focus_var}",
            )
        else:
            counts = df[focus_var].astype(str).value_counts(dropna=False).reset_index()
            counts.columns = [focus_var, "count"]
            fig = px.bar(
                counts,
                x=focus_var,
                y="count",
                title=f"Count of {focus_var}",
            )

        fig.update_layout(height=420, margin=dict(l=20, r=20, t=60, b=20))
        return fig

    @render.download(filename="cleaned_data_simple.csv")
    def download_cleaned():
        elapsed = (datetime.utcnow() - session_started_at).total_seconds()
        df = cleaned_df()
        log_usage(
            session_id,
            "download_cleaned",
            elapsed_seconds=round(elapsed, 2),
            rows=df.shape[0],
            cols=df.shape[1],
            data_source=input.data_source(),
            na_method=input.na_method(),
            remove_duplicates=input.remove_duplicates(),
        )
        yield df.to_csv(index=False)

    @reactive.effect
    @reactive.event(input.submit_feedback)
    def _save_feedback():
        elapsed = (datetime.utcnow() - session_started_at).total_seconds()
        df = cleaned_df()

        feedback_row = {
            "timestamp_utc": now_utc(),
            "session_id": session_id,
            "app_version": APP_VERSION,
            "elapsed_seconds": round(elapsed, 2),
            "data_source": input.data_source(),
            "na_method": input.na_method(),
            "remove_duplicates": input.remove_duplicates(),
            "rows_after_cleaning": df.shape[0],
            "cols_after_cleaning": df.shape[1],
            "ease_of_use": input.ease_of_use(),
            "visual_clarity": input.visual_clarity(),
            "comments": input.feedback_text().strip(),
        }
        append_row(FEEDBACK_LOG, feedback_row)

        log_usage(
            session_id,
            "feedback_submitted",
            elapsed_seconds=round(elapsed, 2),
            ease_of_use=input.ease_of_use(),
            visual_clarity=input.visual_clarity(),
            has_comment=bool(input.feedback_text().strip()),
        )

        feedback_status.set("Thanks - feedback saved to logs/feedback_log.csv")

    @render.text
    def feedback_status():
        return feedback_status.get()


app = App(app_ui, server)
