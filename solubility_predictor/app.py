import streamlit as st
import pandas as pd
import time

from src.csv_processor import (
    process_uploaded_csv
)

from src.smiles_detector import (
    detect_smiles_column
)

st.set_page_config(
    page_title="Solubility Predictor",
    layout="wide"
)

st.title(
    "Solubility Predictor"
)

uploaded_file = st.file_uploader(
    "Upload Dataset",
    type=["csv", "xlsx", "xls"]
)

if uploaded_file is not None:

    if uploaded_file.name.lower().endswith(
        ".csv"
    ):

        preview_df = pd.read_csv(
            uploaded_file,
            nrows=100
        )

        uploaded_file.seek(0)

    elif uploaded_file.name.lower().endswith(
        (".xlsx", ".xls")
    ):

        preview_df = pd.read_excel(
            uploaded_file,
            nrows=100
        )

        uploaded_file.seek(0)

    smiles_col = detect_smiles_column(
        preview_df
    )

    file_size_mb = round(
        uploaded_file.size /
        (1024 * 1024),
        2
    )

    st.subheader(
        "Dataset Information"
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Mode",
        "Streaming"
    )

    col2.metric(
        "Columns",
        len(preview_df.columns)
    )

    col3.metric(
        "SMILES Column",
        smiles_col
    )

    col4.metric(
        "File Size (MB)",
        file_size_mb
    )

    st.info(
        f"Uploaded File: {uploaded_file.name}"
    )

    st.subheader(
        "Preview (First 100 Rows)"
    )

    st.dataframe(
        preview_df.head(),
        use_container_width=True
    )

    if st.button(
        "Predict Solubility"
    ):

        start_time = time.time()

        progress_bar = st.progress(
            0
        )

        status_text = st.empty()

        process_uploaded_csv(
            uploaded_file,
            progress_bar,
            status_text
        )

        end_time = time.time()

        runtime = round(
            end_time - start_time,
            2
        )

        progress_bar.progress(
            100
        )

        status_text.success(
            "Prediction Complete"
        )

        st.success(
            "Prediction Complete"
        )

        st.info(
            f"Runtime: {runtime} seconds"
        )

        st.subheader(
            "Prediction Preview"
        )

        preview_predictions = pd.read_csv(
            "output/predictions.csv",
            nrows=20
        )

        st.dataframe(
            preview_predictions,
            use_container_width=True
        )

        with open(
            "output/predictions.csv",
            "rb"
        ) as f:

            st.download_button(
                label=" Download Predictions CSV",
                data=f,
                file_name="predictions.csv",
                mime="text/csv"
            )