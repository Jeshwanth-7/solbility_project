import pandas as pd
from joblib import (
    Parallel,
    delayed
)
from src.smiles_detector import (
    detect_smiles_column
)

from src.predictor import (
    predict_solubility
)

from config import (
    CHUNK_SIZE
)


def process_dataframe(df):

    smiles_col = (
        detect_smiles_column(df)
    )

    df["predicted_solubility"] = (

        df[smiles_col]

        .astype(str)

        .apply(
            predict_solubility
        )

    )

    return df


def process_dataframe_with_progress(
    df,
    progress_bar,
    status_text
):

    smiles_col = (
        detect_smiles_column(df)
    )

    total = len(df)

    predictions = []

    valid_count = 0

    invalid_count = 0

    processed = 0

    for start in range(
        0,
        total,
        CHUNK_SIZE
    ):

        end = min(
            start + CHUNK_SIZE,
            total
        )

        chunk = df.iloc[
            start:end
        ]

        for smiles in chunk[
            smiles_col
        ]:

            pred = predict_solubility(
                str(smiles)
            )

            predictions.append(
                pred
            )

            processed += 1

            if pd.isna(pred):

                invalid_count += 1

            else:

                valid_count += 1

        progress = (
            processed / total
        )

        progress_bar.progress(
            progress
        )

        status_text.text(
            f"""
Processed : {processed}/{total}

Valid     : {valid_count}

Invalid   : {invalid_count}

Chunk Size: {CHUNK_SIZE}

Progress  : {round(progress * 100, 2)}%
"""
        )

    df[
        "predicted_solubility"
    ] = predictions

    return df


def process_csv(
    input_csv,
    output_csv
):

    print(
        f"Loading CSV: {input_csv}"
    )

    df = pd.read_csv(
        input_csv
    )

    print(
        f"Rows Found: {len(df)}"
    )

    df = process_dataframe(
        df
    )

    df.to_csv(
        output_csv,
        index=False
    )

    print(
        f"Saved Output: {output_csv}"
    )

    return df

import os
import time

from config import (
    CHUNK_SIZE,
    OUTPUT_FILE
)


def process_large_dataframe(
    df,
    progress_bar,
    status_text
):

    smiles_col = detect_smiles_column(
        df
    )

    total = len(df)

    processed = 0

    valid_count = 0

    invalid_count = 0

    start_time = time.time()

    if os.path.exists(
        OUTPUT_FILE
    ):
        os.remove(
            OUTPUT_FILE
        )

    first_chunk = True

    for start in range(
        0,
        total,
        CHUNK_SIZE
    ):

        end = min(
            start + CHUNK_SIZE,
            total
        )

        chunk = df.iloc[
            start:end
        ].copy()

        predictions = []

        for smiles in chunk[
            smiles_col
        ]:

            pred = predict_solubility(
                str(smiles)
            )

            predictions.append(
                pred
            )

            if pd.isna(pred):

                invalid_count += 1

            else:

                valid_count += 1

        chunk[
            "predicted_solubility"
        ] = predictions

        chunk.to_csv(
            OUTPUT_FILE,
            mode="a",
            header=first_chunk,
            index=False
        )

        first_chunk = False

        processed += len(chunk)

        progress = (
            processed / total
        )

        elapsed = (
            time.time()
            - start_time
        )

        speed = (
            processed / elapsed
        )

        remaining = (
            total - processed
        )

        eta_seconds = (
            remaining / speed
            if speed > 0
            else 0
        )

        eta_minutes = round(
            eta_seconds / 60,
            1
        )

        progress_bar.progress(
            progress
        )

        status_text.text(
            f"""
Processed : {processed}/{total}

Valid     : {valid_count}

Invalid   : {invalid_count}

Chunk Size: {CHUNK_SIZE}

ETA       : {eta_minutes} min
"""
        )

    return None


import pandas as pd
import os
import time

from src.smiles_detector import (
    detect_smiles_column
)

from src.predictor import (
    predict_solubility
)

from config import (
    CHUNK_SIZE,
    OUTPUT_FILE
)


def process_uploaded_csv(
    uploaded_file,
    progress_bar,
    status_text
):

    uploaded_file.seek(0)

    preview_df = pd.read_csv(
        uploaded_file,
        nrows=100
    )

    smiles_col = detect_smiles_column(
        preview_df
    )

    uploaded_file.seek(0)

    if os.path.exists(
        OUTPUT_FILE
    ):
        os.remove(
            OUTPUT_FILE
        )

    first_chunk = True

    processed = 0

    valid_count = 0

    invalid_count = 0

    start_time = time.time()

    for chunk in pd.read_csv(
        uploaded_file,
        chunksize=CHUNK_SIZE
    ):

        predictions = Parallel(
            n_jobs=12
        )(
            delayed(
                predict_solubility
            )(
                str(smiles)
            )

            for smiles in chunk[
                smiles_col
            ]
        )

        processed += len(
            predictions
        )

        invalid_chunk = sum(
            pd.isna(pred)
            for pred in predictions
        )

        invalid_count += (
            invalid_chunk
        )

        valid_count += (
            len(predictions)
            - invalid_chunk
        )

        chunk[
            "predicted_solubility"
        ] = predictions

        chunk.to_csv(
            OUTPUT_FILE,
            mode="a",
            header=first_chunk,
            index=False
        )

        first_chunk = False

        elapsed = (
            time.time()
            - start_time
        )

        speed = (
            processed / elapsed
        )

        status_text.text(
            f"""
Processed : {processed}

Valid     : {valid_count}

Invalid   : {invalid_count}

Chunk Size: {CHUNK_SIZE}

Speed     : {round(speed, 2)} mol/sec
"""
        )

    return None