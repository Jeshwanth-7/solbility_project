from src.csv_processor import (
    process_csv
)

process_csv(
    input_csv="input/aqsoldb.csv",
    output_csv="output/predictions.csv"
)