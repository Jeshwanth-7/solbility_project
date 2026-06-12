import joblib
import numpy as np

import warnings

warnings.filterwarnings(
    "ignore",
    category=UserWarning
)

from src.feature_generator import (
    generate_features
)

pipeline = joblib.load(
    "artifacts/best_lgbm_pipeline.pkl"
)

final_feature_columns = joblib.load(
    "artifacts/final_feature_columns.pkl"
)




def predict_solubility(smiles):

    

    X_new = generate_features(smiles)

    

    

    if X_new is None:
        return np.nan

    X_new = X_new.reindex(
        columns=final_feature_columns,
        fill_value=np.nan
    )

    prediction = pipeline.predict(
        X_new
    )

    return prediction[0]