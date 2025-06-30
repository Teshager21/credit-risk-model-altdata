from imblearn.over_sampling import SMOTE

# import pandas as pd


def drop_leakage_cols(X, cols_to_drop):
    return X.drop(
        columns=[col for col in cols_to_drop if col in X.columns], errors="ignore"
    )


def apply_smote(X, y, random_state=42):
    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled
