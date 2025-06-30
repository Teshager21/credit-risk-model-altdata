from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

# import numpy as np


def train_random_forest(X, y, param_grid, random_state=42):
    rf = RandomForestClassifier(random_state=random_state, class_weight="balanced")
    search = RandomizedSearchCV(
        rf,
        param_distributions=param_grid,
        n_iter=10,
        scoring="roc_auc",
        cv=3,
        n_jobs=-1,
        random_state=random_state,
        verbose=2,
    )
    search.fit(X, y)
    return search.best_estimator_, search.best_params_
