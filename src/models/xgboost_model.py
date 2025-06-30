from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV


def train_xgboost(X, y, param_grid, random_state=42):
    xgb = XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        use_label_encoder=False,
        random_state=random_state,
    )
    search = RandomizedSearchCV(
        xgb,
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
