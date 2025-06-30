from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV


def train_gradient_boosting(X, y, param_grid, random_state=42):
    gbm = GradientBoostingClassifier(random_state=random_state)
    search = GridSearchCV(
        gbm,
        param_grid,
        scoring="f1",
        cv=3,
        n_jobs=-1,
        verbose=2,
    )
    search.fit(X, y)
    return search.best_estimator_, search.best_params_
