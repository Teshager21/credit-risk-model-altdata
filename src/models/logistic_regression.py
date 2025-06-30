from sklearn.linear_model import LogisticRegression


def train_logistic_regression(X_train, y_train, random_state=42):
    model = LogisticRegression(
        random_state=random_state,
        class_weight="balanced",
        max_iter=500,
    )
    model.fit(X_train, y_train)
    return model
