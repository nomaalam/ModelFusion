from sklearn.ensemble import RandomForestClassifier
from utils.evaluate_model import evaluate_model


def random_forest(params, X_train=None, y_train=None, X_test=None, y_test=None, return_model=False):
    rf = RandomForestClassifier(n_estimators=params["rf_estimators"],
    max_depth=params["rf_depth"],
    random_state=42)
    if return_model:
        return rf
    y_pred_rf, _ = evaluate_model(rf, X_train, y_train, X_test, y_test, "Random Forest")