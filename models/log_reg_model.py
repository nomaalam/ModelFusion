from sklearn.linear_model import LogisticRegression
from utils.evaluate_model import evaluate_model
def log_reg(params, X_train=None, y_train=None, X_test=None, y_test=None, return_model=False):
    logreg = LogisticRegression(C=params["logreg_c"], max_iter=1000)
    if return_model or X_train is None or y_train is None:
        return logreg
    y_pred_logreg, _ = evaluate_model(logreg, X_train, y_train, X_test, y_test, "Logistic Regression")