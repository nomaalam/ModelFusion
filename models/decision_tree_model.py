from sklearn.tree import DecisionTreeClassifier
from utils.evaluate_model import evaluate_model
from utils.decision_boundary import plot_decision_boundary


def decision_tree(params, X_train=None, y_train=None, X_test=None, y_test=None, return_model=False):
    dt = DecisionTreeClassifier(max_depth=params["dt_depth"], criterion=params["dt_crit"])
    if return_model:
        return dt
    y_pred_dt, _ = evaluate_model(dt, X_train, y_train, X_test, y_test, "Decision Tree")
    plot_decision_boundary(dt, X_train, y_train, "Decision Tree")