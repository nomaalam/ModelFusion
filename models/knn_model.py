from sklearn.neighbors import KNeighborsClassifier
from utils.evaluate_model import evaluate_model
from utils.decision_boundary import plot_decision_boundary

def k_neighbors(params, X_train=None, y_train=None, X_test=None, y_test=None, return_model=False):
    knn = KNeighborsClassifier(n_neighbors=params["knn_n"])
    if return_model or X_train is None or y_train is None:
        return knn
    y_pred_knn, _ = evaluate_model(knn, X_train, y_train, X_test, y_test, "KNN")
    plot_decision_boundary(knn, X_train, y_train, "KNN")