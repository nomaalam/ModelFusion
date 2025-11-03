from sklearn.svm import SVC
from utils.evaluate_model import evaluate_model
from utils.decision_boundary import plot_decision_boundary

def svm_classifier(params, X_train=None, y_train=None, X_test=None, y_test=None, return_model=False):
    svm = SVC(kernel=params["svm_kernel"], C=params["svm_c"], max_iter=1000)
    if return_model or X_train is None or y_train is None:
        return svm
    y_pred_svm, _ = evaluate_model(svm, X_train, y_train, X_test, y_test, "SVM")
    plot_decision_boundary(svm, X_train, y_train, "SVM")