import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from utils.cross_validation import cross_validate_model
from utils.feature_selection import select_k_best_features
from models.knn_model import k_neighbors
from models.svm_model import svm_classifier
from models.decision_tree_model import decision_tree
from models.log_reg_model import log_reg
from models.random_forest_model import random_forest
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="ModelFusion", layout="wide")
st.title("ModelFusion: Multi-Model Classification Explorer")

uploaded_file = st.sidebar.file_uploader("📁 Upload CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df.head())

    if df.isnull().values.any():
        st.warning("Missing values detected. Rows will be dropped.")
        df.dropna(inplace=True)

    all_columns = df.columns.tolist()
    target = st.sidebar.selectbox("🎯 Select Target Column", all_columns)

    # Add a checkbox to select all features
    select_all = st.sidebar.checkbox("Select All Features")
    if select_all:
        features = [c for c in all_columns if c != target]
    else:
        features = st.sidebar.multiselect(
            "🧮 Select Feature Columns",
            [c for c in all_columns if c != target]
        )

    if features and target:
        X = df[features].copy()
        y = df[target].copy()

        # Encode categorical features in X (e.g., 'Mar', 'Male', etc.)
        X = pd.get_dummies(X)

        scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

        # Encode target if it's categorical (object or string)
        if y.dtype == 'object' or y.dtype.name == 'category':
            le = LabelEncoder()
            y = le.fit_transform(y)



        split_ratio = st.sidebar.slider("🔀 Train/Test Split", 0.1, 0.9, 0.7)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-split_ratio, random_state=42)

        if st.sidebar.checkbox("Enable Feature Selection"):
            k = st.sidebar.slider("Number of Features to Select", 1, len(features), min(5, len(features)))
            X_train_selected = select_k_best_features(X_train, y_train, k)
            # Use the same features for X_test
            X_test_selected = X_test[X_train_selected.columns]
            selected_features = X_train_selected.columns.tolist()
            st.subheader("✅ Selected Features for Training")
            st.write(selected_features)
        else:
            X_train_selected = X_train
            X_test_selected = X_test

        scaler = StandardScaler()
        X_train_selected = pd.DataFrame(scaler.fit_transform(X_train_selected), columns=X_train_selected.columns)
        X_test_selected = pd.DataFrame(scaler.transform(X_test_selected), columns=X_test_selected.columns)
        
        all_models = ["KNN", "SVM", "Decision Tree", "Logistic Regression", "Random Forest"]
        select_all_models = st.sidebar.checkbox("Select All Models")
        if select_all_models:
            model_choice = all_models
        else:
            model_choice = st.sidebar.multiselect("🤖 Select Models", all_models)

        params = {}

        if "KNN" in model_choice:
            params["knn_n"] = st.sidebar.slider("KNN: n_neighbors", 1, 15, 5)

        if "SVM" in model_choice:
            params["svm_kernel"] = st.sidebar.selectbox("SVM: Kernel", ["linear", "rbf", "poly"])
            params["svm_c"] = st.sidebar.slider("SVM: C", 0.1, 10.0, 1.0)

        if "Decision Tree" in model_choice:
            params["dt_depth"] = st.sidebar.slider("DT: Max Depth", 1, 20, 5)
            params["dt_crit"] = st.sidebar.selectbox("DT: Criterion", ["gini", "entropy"])

        if "Logistic Regression" in model_choice:
            params["logreg_c"] = st.sidebar.slider("LogReg: C (Inverse Regularization)", 0.01, 10.0, 1.0)

        if "Random Forest" in model_choice:
            params["rf_estimators"] = st.sidebar.slider("RF: n_estimators", 10, 200, 100)
            params["rf_depth"] = st.sidebar.slider("RF: max_depth", 1, 30, 10)


        if st.sidebar.button("🚀 Train & Evaluate"):
            if "KNN" in model_choice:
                k_neighbors(params, X_train_selected, y_train, X_test_selected, y_test)

            if "SVM" in model_choice:
                svm_classifier(params, X_train_selected, y_train, X_test_selected, y_test)

            if "Decision Tree" in model_choice:
                decision_tree(params, X_train_selected, y_train, X_test_selected, y_test)

            if "Logistic Regression" in model_choice:
                log_reg(params, X_train_selected, y_train, X_test_selected, y_test)

            if "Random Forest" in model_choice:
                random_forest(params, X_train_selected, y_train, X_test_selected, y_test)
        


        if st.sidebar.checkbox("Run 5-Fold Cross-Validation"):
            cv_model_name = st.sidebar.selectbox("Select Model for Cross-Validation", ["KNN", "SVM", "Decision Tree", "Logistic Regression", "Random Forest"])
            if cv_model_name == "KNN":
                from models.knn_model import k_neighbors
                model = k_neighbors(params, return_model=True)
            elif cv_model_name == "SVM":
                from models.svm_model import svm_classifier
                model = svm_classifier(params, return_model=True)
            elif cv_model_name == "Decision Tree":
                from models.decision_tree_model import decision_tree
                model = decision_tree(params, return_model=True)
            elif cv_model_name == "Logistic Regression":
                from models.log_reg_model import log_reg
                model = log_reg(params, return_model=True)
            elif cv_model_name == "Random Forest":
                from models.random_forest_model import random_forest
                model = random_forest(params, return_model=True)
            cross_validate_model(model, X[X_train_selected.columns], y)


