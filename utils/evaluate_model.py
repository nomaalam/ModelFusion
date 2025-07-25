from sklearn.metrics import classification_report, confusion_matrix, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

def evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.subheader(f"📊 {model_name} Evaluation")
    st.text("Classification Report")
    st.text(classification_report(y_test, y_pred))

    f1 = f1_score(y_test, y_pred, average='weighted')
    st.metric(label=f"{model_name} F1 Score", value=round(f1, 3))

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title(f"{model_name} - Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    return y_pred, f1
