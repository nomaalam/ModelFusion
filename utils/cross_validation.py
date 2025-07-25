from sklearn.model_selection import cross_val_score
import streamlit as st

def cross_validate_model(model, X, y):
    with st.spinner("Running cross-validation..."):
        scores = cross_val_score(model, X, y, cv=5, scoring='f1_weighted')
        st.success(f"Mean CV F1 Score: {scores.mean():.3f}")
        st.write("CV Scores:", scores)
