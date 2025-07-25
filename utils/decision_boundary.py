import numpy as np
import streamlit as st

def plot_decision_boundary(model, X, y, model_name):
    if X.shape[1] != 2:
        st.warning("Decision boundary is only available for exactly 2 selected features.")
        return

    model.fit(X, y)
    h = 0.5
    x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
    y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    import matplotlib.pyplot as plt
    import seaborn as sns
    fig, ax = plt.subplots()
    ax.contourf(xx, yy, Z, alpha=0.3)
    sns.scatterplot(x=X.iloc[:, 0], y=X.iloc[:, 1], hue=y, edgecolor='k', ax=ax)
    ax.set_title(f"{model_name} - Decision Boundary")
    ax.set_xlabel(X.columns[0])
    ax.set_ylabel(X.columns[1])
    st.pyplot(fig)
