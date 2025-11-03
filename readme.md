# ModelFusion

ModelFusion is an interactive Streamlit web application for exploring and comparing multiple classification models on your own datasets. Upload a CSV, select features and target, tune model hyperparameters, perform feature selection, visualize decision boundaries, and evaluate model performance—all in one place.

**Deployed App:** https://modelfusion.streamlit.app/
## Features

- **Upload CSV**: Easily upload your dataset for analysis.
- **Feature & Target Selection**: Choose which columns to use as features and target.
- **Feature Selection**: Optionally select the top K features using ANOVA F-value.
- **Preprocessing**: Automatic encoding of categorical variables and feature scaling.
- **Model Selection**: Choose from KNN, SVM, Decision Tree, Logistic Regression, and Random Forest.
- **Hyperparameter Tuning**: Adjust key hyperparameters for each model via the sidebar.
- **Model Evaluation**: View classification reports, F1 scores, and confusion matrices.
- **Decision Boundary Visualization**: Visualize decision boundaries for models with 2 features.
- **Cross-Validation**: Run 5-fold cross-validation and see average F1 scores.

## Getting Started

### Prerequisites

- Python 3.8+
- See [requirements.txt](requirements.txt) for all dependencies.

### Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/nomaalam/ModelFusion.git
    cd ModelFusion
    ```

2. Install dependencies:
    ```sh
    pip install -r requirements.txt
    ```

### Running the App

```sh
streamlit run app.py
```

Open the provided local URL in your browser.

## Usage

1. Upload your CSV file.
2. Select the target column and features.
3. (Optional) Enable feature selection and choose the number of features.
4. Choose models and set their hyperparameters.
5. Click "Train & Evaluate" to see results.
6. (Optional) Run cross-validation for a selected model.

## Project Structure

- `app.py` — Main Streamlit app.
- `models/` — Model definitions for each classifier.
- `utils/` — Utility functions for evaluation, feature selection, cross-validation, and visualization.
- `requirements.txt` — Python dependencies.

## License

MIT License
