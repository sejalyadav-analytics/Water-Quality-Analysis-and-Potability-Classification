# Water Quality Analysis and Potability Classification

## Project Overview

This project analyzes water quality parameters and applies machine-learning classification techniques to predict the potability class represented in the dataset.

## Objectives

- Explore water quality data and identify missing values.
- Perform descriptive statistics and data analysis.
- Train and compare machine-learning classification models.
- Evaluate model performance using classification metrics.
- Provide an interactive interface for educational demonstrations.

## Technologies Used

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- XGBoost
- Streamlit

## Machine Learning Models

The project compares Logistic Regression, Decision Tree, Random Forest, Extra Trees, and XGBoost, where supported by the implemented code.

## Dashboard Features

- Dataset overview and exploration
- Data-quality summaries
- Model performance comparison
- Water quality input demonstration
- Project conclusion and future scope

## Project Structure

- `main.py` — data analysis and model training
- `app.py` — Streamlit dashboard
- `data/` — dataset files, where redistribution is permitted
- `outputs/` — generated reports and analysis results
- `requirements.txt` — required Python libraries

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the machine-learning workflow:

```bash
python main.py
```

Run the interactive dashboard:

```bash
python -m streamlit run app.py
```

## Limitations

Predictions depend on the dataset, selected features, and trained models. This project is for educational analysis and does not certify that water is safe to drink.

## Future Scope

- Improve model performance through additional experiments.
- Test the models on independent water-quality datasets.
- Expand visualizations and dashboard functionality.
