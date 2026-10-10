
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================
st.set_page_config(
    page_title="AquaInsight | Water Quality Analytics",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE = Path(__file__).resolve().parent
REPORTS = BASE / "outputs" / "reports"

FEATURES = [
    "ph", "Hardness", "Solids", "Chloramines", "Sulfate",
    "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity"
]

# =====================================================
# 2. GLOBAL STYLE - HIGH CONTRAST TEXT
# =====================================================
st.markdown("""
<style>
.stApp {
    background-color: #f4f8fc;
    color: #18354b;
}

[data-testid="stAppViewContainer"] {
    background-color: #f4f8fc;
}

.block-container {
    max-width: 1450px;
    padding-top: 1.8rem;
    padding-bottom: 3rem;
}

.stApp p,
.stApp li,
.stApp label,
.stApp span,
.stApp div[data-testid="stMarkdownContainer"] {
    color: #243e53;
}

.stApp h1, .stApp h2, .stApp h3,
.stApp h4, .stApp h5 {
    color: #103c5b !important;
    font-weight: 700 !important;
}

section[data-testid="stSidebar"] {
    background-color: #e6f0f8;
    border-right: 1px solid #d1e1ed;
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] span {
    color: #173e5a !important;
}

[data-testid="stMetric"] {
    background-color: #ffffff;
    border: 1px solid #d0e1ed;
    border-radius: 14px;
    padding: 18px;
    box-shadow: 0 2px 8px rgba(24, 61, 86, 0.06);
}

[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] p,
[data-testid="stMetricLabel"] label {
    color: #36556c !important;
}

[data-testid="stMetricValue"],
[data-testid="stMetricValue"] div {
    color: #103c5b !important;
    font-weight: 750 !important;
}

[data-testid="stMetricDelta"] {
    color: #176b45 !important;
}

[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] label,
[data-testid="stWidgetLabel"] span {
    color: #173e5a !important;
    font-weight: 600 !important;
}

.stCaption, [data-testid="stCaptionContainer"] p {
    color: #50677a !important;
}

.stNumberInput input,
.stTextInput input,
.stSelectbox div[data-baseweb="select"] {
    color: #18354b !important;
    background-color: #ffffff !important;
}

.stButton > button,
.stDownloadButton > button {
    background-color: #126a91;
    color: #ffffff !important;
    border: none;
    border-radius: 9px;
    font-weight: 650;
    padding: 0.55rem 1rem;
}

.stButton > button p,
.stDownloadButton > button p {
    color: #ffffff !important;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    background-color: #0b5274;
    border: none;
}

[data-testid="stDataFrame"] {
    border: 1px solid #d1e1ed;
    border-radius: 8px;
}

[data-testid="stAlert"] p {
    color: #18354b !important;
}

hr {
    border-color: #d5e3ed;
}
</style>
""", unsafe_allow_html=True)


# =====================================================
# 3. DATASET DISCOVERY
# =====================================================
@st.cache_data
def load_dataset():
    candidates = []

    for folder in [BASE / "data", BASE]:
        if folder.exists():
            candidates.extend(folder.glob("*.csv"))

    for path in candidates:
        if "outputs" in path.parts:
            continue

        try:
            preview = pd.read_csv(path, nrows=3)
            preview.columns = preview.columns.str.strip()
            column_map = {c.lower(): c for c in preview.columns}

            if (
                "potability" in column_map
                and all(f.lower() in column_map for f in FEATURES)
            ):
                data = pd.read_csv(path)
                data.columns = data.columns.str.strip()
                return data, str(path)

        except Exception:
            continue

    return None, None


df, dataset_path = load_dataset()


# =====================================================
# 4. TRAIN MODEL FOR THE DEMONSTRATION
# =====================================================
@st.cache_resource
def train_demo_model(csv_path):
    data = pd.read_csv(csv_path)
    data.columns = data.columns.str.strip()

    column_map = {c.lower(): c for c in data.columns}
    feature_cols = [column_map[f.lower()] for f in FEATURES]
    target_col = column_map["potability"]

    X = data[feature_cols].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(data[target_col], errors="coerce")

    valid = y.notna() & y.isin([0, 1])
    X = X.loc[valid]
    y = y.loc[valid].astype(int)

    if y.nunique() < 2:
        raise ValueError(
            "The dataset must contain both target classes."
        )

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", DecisionTreeClassifier(
            max_depth=10,
            random_state=42,
            class_weight="balanced"
        ))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    model.fit(X_train, y_train)
    test_predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, test_predictions)
    f1 = f1_score(
        y_test,
        test_predictions,
        zero_division=0
    )

    # Refit on all labelled records for the interactive demonstration.
    model.fit(X, y)

    return model, feature_cols, accuracy, f1, X


model = None
feature_cols = None
accuracy = None
f1 = None
X_data = None
model_error = None

if dataset_path:
    try:
        model, feature_cols, accuracy, f1, X_data = train_demo_model(
            dataset_path
        )
    except Exception as error:
        model_error = str(error)


# =====================================================
# 5. PARAMETER DESCRIPTIONS
# =====================================================
PARAMETER_INFO = {
    "ph": {
        "name": "pH Level",
        "unit": "pH scale",
        "description": "Measures how acidic or alkaline the water is.",
        "reference": "A commonly used guideline range is pH 6.5–8.5."
    },
    "Hardness": {
        "name": "Water Hardness",
        "unit": "mg/L",
        "description": (
            "Measures mineral content, mainly calcium and magnesium."
        ),
        "reference": (
            "Hardness alone does not determine drinking-water safety."
        )
    },
    "Solids": {
        "name": "Total Dissolved Solids",
        "unit": "mg/L, as represented in the dataset",
        "description": "Represents dissolved substances in the water.",
        "reference": (
            "Compare with the applicable local water-quality standard."
        )
    },
    "Chloramines": {
        "name": "Chloramines",
        "unit": "Dataset measurement convention",
        "description": (
            "A chemical indicator related to water disinfection."
        ),
        "reference": (
            "Check the dataset units before comparing with standards."
        )
    },
    "Sulfate": {
        "name": "Sulfate Concentration",
        "unit": "Dataset measurement convention",
        "description": "Measures sulfate present in the water.",
        "reference": (
            "Compare with the applicable local water-quality standard."
        )
    },
    "Conductivity": {
        "name": "Electrical Conductivity",
        "unit": "Dataset measurement convention",
        "description": (
            "Indicates how readily the water conducts electricity."
        ),
        "reference": (
            "Conductivity alone cannot establish drinking-water safety."
        )
    },
    "Organic_carbon": {
        "name": "Organic Carbon",
        "unit": "Dataset measurement convention",
        "description": "Represents organic carbon in the water sample.",
        "reference": (
            "Interpret alongside other water-quality measurements."
        )
    },
    "Trihalomethanes": {
        "name": "Trihalomethanes (THMs)",
        "unit": "Dataset measurement convention",
        "description": (
            "Represents disinfection by-products that can form during treatment."
        ),
        "reference": (
            "Confirm the dataset units before comparing with standards."
        )
    },
    "Turbidity": {
        "name": "Water Turbidity",
        "unit": "Dataset measurement convention",
        "description": "Indicates how cloudy the water is.",
        "reference": (
            "Low turbidity does not prove that water is safe."
        )
    }
}


# =====================================================
# 6. SIDEBAR
# =====================================================
st.sidebar.title("💧 AquaInsight")
st.sidebar.write("Water Quality Intelligence")
st.sidebar.divider()

page = st.sidebar.radio(
    "SELECT A PAGE",
    [
        "Overview",
        "Explore Data",
        "Model Performance",
        "Water Quality Demonstration",
        "Conclusion"
    ]
)

st.sidebar.divider()
st.sidebar.caption("IBM SkillsBuild Project")


# =====================================================
# 7. HEADER
# =====================================================
st.title("💧 AquaInsight")
st.subheader(
    "Water Quality Analysis and Potability Classification"
)
st.caption("IBM SkillsBuild | Water Quality Intelligence")

if df is None:
    st.error(
        "Dataset not found. Put the original water potability CSV "
        "inside the project's data folder."
    )

if model_error:
    st.error(
        f"Could not train the demonstration model: {model_error}"
    )


# =====================================================
# 8. OVERVIEW PAGE
# =====================================================
if page == "Overview":
    st.header("Project Overview")

    st.write(
        "Analyse water-quality measurements, explore data patterns, "
        "compare machine-learning results and demonstrate a "
        "classification model."
    )

    if df is not None:
        target = next(
            (c for c in df.columns if c.lower() == "potability"),
            None
        )

        total_samples = len(df)
        missing_values = int(df.isna().sum().sum())
        parameter_count = len(FEATURES)

        potable_count = (
            int((df[target] == 1).sum()) if target else 0
        )
        nonpotable_count = (
            int((df[target] == 0).sum()) if target else 0
        )

        k1, k2, k3, k4 = st.columns(4)

        k1.metric("Total Water Samples", f"{total_samples:,}")
        k2.metric("Water Parameters", f"{parameter_count}")
        k3.metric("Missing Values", f"{missing_values:,}")
        k4.metric("Potable-Class Samples", f"{potable_count:,}")

        st.divider()

        left, right = st.columns(2)

        with left:
            st.subheader("Potability Class Distribution")

            if target:
                chart_data = pd.DataFrame({
                    "Class": [
                        "Non-potable class",
                        "Potable class"
                    ],
                    "Samples": [
                        nonpotable_count,
                        potable_count
                    ]
                })

                fig = px.pie(
                    chart_data,
                    names="Class",
                    values="Samples",
                    hole=0.55,
                    color="Class",
                    color_discrete_map={
                        "Non-potable class": "#277da1",
                        "Potable class": "#43aa8b"
                    }
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

        with right:
            st.subheader("Missing Values by Parameter")

            missing_data = (
                df.isna().sum()
                .rename_axis("Parameter")
                .reset_index(name="Missing Values")
                .sort_values("Missing Values")
            )

            fig = px.bar(
                missing_data,
                x="Missing Values",
                y="Parameter",
                orientation="h",
                color_discrete_sequence=["#277da1"]
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        if model is not None:
            st.subheader("Demonstration Model Performance")

            m1, m2 = st.columns(2)
            m1.metric("Test Accuracy", f"{accuracy:.1%}")
            m2.metric("Test F1-score", f"{f1:.1%}")

            st.caption(
                "These metrics belong to the dashboard's Decision Tree "
                "model and may differ from the models evaluated in main.py."
            )


# =====================================================
# 9. DATA EXPLORATION
# =====================================================
elif page == "Explore Data":
    st.header("Interactive Data Exploration")

    if df is not None:
        st.write(
            "Select columns, explore measurement distributions, "
            "and investigate correlations."
        )

        chosen_columns = st.multiselect(
            "Select dataset columns to display",
            df.columns.tolist(),
            default=df.columns.tolist()
        )

        if chosen_columns:
            st.dataframe(
                df[chosen_columns].head(500),
                use_container_width=True,
                hide_index=True
            )

        numeric_columns = df.select_dtypes(
            include=np.number
        ).columns.tolist()

        if numeric_columns:
            st.subheader("Explore a Measurement")

            feature = st.selectbox(
                "Choose a water-quality parameter",
                numeric_columns
            )

            chart_type = st.radio(
                "Choose a chart",
                ["Histogram", "Box Plot"],
                horizontal=True
            )

            if chart_type == "Histogram":
                fig = px.histogram(
                    df,
                    x=feature,
                    nbins=35,
                    title=f"Distribution of {feature}"
                )
            else:
                fig = px.box(
                    df,
                    y=feature,
                    title=f"Distribution and outliers: {feature}"
                )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            st.subheader("Correlation Heatmap")

            fig = px.imshow(
                df[numeric_columns].corr(),
                text_auto=".2f",
                aspect="auto",
                color_continuous_scale="RdBu_r"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        st.download_button(
            "Download Dataset as CSV",
            df.to_csv(index=False).encode("utf-8"),
            file_name="water_quality_data.csv",
            mime="text/csv"
        )


# =====================================================
# 10. MODEL PERFORMANCE
# =====================================================
elif page == "Model Performance":
    st.header("Machine-Learning Model Performance")

    if model is not None:
        a, b = st.columns(2)

        a.metric(
            "Decision Tree Test Accuracy",
            f"{accuracy:.1%}"
        )
        b.metric(
            "Decision Tree Test F1-score",
            f"{f1:.1%}"
        )

    comparison_path = REPORTS / "model_comparison.csv"

    if comparison_path.exists():
        results = pd.read_csv(comparison_path)

        st.subheader("Original Project Model Comparison")

        st.dataframe(
            results,
            use_container_width=True,
            hide_index=True
        )

        model_column = next(
            (
                c for c in results.columns
                if c.lower() == "model"
            ),
            None
        )

        metric_columns = [
            c for c in results.select_dtypes(
                include=np.number
            ).columns
            if any(
                key in c.lower()
                for key in [
                    "accuracy", "precision", "recall",
                    "f1", "roc"
                ]
            )
        ]

        if model_column and metric_columns:
            chart_data = results.melt(
                id_vars=model_column,
                value_vars=metric_columns,
                var_name="Metric",
                value_name="Score"
            )

            fig = px.bar(
                chart_data,
                x=model_column,
                y="Score",
                color="Metric",
                barmode="group",
                title="Compare Model Evaluation Metrics"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    else:
        st.info(
            "The original model_comparison.csv report was not found."
        )

    importance_path = next(
        iter(
            REPORTS.glob(
                "permutation_feature_importance*.csv"
            )
        ),
        None
    )

    if importance_path:
        st.subheader("Feature Importance")

        importance = pd.read_csv(importance_path)

        st.dataframe(
            importance,
            use_container_width=True
        )

        numeric_cols = importance.select_dtypes(
            include=np.number
        ).columns.tolist()

        feature_column = next(
            (
                c for c in importance.columns
                if c.lower() in ["feature", "features"]
            ),
            None
        )

        if numeric_cols and feature_column:
            score_column = st.selectbox(
                "Select importance score",
                numeric_cols
            )

            plot_data = importance.sort_values(
                score_column
            ).tail(15)

            fig = px.bar(
                plot_data,
                x=score_column,
                y=feature_column,
                orientation="h",
                title="Most Important Features"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


# =====================================================
# 11. WATER QUALITY DEMONSTRATION
# =====================================================
elif page == "Water Quality Demonstration":
    st.header("🧪 Interactive Water Quality Demonstration")

    st.write(
        "Enter measured values from a water sample and submit them "
        "to the machine-learning model."
    )

    st.info(
        "The initial values are the medians of the dataset. "
        "Replace them with your sample's measurements when available."
    )

    if model is not None and X_data is not None:
        values = {}

        left, right = st.columns(2)

        for i, feature in enumerate(feature_cols):
            series = pd.to_numeric(
                X_data[feature],
                errors="coerce"
            ).dropna()

            if series.empty:
                continue

            minimum = float(series.min())
            maximum = float(series.max())
            default = float(series.median())

            info = PARAMETER_INFO.get(
                feature,
                {
                    "name": feature.replace("_", " "),
                    "unit": "Dataset units",
                    "description": "Enter the measured value.",
                    "reference": (
                        "Check the relevant water-quality standard."
                    )
                }
            )

            with (left if i % 2 == 0 else right):
                st.markdown(f"### {info['name']}")
                st.caption(
                    f"Measurement unit: {info['unit']}"
                )
                st.write(info["description"])

                values[feature] = st.number_input(
                    label=f"Enter {info['name']}",
                    min_value=minimum,
                    max_value=maximum,
                    value=default,
                    key=f"measurement_{feature}",
                    help=(
                        f"Observed dataset range: "
                        f"{minimum:.3f} to {maximum:.3f}. "
                        f"{info['reference']}"
                    )
                )

                with st.expander("Reference information"):
                    st.write(info["reference"])
                    st.caption(
                        "The dataset range is not a safety limit. "
                        "Units and applicable standards must be verified."
                    )

        st.divider()

        if st.button(
            "🔍 Analyse Water Sample",
            type="primary",
            use_container_width=True
        ):
            sample = pd.DataFrame(
                [[values[c] for c in feature_cols]],
                columns=feature_cols
            )

            prediction = int(model.predict(sample)[0])
            probabilities = model.predict_proba(sample)[0]
            classes = list(model.classes_)

            predicted_probability = float(
                probabilities[classes.index(prediction)]
            )

            st.subheader("Analysis Result")

            result_left, result_right = st.columns(2)

            with result_left:
                if prediction == 1:
                    st.success(
                        "Predicted Class: POTABLE (Class 1)"
                    )
                else:
                    st.error(
                        "Predicted Class: NON-POTABLE (Class 0)"
                    )

            with result_right:
                st.metric(
                    "Model Probability for Predicted Class",
                    f"{predicted_probability:.1%}"
                )

            st.subheader("Submitted Measurements")

            st.dataframe(
                sample,
                use_container_width=True,
                hide_index=True
            )

            st.warning(
                "This is a machine-learning classification, not a "
                "drinking-water safety certificate. The model may make "
                "incorrect predictions. Do not use this result alone to "
                "decide whether water is safe to drink. Obtain appropriate "
                "certified laboratory testing."
            )

    else:
        st.error(
            "The model could not be prepared. Check that the original "
            "dataset exists and includes all nine required features "
            "and the Potability target."
        )


# =====================================================
# 12. CONCLUSION PAGE
# =====================================================
elif page == "Conclusion":

    st.header("Project Conclusion")

    st.write(
        "Water Quality Analysis and Potability Classification uses "
        "data analysis and machine learning to study water-quality "
        "measurements and predict the potability class."
    )

    st.divider()

    st.subheader("1. Project Objectives")

    st.markdown("""
    - Analyse nine water-quality parameters.
    - Identify missing values and explore data patterns.
    - Train and compare machine-learning classification models.
    - Evaluate models using standard performance metrics.
    - Develop an interactive dashboard for data exploration and prediction.
    """)

    st.divider()

    st.subheader("2. Techniques Applied")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **Data Analysis**
        - Data cleaning and missing-value handling
        - Descriptive statistics
        - Distribution and correlation analysis

        **Machine Learning**
        - Logistic Regression
        - Decision Tree
        - Random Forest
        - Extra Trees
        - XGBoost
        """)

    with col2:
        st.markdown("""
        **Model Evaluation**
        - Accuracy, precision and recall
        - F1-score and ROC-AUC
        - Confusion matrices
        - Permutation feature importance

        **Interactive Application**
        - Streamlit dashboard
        - Interactive charts
        - User-entered water measurements
        - Potability class prediction
        """)

    st.divider()

    st.subheader("3. Benefits of the Project")

    st.markdown("""
    - **Data-driven insights:** Helps explore patterns in water-quality data.
    - **Model comparison:** Shows how different classification algorithms perform.
    - **Interactive analysis:** Makes analysis more accessible through a dashboard.
    - **Feature importance:** Helps investigate which parameters influence model predictions.
    - **Practical learning:** Demonstrates data science applied to an environmental problem.
    """)

    st.divider()

    st.subheader("4. Future Scope")

    st.markdown("""
    - Improve model performance through hyperparameter tuning.
    - Add SHAP explanations for individual predictions.
    - Integrate additional, verified water-quality datasets.
    - Add comparisons with applicable water-quality standards.
    - Connect IoT sensors for real-time monitoring.
    - Deploy the dashboard online for wider access.
    """)

    st.divider()

    st.subheader("5. Final Conclusion")

    st.success(
        "The project demonstrates how data analysis, machine learning, "
        "and interactive visualisation can help study water quality "
        "and classify samples using measured parameters."
    )

    st.warning(
        "A model prediction is not a guarantee of drinking-water safety. "
        "Appropriate laboratory testing and applicable water-quality "
        "standards are necessary to assess actual safety."
    )


# =====================================================
# 13. FOOTER
# =====================================================
st.divider()

st.caption(
    "AquaInsight | Water Quality Analysis and Potability Classification"
)
