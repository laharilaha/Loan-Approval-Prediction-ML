import os
import pandas as pd
import streamlit as st
import plotly.express as px
import shap

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.stApp {
    background-color: #071A2F;
}

/* Main content */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}

/* Main title */
.main-title {
    font-size: 42px;
    font-weight: 800;
    color: #FFFFFF;
    margin-bottom: 5px;
}

.subtitle {
    font-size: 17px;
    color: #8ED8FF;
    margin-bottom: 25px;
}

/* Section headings */
.section-title {
    color: #20C4FF;
    font-size: 25px;
    font-weight: 700;
    margin-top: 20px;
    margin-bottom: 15px;
}

/* Cards */
.card {
    background: #0D2742;
    border: 1px solid #164A70;
    border-radius: 16px;
    padding: 22px;
    margin-bottom: 18px;
    box-shadow: 0 5px 20px rgba(0,0,0,0.25);
}

/* Labels */
label,
.stSelectbox label,
.stNumberInput label {
    color: #FFFFFF !important;
    font-weight: 600 !important;
}

/* Number inputs */
.stNumberInput input {
    color: #071A2F !important;
    background-color: #FFFFFF !important;
    font-weight: 700 !important;
    border-radius: 10px !important;
}

/* Select boxes */
.stSelectbox div[data-baseweb="select"] > div {
    color: #071A2F !important;
    background-color: #FFFFFF !important;
    font-weight: 700 !important;
    border-radius: 10px !important;
}

/* Selectbox input */
.stSelectbox input {
    color: #071A2F !important;
}

/* Dropdown options */
div[role="option"] {
    color: #071A2F !important;
    background-color: #FFFFFF !important;
}

/* Buttons */
.stButton > button {
    width: 100%;
    background: linear-gradient(90deg, #087CF5, #08B7E8);
    color: white !important;
    border: none;
    border-radius: 10px;
    padding: 12px 20px;
    font-size: 17px;
    font-weight: 700;
}

.stButton > button:hover {
    background: linear-gradient(90deg, #0969D8, #0799C5);
}

/* Prediction approved */
.approved {
    background: #0C3B35;
    border: 2px solid #18D6B0;
    border-radius: 15px;
    padding: 25px;
    text-align: center;
    margin-bottom: 20px;
}

.approved-title {
    color: #18D6B0;
    font-size: 32px;
    font-weight: 800;
}

/* Prediction rejected */
.rejected {
    background: #3D2027;
    border: 2px solid #FF5C70;
    border-radius: 15px;
    padding: 25px;
    text-align: center;
    margin-bottom: 20px;
}

.rejected-title {
    color: #FF5C70;
    font-size: 32px;
    font-weight: 800;
}

/* Confidence */
.confidence {
    color: #FFFFFF;
    font-size: 20px;
    font-weight: 600;
}

/* Factor cards */
.factor-card {
    background: #102F4D;
    border-left: 5px solid #16C6FF;
    border-radius: 10px;
    padding: 13px 18px;
    margin-bottom: 10px;
}

.factor-name {
    color: #FFFFFF;
    font-size: 16px;
    font-weight: 700;
}

.factor-value {
    color: #8ED8FF;
    font-size: 14px;
}

.factor-direction {
    color: #FFFFFF;
    font-size: 14px;
    margin-top: 4px;
}

/* Footer */
.footer {
    text-align: center;
    color: #6E9AB8;
    margin-top: 40px;
    padding: 20px;
    font-size: 13px;
}

/* Hide Streamlit menu/footer */
#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# PATH
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "loan_approval_dataset.csv"
)


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():

    df = pd.read_csv(DATA_PATH)

    # Remove unwanted spaces from column names
    df.columns = df.columns.str.strip()

    # Remove spaces from text values
    for column in df.select_dtypes(include="object").columns:
        df[column] = df[column].str.strip()

    return df


df = load_data()


# =========================================================
# PREPARE DATA
# =========================================================

target_column = "loan_status"

features = [
    "no_of_dependents",
    "education",
    "self_employed",
    "income_annum",
    "loan_amount",
    "loan_term",
    "cibil_score",
    "residential_assets_value",
    "commercial_assets_value",
    "luxury_assets_value",
    "bank_asset_value"
]

categorical_features = [
    "education",
    "self_employed"
]

numeric_features = [
    "no_of_dependents",
    "income_annum",
    "loan_amount",
    "loan_term",
    "cibil_score",
    "residential_assets_value",
    "commercial_assets_value",
    "luxury_assets_value",
    "bank_asset_value"
]


X = df[features]
y = df[target_column]


# =========================================================
# MODEL
# =========================================================

@st.cache_resource
def train_model(X, y):

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                ),
                categorical_features
            ),
            (
                "num",
                "passthrough",
                numeric_features
            )
        ]
    )

    model = RandomForestClassifier(
        n_estimators=150,
        random_state=42,
        class_weight="balanced"
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    return pipeline, accuracy


model, accuracy = train_model(X, y)


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">🏦 Loan Approval Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered loan eligibility analysis using Machine Learning'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# APPLICATION ANALYTICS
# =========================================================

st.markdown(
    '<div class="section-title">📊 Application Analytics</div>',
    unsafe_allow_html=True
)


# ---------------------------------------------------------
# GRAPH 1 - APPROVED VS REJECTED
# ---------------------------------------------------------

col1, col2 = st.columns(2)


with col1:

    status_counts = (
        df["loan_status"]
        .value_counts()
        .reset_index()
    )

    status_counts.columns = [
        "Loan Status",
        "Applications"
    ]

    fig_status = px.bar(
        status_counts,
        x="Loan Status",
        y="Applications",
        title="Approved vs Rejected Applications",
        text="Applications"
    )

    fig_status.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0D2742",
        plot_bgcolor="#0D2742",
        font=dict(color="white"),
        title_font=dict(size=20),
        margin=dict(l=20, r=20, t=60, b=20)
    )

    fig_status.update_traces(
        textposition="outside"
    )

    st.plotly_chart(
        fig_status,
        use_container_width=True
    )


# ---------------------------------------------------------
# GRAPH 2 - CIBIL VS LOAN AMOUNT
# ---------------------------------------------------------

with col2:

    fig_cibil = px.scatter(
        df,
        x="cibil_score",
        y="loan_amount",
        color="loan_status",
        hover_data=[
            "income_annum",
            "loan_term"
        ],
        title="CIBIL Score vs Loan Amount"
    )

    fig_cibil.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0D2742",
        plot_bgcolor="#0D2742",
        font=dict(color="white"),
        title_font=dict(size=20),
        margin=dict(l=20, r=20, t=60, b=20)
    )

    st.plotly_chart(
        fig_cibil,
        use_container_width=True
    )


# =========================================================
# APPLICANT INFORMATION
# =========================================================

st.markdown(
    '<div class="section-title">👤 Applicant Information</div>',
    unsafe_allow_html=True
)


st.markdown(
    '<div class="card">',
    unsafe_allow_html=True
)


# First row
col1, col2, col3 = st.columns(3)

with col1:
    no_of_dependents = st.number_input(
        "Number of Dependents",
        min_value=0,
        max_value=20,
        value=2,
        step=1
    )

with col2:
    education = st.selectbox(
        "Education",
        ["Graduate", "Not Graduate"]
    )

with col3:
    self_employed = st.selectbox(
        "Self Employed",
        ["Yes", "No"]
    )


# Second row
col1, col2, col3 = st.columns(3)

with col1:
    income_annum = st.number_input(
        "Annual Income (₹)",
        min_value=0,
        value=500000,
        step=10000
    )

with col2:
    loan_amount = st.number_input(
        "Loan Amount (₹)",
        min_value=0,
        value=2000000,
        step=10000
    )

with col3:
    loan_term = st.number_input(
        "Loan Term (Years)",
        min_value=1,
        max_value=50,
        value=10,
        step=1
    )


# Third row
col1, col2, col3 = st.columns(3)

with col1:
    cibil_score = st.number_input(
        "CIBIL Score",
        min_value=300,
        max_value=900,
        value=700,
        step=1
    )

with col2:
    residential_assets_value = st.number_input(
        "Residential Assets (₹)",
        min_value=0,
        value=2000000,
        step=10000
    )

with col3:
    commercial_assets_value = st.number_input(
        "Commercial Assets (₹)",
        min_value=0,
        value=1000000,
        step=10000
    )


# Fourth row
col1, col2, col3 = st.columns(3)

with col1:
    luxury_assets_value = st.number_input(
        "Luxury Assets (₹)",
        min_value=0,
        value=500000,
        step=10000
    )

with col2:
    bank_asset_value = st.number_input(
        "Bank Assets (₹)",
        min_value=0,
        value=1000000,
        step=10000
    )

with col3:
    st.write("")


st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# PREDICTION
# =========================================================

st.markdown(
    '<div class="section-title">🔍 Loan Prediction</div>',
    unsafe_allow_html=True
)


predict_button = st.button(
    "🔍 Predict Loan Approval",
    type="primary",
    use_container_width=True
)


# =========================================================
# INDIVIDUAL SHAP EXPLANATION
# =========================================================

def get_individual_explanation(input_data):

    preprocessor = model.named_steps["preprocessor"]
    rf_model = model.named_steps["model"]

    transformed_data = preprocessor.transform(input_data)

    # Get transformed feature names
    feature_names = preprocessor.get_feature_names_out()

    try:

        explainer = shap.TreeExplainer(rf_model)

        shap_values = explainer.shap_values(
            transformed_data
        )

        # Handle different SHAP output formats
        if isinstance(shap_values, list):

            class_names = list(
                rf_model.classes_
            )

            predicted_class_index = list(
                class_names
            ).index(
                model.predict(input_data)[0]
            )

            values = shap_values[
                predicted_class_index
            ][0]

        else:

            shap_array = shap_values

            if len(shap_array.shape) == 3:

                predicted_class_index = list(
                    rf_model.classes_
                ).index(
                    model.predict(input_data)[0]
                )

                values = shap_array[
                    0,
                    :,
                    predicted_class_index
                ]

            else:

                values = shap_array[0]

    except Exception:

        # Fallback if SHAP has a version compatibility issue
        importances = rf_model.feature_importances_

        values = (
            transformed_data[0]
            * importances
        )

    explanation_df = pd.DataFrame({
        "feature": feature_names,
        "impact": values
    })

    # Remove preprocessing prefixes
    explanation_df["feature"] = (
        explanation_df["feature"]
        .str.replace(
            "cat__",
            "",
            regex=False
        )
        .str.replace(
            "num__",
            "",
            regex=False
        )
    )

    # -----------------------------------------------------
    # Combine one-hot categorical features
    # -----------------------------------------------------

    grouped_features = {}

    for _, row in explanation_df.iterrows():

        feature = row["feature"]
        impact = row["impact"]

        if feature.startswith("education_"):

            key = "Education"

        elif feature.startswith("self_employed_"):

            key = "Self Employment"

        else:

            key = feature

        grouped_features[key] = (
            grouped_features.get(key, 0)
            + impact
        )

    result = pd.DataFrame(
        list(grouped_features.items()),
        columns=[
            "Feature",
            "Impact"
        ]
    )

    result["Absolute Impact"] = (
        result["Impact"].abs()
    )

    result = result.sort_values(
        "Absolute Impact",
        ascending=False
    )

    return result.head(5)


# =========================================================
# SHOW PREDICTION
# =========================================================

if predict_button:

    applicant = pd.DataFrame(
        {
            "no_of_dependents": [
                no_of_dependents
            ],
            "education": [
                education
            ],
            "self_employed": [
                self_employed
            ],
            "income_annum": [
                income_annum
            ],
            "loan_amount": [
                loan_amount
            ],
            "loan_term": [
                loan_term
            ],
            "cibil_score": [
                cibil_score
            ],
            "residential_assets_value": [
                residential_assets_value
            ],
            "commercial_assets_value": [
                commercial_assets_value
            ],
            "luxury_assets_value": [
                luxury_assets_value
            ],
            "bank_asset_value": [
                bank_asset_value
            ]
        }
    )

    prediction = model.predict(
        applicant
    )[0]

    probabilities = model.predict_proba(
        applicant
    )[0]

    class_list = list(
        model.classes_
    )

    prediction_index = class_list.index(
        prediction
    )

    confidence = (
        probabilities[prediction_index]
        * 100
    )


    # =====================================================
    # RESULT + INDIVIDUAL FACTORS
    # =====================================================

    result_col, factor_col = st.columns(
        [1, 1]
    )


    # -----------------------------------------------------
    # PREDICTION RESULT
    # -----------------------------------------------------

    with result_col:

        if prediction == "Approved":

            st.markdown(
                f"""
                <div class="approved">
                    <div class="approved-title">
                        ✅ LOAN APPROVED
                    </div>
                    <div class="confidence">
                        Confidence: {confidence:.2f}%
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                f"""
                <div class="rejected">
                    <div class="rejected-title">
                        ❌ LOAN REJECTED
                    </div>
                    <div class="confidence">
                        Confidence: {confidence:.2f}%
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )


    # -----------------------------------------------------
    # INDIVIDUAL FACTORS
    # -----------------------------------------------------

    with factor_col:

        st.markdown(
            '<div class="section-title">'
            '🎯 Individual Factors'
            '</div>',
            unsafe_allow_html=True
        )

        factors = get_individual_explanation(
            applicant
        )

        display_names = {
            "no_of_dependents":
                "Number of Dependents",

            "income_annum":
                "Annual Income",

            "loan_amount":
                "Loan Amount",

            "loan_term":
                "Loan Term",

            "cibil_score":
                "CIBIL Score",

            "residential_assets_value":
                "Residential Assets",

            "commercial_assets_value":
                "Commercial Assets",

            "luxury_assets_value":
                "Luxury Assets",

            "bank_asset_value":
                "Bank Assets"
        }


        for _, row in factors.iterrows():

            feature = row["Feature"]

            impact = row["Impact"]

            readable_name = display_names.get(
                feature,
                feature.replace("_", " ").title()
            )

            if prediction == "Approved":

                if impact > 0:

                    direction = (
                        "✓ Supports approval"
                    )

                else:

                    direction = (
                        "⚠ Works against approval"
                    )

            else:

                if impact > 0:

                    direction = (
                        "✓ Supports rejection"
                    )

                else:

                    direction = (
                        "⚠ Works against rejection"
                    )


            st.markdown(
                f"""
                <div class="factor-card">
                    <div class="factor-name">
                        {readable_name}
                    </div>
                    <div class="factor-value">
                        Model impact: {impact:.4f}
                    </div>
                    <div class="factor-direction">
                        {direction}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )


    # =====================================================
    # INDIVIDUAL FACTOR GRAPH
    # =====================================================

    st.markdown(
        '<div class="section-title">'
        '📈 Factors Affecting This Applicant'
        '</div>',
        unsafe_allow_html=True
    )

    graph_df = factors.copy()

    graph_df["Feature"] = (
        graph_df["Feature"]
        .map(
            lambda x: display_names.get(
                x,
                x.replace("_", " ").title()
            )
        )
    )

    graph_df = graph_df.sort_values(
        "Impact"
    )

    fig_factors = px.bar(
        graph_df,
        x="Impact",
        y="Feature",
        orientation="h",
        title="Top 5 Individual Model Factors",
        text="Impact"
    )

    fig_factors.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0D2742",
        plot_bgcolor="#0D2742",
        font=dict(color="white"),
        title_font=dict(size=20),
        margin=dict(l=20, r=20, t=60, b=20),
        xaxis_title="Model Impact",
        yaxis_title=""
    )

    fig_factors.update_traces(
        texttemplate="%{text:.3f}",
        textposition="outside"
    )

    st.plotly_chart(
        fig_factors,
        use_container_width=True
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">
        Loan Approval Prediction System •
        Random Forest + SHAP Explainability
    </div>
    """,
    unsafe_allow_html=True
)