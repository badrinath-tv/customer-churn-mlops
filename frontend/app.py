import streamlit as st
import requests


# =============================================================================
# CONFIGURATION
# =============================================================================

API_URL = "https://customer-churn-mlops-pvox.onrender.com/predict"


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
)


# =============================================================================
# CUSTOM CSS
# =============================================================================

st.markdown(
    """
    <style>
        .main-title {
            font-size: 38px;
            font-weight: 700;
            text-align: center;
            margin-bottom: 5px;
        }

        .subtitle {
            text-align: center;
            font-size: 18px;
            margin-bottom: 30px;
        }

        .result-box {
            padding: 25px;
            border-radius: 12px;
            text-align: center;
            margin-top: 20px;
        }

        .metric-value {
            font-size: 32px;
            font-weight: 700;
        }

        .footer {
            text-align: center;
            margin-top: 40px;
            font-size: 14px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# HEADER
# =============================================================================

st.markdown(
    '<div class="main-title">AI-Powered Customer Churn Prediction</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    "CatBoost + SMOTE | MLOps-Based Customer Prediction System"
    "</div>",
    unsafe_allow_html=True,
)


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:

    st.header("⚙️ Model Information")

    st.write("**Model:** CatBoost + SMOTE")
    st.write("**Pipeline:** MLOps")
    st.write("**Deployment:** Render")
    st.write("**API:** FastAPI")

    st.divider()

    st.write("### Model Performance")

    st.metric("Accuracy", "86.50%")
    st.metric("F1-Score", "63.51%")
    st.metric("ROC-AUC", "0.8703")


# =============================================================================
# CUSTOMER INPUT
# =============================================================================

st.header("👤 Customer Information")

col1, col2 = st.columns(2)


with col1:

    credit_score = st.number_input(
        "Credit Score",
        min_value=300.0,
        max_value=900.0,
        value=650.0,
        step=1.0,
    )

    geography = st.selectbox(
        "Geography",
        ["France", "Germany", "Spain"],
    )

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"],
    )

    age = st.number_input(
        "Age",
        min_value=18.0,
        max_value=100.0,
        value=40.0,
        step=1.0,
    )

    tenure = st.number_input(
        "Tenure",
        min_value=0.0,
        max_value=10.0,
        value=3.0,
        step=1.0,
    )


with col2:

    balance = st.number_input(
        "Balance",
        min_value=0.0,
        value=60000.0,
        step=1000.0,
    )

    num_products = st.number_input(
        "Number of Products",
        min_value=1.0,
        max_value=4.0,
        value=2.0,
        step=1.0,
    )

    has_card = st.selectbox(
        "Has Credit Card",
        ["Yes", "No"],
    )

    active_member = st.selectbox(
        "Is Active Member",
        ["Yes", "No"],
    )

    estimated_salary = st.number_input(
        "Estimated Salary",
        min_value=0.0,
        value=100000.0,
        step=1000.0,
    )


# =============================================================================
# CONVERT INPUT VALUES
# =============================================================================

has_card_value = 1 if has_card == "Yes" else 0
active_member_value = 1 if active_member == "Yes" else 0


# =============================================================================
# PREDICTION BUTTON
# =============================================================================

st.divider()

predict_button = st.button(
    "🔮 Predict Customer Churn",
    type="primary",
    use_container_width=True,
)


# =============================================================================
# PREDICTION
# =============================================================================

if predict_button:

    customer_data = {
        "CreditScore": credit_score,
        "Geography": geography,
        "Gender": gender,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_products,
        "HasCrCard": has_card_value,
        "IsActiveMember": active_member_value,
        "EstimatedSalary": estimated_salary,
    }

    try:

        with st.spinner("Connecting to prediction API..."):

            response = requests.post(
                API_URL,
                json=customer_data,
                timeout=90,
            )

        if response.status_code == 200:

            result = response.json()

            prediction = result["prediction"]
            churn_status = result["churn_status"]
            probability = result["churn_probability"]

            st.header("📊 Prediction Result")

            result_col1, result_col2 = st.columns(2)

            with result_col1:

                st.metric(
                    "Prediction",
                    churn_status,
                )

            with result_col2:

                st.metric(
                    "Churn Probability",
                    f"{probability * 100:.2f}%",
                )

            if prediction == 1:

                st.error(
                    "⚠️ The model predicts that this customer "
                    "is likely to churn."
                )

            else:

                st.success(
                    "✅ The model predicts that this customer "
                    "is unlikely to churn."
                )

            st.info(
                f"Model used: {result.get('model', 'CatBoost + SMOTE')}"
            )

        else:

            try:
                error_detail = response.json().get(
                    "detail",
                    response.text,
                )
            except Exception:
                error_detail = response.text

            st.error(
                f"Prediction API error ({response.status_code}): "
                f"{error_detail}"
            )

    except requests.exceptions.Timeout:

        st.error(
            "The prediction service took too long to respond. "
            "The Render free instance may be waking up."
        )

    except requests.exceptions.ConnectionError:

        st.error(
            "Could not connect to the prediction API. "
            "Please check whether the Render service is running."
        )

    except Exception as e:

        st.error(
            f"Unexpected error: {str(e)}"
        )


# =============================================================================
# FOOTER
# =============================================================================

st.divider()

st.markdown(
    '<div class="footer">'
    "AI-Powered Customer Churn Prediction with Automated MLOps Pipeline"
    "<br>"
    "CatBoost + SMOTE • FastAPI • MLflow • DVC • GitHub • Render • Evidently AI"
    "</div>",
    unsafe_allow_html=True,
)