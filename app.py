import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from xgboost import XGBClassifier

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="FraudShield AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "fraud_model.json"
FEATURE_PATH = BASE_DIR / "models" / "feature_names.joblib"
METADATA_PATH = BASE_DIR / "models" / "model_metadata.joblib"


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = XGBClassifier()

    model.load_model(MODEL_PATH)

    return model


# ============================================================
# LOAD FEATURE NAMES
# ============================================================

@st.cache_data
def load_features():

    return joblib.load(FEATURE_PATH)


# ============================================================
# LOAD MODEL METADATA
# ============================================================

@st.cache_data
def load_metadata():

    return joblib.load(METADATA_PATH)


# ============================================================
# LOAD ALL MODEL FILES
# ============================================================

try:

    model = load_model()
    feature_names = load_features()
    metadata = load_metadata()

except Exception as e:

    st.error("Model loading failed.")

    st.code(str(e))

    st.info(
        """
        Please make sure the following files exist:

        models/
        ├── fraud_model.json
        ├── feature_names.joblib
        └── model_metadata.joblib
        """
    )

    st.stop()


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main {
        background-color: #f8fafc;
    }

    .hero {
        padding: 40px;
        border-radius: 20px;
        background: linear-gradient(
            135deg,
            #0f172a,
            #172554
        );
        color: white;
        margin-bottom: 30px;
    }

    .hero h1 {
        font-size: 48px;
        margin-bottom: 10px;
    }

    .hero p {
        font-size: 20px;
        color: #cbd5e1;
    }

    .card {
        padding: 25px;
        border-radius: 15px;
        background-color: white;
        border: 1px solid #e2e8f0;
        box-shadow: 0px 4px 15px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }

    .risk-low {
        padding: 20px;
        border-radius: 12px;
        background-color: #dcfce7;
        border: 1px solid #22c55e;
    }

    .risk-medium {
        padding: 20px;
        border-radius: 12px;
        background-color: #fef9c3;
        border: 1px solid #eab308;
    }

    .risk-high {
        padding: 20px;
        border-radius: 12px;
        background-color: #fed7aa;
        border: 1px solid #f97316;
    }

    .risk-critical {
        padding: 20px;
        border-radius: 12px;
        background-color: #fee2e2;
        border: 1px solid #ef4444;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_risk_level(probability):

    if probability < 0.30:

        return "Low"

    elif probability < 0.70:

        return "Medium"

    elif probability < 0.90:

        return "High"

    else:

        return "Critical"


def get_recommended_action(probability, threshold):

    if probability < threshold:

        return "Approve"

    elif probability < 0.90:

        return "Review Manually"

    else:

        return "Temporarily Block"


def get_risk_class(risk):

    if risk == "Low":

        return "risk-low"

    elif risk == "Medium":

        return "risk-medium"

    elif risk == "High":

        return "risk-high"

    else:

        return "risk-critical"


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🛡️ FraudShield AI")

st.sidebar.write(
    "AI-powered transaction risk detection"
)

st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Fraud Detection",
        "About"
    ]
)

st.sidebar.divider()

st.sidebar.caption(
    "Educational / portfolio demonstration"
)


# ============================================================
# HOME PAGE
# ============================================================

if page == "Home":

    st.markdown(
        """
        <div class="hero">

        <h1>🛡️ FraudShield AI</h1>

        <p>
        Detect Fraud. Protect Every Transaction.
        </p>

        <p>
        Machine-learning powered transaction risk
        detection for modern financial systems.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.subheader("Model Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "ROC-AUC",
        f"{metadata['roc_auc']:.3f}"
    )

    col2.metric(
        "PR-AUC",
        f"{metadata['pr_auc']:.3f}"
    )

    col3.metric(
        "Fraud Recall",
        f"{metadata['recall']:.1%}"
    )

    col4.metric(
        "Features",
        metadata["n_features"]
    )

    st.divider()

    st.header("How FraudShield AI Works")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown(
            """
            ### 1️⃣ Enter Transaction

            Provide transaction information including
            transaction time, amount, and anonymized
            transaction features.
            """
        )

    with col2:

        st.markdown(
            """
            ### 2️⃣ AI Risk Analysis

            The trained XGBoost model calculates the
            probability that the transaction is fraudulent.
            """
        )

    with col3:

        st.markdown(
            """
            ### 3️⃣ Review Risk

            The system converts the model probability
            into a risk level and suggested action.
            """
        )

    st.divider()

    st.header("Key Capabilities")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            """
            **🎯 Risk Scoring**

            Calculates a fraud probability for every
            transaction.

            **📊 Imbalanced-Learning Awareness**

            Evaluates fraud using precision, recall,
            F1, ROC-AUC and PR-AUC rather than accuracy alone.
            """
        )

    with col2:

        st.markdown(
            """
            **⚙️ Configurable Threshold**

            Fraud decisions can be adjusted according
            to business risk and cost.

            **🔍 Human Review**

            High-risk transactions can be flagged
            for additional investigation.
            """
        )

    st.divider()

    st.info(
        """
        ⚠️ FraudShield AI is an educational and portfolio
        demonstration. Model predictions are risk signals,
        not proof that a transaction is fraudulent.
        """
    )


# ============================================================
# FRAUD DETECTION PAGE
# ============================================================

elif page == "Fraud Detection":

    st.title("🔎 Fraud Detection")

    st.write(
        """
        Enter the transaction features below and let
        FraudShield AI estimate the probability of fraud.
        """
    )

    st.divider()

    # --------------------------------------------------------
    # Transaction Inputs
    # --------------------------------------------------------

    input_values = {}

    st.subheader("Transaction Information")

    # Amount and Time first
    col1, col2 = st.columns(2)

    with col1:

        input_values["Time"] = st.number_input(
            "Time",
            value=0.0,
            help="Seconds elapsed from the beginning of the dataset."
        )

    with col2:

        input_values["Amount"] = st.number_input(
            "Amount",
            min_value=0.0,
            value=10.0,
            help="Transaction amount."
        )

    st.subheader("Anonymized Transaction Features")

    cols = st.columns(4)

    for index, feature in enumerate(feature_names):

        with cols[index % 4]:

            input_values[feature] = st.number_input(
                feature,
                value=0.0,
                format="%.6f"
            )

    st.divider()

    analyze = st.button(
        "🔍 Analyze Transaction",
        use_container_width=True
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    if analyze:

        input_df = pd.DataFrame(
            [input_values],
            columns=feature_names
        )

        probability = model.predict_proba(
            input_df
        )[0][1]

        threshold = float(
            metadata["threshold"]
        )

        prediction = int(
            probability >= threshold
        )

        risk = get_risk_level(
            probability
        )

        action = get_recommended_action(
            probability,
            threshold
        )

        # ----------------------------------------------------
        # Results
        # ----------------------------------------------------

        st.divider()

        st.header("Transaction Risk Assessment")

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Fraud Probability",
                f"{probability:.2%}"
            )

        with col2:

            st.metric(
                "Risk Level",
                risk
            )

        with col3:

            st.metric(
                "Recommended Action",
                action
            )

        st.progress(
            float(probability)
        )

        st.divider()

        if prediction == 1:

            st.error(
                "🚨 Potential Fraud Detected"
            )

        else:

            st.success(
                "✅ Transaction Classified as Legitimate"
            )

        risk_class = get_risk_class(risk)

        st.markdown(
            f"""
            <div class="{risk_class}">

            <h3>Risk Assessment</h3>

            <p>
            <strong>Fraud Probability:</strong>
            {probability:.2%}
            </p>

            <p>
            <strong>Risk Level:</strong>
            {risk}
            </p>

            <p>
            <strong>Suggested Action:</strong>
            {action}
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )

        st.divider()

        st.caption(
            """
            This prediction is a machine-learning risk signal.
            It should not be treated as definitive proof of fraud.
            Production systems should combine model predictions
            with transaction rules, monitoring, human review,
            security controls and appropriate governance.
            """
        )


# ============================================================
# ABOUT PAGE
# ============================================================

elif page == "About":

    st.title("ℹ️ About FraudShield AI")

    st.markdown(
        """
        ## What is FraudShield AI?

        FraudShield AI is a machine-learning based credit-card
        transaction fraud detection application.

        The system analyzes transaction features and produces
        a probability that a transaction may be fraudulent.

        ## Why Fraud Detection is Challenging

        Fraud detection is an extremely imbalanced classification
        problem.

        In the dataset used for this project, legitimate
        transactions greatly outnumber fraudulent transactions.

        Therefore, accuracy alone can be misleading.

        A useful fraud detection system should consider:

        - Precision
        - Recall
        - F1-score
        - ROC-AUC
        - PR-AUC
        - False positives
        - False negatives
        - Business costs

        ## Model

        FraudShield AI uses an XGBoost classification model.

        The model was trained using transaction features including:

        - Time
        - V1–V28 anonymized features
        - Amount

        ## Important Limitation

        V1–V28 are anonymized features. Their individual meanings
        are not available in the public dataset.

        Therefore, the application should not claim that a particular
        V feature represents a specific customer behavior.

        ## Production Considerations

        A real financial institution would require much more than
        a machine-learning model, including:

        - Real-time API integration
        - Model monitoring
        - Data drift detection
        - Security controls
        - Authentication
        - Audit logging
        - Human investigation workflows
        - Model retraining
        - Governance
        - Privacy protection

        ## Disclaimer

        FraudShield AI is an educational and portfolio project.

        It is not a live banking fraud prevention system and should
        not be used to make real financial decisions without proper
        validation, monitoring, security and governance.
        """
    )

    st.divider()

    st.success(
        "Built as an end-to-end machine learning portfolio project."
    )