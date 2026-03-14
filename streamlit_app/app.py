# =============================================================================
# app.py  —  Car Insurance Fraud Detection — Streamlit Web Interface
# =============================================================================
#
# WHAT IS STREAMLIT?
# ───────────────────
# Streamlit is a Python library that turns a plain Python script into an
# interactive web app — NO HTML, CSS, or JavaScript needed.
#
# You just write Python and Streamlit renders buttons, sliders, tables,
# and charts automatically in the browser.
#
# HOW DOES IT CONNECT TO THE MODEL?
# ───────────────────────────────────
# This app talks to the FastAPI server using the 'requests' library.
# The user fills in a form → Streamlit sends the data to FastAPI → FastAPI
# returns the prediction → Streamlit displays the result.
#
# This is the standard real-world architecture:
#   [User] → [Streamlit UI] → [FastAPI API] → [ML Model] → [Result]
#
# HOW TO RUN:
# ────────────
# 1. Make sure the FastAPI server is running first:
#      uvicorn main:app --reload   (in the api/ folder)
#
# 2. Then in a separate terminal, run the Streamlit app:
#      streamlit run app.py        (in the streamlit_app/ folder)
#
# 3. Your browser will open automatically at http://localhost:8501
# =============================================================================

import streamlit as st
import requests
import json

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
# Must be the FIRST Streamlit command in the script.
# Sets the browser tab title, icon, and layout.

st.set_page_config(page_title="Fraud Detection", page_icon="🚗", layout="wide")

# =============================================================================
# API URL
# =============================================================================
# This is the address of the FastAPI server.
# If you deploy both apps online, replace this with your deployed API URL.

API_URL = "http://127.0.0.1:8000/predict"


# =============================================================================
# HELPER FUNCTION: Call the API
# =============================================================================


def call_prediction_api(payload: dict) -> dict:
    """
    Sends claim data to the FastAPI server and returns the prediction.

    Args:
        payload: dictionary of claim features

    Returns:
        API response as a dictionary, or None on error
    """
    try:
        response = requests.post(
            API_URL,
            json=payload,
            timeout=10,  # Don't wait more than 10 seconds
        )
        response.raise_for_status()  # Raise error for HTTP 4xx / 5xx responses
        return response.json()

    except requests.exceptions.ConnectionError:
        st.error(
            "❌ Cannot connect to the API. Make sure FastAPI is running on port 8000."
        )
        return None
    except requests.exceptions.Timeout:
        st.error("⏱️ Request timed out. The server took too long to respond.")
        return None
    except requests.exceptions.HTTPError as e:
        try:
            detail = e.response.json().get("detail", str(e))
        except Exception:
            detail = e.response.text or str(e)
        st.error(f"API Error: {detail}")
        st.code(
            f"Status code: {e.response.status_code}\nRaw response: {e.response.text}"
        )
        return None


# =============================================================================
# PAGE HEADER
# =============================================================================

st.title("🚗 Car Insurance Fraud Detection")
st.markdown(
    "Fill in the claim details below and click **Predict** to check whether "
    "this claim is likely to be **FRAUDULENT** or **LEGITIMATE**."
)
st.divider()


# =============================================================================
# INPUT FORM
# =============================================================================
# st.form() groups all inputs into one block.
# The prediction only runs when the user clicks the submit button —
# not on every keystroke (which would be annoying and slow).

with st.form("claim_form"):
    st.subheader("📋 Policy Information")

    # ── Row 1: Policy details ─────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)

    with col1:
        policy_state = st.selectbox(
            "Policy State",
            options=["CA", "GA", "MI", "NC", "NY", "OH", "PA", "SC", "TX", "WA"],
            index=0,
        )
    with col2:
        policy_deductible = st.selectbox(
            "Policy Deductible ($)",
            options=[300, 400, 500, 600, 700, 1000, 2000],
            index=2,
        )
    with col3:
        policy_annual_premium = st.number_input(
            "Annual Premium ($)",
            min_value=100.0,
            max_value=5000.0,
            value=1200.0,
            step=50.0,
        )

    st.divider()
    st.subheader("👤 Insured Person")

    # ── Row 2: Insured person info ────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        insured_age = st.slider("Age", min_value=18, max_value=80, value=35)
    with col2:
        insured_sex = st.selectbox("Sex", options=["MALE", "FEMALE", "OTHER"])
    with col3:
        insured_education_level = st.selectbox(
            "Education",
            options=["High School", "College", "Masters", "PhD", "Associate"],
        )
    with col4:
        insured_occupation = st.selectbox(
            "Occupation",
            options=[
                "Manager",
                "Doctor",
                "Lawyer",
                "Teacher",
                "Engineer",
                "Accountant",
                "Sales",
                "Nurse",
                "Mechanic",
                "Exec-managerial",
            ],
        )

    insured_hobbies = st.selectbox(
        "Hobby",
        options=[
            "reading",
            "chess",
            "golf",
            "hiking",
            "movies",
            "paintball",
            "polo",
            "yachting",
            "base jumping",
            "skydiving",
        ],
    )

    st.divider()
    st.subheader("🚨 Incident Details")

    # ── Row 3: Incident info ──────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)

    with col1:
        incident_type = st.selectbox(
            "Incident Type",
            options=[
                "Multi-vehicle Collision",
                "Single Vehicle Collision",
                "Vehicle Theft",
                "Parked Car",
            ],
        )
    with col2:
        collision_type = st.selectbox(
            "Collision Type", options=["Front", "Rear", "Side", "Unknown"]
        )
    with col3:
        incident_severity = st.selectbox(
            "Severity",
            options=["Minor Damage", "Major Damage", "Total Loss", "Trivial Damage"],
        )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        authorities_contacted = st.selectbox(
            "Authorities Contacted",
            options=["Police", "Fire", "Ambulance", "Other", "None"],
        )
    with col2:
        incident_state = st.selectbox(
            "Incident State",
            options=["CA", "GA", "MI", "NC", "NY", "OH", "PA", "SC", "TX", "WA"],
        )
    with col3:
        incident_hour_of_the_day = st.slider("Hour of Incident (0–23)", 0, 23, 14)
    with col4:
        police_report_available = st.selectbox("Police Report?", options=["Yes", "No"])

    col1, col2, col3 = st.columns(3)

    with col1:
        number_of_vehicles_involved = st.number_input(
            "Vehicles Involved", min_value=1, max_value=6, value=1
        )
    with col2:
        bodily_injuries = st.number_input(
            "Bodily Injuries", min_value=0, max_value=6, value=0
        )
    with col3:
        witnesses = st.number_input("Witnesses", min_value=0, max_value=5, value=1)

    st.divider()
    st.subheader("💰 Claim Amounts")

    col1, col2 = st.columns(2)

    with col1:
        claim_amount = st.number_input(
            "Claim Amount ($)",
            min_value=0.0,
            max_value=200000.0,
            value=15000.0,
            step=500.0,
        )
    with col2:
        total_claim_amount = st.number_input(
            "Total Claim Amount ($)",
            min_value=0.0,
            max_value=200000.0,
            value=18000.0,
            step=500.0,
        )

    # ── Submit button ─────────────────────────────────────────────────────────
    # Every st.form() must end with st.form_submit_button()
    submitted = st.form_submit_button(
        "🔍 Predict Fraud", use_container_width=True, type="primary"
    )


# =============================================================================
# PREDICTION RESULT
# =============================================================================
# This code only runs AFTER the user clicks the submit button.

if submitted:
    # Build the payload dictionary matching ClaimInput in main.py
    payload = {
        "policy_state": policy_state,
        "policy_deductible": policy_deductible,
        "policy_annual_premium": policy_annual_premium,
        "insured_age": insured_age,
        "insured_sex": insured_sex,
        "insured_education_level": insured_education_level,
        "insured_occupation": insured_occupation,
        "insured_hobbies": insured_hobbies,
        "incident_type": incident_type,
        "collision_type": collision_type,
        "incident_severity": incident_severity,
        "authorities_contacted": authorities_contacted,
        "incident_state": incident_state,
        "incident_hour_of_the_day": int(incident_hour_of_the_day),
        "number_of_vehicles_involved": int(number_of_vehicles_involved),
        "bodily_injuries": int(bodily_injuries),
        "witnesses": int(witnesses),
        "police_report_available": police_report_available,
        "claim_amount": claim_amount,
        "total_claim_amount": total_claim_amount,
    }

    # Show a spinner while waiting for the API response
    with st.spinner("Analyzing claim..."):
        result = call_prediction_api(payload)

    # ── Display results ───────────────────────────────────────────────────────
    if result:
        st.divider()
        st.subheader("📊 Prediction Result")

        col1, col2, col3 = st.columns(3)

        # Main verdict
        with col1:
            if result["prediction"] == "FRAUD":
                st.error(f"🚨 **{result['prediction']}**")
            else:
                st.success(f"✅ **{result['prediction']}**")

        # Fraud probability as a progress bar
        with col2:
            prob = result["fraud_probability"]
            st.metric("Fraud Probability", f"{prob * 100:.1f}%")
            st.progress(prob)

        # Confidence level
        with col3:
            confidence = result["confidence"]
            color_map = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}
            st.metric("Confidence", f"{color_map.get(confidence, '')} {confidence}")

        # Show the raw JSON response (educational — shows what the API returned)
        with st.expander("🔧 Raw API Response (JSON)"):
            st.json(result)

        # Show the payload that was sent (helps with debugging)
        with st.expander("📤 Data Sent to API"):
            st.json(payload)
