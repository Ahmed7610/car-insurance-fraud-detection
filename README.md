# 🚗 Car Insurance Fraud Detection — Complete Project

## 📁 Project Structure

```
fraud_project/
│
├── notebook/
│   └── car_insurance_fraud_detection.ipynb   ← Run this first in Google Colab
│
├── model_artifacts/                           ← Created by the notebook
│   ├── model.pkl                              ← Trained ML model
│   ├── scaler.pkl                             ← StandardScaler (fitted on training data)
│   ├── label_encoders.pkl                     ← LabelEncoders for text columns
│   └── feature_columns.pkl                    ← Column order used during training
│
├── api/
│   └── main.py                                ← FastAPI server
│
├── streamlit_app/
│   └── app.py                                 ← Streamlit web interface
│
├── requirements.txt                           ← All dependencies
└── README.md                                  ← This file
```

---

## 🧠 Understanding Each Part

### Part 1: The Notebook (Google Colab)
**What it does:** Loads the data, trains 3 ML models, picks the best one, and saves it.

**Key output:** 4 `.pkl` files inside `model_artifacts/`

**Run it:** Upload to Google Colab, run all cells top to bottom, then download the `.pkl` files.

---

### Part 2: Pickle Files (`.pkl`)
**What is Pickle?**
Pickle is Python's built-in way to save any object to a file.

A trained ML model is just a Python object with numbers inside it (the learned patterns).
Without saving it, you'd have to retrain from scratch every time.

```python
# Saving
import pickle
with open('model.pkl', 'wb') as f:   # 'wb' = write binary
    pickle.dump(my_model, f)

# Loading
with open('model.pkl', 'rb') as f:   # 'rb' = read binary
    my_model = pickle.load(f)
```

**Why save 4 files, not just the model?**
When new data arrives, it must go through the EXACT same transformations
as the training data used. So we save:
- `model.pkl` → the trained model itself
- `scaler.pkl` → to normalize new data the same way
- `label_encoders.pkl` → to encode text fields the same way
- `feature_columns.pkl` → to ensure column order matches

---

### Part 3: FastAPI (`api/main.py`)
**What is an API?**
An API (Application Programming Interface) is a way for programs to talk to each other.
Our API is a web server that:
- Listens for incoming claim data (sent as JSON)
- Runs it through our model
- Returns a prediction (also as JSON)

**What is FastAPI?**
FastAPI is a Python library that makes building APIs easy. It handles:
- Receiving HTTP requests
- Validating input data
- Generating interactive documentation automatically

**How data flows through the API:**
```
1. Client sends POST /predict with JSON data
2. FastAPI validates the JSON against ClaimInput schema
3. Categorical fields are encoded with saved LabelEncoders
4. Features are scaled with saved StandardScaler
5. Model predicts fraud probability
6. Result is returned as JSON
```

**The auto-generated docs:**
When the API is running, visit http://127.0.0.1:8000/docs
You can test the API directly from your browser — no code needed!

---

### Part 4: Streamlit (`streamlit_app/app.py`)
**What is Streamlit?**
Streamlit is a Python library that creates interactive web apps from plain Python scripts.
No HTML, CSS, or JavaScript needed — just Python.

**How it connects to the model:**
The Streamlit app does NOT load the model directly.
Instead, it sends the user's form data to the FastAPI server and displays the response.

```
[User fills form] → [Streamlit sends request] → [FastAPI runs model] → [Streamlit shows result]
```

This separation is good practice because:
- The model logic lives in one place (the API)
- Multiple apps (mobile, web, etc.) can use the same API
- Easy to update the model without changing the UI

---

## 🚀 How to Run Everything

### Step 1: Run the Notebook
1. Open Google Colab
2. Upload `car_insurance_fraud_detection.ipynb`
3. Upload `car_insurance_fraud_dataset.csv`
4. Run all cells
5. Download the 4 `.pkl` files from `/content/model_artifacts/`

### Step 2: Set Up Your Local Environment
```bash
# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate      # Mac/Linux
venv\Scripts\activate         # Windows

# Install all dependencies
pip install -r requirements.txt
```

### Step 3: Place the Pickle Files
```
fraud_project/
└── model_artifacts/
    ├── model.pkl
    ├── scaler.pkl
    ├── label_encoders.pkl
    └── feature_columns.pkl
```

### Step 4: Start the FastAPI Server
```bash
cd api/
uvicorn main:app --reload
```
→ API is now running at http://127.0.0.1:8000
→ Interactive docs at http://127.0.0.1:8000/docs

### Step 5: Start the Streamlit App (in a NEW terminal)
```bash
cd streamlit_app/
streamlit run app.py
```
→ Web app opens automatically at http://localhost:8501

---

## 🌐 API Endpoints

| Method | URL | Description |
|--------|-----|-------------|
| GET | `/` | Health check — is the server alive? |
| POST | `/predict` | Submit claim data, get fraud prediction |
| GET | `/model-info` | See model type and feature names |
| GET | `/docs` | Interactive API documentation (auto-generated) |

### Example API Request
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "policy_state": "CA",
       "policy_deductible": 500,
       "policy_annual_premium": 1350.75,
       "insured_age": 42,
       "insured_sex": "MALE",
       "insured_education_level": "College",
       "insured_occupation": "Manager",
       "insured_hobbies": "reading",
       "incident_type": "Multi-vehicle Collision",
       "collision_type": "Rear",
       "incident_severity": "Total Loss",
       "authorities_contacted": "Police",
       "incident_state": "OH",
       "incident_hour_of_the_day": 14,
       "number_of_vehicles_involved": 2,
       "bodily_injuries": 1,
       "witnesses": 2,
       "police_report_available": "Yes",
       "claim_amount": 45000.00,
       "total_claim_amount": 50000.00
     }'
```

### Example API Response
```json
{
  "prediction": "FRAUD",
  "fraud_probability": 0.7832,
  "confidence": "HIGH"
}
```

---

## 🔑 Key Concepts Summary

| Concept | Plain English Explanation |
|---|---|
| **Pickle** | Saving a trained Python object to a file so you don't have to retrain |
| **API** | A door that lets programs talk to each other over the internet |
| **FastAPI** | Python library for building APIs quickly with automatic validation and docs |
| **Pydantic** | Data validation library used by FastAPI to check incoming JSON |
| **POST request** | Sending data TO a server (vs GET which just fetches data) |
| **JSON** | A text format for exchanging data between programs — like a Python dict |
| **Streamlit** | Turns a Python script into a web app without any HTML/CSS/JS |
| **`requests` library** | Python library for making HTTP requests (used by Streamlit to call the API) |
