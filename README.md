# 🚗 Car Insurance Fraud Detection

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green?logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red?logo=streamlit)
![LightGBM](https://img.shields.io/badge/Model-LightGBM-orange)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

A complete end-to-end Machine Learning project that detects fraudulent car insurance claims.
Built with a full ML pipeline: data exploration, SMOTE balancing, model tuning, a REST API, and an interactive web interface.

---

## 📸 Demo

> Fill in a claim → get an instant fraud prediction with confidence score.

<!-- After you take a screenshot of the Streamlit app, upload it to your repo
     and replace the line below with: ![Demo](screenshots/demo.png) -->
*Screenshot coming soon*

---

## 🧠 What This Project Does

Insurance companies lose billions every year to fraudulent claims. This project builds
a machine learning model that looks at a claim's details and predicts:

> **"Is this claim FRAUD or LEGITIMATE?"**

---

## 🏗️ Project Architecture

```
User fills Streamlit form
        ↓
Streamlit sends claim data (JSON)
        ↓
FastAPI receives request → loads model → runs prediction
        ↓
Returns: prediction + fraud probability + confidence
        ↓
Streamlit displays the result
```

---

## 📁 Project Structure

```
car-insurance-fraud-detection/
│
├── notebook/
│   ├── car_insurance_fraud_detection.ipynb        # Original model (v1)
│   └── car_insurance_fraud_detection_v2.ipynb     # Improved model (v2) ← recommended
│
├── api/
│   └── main.py             # FastAPI server — serves predictions via HTTP
│
├── streamlit_app/
│   └── app.py              # Streamlit web interface
│
├── model_artifacts/        # ← YOU must generate these by running the notebook
│   ├── model.pkl           # Trained LightGBM model
│   ├── scaler.pkl          # StandardScaler (fitted on training data)
│   ├── label_encoders.pkl  # LabelEncoders for categorical columns
│   └── feature_columns.pkl # Feature column order
│
├── requirements.txt        # All dependencies
└── README.md
```

---

## ⚙️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| ML Models | LightGBM, XGBoost, Random Forest |
| Imbalance Fix | SMOTE (imbalanced-learn) |
| Tuning | GridSearchCV |
| API | FastAPI + Uvicorn |
| Web Interface | Streamlit |
| Model Saving | Pickle |

---

## 🚀 How to Run This Project (Step by Step)

### Prerequisites
- Python 3.10 or higher installed
- Git installed
- A Google account (for running the notebook in Colab)

---

### Step 1 — Clone the Repository

```bash
git clone https://github.com/Ahmed7610/car-insurance-fraud-detection.git
cd car-insurance-fraud-detection
```

---

### Step 2 — Install Dependencies

```bash
pip install -r requirements.txt
```

On Ubuntu, if you get a permissions error:
```bash
pip install -r requirements.txt --break-system-packages
```

---

### Step 3 — Generate the Model Files (pkl files)

The trained model files are not included in this repo (they are too large).
You need to generate them yourself by running the notebook.

1. Go to [Google Colab](https://colab.research.google.com)
2. Upload `notebook/car_insurance_fraud_detection_v2.ipynb`
3. Upload the dataset file `car_insurance_fraud_dataset.csv`
4. Click **Runtime → Restart session and run all**
5. Wait for all cells to finish — the last cell downloads 4 files automatically
6. Move all 4 files into the `model_artifacts/` folder

---

### Step 4 — Start the FastAPI Server (Terminal 1)

```bash
cd api/
uvicorn main:app --reload
```

You should see:
```
✅ Model and artifacts loaded successfully!
INFO: Uvicorn running on http://127.0.0.1:8000
```

> 💡 Visit **http://127.0.0.1:8000/docs** to test the API in your browser

---

### Step 5 — Start the Streamlit App (Terminal 2)

Open a **new terminal** — keep Terminal 1 running!

```bash
cd streamlit_app/
streamlit run app.py
```

Your browser opens at **http://localhost:8501** — fill in the form and click Predict.

---

## 🔌 API Reference

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check |
| POST | `/predict` | Submit a claim, get fraud prediction |
| GET | `/model-info` | See model type and features |
| GET | `/docs` | Interactive API documentation |

### Example Response

```json
{
  "prediction": "FRAUD",
  "fraud_probability": 0.7832,
  "confidence": "HIGH"
}
```

---

## 🧪 ML Pipeline Summary

```
Raw Data (30,000 claims)
        ↓
Drop irrelevant columns → Label Encode → Train/Test Split
        ↓
StandardScaler → SMOTE (train only) → 5-Fold Cross Validation
        ↓
GridSearchCV (hyperparameter tuning)
        ↓
Best Model: LightGBM
        ↓
Save with Pickle → Deploy with FastAPI → Serve via Streamlit
```

---

## 📊 Model Performance

| Model | Accuracy | ROC-AUC |
|---|---|---|
| Random Forest | ~0.85 | ~0.88 |
| XGBoost | ~0.87 | ~0.91 |
| LightGBM (tuned) | ~0.88 | ~0.93 |

---

## 💡 Key Concepts Used

| Concept | Why it matters |
|---|---|
| **SMOTE** | Fixes class imbalance — without it, model ignores fraud cases |
| **StratifiedKFold** | Keeps fraud ratio equal across all CV folds |
| **GridSearchCV** | Finds optimal hyperparameters automatically |
| **LightGBM** | Faster and more accurate than Random Forest on tabular data |
| **Pickle** | Saves trained model so we don't retrain on every request |
| **FastAPI** | Serves the model as an HTTP API any app can call |
| **Streamlit** | Turns a Python script into a usable web interface |

---

## 👤 Author

**Ahmed** — [@Ahmed7610](https://github.com/Ahmed7610)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
