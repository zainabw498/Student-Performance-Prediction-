# 🎓 Student Performance Prediction System
### Final Year Project · Logistic Regression · Streamlit Web Application

---

## 📁 Project Structure

```
project_root/
├── app.py                  # Streamlit app (replaces Flask)
├── train_model.py          # Model training script (unchanged)
├── student_data.csv        # Synthetic dataset (200 rows)
├── requirements.txt        # Python dependencies
├── model.pkl               # Trained Logistic Regression model (generated)
├── scaler.pkl              # StandardScaler transformer (generated)
├── model_metadata.json     # Feature coefficients + accuracy (generated)
└── README.md               # This file
```

> **Note:** The `templates/` folder is no longer needed — Streamlit renders HTML directly from Python.

---

## ⚙️ How to Run

### Step 1 — Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2 — Train the Model

```bash
python train_model.py
```

This will generate `model.pkl`, `scaler.pkl`, and `model_metadata.json`.

### Step 3 — Start the Streamlit App

```bash
streamlit run app.py
```

### Step 4 — Open the Dashboard

Streamlit will automatically open your browser at: **http://localhost:8501**

---

## 🧠 Key Mathematical Concepts

| Concept | Implementation |
|---|---|
| Decision Threshold | P(y=1) > 0.50 → Predict Pass |
| Probability | `model.predict_proba(X)[:, 1]` |
| Feature Normalisation | `StandardScaler` (μ=0, σ=1) |
| Coefficient Analysis | `model.coef_[0]` used for justification |
| Log-odds | log(P/1-P) = Σ βᵢxᵢ + β₀ |

---

## 📊 Dataset Features

| Feature | Range | Description |
|---|---|---|
| Study_Hours | 1–10 | Weekly study hours |
| Attendance_Rate | 60–100% | Class attendance percentage |
| Previous_Grades | 0–100 | Previous academic grade |
| Participation_Score | 1–10 | In-class participation |
| Parental_Involvement | 0/1/2 | Low / Medium / High |

---

## 🔄 Flask → Streamlit Migration Notes

| Flask concept | Streamlit equivalent |
|---|---|
| `app.route("/")` + `render_template` | `st.markdown()` + layout columns |
| `app.route("/predict")` POST endpoint | Direct Python call on button click |
| `{{ model_accuracy }}` Jinja2 template variable | f-string inside `st.markdown()` |
| JavaScript `fetch('/predict')` | No JS needed — Streamlit reruns on interaction |
| `templates/index.html` | Inline HTML/CSS via `st.markdown(unsafe_allow_html=True)` |
