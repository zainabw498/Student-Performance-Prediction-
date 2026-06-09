"""
app.py  —  Student Performance Prediction System (Streamlit Cloud Ready)
Model is trained at runtime from student_data.csv — no .pkl files needed.
"""

import json
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Student Analytics Dashboard", page_icon="🎓", layout="wide")

# ─────────────────────────────────────────────
# Train model at startup (cached — runs once)
# ─────────────────────────────────────────────
@st.cache_resource
def train_and_load():
    FEATURE_COLS = [
        "Study_Hours", "Attendance_Rate", "Previous_Grades",
        "Participation_Score", "Parental_Involvement",
    ]
    TARGET_COL = "Passed"

    df = pd.read_csv("student_data.csv")
    X  = df[FEATURE_COLS].values
    y  = df[TARGET_COL].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    model.fit(X_train_sc, y_train)

    accuracy  = accuracy_score(y_test, model.predict(X_test_sc))
    coef_dict = dict(zip(FEATURE_COLS, model.coef_[0].tolist()))

    metadata = {
        "feature_names": FEATURE_COLS,
        "coefficients":  coef_dict,
        "accuracy":      round(accuracy * 100, 2),
    }
    return model, scaler, metadata

MODEL, SCALER, METADATA = train_and_load()
FEATURE_NAMES = METADATA["feature_names"]
COEFFICIENTS  = METADATA["coefficients"]
FEATURE_LABELS = {
    "Study_Hours": "Study Hours", "Attendance_Rate": "Attendance Rate",
    "Previous_Grades": "Previous Grades", "Participation_Score": "Participation Score",
    "Parental_Involvement": "Parental Involvement",
}

# ★  Change header background colour here  ★
HEADER_BG = "#1a56db"

st.markdown(f"""<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {{
  --header-bg: {HEADER_BG};
  --ink: #0d1117; --ink-light: #374151; --ink-muted: #6b7280;
  --paper: #f8f7f4; --white: #fff;
  --accent: #1a56db; --accent-lt: #dbeafe;
  --pass: #059669; --pass-lt: #d1fae5;
  --fail: #dc2626; --fail-lt: #fee2e2;
  --border: #e5e7eb;
}}

html, body, [class*="css"] {{ font-family: 'DM Sans', sans-serif; color: var(--ink); }}
#MainMenu, footer, header {{ visibility: hidden; }}
.block-container {{ padding-top: 0 !important; max-width: 100% !important; }}

.app-header {{
  background: var(--header-bg); color: #fff;
  padding: 0 2rem; height: 64px;
  display: flex; align-items: center; justify-content: space-between;
  box-shadow: 0 2px 12px rgba(0,0,0,.3);
  margin: -1rem -1rem 1.5rem -1rem;
  position: sticky; top: 0; z-index: 999;
}}
.header-brand {{ display: flex; align-items: center; gap: .75rem; }}
.header-icon {{
  width:36px; height:36px; border-radius:8px;
  background: rgba(255,255,255,.2);
  display:flex; align-items:center; justify-content:center; font-size:1.1rem;
}}
.header-title {{ font-family:'Syne',sans-serif; font-size:1.05rem; font-weight:700; }}
.header-subtitle {{ font-size:.75rem; color:rgba(255,255,255,.65); }}
.badge-accuracy {{
  background:rgba(0,0,0,.25); border:1px solid rgba(255,255,255,.2);
  color:#e5e7eb; padding:.3rem .8rem; border-radius:999px; font-size:.78rem; font-weight:500;
}}

.info-bar {{ display:flex; flex-wrap:wrap; gap:.75rem; margin-bottom:1.5rem; }}
.info-pill {{
  display:flex; align-items:center; gap:.5rem;
  background:#fff; border:1px solid #e5e7eb; border-radius:999px;
  padding:.35rem 1rem .35rem .6rem; font-size:.78rem; color:#374151;
  box-shadow:0 1px 3px rgba(0,0,0,.08);
}}
.pill-dot {{ width:8px; height:8px; border-radius:50%; display:inline-block; }}

.section-eyebrow {{
  font-size:.72rem; font-weight:600; letter-spacing:.12em;
  text-transform:uppercase; color:#1a56db; margin-bottom:.35rem;
}}
.section-title {{
  font-family:'Syne',sans-serif; font-size:1.55rem; font-weight:800;
  line-height:1.2; margin-bottom:.75rem; color:#0d1117;
}}

.form-card {{
  background:#fff; border:1px solid #e5e7eb; border-radius:14px;
  padding:1.4rem 1.5rem .5rem; box-shadow:0 1px 3px rgba(0,0,0,.08);
  margin-bottom:.75rem;
}}

.slabel {{
  display:flex; justify-content:space-between; align-items:center;
  margin-bottom:2px; margin-top:10px;
}}
.slabel-name {{ font-size:.82rem; font-weight:600; color:#374151; }}
.slabel-val {{
  font-size:.76rem; font-weight:700; color:#1a56db;
  background:#dbeafe; padding:1px 9px; border-radius:999px;
}}

div[data-testid="stSlider"] label {{ display:none !important; }}
div[data-testid="stSlider"] {{ margin-bottom:6px !important; }}
div[data-testid="stSlider"] > div {{ padding-top:2px !important; padding-bottom:0 !important; }}

div[data-testid="stSelectbox"] label {{
  font-size:.82rem !important; font-weight:600 !important; color:#374151 !important; margin-top:10px;
}}

div.stButton > button {{
  width:100%; background:#0d1117 !important; color:#fff !important;
  border:none !important; font-family:'Syne',sans-serif !important;
  font-weight:700 !important; font-size:.95rem !important;
  letter-spacing:.03em !important; padding:.85rem 1.5rem !important;
  border-radius:8px !important; margin-top:.25rem;
}}
div.stButton > button:hover {{ background:#1f2937 !important; }}

.math-note {{
  padding:.85rem 1rem; background:#f0fdf4; border:1px solid #bbf7d0;
  border-radius:8px; font-size:.78rem; color:#166534; line-height:1.55; margin-top:.75rem;
}}

.card {{
  background:#fff; border:1px solid #e5e7eb; border-radius:14px;
  padding:1.75rem; box-shadow:0 1px 3px rgba(0,0,0,.08); margin-bottom:1rem;
}}

.verdict-banner {{
  border-radius:8px; padding:1.25rem 1.5rem; margin-bottom:1.5rem;
  display:flex; align-items:center; gap:1rem;
}}
.verdict-banner.pass {{ background:#d1fae5; border:1.5px solid #a7f3d0; }}
.verdict-banner.fail {{ background:#fee2e2; border:1.5px solid #fca5a5; }}
.verdict-icon {{ font-size:2.2rem; flex-shrink:0; }}
.verdict-label {{ font-family:'Syne',sans-serif; font-size:1.6rem; font-weight:800; }}
.verdict-label.pass {{ color:#059669; }} .verdict-label.fail {{ color:#dc2626; }}
.verdict-sub {{ font-size:.85rem; color:#6b7280; }}

.prob-section {{ margin-bottom:1.5rem; }}
.prob-header {{ display:flex; justify-content:space-between; align-items:baseline; margin-bottom:.5rem; }}
.prob-label {{ font-size:.82rem; font-weight:600; color:#374151; }}
.prob-value-pass {{ font-family:'Syne',sans-serif; font-size:1.4rem; font-weight:800; color:#059669; }}
.prob-value-fail {{ font-family:'Syne',sans-serif; font-size:1.4rem; font-weight:800; color:#dc2626; }}
.prob-track {{ height:12px; background:#e5e7eb; border-radius:999px; overflow:hidden; }}
.prob-fill-pass {{ height:100%; border-radius:999px; background:#059669; }}
.prob-fill-fail {{ height:100%; border-radius:999px; background:#dc2626; }}
.threshold-label {{ font-size:.68rem; color:#6b7280; text-align:center; margin-top:.3rem; }}

.contrib-section {{ margin-bottom:1.5rem; }}
.contrib-title {{ font-size:.82rem; font-weight:600; color:#374151; margin-bottom:.75rem; }}
.contrib-row {{ display:grid; grid-template-columns:130px 1fr 52px; align-items:center; gap:.5rem; margin-bottom:.45rem; font-size:.78rem; }}
.contrib-feat {{ color:#374151; font-weight:500; }}
.contrib-track {{ height:7px; background:#e5e7eb; border-radius:999px; overflow:hidden; }}
.contrib-bar-pos {{ height:100%; border-radius:999px; background:#059669; }}
.contrib-bar-neg {{ height:100%; border-radius:999px; background:#dc2626; }}
.contrib-num {{ text-align:right; color:#6b7280; font-size:.74rem; }}

.justification {{
  background:#f8f7f4; border:1px solid #e5e7eb;
  border-radius:8px; padding:1rem 1.15rem; margin-bottom:1rem;
}}
.just-header {{ font-size:.78rem; font-weight:700; color:#374151; text-transform:uppercase; letter-spacing:.08em; margin-bottom:.6rem; }}
.just-text {{ font-size:.85rem; line-height:1.6; color:#374151; }}
.just-text strong {{ color:#0d1117; }}
.just-reasons {{ margin-top:.6rem; padding-left:1rem; }}
.just-reasons li {{ font-size:.82rem; color:#6b7280; margin-bottom:.25rem; }}

.norm-note {{
  display:flex; gap:.6rem; background:#dbeafe; border:1px solid #bfdbfe;
  border-radius:8px; padding:.75rem 1rem; font-size:.78rem; color:#1e40af; line-height:1.55;
}}
.norm-icon {{ flex-shrink:0; font-size:1rem; margin-top:.05rem; }}

.empty-state {{
  display:flex; flex-direction:column; align-items:center;
  justify-content:center; padding:3rem 1.5rem; text-align:center; color:#6b7280; gap:.75rem;
}}
.empty-icon {{
  width:64px; height:64px; background:#f8f7f4; border-radius:50%;
  display:flex; align-items:center; justify-content:center;
  font-size:1.8rem; margin:0 auto;
}}
</style>""", unsafe_allow_html=True)


def build_justification(fv, prediction, probability):
    scaled  = SCALER.transform([fv])[0]
    contribs = {feat: COEFFICIENTS[feat] * scaled[i] for i, feat in enumerate(FEATURE_NAMES)}
    sorted_c = sorted(contribs.items(), key=lambda x: abs(x[1]), reverse=True)
    top_feat, top_val = sorted_c[0]
    top_lbl   = FEATURE_LABELS[top_feat]
    direction = "positive" if top_val >= 0 else "negative"
    if prediction == 1:
        main = (f"The model predicts <strong>Pass</strong> with {probability:.1f}% confidence. "
                f"The strongest driver is <strong>{top_lbl}</strong>, which contributed the "
                f"highest {direction} weight to the log-odds. "
                f"Since P(Pass) = {probability:.1f}% > 50%, the threshold rule classifies this as a Pass.")
    else:
        main = (f"The model predicts <strong>Fail</strong> with {100-probability:.1f}% confidence. "
                f"<strong>{top_lbl}</strong> had the largest {direction} impact on reducing "
                f"the pass probability below the 50% decision threshold.")
    reasons = [
        f"{FEATURE_LABELS[f]} (coef {COEFFICIENTS[f]:+.3f}) "
        f"{'increases' if COEFFICIENTS[f]>=0 else 'decreases'} the pass probability."
        for f, _ in sorted_c[:3]
    ]
    return {
        "outcome": "Pass" if prediction == 1 else "Fail",
        "probability": round(probability, 2),
        "main_text": main, "top_reasons": reasons,
        "normalisation_note": (
            "All input features were normalised using StandardScaler (μ=0, σ=1) before "
            "being passed to Logistic Regression, ensuring fair comparison across different "
            "numerical scales (e.g. Study Hours vs Attendance %)."),
        "contributions": {FEATURE_LABELS[k]: round(v, 4) for k, v in sorted_c},
        "model_accuracy": METADATA["accuracy"],
    }


# ── Header ───────────────────────────────────────────────
st.markdown(f"""
<div class="app-header">
  <div class="header-brand">
    <div class="header-icon">🎓</div>
    <div>
      <div class="header-title">Student Analytics Dashboard</div>
      <div class="header-subtitle">Logistic Regression · Final Year Project</div>
    </div>
  </div>
  <div class="badge-accuracy">Model Accuracy: {METADATA['accuracy']}%</div>
</div>""", unsafe_allow_html=True)

# ── Info pills ───────────────────────────────────────────
st.markdown("""
<div class="info-bar">
  <div class="info-pill"><span class="pill-dot" style="background:#1a56db"></span>Logistic Regression</div>
  <div class="info-pill"><span class="pill-dot" style="background:#059669"></span>StandardScaler Normalisation</div>
  <div class="info-pill"><span class="pill-dot" style="background:#d97706"></span>predict_proba Threshold &gt; 50%</div>
  <div class="info-pill"><span class="pill-dot" style="background:#7c3aed"></span>Coefficient-Based Justification</div>
</div>""", unsafe_allow_html=True)

# ── Two columns ──────────────────────────────────────────
left_col, right_col = st.columns(2, gap="large")

with left_col:
    st.markdown('<div class="section-eyebrow">Input Parameters</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Student Profile</div>', unsafe_allow_html=True)
    st.markdown('<div class="form-card">', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        study_hours = st.slider("sh", 1.0, 10.0, 5.0, 0.1, format="%.1f h", label_visibility="collapsed")
        st.markdown(f'<div class="slabel"><span class="slabel-name">Study Hours / Week</span><span class="slabel-val">{study_hours:.1f} h</span></div>', unsafe_allow_html=True)
    with c2:
        attendance = st.slider("att", 60, 100, 80, 1, format="%d%%", label_visibility="collapsed")
        st.markdown(f'<div class="slabel"><span class="slabel-name">Attendance Rate</span><span class="slabel-val">{attendance}%</span></div>', unsafe_allow_html=True)

    c3, c4 = st.columns(2)
    with c3:
        prev_grades = st.slider("pg", 0, 100, 65, 1, label_visibility="collapsed")
        st.markdown(f'<div class="slabel"><span class="slabel-name">Previous Grades</span><span class="slabel-val">{prev_grades}</span></div>', unsafe_allow_html=True)
    with c4:
        participation = st.slider("ps", 1, 10, 5, 1, label_visibility="collapsed")
        st.markdown(f'<div class="slabel"><span class="slabel-name">Participation Score</span><span class="slabel-val">{participation}</span></div>', unsafe_allow_html=True)

    parental = st.selectbox(
        "Parental Involvement", options=[0, 1, 2], index=1,
        format_func=lambda x: {
            0: "0 — Low (Minimal home academic support)",
            1: "1 — Medium (Occasional involvement)",
            2: "2 — High (Active, regular engagement)",
        }[x])

    predict_clicked = st.button("▶  Run Prediction", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="math-note">
      <strong>📐 Logistic Regression Decision Rule:</strong><br>
      P(Pass) = σ(β₀ + β₁·x₁ + … + β₅·x₅) where σ is the sigmoid function.
      If P(Pass) &gt; 0.50, the student is classified as <em>Pass</em>.
      All inputs are first normalised via <code>StandardScaler</code>.
    </div>""", unsafe_allow_html=True)

with right_col:
    st.markdown('<div class="section-eyebrow">Prediction Output</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Analysis Result</div>', unsafe_allow_html=True)

    if not predict_clicked:
        st.markdown("""
        <div class="card">
          <div class="empty-state">
            <div class="empty-icon">📊</div>
            <div style="font-weight:600;color:#0d1117">No prediction yet</div>
            <div style="font-size:.85rem">Fill in the student profile and click<br>
              <strong>Run Prediction</strong> to see the analysis.</div>
          </div>
        </div>""", unsafe_allow_html=True)
    else:
        fv = np.array([float(study_hours), float(attendance),
                       float(prev_grades), float(participation), float(parental)])
        scaled      = SCALER.transform([fv])
        prediction  = int(MODEL.predict(scaled)[0])
        probability = float(MODEL.predict_proba(scaled)[0][1]) * 100
        data        = build_justification(fv, prediction, probability)
        is_pass     = data["outcome"] == "Pass"
        css_cls     = "pass" if is_pass else "fail"
        icon        = "✅" if is_pass else "❌"
        conf        = data["probability"] if is_pass else 100 - data["probability"]
        pvc         = "prob-value-pass" if is_pass else "prob-value-fail"
        pfc         = "prob-fill-pass"  if is_pass else "prob-fill-fail"

        contribs  = data["contributions"]
        max_abs   = max(abs(v) for v in contribs.values()) or 1
        bars_html = '<div class="contrib-title">📊 Feature Contributions (coef × scaled value)</div>'
        for feat, val in contribs.items():
            pct = abs(val) / max_abs * 100
            bc  = "contrib-bar-pos" if val >= 0 else "contrib-bar-neg"
            sg  = "+" if val >= 0 else ""
            bars_html += (
                f'<div class="contrib-row">'
                f'<div class="contrib-feat">{feat}</div>'
                f'<div class="contrib-track"><div class="{bc}" style="width:{pct:.1f}%"></div></div>'
                f'<div class="contrib-num">{sg}{val:.3f}</div></div>')

        reasons_li = "".join(f"<li>{r}</li>" for r in data["top_reasons"])

        st.markdown(f"""
        <div class="card">
          <div class="verdict-banner {css_cls}">
            <div class="verdict-icon">{icon}</div>
            <div>
              <div class="verdict-label {css_cls}">{data['outcome']}</div>
              <div class="verdict-sub">Confidence: {conf:.1f}% · Model accuracy: {data['model_accuracy']}%</div>
            </div>
          </div>
          <div class="prob-section">
            <div class="prob-header">
              <span class="prob-label">Pass Probability  P(y=1)</span>
              <span class="{pvc}">{data['probability']:.1f}%</span>
            </div>
            <div class="prob-track"><div class="{pfc}" style="width:{data['probability']}%"></div></div>
            <div class="threshold-label">▲ 50% decision threshold</div>
          </div>
          <div class="contrib-section">{bars_html}</div>
          <div class="justification">
            <div class="just-header">🔍 Model Reasoning</div>
            <div class="just-text">{data['main_text']}</div>
            <ul class="just-reasons">{reasons_li}</ul>
          </div>
          <div class="norm-note">
            <div class="norm-icon">⚖️</div>
            <div>{data['normalisation_note']}</div>
          </div>
        </div>""", unsafe_allow_html=True)
