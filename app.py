import streamlit as st
import numpy as np
import pandas as pd
import pickle
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MediPredict · AI Health Intelligence",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Load models ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    with open("columns.pkl", "rb") as f:
        columns = pickle.load(f)
    with open("le.pkl", "rb") as f:
        le = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("reg.pkl", "rb") as f:
        reg = pickle.load(f)
    with open("clf.pkl", "rb") as f:
        clf = pickle.load(f)
    return columns, le, scaler, reg, clf

columns, le, scaler, reg, clf = load_models()

# ── Global CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Playfair+Display:ital,wght@0,600;0,700;1,600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding: 2rem 1rem 4rem 1rem;
    max-width: 820px;
    background: #eef1f7;
}

/* Page background */
.stApp { background: #eef1f7; }

/* ── Step tracker ───────────────────────────────────────────────────────── */
.tracker-wrap {
    display: flex;
    align-items: center;
    margin: 0 0 2rem 0;
    padding: 0 0.5rem;
}
.tracker-step {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.3rem;
    position: relative;
    z-index: 1;
}
.tracker-circle {
    width: 42px; height: 42px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-weight: 700; font-size: 1rem;
    flex-shrink: 0;
}
.tracker-circle.done   { background: #3d8c7c; color: #fff; }
.tracker-circle.active { background: #2563eb; color: #fff; }
.tracker-circle.future { background: #fff; color: #94a3b8; border: 2px solid #cbd5e1; }
.tracker-label {
    font-size: 0.7rem; font-weight: 600;
    letter-spacing: 0.06em; text-transform: uppercase;
    color: #64748b;
}
.tracker-line {
    flex: 1;
    height: 3px;
    border-radius: 999px;
    margin: 0 0.3rem;
    margin-bottom: 1.2rem;
}
.tracker-line.done   { background: #3d8c7c; }
.tracker-line.future { background: #cbd5e1; }

/* ── Step card ──────────────────────────────────────────────────────────── */
.step-card {
    background: #fff;
    border-radius: 20px;
    padding: 2rem 2.2rem 1.8rem;
    margin-bottom: 2rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    border-top: 4px solid transparent;
    border-image: linear-gradient(90deg, #2563eb, #3d8c7c) 1;
    border-top-left-radius: 20px;
    border-top-right-radius: 20px;
}
.step-pill {
    display: inline-block;
    background: #e8f0fe;
    color: #2563eb;
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    padding: 0.3rem 0.8rem;
    border-radius: 999px;
    margin-bottom: 0.9rem;
}
.step-title {
    font-family: 'Inter', sans-serif;
    font-size: 1.35rem;
    font-weight: 400;
    color: #1e293b;
    margin: 0 0 0.4rem;
}
.step-title em {
    font-family: 'Playfair Display', serif;
    font-style: italic;
    font-weight: 600;
    color: #2563eb;
}
.step-desc {
    font-size: 0.9rem;
    color: #64748b;
    line-height: 1.6;
}

/* ── Field label ────────────────────────────────────────────────────────── */
.field-lbl {
    font-size: 0.85rem;
    font-weight: 500;
    color: #334155;
    margin: 1.4rem 0 0.6rem;
}

/* ── Pill-style buttons ─────────────────────────────────────────────────── */
.stButton > button {
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    border-radius: 999px !important;
    padding: 0.6rem 1.4rem !important;
    font-size: 0.9rem !important;
    transition: all 0.2s ease !important;
}

/* Selected pill button style via data attribute */
[data-pill-selected="true"] > button {
    background-color: #2563eb !important;
    color: #fff !important;
    border: none !important;
}

/* ── Slider overrides ───────────────────────────────────────────────────── */
[data-testid="stSlider"] label {
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    color: #334155 !important;
    letter-spacing: 0 !important;
    text-transform: none !important;
}

/* ── Results page ───────────────────────────────────────────────────────── */
.result-eyebrow {
    font-size: 0.65rem;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: #2563eb;
    font-weight: 700;
    text-align: center;
    margin-bottom: 0.4rem;
}
.result-headline {
    font-family: 'Playfair Display', serif;
    font-size: 2.6rem;
    font-weight: 700;
    color: #1e293b;
    text-align: center;
    margin: 0 0 0.3rem;
}
.result-sub {
    font-size: 0.9rem;
    color: #64748b;
    text-align: center;
    margin-bottom: 2rem;
}
.stat-card {
    background: #fff;
    border-radius: 18px;
    padding: 1.6rem 1.8rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    height: 100%;
}
.stat-card.blue-accent  { border-top: 4px solid #2563eb; }
.stat-card.red-accent   { border-top: 4px solid #ef4444; }
.stat-card.green-accent { border-top: 4px solid #3d8c7c; }
.stat-eyebrow {
    font-size: 0.62rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #94a3b8;
    font-weight: 600;
    margin-bottom: 0.5rem;
}
.stat-value {
    font-family: 'Playfair Display', serif;
    font-size: 2.2rem;
    font-weight: 700;
    color: #1e293b;
    line-height: 1.1;
}
.stat-note {
    font-size: 0.8rem;
    color: #64748b;
    margin-top: 0.4rem;
}
.chart-card {
    background: #fff;
    border-radius: 18px;
    padding: 1.4rem 1.6rem 1rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    margin-bottom: 1.2rem;
}
.chart-title {
    font-size: 0.95rem;
    font-weight: 600;
    color: #1e293b;
    margin-bottom: 0.2rem;
    text-align: center;
}
.rec-section-title {
    font-size: 0.65rem;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: #2563eb;
    font-weight: 700;
    margin: 2rem 0 1rem;
}
.rec-card {
    background: #fff;
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.8rem;
    border-left: 4px solid #2563eb;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
.rec-card-title {
    font-weight: 600;
    font-size: 0.9rem;
    color: #1e293b;
    margin-bottom: 0.3rem;
}
.rec-card-body {
    font-size: 0.83rem;
    color: #64748b;
    line-height: 1.5;
}
</style>
""", unsafe_allow_html=True)

# ── Session state ────────────────────────────────────────────────────────────
defaults = {
    "step": 0,
    # Step 1 — Personal
    "gender": "Male",
    "insurance": "Basic",
    "city": "Urban",
    # Step 2 — Lifestyle
    "smoker": "No",
    "act_level": "Medium",
    # Step 3 — Conditions
    "diabetes": "No",
    "hypertension": "No",
    "heart_disease": "No",
    "asthma": "No",
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

def go(n):
    st.session_state.step = n
    st.rerun()


# ── Helper: pill radio ───────────────────────────────────────────────────────
def pill_radio(key, options, num_cols=None):
    """Render pill-style radio buttons using only Streamlit buttons (no duplicate HTML divs)."""
    current = st.session_state.get(key, options[0])
    if num_cols is None:
        num_cols = len(options)
    cols = st.columns(num_cols)
    for i, opt in enumerate(options):
        with cols[i]:
            is_selected = (current == opt)
            # Use Streamlit's type parameter to visually distinguish selected vs unselected
            btn_type = "primary" if is_selected else "secondary"
            if st.button(opt, key=f"btn_{key}_{opt}", use_container_width=True, type=btn_type):
                st.session_state[key] = opt
                st.rerun()
    return st.session_state.get(key, options[0])


# ── Helper: step tracker HTML ────────────────────────────────────────────────
def render_tracker(active_step, labels):
    """active_step is 1-indexed"""
    parts = []
    for i, lbl in enumerate(labels):
        snum = i + 1
        if snum < active_step:
            cclass = "done"
        elif snum == active_step:
            cclass = "active"
        else:
            cclass = "future"
        parts.append(f"""
        <div class="tracker-step">
            <div class="tracker-circle {cclass}">{snum}</div>
            <div class="tracker-label">{lbl}</div>
        </div>""")
        if i < len(labels) - 1:
            lclass = "done" if snum < active_step else "future"
            parts.append(f'<div class="tracker-line {lclass}"></div>')
    st.markdown(f'<div class="tracker-wrap">{"".join(parts)}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 0 — HERO
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.step == 0:
    st.markdown("""
    <div style="text-align:center; padding: 4rem 1rem 2rem;">
        <div style="font-size:0.65rem; letter-spacing:0.22em; text-transform:uppercase;
                    color:#2563eb; font-weight:700; margin-bottom:0.6rem;">
            AI · HEALTH INTELLIGENCE
        </div>
        <div style="font-family:'Playfair Display',serif; font-size:3.4rem; font-weight:700;
                    color:#1e293b; line-height:1.1; margin-bottom:0.8rem;">
            Medi<em style="color:#2563eb; font-style:italic;">Predict</em>
        </div>
        <div style="font-size:1rem; color:#64748b; max-width:480px; margin:0 auto 2.5rem;
                    line-height:1.7;">
            Enter your health profile and receive an instant estimate of your
            annual medical cost alongside a personalised disease risk score.
        </div>
        <div style="display:flex; justify-content:center; gap:2.5rem; margin-bottom:3rem;">
            <div style="text-align:center;">
                <div style="font-family:'Playfair Display',serif; font-size:2rem;
                            font-weight:700; color:#2563eb;">3</div>
                <div style="font-size:0.7rem; text-transform:uppercase; letter-spacing:0.1em;
                            color:#94a3b8; font-weight:600;">Steps</div>
            </div>
            <div style="text-align:center;">
                <div style="font-family:'Playfair Display',serif; font-size:2rem;
                            font-weight:700; color:#2563eb;">ML</div>
                <div style="font-size:0.7rem; text-transform:uppercase; letter-spacing:0.1em;
                            color:#94a3b8; font-weight:600;">Powered</div>
            </div>
            <div style="text-align:center;">
                <div style="font-family:'Playfair Display',serif; font-size:2rem;
                            font-weight:700; color:#2563eb;">2</div>
                <div style="font-size:0.7rem; text-transform:uppercase; letter-spacing:0.1em;
                            color:#94a3b8; font-weight:600;">Outputs</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Begin Health Assessment →", type="primary", use_container_width=True):
            go(1)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — PERSONAL PROFILE
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.step == 1:
    render_tracker(1, ["Profile", "Lifestyle", "History"])

    st.markdown("""
    <div class="step-card">
        <div class="step-pill">STEP 1 OF 3</div>
        <div class="step-title">Personal <em>Profile</em></div>
        <div class="step-desc">Tell us about yourself — age, body metrics, and coverage details.</div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        age = st.slider("Age", 18, 85, 30)
    with c2:
        bmi = st.slider("BMI", 10.0, 50.0, 24.0, step=0.5)

    c3, c4 = st.columns(2)
    with c3:
        children = st.slider("Dependents", 0, 10, 0)
    with c4:
        st.markdown('<div class="field-lbl">Gender</div>', unsafe_allow_html=True)
        pill_radio("gender", ["Male", "Female"], num_cols=2)

    st.markdown('<div class="field-lbl">Insurance</div>', unsafe_allow_html=True)
    pill_radio("insurance", ["Basic", "Premium"], num_cols=2)

    st.markdown('<div class="field-lbl">City</div>', unsafe_allow_html=True)
    pill_radio("city", ["Urban", "Semi-Urban", "Rural"], num_cols=3)

    st.markdown("<br>", unsafe_allow_html=True)
    cb, cn = st.columns([1, 1])
    with cb:
        if st.button("← Back", use_container_width=True):
            go(0)
    with cn:
        if st.button("Next →", type="primary", use_container_width=True):
            st.session_state.s1 = dict(
                age=age, bmi=bmi, children=children,
                gender=st.session_state.gender,
                insurance=st.session_state.insurance,
                city=st.session_state.city,
            )
            go(2)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — LIFESTYLE HABITS
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.step == 2:
    render_tracker(2, ["Profile", "Lifestyle", "History"])

    st.markdown("""
    <div class="step-card">
        <div class="step-pill">STEP 2 OF 3</div>
        <div class="step-title">Lifestyle <em>Habits</em></div>
        <div class="step-desc">Smoking and activity level are among the strongest cost predictors.</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="field-lbl">Do you currently smoke?</div>', unsafe_allow_html=True)
    pill_radio("smoker", ["No", "Yes"], num_cols=2)

    st.markdown('<div class="field-lbl">Physical Activity Level</div>', unsafe_allow_html=True)
    pill_radio("act_level", ["Low", "Medium", "High"], num_cols=3)

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        daily_steps = st.slider("Daily Steps", 0, 20000, 7500, step=500)
    with c2:
        sleep_hours = st.slider("Sleep Hours / Night", 3.0, 12.0, 7.0, step=0.5)

    c3, c4 = st.columns(2)
    with c3:
        stress_level = st.slider("Stress Level (1=Low, 10=High)", 1, 10, 4)
    with c4:
        doc_visits = st.slider("Doctor Visits / Year", 0, 20, 2)

    c5, c6 = st.columns(2)
    with c5:
        hosp_admissions = st.slider("Hospital Admissions / Year", 0, 10, 0)
    with c6:
        medication_count = st.slider("No. of Medications", 0, 15, 1)

    prev_cost = st.number_input("Previous Year Medical Cost (₹)", 0, 500000, 2000, step=500)

    st.markdown("<br>", unsafe_allow_html=True)
    cb, cn = st.columns([1, 1])
    with cb:
        if st.button("← Back", use_container_width=True):
            go(1)
    with cn:
        if st.button("Next →", type="primary", use_container_width=True):
            st.session_state.s2 = dict(
                smoker=1 if st.session_state.smoker == "Yes" else 0,
                act_level=st.session_state.act_level,
                daily_steps=daily_steps,
                sleep_hours=sleep_hours,
                stress_level=stress_level,
                doc_visits=doc_visits,
                hosp_admissions=hosp_admissions,
                medication_count=medication_count,
                prev_cost=prev_cost,
            )
            go(3)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — MEDICAL HISTORY
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.step == 3:
    render_tracker(3, ["Profile", "Lifestyle", "History"])

    st.markdown("""
    <div class="step-card">
        <div class="step-pill">STEP 3 OF 3</div>
        <div class="step-title">Medical <em>History</em></div>
        <div class="step-desc">Existing conditions help calibrate the risk model. Select all that apply.</div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="field-lbl">Diabetes</div>', unsafe_allow_html=True)
        pill_radio("diabetes", ["No", "Yes"], num_cols=2)
    with c2:
        st.markdown('<div class="field-lbl">Hypertension</div>', unsafe_allow_html=True)
        pill_radio("hypertension", ["No", "Yes"], num_cols=2)

    c3, c4 = st.columns(2)
    with c3:
        st.markdown('<div class="field-lbl">Heart Disease</div>', unsafe_allow_html=True)
        pill_radio("heart_disease", ["No", "Yes"], num_cols=2)
    with c4:
        st.markdown('<div class="field-lbl">Asthma</div>', unsafe_allow_html=True)
        pill_radio("asthma", ["No", "Yes"], num_cols=2)

    st.markdown("<br>", unsafe_allow_html=True)
    cb, cn, cr = st.columns([1, 1, 1])
    with cb:
        if st.button("← Back", use_container_width=True):
            go(2)
    with cn:
        if st.button("Predict Now →", type="primary", use_container_width=True):
            st.session_state.s3 = dict(
                diabetes=1 if st.session_state.diabetes == "Yes" else 0,
                hypertension=1 if st.session_state.hypertension == "Yes" else 0,
                heart_disease=1 if st.session_state.heart_disease == "Yes" else 0,
                asthma=1 if st.session_state.asthma == "Yes" else 0,
            )
            go(4)
    with cr:
        if st.button("Restart", use_container_width=True):
            for key in ["s1", "s2", "s3", "gender", "insurance", "city",
                        "smoker", "act_level", "diabetes", "hypertension",
                        "heart_disease", "asthma"]:
                st.session_state.pop(key, None)
            go(0)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — RESULTS
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.step == 4:
    s1 = st.session_state.get("s1", {})
    s2 = st.session_state.get("s2", {})
    s3 = st.session_state.get("s3", {})

    # ── Encode ────────────────────────────────────────────────────────────
    gender_enc    = 1 if s1.get("gender", "Male") == "Male" else 0
    insurance_enc = 1 if s1.get("insurance", "Basic") == "Premium" else 0
    city_enc      = le.transform([s1.get("city", "Urban")])[0]
    act_map       = {"Low": 0, "Medium": 1, "High": 2}
    act_enc       = act_map.get(s2.get("act_level", "Medium"), 1)

    input_data = pd.DataFrame([[
        s1.get("age", 30), gender_enc, s1.get("bmi", 24.0),
        s2.get("smoker", 0), s3.get("diabetes", 0), s3.get("hypertension", 0),
        s3.get("heart_disease", 0), s3.get("asthma", 0),
        act_enc, s2.get("daily_steps", 7500), s2.get("sleep_hours", 7.0),
        s2.get("stress_level", 4),
        s2.get("doc_visits", 2), s2.get("hosp_admissions", 0),
        s2.get("medication_count", 1),
        insurance_enc, 60.0,
        city_enc, s2.get("prev_cost", 2000)
    ]], columns=columns)

    scaled_input = scaler.transform(input_data)

    clf_pred   = clf.predict(scaled_input)[0]
    clf_prob   = clf.predict_proba(scaled_input)[0]
    risk_score = round(float(clf_prob[1]) * 100, 1)

    if clf_pred == 0:
        medical_cost = 1000
    else:
        medical_cost = max(1000, int(reg.predict(scaled_input)[0]))

    # ── Header ────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="result-eyebrow">ASSESSMENT COMPLETE</div>
    <div class="result-headline">Your Health Report</div>
    <div class="result-sub">Based on the profile you provided</div>
    """, unsafe_allow_html=True)

    # ── Top stat cards ────────────────────────────────────────────────────
    risk_label = "High Risk" if clf_pred == 1 else "Low Risk"
    risk_note  = "Specialist consultation advised" if clf_pred == 1 else "Keep up your healthy habits"
    risk_accent = "red-accent" if clf_pred == 1 else "green-accent"

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
        <div class="stat-card blue-accent">
            <div class="stat-eyebrow">ANNUAL MEDICAL COST</div>
            <div class="stat-value">₹{medical_cost:,}</div>
            <div class="stat-note">Estimated for your profile</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="stat-card {risk_accent}">
            <div class="stat-eyebrow">DISEASE RISK</div>
            <div class="stat-value">{risk_label}</div>
            <div class="stat-note">{risk_note}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Charts row 1: Profile Overview + Condition Flags ─────────────────
    BG = "#ffffff"
    TEXT = "#1e293b"
    MUTED = "#94a3b8"

    c3, c4 = st.columns(2)

    with c3:
        st.markdown('<div class="chart-card"><div class="chart-title">Profile Overview</div>', unsafe_allow_html=True)
        act_val = act_map.get(s2.get("act_level", "Medium"), 1)
        fig1, ax1 = plt.subplots(figsize=(4, 2.8))
        fig1.patch.set_facecolor(BG)
        ax1.set_facecolor(BG)
        labels_bar = ["Age", "BMI", "Children", "Activity"]
        values_bar = [s1.get("age", 30), s1.get("bmi", 24.0), s1.get("children", 0), act_val]
        colors_bar = ["#3b82f6", "#3d8c7c", "#94a3b8", "#a78bfa"]
        bars = ax1.bar(labels_bar, values_bar, color=colors_bar, width=0.55, zorder=3)
        for bar, val in zip(bars, values_bar):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     str(round(val, 1)), ha="center", va="bottom", fontsize=8,
                     color=TEXT, fontweight="500")
        ax1.set_axisbelow(True)
        ax1.yaxis.set_visible(False)
        ax1.spines[["top", "right", "left"]].set_visible(False)
        ax1.spines["bottom"].set_color("#e2e8f0")
        ax1.tick_params(axis="x", labelsize=8, colors=MUTED)
        plt.tight_layout(pad=0.4)
        st.pyplot(fig1, use_container_width=True)
        plt.close(fig1)
        st.markdown('</div>', unsafe_allow_html=True)

    with c4:
        st.markdown('<div class="chart-card"><div class="chart-title">Condition Flags</div>', unsafe_allow_html=True)
        conditions = {
            "Smoker": s2.get("smoker", 0),
            "Asthma": s3.get("asthma", 0),
            "Heart Dis.": s3.get("heart_disease", 0),
            "Hypertension": s3.get("hypertension", 0),
            "Diabetes": s3.get("diabetes", 0),
        }
        fig2, ax2 = plt.subplots(figsize=(4, 2.8))
        fig2.patch.set_facecolor(BG)
        ax2.set_facecolor(BG)
        y_pos = range(len(conditions))
        cond_names = list(conditions.keys())
        cond_vals  = list(conditions.values())
        for i, (name, val) in enumerate(zip(cond_names, cond_vals)):
            color = "#ef4444" if val else "#3d8c7c"
            label_txt = "Yes" if val else "No"
            ax2.barh(i, 1, color=color, height=0.45, zorder=3)
            ax2.text(1.08, i, label_txt, va="center", fontsize=8.5,
                     color=TEXT, fontweight="500")
        ax2.set_yticks(list(y_pos))
        ax2.set_yticklabels([f"{n} –" for n in cond_names], fontsize=8, color=MUTED)
        ax2.set_xlim(0, 1.6)
        ax2.xaxis.set_visible(False)
        ax2.spines[["top", "right", "bottom"]].set_visible(False)
        ax2.spines["left"].set_color("#e2e8f0")
        present_patch = mpatches.Patch(color="#ef4444", label="Present")
        absent_patch  = mpatches.Patch(color="#3d8c7c", label="Absent")
        ax2.legend(handles=[present_patch, absent_patch], loc="upper right",
                   fontsize=7, frameon=False)
        plt.tight_layout(pad=0.4)
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Charts row 2: Cost Factors donut + Coverage bar ───────────────────
    age_val = s1.get("age", 30)
    bmi_val = s1.get("bmi", 24.0)
    smoker_val = s2.get("smoker", 0)
    cond_count = sum([s3.get("diabetes", 0), s3.get("hypertension", 0),
                      s3.get("heart_disease", 0), s3.get("asthma", 0)])
    base_share = 5000
    age_share  = max(0, (age_val - 18) * 50)
    bmi_share  = max(0, (bmi_val - 18.5) * 80) if bmi_val > 18.5 else 0
    smoke_share = 3000 if smoker_val else 0
    cond_share  = cond_count * 500
    total_fake  = base_share + age_share + bmi_share + smoke_share + cond_share
    if total_fake == 0:
        total_fake = base_share

    pct_age   = age_share / total_fake * 100
    pct_bmi   = bmi_share / total_fake * 100
    pct_smoke = smoke_share / total_fake * 100
    pct_cond  = cond_share / total_fake * 100
    pct_base  = base_share / total_fake * 100

    coverage_pct  = 0.60
    covered_cost  = int(medical_cost * coverage_pct)
    oop_cost      = medical_cost - covered_cost

    c5, c6 = st.columns(2)
    with c5:
        st.markdown('<div class="chart-card"><div class="chart-title">Cost Factors</div>', unsafe_allow_html=True)
        slices  = [pct_age, pct_bmi, pct_smoke, pct_cond, pct_base]
        clabels = [f"{v:.0f}%" if v >= 5 else "" for v in slices]
        ccolors = ["#3b82f6", "#3d8c7c", "#ef4444", "#f59e0b", "#a78bfa"]
        fig3, ax3 = plt.subplots(figsize=(4, 3.2))
        fig3.patch.set_facecolor(BG)
        ax3.set_facecolor(BG)
        wedges, texts = ax3.pie(
            slices, labels=clabels, colors=ccolors,
            startangle=90, wedgeprops=dict(width=0.52),
            textprops=dict(fontsize=8.5, color=TEXT, fontweight="600")
        )
        legend_labels = ["Age", "BMI", "Smoking", "Conditions", "Base"]
        legend_patches = [mpatches.Patch(color=c, label=l) for c, l in zip(ccolors, legend_labels)]
        ax3.legend(handles=legend_patches, loc="lower center",
                   bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=7, frameon=False)
        plt.tight_layout(pad=0.4)
        st.pyplot(fig3, use_container_width=True)
        plt.close(fig3)
        st.markdown('</div>', unsafe_allow_html=True)

    with c6:
        st.markdown('<div class="chart-card"><div class="chart-title">Coverage (60%)</div>', unsafe_allow_html=True)
        fig4, ax4 = plt.subplots(figsize=(4, 3.2))
        fig4.patch.set_facecolor(BG)
        ax4.set_facecolor(BG)
        bar_labels  = ["Total", "Covered", "Out of Pocket"]
        bar_values  = [medical_cost, covered_cost, oop_cost]
        bar_colors  = ["#3b82f6", "#3d8c7c", "#ef8c8c"]
        bars4 = ax4.bar(bar_labels, bar_values, color=bar_colors, width=0.45, zorder=3)
        for bar, val in zip(bars4, bar_values):
            ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(bar_values)*0.02,
                     f"₹{val:,}", ha="center", va="bottom", fontsize=7.5,
                     color=TEXT, fontweight="500")
        ax4.set_axisbelow(True)
        ax4.yaxis.set_visible(False)
        ax4.spines[["top", "right", "left"]].set_visible(False)
        ax4.spines["bottom"].set_color("#e2e8f0")
        ax4.tick_params(axis="x", labelsize=8, colors=MUTED)
        plt.tight_layout(pad=0.4)
        st.pyplot(fig4, use_container_width=True)
        plt.close(fig4)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Recommendations ───────────────────────────────────────────────────
    st.markdown('<div class="rec-section-title">RECOMMENDATIONS</div>', unsafe_allow_html=True)

    recs = []
    if s2.get("smoker", 0):
        recs.append(("Quit Smoking",
                     "Quitting smoking can reduce your disease risk by up to 50% within 5 years and significantly lower your annual medical costs."))
    if s1.get("bmi", 24) > 27.5:
        recs.append(("Manage Your Weight",
                     f"Your BMI of {s1['bmi']} is above the ideal range. Aim for 18.5–24.9 through balanced diet and regular activity."))
    if s2.get("stress_level", 4) > 6:
        recs.append(("Reduce Stress",
                     "High stress is a silent driver of cardiovascular risk. Try mindfulness, structured breaks, or speaking to a counsellor."))
    if s2.get("sleep_hours", 7) < 6:
        recs.append(("Improve Sleep",
                     "Less than 6 hours of sleep raises cortisol and inflammation markers significantly. Aim for 7–9 hours nightly."))
    if s2.get("daily_steps", 7500) < 5000:
        recs.append(("Increase Activity",
                     "Aim for 8,000–10,000 steps a day to meaningfully lower chronic disease risk and improve overall wellbeing."))
    if not recs:
        recs.append(("General Wellness",
                     "Even without specific conditions flagged, maintaining a healthy lifestyle helps reduce overall risk. Keep up your good habits and attend annual checkups."))

    for title, body in recs:
        st.markdown(f"""
        <div class="rec-card">
            <div class="rec-card-title">{title}</div>
            <div class="rec-card-body">{body}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("← Start Over", use_container_width=True):
            for key in ["s1", "s2", "s3", "gender", "insurance", "city",
                        "smoker", "act_level", "diabetes", "hypertension",
                        "heart_disease", "asthma"]:
                st.session_state.pop(key, None)
            go(0)
