import streamlit as st
import numpy as np
import pandas as pd
import pickle
import warnings
warnings.filterwarnings("ignore")

# ── Page config ─────────────────────────────────────────────────────────────
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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Playfair+Display:ital,wght@0,700;1,600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 1rem 3rem 1rem; max-width: 760px; }

/* ── Base palette ──────────────────────────────────────────────────────── */
:root {
    --navy:    #1a2744;
    --blue:    #2563eb;
    --sky:     #60a5fa;
    --teal:    #0d9488;
    --mint:    #ccfbf1;
    --cream:   #f8fafc;
    --slate:   #64748b;
    --border:  #e2e8f0;
    --red:     #ef4444;
    --amber:   #f59e0b;
    --green:   #22c55e;
}

/* ── Hero ──────────────────────────────────────────────────────────────── */
.hero {
    background: linear-gradient(135deg, #1a2744 0%, #1e3a5f 50%, #0f4c75 100%);
    border-radius: 20px;
    padding: 3rem 2.5rem 2.5rem;
    margin: 1.5rem 0 2rem;
    position: relative;
    overflow: hidden;
    text-align: center;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 220px; height: 220px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(96,165,250,0.15) 0%, transparent 70%);
}
.hero::after {
    content: '';
    position: absolute;
    bottom: -40px; left: -40px;
    width: 160px; height: 160px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(13,148,136,0.12) 0%, transparent 70%);
}
.hero-eyebrow {
    font-size: 0.68rem;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: var(--sky);
    font-weight: 500;
    margin-bottom: 0.8rem;
}
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 3.2rem;
    font-weight: 700;
    color: #ffffff;
    line-height: 1.1;
    margin: 0 0 0.3rem;
}
.hero-title em {
    font-style: italic;
    color: var(--sky);
}
.hero-sub {
    color: rgba(255,255,255,0.72);
    font-size: 0.95rem;
    max-width: 480px;
    margin: 0.6rem auto 1.8rem;
    line-height: 1.6;
}
.hero-stats {
    display: flex;
    justify-content: center;
    gap: 1px;
    background: rgba(255,255,255,0.08);
    border-radius: 12px;
    overflow: hidden;
    max-width: 380px;
    margin: 0 auto;
}
.hero-stat {
    flex: 1;
    padding: 1rem 0.5rem;
    text-align: center;
    background: rgba(255,255,255,0.04);
}
.hero-stat:not(:last-child) { border-right: 1px solid rgba(255,255,255,0.08); }
.stat-num {
    font-family: 'Playfair Display', serif;
    font-size: 1.9rem;
    color: var(--sky);
    font-weight: 700;
    line-height: 1;
}
.stat-label {
    font-size: 0.6rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: rgba(255,255,255,0.45);
    margin-top: 0.3rem;
}

/* ── Step card ─────────────────────────────────────────────────────────── */
.step-card {
    background: #ffffff;
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.6rem 1.8rem 1rem;
    margin-bottom: 1.6rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04), 0 4px 16px rgba(0,0,0,0.04);
}
.step-pill {
    display: inline-block;
    background: var(--mint);
    color: var(--teal);
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    padding: 0.25rem 0.7rem;
    border-radius: 999px;
    margin-bottom: 0.6rem;
}
.step-title {
    font-size: 1.1rem;
    font-weight: 600;
    color: var(--navy);
    margin: 0 0 0.25rem;
}
.step-title em { font-style: italic; color: var(--blue); }
.step-desc {
    font-size: 0.83rem;
    color: var(--slate);
    margin-bottom: 0.2rem;
    line-height: 1.5;
}

/* ── Progress bar ──────────────────────────────────────────────────────── */
.progress-wrap {
    background: var(--border);
    border-radius: 999px;
    height: 4px;
    margin-bottom: 2rem;
    overflow: hidden;
}
.progress-fill {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, var(--teal), var(--blue));
    transition: width 0.4s ease;
}

/* ── Section label ─────────────────────────────────────────────────────── */
.field-label {
    font-size: 0.78rem;
    font-weight: 600;
    color: var(--navy);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin: 0.8rem 0 0.1rem;
}

/* ── Nav buttons ───────────────────────────────────────────────────────── */
.stButton > button {
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    border-radius: 12px !important;
    transition: all 0.2s ease !important;
}
div[data-testid="stHorizontalBlock"] .stButton > button {
    width: 100%;
}

/* ── Result card ───────────────────────────────────────────────────────── */
.result-hero {
    background: linear-gradient(135deg, #f0fdf4 0%, #ecfdf5 100%);
    border: 1px solid #bbf7d0;
    border-radius: 18px;
    padding: 2rem 2rem 1.6rem;
    margin: 1rem 0;
    text-align: center;
}
.result-hero.risk {
    background: linear-gradient(135deg, #fff7ed 0%, #fef3c7 100%);
    border-color: #fed7aa;
}
.result-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: #dcfce7;
    color: #16a34a;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 0.3rem 0.8rem;
    border-radius: 999px;
    margin-bottom: 0.8rem;
}
.result-badge.risk { background: #fed7aa; color: #c2410c; }
.result-main {
    font-family: 'Playfair Display', serif;
    font-size: 2.4rem;
    font-weight: 700;
    color: var(--navy);
    line-height: 1.1;
    margin: 0.2rem 0;
}
.result-sub {
    font-size: 0.85rem;
    color: var(--slate);
    margin-top: 0.3rem;
}
.cost-card {
    background: #fff;
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    text-align: center;
    margin-top: 1rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}
.cost-label {
    font-size: 0.7rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--slate);
    font-weight: 600;
    margin-bottom: 0.4rem;
}
.cost-amount {
    font-family: 'Playfair Display', serif;
    font-size: 2.6rem;
    font-weight: 700;
    color: var(--blue);
}
.cost-note {
    font-size: 0.75rem;
    color: var(--slate);
    margin-top: 0.3rem;
}
.rec-section {
    margin-top: 1.5rem;
}
.rec-eyebrow {
    font-size: 0.65rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--teal);
    font-weight: 700;
    margin-bottom: 0.8rem;
}
.rec-item {
    display: flex;
    align-items: flex-start;
    gap: 0.7rem;
    padding: 0.7rem 0;
    border-bottom: 1px solid var(--border);
    font-size: 0.85rem;
    color: #374151;
    line-height: 1.45;
}
.rec-item:last-child { border-bottom: none; }
.rec-icon {
    width: 26px; height: 26px;
    border-radius: 8px;
    background: var(--mint);
    display: flex; align-items: center; justify-content: center;
    font-size: 0.85rem;
    flex-shrink: 0;
    margin-top: 1px;
}

/* ── Disclaimer ────────────────────────────────────────────────────────── */
.disclaimer {
    background: #fffbeb;
    border: 1px solid #fde68a;
    border-radius: 10px;
    padding: 0.75rem 1rem;
    font-size: 0.75rem;
    color: #92400e;
    line-height: 1.5;
    margin-top: 1.5rem;
    text-align: center;
}

/* ── Divider ───────────────────────────────────────────────────────────── */
.divider { height: 1px; background: var(--border); margin: 1.5rem 0; }

/* ── Slider label override ─────────────────────────────────────────────── */
[data-testid="stSlider"] label {
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    color: var(--navy) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
}

/* ── Selectbox override ────────────────────────────────────────────────── */
[data-testid="stSelectbox"] label {
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    color: var(--navy) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
}

/* ── Checkbox override ─────────────────────────────────────────────────── */
[data-testid="stCheckbox"] label {
    font-size: 0.9rem !important;
    font-weight: 500 !important;
    color: #374151 !important;
}

/* ── Number input override ─────────────────────────────────────────────── */
[data-testid="stNumberInput"] label {
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    color: var(--navy) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
}
</style>
""", unsafe_allow_html=True)

# ── Session state init ───────────────────────────────────────────────────────
if "step" not in st.session_state:
    st.session_state.step = 0  # 0=hero, 1,2,3=form steps, 4=result

def go(n): st.session_state.step = n

# ── HERO PAGE ────────────────────────────────────────────────────────────────
if st.session_state.step == 0:
    st.markdown("""
    <div class="hero">
        <div class="hero-eyebrow">AI · Health Intelligence</div>
        <div class="hero-title">Medi<em>Predict</em></div>
        <div class="hero-sub">
            Enter your health profile and receive an instant estimate of your
            annual medical cost alongside a personalised disease risk score.
        </div>
        <div class="hero-stats">
            <div class="hero-stat">
                <div class="stat-num">3</div>
                <div class="stat-label">Steps</div>
            </div>
            <div class="hero-stat">
                <div class="stat-num">ML</div>
                <div class="stat-label">Powered</div>
            </div>
            <div class="hero-stat">
                <div class="stat-num">2</div>
                <div class="stat-label">Outputs</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Begin Health Assessment →", type="primary", use_container_width=True):
            go(1)

# ── STEP 1: Personal Profile ─────────────────────────────────────────────────
elif st.session_state.step == 1:
    progress = 33
    st.markdown(f"""
    <div class="progress-wrap">
        <div class="progress-fill" style="width:{progress}%"></div>
    </div>
    <div class="step-card">
        <div class="step-pill">Step 1 of 3</div>
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
        children = st.slider("No. of Children", 0, 10, 0)
    with c4:
        gender_sel = st.radio("Gender", ["Male", "Female"], horizontal=True)

    c5, c6 = st.columns(2)
    with c5:
        insurance_sel = st.radio("Insurance Plan", ["Basic", "Premium"], horizontal=True)
    with c6:
        city_sel = st.selectbox("City Type", ["Urban", "Semi-Urban", "Rural"])

    st.markdown("<br>", unsafe_allow_html=True)
    col_back, col_next = st.columns([1, 1])
    with col_back:
        if st.button("← Back", use_container_width=True):
            go(0)
    with col_next:
        if st.button("Next →", type="primary", use_container_width=True):
            st.session_state.s1 = dict(
                age=age, bmi=bmi, children=children,
                gender=gender_sel, insurance=insurance_sel, city=city_sel
            )
            go(2)

# ── STEP 2: Health Conditions ────────────────────────────────────────────────
elif st.session_state.step == 2:
    progress = 66
    st.markdown(f"""
    <div class="progress-wrap">
        <div class="progress-fill" style="width:{progress}%"></div>
    </div>
    <div class="step-card">
        <div class="step-pill">Step 2 of 3</div>
        <div class="step-title">Health <em>Conditions</em></div>
        <div class="step-desc">Select any diagnosed conditions and your smoking status.</div>
    </div>
    """, unsafe_allow_html=True)

    smoker = st.checkbox("🚬  Smoker")
    c1, c2 = st.columns(2)
    with c1:
        diabetes      = st.checkbox("🩸  Diabetes")
        hypertension  = st.checkbox("💊  Hypertension")
    with c2:
        heart_disease = st.checkbox("❤️  Heart Disease")
        asthma        = st.checkbox("🫁  Asthma")

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    c3, c4 = st.columns(2)
    with c3:
        doc_visits = st.slider("Doctor Visits / Year", 0, 20, 2)
    with c4:
        hosp_admissions = st.slider("Hospital Admissions / Year", 0, 10, 0)

    c5, c6 = st.columns(2)
    with c5:
        medication_count = st.slider("No. of Medications", 0, 15, 1)
    with c6:
        prev_cost = st.number_input("Previous Year Medical Cost (₹)", 0, 500000, 2000, step=500)

    st.markdown("<br>", unsafe_allow_html=True)
    col_back, col_next = st.columns([1, 1])
    with col_back:
        if st.button("← Back", use_container_width=True):
            go(1)
    with col_next:
        if st.button("Next →", type="primary", use_container_width=True):
            st.session_state.s2 = dict(
                smoker=int(smoker), diabetes=int(diabetes),
                hypertension=int(hypertension), heart_disease=int(heart_disease),
                asthma=int(asthma), doc_visits=doc_visits,
                hosp_admissions=hosp_admissions, medication_count=medication_count,
                prev_cost=prev_cost
            )
            go(3)

# ── STEP 3: Lifestyle ────────────────────────────────────────────────────────
elif st.session_state.step == 3:
    progress = 90
    st.markdown(f"""
    <div class="progress-wrap">
        <div class="progress-fill" style="width:{progress}%"></div>
    </div>
    <div class="step-card">
        <div class="step-pill">Step 3 of 3</div>
        <div class="step-title">Daily <em>Lifestyle</em></div>
        <div class="step-desc">Your habits shape long-term health costs more than any single factor.</div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        act_level_sel = st.selectbox("Physical Activity Level", ["Low", "Medium", "High"])
        daily_steps   = st.slider("Daily Steps", 0, 20000, 7500, step=500)
    with c2:
        sleep_hours  = st.slider("Sleep Hours / Night", 3.0, 12.0, 7.0, step=0.5)
        stress_level = st.slider("Stress Level (1 = Low, 10 = High)", 1, 10, 4)

    st.markdown("<br>", unsafe_allow_html=True)
    col_back, col_predict = st.columns([1, 1])
    with col_back:
        if st.button("← Back", use_container_width=True):
            go(2)
    with col_predict:
        if st.button("Generate Prediction ✦", type="primary", use_container_width=True):
            st.session_state.s3 = dict(
                act_level=act_level_sel, daily_steps=daily_steps,
                sleep_hours=sleep_hours, stress_level=stress_level
            )
            go(4)

# ── STEP 4: Results ──────────────────────────────────────────────────────────
elif st.session_state.step == 4:
    s1 = st.session_state.get("s1", {})
    s2 = st.session_state.get("s2", {})
    s3 = st.session_state.get("s3", {})

    # ── Encode inputs ──────────────────────────────────────────────────────
    gender_enc    = 1 if s1["gender"] == "Male" else 0
    insurance_enc = 1 if s1["insurance"] == "Premium" else 0
    city_enc      = le.transform([s1["city"]])[0]   # Rural=0, Semi-Urban=1, Urban=2
    act_map       = {"Low": 0, "Medium": 1, "High": 2}
    act_enc       = act_map[s3["act_level"]]

    input_data = pd.DataFrame([[
        s1["age"], gender_enc, s1["bmi"],
        s2["smoker"], s2["diabetes"], s2["hypertension"],
        s2["heart_disease"], s2["asthma"],
        act_enc, s3["daily_steps"], s3["sleep_hours"], s3["stress_level"],
        s2["doc_visits"], s2["hosp_admissions"], s2["medication_count"],
        insurance_enc, 60.0,   # insurance_coverage_pct — sensible default
        city_enc, s2["prev_cost"]
    ]], columns=columns)

    scaled_input = scaler.transform(input_data)

    # ── Predictions ────────────────────────────────────────────────────────
    clf_pred   = clf.predict(scaled_input)[0]          # 0 = No Disease, 1 = Disease
    clf_prob   = clf.predict_proba(scaled_input)[0]    # [prob_0, prob_1]
    risk_score = round(float(clf_prob[1]) * 100, 1)    # % probability of disease

    # ── Cost logic ─────────────────────────────────────────────────────────
    # If no disease detected, show a nominal ₹1,000 checkup cost rather than
    # the regressor's ~₹8,000 intercept baseline, which is misleading for
    # healthy individuals.
    if clf_pred == 0:
        medical_cost = 1000
    else:
        medical_cost = max(1000, int(reg.predict(scaled_input)[0]))

    # ── Progress bar (100%) ────────────────────────────────────────────────
    st.markdown("""
    <div class="progress-wrap">
        <div class="progress-fill" style="width:100%"></div>
    </div>
    """, unsafe_allow_html=True)

    # ── Verdict card ───────────────────────────────────────────────────────
    if clf_pred == 0:
        verdict_html = f"""
        <div class="result-hero">
            <div class="result-badge">✓ No Disease Detected</div>
            <div class="result-main">You're in good shape!</div>
            <div class="result-sub">
                Disease probability: <strong>{risk_score}%</strong> — well within the healthy range.
            </div>
        </div>
        """
    else:
        verdict_html = f"""
        <div class="result-hero risk">
            <div class="result-badge risk">⚠ Disease Risk Detected</div>
            <div class="result-main">Medical attention advised</div>
            <div class="result-sub">
                Disease probability: <strong>{risk_score}%</strong> — please consult a doctor.
            </div>
        </div>
        """

    st.markdown(verdict_html, unsafe_allow_html=True)

    # ── Cost card ──────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="cost-card">
        <div class="cost-label">Estimated Annual Medical Cost</div>
        <div class="cost-amount">₹{medical_cost:,}</div>
        <div class="cost-note">
            {"Basic consultation / checkup cost" if clf_pred == 0 else "Model-estimated treatment & care cost"}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Recommendations ────────────────────────────────────────────────────
    recs = []
    if s2["smoker"]:
        recs.append(("🚭", "Quitting smoking can reduce your disease risk by up to 50% within 5 years."))
    if s1["bmi"] > 27.5:
        recs.append(("⚖️", f"Your BMI of {s1['bmi']} is above the ideal range. Aim for 18.5–24.9."))
    if s3["stress_level"] > 6:
        recs.append(("🧘", "High stress is a silent driver of cardiovascular risk. Try mindfulness or structured breaks."))
    if s3["sleep_hours"] < 6:
        recs.append(("😴", "Less than 6 hours of sleep raises cortisol and inflammation markers significantly."))
    if s3["daily_steps"] < 5000:
        recs.append(("🚶", "Aim for 8,000–10,000 steps a day to meaningfully lower chronic disease risk."))
    if not recs:
        recs.append(("🌿", "Keep up the healthy habits — annual checkups and consistent activity are your best investment."))

    rec_items = "".join(
        f'<div class="rec-item"><div class="rec-icon">{icon}</div><div>{text}</div></div>'
        for icon, text in recs
    )
    st.markdown(f"""
    <div class="rec-section">
        <div class="rec-eyebrow">Personalised Recommendations</div>
        {rec_items}
    </div>
    """, unsafe_allow_html=True)

    # ── Disclaimer ─────────────────────────────────────────────────────────
    st.markdown("""
    <div class="disclaimer">
        ⚠️ <strong>For informational purposes only.</strong>
        This tool uses a machine learning model trained on anonymised data and does not constitute
        medical advice. Always consult a qualified healthcare professional for diagnosis and treatment.
    </div>
    """, unsafe_allow_html=True)

    # ── Start over ─────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("← Start Over", use_container_width=True):
            for key in ["s1", "s2", "s3"]:
                st.session_state.pop(key, None)
            go(0)
