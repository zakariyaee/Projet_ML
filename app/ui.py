import streamlit as st
import requests
import pandas as pd
import io

API_URL = "http://api:8000"

st.set_page_config(
    page_title="PHA Sentinel · Classification d'Astéroïdes",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #f8f9fb;
    color: #1a1f36;
}

.main { background-color: #f8f9fb; }
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 3rem;
    max-width: 1280px;
}

/* ── Top nav bar ── */
.pha-nav {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.85rem 1.8rem;
    background: #ffffff;
    border: 1px solid #e8eaf0;
    border-radius: 14px;
    margin-bottom: 1.8rem;
}
.nav-brand {
    display: flex;
    align-items: center;
    gap: 10px;
}
.nav-dot {
    width: 32px;
    height: 32px;
    background: linear-gradient(135deg, #2171B5, #2171B5);
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 15px;
}
.nav-title {
    font-family: 'Outfit', sans-serif;
    font-size: 1.05rem;
    font-weight: 700;
    color: #1a1f36;
    letter-spacing: -0.02em;
}
.nav-subtitle {
    font-size: 0.72rem;
    color: #8b93a7;
    font-weight: 400;
}
.nav-meta {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    color: #a0aab8;
    display: flex;
    gap: 1.5rem;
    align-items: center;
}
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 3px 10px 3px 7px;
    border-radius: 20px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    font-weight: 500;
}
.status-pill.online {
    background: #ecfdf5;
    color: #065f46;
    border: 1px solid #a7f3d0;
}
.status-pill.offline {
    background: #fef2f2;
    color: #991b1b;
    border: 1px solid #fecaca;
}
.status-pill::before {
    content: '';
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: currentColor;
    display: inline-block;
}

/* ── Hero section ── */
.pha-hero {
    background: linear-gradient(135deg, #2171B5 0%, #2171B5 55%, #2171B5 100%);
    border-radius: 16px;
    padding: 2.8rem 2.5rem 2.4rem;
    margin-bottom: 1.8rem;
    position: relative;
    overflow: hidden;
    color: white;
}
.hero-bg-circle {
    position: absolute;
    width: 360px;
    height: 360px;
    background: rgba(255,255,255,0.05);
    border-radius: 50%;
    top: -120px;
    right: -80px;
}
.hero-bg-circle2 {
    position: absolute;
    width: 220px;
    height: 220px;
    background: rgba(255,255,255,0.04);
    border-radius: 50%;
    bottom: -80px;
    right: 220px;
}
.hero-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: rgba(255,255,255,0.65);
    margin-bottom: 0.7rem;
}
.hero-heading {
    font-family: 'Outfit', sans-serif;
    font-size: 2.4rem;
    font-weight: 800;
    line-height: 1.1;
    letter-spacing: -0.03em;
    color: #ffffff;
    margin-bottom: 0.8rem;
}
.hero-heading em {
    font-style: normal;
    color: rgba(255,255,255,0.7);
}
.hero-body {
    font-size: 0.95rem;
    color: rgba(255,255,255,0.75);
    line-height: 1.7;
    max-width: 560px;
    margin-bottom: 1.8rem;
}
.hero-stats {
    display: flex;
    gap: 2.5rem;
    flex-wrap: wrap;
}
.hero-stat {
    display: flex;
    flex-direction: column;
    gap: 2px;
}
.hero-stat-val {
    font-family: 'Outfit', sans-serif;
    font-size: 1.5rem;
    font-weight: 700;
    color: #ffffff;
    line-height: 1;
}
.hero-stat-lbl {
    font-size: 0.72rem;
    color: rgba(255,255,255,0.55);
    font-weight: 400;
    letter-spacing: 0.04em;
}
.hero-divider {
    width: 1px;
    height: 36px;
    background: rgba(255,255,255,0.2);
    align-self: center;
}

/* ── Metric bar ── */
.metrics-bar {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin-bottom: 1.8rem;
}
.metric-card {
    background: #ffffff;
    border: 1px solid #e8eaf0;
    border-radius: 12px;
    padding: 1.2rem 1.3rem;
}
.metric-label {
    font-size: 0.72rem;
    color: #8b93a7;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 500;
    margin-bottom: 0.5rem;
}
.metric-value {
    font-family: 'Outfit', sans-serif;
    font-size: 1.45rem;
    font-weight: 700;
    color: #2171B5;
    line-height: 1.1;
}
.metric-value.green { color: #059669; }
.metric-value.red { color: #dc2626; }
.metric-value.amber { color: #d97706; }
.metric-value.gray { color: #1a1f36; font-size: 0.95rem; margin-top: 4px; }
.metric-tag {
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem;
    padding: 2px 6px;
    border-radius: 4px;
    background: #eef0fc;
    color: #2171B5;
    margin-top: 4px;
}

/* ── Tab bar ── */
.stTabs [data-baseweb="tab-list"] {
    background: #ffffff !important;
    border-radius: 10px !important;
    border: 1px solid #e8eaf0 !important;
    padding: 4px !important;
    gap: 2px !important;
    margin-bottom: 1.5rem !important;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    color: #6b7280 !important;
    background: transparent !important;
    border-radius: 8px !important;
    padding: 0.5rem 1.3rem !important;
    border: none !important;
    letter-spacing: 0.01em !important;
}
.stTabs [aria-selected="true"] {
    background: #eef0fc !important;
    color: #2171B5 !important;
}

/* ── Section headers ── */
.section-hd {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 1.2rem;
    margin-top: 0.5rem;
}
.section-hd-bar {
    width: 3px;
    height: 18px;
    border-radius: 2px;
    background: #2171B5;
    flex-shrink: 0;
}
.section-hd-bar.amber { background: #d97706; }
.section-hd-bar.green { background: #059669; }
.section-hd-title {
    font-family: 'Outfit', sans-serif;
    font-size: 0.95rem;
    font-weight: 600;
    color: #1a1f36;
    letter-spacing: -0.01em;
}
.section-hd-rule {
    flex: 1;
    height: 1px;
    background: #e8eaf0;
}

/* ── Form cards ── */
.form-card {
    background: #ffffff;
    border: 1px solid #e8eaf0;
    border-radius: 12px;
    padding: 1.4rem 1.5rem;
    height: 100%;
}
.form-card-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #2171B5;
    padding-bottom: 0.8rem;
    margin-bottom: 0.8rem;
    border-bottom: 1px solid #e8eaf0;
    font-weight: 500;
}

/* ── Input overrides ── */
.stNumberInput label,
.stSelectbox label,
.stTextInput label {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.72rem !important;
    font-weight: 500 !important;
    color: #6b7280 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
}
.stNumberInput > div > div > input,
.stTextInput > div > div > input {
    background-color: #f8f9fb !important;
    border: 1px solid #e0e3eb !important;
    color: #1a1f36 !important;
    border-radius: 8px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.82rem !important;
}
.stNumberInput > div > div > input:focus,
.stTextInput > div > div > input:focus {
    border-color: #2171B5 !important;
    box-shadow: 0 0 0 3px rgba(33,113,181,0.08) !important;
    background: #ffffff !important;
}
.stSelectbox > div > div {
    background-color: #f8f9fb !important;
    border: 1px solid #e0e3eb !important;
    color: #1a1f36 !important;
    border-radius: 8px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.82rem !important;
}

/* ── Engineered features band ── */
.eng-band {
    background: #f4f5fd;
    border: 1px solid #dde0f5;
    border-radius: 10px;
    padding: 1.2rem 1.3rem 0.8rem;
    margin: 1rem 0;
}
.eng-band-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #2171B5;
    margin-bottom: 0.8rem;
    font-weight: 500;
}

/* ── Submit button ── */
.stFormSubmitButton > button {
    width: 100% !important;
    background: #2171B5 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.85rem 2rem !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em !important;
    cursor: pointer !important;
    margin-top: 1rem !important;
    transition: all 0.15s ease !important;
}
.stFormSubmitButton > button:hover {
    background: #2171B5 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(33,113,181,0.3) !important;
}

/* ── Result cards ── */
.result-wrapper {
    margin: 1.5rem 0;
}
.result-header {
    display: flex;
    align-items: flex-start;
    gap: 1.2rem;
    margin-bottom: 1.2rem;
}
.result-icon {
    width: 52px;
    height: 52px;
    border-radius: 14px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.5rem;
    flex-shrink: 0;
}
.result-icon.danger { background: #fef2f2; }
.result-icon.safe { background: #ecfdf5; }

.result-danger-card {
    background: #ffffff;
    border: 1px solid #fca5a5;
    border-top: 4px solid #dc2626;
    border-radius: 14px;
    padding: 1.8rem;
}
.result-safe-card {
    background: #ffffff;
    border: 1px solid #6ee7b7;
    border-top: 4px solid #059669;
    border-radius: 14px;
    padding: 1.8rem;
}
.result-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    font-weight: 500;
    margin-bottom: 0.3rem;
}
.result-eyebrow.danger { color: #dc2626; }
.result-eyebrow.safe { color: #059669; }
.result-heading {
    font-family: 'Outfit', sans-serif;
    font-size: 1.45rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    line-height: 1.2;
}
.result-heading.danger { color: #b91c1c; }
.result-heading.safe { color: #047857; }

.result-metrics {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
    margin-top: 1.2rem;
    padding-top: 1.2rem;
    border-top: 1px solid #f0f1f4;
}
.result-met {
    display: flex;
    flex-direction: column;
    gap: 3px;
}
.result-met-lbl {
    font-size: 0.68rem;
    color: #8b93a7;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 500;
}
.result-met-val {
    font-family: 'Outfit', sans-serif;
    font-size: 1.2rem;
    font-weight: 700;
    color: #1a1f36;
}
.result-met-val.danger { color: #dc2626; }
.result-met-val.safe { color: #059669; }
.result-met-val.conf-high { color: #059669; }
.result-met-val.conf-medium { color: #d97706; }
.result-met-val.conf-low { color: #dc2626; }

/* ── Threat meter bar ── */
.threat-bar-wrap {
    margin: 1rem 0 0.5rem;
}
.threat-bar-label {
    font-size: 0.7rem;
    color: #8b93a7;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-weight: 500;
    margin-bottom: 6px;
}
.threat-bar-track {
    height: 8px;
    background: #f0f1f4;
    border-radius: 20px;
    overflow: hidden;
}
.threat-bar-fill {
    height: 100%;
    border-radius: 20px;
    transition: width 0.5s ease;
}

/* ── About section ── */
.about-card {
    background: #ffffff;
    border: 1px solid #e8eaf0;
    border-radius: 12px;
    padding: 1.5rem;
}
.about-body {
    font-size: 0.9rem;
    color: #4b5563;
    line-height: 1.75;
}
.about-body strong { color: #1a1f36; font-weight: 600; }
.info-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.65rem 0;
    border-bottom: 1px solid #f0f1f4;
}
.info-row:last-child { border-bottom: none; }
.info-row-key {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: #8b93a7;
}
.info-row-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: #2171B5;
    font-weight: 500;
    background: #eef0fc;
    padding: 2px 8px;
    border-radius: 5px;
}
.endpoint-row {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 0.55rem 0;
    border-bottom: 1px solid #f0f1f4;
}
.endpoint-row:last-child { border-bottom: none; }
.method-badge {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem;
    padding: 2px 7px;
    border-radius: 4px;
    font-weight: 600;
    min-width: 36px;
    text-align: center;
}
.method-badge.get {
    background: #ecfdf5;
    color: #065f46;
    border: 1px solid #a7f3d0;
}
.method-badge.post {
    background: #eff6ff;
    color: #1e40af;
    border: 1px solid #bfdbfe;
}
.endpoint-path {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: #374151;
}

/* ── Performance metrics ── */
.perf-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 10px;
    margin: 0.8rem 0;
}
.perf-card {
    background: #f8f9fb;
    border: 1px solid #e8eaf0;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    text-align: center;
}
.perf-val {
    font-family: 'Outfit', sans-serif;
    font-size: 1.3rem;
    font-weight: 700;
    color: #2171B5;
    display: block;
}
.perf-lbl {
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: #8b93a7;
    font-weight: 500;
}

/* ── Batch section ── */
.upload-zone-wrap { margin-bottom: 1rem; }
.batch-summary {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
    margin: 1.5rem 0;
}
.batch-stat {
    background: #ffffff;
    border: 1px solid #e8eaf0;
    border-radius: 10px;
    padding: 1.2rem;
    text-align: center;
}
.batch-stat-num {
    font-family: 'Outfit', sans-serif;
    font-size: 1.8rem;
    font-weight: 800;
    display: block;
    line-height: 1;
    margin-bottom: 4px;
}
.batch-stat-lbl {
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #8b93a7;
    font-weight: 500;
}

.format-card {
    background: #f4f5fd;
    border: 1px solid #dde0f5;
    border-radius: 10px;
    padding: 1.2rem 1.3rem;
}
.format-card-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.13em;
    text-transform: uppercase;
    color: #2171B5;
    margin-bottom: 0.7rem;
    font-weight: 500;
}
.format-item {
    font-size: 0.82rem;
    color: #4b5563;
    line-height: 2;
    display: flex;
    align-items: center;
    gap: 6px;
}
.format-item::before {
    content: '·';
    color: #2171B5;
    font-weight: 700;
}

/* ── Misc ── */
.stSuccess {
    background: #ecfdf5 !important;
    border: 1px solid #6ee7b7 !important;
    color: #065f46 !important;
    border-radius: 8px !important;
}
.stError {
    background: #fef2f2 !important;
    border: 1px solid #fca5a5 !important;
    color: #991b1b !important;
    border-radius: 8px !important;
}
.stInfo {
    background: #eff6ff !important;
    border: 1px solid #bfdbfe !important;
    color: #1e40af !important;
    border-radius: 8px !important;
}
.stSpinner > div { border-top-color: #2171B5 !important; }
.stDataFrame {
    border: 1px solid #e8eaf0 !important;
    border-radius: 8px !important;
}
.stDownloadButton > button {
    background: #eef0fc !important;
    border: 1px solid #c7cdf0 !important;
    color: #2171B5 !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.8rem !important;
}
.stDownloadButton > button:hover {
    background: #dde0f8 !important;
}
.stFileUploader > div {
    background: #f8f9fb !important;
    border: 2px dashed #c7cdf0 !important;
    border-radius: 10px !important;
}
.stButton > button {
    background: #2171B5 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
}
.stButton > button:hover {
    background: #2171B5 !important;
    box-shadow: 0 4px 15px rgba(33,113,181,0.25) !important;
}
hr { border-color: #e8eaf0 !important; }
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ── Fetch model info ──────────────────────────────────────────────────────────
model_info_data = None
try:
    r = requests.get(f"{API_URL}/model/info", timeout=5)
    if r.status_code == 200:
        model_info_data = r.json()
except Exception:
    pass

threshold_val = model_info_data.get("threshold", "—") if model_info_data else "—"
model_key = model_info_data.get("model_key", "—") if model_info_data else "—"
strategy = model_info_data.get("strategy", "—") if model_info_data else "—"
is_online = model_info_data is not None

# ── Top Nav ───────────────────────────────────────────────────────────────────
status_html = '<span class="status-pill online">API en ligne</span>' if is_online else '<span class="status-pill offline">API hors ligne</span>'
st.markdown(f"""
<div class="pha-nav">
    <div class="nav-brand">
        <div>
            <div class="nav-title">PHA Sentinel</div>
            <div class="nav-subtitle">Détection d'astéroïdes géocroiseurs</div>
        </div>
    </div>
    <div class="nav-meta">
        {status_html}
    </div>
</div>
""", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="pha-hero">
    <div class="hero-bg-circle"></div>
    <div class="hero-bg-circle2"></div>
    <div class="hero-eyebrow">Système de veille spatiale · Machine Learning</div>
    <div class="hero-heading">Classification des<br><em>Astéroïdes Dangereux</em></div>
    <div class="hero-body">
        Identifiez automatiquement si un astéroïde géocroiseur constitue une menace potentielle 
        à partir de ses caractéristiques orbitales et physiques, grâce à un modèle XGBoost 
        entraîné sur 20 000 objets de la base NASA NeoWs.
    </div>
    <div class="hero-stats">
        <div class="hero-stat">
            <div class="hero-stat-val">20 000</div>
            <div class="hero-stat-lbl">Astéroïdes d'entraînement</div>
        </div>
        <div class="hero-divider"></div>
        <div class="hero-stat">
            <div class="hero-stat-val">22</div>
            <div class="hero-stat-lbl">Variables d'entrée</div>
        </div>
        <div class="hero-divider"></div>
        <div class="hero-stat">
            <div class="hero-stat-val">XGBoost</div>
            <div class="hero-stat-lbl">Algorithme retenu</div>
        </div>
        <div class="hero-divider"></div>
        <div class="hero-stat">
            <div class="hero-stat-val">FN×1000</div>
            <div class="hero-stat-lbl">Coût asymétrique</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Metrics bar ───────────────────────────────────────────────────────────────
status_class = "green" if is_online else "red"
status_label = "EN LIGNE" if is_online else "HORS LIGNE"

st.markdown(f"""
<div class="metrics-bar">
    <div class="metric-card">
        <div class="metric-label">Statut API</div>
        <div class="metric-value {status_class}">{status_label}</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">Modèle actif</div>
        <div class="metric-value gray">{model_key}</div>
        <div class="metric-tag">pipeline complet</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">Seuil décision</div>
        <div class="metric-value amber">{threshold_val}</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">Features</div>
        <div class="metric-value">22</div>
        <div class="metric-tag">orbitales + physiques</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([" À propos du modèle", "  Prédiction unitaire", "  Prédiction par lot"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — About
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    col_a, col_b = st.columns([3, 2], gap="large")

    with col_a:
        st.markdown("""
<div class="section-hd">
    <div class="section-hd-bar"></div>
    <div class="section-hd-title">Mission et contexte métier</div>
    <div class="section-hd-rule"></div>
</div>
""", unsafe_allow_html=True)

        st.markdown("""
<div class="about-card">
    <div class="about-body">
        Ce système utilise un modèle <strong>XGBoost</strong> entraîné sur 20 000 astéroïdes 
        géocroiseurs pour déterminer si un objet spatial est un 
        <strong>PHA (Potentially Hazardous Asteroid)</strong>. L'objectif est d'optimiser 
        l'allocation des ressources d'observation télescopique en priorisant les objets à risque réel.
    </div>
</div>
""", unsafe_allow_html=True)

        st.markdown("<br/>", unsafe_allow_html=True)

        st.markdown("""
<div class="section-hd">
    <div class="section-hd-bar amber"></div>
    <div class="section-hd-title">Asymétrie des coûts</div>
    <div class="section-hd-rule"></div>
</div>
""", unsafe_allow_html=True)

        st.markdown("""
<div class="about-card">
    <div class="about-body">
        Manquer un astéroïde dangereux représente un <strong>risque civilisationnel</strong> 
        (coût FN = 1 000). Une fausse alerte coûte simplement du temps d'observation (coût FP = 20). 
        Le seuil de décision a été optimisé via une matrice de coût asymétrique pour 
        <strong>maximiser le Rappel</strong> au détriment de la Précision.
    </div>
</div>
""", unsafe_allow_html=True)

        st.markdown("<br/>", unsafe_allow_html=True)

        st.markdown("""
<div class="section-hd">
    <div class="section-hd-bar green"></div>
    <div class="section-hd-title">Performances sur le jeu de test</div>
    <div class="section-hd-rule"></div>
</div>
""", unsafe_allow_html=True)

        if model_info_data:
            report = model_info_data.get("test_report", {})
            if report:
                pha = report.get("PHA", report.get("1", {}))
                accuracy = report.get("accuracy", None)
                f1 = pha.get('f1-score', 0)
                recall = pha.get('recall', 0)
                precision = pha.get('precision', 0)

                st.markdown(f"""
<div class="perf-grid">
    <div class="perf-card">
        <span class="perf-val">{f1:.3f}</span>
        <span class="perf-lbl">F1-Score (PHA)</span>
    </div>
    <div class="perf-card">
        <span class="perf-val">{recall:.3f}</span>
        <span class="perf-lbl">Rappel (PHA)</span>
    </div>
    <div class="perf-card">
        <span class="perf-val">{precision:.3f}</span>
        <span class="perf-lbl">Précision (PHA)</span>
    </div>
</div>
""", unsafe_allow_html=True)
                if accuracy:
                    st.markdown(f"""
<div class="about-card" style="margin-top:0.8rem; padding: 0.9rem 1.2rem; display:flex; justify-content:space-between; align-items:center;">
    <span style="font-size:0.8rem; color:#6b7280; font-weight:500; text-transform:uppercase; letter-spacing:0.07em;">Accuracy globale</span>
    <span style="font-family:'Outfit',sans-serif; font-size:1.2rem; font-weight:700; color:#2171B5;">{accuracy:.3f}</span>
</div>
""", unsafe_allow_html=True)
            else:
                st.json(model_info_data)
        else:
            st.error("Impossible de récupérer les métriques — API hors ligne.")

    with col_b:
        st.markdown("""
<div class="section-hd">
    <div class="section-hd-bar"></div>
    <div class="section-hd-title">Architecture du pipeline</div>
    <div class="section-hd-rule"></div>
</div>
""", unsafe_allow_html=True)

        st.markdown("""
<div class="about-card">
    <div class="info-row">
        <span class="info-row-key">Algorithme</span>
        <span class="info-row-val">XGBoost</span>
    </div>
    <div class="info-row">
        <span class="info-row-key">Features</span>
        <span class="info-row-val">22 variables</span>
    </div>
    <div class="info-row">
        <span class="info-row-key">Déséquilibre</span>
        <span class="info-row-val">SMOTE + class_weight</span>
    </div>
    <div class="info-row">
        <span class="info-row-key">Tuning</span>
        <span class="info-row-val">GridSearchCV 5-fold</span>
    </div>
    <div class="info-row">
        <span class="info-row-key">Source données</span>
        <span class="info-row-val">NASA NeoWs API</span>
    </div>
    <div class="info-row">
        <span class="info-row-key">Serialisation</span>
        <span class="info-row-val">joblib (.joblib)</span>
    </div>
</div>
""", unsafe_allow_html=True)

        st.markdown("<br/>", unsafe_allow_html=True)
        st.markdown("""
<div class="section-hd">
    <div class="section-hd-bar"></div>
    <div class="section-hd-title">Endpoints disponibles</div>
    <div class="section-hd-rule"></div>
</div>
""", unsafe_allow_html=True)

        st.markdown("""
<div class="about-card">
    <div class="endpoint-row">
        <span class="method-badge get">GET</span>
        <span class="endpoint-path">/health</span>
    </div>
    <div class="endpoint-row">
        <span class="method-badge get">GET</span>
        <span class="endpoint-path">/model/info</span>
    </div>
    <div class="endpoint-row">
        <span class="method-badge post">POST</span>
        <span class="endpoint-path">/predict</span>
    </div>
    <div class="endpoint-row">
        <span class="method-badge post">POST</span>
        <span class="endpoint-path">/predict/batch</span>
    </div>
</div>
""", unsafe_allow_html=True)

        st.markdown("<br/>", unsafe_allow_html=True)
        st.markdown("""
<div class="section-hd">
    <div class="section-hd-bar amber"></div>
    <div class="section-hd-title">Limites connues</div>
    <div class="section-hd-rule"></div>
</div>
<div class="about-card">
    <div class="about-body" style="font-size:0.82rem;">
        <strong>Biais temporel :</strong> données collectées à un instant donné via NASA NeoWs.<br/>
        <strong>Features ingéniées :</strong> doivent être recalculées correctement.<br/>
        <strong>Classe UNKNOWN :</strong> prédictions moins fiables pour les corps inconnus.<br/>
        <strong>Pas de MLOps :</strong> ré-entraînement non automatique.
    </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Single Prediction
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("""
<div class="section-hd">
    <div class="section-hd-bar"></div>
    <div class="section-hd-title">Renseignez les paramètres de l'astéroïde</div>
    <div class="section-hd-rule"></div>
</div>
<p style="font-size:0.85rem; color:#6b7280; margin-bottom:1.5rem; margin-top:-0.5rem;">
Saisissez les caractéristiques orbitales et physiques pour obtenir une évaluation de risque instantanée.
</p>
""", unsafe_allow_html=True)

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3, gap="medium")

        with col1:
            st.markdown('<div class="form-card"><div class="form-card-title">⬡ Caractéristiques physiques</div>', unsafe_allow_html=True)
            absolute_magnitude_h = st.number_input("Magnitude absolue (H)", value=16.53)
            diameter_mean_km = st.number_input("Diamètre moyen (km)", value=2.12)
            diameter_uncertainty = st.number_input("Incertitude diamètre (km)", value=1.62)
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="form-card"><div class="form-card-title">◎ Approches et vitesses</div>', unsafe_allow_html=True)
            orbiting_body = st.selectbox("Corps orbité", ["EARTH", "JUPTR", "MARS", "VENUS", "UNKNOWN"])
            is_sentry_object = st.selectbox("Objet Sentry", [0, 1], format_func=lambda x: "Non (0)" if x == 0 else "Oui (1)")
            n_approaches = st.number_input("Nombre d'approches", value=18, step=1)
            relative_velocity_km_per_second = st.number_input("Vitesse relative (km/s)", value=26.25)
            max_velocity_km_s = st.number_input("Vitesse maximale (km/s)", value=34.13)
            miss_distance_astronomical = st.number_input("Distance de passage (UA)", value=0.106)
            min_miss_distance_au = st.number_input("Distance min. de passage (UA)", value=0.039)
            st.markdown('</div>', unsafe_allow_html=True)

        with col3:
            st.markdown('<div class="form-card"><div class="form-card-title">◫ Paramètres orbitaux</div>', unsafe_allow_html=True)
            orbit_class_type = st.selectbox("Classe d'orbite", ["APO", "ATE", "AMO", "UNKNOWN"])
            semi_major_axis = st.number_input("Demi-grand axe (UA)", value=1.078)
            inclination = st.number_input("Inclinaison (°)", value=22.8)
            perihelion_distance = st.number_input("Distance au périhélie (UA)", value=0.186)
            perihelion_argument = st.number_input("Argument du périhélie (°)", value=31.43)
            minimum_orbit_intersection = st.number_input("MOID (UA)", value=0.033)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("""
<div class="eng-band">
    <div class="eng-band-title">⚙ Variables ingéniérées</div>
""", unsafe_allow_html=True)

        col4, col5, col6, col7, col8, col9 = st.columns(6)
        orbit_uncertainty = col4.number_input("Incertitude orbitale", value=0.0)
        data_arc_in_days = col5.number_input("Arc données (j)", value=27807.0)
        perihelion_to_aphelion_ratio = col6.number_input("Ratio Péri./Aphélie", value=0.094)
        threat_ratio = col7.number_input("Ratio de menace", value=0.015)
        velocity_distance_ratio = col8.number_input("Ratio Vit./Distance", value=245.43)
        observation_reliability = col9.number_input("Fiabilité obs.", value=0.0)

        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<br/>", unsafe_allow_html=True)
        submit_button = st.form_submit_button(label="  Lancer l'analyse de risque")

    # ── Result ────────────────────────────────────────────────────────────────
    if submit_button:
        payload = {
            "absolute_magnitude_h": absolute_magnitude_h,
            "is_sentry_object": int(is_sentry_object),
            "relative_velocity_km_per_second": relative_velocity_km_per_second,
            "miss_distance_astronomical": miss_distance_astronomical,
            "orbiting_body": orbiting_body,
            "n_approaches": int(n_approaches),
            "min_miss_distance_au": min_miss_distance_au,
            "max_velocity_km_s": max_velocity_km_s,
            "semi_major_axis": semi_major_axis,
            "inclination": inclination,
            "perihelion_distance": perihelion_distance,
            "perihelion_argument": perihelion_argument,
            "orbit_uncertainty": orbit_uncertainty,
            "minimum_orbit_intersection": minimum_orbit_intersection,
            "data_arc_in_days": data_arc_in_days,
            "orbit_class_type": orbit_class_type,
            "diameter_mean_km": diameter_mean_km,
            "diameter_uncertainty": diameter_uncertainty,
            "perihelion_to_aphelion_ratio": perihelion_to_aphelion_ratio,
            "threat_ratio": threat_ratio,
            "velocity_distance_ratio": velocity_distance_ratio,
            "observation_reliability": observation_reliability,
        }

        with st.spinner("Analyse en cours..."):
            try:
                response = requests.post(f"{API_URL}/predict", json=payload)
                if response.status_code == 200:
                    result = response.json()
                    is_pha = "Risque élevé" in result["prediction"]
                    prob_pct = result["probability"] * 100
                    conf = result.get("confidence", "medium").lower()
                    conf_class = f"conf-{conf}"

                    if prob_pct >= 70:
                        bar_color = "linear-gradient(90deg, #dc2626, #ef4444)"
                    elif prob_pct >= 40:
                        bar_color = "linear-gradient(90deg, #d97706, #f59e0b)"
                    else:
                        bar_color = "linear-gradient(90deg, #059669, #10b981)"

                    if is_pha:
                        st.markdown(f"""
<div class="result-danger-card">
    <div class="result-header">
        <div>
            <div class="result-eyebrow danger">Alerte — Objet potentiellement dangereux</div>
            <div class="result-heading danger">PHA DÉTECTÉ</div>
        </div>
    </div>
    <div class="threat-bar-wrap">
        <div class="threat-bar-label">Niveau de menace — {prob_pct:.1f}%</div>
        <div class="threat-bar-track">
            <div class="threat-bar-fill" style="width:{min(prob_pct,100):.1f}%; background:{bar_color};"></div>
        </div>
    </div>
    <div class="result-metrics">
        <div class="result-met">
            <div class="result-met-lbl">Probabilité PHA</div>
            <div class="result-met-val danger">{prob_pct:.1f}%</div>
        </div>
        <div class="result-met">
            <div class="result-met-lbl">Seuil appliqué</div>
            <div class="result-met-val">{result['threshold']}</div>
        </div>
        <div class="result-met">
            <div class="result-met-lbl">Confiance</div>
            <div class="result-met-val {conf_class}">{conf.capitalize()}</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
<div class="result-safe-card">
    <div class="result-header">
        <div>
            <div class="result-eyebrow safe">Objet non dangereux — Aucune alerte</div>
            <div class="result-heading safe">NON-PHA · SÉCURISÉ</div>
        </div>
    </div>
    <div class="threat-bar-wrap">
        <div class="threat-bar-label">Niveau de menace — {prob_pct:.1f}%</div>
        <div class="threat-bar-track">
            <div class="threat-bar-fill" style="width:{min(prob_pct,100):.1f}%; background:{bar_color};"></div>
        </div>
    </div>
    <div class="result-metrics">
        <div class="result-met">
            <div class="result-met-lbl">Probabilité PHA</div>
            <div class="result-met-val safe">{prob_pct:.1f}%</div>
        </div>
        <div class="result-met">
            <div class="result-met-lbl">Seuil appliqué</div>
            <div class="result-met-val">{result['threshold']}</div>
        </div>
        <div class="result-met">
            <div class="result-met-lbl">Confiance</div>
            <div class="result-met-val {conf_class}">{conf.capitalize()}</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

                else:
                    st.error(f"Erreur API : {response.text}")
            except Exception as e:
                st.error(f"Connexion impossible : {e}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Batch
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("""
<div class="section-hd">
    <div class="section-hd-bar"></div>
    <div class="section-hd-title">Analyse par lot (CSV)</div>
    <div class="section-hd-rule"></div>
</div>
<p style="font-size:0.85rem; color:#6b7280; margin-bottom:1.5rem; margin-top:-0.5rem;">
Importez un fichier CSV contenant plusieurs astéroïdes pour obtenir leurs classifications en masse.
</p>
""", unsafe_allow_html=True)

    col_up, col_info = st.columns([2, 1], gap="large")

    with col_up:
        st.markdown('<div class="upload-zone-wrap">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Déposez votre fichier CSV ou cliquez pour parcourir",
            type=["csv"],
            help="Le fichier doit contenir les 22 colonnes de features du modèle"
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with col_info:
        st.markdown("""
<div class="format-card">
    <div class="format-card-title">Format attendu</div>
    <div class="format-item">22 colonnes de features</div>
    <div class="format-item">Headers en première ligne</div>
    <div class="format-item">Encodage UTF-8</div>
    <div class="format-item">Séparateur virgule</div>
</div>
""", unsafe_allow_html=True)

    if uploaded_file is not None:
        df_preview = pd.read_csv(uploaded_file)

        st.markdown("""
<div class="section-hd" style="margin-top:1.5rem;">
    <div class="section-hd-bar"></div>
    <div class="section-hd-title">Aperçu du fichier importé</div>
    <div class="section-hd-rule"></div>
</div>
""", unsafe_allow_html=True)

        info_col1, info_col2, info_col3 = st.columns(3)
        with info_col1:
            st.markdown(f"""
<div class="metric-card">
    <div class="metric-label">Lignes détectées</div>
    <div class="metric-value">{len(df_preview)}</div>
</div>
""", unsafe_allow_html=True)
        with info_col2:
            st.markdown(f"""
<div class="metric-card">
    <div class="metric-label">Colonnes</div>
    <div class="metric-value">{len(df_preview.columns)}</div>
</div>
""", unsafe_allow_html=True)
        with info_col3:
            st.markdown(f"""
<div class="metric-card">
    <div class="metric-label">Fichier</div>
    <div class="metric-value gray">{uploaded_file.name}</div>
</div>
""", unsafe_allow_html=True)

        st.markdown("<br/>", unsafe_allow_html=True)
        st.dataframe(df_preview.head(5), use_container_width=True)

        uploaded_file.seek(0)

        if st.button(" Lancer l'analyse par lot", use_container_width=True):
            with st.spinner("Traitement en cours..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file, "text/csv")}
                    response = requests.post(f"{API_URL}/predict/batch", files=files)

                    if response.status_code == 200:
                        result_df = pd.read_csv(io.BytesIO(response.content))

                        n_pha = (result_df["predicted_class"] == "PHA").sum() if "predicted_class" in result_df.columns else 0
                        n_total = len(result_df)
                        n_safe = n_total - n_pha
                        pct_pha = (n_pha / n_total * 100) if n_total > 0 else 0

                        st.markdown("""
<div class="section-hd" style="margin-top:1.5rem;">
    <div class="section-hd-bar green"></div>
    <div class="section-hd-title">Résultats de la classification</div>
    <div class="section-hd-rule"></div>
</div>
""", unsafe_allow_html=True)

                        st.markdown(f"""
<div class="batch-summary">
    <div class="batch-stat" style="border-top: 3px solid #2171B5;">
        <span class="batch-stat-num" style="color:#2171B5;">{n_total}</span>
        <div class="batch-stat-lbl">Astéroïdes analysés</div>
    </div>
    <div class="batch-stat" style="border-top: 3px solid #dc2626;">
        <span class="batch-stat-num" style="color:#dc2626;">{n_pha}</span>
        <div class="batch-stat-lbl">PHAs détectés ({pct_pha:.1f}%)</div>
    </div>
    <div class="batch-stat" style="border-top: 3px solid #059669;">
        <span class="batch-stat-num" style="color:#059669;">{n_safe}</span>
        <div class="batch-stat-lbl">Non-PHAs sécurisés</div>
    </div>
</div>
""", unsafe_allow_html=True)

                        st.dataframe(result_df, use_container_width=True)

                        st.download_button(
                            label="⬇  Télécharger les prédictions (CSV)",
                            data=response.content,
                            file_name=f"pha_predictions_{uploaded_file.name}",
                            mime="text/csv",
                            use_container_width=True,
                        )
                    else:
                        st.error(f"Erreur API : {response.text}")
                except Exception as e:
                    st.error(f"Connexion impossible : {e}")