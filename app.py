import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import os
import time
import numpy as np

st.set_page_config(
    page_title="SENTINEL // SOC",
    layout="wide",
    page_icon="⬡",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@300;500;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Rajdhani', sans-serif;
}

/* Dark background with subtle grid */
.stApp {
    background-color: #080c10;
    background-image:
        linear-gradient(rgba(0, 255, 140, 0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0, 255, 140, 0.03) 1px, transparent 1px);
    background-size: 40px 40px;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #070b0f !important;
    border-right: 1px solid rgba(0, 255, 140, 0.15) !important;
}

[data-testid="stSidebar"] * {
    color: #a0e8c8 !important;
}

/* Radio buttons */
[data-testid="stSidebar"] .stRadio label {
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 13px !important;
    letter-spacing: 1px;
    text-transform: uppercase;
    padding: 6px 10px;
    border-left: 2px solid transparent;
    transition: all 0.2s;
}

[data-testid="stSidebar"] .stRadio label:hover {
    border-left: 2px solid #00ff8c;
    color: #00ff8c !important;
    background: rgba(0,255,140,0.05);
}

/* Metric cards */
[data-testid="stMetricValue"] {
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 2rem !important;
    color: #00ff8c !important;
    text-shadow: 0 0 20px rgba(0,255,140,0.4);
}

[data-testid="stMetricLabel"] {
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.7rem !important;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #4a7a60 !important;
}

[data-testid="stMetricDelta"] {
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.75rem !important;
}

/* Metric container */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, #0a1a12 0%, #0d1f17 100%);
    border: 1px solid rgba(0, 255, 140, 0.2);
    border-top: 2px solid #00ff8c;
    padding: 20px !important;
    border-radius: 4px !important;
    position: relative;
    overflow: hidden;
}

[data-testid="metric-container"]::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background: radial-gradient(circle at top left, rgba(0,255,140,0.05), transparent 60%);
    pointer-events: none;
}

/* Titles */
h1, h2, h3 {
    font-family: 'Rajdhani', sans-serif !important;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #e0ffe8 !important;
}

h1 { font-size: 2rem !important; font-weight: 700 !important; }
h2 { font-size: 1.3rem !important; font-weight: 500 !important; color: #00ff8c !important; }
h3 { font-size: 1.1rem !important; font-weight: 500 !important; }

/* Divider */
hr {
    border: none !important;
    border-top: 1px solid rgba(0, 255, 140, 0.12) !important;
    margin: 1.5rem 0 !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: #0a1a12;
    border: 1px dashed rgba(0,255,140,0.3) !important;
    border-radius: 4px !important;
    padding: 1rem;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    background: #0a1312 !important;
    border: 1px solid rgba(0,255,140,0.15) !important;
}

/* Warning/error/info */
.stAlert {
    background: rgba(0, 50, 30, 0.6) !important;
    border: 1px solid rgba(0,255,140,0.25) !important;
    border-radius: 4px !important;
    color: #a0e8c8 !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 13px !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #080c10; }
::-webkit-scrollbar-thumb { background: #00ff8c44; border-radius: 2px; }

/* Scan line animation */
@keyframes scan {
    0% { transform: translateY(-100%); }
    100% { transform: translateY(100vh); }
}

/* Blink animation */
@keyframes blink {
    0%, 100% { opacity: 1; }
    50% { opacity: 0; }
}

.cursor {
    display: inline-block;
    width: 10px;
    height: 1em;
    background: #00ff8c;
    margin-left: 4px;
    animation: blink 1s step-start infinite;
    vertical-align: text-bottom;
}

/* Status badge */
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(0, 255, 140, 0.08);
    border: 1px solid rgba(0, 255, 140, 0.3);
    padding: 4px 12px;
    border-radius: 2px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 12px;
    color: #00ff8c;
    letter-spacing: 2px;
    text-transform: uppercase;
}

.status-dot {
    width: 6px;
    height: 6px;
    background: #00ff8c;
    border-radius: 50%;
    box-shadow: 0 0 6px #00ff8c;
    animation: blink 2s ease-in-out infinite;
}

.threat-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(255, 60, 60, 0.1);
    border: 1px solid rgba(255, 60, 60, 0.4);
    padding: 4px 12px;
    border-radius: 2px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 12px;
    color: #ff6060;
    letter-spacing: 2px;
}

/* Hex decorations */
.hex-label {
    font-family: 'Share Tech Mono', monospace;
    font-size: 11px;
    color: #1a4a2e;
    letter-spacing: 1px;
}

.section-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 1rem;
}

.section-header::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(to right, rgba(0,255,140,0.3), transparent);
}

/* About section */
.about-card {
    background: linear-gradient(135deg, #0a1a12 0%, #0d1c15 100%);
    border: 1px solid rgba(0,255,140,0.2);
    border-left: 3px solid #00ff8c;
    padding: 1.5rem;
    border-radius: 4px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 13px;
    line-height: 2;
    color: #7abf9a;
}

.about-card strong {
    color: #00ff8c;
}

/* Sidebar logo */
.sidebar-logo {
    font-family: 'Share Tech Mono', monospace;
    font-size: 22px;
    font-weight: bold;
    color: #00ff8c;
    letter-spacing: 4px;
    text-shadow: 0 0 20px rgba(0,255,140,0.5);
    text-align: center;
    padding: 1rem 0;
}

.sidebar-sub {
    font-family: 'Share Tech Mono', monospace;
    font-size: 10px;
    color: #2a6a46;
    letter-spacing: 3px;
    text-align: center;
    margin-top: -8px;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_assets():
    model = joblib.load('ids_random_forest.pkl')
    scaler = joblib.load('ids_scaler.pkl')
    return model, scaler


# -- SIDEBAR --
with st.sidebar:
    st.markdown('<div class="sidebar-logo">⬡ SENTINEL</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-sub">SECURITY OPERATIONS CENTER</div>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown(
        '<div class="hex-label">// NAVIGATION MODULE</div>',
        unsafe_allow_html=True
    )
    st.markdown("<br>", unsafe_allow_html=True)

    menu = st.radio(
        "",
        ["⬡  TABLEAU DE BORD", "⬡  ANALYSE EN TEMPS RÉEL", "⬡  A PROPOS"],
        label_visibility="collapsed"
    )

    st.markdown("---")

    # System status
    st.markdown(
        '<div class="status-badge"><span class="status-dot"></span>SYSTÈME OPÉRATIONNEL</div>',
        unsafe_allow_html=True
    )
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        f'<div class="hex-label">// SESSION: {int(time.time()) & 0xFFFF:04X}</div>',
        unsafe_allow_html=True
    )

try:
    model, scaler = load_assets()
    model_loaded = True
except Exception as e:
    model_loaded = False


# === DASHBOARD ===
if "TABLEAU" in menu:

    # Header
    st.markdown("""
    <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:0.5rem;">
        <div>
            <div style="font-family:'Share Tech Mono',monospace; font-size:11px; color:#2a6a46; letter-spacing:3px; margin-bottom:4px;">
                // MODULE_01 :: SECURITY_INTELLIGENCE
            </div>
            <h1>TABLEAU DE BORD</h1>
        </div>
        <div class="threat-badge">⚠ SURVEILLANCE ACTIVE</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("MODÈLE IA", "RANDOM FOREST", delta="↑ OPTIMISÉ")
    with col2:
        st.metric("PRÉCISION", "98.5%", delta="STABLE")
    with col3:
        st.metric("STATUT", "PROTÉGÉ", delta="ACTIF")
    with col4:
        st.metric("MENACES / 24H", "0", delta="AUCUNE ALERTE")

    st.markdown("---")

    # Charts
    st.markdown('<div class="section-header"><h2>PERFORMANCE DU MODÈLE</h2></div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        if os.path.exists("confusion_matrix.png"):
            st.markdown('<div class="hex-label">// MATRICE DE CONFUSION</div>', unsafe_allow_html=True)
            st.image("confusion_matrix.png")
        else:
            # Fake heatmap placeholder
            st.markdown('<div class="hex-label">// MATRICE DE CONFUSION [DEMO]</div>', unsafe_allow_html=True)
            z = [[49823, 312], [187, 48901]]
            labels = ["NORMAL", "ATTAQUE"]
            fig = go.Figure(data=go.Heatmap(
                z=z, x=labels, y=labels,
                colorscale=[[0, "#080c10"], [0.5, "#0d3a22"], [1, "#00ff8c"]],
                text=[[str(v) for v in row] for row in z],
                texttemplate="%{text}",
                textfont={"family": "Share Tech Mono", "size": 18, "color": "#e0ffe8"},
                showscale=False
            ))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#080c10",
                font=dict(family="Rajdhani", color="#7abf9a", size=13),
                xaxis=dict(showgrid=False, color="#4a7a60"),
                yaxis=dict(showgrid=False, color="#4a7a60"),
                margin=dict(l=20, r=20, t=30, b=20),
                height=280
            )
            st.plotly_chart(fig, use_container_width=True)

    with c2:
        if os.path.exists("feature_importance.png"):
            st.markdown('<div class="hex-label">// IMPORTANCE DES VARIABLES</div>', unsafe_allow_html=True)
            st.image("feature_importance.png")
        else:
            st.markdown('<div class="hex-label">// IMPORTANCE DES VARIABLES [DEMO]</div>', unsafe_allow_html=True)
            features = ["dst_bytes", "src_bytes", "count", "srv_count", "duration", "land", "wrong_frag", "urgent"]
            importance = [0.31, 0.24, 0.17, 0.10, 0.07, 0.05, 0.04, 0.02]
            fig = go.Figure(go.Bar(
                x=importance[::-1],
                y=features[::-1],
                orientation='h',
                marker=dict(
                    color=importance[::-1],
                    colorscale=[[0, "#0d3a22"], [1, "#00ff8c"]],
                    line=dict(color="rgba(0,255,140,0.3)", width=1)
                )
            ))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#080c10",
                font=dict(family="Share Tech Mono", color="#7abf9a", size=11),
                xaxis=dict(showgrid=True, gridcolor="rgba(0,255,140,0.07)", color="#4a7a60", tickformat=".0%"),
                yaxis=dict(showgrid=False, color="#4a7a60"),
                margin=dict(l=100, r=20, t=30, b=20),
                height=280
            )
            st.plotly_chart(fig, use_container_width=True)


# === REAL-TIME ANALYSIS ===
elif "ANALYSE" in menu:

    st.markdown("""
    <div style="margin-bottom:0.5rem;">
        <div style="font-family:'Share Tech Mono',monospace; font-size:11px; color:#2a6a46; letter-spacing:3px; margin-bottom:4px;">
            // MODULE_02 :: TRAFFIC_ANALYZER
        </div>
        <h1>ANALYSE EN TEMPS RÉEL</h1>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    st.markdown('<div class="hex-label">// IMPORTER LES LOGS RÉSEAU [FORMAT: CSV]</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("", type="csv", label_visibility="collapsed")

    if uploaded_file:
        with st.spinner("⬡  ANALYSE EN COURS..."):
            df = pd.read_csv(uploaded_file)

            if not model_loaded:
                st.error("// ERREUR: Modèles non chargés. Vérifiez les fichiers .pkl")
            else:
                features = scaler.feature_names_in_.tolist()
                X_input = df.reindex(columns=features).fillna(0)
                X_scaled = scaler.transform(X_input)
                preds = model.predict(X_scaled)
                df['Verdict'] = preds

                # Summary metrics
                total = len(df)
                attacks = (df['Verdict'] != 'normal').sum() if df['Verdict'].dtype == object else (df['Verdict'] != 0).sum()
                normal = total - attacks
                rate = (attacks / total * 100) if total > 0 else 0

                st.markdown("---")
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.metric("PAQUETS ANALYSÉS", f"{total:,}")
                with m2:
                    st.metric("INTRUSIONS DÉTECTÉES", f"{attacks:,}", delta=f"{'↑ CRITIQUE' if rate > 10 else '↓ FAIBLE'}")
                with m3:
                    st.metric("TAUX DE MENACE", f"{rate:.1f}%")

                st.markdown("---")

                # Threat chart
                st.markdown('<div class="section-header"><h2>RÉPARTITION DES MENACES</h2></div>', unsafe_allow_html=True)

                counts = df['Verdict'].value_counts().reset_index()
                counts.columns = ['Menace', 'Nombre']

                colors = []
                for m in counts['Menace']:
                    if str(m).lower() in ['normal', '0']:
                        colors.append('#00ff8c')
                    else:
                        colors.append('#ff4040')

                fig = go.Figure(go.Bar(
                    x=counts['Menace'].astype(str),
                    y=counts['Nombre'],
                    marker=dict(
                        color=colors,
                        line=dict(color='rgba(255,255,255,0.1)', width=1)
                    ),
                    text=counts['Nombre'],
                    textposition='outside',
                    textfont=dict(family='Share Tech Mono', size=13, color='#7abf9a')
                ))
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='#080c10',
                    font=dict(family='Rajdhani', color='#7abf9a', size=13),
                    xaxis=dict(showgrid=False, color='#4a7a60', tickfont=dict(family='Share Tech Mono', size=11)),
                    yaxis=dict(showgrid=True, gridcolor='rgba(0,255,140,0.07)', color='#4a7a60'),
                    margin=dict(l=20, r=20, t=40, b=20),
                    height=320,
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)

                # Audit log
                st.markdown("---")
                st.markdown('<div class="section-header"><h2>JOURNAL D\'AUDIT</h2></div>', unsafe_allow_html=True)
                st.markdown('<div class="hex-label">// AFFICHAGE DES 500 DERNIERS ENREGISTREMENTS</div>', unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                st.dataframe(
                    df.tail(500),
                    use_container_width=True,
                    height=350
                )
    else:
        # Empty state with terminal feel
        st.markdown("""
        <div style="
            background: #0a1312;
            border: 1px dashed rgba(0,255,140,0.2);
            padding: 3rem 2rem;
            text-align: center;
            border-radius: 4px;
            margin-top: 1rem;
        ">
            <div style="font-family:'Share Tech Mono',monospace; font-size:36px; color:#1a4a2e; margin-bottom:1rem;">⬡</div>
            <div style="font-family:'Share Tech Mono',monospace; font-size:13px; color:#2a6a46; letter-spacing:2px;">
                EN ATTENTE D'ENTRÉE &gt;&gt;<span class="cursor"></span>
            </div>
            <div style="font-family:'Share Tech Mono',monospace; font-size:11px; color:#1a3a26; margin-top:8px; letter-spacing:1px;">
                DÉPOSER UN FICHIER .CSV POUR DÉMARRER L'ANALYSE
            </div>
        </div>
        """, unsafe_allow_html=True)


# === ABOUT ===
elif "PROPOS" in menu:

    st.markdown("""
    <div style="margin-bottom:0.5rem;">
        <div style="font-family:'Share Tech Mono',monospace; font-size:11px; color:#2a6a46; letter-spacing:3px; margin-bottom:4px;">
            // MODULE_03 :: SYSTEM_INFO
        </div>
        <h1>INFORMATIONS PROJET</h1>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        <div class="about-card">
            <strong>// PROJET</strong><br>
            Système Expert de Détection d'Intrusions (IDS)<br><br>
            <strong>// DÉVELOPPEUR</strong><br>
            Chihab Eddine Gheraibia / Nezar Aymen<br><br>
            <strong>// UNIVERSITÉ</strong><br>
            Badji Mokhtar — Annaba<br><br>
            <strong>// STACK TECHNIQUE</strong><br>
            Python · Scikit-Learn · Streamlit · Plotly<br><br>
            <strong>// ALGORITHME</strong><br>
            Random Forest Classifier — 98.5% précision
        </div>
        """, unsafe_allow_html=True)

    with col2:
        # Architecture mini-diagram
        st.markdown('<div class="hex-label">// PIPELINE D\'ANALYSE</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        steps = [
            ("01", "INGESTION", "Fichier CSV"),
            ("02", "NORMALISATION", "StandardScaler"),
            ("03", "CLASSIFICATION", "Random Forest"),
            ("04", "VERDICT", "Normal / Attaque"),
        ]
        for code, title, sub in steps:
            st.markdown(f"""
            <div style="
                background: #0a1a12;
                border: 1px solid rgba(0,255,140,0.15);
                border-left: 2px solid #00ff8c;
                padding: 10px 14px;
                margin-bottom: 6px;
                border-radius: 2px;
            ">
                <div style="font-family:'Share Tech Mono',monospace; font-size:10px; color:#2a6a46;">{code}</div>
                <div style="font-family:'Rajdhani',sans-serif; font-size:15px; font-weight:600; color:#e0ffe8; letter-spacing:2px;">{title}</div>
                <div style="font-family:'Share Tech Mono',monospace; font-size:11px; color:#4a7a60;">{sub}</div>
            </div>
            """, unsafe_allow_html=True)
    
