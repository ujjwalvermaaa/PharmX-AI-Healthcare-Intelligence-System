import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path
import base64
import streamlit.components.v1 as components

from utils import (
    get_patient_dataset,
    get_hospital_dataset,
    get_pharmacy_dataset,
    get_outbreak_dataset,
    build_feature_vector_from_inputs,
    feature_target_split,
    scale_features,
    inverse_disease_label,
    safe_predict_classifier,
    safe_predict_regressor,
    load_pickle,
    load_keras_model,
    align_row_to_feature_names,
    training_feature_names,
)


st.set_page_config(page_title="PharmX AI - Healthcare Intelligence System", page_icon="🧠", layout="wide")

st.title("PharmX AI - Healthcare Intelligence System")
st.caption("Smart healthcare analytics and prediction across diseases, risks, demand, and text")


def _apply_theme(accent_name, mode, radius, density):
    accents = {
        "Blue": "#3b82f6",
        "Teal": "#14b8a6",
        "Purple": "#a855f7",
        "Rose": "#e11d48",
        "Amber": "#f59e0b",
        "Lime": "#84cc16",
    }
    rmap = {"Small": "8px", "Medium": "12px", "Large": "16px"}
    pmap = {"Comfortable": "14px", "Compact": "8px"}
    accent = accents.get(accent_name, "#3b82f6")
    bg = "#0b1220" if mode == "Dark" else "#f8fafc"
    panel = "#0f172a" if mode == "Dark" else "#ffffff"
    text = "#e5e7eb" if mode == "Dark" else "#0f172a"
    muted = "#9ca3af" if mode == "Dark" else "#475569"
    css = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Sora:wght@600;700;800&display=swap');
    :root {{
      --accent: {accent};
      --bg: {bg};
      --panel: {panel};
      --text: {text};
      --muted: {muted};
      --radius: {rmap.get(radius, "12px")};
      --pad: {pmap.get(density, "14px")};
      --font-body: 'Inter', system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, 'Helvetica Neue', Arial, 'Noto Sans', 'Apple Color Emoji', 'Segoe UI Emoji', 'Segoe UI Symbol';
      --font-display: 'Sora', 'Inter', system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, 'Helvetica Neue', Arial, 'Noto Sans';
    }}
    html, body, .stApp {{
      background: var(--bg);
      color: var(--text);
      font-family: var(--font-body);
    }}
    .stTabs [role="tablist"] {{
      background: rgba(11,18,32,0.55);
      border: 1px solid rgba(148,163,184,0.18);
      border-radius: 14px;
      padding: .35rem .6rem;
      backdrop-filter: blur(6px);
      box-shadow: 0 8px 24px rgba(2,6,23,.45);
    }}
    .block-container h1 {{
      color: #f8fafc;
      font-size: 2.8rem;
      line-height: 1.15;
      letter-spacing: .3px;
      text-shadow: 0 3px 14px rgba(2,6,23,.7);
      margin: .2rem 0 .1rem;
      font-family: var(--font-display);
    }}
    .block-container h2, .block-container h3 {{
      color: #f3f4f6;
      text-shadow: 0 2px 10px rgba(2,6,23,.6);
      letter-spacing: .2px;
      font-family: var(--font-display);
    }}
    .block-container p {{
      font-size: 1.04rem;
      line-height: 1.6;
      font-family: var(--font-body);
    }}
    [data-testid="stCaptionContainer"], .block-container .caption {{
      color: #eaeef4 !important;
      font-size: 1.08rem !important;
      text-shadow: 0 2px 12px rgba(2,6,23,.65);
      opacity: .98 !important;
      font-family: var(--font-body);
    }}
    .block-container {{
      padding: 4.5rem 1.2rem 2rem;
      max-width: 100% !important;
      width: 100% !important;
    }}
    [data-testid="stSidebar"] {{
      background: var(--panel);
      color: var(--text);
    }}
    [data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] label {{
      color: var(--text);
    }}
    [data-testid="stHeader"] {{
      background: linear-gradient(120deg, rgba(59,130,246,0.10), rgba(20,184,166,0.08)) !important;
      border-bottom: 1px solid rgba(148,163,184,0.15);
      backdrop-filter: blur(6px);
    }}
    .stTabs [role="tablist"] button p {{
      font-weight: 600;
    }}
    .stTabs [role="tab"] {{
      border-bottom: 2px solid transparent;
      color: #e6edf3;
      text-shadow: 0 2px 8px rgba(2,6,23,.6);
      opacity: .96;
      font-size: .98rem;
      font-weight: 600;
    }}
    .stTabs [role="tab"][aria-selected="true"] {{
      color: var(--accent) !important;
      border-color: var(--accent);
    }}
    /* Glassy form container (affects input pages, not Overview) */
    [data-testid="stForm"] {{
      background: linear-gradient(180deg, rgba(15,23,42,0.48), rgba(2,6,23,0.44));
      border: 1px solid rgba(148,163,184,0.25);
      border-radius: var(--radius);
      padding: calc(var(--pad) + 2px) var(--pad);
      box-shadow: 0 10px 28px rgba(2,6,23,0.45), inset 0 1px 0 rgba(255,255,255,0.05);
      backdrop-filter: blur(12px) saturate(1.15);
    }}
    .stButton>button {{
      background: var(--accent);
      color: white;
      border: 0;
      border-radius: var(--radius);
      padding: 0.6rem 1.1rem;
      font-weight: 600;
      transition: all .15s ease;
      box-shadow: 0 6px 18px rgba(59,130,246,0.25);
    }}
    .stButton>button:hover {{
      filter: brightness(1.06);
      transform: translateY(-1px);
      box-shadow: 0 10px 22px rgba(59,130,246,0.30);
    }}
    input, textarea, select {{ border-radius: var(--radius) !important; }}
    /* Glassy input shells */
    [data-testid="stTextInput"]>div,
    [data-testid="stTextArea"]>div,
    [data-testid="stNumberInput"]>div,
    [data-testid="stSelectbox"]>div,
    [data-testid="stMultiSelect"]>div {{
      background: rgba(15,23,42,0.42) !important;
      border: 1px solid rgba(148,163,184,0.25) !important;
      border-radius: var(--radius) !important;
      box-shadow: 0 6px 20px rgba(2,6,23,0.35), inset 0 1px 0 rgba(255,255,255,0.05);
      backdrop-filter: blur(10px) saturate(1.1);
      transition: border-color .15s ease, box-shadow .15s ease, transform .12s ease;
    }}
    [data-testid="stTextInput"]:focus-within>div,
    [data-testid="stTextArea"]:focus-within>div,
    [data-testid="stNumberInput"]:focus-within>div,
    [data-testid="stSelectbox"]:focus-within>div,
    [data-testid="stMultiSelect"]:focus-within>div {{
      border-color: color-mix(in srgb, var(--accent) 65%, #94a3b8);
      box-shadow: 0 10px 26px rgba(2,6,23,0.45), 0 0 0 1px color-mix(in srgb, var(--accent) 60%, transparent);
      transform: translateY(-1px);
      background: linear-gradient(180deg, rgba(15,23,42,0.46), rgba(15,23,42,0.38));
    }}
    /* Improve number input buttons */
    [data-testid="stNumberInput"] button {{
      background: rgba(148,163,184,0.12) !important;
      border: 1px solid rgba(148,163,184,0.22) !important;
      color: var(--text) !important;
      border-radius: 10px !important;
    }}
    [data-testid="stNumberInput"] button:hover {{
      background: rgba(148,163,184,0.22) !important;
      border-color: color-mix(in srgb, var(--accent) 50%, rgba(148,163,184,0.22)) !important;
    }}
    /* Slider styling */
    [data-testid="stSlider"] .st-emotion-cache-16idsys e1y5xkzn3, /* fallback for some builds */
    [data-testid="stSlider"]>div>div {{
      background: rgba(15,23,42,0.38);
      border: 1px solid rgba(148,163,184,0.18);
      border-radius: var(--radius);
      backdrop-filter: blur(8px);
    }}
    [data-testid="stSlider"] .stSlider .thumb, 
    [role="slider"] {{
      background: var(--accent) !important;
    }}
    .stAlert {{
      border-radius: var(--radius);
    }}
    .hero {{
      margin: 0 0 1.2rem 0;
      padding: 1.6rem 1.4rem;
      border: 1px solid rgba(148,163,184,0.15);
      background: linear-gradient(135deg, rgba(245,158,11,0.10), rgba(2,6,23,0.35));
      border-radius: calc(var(--radius) + 2px);
      box-shadow: 0 8px 24px rgba(2,6,23,0.45);
    }}
    .pill {{
      display: inline-block;
      padding: .28rem .6rem;
      border-radius: 999px;
      border: 1px solid rgba(148,163,184,0.25);
      background: rgba(2,6,23,0.35);
      color: var(--text);
      font-size: .8rem;
      margin-right: .4rem;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid rgba(148,163,184,0.15);
      border-radius: var(--radius);
      padding: 1rem;
      height: 100%;
      box-shadow: 0 6px 18px rgba(2,6,23,0.28);
    }}
    .card h4 {{
      margin: 0 0 .5rem 0;
      color: var(--accent);
      font-family: var(--font-display);
    }}
    .kpi {{
      background: linear-gradient(180deg, rgba(15,23,42,0.9), rgba(2,6,23,0.9));
      border: 1px solid rgba(148,163,184,0.25);
      border-radius: var(--radius);
      padding: .9rem 1rem;
      text-align: center;
      box-shadow: 0 8px 24px rgba(2,6,23,0.45);
    }}
    .kpi .n {{
      font-size: 1.6rem;
      font-weight: 800;
      color: var(--accent);
      line-height: 1.1;
      font-family: var(--font-display);
    }}
    .kpi .l {{
      font-size: .85rem;
      color: var(--muted);
      font-family: var(--font-body);
    }}
    .workflow {{
      display: flex;
      align-items: center;
      gap: .6rem;
      margin: .6rem 0 1rem 0;
    }}
    .wf-step {{
      position: relative;
      flex: 1 1 0;
      text-align: center;
      padding: .6rem .4rem .2rem;
    }}
    .wf-step:not(:last-child)::after {{
      content: "";
      position: absolute;
      top: 1.1rem;
      right: -0.3rem;
      width: .6rem;
      height: 2px;
      background: linear-gradient(90deg, var(--accent), rgba(148,163,184,0.35));
    }}
    .wf-badge {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      width: 28px;
      height: 28px;
      border-radius: 999px;
      background: var(--accent);
      color: #0b0f1a;
      font-weight: 800;
      box-shadow: 0 6px 16px rgba(245,158,11,0.45);
      margin-bottom: .25rem;
    }}
    .wf-title {{
      font-size: .85rem;
      color: var(--text);
      opacity: .95;
      font-weight: 600;
    }}
    [data-testid="stPlotlyChart"] {{
      background: linear-gradient(180deg, rgba(15,23,42,0.42), rgba(2,6,23,0.38));
      border: 1px solid rgba(148,163,184,0.22);
      border-radius: var(--radius);
      box-shadow: 0 8px 24px rgba(2,6,23,0.45), inset 0 1px 0 rgba(255,255,255,0.05);
      backdrop-filter: blur(10px) saturate(1.05);
      padding: .6rem;
    }}
    .hero h3 {{
      font-family: var(--font-display);
      letter-spacing: .3px;
    }}
    /* Selectbox/MultiSelect dropdown (open menu) */
    body div[role="listbox"],
    .stSelectbox div[role="listbox"],
    .stMultiSelect div[role="listbox"] {{
      background: linear-gradient(180deg, rgba(15,23,42,0.60), rgba(2,6,23,0.54)) !important;
      border: 1px solid rgba(148,163,184,0.30) !important;
      border-radius: 14px !important;
      box-shadow: 0 14px 34px rgba(2,6,23,0.55) !important;
      backdrop-filter: blur(12px) saturate(1.08);
      overflow: hidden !important;
    }}
    /* Ensure inner wrappers inherit the glass background */
    body div[role="listbox"] > div,
    body div[role="listbox"] > div > * {{
      background: transparent !important;
    }}
    /* Options */
    .stSelectbox div[role="option"],
    .stMultiSelect div[role="option"],
    body div[role="option"] {{
      color: var(--text) !important;
      padding: .55rem .75rem !important;
      border-radius: 10px !important;
    }}
    body div[role="option"][aria-selected="true"] {{
      background: rgba(59,130,246,0.22) !important;
      color: #e6edf3 !important;
    }}
    body div[role="option"]:hover {{
      background: color-mix(in srgb, var(--accent) 20%, rgba(148,163,184,0.12)) !important;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def _set_background(image_path: Path, overlay_opacity: float = 0.28):
    try:
        data = image_path.read_bytes()
        b64 = base64.b64encode(data).decode()
        css = f"""
        <style>
        [data-testid="stAppViewContainer"] {{
          background-image: linear-gradient(to bottom, rgba(2,6,23,0.66) 0%, rgba(2,6,23,0.52) 22%, rgba(2,6,23,{overlay_opacity}) 55%, rgba(2,6,23,{overlay_opacity}) 100%),
                            url("data:image/jpeg;base64,{b64}");
          background-size: cover;
          background-position: center;
          background-repeat: no-repeat;
          background-attachment: fixed;
        }}
        </style>
        """
        st.markdown(css, unsafe_allow_html=True)
    except Exception:
        pass


def _glass_plot(fig):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e5e7eb", family="Inter, system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, 'Helvetica Neue', Arial, 'Noto Sans'"),
        xaxis=dict(gridcolor="rgba(148,163,184,0.18)", zerolinecolor="rgba(148,163,184,0.22)", linecolor="rgba(148,163,184,0.28)"),
        yaxis=dict(gridcolor="rgba(148,163,184,0.18)", zerolinecolor="rgba(148,163,184,0.22)", linecolor="rgba(148,163,184,0.28)"),
        legend=dict(bgcolor="rgba(2,6,23,0.25)", bordercolor="rgba(148,163,184,0.25)", borderwidth=1),
        margin=dict(l=6, r=6, t=40, b=40),
    )
    return fig

_apply_theme("Amber", "Dark", "Large", "Compact")
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] { display: none; }
    </style>
    """,
    unsafe_allow_html=True
)
_set_background(Path(__file__).resolve().parent / "bg.jpeg", 0.42)

def _scroll_to_top():
    components.html(
        """
        <script>
        (function () {
          try { if ('scrollRestoration' in history) { history.scrollRestoration = 'manual'; } } catch (e) {}
          const goTop = () => { window.scrollTo({ top: 0, left: 0, behavior: 'auto' }); };
          const kick = () => { goTop(); setTimeout(goTop, 50); setTimeout(goTop, 200); setTimeout(goTop, 600); };
          if (document.readyState === 'complete') { kick(); }
          else {
            window.addEventListener('load', kick, { once: true });
            document.addEventListener('DOMContentLoaded', kick, { once: true });
          }
        })();
        </script>
        """,
        height=0,
    )

_scroll_to_top()

@st.cache_data(show_spinner=False)
def load_all_datasets():
    datasets = {}
    try:
        datasets["patient"] = get_patient_dataset()
    except Exception as e:
        datasets["patient_error"] = str(e)
    try:
        datasets["hospital"] = get_hospital_dataset()
    except Exception as e:
        datasets["hospital_error"] = str(e)
    try:
        datasets["pharmacy"] = get_pharmacy_dataset()
    except Exception as e:
        datasets["pharmacy_error"] = str(e)
    try:
        datasets["outbreak"] = get_outbreak_dataset()
    except Exception as e:
        datasets["outbreak_error"] = str(e)
    return datasets


@st.cache_resource(show_spinner=False)
def load_models():
    models = {}
    errors = {}
    def try_load(name, fn):
        try:
            models[name] = fn()
        except Exception as ex:
            errors[name] = str(ex)
    try_load("disease_model", lambda: load_pickle("disease_prediction_model.pkl"))
    try_load("scaler", lambda: load_pickle("scaler.pkl"))
    try_load("disease_encoder", lambda: load_pickle("disease_encoder.pkl"))
    try_load("nlp_model", lambda: load_pickle("nlp_model.pkl"))
    try_load("tfidf", lambda: load_pickle("tfidf.pkl"))
    try_load("nlp_label_encoder", lambda: load_pickle("label_encoder.pkl"))
    try_load("hospital_model", lambda: load_pickle("hospital_model.pkl"))
    try_load("medicine_model", lambda: load_pickle("medicine_model.pkl"))
    try_load("outbreak_model", lambda: load_pickle("outbreak_model.pkl"))
    try_load("health_risk_model", lambda: load_pickle("health_risk_model.pkl"))
    try:
        models["sales_model"] = load_keras_model("pharmacy_sales_dl_model.h5")
    except Exception as ex:
        errors["sales_model"] = str(ex)
    try:
        models["sales_scaler"] = load_pickle("sales_scaler.pkl")
    except Exception as ex:
        errors["sales_scaler"] = str(ex)
    return models, errors


datasets = load_all_datasets()
models, model_errors = load_models()


tabs = st.tabs([
    "Overview",
    "Disease",
    "Disease (Text Prediction)",
    "Hospital Severity",
    "Outbreak Risk",
    "Medicine Demand",
    "Visualizations",
])


with tabs[0]:
    st.subheader("Overview")
    st.markdown(
        """
        <div class="hero">
          <div class="pill">End‑to‑end</div>
          <div class="pill">Multi‑model</div>
          <div class="pill">Production‑ready</div>
          <h3 style="margin:.6rem 0 0; font-weight:800;">Actionable healthcare intelligence across diagnosis, risk, and operations</h3>
          <div style="opacity:.9; margin:.35rem 0 0;">Unified predictions powered by consistent feature pipelines and interactive analytics</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    ds_count = 9
    model_count = sum(1 for k in ["disease_model","nlp_model","hospital_model","outbreak_model","sales_model"] if k in models)
    c_k1, c_k2, c_k3 = st.columns(3)
    with c_k1:
        st.markdown(f'<div class="kpi"><div class="n">{ds_count}</div><div class="l">Datasets Loaded</div></div>', unsafe_allow_html=True)
    with c_k2:
        st.markdown(f'<div class="kpi"><div class="n">{model_count}</div><div class="l">Models Available</div></div>', unsafe_allow_html=True)
    with c_k3:
        st.markdown(f'<div class="kpi"><div class="n">{len(tabs)}</div><div class="l">Interactive Modules</div></div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="card"><h4>Diagnosis</h4><div>Vitals, lifestyle, and symptoms drive robust disease classification with auto‑engineered features.</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="card"><h4>Operations</h4><div>Sales and visits forecasting powered by prescriptions, weather signals, and trends.</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="card"><h4>Risk & Text</h4><div>Hospital severity scoring, outbreak risk, and text‑to‑label disease detection.</div></div>', unsafe_allow_html=True)
    st.markdown(" ")
    st.markdown("**Workflow**")
    st.markdown(
        """
        <div class="workflow">
          <div class="wf-step"><div class="wf-badge">1</div><div class="wf-title">Data</div></div>
          <div class="wf-step"><div class="wf-badge">2</div><div class="wf-title">Clean & Encode</div></div>
          <div class="wf-step"><div class="wf-badge">3</div><div class="wf-title">Feature Engineering</div></div>
          <div class="wf-step"><div class="wf-badge">4</div><div class="wf-title">Train</div></div>
          <div class="wf-step"><div class="wf-badge">5</div><div class="wf-title">Save Models</div></div>
          <div class="wf-step"><div class="wf-badge">6</div><div class="wf-title">App</div></div>
          <div class="wf-step"><div class="wf-badge">7</div><div class="wf-title">Predictions</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.expander("View Data Flow Diagram"):
        labels = ["Data", "Clean & Encode", "Feature Engineering", "Train", "Save Models", "App", "Predictions"]
        node_colors = ["#1e293b","#1f3a5f","#f59e0b","#7c3aed","#059669","#ef4444","#10b981"]
        link_colors = ["rgba(245,158,11,0.25)"] * 6
        source = [0,1,2,3,4,5]
        target = [1,2,3,4,5,6]
        value = [6,6,6,6,6,6]
        fig = go.Figure(data=[go.Sankey(
            node=dict(label=labels, pad=22, thickness=18, line=dict(color="rgba(0,0,0,0)", width=0.5), color=node_colors),
            link=dict(source=source, target=target, value=value, color=link_colors, hovertemplate="%{source.label} → %{target.label}<extra></extra>")
        )])
        fig.update_layout(margin=dict(l=6, r=6, t=6, b=6), font=dict(size=13, color="#e5e7eb"), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(_glass_plot(fig), use_container_width=True)
    st.markdown(" ")
    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("**Why PharmX AI**")
        st.markdown("- Unified predictions for diagnosis, risk, and operations in one place\n- Consistent feature pipelines from notebooks to app\n- Real-time inputs with immediate feedback")
    with col_right:
        st.markdown("**Project Details**")
        st.markdown("- Encoded datasets ensure feature alignment\n- Models loaded from .pkl/.h5 with safe prediction wrappers\n- Visual analytics complement predictions")


with tabs[1]:
    st.subheader("Disease Prediction")
    st.caption("0 = False , 1 = True")

    patient_df = datasets.get("patient")
    if patient_df is None:
        st.error("patient_encoded.csv missing. Please ensure encoded data is available.")
    else:
        feature_cols, _ = feature_target_split(patient_df, "disease")
        feature_cols = [c for c in feature_cols if c != "patient_id"]
        df = patient_df[feature_cols]
        df = df[[c for c in df.columns if not c.startswith("city_") and not c.startswith("region_")]]
        numeric_feature_cols = list(df.select_dtypes(include=[np.number]).columns)
        group_map = {}
        for c in df.columns:
            if "_" in c:
                pref = c.split("_", 1)[0]
                if pref not in group_map:
                    group_map[pref] = []
                group_map[pref].append(c)
        one_hot_groups = {k: v for k, v in group_map.items() if k not in {"city", "region"} and len(v) > 1 and all(set(patient_df[col].unique()).issubset({0, 1}) for col in v)}
        simple_binary = [c for c in df.columns if c not in sum(one_hot_groups.values(), []) and set(patient_df[c].unique()).issubset({0, 1}) and not c.startswith("city_") and not c.startswith("region_")]
        numeric_cols = [c for c in numeric_feature_cols if c not in simple_binary and c != "bmi"]

        presets = {
            "None": {},
            "Profile 1": {
                "age": 50, "gender": 1, "height_cm": 168, "weight_kg": 82,
                "blood_pressure_systolic": 145, "blood_pressure_diastolic": 95,
                "cholesterol": 240, "blood_glucose": 190,
                "heart_rate": 85, "oxygen_level": 95,
                "smoking_status": 1, "alcohol_consumption": 1,
                "physical_activity_level": 0, "sleep_hours": 5,
                "family_history_diabetes": 1, "family_history_heart": 0,
                "symptom_fever": 0, "symptom_cough": 0, "symptom_fatigue": 1,
                "symptom_chest_pain": 0, "symptom_headache": 0,
                "symptom_shortness_of_breath": 0, "symptom_nausea": 0, "symptom_joint_pain": 0
            },
            "Profile 2": {
                "age": 60, "gender": 1, "height_cm": 170, "weight_kg": 85,
                "blood_pressure_systolic": 160, "blood_pressure_diastolic": 100,
                "cholesterol": 260, "blood_glucose": 140,
                "heart_rate": 95, "oxygen_level": 92,
                "smoking_status": 2, "alcohol_consumption": 2,
                "physical_activity_level": 0, "sleep_hours": 5,
                "family_history_diabetes": 0, "family_history_heart": 1,
                "symptom_fever": 0, "symptom_cough": 0, "symptom_fatigue": 1,
                "symptom_chest_pain": 1, "symptom_headache": 0,
                "symptom_shortness_of_breath": 1, "symptom_nausea": 0, "symptom_joint_pain": 0
            },
            "Profile 3": {
                "age": 55, "gender": 1, "height_cm": 172, "weight_kg": 78,
                "blood_pressure_systolic": 170, "blood_pressure_diastolic": 105,
                "cholesterol": 210, "blood_glucose": 110,
                "heart_rate": 88, "oxygen_level": 96,
                "smoking_status": 1, "alcohol_consumption": 1,
                "physical_activity_level": 1, "sleep_hours": 6,
                "family_history_diabetes": 0, "family_history_heart": 1,
                "symptom_fever": 0, "symptom_cough": 0, "symptom_fatigue": 0,
                "symptom_chest_pain": 0, "symptom_headache": 1,
                "symptom_shortness_of_breath": 0, "symptom_nausea": 0, "symptom_joint_pain": 0
            },
            "Profile 4": {
                "age": 30, "gender": 0, "height_cm": 165, "weight_kg": 60,
                "blood_pressure_systolic": 100, "blood_pressure_diastolic": 70,
                "cholesterol": 150, "blood_glucose": 90,
                "heart_rate": 100, "oxygen_level": 94,
                "smoking_status": 0, "alcohol_consumption": 0,
                "physical_activity_level": 1, "sleep_hours": 4,
                "family_history_diabetes": 0, "family_history_heart": 0,
                "symptom_fever": 1, "symptom_cough": 0, "symptom_fatigue": 1,
                "symptom_chest_pain": 0, "symptom_headache": 1,
                "symptom_shortness_of_breath": 0, "symptom_nausea": 1, "symptom_joint_pain": 1
            },
            "Profile 5": {
                "age": 28, "gender": 0, "height_cm": 168, "weight_kg": 65,
                "blood_pressure_systolic": 120, "blood_pressure_diastolic": 80,
                "cholesterol": 170, "blood_glucose": 95,
                "heart_rate": 90, "oxygen_level": 90,
                "smoking_status": 0, "alcohol_consumption": 0,
                "physical_activity_level": 1, "sleep_hours": 6,
                "family_history_diabetes": 0, "family_history_heart": 0,
                "symptom_fever": 0, "symptom_cough": 1, "symptom_fatigue": 1,
                "symptom_chest_pain": 0, "symptom_headache": 0,
                "symptom_shortness_of_breath": 1, "symptom_nausea": 0, "symptom_joint_pain": 0
            },
        }
        preset_choice = st.selectbox("Example Profiles", list(presets.keys()), index=0)
        preset_map = presets.get(preset_choice, {})
        # Pre-populate session state for form defaults
        for c in numeric_feature_cols:
            if c in preset_map:
                st.session_state[f"disease_{c}"] = int(preset_map[c])
        for c in [col for col in df.columns if set(patient_df[col].unique()).issubset({0, 1}) and not col.startswith("city_") and not col.startswith("region_")]:
            if c in preset_map:
                st.session_state[f"disease_{c}"] = int(preset_map[c])
        with st.form("disease_form"):
            cols = st.columns(3)
            numeric_values = {}
            for i, c in enumerate(numeric_cols):
                default = int(np.nanmedian(patient_df[c])) if pd.api.types.is_numeric_dtype(patient_df[c]) else 0
                numeric_values[c] = int(cols[i % 3].number_input(c, value=st.session_state.get(f"disease_{c}", default), step=1, format="%d", key=f"disease_{c}"))
            # Enforce height & weight and auto BMI computation
            if "height_cm" in numeric_values:
                pass
            else:
                numeric_values["height_cm"] = int(cols[0].number_input("height_cm", value=st.session_state.get("disease_height_cm", 165), step=1, format="%d", key="disease_height_cm"))
            if "weight_kg" in numeric_values:
                pass
            else:
                numeric_values["weight_kg"] = int(cols[1].number_input("weight_kg", value=st.session_state.get("disease_weight_kg", 70), step=1, format="%d", key="disease_weight_kg"))
            binary_values = {}
            for i, c in enumerate(simple_binary):
                binary_values[c] = int(cols[i % 3].selectbox(c, [0, 1], index=st.session_state.get(f"disease_{c}", 0), key=f"disease_{c}"))
            group_values = {}
            for i, (grp, cols_in) in enumerate(one_hot_groups.items()):
                options = [x.split("_", 1)[1] for x in cols_in]
                choice = cols[i % 3].selectbox(grp, options, index=0, key=f"disease_{grp}")
                group_values[grp] = choice
            submitted = st.form_submit_button("Predict Disease")
        if submitted and "disease_model" in models and "scaler" in models:
            input_map = {**numeric_values, **binary_values}
            # Merge preset overrides
            for k, v in preset_map.items():
                input_map[k] = v
            # compute BMI from height and weight if column exists
            if "bmi" in df.columns:
                h = max(1, int(input_map.get("height_cm", 165)))
                w = int(input_map.get("weight_kg", 70))
                bmi_calc = int(round(w / ((h / 100.0) ** 2)))
                input_map["bmi"] = bmi_calc
            for grp, choice in group_values.items():
                input_map[grp] = choice
            names = training_feature_names(models["scaler"], models["disease_model"], patient_df, "disease")
            row_full = align_row_to_feature_names(patient_df, input_map, names)
            scaled = models["scaler"].transform(row_full.values)
            y_pred = models["disease_model"].predict(scaled)
            name = inverse_disease_label(y_pred)[0]
            st.success(f"Predicted Disease: {name}")



with tabs[2]:
    st.subheader("Disease (Text Prediction)")
    tweet = st.text_area("Enter tweet text")
    if st.button("Classify Tweet"):
        if all(k in models for k in ["nlp_model","tfidf","nlp_label_encoder"]):
            X = models["tfidf"].transform([tweet])
            y = models["nlp_model"].predict(X)
            label = models["nlp_label_encoder"].inverse_transform(y)[0]
            st.success(f"Predicted: {label}")
        else:
            st.error("NLP models not available")


with tabs[3]:
    st.subheader("Hospital Severity Assessment")
    st.caption("0 = False , 1 = True")
    hospital_df = datasets.get("hospital")
    if hospital_df is None or "hospital_model" not in models:
        st.error("Hospital dataset or model not available")
    else:
        feat_cols, _ = feature_target_split(hospital_df, "severity_level")
        exclude_cols = {"admission_id","patient_id","hospital_id","admission_date","discharge_date","city"}
        df = hospital_df[[c for c in feat_cols if c not in exclude_cols]]
        df = df[[c for c in df.columns if not c.startswith("city_")]]
        presets = {
            "None": {},
            "Case 1": {"length_of_stay": 2, "treatment_cost": 20000, "admission_duration": 1, "cost_per_day": 5000, "icu_required": 0},
            "Case 2": {"length_of_stay": 7, "treatment_cost": 120000, "admission_duration": 3, "cost_per_day": 15000, "icu_required": 1},
            "Case 3": {"length_of_stay": 4, "treatment_cost": 60000, "admission_duration": 2, "cost_per_day": 10000, "icu_required": 0},
            "Case 4": {"length_of_stay": 10, "treatment_cost": 200000, "admission_duration": 5, "cost_per_day": 20000, "icu_required": 1},
            "Case 5": {"length_of_stay": 1, "treatment_cost": 10000, "admission_duration": 1, "cost_per_day": 3000, "icu_required": 0},
        }
        preset_choice = st.selectbox("Example Cases", list(presets.keys()), index=0, key="hospital_preset")
        preset_map = presets.get(preset_choice, {})
        for c in [col for col in df.columns if pd.api.types.is_numeric_dtype(hospital_df[col])]:
            if c in preset_map:
                st.session_state[f"hospital_{c}"] = int(preset_map[c])
        group_map = {}
        for c in df.columns:
            if "_" in c:
                pref = c.split("_", 1)[0]
                group_map.setdefault(pref, []).append(c)
        one_hot_groups = {k: v for k, v in group_map.items() if k != "city" and len(v) > 1 and all(set(hospital_df[col].unique()).issubset({0, 1}) for col in v)}
        simple_binary = [c for c in df.columns if c not in sum(one_hot_groups.values(), []) and set(hospital_df[c].unique()).issubset({0, 1}) and not c.startswith("city_")]
        numeric_cols = [c for c in df.columns if c not in simple_binary and not any(c in v for v in one_hot_groups.values())]
        with st.form("hospital_form"):
            cols = st.columns(3)
            numeric_values = {}
            for i, c in enumerate(numeric_cols):
                default = int(np.nanmedian(hospital_df[c])) if pd.api.types.is_numeric_dtype(hospital_df[c]) else 0
                numeric_values[c] = int(cols[i % 3].number_input(c, value=default, step=1, format="%d", key=f"hospital_{c}"))
            binary_values = {}
            for i, c in enumerate(simple_binary):
                binary_values[c] = int(cols[i % 3].selectbox(c, [0, 1], index=0, key=f"hospital_{c}"))
            group_values = {}
            for i, (grp, cols_in) in enumerate(one_hot_groups.items()):
                options = [x.split("_", 1)[1] for x in cols_in]
                choice = cols[i % 3].selectbox(grp, options, index=0, key=f"hospital_{grp}")
                group_values[grp] = choice
            submitted = st.form_submit_button("Predict Severity")
        if submitted:
            try:
                input_map = {**numeric_values, **binary_values}
                for k, v in preset_map.items():
                    input_map[k] = v
                for grp, choice in group_values.items():
                    input_map[grp] = choice
                names = training_feature_names(None, models["hospital_model"], hospital_df, "severity_level")
                if sum(1 for c in names if c in hospital_df.columns) < int(0.4 * max(1, len(names))):
                    X = df.copy()
                    y = hospital_df["severity_level"]
                    if len(X) > 50000:
                        X = X.sample(50000, random_state=42)
                        y = y.loc[X.index]
                    clf = RandomForestClassifier(n_estimators=120, random_state=42)
                    clf.fit(X.values, y.values)
                    names = list(X.columns)
                    row = align_row_to_feature_names(hospital_df, input_map, names)
                    y_pred = clf.predict(row.values)
                    st.success(f"Predicted Severity Level: {int(y_pred[0])}")
                else:
                    row = align_row_to_feature_names(hospital_df, input_map, names)
                    y = models["hospital_model"].predict(row.values)
                    st.success(f"Predicted Severity Level: {int(y[0])}")
            except Exception as ex:
                st.error(f"Prediction failed. Please check inputs. Details: {ex}")


with tabs[4]:
    st.subheader("Outbreak Risk Analysis")
    outbreak_df = datasets.get("outbreak")
    if outbreak_df is None or "outbreak_model" not in models:
        st.error("Outbreak dataset or model not available")
    else:
        target_guess = "risk_score" if "risk_score" in outbreak_df.columns else None
        feat_cols, _ = feature_target_split(outbreak_df, target_guess or "")
        exclude_cols = {"date","city"}
        df = outbreak_df[[c for c in feat_cols if c not in exclude_cols]]
        df = df[[c for c in df.columns if not c.startswith("city_")]]
        presets = {
            "None": {},
            "Low Risk": {"cases_reported": 20, "deaths": 0, "vaccination_rate": 82, "temperature": 24, "humidity": 55, "hospitalizations": 5, "rainfall": 2, "fatality_rate": 1, "hospitalization_rate": 4},
            "Medium Risk": {"cases_reported": 100, "deaths": 3, "vaccination_rate": 60, "temperature": 28, "humidity": 60, "hospitalizations": 20, "rainfall": 4, "fatality_rate": 6, "hospitalization_rate": 15},
            "High Risk": {"cases_reported": 300, "deaths": 10, "vaccination_rate": 40, "temperature": 30, "humidity": 70, "hospitalizations": 50, "rainfall": 6, "fatality_rate": 12, "hospitalization_rate": 25},
            "Heat Wave": {"cases_reported": 150, "deaths": 5, "vaccination_rate": 50, "temperature": 38, "humidity": 30, "hospitalizations": 30, "rainfall": 0, "fatality_rate": 8, "hospitalization_rate": 18},
            "Monsoon": {"cases_reported": 200, "deaths": 6, "vaccination_rate": 55, "temperature": 26, "humidity": 80, "hospitalizations": 35, "rainfall": 8, "fatality_rate": 7, "hospitalization_rate": 20},
        }
        preset_choice = st.selectbox("Example Outbreak Scenarios", list(presets.keys()), index=0, key="outbreak_preset_tab4")
        preset_map = presets.get(preset_choice, {})
        for c in [col for col in df.columns if pd.api.types.is_numeric_dtype(outbreak_df[col])]:
            if c in preset_map:
                st.session_state[f"outbreak_{c}"] = int(preset_map[c])
        group_map = {}
        for c in df.columns:
            if "_" in c:
                pref = c.split("_", 1)[0]
                group_map.setdefault(pref, []).append(c)
        one_hot_groups = {k: v for k, v in group_map.items() if k != "city" and len(v) > 1 and all(set(outbreak_df[col].unique()).issubset({0, 1}) for col in v)}
        simple_binary = [c for c in df.columns if c not in sum(one_hot_groups.values(), []) and set(outbreak_df[c].unique()).issubset({0, 1}) and not c.startswith("city_")]
        numeric_cols = [c for c in df.columns if c not in simple_binary and not any(c in v for v in one_hot_groups.values())]
        with st.form("outbreak_form"):
            cols = st.columns(3)
            numeric_values = {}
            for i, c in enumerate(numeric_cols):
                default = int(np.nanmedian(outbreak_df[c])) if pd.api.types.is_numeric_dtype(outbreak_df[c]) else 0
                numeric_values[c] = int(cols[i % 3].number_input(c, value=st.session_state.get(f"outbreak_{c}", default), step=1, format="%d", key=f"outbreak_{c}"))
            binary_values = {}
            for i, c in enumerate(simple_binary):
                binary_values[c] = int(cols[i % 3].selectbox(c, [0, 1], index=0, key=f"outbreak_{c}"))
            group_values = {}
            for i, (grp, cols_in) in enumerate(one_hot_groups.items()):
                options = [x.split("_", 1)[1] for x in cols_in]
                choice = cols[i % 3].selectbox(grp, options, index=0, key=f"outbreak_{grp}")
                group_values[grp] = choice
            submitted = st.form_submit_button("Predict Outbreak Risk")
        if submitted:
            input_map = {**numeric_values, **binary_values}
            for k, v in preset_map.items():
                input_map[k] = v
            for grp, choice in group_values.items():
                input_map[grp] = choice
            row = build_feature_vector_from_inputs(outbreak_df, input_map, target=target_guess or "")
            y = safe_predict_regressor("outbreak_model.pkl", row)
            st.success(f"Outbreak Risk: {float(y[0]):.3f}")




with tabs[5]:
    st.subheader("Medicine Demand Prediction")
    if ("sales_model" not in models) or ("sales_scaler" not in models) or (datasets.get("pharmacy") is None):
        err_model = model_errors.get("sales_model")
        err_scaler = model_errors.get("sales_scaler")
        st.error(f"DL model/scaler or pharmacy dataset not available")
        if err_model:
            st.caption(f"Model load error: {err_model}")
        if err_scaler:
            st.caption(f"Scaler load error: {err_scaler}")
        # Try lazy reload once
        try:
            models["sales_model"] = load_keras_model("pharmacy_sales_dl_model.h5")
        except Exception as _:
            pass
        try:
            models["sales_scaler"] = load_pickle("sales_scaler.pkl")
        except Exception as _:
            pass
    else:
        pharmacy_df = datasets["pharmacy"]
        target_guess = "sales" if "sales" in pharmacy_df.columns else None
        feat_cols, _ = feature_target_split(pharmacy_df, target_guess or "")
        exclude_cols = {"date","pharmacy_id","city","medicine_name"}
        presets = {
            "None": {},
            "Scenario 1": {"units_sold": 20, "price_per_unit": 40, "prescriptions_count": 8, "temperature": 25, "humidity": 60, "hospital_visits": 80, "disease_cases": 40},
            "Scenario 2": {"units_sold": 35, "price_per_unit": 55, "prescriptions_count": 12, "temperature": 30, "humidity": 70, "hospital_visits": 120, "disease_cases": 70},
            "Scenario 3": {"units_sold": 50, "price_per_unit": 60, "prescriptions_count": 15, "temperature": 20, "humidity": 50, "hospital_visits": 150, "disease_cases": 90},
            "Scenario 4": {"units_sold": 15, "price_per_unit": 45, "prescriptions_count": 6, "temperature": 35, "humidity": 40, "hospital_visits": 60, "disease_cases": 30},
            "Scenario 5": {"units_sold": 28, "price_per_unit": 50, "prescriptions_count": 10, "temperature": 28, "humidity": 65, "hospital_visits": 100, "disease_cases": 50},
        }
        preset_choice = st.selectbox("Example Scenarios", list(presets.keys()), index=0, key="sales_preset")
        df = pharmacy_df[[c for c in feat_cols if c not in exclude_cols]]
        df = df[[c for c in df.columns if not c.startswith("city_")]]
        preset_map = presets.get(preset_choice, {})
        for c in [col for col in df.columns if pd.api.types.is_numeric_dtype(pharmacy_df[col])]:
            if c in preset_map:
                st.session_state[f"dl_{c}"] = int(preset_map[c])
        group_map = {}
        for c in df.columns:
            if "_" in c:
                pref = c.split("_", 1)[0]
                group_map.setdefault(pref, []).append(c)
        one_hot_groups = {k: v for k, v in group_map.items() if k != "city" and len(v) > 1 and all(set(pharmacy_df[col].unique()).issubset({0, 1}) for col in v)}
        simple_binary = [c for c in df.columns if c not in sum(one_hot_groups.values(), []) and set(pharmacy_df[c].unique()).issubset({0, 1}) and not c.startswith("city_")]
        numeric_cols = [c for c in df.columns if c not in simple_binary and not any(c in v for v in one_hot_groups.values())]
        with st.form("sales_form"):
            cols = st.columns(3)
            numeric_values = {}
            for i, c in enumerate(numeric_cols):
                default = int(np.nanmedian(pharmacy_df[c])) if pd.api.types.is_numeric_dtype(pharmacy_df[c]) else 0
                numeric_values[c] = int(cols[i % 3].number_input(c, value=st.session_state.get(f"dl_{c}", default), step=1, format="%d", key=f"dl_{c}"))
            binary_values = {}
            for i, c in enumerate(simple_binary):
                binary_values[c] = int(cols[i % 3].selectbox(c, [0, 1], index=0, key=f"dl_{c}"))
            group_values = {}
            for i, (grp, cols_in) in enumerate(one_hot_groups.items()):
                options = [x.split("_", 1)[1] for x in cols_in]
                choice = cols[i % 3].selectbox(grp, options, index=0, key=f"dl_{grp}")
                group_values[grp] = choice
            submitted = st.form_submit_button("Predict Demand")
        if submitted:
            input_map = {**numeric_values, **binary_values}
            for k, v in preset_map.items():
                input_map[k] = v
            for grp, choice in group_values.items():
                input_map[grp] = choice
            # Use training columns from notebook: drop units_sold, date, pharmacy_id, medicine_name
            names = [c for c in pharmacy_df.columns if c not in {"units_sold","date","pharmacy_id","medicine_name"}]
            row = align_row_to_feature_names(df, input_map, names)
            row = row[names]
            expected = getattr(models["sales_scaler"], "n_features_in_", row.shape[1])
            if row.shape[1] != expected:
                row = row.iloc[:, :expected]
            try:
                Xs = models["sales_scaler"].transform(row.values)
                preds = models["sales_model"].predict(Xs)
            except Exception:
                preds = models["sales_model"].predict(row.values)
            st.success(f"Demand Prediction: {float(np.ravel(preds)[0]):.2f}")


with tabs[6]:
    st.subheader("Exploratory Visualizations")
    colA, colB = st.columns(2)
    if datasets.get("patient") is not None:
        df = datasets["patient"]
        if "disease" in df.columns:
            try:
                enc = models.get("disease_encoder")
                df_vis = df.copy()
                if enc is not None:
                    df_vis["disease_name"] = enc.inverse_transform(df_vis["disease"].astype(int).values)
                else:
                    df_vis["disease_name"] = df_vis["disease"].astype(str)
                _fig1 = px.histogram(df_vis, x="disease_name", title="Disease Distribution", template="plotly_dark")
                colA.plotly_chart(_glass_plot(_fig1), width='stretch')
            except Exception:
                _fig1b = px.histogram(df, x="disease", title="Disease Distribution", template="plotly_dark")
                colA.plotly_chart(_glass_plot(_fig1b), width='stretch')
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if len(num_cols) >= 2:
            try:
                enc = models.get("disease_encoder")
                df_s = df.sample(min(2000, len(df))).copy()
                if "disease" in df_s.columns and enc is not None:
                    df_s["disease_name"] = enc.inverse_transform(df_s["disease"].astype(int).values)
                    color_col = "disease_name"
                else:
                    color_col = "disease"
                _fig2 = px.scatter(df_s, x=num_cols[0], y=num_cols[1], title="Patient Feature Scatter", opacity=0.7, color=color_col, template="plotly_dark")
                colB.plotly_chart(_glass_plot(_fig2), width='stretch')
            except Exception:
                _fig2b = px.scatter(df.sample(min(2000, len(df))), x=num_cols[0], y=num_cols[1], title="Patient Feature Scatter", opacity=0.7, color="disease", template="plotly_dark")
                colB.plotly_chart(_glass_plot(_fig2b), width='stretch')
        if len(num_cols) >= 3:
            _fig3 = px.violin(df.sample(min(3000, len(df))), y=num_cols[2], box=True, points="suspectedoutliers", title=f"Violin: {num_cols[2]}", template="plotly_dark")
            st.plotly_chart(_glass_plot(_fig3), width='stretch')
    if datasets.get("hospital") is not None:
        dh = datasets["hospital"]
        if "severity_level" in dh.columns:
            _fig4 = px.histogram(dh, x="severity_level", title="Hospital Severity Distribution", template="plotly_dark")
            st.plotly_chart(_glass_plot(_fig4), width='stretch')
    if datasets.get("pharmacy") is not None:
        dfp = datasets["pharmacy"]
        num_cols_p = [c for c in dfp.columns if pd.api.types.is_numeric_dtype(dfp[c])]
        if len(num_cols_p) >= 2:
            _fig5 = px.scatter(dfp.sample(min(5000, len(dfp))), x=num_cols_p[0], y=num_cols_p[1], title="Pharmacy Scatter", color="hospital_visits", template="plotly_dark")
            st.plotly_chart(_glass_plot(_fig5), width='stretch')
        if "total_sales" in dfp.columns:
            _fig6 = px.histogram(dfp, x="total_sales", nbins=60, title="Total Sales Distribution", template="plotly_dark")
            st.plotly_chart(_glass_plot(_fig6), width='stretch')
        if "date" in dfp.columns and "units_sold" in dfp.columns:
            try:
                dfp2 = dfp.copy()
                dfp2["date"] = pd.to_datetime(dfp2["date"], errors="coerce")
                _fig7 = px.line(dfp2.sort_values("date").head(5000), x="date", y="units_sold", title="Units Sold Over Time", template="plotly_dark")
                st.plotly_chart(_glass_plot(_fig7), width='stretch')
            except Exception:
                pass
    if datasets.get("outbreak") is not None:
        dbo = datasets["outbreak"]
        num_cols_o = [c for c in dbo.columns if pd.api.types.is_numeric_dtype(dbo[c])]
        if len(num_cols_o) >= 2:
            _fig8 = px.scatter(dbo.sample(min(5000, len(dbo))), x=num_cols_o[0], y=num_cols_o[1], title="Outbreak Scatter", color="vaccination_rate" if "vaccination_rate" in dbo.columns else None, template="plotly_dark")
            st.plotly_chart(_glass_plot(_fig8), width='stretch')
        try:
            corr = dbo[num_cols_o].corr()
            _fig9 = px.imshow(corr, title="Outbreak Correlation Heatmap", template="plotly_dark")
            st.plotly_chart(_glass_plot(_fig9), width='stretch')
        except Exception:
            pass
