"""
app/streamlit_app.py
─────────────────────
Multi-page Streamlit UI for Multilingual Hate Speech & Misinformation Detector.

Pages:
  1. 🏠 Home          — project overview
  2. 🔍 Detect        — live prediction + LIME explanation
  3. 📊 Model Results — confusion matrix, metrics comparison
  4. ⚖️  Bias Analysis — fairness across language groups
  5. 📡 Monitoring    — data drift detection

Run: streamlit run app/streamlit_app.py
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import yaml

# ── Page config (must be first Streamlit call) ──
st.set_page_config(
    page_title="MuRIL Hate & Misinfo Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0f1117; color: #e8e8e8; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1f2e 0%, #0f1117 100%);
        border-right: 1px solid #2d3748;
    }

    /* Cards */
    .metric-card {
        background: #1e2433;
        border: 1px solid #2d3748;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        transition: transform 0.2s;
    }
    .metric-card:hover { transform: translateY(-2px); }
    .metric-value { font-size: 2rem; font-weight: 700; color: #60a5fa; }
    .metric-label { font-size: 0.85rem; color: #9ca3af; margin-top: 4px; }

    /* Prediction boxes */
    .pred-hate {
        background: #2d1515; border: 2px solid #ef4444;
        border-radius: 12px; padding: 16px; text-align: center;
    }
    .pred-safe {
        background: #0d2d1a; border: 2px solid #22c55e;
        border-radius: 12px; padding: 16px; text-align: center;
    }
    .pred-fake {
        background: #2d2015; border: 2px solid #f59e0b;
        border-radius: 12px; padding: 16px; text-align: center;
    }

    /* Headers */
    h1, h2, h3 { color: #e8e8e8 !important; }

    /* Tabs */
    .stTabs [data-baseweb="tab"] { color: #9ca3af; }
    .stTabs [aria-selected="true"] { color: #60a5fa !important; }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6, #8b5cf6);
        color: white; border: none; border-radius: 8px;
        padding: 0.5rem 2rem; font-weight: 600;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.85; }

    /* Text area */
    .stTextArea textarea {
        background: #1e2433; color: #e8e8e8;
        border: 1px solid #2d3748; border-radius: 8px;
    }

    /* Info boxes */
    .stAlert { border-radius: 8px; }

    /* Hide Streamlit branding */
    #MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  Config & Model loading
# ─────────────────────────────────────────────

@st.cache_resource
def load_cfg():
    with open("configs/config.yaml") as f:
        return yaml.safe_load(f)


@st.cache_resource
def load_models():
    """Load both trained models. Returns None if not found."""
    from src.data.preprocessor import MuRILTokenizerWrapper
    from src.models.muril_classifier import load_model

    cfg    = load_cfg()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer   = MuRILTokenizerWrapper(cfg)
    hate_model  = None
    misinfo_model = None

    hate_path   = cfg["paths"]["model_dir"] + "/muril_hate_best.pt"
    misinfo_path= cfg["paths"]["model_dir"] + "/muril_misinfo_best.pt"

    if os.path.exists(hate_path):
        hate_model = load_model(cfg, hate_path, task="hate")
        hate_model = hate_model.to(device)
    if os.path.exists(misinfo_path):
        misinfo_model = load_model(cfg, misinfo_path, task="misinfo")
        misinfo_model = misinfo_model.to(device)

    return tokenizer, hate_model, misinfo_model, device, cfg


def predict(text: str, model, tokenizer, device, label_map: dict) -> dict:
    """Run inference on a single text."""
    model.eval()
    enc = tokenizer.tokenize_single(text)
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        _, logits = model(**enc)
        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
    pred_idx = int(np.argmax(probs))
    return {
        "label":      label_map[pred_idx],
        "confidence": float(probs[pred_idx]) * 100,
        "probs":      {label_map[i]: float(probs[i]) * 100 for i in range(len(probs))},
    }


# ─────────────────────────────────────────────
#  Sidebar Navigation
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🛡️ MuRIL Detector")
    st.markdown("*Multilingual Hate & Misinfo*")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["🏠 Home", "🔍 Detect", "📊 Model Results", "⚖️ Bias Analysis", "📡 Monitoring"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("**Project Info**")
    st.markdown("👤 Jay Modhiya")
    st.markdown("👤 Krishna Nandi")
    st.markdown("🏛️ SIT Pune")
    st.markdown("📚 NLPA + MLOps")
    st.markdown("---")

    # Device status
    device_str = "🟢 GPU" if torch.cuda.is_available() else "🟡 CPU"
    st.markdown(f"**Device:** {device_str}")

    # Model status
    cfg = load_cfg()
    hate_exists   = os.path.exists(cfg["paths"]["model_dir"] + "/muril_hate_best.pt")
    misinfo_exists= os.path.exists(cfg["paths"]["model_dir"] + "/muril_misinfo_best.pt")
    st.markdown(f"**Hate model:** {'🟢 Loaded' if hate_exists else '🔴 Not found'}")
    st.markdown(f"**Misinfo model:** {'🟢 Loaded' if misinfo_exists else '🔴 Not found'}")


# ─────────────────────────────────────────────
#  Page 1 — Home
# ─────────────────────────────────────────────

if page == "🏠 Home":
    st.title("🛡️ Multilingual Hate Speech & Misinformation Detector")
    st.markdown("*Using MuRIL transformer for Hindi, Hinglish & English*")
    st.markdown("---")

    # Stats row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""<div class="metric-card">
            <div class="metric-value">3</div>
            <div class="metric-label">Languages</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class="metric-card">
            <div class="metric-value">59K+</div>
            <div class="metric-label">Training Samples</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""<div class="metric-card">
            <div class="metric-value">238M</div>
            <div class="metric-label">Model Parameters</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown("""<div class="metric-card">
            <div class="metric-value">2</div>
            <div class="metric-label">Tasks</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 📊 Datasets Used")
        datasets_df = pd.DataFrame({
            "Dataset":    ["Davidson", "HASOC 2019", "FakeNewsNet"],
            "Language":   ["English", "Hindi/Hinglish", "English"],
            "Task":       ["Hate Speech", "Hate Speech", "Misinformation"],
            "Samples":    ["12,970", "5,983", "40,587"],
        })
        st.dataframe(datasets_df, hide_index=True, use_container_width=True)

    with col2:
        st.markdown("### 🏗️ Architecture")
        st.markdown("""
        - **Backbone:** `google/muril-base-cased`
        - **Parameters:** 238M
        - **Head:** Dropout(0.3) → Linear(768→2)
        - **Optimizer:** AdamW (lr=1e-5)
        - **Scheduler:** Linear warmup
        - **Training:** Kaggle P100 GPU
        """)

    st.markdown("### 🔬 MLOps Stack")
    col1, col2, col3, col4, col5 = st.columns(5)
    for col, tool, desc in zip(
        [col1, col2, col3, col4, col5],
        ["DVC", "MLflow", "Docker", "GitHub Actions", "AWS EC2"],
        ["Data versioning", "Experiment tracking", "Containerization", "CI/CD", "Cloud deployment"]
    ):
        with col:
            st.markdown(f"""<div class="metric-card">
                <div style="font-size:1.1rem;font-weight:600;color:#60a5fa">{tool}</div>
                <div class="metric-label">{desc}</div>
            </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  Page 2 — Detect
# ─────────────────────────────────────────────

elif page == "🔍 Detect":
    st.title("🔍 Live Detection")
    st.markdown("Enter text in **English, Hindi, or Hinglish** to detect hate speech or misinformation.")
    st.markdown("---")

    tokenizer, hate_model, misinfo_model, device, cfg = load_models()

    # Sample texts
    st.markdown("**Quick samples:**")
    sample_col1, sample_col2, sample_col3 = st.columns(3)
    samples = {
        "English hate":   "You people don't belong here, go back to where you came from.",
        "Hindi hate":     "तुम जैसे लोगों को यहाँ नहीं रहना चाहिए।",
        "Fake news":      "Scientists confirm that 5G towers are spreading the virus to control population.",
    }
    for col, (label, text) in zip([sample_col1, sample_col2, sample_col3], samples.items()):
        with col:
            if st.button(f"📋 {label}"):
                st.session_state["input_text"] = text

    # Text input
    input_text = st.text_area(
        "Enter text to analyze:",
        value=st.session_state.get("input_text", ""),
        height=120,
        placeholder="Type or paste text here — supports English, Hindi, Hinglish...",
    )

    col_hate, col_misinfo = st.columns(2)

    # ── Hate detection ──
    with col_hate:
        if st.button("🔍 Check Hate Speech", use_container_width=True):
            if not input_text.strip():
                st.warning("Please enter some text first.")
            elif hate_model is None:
                st.error("Hate model not found. Please train and download the model first.")
                st.info("Expected at: models/saved/muril_hate_best.pt")
            else:
                with st.spinner("Analyzing..."):
                    label_map = {0: "not_hate", 1: "hate"}
                    result = predict(input_text, hate_model, tokenizer, device, label_map)

                    if result["label"] == "hate":
                        st.markdown(f"""<div class="pred-hate">
                            <h3 style="color:#ef4444">🚨 HATE SPEECH DETECTED</h3>
                            <p style="font-size:1.5rem;font-weight:700;color:#ef4444">
                                {result['confidence']:.1f}% confidence
                            </p>
                        </div>""", unsafe_allow_html=True)
                    else:
                        st.markdown(f"""<div class="pred-safe">
                            <h3 style="color:#22c55e">✅ NOT HATE SPEECH</h3>
                            <p style="font-size:1.5rem;font-weight:700;color:#22c55e">
                                {result['confidence']:.1f}% confidence
                            </p>
                        </div>""", unsafe_allow_html=True)

                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown("**Probability breakdown:**")
                    for label, prob in result["probs"].items():
                        st.progress(prob / 100, text=f"{label}: {prob:.1f}%")

                    # LIME explanation
                    if st.checkbox("Show LIME explanation", key="lime_hate"):
                        with st.spinner("Generating explanation (takes ~30s)..."):
                            try:
                                from src.explainability.lime_explainer import HateSpeechLIMEExplainer
                                explainer = HateSpeechLIMEExplainer(
                                    hate_model, tokenizer, device, label_map, cfg
                                )
                                exp_result = explainer.explain(input_text)
                                fig = explainer.plot(exp_result)
                                st.pyplot(fig)
                                plt.close()
                            except Exception as e:
                                st.error(f"LIME error: {e}")

    # ── Misinfo detection ──
    with col_misinfo:
        if st.button("🔍 Check Misinformation", use_container_width=True):
            if not input_text.strip():
                st.warning("Please enter some text first.")
            elif misinfo_model is None:
                st.error("Misinfo model not found. Please train and download the model first.")
                st.info("Expected at: models/saved/muril_misinfo_best.pt")
            else:
                with st.spinner("Analyzing..."):
                    label_map = {0: "real", 1: "fake"}
                    result = predict(input_text, misinfo_model, tokenizer, device, label_map)

                    if result["label"] == "fake":
                        st.markdown(f"""<div class="pred-fake">
                            <h3 style="color:#f59e0b">⚠️ LIKELY MISINFORMATION</h3>
                            <p style="font-size:1.5rem;font-weight:700;color:#f59e0b">
                                {result['confidence']:.1f}% confidence
                            </p>
                        </div>""", unsafe_allow_html=True)
                    else:
                        st.markdown(f"""<div class="pred-safe">
                            <h3 style="color:#22c55e">✅ LIKELY REAL NEWS</h3>
                            <p style="font-size:1.5rem;font-weight:700;color:#22c55e">
                                {result['confidence']:.1f}% confidence
                            </p>
                        </div>""", unsafe_allow_html=True)

                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown("**Probability breakdown:**")
                    for label, prob in result["probs"].items():
                        st.progress(prob / 100, text=f"{label}: {prob:.1f}%")


# ─────────────────────────────────────────────
#  Page 3 — Model Results
# ─────────────────────────────────────────────

elif page == "📊 Model Results":
    st.title("📊 Model Performance Results")
    st.markdown("---")

    # Results table
    st.markdown("### 🏆 MuRIL Performance Summary")
    results_df = pd.DataFrame({
        "Model":     ["MuRIL (Hate — EN+HI)", "MuRIL (Misinfo — EN)"],
        "Dataset":   ["Davidson + HASOC",      "FakeNewsNet"],
        "Accuracy":  ["—",                     "—"],
        "Precision": ["—",                     "—"],
        "Recall":    ["—",                     "—"],
        "F1-Score":  ["—",                     "—"],
    })

    # Load actual results from MLflow if available
    try:
        import mlflow
        mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
        client = mlflow.tracking.MlflowClient()
        rows = []
        for task, dataset in [("hate", "Davidson+HASOC"), ("misinfo", "FakeNewsNet")]:
            exp_name = f"{cfg['mlflow']['experiment_name']}-{task}"
            exp = client.get_experiment_by_name(exp_name)
            if exp:
                runs = client.search_runs(exp.experiment_id, order_by=["metrics.test_f1 DESC"])
                if runs:
                    r = runs[0]
                    m = r.data.metrics
                    rows.append({
                        "Model":     f"MuRIL ({task.upper()})",
                        "Dataset":   dataset,
                        "Accuracy":  f"{m.get('test_accuracy', 0):.2f}%",
                        "Precision": f"{m.get('test_precision', 0):.2f}%",
                        "Recall":    f"{m.get('test_recall', 0):.2f}%",
                        "F1-Score":  f"{m.get('test_f1', 0):.2f}%",
                    })
        if rows:
            results_df = pd.DataFrame(rows)
    except Exception:
        pass

    st.dataframe(results_df, hide_index=True, use_container_width=True)

    # Confusion matrices
    st.markdown("### 🔲 Confusion Matrices")
    col1, col2 = st.columns(2)
    for col, task in zip([col1, col2], ["hate", "misinfo"]):
        with col:
            cm_path = f"outputs/confusion_matrix_{task}.png"
            if os.path.exists(cm_path):
                st.image(cm_path, caption=f"MuRIL — {task.upper()} Task",
                         use_container_width=True)
            else:
                st.info(f"Confusion matrix not found.\nTrain model and it will appear here.\nExpected: {cm_path}")

    # Training curves from MLflow
    st.markdown("### 📈 Training Progress (from MLflow)")
    try:
        import mlflow
        mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
        client = mlflow.tracking.MlflowClient()

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.patch.set_facecolor("#1e2433")

        for ax, task in zip(axes, ["hate", "misinfo"]):
            exp_name = f"{cfg['mlflow']['experiment_name']}-{task}"
            exp = client.get_experiment_by_name(exp_name)
            if exp:
                runs = client.search_runs(exp.experiment_id)
                if runs:
                    run_id = runs[0].info.run_id
                    history = client.get_metric_history(run_id, "val_f1")
                    epochs = [h.step for h in history]
                    values = [h.value for h in history]
                    ax.plot(epochs, values, "o-", color="#60a5fa", linewidth=2)
                    ax.set_facecolor("#1e2433")
                    ax.set_title(f"{task.upper()} — Val F1", color="white")
                    ax.set_xlabel("Epoch", color="#9ca3af")
                    ax.set_ylabel("F1 (%)", color="#9ca3af")
                    ax.tick_params(colors="#9ca3af")
            else:
                ax.text(0.5, 0.5, "No MLflow data yet", ha="center",
                        va="center", color="#9ca3af", transform=ax.transAxes)
                ax.set_facecolor("#1e2433")

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    except Exception as e:
        st.info(f"MLflow data not available yet. Train the model first.")


# ─────────────────────────────────────────────
#  Page 4 — Bias Analysis
# ─────────────────────────────────────────────

elif page == "⚖️ Bias Analysis":
    st.title("⚖️ Bias & Fairness Analysis")
    st.markdown("Comparing model performance across **English**, **Hindi**, and **Hinglish** groups.")
    st.markdown("---")

    tokenizer, hate_model, misinfo_model, device, cfg = load_models()

    if hate_model is None:
        st.error("Hate model not found. Please train and download the model first.")
    else:
        with st.spinner("Running bias analysis on test set..."):
            try:
                from src.data.loader import load_davidson, load_hasoc
                from src.bias.bias_analyzer import generate_bias_report

                davidson = load_davidson(cfg)
                hasoc    = load_hasoc(cfg)

                # Build test dataframe with language labels
                test_en = davidson["test"].copy()
                test_en["language"] = "English"
                test_hi = hasoc["test"].copy()
                test_hi["language"] = "Hindi/Hinglish"

                test_df = pd.concat([test_en, test_hi], ignore_index=True)

                # Get predictions
                label_map = {0: "not_hate", 1: "hate"}
                preds = []
                for text in test_df["text"].tolist():
                    result = predict(text, hate_model, tokenizer, device, label_map)
                    preds.append(1 if result["label"] == "hate" else 0)
                test_df["pred"] = preds

                report = generate_bias_report(
                    test_df, pred_col="pred", label_col="label",
                    group_col="language",
                    save_path="outputs/bias_plot.png"
                )

                # Verdict
                verdict = report["verdict"]
                if "✅" in verdict:
                    st.success(verdict)
                elif "⚠️" in verdict:
                    st.warning(verdict)
                else:
                    st.error(verdict)

                st.markdown("### 📊 Per-Language Metrics")
                st.dataframe(report["metrics_df"], use_container_width=True)

                st.markdown("### 📏 Equalized Odds Gaps")
                gaps = report["gaps"]
                g1, g2, g3 = st.columns(3)
                with g1:
                    st.metric("F1 Gap", f"{gaps['f1_gap']}%",
                              delta="Lower is fairer", delta_color="inverse")
                with g2:
                    st.metric("FPR Gap", f"{gaps['fpr_gap']}%",
                              delta="Lower is fairer", delta_color="inverse")
                with g3:
                    st.metric("FNR Gap", f"{gaps['fnr_gap']}%",
                              delta="Lower is fairer", delta_color="inverse")

                st.pyplot(report["figure"])
                plt.close()

            except Exception as e:
                st.error(f"Bias analysis error: {e}")
                st.exception(e)


# ─────────────────────────────────────────────
#  Page 5 — Monitoring
# ─────────────────────────────────────────────

elif page == "📡 Monitoring":
    st.title("📡 Data Drift Monitoring")
    st.markdown("Detect when incoming text shifts from the training distribution.")
    st.markdown("---")

    from src.monitoring.drift_detector import DriftDetector
    from src.data.loader import load_davidson

    cfg = load_cfg()

    # Load reference data
    @st.cache_resource
    def get_detector():
        detector = DriftDetector(cfg)
        davidson = load_davidson(cfg)
        ref_texts = davidson["train"]["text"].tolist()[:2000]
        detector.fit(ref_texts)
        return detector

    detector = get_detector()

    st.markdown("### 🧪 Test for Drift")
    st.markdown("Paste incoming/production text samples below (one per line):")

    incoming_text = st.text_area(
        "Incoming texts:",
        height=150,
        placeholder="Paste new texts here, one per line...\nExample:\nALERT! URGENT MESSAGE!!!\nBREAKING NEWS: Everything is fake!!!",
    )

    if st.button("🔍 Run Drift Detection", use_container_width=False):
        if not incoming_text.strip():
            st.warning("Please enter some texts.")
        else:
            texts = [t.strip() for t in incoming_text.strip().split("\n") if t.strip()]
            with st.spinner(f"Analyzing {len(texts)} texts..."):
                result = detector.detect(texts)

            if result["drift_detected"]:
                st.error(f"⚠️ {result['verdict']}")
            else:
                st.success(f"✅ {result['verdict']}")

            st.markdown("### 📊 Feature-Level KS Test Results")
            ks_rows = []
            for feat, vals in result["feature_results"].items():
                ks_rows.append({
                    "Feature":       feat.replace("_", " ").title(),
                    "KS Statistic":  vals["ks_statistic"],
                    "P-Value":       vals["p_value"],
                    "Drift":         "🔴 Yes" if vals["drift"] else "🟢 No",
                })
            st.dataframe(pd.DataFrame(ks_rows), hide_index=True, use_container_width=True)

            # Distribution plot
            if len(texts) >= 5:
                st.markdown("### 📈 Distribution Comparison")
                fig = detector.plot_distribution(texts, feature="text_length")
                st.pyplot(fig)
                plt.close()

    # Drift history
    st.markdown("### 📋 Drift Check History")
    history_df = detector.get_drift_summary()
    if history_df.empty:
        st.info("No drift checks run yet. Use the tool above.")
    else:
        st.dataframe(history_df, hide_index=True, use_container_width=True)
