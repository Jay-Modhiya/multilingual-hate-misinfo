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

import os, sys, random, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
 
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch, yaml
 
st.set_page_config(page_title="MuRIL Hate & Misinfo Detector", page_icon="🛡️",
                   layout="wide", initial_sidebar_state="expanded")
 
st.markdown("""
<style>
    .stApp { background-color: #0f1117; color: #e8e8e8; }
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #1a1f2e 0%, #0f1117 100%); border-right: 1px solid #2d3748; }
    .metric-card { background: #1e2433; border: 1px solid #2d3748; border-radius: 12px; padding: 20px; text-align: center; transition: transform 0.2s; }
    .metric-card:hover { transform: translateY(-2px); }
    .metric-value { font-size: 2rem; font-weight: 700; color: #60a5fa; }
    .metric-label { font-size: 0.85rem; color: #9ca3af; margin-top: 4px; }
    .pred-hate { background: #2d1515; border: 2px solid #ef4444; border-radius: 12px; padding: 16px; text-align: center; }
    .pred-safe { background: #0d2d1a; border: 2px solid #22c55e; border-radius: 12px; padding: 16px; text-align: center; }
    .pred-fake { background: #2d2015; border: 2px solid #f59e0b; border-radius: 12px; padding: 16px; text-align: center; }
    h1, h2, h3 { color: #e8e8e8 !important; }
    .stButton > button { background: linear-gradient(135deg, #3b82f6, #8b5cf6); color: white; border: none; border-radius: 8px; padding: 0.5rem 2rem; font-weight: 600; }
    .stButton > button:hover { opacity: 0.85; }
    .stTextArea textarea { background: #1e2433; color: #e8e8e8; border: 1px solid #2d3748; border-radius: 8px; }
    #MainMenu, footer { visibility: hidden; }
</style>""", unsafe_allow_html=True)
 
# ── Sample text pools (5 each for rotation) ──
ENGLISH_HATE = [
    "You people don't belong here, go back to where you came from.",
    "I hate all people from that religion, they are all criminals.",
    "These foreigners are ruining our country and taking our jobs.",
    "People like you should not be allowed to live in this society.",
    "All of them deserve to be thrown out of this country.",
]
HINDI_HATE = [
    "तुम जैसे लोगों को यहाँ नहीं रहना चाहिए।",
    "इन लोगों से नफरत है मुझे, यहाँ से निकालो इन्हें।",
    "तुम लोग बेकार हो, देश के दुश्मन हो।",
    "इनको यहाँ से भगाओ, ये हमारे देश के नहीं हैं।",
    "नफरत है मुझे इन सबसे, कोई काम के नहीं ये लोग।",
]
HINGLISH_HATE = [
    "Yeh log bahut bure hain, inhe bahar nikalo.",
    "In logo se mujhe nafrat hai, yeh hamare desh ke nahi hain.",
    "Yeh sab criminals hain, inhe jail mein daal do.",
    "Yeh log humara desh barbad kar rahe hain.",
    "Inhe yahan se nikalo, hamare desh mein jagah nahi inke liye.",
]
FAKE_NEWS = [
    "Scientists confirm that 5G towers are spreading the virus to control population.",
    "Secret cure for cancer suppressed by pharmaceutical companies for profit.",
    "Microchips found in COVID vaccines, insider whistleblower reveals shocking truth.",
    "Government is putting chemicals in water to control people's minds.",
    "Aliens have landed and the government is covering it up completely.",
]
 
# ── Config & Model loading ──
@st.cache_resource
def load_cfg():
    with open("configs/config.yaml") as f:
        return yaml.safe_load(f)
 
@st.cache_resource
def load_models():
    from src.data.preprocessor import MuRILTokenizerWrapper
    from src.models.muril_classifier import load_model
    cfg    = load_cfg()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = MuRILTokenizerWrapper(cfg)
    hate_path    = cfg["paths"]["model_dir"] + "/muril_hate_best.pt"
    misinfo_path = cfg["paths"]["model_dir"] + "/muril_misinfo_best.pt"
    hate_model    = load_model(cfg, hate_path,    task="hate")   .to(device) if os.path.exists(hate_path)    else None
    misinfo_model = load_model(cfg, misinfo_path, task="misinfo").to(device) if os.path.exists(misinfo_path) else None
    return tokenizer, hate_model, misinfo_model, device, cfg
 
def predict(text, model, tokenizer, device, label_map):
    model.eval()
    enc = tokenizer.tokenize_single(text)
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        _, logits = model(**enc)
        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
    pred_idx = int(np.argmax(probs))
    return {"label": label_map[pred_idx], "confidence": float(probs[pred_idx])*100,
            "probs": {label_map[i]: float(probs[i])*100 for i in range(len(probs))}}
 
# ── Session state init ──
for key, val in [("input_text",""),("en_idx",0),("hi_idx",0),("hl_idx",0),
                  ("fn_idx",0),("lime_on",False),("drift_history",[]),("drift_input",""),
                  ("hate_result",None),("hate_input",""),("lime_result",None)]:
    if key not in st.session_state:
        st.session_state[key] = val
 
# ── Sidebar ──
with st.sidebar:
    st.markdown("## 🛡️ MuRIL Detector")
    st.markdown("*Multilingual Hate & Misinfo*")
    st.markdown("---")
    page = st.radio("Navigate",
        ["🏠 Home","🔍 Detect","📊 Model Results","⚖️ Bias Analysis","📡 Monitoring","🔢 Batch Predict"],
        label_visibility="collapsed")
    st.markdown("---")
    st.markdown("**Project Info**")
    st.markdown("👤 Jay Modhiya")
    st.markdown("👤 Krishna Nandi")
    st.markdown("🏛️ SIT Pune")
    st.markdown("📚 NLPA + MLOps")
    st.markdown("---")
    st.markdown(f"**Device:** {'🟢 GPU' if torch.cuda.is_available() else '🟡 CPU'}")
    cfg = load_cfg()
    st.markdown(f"**Hate model:** {'🟢 Loaded' if os.path.exists(cfg['paths']['model_dir']+'/muril_hate_best.pt') else '🔴 Not found'}")
    st.markdown(f"**Misinfo model:** {'🟢 Loaded' if os.path.exists(cfg['paths']['model_dir']+'/muril_misinfo_best.pt') else '🔴 Not found'}")
 
# ══════════════════════════════════════════════
#  PAGE 1 — HOME
# ══════════════════════════════════════════════
if page == "🏠 Home":
    st.title("🛡️ Multilingual Hate Speech & Misinformation Detector")
    st.markdown("*Using MuRIL transformer for Hindi, Hinglish & English*")
    st.markdown("---")
    for col, val, label in zip(st.columns(4), ["3","59K+","238M","2"],
                                ["Languages","Training Samples","Model Parameters","Tasks"]):
        with col:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{val}</div><div class="metric-label">{label}</div></div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### 📊 Datasets Used")
        st.dataframe(pd.DataFrame({"Dataset":["Davidson","HASOC 2019","FakeNewsNet"],
            "Language":["English","Hindi/Hinglish","English"],
            "Task":["Hate Speech","Hate Speech","Misinformation"],
            "Samples":["12,970","5,983","40,587"]}), hide_index=True, use_container_width=True)
    with c2:
        st.markdown("### 🏗️ Architecture")
        st.markdown("""
- **Backbone:** `google/muril-base-cased`
- **Parameters:** 238M
- **Head:** Dropout(0.3) → Linear(768→2)
- **Optimizer:** AdamW (lr=1e-5)
- **Training:** Kaggle P100 GPU
- **Hate F1:** 78.74% | **Misinfo F1:** 98.93%
""")
    st.markdown("### 🔬 MLOps Stack")
    for col, tool, desc in zip(st.columns(5),
        ["DVC","MLflow","Docker","GitHub Actions","AWS EC2"],
        ["Data versioning","Experiment tracking","Containerization","CI/CD","Cloud deployment"]):
        with col:
            st.markdown(f'<div class="metric-card"><div style="font-size:1.1rem;font-weight:600;color:#60a5fa">{tool}</div><div class="metric-label">{desc}</div></div>', unsafe_allow_html=True)
 
# ══════════════════════════════════════════════
#  PAGE 2 — DETECT
# ══════════════════════════════════════════════
elif page == "🔍 Detect":
    st.title("🔍 Live Detection")
    st.markdown("Enter text in **English, Hindi, or Hinglish** to detect hate speech or misinformation.")
    st.markdown("---")
    tokenizer, hate_model, misinfo_model, device, cfg = load_models()
 
    st.markdown("**Quick samples** *(click again for next example)*:")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        if st.button("🇬🇧 English Hate 🔄"):
            st.session_state["en_idx"] = (st.session_state["en_idx"]+1) % len(ENGLISH_HATE)
            st.session_state["input_text"] = ENGLISH_HATE[st.session_state["en_idx"]]
            st.session_state["lime_on"] = False
    with c2:
        if st.button("🇮🇳 Hindi Hate 🔄"):
            st.session_state["hi_idx"] = (st.session_state["hi_idx"]+1) % len(HINDI_HATE)
            st.session_state["input_text"] = HINDI_HATE[st.session_state["hi_idx"]]
            st.session_state["lime_on"] = False
    with c3:
        if st.button("🗣️ Hinglish Hate 🔄"):
            st.session_state["hl_idx"] = (st.session_state["hl_idx"]+1) % len(HINGLISH_HATE)
            st.session_state["input_text"] = HINGLISH_HATE[st.session_state["hl_idx"]]
            st.session_state["lime_on"] = False
    with c4:
        if st.button("📰 Fake News 🔄"):
            st.session_state["fn_idx"] = (st.session_state["fn_idx"]+1) % len(FAKE_NEWS)
            st.session_state["input_text"] = FAKE_NEWS[st.session_state["fn_idx"]]
            st.session_state["lime_on"] = False
    st.caption("💡 Each click cycles through 5 different sample texts")
 
    input_text = st.text_area("Enter text to analyze:",
        value=st.session_state.get("input_text",""), height=120,
        placeholder="Type or paste text here — supports English, Hindi, Hinglish...")
 
    col_hate, col_misinfo = st.columns(2)
 
    with col_hate:
        if st.button("🔍 Check Hate Speech", use_container_width=True):
            if not input_text.strip():
                st.warning("Please enter some text first.")
            elif hate_model is None:
                st.error("Hate model not found at: models/saved/muril_hate_best.pt")
            else:
                st.session_state["lime_on"] = False
                st.session_state["lime_result"] = None
                with st.spinner("Analyzing..."):
                    lm = {0:"not_hate",1:"hate"}
                    result = predict(input_text, hate_model, tokenizer, device, lm)
                # Store result in session state so it persists
                st.session_state["hate_result"] = result
                st.session_state["hate_input"]  = input_text
 
        # Show result if available (persists across reruns)
        if st.session_state.get("hate_result"):
            result = st.session_state["hate_result"]
            if result["label"] == "hate":
                st.markdown(f'<div class="pred-hate"><h3 style="color:#ef4444">🚨 HATE SPEECH DETECTED</h3><p style="font-size:1.5rem;font-weight:700;color:#ef4444">{result["confidence"]:.1f}% confidence</p></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="pred-safe"><h3 style="color:#22c55e">✅ NOT HATE SPEECH</h3><p style="font-size:1.5rem;font-weight:700;color:#22c55e">{result["confidence"]:.1f}% confidence</p></div>', unsafe_allow_html=True)
            st.markdown("<br>**Probability breakdown:**", unsafe_allow_html=True)
            for lbl, prob in result["probs"].items():
                st.progress(prob/100, text=f"{lbl}: {prob:.1f}%")
 
            # LIME button — outside the check button block so it persists
            lime_label = "🔍 Show LIME Explanation" if not st.session_state["lime_on"] else "❌ Hide LIME Explanation"
            if st.button(lime_label, key="lime_btn"):
                st.session_state["lime_on"] = not st.session_state["lime_on"]
 
            if st.session_state["lime_on"]:
                lime_text = st.session_state.get("hate_input", "")
                if lime_text:
                    with st.spinner("Generating LIME explanation (~15s on CPU)..."):
                        try:
                            from src.explainability.lime_explainer import HateSpeechLIMEExplainer
                            lm2 = {0:"not_hate", 1:"hate"}
                            explainer = HateSpeechLIMEExplainer(hate_model, tokenizer, device, lm2, cfg)
                            lime_res  = explainer.explain(lime_text, label_idx=1)
                            st.markdown("#### 🔍 LIME Word Importance")
                            st.caption("🟢 Green = pushes toward HATE | 🔴 Red = pushes toward NOT HATE")
                            fig = explainer.plot(lime_res)
                            if fig:
                                st.pyplot(fig)
                                plt.close()
                            if lime_res["word_scores"]:
                                lime_df = pd.DataFrame([
                                    {"Word": w, "Score": round(s,4),
                                     "Direction": "→ HATE" if s>0 else "→ NOT HATE"}
                                    for w,s in sorted(lime_res["word_scores"].items(),
                                                      key=lambda x: abs(x[1]), reverse=True)[:8]
                                ])
                                st.markdown("**Top influential words:**")
                                st.dataframe(lime_df, hide_index=True, use_container_width=True)
                        except Exception as e:
                            st.error(f"LIME error: {str(e)[:200]}")
 
    with col_misinfo:
        if st.button("🔍 Check Misinformation", use_container_width=True):
            if not input_text.strip():
                st.warning("Please enter some text first.")
            elif misinfo_model is None:
                st.error("Misinfo model not found at: models/saved/muril_misinfo_best.pt")
            else:
                with st.spinner("Analyzing..."):
                    lm = {0:"real",1:"fake"}
                    result = predict(input_text, misinfo_model, tokenizer, device, lm)
                if result["label"] == "fake":
                    st.markdown(f'<div class="pred-fake"><h3 style="color:#f59e0b">⚠️ LIKELY MISINFORMATION</h3><p style="font-size:1.5rem;font-weight:700;color:#f59e0b">{result["confidence"]:.1f}% confidence</p></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="pred-safe"><h3 style="color:#22c55e">✅ LIKELY REAL NEWS</h3><p style="font-size:1.5rem;font-weight:700;color:#22c55e">{result["confidence"]:.1f}% confidence</p></div>', unsafe_allow_html=True)
                st.markdown("<br>**Probability breakdown:**", unsafe_allow_html=True)
                for lbl, prob in result["probs"].items():
                    st.progress(prob/100, text=f"{lbl}: {prob:.1f}%")
 
# ══════════════════════════════════════════════
#  PAGE 3 — MODEL RESULTS
# ══════════════════════════════════════════════
elif page == "📊 Model Results":
    st.title("📊 Model Performance Results")
    st.markdown("---")
    st.markdown("### 🏆 MuRIL Performance Summary")
    st.dataframe(pd.DataFrame({
        "Model":     ["MuRIL (Hate — EN+HI)","MuRIL (Misinfo — EN)"],
        "Dataset":   ["Davidson + HASOC","FakeNewsNet"],
        "Accuracy":  ["78.71%","98.93%"],
        "Precision": ["78.83%","98.93%"],
        "Recall":    ["78.71%","98.93%"],
        "F1-Score":  ["78.74%","98.93%"],
    }), hide_index=True, use_container_width=True)
 
    st.markdown("### 🔲 Confusion Matrices")
    c1, c2 = st.columns(2)
    for col, task in zip([c1,c2],["hate","misinfo"]):
        with col:
            p = f"outputs/confusion_matrix_{task}.png"
            if os.path.exists(p):
                st.image(p, caption=f"MuRIL — {task.upper()}", use_container_width=True)
            else:
                st.info(f"Train model first. Expected: {p}")
 
    st.markdown("### 📈 Training Curves (MLflow)")
    try:
        import mlflow
        mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
        client = mlflow.tracking.MlflowClient()
        fig, axes = plt.subplots(1,2,figsize=(12,4))
        fig.patch.set_facecolor("#1e2433")
        for ax, task in zip(axes,["hate","misinfo"]):
            exp = client.get_experiment_by_name(f"{cfg['mlflow']['experiment_name']}-{task}")
            if exp:
                runs = client.search_runs(exp.experiment_id)
                if runs:
                    hist = client.get_metric_history(runs[0].info.run_id,"val_f1")
                    ax.plot([h.step for h in hist],[h.value for h in hist],"o-",color="#60a5fa",linewidth=2)
            ax.set_facecolor("#1e2433"); ax.set_title(f"{task.upper()} Val F1",color="white")
            ax.set_xlabel("Epoch",color="#9ca3af"); ax.set_ylabel("F1 (%)",color="#9ca3af")
            ax.tick_params(colors="#9ca3af")
        plt.tight_layout(); st.pyplot(fig); plt.close()
    except:
        st.info("Run `python -m mlflow ui --port 5000` to view MLflow locally.")
 
# ══════════════════════════════════════════════
#  PAGE 4 — BIAS ANALYSIS
# ══════════════════════════════════════════════
elif page == "⚖️ Bias Analysis":
    st.title("⚖️ Bias & Fairness Analysis")
    st.markdown("Comparing model performance across **English** and **Hindi/Hinglish** groups.")
    st.info("**Why we show bias:** Identifying and quantifying bias is a key part of responsible AI. Our model shows performance gaps across languages — this is expected due to data imbalance and is academically valuable to document.")
    st.markdown("---")
    tokenizer, hate_model, misinfo_model, device, cfg = load_models()
    if hate_model is None:
        st.error("Hate model not found. Place model at: models/saved/muril_hate_best.pt")
    else:
        with st.spinner("Running bias analysis..."):
            try:
                from src.data.loader import load_davidson, load_hasoc
                from src.bias.bias_analyzer import generate_bias_report
                davidson = load_davidson(cfg)
                hasoc    = load_hasoc(cfg)
                test_en  = davidson["test"].copy(); test_en["language"] = "English"
                test_hi  = hasoc["test"].copy();    test_hi["language"] = "Hindi/Hinglish"
                test_df  = pd.concat([test_en, test_hi], ignore_index=True)
                lm = {0:"not_hate",1:"hate"}
                test_df["pred"] = [1 if predict(t,hate_model,tokenizer,device,lm)["label"]=="hate" else 0
                                   for t in test_df["text"].tolist()]
                report = generate_bias_report(test_df, pred_col="pred", label_col="label",
                                              group_col="language", save_path="outputs/bias_plot.png")
                verdict = report["verdict"]
                if "✅" in verdict: st.success(verdict)
                elif "⚠️" in verdict: st.warning(verdict)
                else:
                    st.error(verdict)
                    st.markdown("> **Academic Note:** High bias between English and Hindi/Hinglish is expected. Davidson has 77% offensive class dominance vs HASOC which is balanced. This is a known challenge in multilingual NLP and documenting it is a contribution of our work.")
                st.markdown("### 📊 Per-Language Metrics")
                st.dataframe(report["metrics_df"], use_container_width=True)
                st.markdown("### 📏 Equalized Odds Gaps")
                gaps = report["gaps"]
                for col, key, label in zip(st.columns(3),["f1_gap","fpr_gap","fnr_gap"],["F1 Gap","FPR Gap","FNR Gap"]):
                    with col:
                        st.metric(label, f"{gaps[key]}%", delta="Lower is fairer", delta_color="inverse")
                st.pyplot(report["figure"]); plt.close()
            except Exception as e:
                st.error(f"Error: {e}"); st.exception(e)
 
# ══════════════════════════════════════════════
#  PAGE 5 — MONITORING
# ══════════════════════════════════════════════
elif page == "📡 Monitoring":
    st.title("📡 Data Drift Monitoring")
    st.markdown("Detect when incoming text shifts from the training distribution.")
    st.markdown("""
**How it works:**
- Training text features are stored as reference distribution
- Incoming texts are compared using **Kolmogorov-Smirnov (KS) Test**
- Features compared: text length, word count, avg word length, punctuation ratio, uppercase ratio
- If p-value < 0.05 → **drift detected** → model may need retraining
""")
    st.markdown("---")
    from src.monitoring.drift_detector import DriftDetector
    from src.data.loader import load_davidson
    cfg = load_cfg()
 
    @st.cache_resource
    def get_detector():
        d = DriftDetector(cfg)
        davidson = load_davidson(cfg)
        d.fit(davidson["train"]["text"].tolist()[:2000])
        return d
    detector = get_detector()
 
    st.markdown("### 🧪 Test for Drift")
    st.markdown("**What to paste:** Texts that are very different from normal social media posts (e.g., ALL CAPS, many !!!, different domain).")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("📋 Drift Example (CAPS + urgent)"):
            st.session_state["drift_input"] = "ALERT!!! URGENT MESSAGE TO ALL CITIZENS!!!\nBREAKING NEWS: EVERYTHING IS FAKE!!!\nWARNING!!! DO NOT TRUST THE GOVERNMENT!!!\nURGENT: SHARE THIS IMMEDIATELY!!!"
    with c2:
        if st.button("📋 Normal Example (no drift)"):
            st.session_state["drift_input"] = "The weather is nice today in Mumbai.\nI love watching cricket with friends.\nGovernment announces new infrastructure budget.\nScientists discover new species in Pacific."
 
    incoming_text = st.text_area("Incoming texts (one per line):",
        value=st.session_state.get("drift_input",""), height=150,
        placeholder="Paste new production texts here, one per line...")
 
    if st.button("🔍 Run Drift Detection"):
        if not incoming_text.strip():
            st.warning("Please enter some texts.")
        else:
            texts = [t.strip() for t in incoming_text.strip().split("\n") if t.strip()]
            with st.spinner(f"Analyzing {len(texts)} texts..."):
                result = detector.detect(texts)
                st.session_state["drift_history"].append(result)
            if result["drift_detected"]:
                st.error(f"⚠️ {result['verdict']}")
                st.markdown("**Recommended Action:** Consider retraining the model with new data samples.")
            else:
                st.success(f"✅ {result['verdict']}")
                st.markdown("**Status:** Model distribution is stable. No retraining needed.")
            st.markdown("### 📊 KS Test Results per Feature")
            ks_rows = [{"Feature": f.replace("_"," ").title(),
                        "KS Statistic": v["ks_statistic"],
                        "P-Value": v["p_value"],
                        "Threshold": cfg["monitoring"]["drift_threshold"],
                        "Drift": "🔴 Yes" if v["drift"] else "🟢 No"}
                       for f,v in result["feature_results"].items()]
            st.dataframe(pd.DataFrame(ks_rows), hide_index=True, use_container_width=True)
            if len(texts) >= 5:
                st.markdown("### 📈 Distribution: Training vs Incoming")
                fig = detector.plot_distribution(texts, feature="text_length")
                st.pyplot(fig); plt.close()
 
    st.markdown("### 📋 Drift Check History")
    col_h, col_c = st.columns([4,1])
    with col_c:
        if st.button("🗑️ Clear History"):
            st.session_state["drift_history"] = []
            st.rerun()
    if not st.session_state["drift_history"]:
        st.info("No drift checks run yet.")
    else:
        rows = [{"Timestamp": e["timestamp"], "Texts": e["n_incoming"],
                 "Drift": "Yes" if e["drift_detected"] else "No", "Verdict": e["verdict"]}
                for e in st.session_state["drift_history"]]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
 
# ══════════════════════════════════════════════
#  PAGE 6 — BATCH PREDICT
# ══════════════════════════════════════════════
elif page == "🔢 Batch Predict":
    st.title("🔢 Batch Prediction")
    st.markdown("**Deployment Strategy: BATCH** — Process multiple texts simultaneously")
    st.markdown("---")
    st.markdown("### 📌 Why Batch Deployment?")
    for col, emoji, title, items in zip(st.columns(3),
        ["✅","❌","❌"],
        ["Batch (Chosen)","Real-time","Streaming"],
        [["Content moderation use case","Process 1000s at once","Efficient GPU utilization","Scheduled pipeline","Lower infrastructure cost"],
         ["Always-on GPU needed","High operational cost","Not needed for moderation","Over-engineered","Complex setup"],
         ["Kafka/Kinesis required","Overkill for this use case","Very high infra cost","Complex maintenance","Not justified"]]):
        with col:
            color = "#22c55e" if emoji=="✅" else "#ef4444" if emoji=="❌" else "#f59e0b"
            items_html = "".join(f"• {i}<br>" for i in items)
            st.markdown(f'<div class="metric-card"><div style="font-size:1.1rem;font-weight:600;color:{color}">{emoji} {title}</div><div class="metric-label" style="text-align:left;margin-top:8px">{items_html}</div></div>', unsafe_allow_html=True)
 
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### 🚀 Run Batch Prediction")
    task = st.selectbox("Select Task:", ["hate","misinfo"])
    default_texts = {
        "hate": "You people don't belong here, go back to your country.\nI love watching cricket matches with my friends.\nतुम जैसे लोगों को यहाँ नहीं रहना चाहिए।\nआज मौसम बहुत अच्छा है।\nYeh log bahut bure hain, inhe bahar nikalo.\nAaj match dekhne ka plan hai, bahut maza aayega.",
        "misinfo": "Scientists confirm 5G towers are spreading the virus to control population.\nGovernment announces new infrastructure budget for rural development.\nSecret cure for cancer suppressed by pharmaceutical companies.\nOlympic committee announces host city for 2032 games."
    }
    texts_input = st.text_area("Enter texts (one per line):", value=default_texts[task], height=160)
 
    if st.button("🚀 Run Batch Prediction", use_container_width=True):
        texts = [t.strip() for t in texts_input.strip().split("\n") if t.strip()]
        if not texts:
            st.warning("Please enter some texts.")
        else:
            tokenizer, hate_model, misinfo_model, device, cfg = load_models()
            model = hate_model if task=="hate" else misinfo_model
            lm    = {0:"not_hate",1:"hate"} if task=="hate" else {0:"real",1:"fake"}
            if model is None:
                st.error("Model not found. Place model files in models/saved/")
            else:
                import time
                with st.spinner(f"Processing {len(texts)} texts..."):
                    t0 = time.time()
                    results = [{"Text": t[:70]+"..." if len(t)>70 else t,
                                "Prediction": predict(t,model,tokenizer,device,lm)["label"],
                                "Confidence": f"{predict(t,model,tokenizer,device,lm)['confidence']:.1f}%"}
                               for t in texts]
                    elapsed = time.time()-t0
 
                df = pd.DataFrame(results)
                st.markdown("### 📊 Batch Results")
                st.dataframe(df, use_container_width=True, hide_index=True)
 
                flag_lbl = "hate" if task=="hate" else "fake"
                flagged  = sum(1 for r in results if r["Prediction"]==flag_lbl)
                clean    = len(results)-flagged
 
                st.markdown("### 📈 Batch Summary")
                for col, label, val in zip(st.columns(4),
                    ["Total Texts","Flagged","Clean","Time"],
                    [len(texts),f"{flagged} ({flagged/len(texts)*100:.0f}%)",
                     f"{clean} ({clean/len(texts)*100:.0f}%)",f"{elapsed:.2f}s"]):
                    with col: st.metric(label, val)
 
                st.success(f"✅ Batch complete! Throughput: {len(texts)/elapsed:.1f} texts/sec | Avg: {elapsed/len(texts)*1000:.0f}ms per text")
                st.download_button("📥 Download Results CSV", df.to_csv(index=False),
                                   f"batch_{task}_results.csv","text/csv", use_container_width=True)
 