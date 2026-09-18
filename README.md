<div align="center">

# 🛡️ FORGESHIELD

### **AI-Based Industrial Safety & Predictive Risk Management**

**Predict. Detect. Explain. Protect.**

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)]()
[![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--learn-F7931E?style=flat-square)]()
[![Deep Learning](https://img.shields.io/badge/Deep%20Learning-TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)]()
[![XAI](https://img.shields.io/badge/XAI-SHAP-7C3AED?style=flat-square)]()
[![RAG](https://img.shields.io/badge/RAG-ChromaDB-5B21B6?style=flat-square)]()
[![LLM](https://img.shields.io/badge/LLM-Qwen3--8B-111827?style=flat-square)]()
[![Dashboard](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)]()

**A research Proof of Concept combining predictive maintenance, anomaly detection, explainable AI, RAG, and generative AI for industrial safety intelligence.**

</div>

---

## ⚡ ForgeShield in One View

**Predict → Detect → Explain → Assess Risk → Retrieve Evidence → Generate Safety Intelligence**

| **Capability** | **What it does** |
|---|---|
| **Predict** | Machine-failure prediction |
| **Detect** | Anomaly detection |
| **Explain** | SHAP-based interpretation |
| **Assess** | Unified machine-risk scoring |
| **Retrieve** | Evidence-grounded RAG |
| **Generate** | Incident intelligence + Safety Copilot |
| **Monitor** | Multi-machine telemetry simulation |

---

## 🧠 Architecture

```text
                    MACHINE DATA
                         │
              ┌──────────┼──────────┐
              ↓          ↓          ↓
          Supervised   Deep      Anomaly
              ML      Learning   Detection
              │          │          │
              ↓          ↓          ↓
         Failure Risk   RUL     Anomaly Score
              └──────────┼──────────┘
                         ↓
                  UNIFIED RISK
                         ↓
                    SHAP / XAI
                         ↓
                  RAG RETRIEVAL
                         ↓
                    QWEN3:8B
                    ↙       ↘
             Incident      Safety
           Intelligence    Copilot
                    ↓
             FORGESHIELD
            COMMAND CENTER
```

---

# 📊 Key Results

### Machine-Failure Prediction

**AI4I 2020 · untouched test set**

| **Model** | **Precision** | **Recall** | **F1** | **ROC-AUC** | **PR-AUC** | **FN** |
|---|---:|---:|---:|---:|---:|---:|
| **Gradient Boosting** | **0.902** | **0.809** | **0.853** | 0.974 | **0.897** | **13** |
| Random Forest | 0.927 | 0.750 | 0.829 | 0.976 | 0.844 | 17 |
| XGBoost | 0.885 | 0.794 | 0.837 | **0.980** | 0.864 | 14 |
| SVM | 0.573 | 0.691 | 0.627 | 0.970 | 0.655 | 21 |

> **Recall and false negatives are explicitly considered because missed failures are important in safety-oriented prediction.**

---

## 🎯 Unified Risk

```text
Risk =
0.60 × Failure Probability
+
0.40 × Anomaly Score
```

| **Score** | **Risk Band** |
|---:|:---|
| `< 0.25` | 🟢 **LOW** |
| `0.25 – < 0.50` | 🟡 **MEDIUM** |
| `0.50 – < 0.75` | 🟠 **HIGH** |
| `≥ 0.75` | 🔴 **CRITICAL** |

---

# 🔬 Research Components

### **Predictive Maintenance**
AI4I 2020 with multiple supervised ML models including Gradient Boosting, Random Forest, XGBoost and SVM.

### **Predictive Health**
NASA C-MAPSS FD001 for **RUL and temporal degradation modeling** using LSTM, GRU, CNN-LSTM and LSTM Autoencoder.

### **Anomaly Detection**
Isolation Forest, One-Class SVM, K-Means, DBSCAN, Agglomerative Clustering and PCA.

### **Explainable AI**
SHAP-based feature importance, summary analysis and dependence analysis.

### **RAG + Generative AI**
Retrieved safety evidence is supplied to a local LLM before generation.

### **Incident Intelligence**
Risk events are converted into structured evidence-grounded incident reports.

### **Safety Copilot**
Local **Qwen3:8B** assistant with evidence-aware grounding and abstention.

---

# 📚 RAG Knowledge Base

| **Evidence** | **Count** |
|---|---:|
| **Synthetic incidents** | **80** |
| **Safety procedures** | **4** |
| **Event types** | **8** |
| **Indexed chunks** | **158** |
| **Embedding dimensions** | **384** |

**Response structure**

`FACTS` · `POSSIBLE HYPOTHESES` · `RECOMMENDED ACTIONS` · `EVIDENCE SOURCES` · `LIMITATIONS`

> The current safety evidence is synthetic research evidence, not validated industrial operating procedure.

---

# 🖥️ Command Center

ForgeShield's Streamlit dashboard brings the research pipeline together:

**Command Center · Live Monitoring · Machine Intelligence · Predictive Health · Anomaly Detection · Explainability · Incident Intelligence · Safety Copilot**

---

# 📡 Live Monitoring

The current POC includes **multi-machine telemetry simulation**.

```text
Telemetry
   ↓
Feature Engineering
   ↓
Failure Prediction
   ↓
Anomaly Scoring
   ↓
Unified Risk
   ↓
Dashboard
```

> **The current implementation uses simulated telemetry. No physical machines or factory sensors are connected.**

---

# 🛠️ Technology

| **Area** | **Technology** |
|---|---|
| **Language** | Python |
| **ML** | Scikit-learn, XGBoost |
| **Deep Learning** | TensorFlow / Keras |
| **XAI** | SHAP |
| **RAG** | ChromaDB, Sentence Transformers |
| **LLM** | Ollama + Qwen3:8B |
| **Dashboard** | Streamlit |
| **Visualization** | Plotly, Matplotlib |
| **Data** | Pandas, NumPy |

---

# 🚀 Run

```bash
git clone https://github.com/aashi2709/ForgeShield.git
cd ForgeShield

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

streamlit run src/dashboard/app.py
```

---

# 🔬 Research Focus

ForgeShield is a **research Proof of Concept** focused on:

**Predictive ML · Deep Learning · Anomaly Detection · Explainable AI · Unified Risk · RAG · Generative AI · Safety Intelligence**

The project emphasizes leakage-safe experimentation, out-of-fold threshold selection, untouched test evaluation, explicit synthetic-data labeling, and explicit simulation labeling.

---

## ⚠️ Scope

ForgeShield is **not a production industrial safety system**.

It does not provide certified safety controls, autonomous machine control, or validated industrial operating procedures.

---

<div align="center">

### **🛡️ FORGESHIELD**

**Predict. Detect. Explain. Protect.**

</div>
