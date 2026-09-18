<div align="center">

# 🛡️ ForgeShield

### AI-Based Industrial Safety & Predictive Risk Management System

**Predict. Detect. Explain. Protect.**

![Status](https://img.shields.io/badge/status-research%20POC-orange)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-lightgrey)
![LLM](https://img.shields.io/badge/LLM-Qwen3%3A8B-purple)

ForgeShield is a research Proof of Concept that combines **predictive maintenance**, **anomaly detection**, **predictive health**, **explainable AI**, **Retrieval-Augmented Generation**, and a **local Safety Copilot** into one industrial safety intelligence workflow.

</div>

<br>


## 🧩 Overview

Industrial safety systems often produce isolated signals — a machine-failure probability here, an anomaly score there, a maintenance estimate somewhere else, a text-based recommendation with no evidence behind it.

**ForgeShield connects these pieces.**

The system takes machine data, produces predictive and behavioral risk signals, explains the model output, retrieves relevant safety evidence, and passes that grounded context to a local language model for incident intelligence and safety assistance.

> **Core idea**
> `Machine Data → Prediction → Detection → Unified Risk → Explanation → Evidence → Safety Intelligence`

> [!NOTE]
> ForgeShield is intentionally developed as a **research-oriented Proof of Concept**, not a production industrial control or certified safety system.

---

## 🏗️ System Architecture

```
                         ┌──────────────────────┐
                         │     MACHINE DATA      │
                         └───────────┬───────────┘
                                     │
                            Data Preprocessing
                                     │
              ┌──────────────────────┼──────────────────────┐
              │                      │                      │
              ▼                      ▼                      ▼
      ┌───────────────┐     ┌────────────────┐     ┌────────────────┐
      │ Supervised ML  │     │ Deep Learning  │     │ Anomaly        │
      │                │     │                │     │ Detection      │
      └───────┬────────┘     └───────┬────────┘     └───────┬────────┘
              │                      │                      │
              ▼                      ▼                      ▼
       Failure Probability     RUL / Health           Anomaly Score
              │                      │                      │
              └──────────────────────┴──────────┬───────────┘
                                                  ▼
                                    ┌──────────────────────┐
                                    │   UNIFIED RISK SCORE  │
                                    └───────────┬───────────┘
                                                 │
                          ┌──────────────────────┼──────────────────────┐
                          ▼                      ▼                      ▼
                    ┌──────────┐        ┌────────────────┐     ┌────────────────┐
                    │ SHAP XAI │        │ Incident       │     │ RAG            │
                    │          │        │ Context        │     │ Retrieval      │
                    └──────────┘        └───────┬────────┘     └───────┬────────┘
                                                 │                      │
                                                 └──────────┬───────────┘
                                                             ▼
                                                   ┌──────────────────┐
                                                   │   Qwen3:8B LLM    │
                                                   └─────────┬─────────┘
                                                              │
                                          ┌───────────────────┴───────────────────┐
                                          ▼                                       ▼
                              Incident Intelligence                    Safety Copilot
                                          │                                       │
                                          └───────────────────┬───────────────────┘
                                                               ▼
                                             ┌──────────────────────────────┐
                                             │   ForgeShield Command Center  │
                                             └──────────────────────────────┘
```

---

## 🔬 Research Contributions

ForgeShield focuses on the **integration and decision layer** connecting traditional machine learning with evidence-grounded generative AI.

| # | Contribution | Description |
|---|---|---|
| 1 | **Predictive Failure Modeling** | Multiple supervised models evaluated for machine-failure prediction |
| 2 | **Unsupervised Risk Discovery** | Anomaly detectors and clustering methods provide a complementary view of abnormal behavior |
| 3 | **Predictive Health** | NASA C-MAPSS trajectories used for RUL and temporal degradation research |
| 4 | **Explainable Risk** | SHAP connects model predictions to interpretable feature-level evidence |
| 5 | **Unified Risk Layer** | Failure probability and anomaly score fused into one operationally interpretable score |
| 6 | **Evidence-Grounded Generative AI** | RAG retrieves relevant safety evidence before it reaches the local LLM |
| 7 | **Fact / Hypothesis Separation** | Generated incident intelligence separates evidence-supported facts from AI hypotheses |
| 8 | **Evidence-Aware Abstention** | The Safety Copilot abstains when retrieved evidence is insufficient or unrelated |

---

## 📊 Key Results

### Machine-Failure Prediction

**Dataset:** AI4I 2020 Predictive Maintenance Dataset
**Evaluation:** Untouched test set
**Thresholds:** Selected from out-of-fold training predictions

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC | False Negatives |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Gradient Boosting** | 0.902 | 0.809 | **0.853** | 0.974 | 0.897 | 13 |
| Random Forest | 0.927 | 0.750 | 0.829 | 0.976 | 0.844 | 17 |
| XGBoost | 0.885 | 0.794 | 0.837 | **0.980** | 0.864 | 14 |
| SVM | 0.573 | 0.691 | 0.627 | 0.970 | 0.655 | 21 |

> [!IMPORTANT]
> Safety-oriented evaluation gives explicit attention to **recall and false negatives**, because missed failures are particularly costly in this application context.

---

## 🎯 Unified Risk Scoring

ForgeShield combines two complementary signals into a single operational score:

```
Failure Probability  +  Anomaly Score  →  Unified Risk Score
```

**Current formulation:**

$$\text{Risk} = 0.60 \times \text{Failure Probability} + 0.40 \times \text{Anomaly Score}$$

### Risk Bands

| Risk Score | Band |
|---|---|
| `< 0.25` | 🟢 **LOW** |
| `0.25 – < 0.50` | 🟡 **MEDIUM** |
| `0.50 – < 0.75` | 🟠 **HIGH** |
| `≥ 0.75` | 🔴 **CRITICAL** |

This unified score is the hand-off point between predictive analytics, explainability, incident intelligence, and the dashboard.

---

## 🤖 Machine-Failure Prediction

The **AI4I dataset** powers the supervised prediction problem.

**Input signals**
- Air temperature · Process temperature
- Rotational speed · Torque · Tool wear
- Product type · Failure-mode indicators

**Preprocessing pipeline**
1. Missing-value handling
2. IQR-based outlier clipping
3. Feature engineering
4. Numerical standardization
5. Categorical encoding
6. Stratified train/test splitting

**Engineered features:** Temperature Differential · Mechanical Power · Tool Wear Rate

**Evaluated models:** Logistic Regression · Decision Tree · Random Forest · KNN · SVM · Naive Bayes · Gradient Boosting · XGBoost

---

## 🔎 Anomaly Detection

Industrial environments often have limited labeled failure examples, so ForgeShield adds an unsupervised layer.

| Method | Purpose |
|---|---|
| **Isolation Forest** | Continuous anomaly scoring |
| One-Class SVM | Novelty / anomaly detection |
| K-Means | Cluster structure analysis |
| DBSCAN | Density-based structure |
| Agglomerative Clustering | Hierarchical structure |
| PCA | Low-dimensional visualization |

- Isolation Forest supplies the **continuous anomaly signal** used by the unified risk layer.
- Clustering is treated as **structural analysis and sanity checking**, not a direct replacement for supervised failure prediction.

---

## ⏱️ Predictive Health & RUL

ForgeShield uses **NASA C-MAPSS FD001** for time-series predictive-health research, investigating how degradation trajectories can be modeled for Remaining Useful Life (RUL) estimation.

**Deep-learning models:** LSTM · GRU · CNN-LSTM · LSTM Autoencoder

- The temporal pipeline operates on engine trajectories.
- Train/test separation is performed **at the engine-unit level** to reduce trajectory leakage.
- The LSTM Autoencoder additionally supports **reconstruction-error analysis** for abnormal temporal behavior.

---

## 🧠 Explainable AI

A high-risk prediction is more useful when the system can show *why* the model produced it. ForgeShield uses **SHAP** (SHapley Additive exPlanations) for model interpretation.

**Explainability outputs:** global feature importance · model-specific SHAP importance · summary plots · dependence plots · sample-level SHAP values

| Feature | Role in the Analysis |
|---|---|
| **Tool Wear** | Strong contributor across explainability results |
| Rotational Speed | Important operating-condition signal |
| Mechanical Power | Derived mechanical behavior signal |
| Torque | Load-related signal |
| Temperature Differential | Engineered thermal signal |

The XAI layer provides an interpretable bridge between model output and downstream safety intelligence.

---

## 📚 RAG Knowledge Base

ForgeShield uses Retrieval-Augmented Generation to ground generative AI in retrieved safety evidence.

```
Safety Evidence → Document Chunking → Embeddings → Semantic Retrieval
                → Metadata Filtering → Relevant Evidence → Local LLM
```

### Current POC Knowledge Base

| Component | Count |
|---|:---:|
| Synthetic incident records | 80 |
| Safety procedures | 4 |
| Event types | 8 |
| Indexed chunks | 158 |
| Embedding dimensions | 384 |

The knowledge base distinguishes between **incident evidence** and **safety procedures**, so retrieved context can be used more deliberately downstream.

> [!WARNING]
> Current incident and procedure records are **synthetic research evidence**. They are **not** validated industrial operating procedures.

---

## 📝 Incident Intelligence

ForgeShield turns a flagged machine event into a structured incident report.

```
Machine Context → Risk Assessment → Retrieved Evidence
                → Safety Procedure → Qwen3:8B → Structured Incident Report
```

**Report structure:**
1. Incident Summary
2. Timeline of Events
3. Abnormal Sensor Behavior
4. Possible Contributing Factors
5. Potential Root Causes
6. Evidence
7. Corrective Actions
8. Preventive Actions

A central design principle is the separation between **evidence-supported facts** and **AI-generated hypotheses**. Generated citations are validated against the retrieved evidence identifiers.

---

## 🛡️ Safety Copilot

The Safety Copilot provides a conversational interface over the retrieved safety knowledge, powered locally by **Ollama + Qwen3:8B**.

Responses are organized into:

`FACTS` → `POSSIBLE HYPOTHESES` → `RECOMMENDED ACTIONS` → `EVIDENCE SOURCES` → `LIMITATIONS`

**Grounding rules** — the assistant is designed to:

- ✅ Use retrieved evidence as the basis for factual claims
- ✅ Clearly label hypotheses
- ✅ Ground recommended actions in available procedures/evidence
- ✅ Expose evidence identifiers used by the response
- ✅ State when evidence is insufficient
- 🚫 Avoid invented operational instructions
- 🚫 Abstain from unsupported or unrelated requests

This makes the Copilot a **grounded research assistant**, not an unrestricted chatbot.

---

## 📡 Live Monitoring

ForgeShield includes a multi-machine live telemetry simulation:

```
Simulated Telemetry → Feature Engineering → Training-Time Preprocessing
                     → Failure Prediction → Anomaly Scoring → Unified Risk → Dashboard
```

The simulator generates telemetry for multiple machines and passes it through the **same trained inference pipeline** used by the risk dashboard.

> [!NOTE]
> This is simulated telemetry. The current POC is not connected to physical machines, PLCs, or factory sensors.

---

## 🖥️ ForgeShield Command Center

The Streamlit dashboard brings all research components into a single interface.

| Module | Purpose |
|---|---|
| **Command Center** | Overall machine-risk overview |
| **Live Monitoring** | Multi-machine telemetry simulation |
| **Machine Intelligence** | Supervised failure prediction |
| **Predictive Health** | RUL and temporal health analysis |
| **Anomaly Detection** | Unsupervised risk discovery |
| **Explainability** | SHAP model interpretation |
| **Incident Intelligence** | Evidence-grounded incident reports |
| **Safety Copilot** | RAG-powered safety assistance |
| **Knowledge Base** | Retrieval evidence overview |

---

## 🎥 Demo

*Demo screenshots and the voice-over walkthrough video will be added here.*

**Planned demonstration flow:**

`Command Center` → `Machine Intelligence` → `Anomaly Detection` → `Predictive Health` → `Explainability` → `Incident Intelligence` → `Safety Copilot` → `Live Monitoring`

<details>
<summary><strong>Demo highlights (click to expand)</strong></summary>

<br>

**1. High-Risk Machine**
Show a machine with its failure probability, anomaly score, unified risk, risk band, and actual failure label.

**2. Explainability**
Show the SHAP feature contributions behind the prediction.

**3. Evidence-Grounded Incident Analysis**
Show the generated report alongside its retrieved evidence.

**4. Safety Copilot**
Demonstrate a supported safety query.

**5. Abstention**
Demonstrate an unrelated or unsupported safety query and show that the system does not fabricate unsupported instructions.

**6. Live Monitoring**
Show multiple simulated machines updating through the inference pipeline in real time.

</details>

---

## 🧪 Research Methodology

ForgeShield emphasizes experimental controls designed to reduce leakage and keep the research workflow reproducible.

<table>
<tr><td width="25%" valign="top"><strong>Classification</strong></td><td>

- Stratified train/test split
- Class weighting for imbalance
- Training-derived preprocessing
- Training-derived outlier bounds
- Out-of-fold threshold selection
- Untouched final test evaluation

</td></tr>
<tr><td valign="top"><strong>Time-Series</strong></td><td>

- Engine-level separation for C-MAPSS
- Sequence-based temporal modeling
- Training/evaluation separation by engine unit

</td></tr>
<tr><td valign="top"><strong>Anomaly Detection</strong></td><td>

- Detector fitting on training data where supported
- Train-derived PCA and clustering structure
- Continuous anomaly scoring for integrated risk

</td></tr>
<tr><td valign="top"><strong>Generative AI</strong></td><td>

- Metadata-aware retrieval
- Evidence-grounded prompting
- Explicit fact/hypothesis separation
- Citation validation
- Abstention when relevant evidence is insufficient

</td></tr>
</table>

---

## 🧰 Technology Stack

| Layer | Technology |
|---|---|
| **Language** | Python |
| **Data Processing** | Pandas, NumPy |
| **Machine Learning** | Scikit-learn, XGBoost |
| **Deep Learning** | TensorFlow / Keras |
| **Explainability** | SHAP |
| **Embeddings** | Sentence Transformers |
| **Vector Store** | ChromaDB |
| **Local LLM** | Ollama + Qwen3:8B |
| **Dashboard** | Streamlit |
| **Visualization** | Plotly, Matplotlib |
| **Model Persistence** | Joblib |

---

## 🚀 Quick Start

```bash
git clone https://github.com/aashi2709/ForgeShield.git
cd ForgeShield

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

streamlit run src/dashboard/app.py
```

**Test live inference**

```bash
python src/live_monitoring/inference.py
```

---

## 📖 Documentation

Detailed research documentation is maintained separately so this README stays focused:

- 🏗️ Architecture
- 🧪 Methodology
- 📊 Datasets
- 🔁 Reproducibility

---

## 📌 Research Scope

ForgeShield is an **academic research Proof of Concept** demonstrating an integrated workflow across:

`Machine Learning → Deep Learning → Anomaly Detection → Risk Scoring → Explainability → RAG → Generative AI → Safety Intelligence`

**It is not intended to:**

- ❌ Directly control industrial machinery
- ❌ Replace certified safety systems
- ❌ Provide guaranteed failure predictions
- ❌ Replace qualified safety personnel
- ❌ Serve as a production industrial control system
- ❌ Provide validated emergency procedures

---

## ⚠️ Limitations

| Area | Current Boundary |
|---|---|
| **Datasets** | AI4I and C-MAPSS are benchmark datasets rather than one unified factory stream |
| **Safety Evidence** | Incident/procedure records are synthetic |
| **Monitoring** | Live telemetry is simulated |
| **RAG Evaluation** | Proof-of-Concept grounding evaluation |
| **LLM** | Output depends on model behavior and retrieved evidence |
| **Risk Fusion** | 0.60 / 0.40 weights are research parameters |
| **Deployment** | Physical deployment would require additional safety and engineering validation |

---

## 🔮 Future Work

Physical sensor integration · PLC connectivity · Streaming telemetry · Online anomaly detection · Adaptive risk calibration · Larger industrial datasets · Domain-validated safety procedures · Human-in-the-loop validation · Edge deployment · Industrial cybersecurity

---

## 📄 Research Paper

ForgeShield is being developed as a **research study**, not a code-only project. The planned paper covers:

> Introduction · Literature Review · Problem Statement · Dataset Description · Methodology · ML Models · DL Models · Experimental Setup · Results · Model Comparison · Explainable AI · RAG Architecture · Discussion · Limitations · Future Work · Conclusion · References

The paper will be supported by the quantitative results, figures, tables, and experiments generated throughout the project.

<br>

<div align="center">

### 🛡️ FORGESHIELD

**Predict. Detect. Explain. Protect.**

*Research Proof of Concept for AI-driven industrial safety intelligence*

</div>
