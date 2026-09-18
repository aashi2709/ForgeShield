# ForgeShield

### AI-Based Industrial Safety & Predictive Risk Management System

> **Predict. Detect. Explain. Protect.**

ForgeShield is a research-oriented Proof of Concept for intelligent industrial safety and predictive risk management. It combines machine-failure prediction, anomaly detection, predictive health monitoring, explainable AI, evidence-grounded Retrieval-Augmented Generation (RAG), and a local AI Safety Copilot into a unified safety intelligence workflow.

The project demonstrates how heterogeneous industrial data can be transformed into predictive risk signals and then converted into interpretable, evidence-grounded safety intelligence.

---

## System Overview

```text
                    INDUSTRIAL DATA
                         │
              ┌──────────┴──────────┐
              │                     │
        AI4I 2020 Dataset     NASA C-MAPSS
              │                     │
              ▼                     ▼
       Data Preprocessing     Time-Series Processing
              │                     │
              ▼                     ▼
      Supervised ML Models    Deep Learning Models
              │                     │
              ▼                     ▼
       Failure Probability       RUL Prediction
              │                     │
              └──────────┬──────────┘
                         │
                         ▼
                 Anomaly Detection
                         │
                         ▼
                  Unified Risk Score
                         │
             ┌───────────┴───────────┐
             │                       │
             ▼                       ▼
       Explainable AI          RAG Knowledge Base
          (SHAP)                     │
             │                       ▼
             │                 Evidence Retrieval
             │                       │
             └───────────┬───────────┘
                         ▼
               Incident Intelligence
                         │
                         ▼
                  Safety Copilot
                         │
                         ▼
                 FORGESHIELD COMMAND
                      CENTER


Research Contributions

ForgeShield focuses on the integration of multiple AI capabilities into a single industrial safety research pipeline:

Predictive machine-failure classification
Unsupervised anomaly detection
Remaining Useful Life (RUL) prediction
Deep-learning-based predictive health monitoring
Unified risk scoring
Explainable AI using SHAP
Evidence-grounded Retrieval-Augmented Generation
AI-generated incident intelligence
Safety-focused conversational assistance
Multi-machine live telemetry simulation
Evidence-aware abstention for unsupported queries

The central research idea is to move from isolated model predictions toward an integrated safety decision layer that combines predictive and anomalous behavior signals with evidence-grounded generative AI.

1. Machine-Failure Prediction
Dataset

The AI4I 2020 Predictive Maintenance Dataset is used for supervised machine-failure prediction.

The dataset contains:

10,000 observations
Machine operating conditions
Product type
Temperature measurements
Rotational speed
Torque
Tool wear
Machine-failure labels
Failure-mode indicators

The preprocessing pipeline includes:

Missing-value handling
IQR-based outlier clipping
Feature engineering
Numerical standardization
Categorical encoding
Stratified train/test splitting
Engineered Features

Examples include:

Temperature Differential
Mechanical Power
Tool Wear Rate
Models

The following supervised learning models were evaluated:

Logistic Regression
Decision Tree
Random Forest
K-Nearest Neighbors
Support Vector Machine
Naive Bayes
Gradient Boosting
XGBoost

Because missed failures are particularly important in safety-oriented applications, recall and false negatives are explicitly considered during evaluation.

Final Test Evaluation

The final evaluation was performed on an untouched test set using thresholds selected from out-of-fold training predictions.

Model	Precision	Recall	F1	ROC-AUC	PR-AUC	FN
Gradient Boosting	0.902	0.809	0.853	0.974	0.897	13
Random Forest	0.927	0.750	0.829	0.976	0.844	17
XGBoost	0.885	0.794	0.837	0.980	0.864	14
SVM	0.573	0.691	0.627	0.970	0.655	21

The results demonstrate the trade-offs between precision, recall, F1, ranking performance, and missed failures across different model families.

2. Anomaly Detection

ForgeShield also investigates unsupervised approaches for discovering unusual machine behavior.

Methods
Isolation Forest
One-Class SVM
K-Means
DBSCAN
Agglomerative Clustering
PCA

Isolation Forest is used as the continuous anomaly component of the integrated risk score.

Clustering methods are treated primarily as structural analysis and sanity checks rather than direct failure classifiers.

Key Observation

The clustering experiments produced ARI values close to zero, indicating weak alignment between unsupervised cluster assignments and the known failure labels.

This supports treating anomaly detection as a complementary signal rather than replacing supervised failure prediction.

3. Predictive Health & Remaining Useful Life

NASA C-MAPSS FD001 is used for predictive-health research.

The dataset contains multivariate engine degradation trajectories and is used to investigate Remaining Useful Life (RUL) prediction.

The predictive-health pipeline explores:

LSTM
GRU
CNN-LSTM
LSTM Autoencoder

The autoencoder component is additionally used to investigate reconstruction-error-based detection of abnormal temporal behavior.

C-MAPSS train/test separation is performed at the engine-unit level to reduce trajectory leakage.

4. Explainable AI

ForgeShield uses SHAP-based explainability to investigate why machine-failure models produce their predictions.

The explainability module provides:

Global feature importance
Model-specific SHAP importance
SHAP summary plots
Feature dependence analysis
Sample-level SHAP values

The global analysis highlights features such as:

Tool Wear
Rotational Speed
Mechanical Power
Torque

Explainability provides an interpretable layer between model predictions and downstream safety intelligence.

5. Unified Risk Scoring

ForgeShield combines supervised failure probability with the continuous anomaly score.

The current research formulation is:

Unified Risk Score =
    0.60 × Failure Probability
  + 0.40 × Anomaly Score

The resulting score is mapped into four risk bands:

Risk Score	Risk Band
< 0.25	Low
0.25 – < 0.50	Medium
0.50 – < 0.75	High
≥ 0.75	Critical

This creates a unified decision layer from two complementary sources of evidence:

Predictive Failure Risk
          +
Behavioral Anomaly Risk
          ↓
     Unified Risk
6. Incident Intelligence

ForgeShield extends predictive risk analysis into evidence-grounded incident interpretation.

The incident-intelligence pipeline combines:

Machine Context
      ↓
Risk Assessment
      ↓
Relevant Historical Evidence
      ↓
Safety Procedure Retrieval
      ↓
Local LLM
      ↓
Structured Incident Report

Generated incident reports contain sections such as:

Incident Summary
Timeline of Events
Abnormal Sensor Behavior
Possible Contributing Factors
Potential Root Causes
Evidence
Corrective Actions
Preventive Actions

Generated evidence identifiers are validated against retrieved evidence to reduce unsupported citations.

7. Retrieval-Augmented Generation

The RAG system provides the evidence layer for ForgeShield's generative-AI components.

The retrieval pipeline is:

Safety Documents
      ↓
Document Chunking
      ↓
Embeddings
      ↓
Vector Retrieval
      ↓
Metadata Filtering
      ↓
Relevant Evidence
      ↓
LLM

The current Proof of Concept knowledge base contains:

80 synthetic incident records
4 synthetic safety procedures
8 incident/event types
158 indexed chunks
384-dimensional embeddings

The synthetic evidence is explicitly treated as research evidence rather than validated industrial operating procedure.

8. Safety Copilot

The ForgeShield Safety Copilot provides a conversational interface over the retrieved safety evidence.

The current implementation uses:

Local LLM: Qwen3:8B
Runtime: Ollama

Responses are structured into:

FACTS
POSSIBLE HYPOTHESES
RECOMMENDED ACTIONS
EVIDENCE SOURCES
LIMITATIONS

The grounding strategy requires:

Facts to be supported by retrieved evidence
Hypotheses to be explicitly labelled
Recommended actions to be grounded in retrieved procedures/evidence
Unsupported claims to be excluded
Unsupported operational instructions to be avoided
Insufficient evidence to trigger abstention

This allows the system to demonstrate evidence-aware behavior instead of treating the LLM as an unrestricted source of safety instructions.

9. Live Monitoring

ForgeShield includes a multi-machine live-monitoring simulation.

Simulated Machine Telemetry
            ↓
Feature Engineering
            ↓
Training-Time Preprocessing
            ↓
Failure Prediction
            ↓
Anomaly Scoring
            ↓
Unified Risk
            ↓
Dashboard

The current implementation simulates telemetry from multiple machines.

Important

This is simulated telemetry, not data from physically connected industrial machines.

The purpose is to demonstrate the real-time inference and visualization workflow using the already-trained research models.

A future physical deployment could extend the ingestion layer to industrial sensors, PLCs, gateways, or messaging infrastructure.

10. ForgeShield Command Center

The dashboard provides a unified interface for the complete research pipeline.

Dashboard Modules
Command Center
Live Monitoring
Machine Intelligence
Predictive Health
Anomaly Detection
Explainability
Incident Intelligence
Safety Copilot
Knowledge Base
Documentation
Settings

The Command Center connects the individual research components into a single visual workflow.

11. End-to-End Pipeline

The complete ForgeShield workflow is:

Industrial / Benchmark Data
          ↓
    Data Engineering
          ↓
      Feature Engineering
          ↓
   ┌──────┴────────┐
   │               │
Supervised ML   Deep Learning
   │               │
Failure Prob.   RUL / Health
   │               │
   └──────┬────────┘
          ↓
   Anomaly Detection
          ↓
   Unified Risk Score
          ↓
      SHAP / XAI
          ↓
    Incident Context
          ↓
     RAG Retrieval
          ↓
    Evidence-Grounded
          LLM
          ↓
   Incident Intelligence
          ↓
     Safety Copilot
          ↓
   ForgeShield Command
         Center
12. Technology Stack
Programming
Python
Machine Learning
Scikit-learn
XGBoost
Deep Learning
TensorFlow / Keras
Explainable AI
SHAP
Generative AI
Ollama
Qwen3:8B
Retrieval
ChromaDB
Sentence-transformer embeddings
Dashboard
Streamlit
Plotly
Experimentation
Pandas
NumPy
Matplotlib
Joblib
13. Repository Structure
ForgeShield/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── knowledge_base/
│   ├── docs/
│   └── chroma_db/
│
├── models/
│
├── reports/
│
├── src/
│   ├── dashboard/
│   │   ├── components/
│   │   └── views/
│   │
│   ├── live_monitoring/
│   └── rag/
│
├── docs/
│   ├── architecture.md
│   ├── methodology.md
│   ├── datasets.md
│   └── reproducibility.md
│
├── .env.example
├── .gitignore
├── LICENSE
├── requirements.txt
└── README.md
14. Installation

Clone the repository:

git clone https://github.com/aashi2709/ForgeShield.git
cd ForgeShield

Create a virtual environment:

python -m venv .venv
source .venv/bin/activate

Install dependencies:

pip install -r requirements.txt
15. Running the Dashboard

Launch ForgeShield with:

streamlit run src/dashboard/app.py

The dashboard provides access to the research modules and the simulated live-monitoring environment.

16. Running the Live Monitoring Simulation

The live inference pipeline can be tested independently:

python src/live_monitoring/inference.py

This generates simulated machine telemetry and passes it through the trained ForgeShield inference pipeline.

17. Research Methodology

The project emphasizes leakage-safe experimentation.

Key principles include:

Training-derived preprocessing
Training-derived outlier boundaries
Stratified classification splitting
Engine-level separation for C-MAPSS
Out-of-fold threshold selection
Untouched final test evaluation
Explicit distinction between real benchmark data and synthetic evidence
Explicit distinction between simulated telemetry and physical machine data

Detailed methodology is documented in:

docs/methodology.md
18. Reproducibility

The project includes a dedicated reproducibility guide:

docs/reproducibility.md

It documents:

Environment setup
Dataset preparation
Model training
Cross-validation
Threshold selection
Final evaluation
Anomaly detection
Predictive-health experiments
Explainability
RAG
Dashboard execution
19. Project Scope

ForgeShield is a research Proof of Concept, not a production industrial safety system.

The current implementation demonstrates the research pipeline using:

Benchmark datasets
Synthetic safety evidence
Simulated live telemetry
Research-trained models
Local generative AI

It does not provide:

Certified industrial safety controls
Direct machine control
Safety-critical autonomous actuation
Validated industrial operating procedures
Physical sensor deployment
Industrial safety certification
20. Limitations

Current limitations include:

Benchmark datasets may not represent every real industrial environment.
Synthetic safety incidents are not equivalent to validated industrial incident records.
Simulated live telemetry does not represent physical sensor noise or communication failures.
The RAG evaluation is a Proof-of-Concept grounding evaluation.
Local LLM outputs remain dependent on model behavior and retrieved evidence.
Deep-learning performance depends on dataset characteristics and training configuration.
Unified risk weights are research parameters rather than universally validated industrial risk coefficients.
21. Future Work

Potential extensions include:

Physical sensor and PLC integration
MQTT/Kafka-based telemetry ingestion
Online model monitoring
Streaming anomaly detection
Adaptive risk calibration
Larger industrial datasets
Human-in-the-loop safety validation
More rigorous RAG evaluation
Domain-validated safety procedures
Edge deployment
Industrial cybersecurity integration
Multimodal industrial safety intelligence
22. Research Paper

ForgeShield is being developed as a research-oriented Proof of Concept with the following intended paper structure:

Abstract
Introduction
Literature Review
Problem Statement
Dataset Description
Methodology
Machine Learning Models
Deep Learning Models
Experimental Setup
Results
Model Comparison
Explainable AI
Generative AI / RAG Architecture
Discussion
Limitations
Future Work
Conclusion
References
23. Project Status

Current status: Research Proof of Concept

Implemented research components include:

 Data preprocessing
 Supervised machine-failure prediction
 Cross-validation
 Leakage-safe threshold selection
 Final test evaluation
 Anomaly detection
 Predictive-health experiments
 Explainable AI
 Unified risk scoring
 RAG knowledge base
 Incident intelligence
 Safety Copilot
 Multi-machine live telemetry simulation
 ForgeShield Command Center dashboard
 Research documentation
License

This project is licensed under the MIT License.

See LICENSE for details.

                      
