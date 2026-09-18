ForgeShield

<p align="center">
  <strong>AI-Based Industrial Safety & Predictive Risk Management System</strong>
</p>

<p align="center">
  <em>Predict. Detect. Explain. Protect.</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Project-Research%20POC-111827?style=flat-square">
  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/ML-Scikit--learn-F7931E?style=flat-square">
  <img src="https://img.shields.io/badge/XAI-SHAP-8A2BE2?style=flat-square">
  <img src="https://img.shields.io/badge/RAG-ChromaDB-6B7280?style=flat-square">
  <img src="https://img.shields.io/badge/LLM-Qwen3--8B-111827?style=flat-square">
  <img src="https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white">
</p>

Overview

ForgeShield is a research-oriented Proof of Concept for intelligent industrial safety and predictive risk management.

The system combines:

machine-failure prediction + anomaly detection + predictive health monitoring + explainable AI + RAG + generative AI

into one safety-intelligence workflow.

Core idea: transform machine data into predictive risk signals, explain why the system produced those signals, retrieve relevant safety evidence, and use a grounded local LLM to generate structured safety intelligence.

What ForgeShield Does

Capability

Purpose

Machine Failure Prediction

Predict the probability of machine failure

Anomaly Detection

Detect unusual machine behavior

Predictive Health

Estimate Remaining Useful Life (RUL) and monitor degradation

Explainable AI

Explain model predictions using SHAP

Unified Risk

Combine failure probability and anomaly score

Incident Intelligence

Generate evidence-grounded incident reports

RAG

Retrieve relevant safety evidence

Safety Copilot

Answer safety queries using retrieved evidence

Live Monitoring

Demonstrate multi-machine real-time inference using simulated telemetry

Command Center

Bring the complete pipeline together in one dashboard

Architecture

flowchart TD
    A[Industrial / Benchmark Data] --> B[Data Preprocessing]

    B --> C[Supervised ML]
    B --> D[Deep Learning]
    B --> E[Anomaly Detection]

    C --> F[Failure Probability]
    D --> G[RUL / Predictive Health]
    E --> H[Anomaly Score]

    F --> I[Unified Risk Score]
    H --> I

    I --> J[Explainable AI - SHAP]
    I --> K[Incident Context]

    K --> L[RAG Retrieval]
    L --> M[Retrieved Safety Evidence]
    M --> N[Local LLM - Qwen3:8B]

    N --> O[Incident Intelligence]
    N --> P[Safety Copilot]

    F --> Q[ForgeShield Command Center]
    H --> Q
    I --> Q
    J --> Q
    O --> Q
    P --> Q

Research Contributions

ForgeShield focuses on integrating multiple AI capabilities rather than treating each model as an isolated component.

1. Predictive failure modeling

Multiple supervised learning models are evaluated for machine-failure prediction.

2. Complementary anomaly detection

Unsupervised methods provide an additional behavioral-risk signal instead of replacing supervised prediction.

3. Predictive health monitoring

NASA C-MAPSS trajectories are used to investigate RUL prediction and temporal degradation modeling.

4. Explainable predictions

SHAP is used to expose the features contributing to model predictions.

5. Unified risk representation

Failure probability and anomaly score are fused into one interpretable risk score.

6. Evidence-grounded generative AI

RAG connects machine context with retrieved safety evidence before information is passed to the local LLM.

7. Safety-aware abstention

The Safety Copilot is designed to avoid inventing unsupported safety instructions when relevant evidence is unavailable.

1. Machine-Failure Prediction

Dataset

The AI4I 2020 Predictive Maintenance Dataset is used for supervised machine-failure prediction.

The dataset contains:

10,000 observations

Machine operating conditions

Product type

Air temperature

Process temperature

Rotational speed

Torque

Tool wear

Machine-failure labels

Failure-mode indicators

Preprocessing

The pipeline includes:

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

Models Evaluated

Logistic Regression

Decision Tree

Random Forest

K-Nearest Neighbors

Support Vector Machine

Naive Bayes

Gradient Boosting

XGBoost

Because missed failures are important in safety-oriented applications, evaluation considers:

Precision

Recall

F1

ROC-AUC

PR-AUC

Confusion matrix

False negatives

Final Test Results

The final evaluation uses an untouched test set. Operating thresholds were selected using out-of-fold training predictions.

Model

Precision

Recall

F1

ROC-AUC

PR-AUC

FN

Gradient Boosting

0.902

0.809

0.853

0.974

0.897

13

Random Forest

0.927

0.750

0.829

0.976

0.844

17

XGBoost

0.885

0.794

0.837

0.980

0.864

14

SVM

0.573

0.691

0.627

0.970

0.655

21

These results illustrate the trade-offs between precision, recall, overall classification performance, ranking metrics, and missed failures.

2. Anomaly Detection

ForgeShield investigates both point-anomaly detection and unsupervised structure discovery.

Methods

Method

Role

Isolation Forest

Continuous anomaly scoring

One-Class SVM

Novelty / anomaly detection

K-Means

Cluster structure

DBSCAN

Density-based structure

Agglomerative Clustering

Hierarchical structure

PCA

Low-dimensional visualization

Isolation Forest provides the continuous anomaly component used by the integrated risk layer.

Research Observation

The clustering experiments produced ARI values close to zero, indicating weak alignment between unsupervised cluster assignments and known failure labels.

Therefore, clustering is treated as structural analysis rather than a direct machine-failure classifier.

3. Predictive Health & RUL

NASA C-MAPSS FD001 is used for predictive-health research.

The dataset provides multivariate engine degradation trajectories for Remaining Useful Life prediction.

Models

LSTM

GRU

CNN-LSTM

LSTM Autoencoder

The autoencoder additionally supports reconstruction-error-based analysis of abnormal temporal behavior.

Leakage Prevention

C-MAPSS trajectories are separated by engine unit rather than randomly splitting individual time windows. This reduces the risk of trajectory leakage between training and evaluation.

4. Explainable AI

ForgeShield uses SHAP to investigate why supervised models produce their predictions.

The explainability module provides:

Global feature importance

Model-specific SHAP importance

SHAP summary plots

Feature dependence analysis

Sample-level SHAP values

Important features observed across the explainability analysis include:

Tool Wear

Rotational Speed

Mechanical Power

Torque

The XAI layer connects predictive output with interpretable machine behavior.

5. Unified Risk Scoring

The central decision layer combines two signals:

Failure Probability
        +
Anomaly Score
        ↓
Unified Risk Score

The current research formulation is:

$$
R = 0.60P_f + 0.40A
$$

where:

$P_f$ = predicted failure probability

$A$ = normalized anomaly score

Risk Bands

Risk Score

Band

< 0.25

Low

0.25 – < 0.50

Medium

0.50 – < 0.75

High

≥ 0.75

Critical

This creates one interpretable risk representation from supervised and unsupervised evidence.

6. Incident Intelligence

ForgeShield extends risk prediction into structured incident interpretation.

flowchart LR
    A[Machine Context] --> B[Risk Assessment]
    B --> C[Relevant Evidence]
    C --> D[Safety Procedure]
    D --> E[Local LLM]
    E --> F[Structured Incident Report]

Generated reports contain sections such as:

Incident Summary

Timeline of Events

Abnormal Sensor Behavior

Possible Contributing Factors

Potential Root Causes

Evidence

Corrective Actions

Preventive Actions

Evidence identifiers are validated against retrieved evidence to reduce unsupported citations.

7. Retrieval-Augmented Generation

The RAG system provides the evidence layer for ForgeShield's generative-AI components.

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
Local LLM

Current POC Knowledge Base

Component

Quantity

Synthetic incident records

80

Synthetic safety procedures

4

Event types

8

Indexed chunks

158

Embedding dimension

384

The current safety evidence is synthetic research evidence and is not validated industrial operating procedure.

8. Safety Copilot

The Safety Copilot provides a conversational interface over retrieved safety evidence.

Current LLM

Qwen3:8B
Runtime: Ollama

Responses are structured into:

FACTS
POSSIBLE HYPOTHESES
RECOMMENDED ACTIONS
EVIDENCE SOURCES
LIMITATIONS

Grounding Principles

The assistant is designed so that:

Facts must be supported by retrieved evidence.

Hypotheses must be explicitly labelled.

Recommended actions must be grounded in retrieved procedures/evidence.

Unsupported claims should not be presented as facts.

Unsupported operational instructions should not be invented.

Insufficient evidence should result in abstention.

This is intended to demonstrate evidence-aware generative AI rather than unrestricted LLM output.

9. Live Monitoring

ForgeShield includes a multi-machine live-monitoring simulation.

flowchart LR
    A[Simulated Machine Telemetry] --> B[Feature Engineering]
    B --> C[Training-Time Preprocessing]
    C --> D[Failure Prediction]
    C --> E[Anomaly Scoring]
    D --> F[Unified Risk]
    E --> F
    F --> G[Live Dashboard]

Important

The current live-monitoring layer uses simulated telemetry.

It is not connected to physical industrial machines, PLCs, or factory sensors.

The purpose is to demonstrate the real-time inference workflow using the trained ForgeShield models.

A future deployment could replace the simulator with an appropriate industrial data-ingestion layer.

10. ForgeShield Command Center

The Streamlit dashboard provides a unified interface for the research system.

Dashboard Modules

Module

Function

Command Center

Overall system view

Live Monitoring

Multi-machine telemetry simulation

Machine Intelligence

Failure prediction analysis

Predictive Health

RUL and temporal health analysis

Anomaly Detection

Unsupervised risk discovery

Explainability

SHAP-based model interpretation

Incident Intelligence

Evidence-grounded incident reports

Safety Copilot

RAG-powered safety interaction

Knowledge Base

Evidence and retrieval overview

Documentation

Research methodology and scope

Settings

Local model and system configuration

11. End-to-End Research Workflow

flowchart TD
    A[Benchmark / Synthetic Data] --> B[Data Engineering]
    B --> C[Feature Engineering]

    C --> D[Supervised ML]
    C --> E[Deep Learning]
    C --> F[Anomaly Detection]

    D --> G[Failure Probability]
    E --> H[RUL / Health Signal]
    F --> I[Anomaly Score]

    G --> J[Unified Risk]
    I --> J

    J --> K[SHAP Explainability]
    J --> L[Incident Context]

    L --> M[RAG]
    M --> N[Evidence]
    N --> O[Generative AI]

    O --> P[Incident Intelligence]
    O --> Q[Safety Copilot]

    J --> R[ForgeShield Command Center]
    K --> R
    P --> R
    Q --> R

12. Technology Stack

Area

Technology

Language

Python

Data Processing

Pandas, NumPy

Machine Learning

Scikit-learn, XGBoost

Deep Learning

TensorFlow / Keras

Explainability

SHAP

Embeddings

Sentence Transformers

Vector Store

ChromaDB

Local LLM

Ollama + Qwen3:8B

Dashboard

Streamlit

Visualization

Plotly, Matplotlib

Model Persistence

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

14. Quick Start

Clone

git clone https://github.com/aashi2709/ForgeShield.git
cd ForgeShield

Create Environment

python -m venv .venv
source .venv/bin/activate

Install Dependencies

pip install -r requirements.txt

Run Dashboard

streamlit run src/dashboard/app.py

Test Live Inference

python src/live_monitoring/inference.py

15. Research Methodology

ForgeShield emphasizes leakage-safe experimentation.

Key principles include:

Training-derived preprocessing

Training-derived outlier boundaries

Stratified classification splitting

Engine-level C-MAPSS separation

Out-of-fold threshold selection

Untouched final test evaluation

Explicit separation of real benchmark data and synthetic evidence

Explicit separation of simulated telemetry and physical machine data

<details>
<summary><strong>Detailed methodology</strong></summary>

Data Preparation

The AI4I pipeline performs feature engineering, missing-value handling, IQR clipping, standardization, categorical encoding, and stratified splitting.

Supervised Learning

Multiple model families are evaluated using classification metrics with particular attention to recall and false negatives.

Threshold Selection

Operating thresholds are selected from out-of-fold training predictions rather than the final test set.

Anomaly Detection

Isolation Forest and One-Class SVM are evaluated alongside clustering-based methods.

Predictive Health

C-MAPSS trajectories are processed as temporal engine histories for RUL and degradation modeling.

Explainability

SHAP is used to interpret model behavior and identify influential features.

RAG

Safety evidence is chunked, embedded, retrieved, filtered by metadata, and supplied to the local LLM.

Generative AI

The Safety Copilot separates evidence-supported facts from hypotheses and recommended actions and supports abstention when evidence is insufficient.

</details>

16. Reproducibility

A complete reproduction workflow is documented in docs/reproducibility.md.

The general sequence is:

Environment
    ↓
Datasets
    ↓
Preprocessing
    ↓
Model Training
    ↓
Cross-Validation
    ↓
Threshold Selection
    ↓
Final Evaluation
    ↓
Anomaly Detection
    ↓
Predictive Health
    ↓
Explainability
    ↓
RAG
    ↓
Dashboard

Large datasets, trained models, generated figures, and local vector-store artifacts are excluded from version control where appropriate.

17. Documentation

Document

Description

docs/architecture.md

System architecture and component flow

docs/methodology.md

Detailed research methodology

docs/datasets.md

Dataset roles, preparation, and limitations

docs/reproducibility.md

Reproduction workflow and experiment controls

18. Project Scope

ForgeShield is a research Proof of Concept, not a production industrial safety system.

The current implementation demonstrates the research workflow using:

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

19. Limitations

Current limitations include:

Benchmark datasets may not represent every real industrial environment.

Synthetic safety incidents are not equivalent to validated industrial incident records.

Simulated telemetry does not reproduce every characteristic of physical sensor systems.

The RAG evaluation is a Proof-of-Concept grounding evaluation.

Local LLM outputs remain dependent on model behavior and retrieved evidence.

Deep-learning performance depends on dataset characteristics and training configuration.

Unified risk weights are research parameters rather than universally validated industrial risk coefficients.

20. Future Work

Potential extensions include:

Physical sensor and PLC integration

Industrial telemetry ingestion

Streaming anomaly detection

Online model monitoring

Adaptive risk calibration

Larger industrial datasets

Human-in-the-loop safety validation

More rigorous RAG evaluation

Domain-validated safety procedures

Edge deployment

Industrial cybersecurity integration

Multimodal industrial safety intelligence

21. Research Paper

ForgeShield is being developed as a research-oriented Proof of Concept.

The planned paper structure is:

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

22. Project Status

Research Proof of Concept

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

ForgeShield Command Center

Research documentation

License

This project is licensed under the MIT License.

See LICENSE for details.

Disclaimer

ForgeShield is an academic research Proof of Concept.

The system demonstrates machine-learning, deep-learning, explainability, retrieval-augmented generation, and safety-intelligence concepts.

It must not be treated as a certified industrial safety system or as a substitute for qualified safety engineering, validated operating procedures, or regulatory compliance.
