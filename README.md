# Spatio-Temporal Predictive Policing Engine Using Graph Attention Networks

**Academic Affiliation:** Department of Computer Science and Engineering (Data Science)  
**Project Guide:** Mrs. V. Munni  

---

##  Abstract & Overview

Traditional spatial mapping in law enforcement often relies on static hotspot analysis, which fails to capture the dynamic, evolving nature of communication and mobility networks. This project introduces a **Spatio-Temporal Predictive Policing Engine** powered by **Graph Attention Networks (ST-GAT)**.

By converting spatial zones and telemetric metadata into a dynamic network graph, the engine models the ebb and flow of anomalous activities. Crucially, to address the legal and constitutional "black box" problem of deep learning in law enforcement, this architecture integrates **Explainable AI (XAI)** via `GNNExplainer`. This ensures that every flagged anomaly provides transparent, mathematically reviewable evidence detailing exactly *why* a node was classified as high-risk, fulfilling the strict operational requirements of modern data protection frameworks.

*Note: Due to legal restrictions regarding the storage and processing of actual Call Data Records (CDRs), this repository utilizes a robust data generation pipeline to create statistically sound, synthetic telemetric distributions for model validation.*

---

##  Core Architecture & Tech Stack

This project is built strictly using a robust, purely Python-based data science and machine learning stack, avoiding unnecessary web-framework overhead.

* **Data Manipulation & Processing:** Python, Pandas, NumPy
* **Deep Learning & Graph Modeling:** PyTorch, PyTorch Geometric, NetworkX
* **Explainable AI (XAI):** GNNExplainer
* **Interactive Deployment Dashboard:** Streamlit, Plotly
* **Reporting Output:** FPDF

---

##  Repository Structure

```text
spatio-temporal-policing-engine/
│
├── data/                       # Directory for generated synthetic CDRs and PDF reports
│
├── src/
│   ├── data_pipeline.py        # Generates synthetic telemetric data and extracts node features
│   ├── gat_model.py            # ST-GAT PyTorch architecture and XAI initialization
│   └── report_engine.py        # Logic for automated PDF dossier generation
│
├── app.py                      # Main Streamlit dashboard deployment file
├── requirements.txt            # Project dependencies
└── README.md                   # Academic documentation


## Key Features
* Interactive Gradio Interface: A functional web-app where users can upload CDR files and receive visual pattern analysis.
* Real-time Prediction UI: Designed custom Gradio components to display risk-priority alerts and temporal trends.
* User-Centric Inputs: Simplified complex data entry into intuitive sliders and file-upload blocks.


Developed as part of the Smart Policing initiative for the Centre for Human Security Studies.
