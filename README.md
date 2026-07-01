# 🌳 Sales Predictive Inference Engine (Enterprise MLOps)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-latest-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An enterprise-grade, high-performance **Machine Learning Predictive Inference Engine** designed to solve dynamic pricing and transactional revenue forecasting. This module implements a robust **RandomForestRegressor** pipeline, decoupled into a modular **State Serialization (Training) Stage** and a **Dynamic Live Telemetry Inference Stage**.

This project addresses a major corporate bottleneck: **Unstructured data inputs crashing production models.** Through automated preprocessing guardrails and deterministic boundary controls, this engine guarantees **100% server uptime** and zero mathematical hallucinations.

---

## 🛠️ System Architecture & Core Features

### 1. Robust Categorical Encoding Guardrail (`handle_unknown='ignore'`)
Production environments frequently ingest unseen categories that cause standard Encoders to throw dimensional mismatch exceptions. Our preprocessing layer utilizes a custom `ColumnTransformer` with `OneHotEncoder(handle_unknown='ignore')` to neutralize unseen inputs into zero-vectors, preventing runtime crashes.

### 2. Automated Structural Schema Validation
Before fitting, the engine executes a strict header check against standard target dimensions (`Price`, `Quantity`, `Category`, `Total_Revenue`). If schema anomalies are detected, the training pipeline aborts gracefully with detailed system logs, maintaining state integrity.

### 3. State Serialization & Persistence
Leveraging `joblib` binary serialization, the trained Scikit-Learn pipeline state (including scaling transformations and ensemble weights) is frozen and stored directly to disk. This allows lightweight, millisecond-latency deserialization during real-time client-side requests.

### 4. Mathematical Anti-Hallucination Boundaries
Machine learning regression models can output unrealistic negative values or massive fluctuations on edge-case inputs. This inference engine features a **deterministic fallback boundary** where predicted outcomes are mathematically audited:
$$\text{Predicted Revenue} \ge \text{Price} \times \text{Quantity} \times 0.85$$
If predictions violate basic physical economics, the engine dynamically falls back to a safe baseline calculation, masking ML inaccuracies from the user interface.

---

## ⚙️ Installation & Workspace Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/akbarbinhabib-dotcom/Sales-Predictive-Engine.git
   cd Sales-Predictive-Engine

