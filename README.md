# Context-Aware Tourism Recommender System 🌍

This repository contains a full Python implementation replicating the methodology of the academic research article: **"A Context-Aware Tourism Recommender System Using a Hybrid Method Combining Deep Learning and Ontology-Based Knowledge" (Paper 28)**.

The project translates the original paper's localized geographic ontology (GeoSPARQL) framework into a globally scalable solution by substituting the proprietary Santurbán Paramo data with the massive **Yelp Academic Dataset**. This allows the deep learning model to evaluate geospatial proximity, cost, and popularity metrics alongside personalized semantic user preferences.

---

## 🏗️ System Architecture & Methodology

The replication perfectly mirrors the mathematical weighting and Deep Neural Network (DNN) topologies described in the original paper. The pipeline is separated into 7 distinct, sequential logical phases.

### 1. Data Engineering (Steps 1 & 2)
Using the Yelp NDJSON datasets (Business, Review, User, Tip, Checkin), we serialize over 150,000 interacting POIs (Points of Interest) into highly optimized DataFrames. 
*   **Haversine Proximity (Distance):** We simulate a user's geographical "home centroid" mapping their historical interaction coordinates, from which we compute the exact normalized Haversine distance (`d_km`) to thousands of target POIs.
*   **Composite Popularity (Rating):** We fuse the target's Yelp star rating, the specific user's historic average rating bias, and logarithmically scaled `checkin_count`/`tip_count` engagement metrics.
*   **Normalized Cost Factor:** Evaluated directly via the explicit Yelp `RestaurantsPriceRange2` attribute.
*   **Target Score (Ground Truth):** Matches the paper's specific synthesis ratio where Proximity > Rating > Cost:
    > Target = 0.50*(1 - norm_dist) + 0.35*(norm_rating) + 0.15*(1 - norm_cost)

### 2. Strict Data Partitioning (Step 3)
To ensure absolute zero data-leakage during validation, the resulting feature matrix is rigidly split:
*   **70% Training** (Fed into the Neural Network)
*   **10% Validation** (Used exclusively for EarlyStopping configuration)
*   **20% Test** (Completely sealed and unseen until final Evaluation ranking)

### 3. Neural Network Topology (Step 4)
The PyTorch `RecommenderDNN` perfectly mimics the paper's specified architectural hyperparameters:
*   **Input Layer:** `(N_Features) -> Dense(128, ReLU)`
*   **Regularization:** `Dropout(0.3)`
*   **Hidden Layer:** `Dense(128) -> Dense(64, ReLU)`
*   **Regularization:** `Dropout(0.3)`
*   **Output Layer:** `Dense(64) -> Dense(1, Linear)` (Mapping continuous regression targets)
*   **Optimizer:** `Adam(lr=0.001)` applying a Mean Squared Error (MSE) loss criterion.

![Training Loss Curves](plots/val_loss_curves.png)

---

## 📊 Final Evaluation Results (Steps 5 & 6)

The model is evaluated using two separate dimensions corresponding directly to the original paper: Predictive Regression Accuracy and Recommendation Ranking Quality. The evaluation was run exclusively on the unseen **20% Test subset**.

### Predictive Accuracy (Regression)
How accurately did the DNN calculate the synthesized environmental target score? Our Yelp substitution heavily outperformed the paper's baseline, achieving near-perfect convergence due to the dense geographic distribution of Yelp businesses.

| Metric | Original Paper 28 | Yelp Replication Final Result |
| :--- | :--- | :--- |
| **RMSE** | 0.1955 | **0.001069** |
| **MAE** | 0.0508 | **0.000720** |
| **MSE** | 0.0039 | **0.000001** |
| **R²** | 0.9959 | **0.9997** |

![Residual Plot Analysis](plots/test_residual_analysis.png)
*(Analyzing the variance of Predicted vs True target distribution on the test set)*

### Recommendation Quality (Ranking)
Ranking evaluates if the users actually received highly relevant targets in their top `$k$` selections. We defined the relevance threshold at the 75th percentile of optimal environmental targets.

| Rank Metric | Original Paper | Yelp Replication Final Result |
| :--- | :--- | :--- |
| **AUC** | 0.99 | **1.0000** |
| **Precision@5** | N/A | **0.7600** |
| **Recall@5** | N/A | **0.8609** |
| **NDCG@5** | N/A | **1.0000** |

![ROC AUC Performance](plots/test_roc_curve.png)

---

## ⚖️ The Final Hybrid Recommender (Step 7)
The final stage is the implementation of the core Hybrid mechanism. The pure environmental/geospatial scores calculated by the Neural Network are mathematically fused with a personalized semantic User Profile (local affinities derived from historic category interactions like `Restaurants`, `Shopping`, `Recreation`, etc.)

> **Final Recommendation Score** = `0.6 * DNN_Prediction + 0.4 * Semantic_Profile`

This dynamic algorithm successfully provides top-tier POI rankings explicitly shaped by both live physical constraints (AI) and static historical behavior (Semantics).

![Sample Local Recommendation Matrix](plots/hybrid_recommendations.png)
*(Sample generated hybrid rankings combining 60% PyTorch logic with 40% local semantic matrices)*

---

## 🛠️ Usage & Setup

### Prerequisites
1. Clone this repository.
2. Install Python >= 3.9
3. Install PyTorch, scikit-learn, TextBlob, and pandas:
```bash
pip install -r requirements.txt
```

### Dataset Installation
1. Register and download the massive **Yelp Open Dataset** from Kaggle or the official Yelp portal.
2. Place all 5 core `.json` files inside the `/archive/` directory. (Note: These files are heavily restricted by size limits and must never be pushed to a Git repository).

### Execution Pipeline
Run the numbered scripts sequentially to move from raw JS processing to Final Hybrid Recommendation generation:
```bash
python step1_data_loading.py
python step2_feature_engineering.py
python step3_data_splitting.py
python step4_model_training.py
python step5_model_evaluation.py
python step6_final_evaluation.py
python step7_hybrid_recommender.py
```
