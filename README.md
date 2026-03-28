# Context-Aware Tourism Recommender System (Paper Replication)

## Overview
This repository contains a full Python implementation replicating the methodology of the research article: *"A Context-Aware Tourism Recommender System Using a Hybrid Method Combining Deep Learning and Ontology-Based Knowledge" (Paper 28)*. 

Since the original proprietary ontology data for the Santurbán region is not public, we adapted the architecture using the massive **Yelp Academic Dataset** to evaluate contextual interactions (proximity, rating, cost) alongside personalized user preferences.

## Key Features
* **Geospatial Processing:** Calculates Haversine distances to generate relative proximity metrics.
* **Deep Neural Network (DNN):** PyTorch implementation of the `128 -> 64 -> 1` regression architecture to predict optimal target scores.
* **Hybrid Scoring Algorithm:** Synthesizes `60%` objective model prediction with `40%` local semantic user preference.
* **Evaluation:** Strict `70/10/20` train/val/test splits evaluating ranking quality (AUC: 1.0, NDCG@5: 1.0) and regression accuracy.

## Project Structure
The pipeline is fully automated across 7 sequential Python scripts:

* `step1_data_loading.py` - Parses raw Yelp NDJSON into optimized pandas pickles.
* `step2_feature_engineering.py` - Calculates distance, popularity, and cost matrices.
* `step3_data_splitting.py` - Implements strict 70-10-20 train/val/test leakage prevention.
* `step4_model_training.py` - Builds and trains the PyTorch DNN with Early Stopping.
* `step5_model_evaluation.py` - Validates initial RMSE/MAE on validation set.
* `step6_final_evaluation.py` - Unseals test set; generates ROC and NDCG ranking metrics.
* `step7_hybrid_recommender.py` - Synthesizes user preferences to generate final ranked destinations.

## Setup Instructions

**1. Install Dependencies**
```bash
pip install -r requirements.txt
```

**2. Download Data**
Download the Yelp Academic Dataset from [Kaggle](https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset) and place the JSON files in the `/archive/` directory.

**3. Run the Pipeline**
Execute the scripts in sequential order:
```bash
python step1_data_loading.py
python step2_feature_engineering.py
# ... continue through step 7
```

## Results & Documentation
For a detailed breakdown of the mathematical target generation, model architecture, and the side-by-side comparison of our results against the original paper, please see the `Report.docx` included in this repository.

## Acknowledgments
* **Original Paper:** *A Context-Aware Tourism Recommender System Using a Hybrid Method Combining Deep Learning and Ontology-Based Knowledge*
* **Dataset:** Yelp Academic Dataset (via Kaggle)
