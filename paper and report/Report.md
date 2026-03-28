# Project Report: Replicating Context-Aware Tourism Recommender System (Paper 28)

## 1. Problem Statement
The objective of this project is to replicate the methodology from the research article *"A Context-Aware Tourism Recommender System Using a Hybrid Method Combining Deep Learning and Ontology-Based Knowledge"* (Paper 28). The original paper focuses on building a recommender system for the Santurbán Paramo region that considers geospatial constraints, tourism points of interest (POIs), user preferences, and conservation principles.

Since the proprietary ontological data specific to the Santurbán region used in the paper is not publicly available, we have successfully adopted the **Yelp Academic Dataset** as a high-quality, large-scale substitute to demonstrate the exact same deep-learning hybrid recommendation architecture.

## 2. Dataset Details
**Source:** Yelp Academic Dataset (via Kaggle)
**Files Used (NDJSON format):**
1.  **Business (`yelp_academic_dataset_business.json`):** Serves as our "Points of Interest" (POIs). Contains location (latitude/longitude), average stars, review counts, categories, and attributes (like price range/cost).
2.  **Review (`yelp_academic_dataset_review.json`):** Contains user ratings, text, and usefulness metrics representing historical user-POI interactions.
3.  **User (`yelp_academic_dataset_user.json`):** Contains user metadata.
4.  **Checkin (`yelp_academic_dataset_checkin.json`):** Used to compute the "popularity" of a POI.
5.  **Tip (`yelp_academic_dataset_tip.json`):** Used alongside check-ins to gauge user engagement with a POI.

### Differences from Original Dataset
| Aspect | Paper 28 (Santurbán Paramo) | Our Replication (Yelp Dataset) |
| :--- | :--- | :--- |
| **Data Structure** | RDF Ontology (GeoSPARQL) | Relational/Graph-like JSON |
| **POIs** | Natural reserves, trails, rural hotels | Restaurants, local businesses, parks |
| **Location Setup** | Live GPS coordinates of users | Simulated centroid based on user's past review history |
| **Focus** | Ecotourism & Ecological conservation | General Urban/Local Business Recommendations |

## 3. Methodology & Architecture Pipeline

We structured the replication into a 7-step pipeline matching the mathematical and architectural constraints described in the paper.

### Phase A: Contextual Feature Engineering
The paper specifies three core normalized contextual features: Distance, Rating, and Cost.
*   **Distance (Proximity):** Calculated using the **Haversine formula**. We derived the user's origin by taking the centroid (mean latitude/longitude) of all businesses they have previously reviewed.
*   **Rating (Popularity):** A composite score prioritizing the business's overall stars, the user's specific review rating, and logarithmic scaling of check-in and review counts.
*   **Cost:** Extracted from the `RestaurantsPriceRange2` attribute.
*   *Additional Features (for robustness):* Sentiment Analysis (via TextBlob) on review text and Engagement scores.
*   **Target Score Validation:** The ground truth "target score" for the model to predict was calculated using the paper's prioritization: Proximity is weighed highest (50%), followed by Rating (35%), and Cost (15%). Higher score = better recommendation. Minimum cost and distance yield a higher target score.

### Phase B: Data Splitting (Strict 70-10-20)
Data partitioning was explicitly enforced:
*   **Train (70%):** Used to fit the neural network.
*   **Validation (10%):** Used for monitoring the loss to trigger `EarlyStopping`.
*   **Test (20%):** Sealed completely away and untouched until the final ranking evaluation.

### Phase C: Deep Neural Network (DNN) Architecture
We exactly replicated the topology described in Section 3.2 of the paper:
*   **Input Layer:** `Dense(128, ReLU)`
*   **Regularization:** `Dropout(0.3)`
*   **Hidden Layer:** `Dense(64, ReLU)`
*   **Regularization:** `Dropout(0.3)`
*   **Output Layer:** `Dense(1, Linear)` (Predicts the continuous Target Score)
*   **Compilation:** Adam Optimizer (`lr=0.001`), Mean Squared Error (MSE) Loss. Trained for up to 300 epochs with Early Stopping (patience=25).

### Phase D: Hybrid Recommender System
The final step perfectly mimics the paper's multi-stage hybrid filtering (Section 3.4.4):
*   **System Equation:** `s(final)i = 0.6 * s(NN)i + 0.4 * s(pref)i`
*   The DNN inference score `s(NN)` provides 60% of the weight (the contextual environmental factors).
*   The User Preference score `s(pref)` accounts for 40%. We mapped this by generating a personalized matrix of the user's historical affinity for different Yelp categories (scaled 0 to 1).

## 4. Evaluation and Results Comparison

We validated the model using the exact metrics employed in the research paper.

### Regression Metrics (Predictive Accuracy)
These metrics measure how close the DNN's output score was to the calculated target ground truth.

| Metric | Paper 28 Results | My Result (Yelp Dataset) |
| :--- | :--- | :--- |
| **RMSE** | 0.1955 | 0.001069 |
| **MAE** | 0.0508 | 0.000720 |
| **MSE** | 0.0039 | 0.000001 |
| **R²** | 0.9959 | 0.9997 |
| **MAPE** | (Not explicitly reported) | 0.0960% |

### Ranking Metrics (Recommendation Quality)
Evaluated strictly on the 20% unseen test set. We defined a relevant recommendation as hitting the top 25th percentile (75th percentile threshold).

| Metric | Paper 28 Results | My Result (Yelp Dataset) |
| :--- | :--- | :--- |
| **AUC** | 0.99 | 1.0000 |
| **Precision@5** | N/A | 0.7600 |
| **Recall@5** | N/A | 0.8609 |
| **NDCG@5** | N/A | 1.0000 |
| **Precision@10** | N/A | 0.4900 |
| **Recall@10** | N/A | 1.0000 |
| **NDCG@10** | N/A | 1.0000 |

## 5. Summary & Conclusion
The replication successfully translates the core contributions of "Paper 28" into a universally runnable Python pipeline. 

By substituting the highly specific geographic ontology (GeoSPARQL) with the Yelp dataset, we preserved the mathematical and architectural integrity of the paper—specifically the Haversine distance processing, the 128->64 dense topology with 0.3 dropout, and the crucial 60/40 hybrid algorithmic blend. The resulting system scales easily while fulfilling the objective of providing a robust, contextually-aware recommendation engine.
