## Adult Census Income Predictor
This project tries to predict the income from a [Kaggle Dataset - Adult Census Income](https://www.kaggle.com/datasets/uciml/adult-census-income)

### Technical Implementation & Workflow Components
To achieve a robust comparative analysis, we implemented a comprehensive data science pipeline in KNIME Analytics Platform. The approach integrates unsupervised feature engineering techniques (PCA and Clustering) with supervised learning algorithms. Crucially, we implemented a "No-Leak" architecture by splitting the data into training and testing sets *prior* to feature engineering. This ensures that standardization parameters, principal components, and cluster centers are derived solely from the training partition and applied transductively to the test set, preventing look-ahead bias.

**Workflow Node Listing:**
* **CSV Reader:** Ingests the raw `adult.csv` dataset.
* **String Manipulation (Multi Column):** Cleans categorical features by stripping whitespace and removing trailing periods (e.g., converting "Private." to "Private").
* **Missing Value:** Imputes missing data points (e.g., replacing '?' with the column mode or mean).
* **Table Partitioner:** Splits the clean dataset into a 70% Training set and a 30% Test set immediately after cleaning to ensure valid evaluation.
* **Normalizer:** Calculates Z-Score standardization parameters (Mean/Variance) solely on the Training partition.
* **PCA Compute:** Performs Principal Component Analysis on the Training data to reduce dimensionality and extract variance.
* **PCA Apply:** Transforms the Training data (and separately the Test data) using the component model derived from `PCA Compute`.
* **k-Means:** Identifies 5 demographic clusters within the Training data to create a new "Cluster" feature.
* **Cluster Assigner:** Assigns rows to the nearest demographic cluster. Used on the Training set (self-assignment) and the Test set (using Training centers).
* **Normalizer (Apply):** Applies the training scaling parameters to the Test set to ensure consistent data ranges without re-calculating statistics.
* **SMOTE (Synthetic Minority Over-sampling Technique):** Balances the Training set by generating synthetic examples of the minority class (>50K) to prevent model bias.
* **Domain Calculator:** Updates the data statistics and domain bounds required for the learner nodes.
* **XGBoost Tree Ensemble Learner:** Trains the gradient-boosted decision tree model (The "Champion" Model).
* **Logistic Regression Learner:** Trains the baseline linear classification model.
* **Column Filter:** Removes categorical string columns specifically for the neural network branch to satisfy input requirements.
* **RProp MLP Learner:** Trains the Multi-Layer Perceptron (Deep Learning) model using the numeric/PCA features.
* **XGBoost / Logistic Regression / MLP Predictors:** Applies the trained models to the unseen Test data to generate classification probabilities.
* **Scorer:** Computes key performance metrics (Accuracy, Recall, Precision) and generates the confusion matrix.
* **Column Appender:** Joins the prediction probability columns from all three branches into a single table for comparison.
* **ROC Curve:** Visualizes the trade-off between sensitivity and specificity, enabling a direct comparison of the Area Under the Curve (AUC) for all three models.
