# Video Game Sales (1980-2024) - Machine Learning Lifecycle

## Project Overview
This repository contains the implementation of a complete Machine Learning lifecycle as part of a university assignment. The objective of this project is to process a raw dataset, clean it, train multiple classification models, and evaluate their performance. 

**Task:** Predicting whether a video game will be a "Hit" (Global Sales > 1 Million copies) based on its Platform/Console and Genre.

## Dataset
* **Source:** Kaggle (Video Games Sales 1980-2024 - Raw)
* **Description:** The dataset contains historical sales data for video games across different platforms and genres.

## Machine Learning Lifecycle Steps Implemented:

### 1. Data Fetching & Cleaning
* Loaded the raw CSV file using `pandas`.
* Standardized column names (lowercased and removed spaces).
* Dropped irrelevant columns (e.g., box art URLs, long text descriptions) that do not contribute to the predictive model.
* Handled missing values by dropping rows with `NaN` in critical columns (Sales, Genre, Platform).

### 2. Feature Engineering & Preprocessing
* **Target Variable Creation:** Created a new binary column `is_hit`. If total sales > 1.0 (million), it is classified as `1` (Hit), otherwise `0`.
* **Encoding:** Machine learning models require numerical input. Converted categorical text data (Console/Platform and Genre) into binary numeric columns using One-Hot Encoding (`pd.get_dummies`).

### 3. Data Splitting
* Separated the features (`X`) and the target variable (`y`).
* Split the dataset using `train_test_split` with an **80/20 ratio** (80% data for training the models, 20% for testing their accuracy).

### 4. Model Training
Trained three different machine learning models to compare their performance:
1. **Logistic Regression** (max_iter=2000)
2. **Decision Tree Classifier**
3. **Random Forest Classifier**

### 5. Model Evaluation
Evaluated the trained models on the 20% unseen test data using:
* **Accuracy Score:** Calculated the percentage of correct predictions for all three models.
* **Classification Report:** Generated a detailed report (Precision, Recall, F1-Score) for the Random Forest model to analyze its performance deeply.

## Tech Stack
* **Language:** Python
* **Libraries:** Pandas, Scikit-learn
