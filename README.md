# Usecase-7
# Player Market Value Prediction

## Introduction

Understanding a player's market value is crucial in football, affecting transfers, team investments, and scouting decisions. This project analyzes various player attributes and performance metrics to predict their market value accurately.

## Objectives

- Develop a regression model to predict market value based on player attributes.
- Evaluate model performance and refine it using feature engineering and tuning.

## Dataset Overview

The dataset includes multiple features related to football players, such as: Name, team, position, height, assists, appearances, clean sheets, etc.

## Exploratory Data Analysis (EDA)

1. **Data Viewing and Profiling:**
   - Checked dataset structure, column names, and data types.
   - Identified and handled missing values.
   - Standardized categorical variables for consistency.

2. **Data Cleaning:**

3. **Univariate Analysis:**
   - Visualized market value distribution using histograms and box plots.
   - Examined player positions and age distribution trends.

4. **Bivariate & Multivariate Analysis:**
   - Used heatmaps to explore correlations between numerical features.
   - Analyzed how age and position influence market value.
   - Compared market values across different positions with box plots.

5. **Outlier Detection:**
   - Applied IQR filtering to remove extreme outliers in market value.

## Machine Learning Modeling

1. **Feature Engineering:**
   - Encoded categorical features (position, awards) into numerical values.
   - Scaled numerical variables to improve model efficiency.

2. **Model Training:**
   - Trained multiple regression models for market value prediction.
   - Split data into training and testing sets for evaluation.

3. **Performance Evaluation:**
   - Used RMSE, MAE, and R-squared to measure model accuracy.

4. **Hyperparameter Tuning:**
   - Fine-tuned model parameters using GridSearchCV for optimization.

5. **Validation & Overfitting Check:**
   - Compared training and test set performance to ensure generalization.
   - Analyzed residuals to check for normal distribution and errors.

## Key Insights

- **Younger players with strong performance metrics tend to have higher market values.**
- **Attackers and midfielders generally command higher market prices.**
- **Injuries and career length impact a player's worth significantly.**
- **Machine learning models provide decent predictions, but more external factors could improve accuracy.**

## Conclusion

This project provides insights into how different attributes influence a player's market value. By combining EDA with machine learning, we built a predictive model that estimates a player's worth. Future improvements can include incorporating external data like contract details, club reputation, and league performance for better accuracy.

## How to Run the Project
1. Install necessary libraries: `pip install -r requirements.txt`
2. Run the Jupyter Notebook to perform EDA and train models.
3. Analyze the generated visualizations and model results.
