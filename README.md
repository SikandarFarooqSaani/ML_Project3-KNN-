# ML_Project3-KNN-
Solar Power Generation Prediction (K-Nearest Neighbors Regression)
This project aims to predict the amount of solar power generated using weather and environmental data. It implements a K-Nearest Neighbors (KNN) Regression model, a fundamental machine learning algorithm, to forecast power output based on various input features.

Project Overview
The goal is to build a model that can accurately predict Expected_power (the target variable) from a dataset containing various sensor readings. This is a classic regression problem in machine learning, where the output is a continuous value (power in Watts).

Dataset Source: Solar Energy Power Generation Dataset on Kaggle

Final Model Performance:

R² Score: 0.78

Root Mean Squared Error (RMSE): 19,203 Watts

An R² score of 0.78 indicates a model that captures a good portion of the variation in the data, but there is significant room for improvement, as the high RMSE suggests the predictions can be quite far off in absolute terms.

Methodology
1. Data Preparation & Exploratory Data Analysis (EDA)
Data Loading & Inspection: The dataset was imported and thoroughly checked for quality.

Data Cleaning: The data was inspected for missing values (nulls), duplicate rows, and incorrect data types. This step ensured a clean and reliable dataset for training.

Correlation Analysis: A heatmap was plotted to visualize the correlation between all features and the target variable (Expected_power). This helped in understanding which factors most directly influence power generation.

2. Initial Modeling & Evaluation
The data was split into features (X) and the target variable (y).

An 80:20 train-test split was applied using a random_state=2 to ensure reproducible results.

Feature Scaling: The StandardScaler was used to normalize all features. This is a critical step for distance-based algorithms like KNN, as it ensures all features contribute equally to the distance calculation.

An initial KNN Regressor model was trained and evaluated.

The first model achieved an R² score of 0.78 and an RMSE of 19,203.

3. Experimentation and Learning
Feature Selection Test: Based on the correlation heatmap, features with low correlation to the target were dropped, and the model was retrained. This resulted in a slightly worse R² score of 0.77.

Key Learning: This experiment reinforced an important lesson: dropping features solely based on their correlation with the target can be counterproductive. A feature might have a low direct correlation but a strong interaction with other features that the model needs to make accurate predictions.

Hyperparameter Tuning: The model was adjusted by experimenting with different parameters:

weights='distance': This makes closer neighbors have a greater influence on the prediction than farther ones.

p=1: This uses Manhattan distance instead of the default Euclidean distance for calculating neighbor proximity.

n_neighbors (K value): Tested with values like 5 and 7.

Results and Conclusion
The best performance was achieved with the initial set of features after scaling, yielding an R² score of 0.78. While this is a decent baseline, it is considered average for this task.

Conclusion:
The KNN algorithm provided a reasonable baseline model for predicting solar power generation. The most significant finding was that simplistic feature removal based on correlation harmed performance, highlighting the complexity of feature relationships. The model's performance suggests that more sophisticated algorithms (like Gradient Boosting, Random Forests, or Neural Networks) or more extensive feature engineering might be necessary to achieve higher accuracy and a lower error margin.

Technologies Used
Python

Libraries:

pandas, numpy - for data manipulation and analysis

matplotlib, seaborn - for data visualization and heatmaps

scikit-learn - for machine learning tools (train_test_split, StandardScaler, KNeighborsRegressor, mean_squared_error, r2_score)
