# Sales Forecasting and Customer Behavior Analysis for Rossmann Pharmaceuticals Using Machine Learning and Deep Learning

##  Overview
This project focuses on predicting future sales for Rossmann Pharmaceuticals' stores using a combination of machine learning and deep learning techniques. The project aims to deliver actionable sales forecasts, allowing the finance team to plan and allocate resources efficiently. The solution includes detailed exploratory data analysis, machine learning modeling, deep learning with LSTM for time series prediction, and serving the predictions through a web API.

## Objective
The primary objective of this project is to build and deploy an end-to-end machine learning and deep learning solution for Rossmann Pharmaceuticals to forecast store sales up to six weeks in advance. The solution will aid the finance team in optimizing resource allocation, planning, and decision-making by providing accurate sales predictions based on various factors such as promotions, competition, holidays, seasonality, and locality. Key objectives include:

- Conducting exploratory data analysis (EDA) to understand customer purchasing behavior.
- Building a machine learning model to predict store sales using scikit-learn pipelines.
- Creating a deep learning LSTM model for time series forecasting.
- Serving predictions through a web-based API for real-time access.

## Methdology
### Exploration of Customer Purchasing Behavior
Goal: Analyze customer behavior based on various factors like promotions, holidays, and competition.
Key Questions Answered:
 - How do promotions impact sales and customer behavior?
 - What is the effect of holidays on sales?
 - How does the distance to competitors affect store sales?
 - Are there observable seasonal trends in purchasing behavior?

### Prediction of Store Sales

- Preprocessing: Handle missing data, outliers, and feature engineering (e.g., creating features from date columns).
- Model Building: Develop machine learning models using scikit-learn pipelines (starting with Random Forest Regressor).
- Loss Function: Choose a loss function that best fits the sales forecasting objective and evaluate the model’s performance.
- Post-Prediction Analysis: Investigate feature importance and estimate confidence intervals for predictions.
- Serialization: Save the trained models with timestamps for tracking and version control.
- Deep Learning Model: Build a two-layer LSTM model for sales prediction using time-series data.

### Model Serving API
Objective: Serve the trained machine learning and deep learning models through a REST API for real-time sales forecasting.
Key Steps:
- Build a REST API using Flask/FastAPI to handle prediction requests.
- Load and serve the serialized models from Task 2.
- Deploy the API for live use.
  
## Conclusion
This project provides a comprehensive machine learning pipeline for sales forecasting and customer behavior analysis. By leveraging data-driven insights and deploying prediction models through a web interface, Rossmann Pharmaceuticals can make informed decisions to optimize store operations and resource allocation.

## License
This project is licensed under the Apache 2.0 License.
