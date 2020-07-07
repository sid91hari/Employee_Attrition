# Employee_Attrition
Employees are the most important part of an organization. Successful employees meet deadlines, make sales, and build the brand through positive customer interactions.

Employee attrition is a major cost to an organization and predicting such attritions is the most important requirement of the Human Resources department in many organizations. The objective of this challenge is to predict the attrition rate of employees of an organization. 

This competition is organized by HackerEarth.

Competition link:
https://www.hackerearth.com/challenges/competitive/hackerearth-machine-learning-challenge-predict-employee-attrition-rate/

### Top 6% Finish (Out of 5000+ participants)

-----------------------------------------------------------------------------------------------------------------------------------------------------
APPROACH
-----------------------------------------------------------------------------------------------------------------------------------------------------

Response variable: Attrition_rate <br>
Evaluation Metric: Score = 100*max(0,1-RMSE(actual_values, predicted_values)) <br>
R language is used for programming. <br>
The R script provides the entire analysis and machine learning models (GBM/XGBoost/Light GBM/CatBoost). <br>
The Jupter Notebook provides the details of the analysis and the final model written in R. <br>

R script: emp_attrition.R <br>
Jupter notebook (R): Emp_Attrition.ipynb

-----------------------------------------------------------------------------------------------------------------------------------------------------
I. Data Cleaning and Exploratory Data Analysis
-----------------------------------------------------------------------------------------------------------------------------------------------------

1. Checking blanks/NAs
2. Null value treatment
2. Analysis of response variable
3. Numerical variables vs Response variable
4. Categorical variables vs Response variable

-----------------------------------------------------------------------------------------------------------------------------------------------------
II. Data Preprocessing
-----------------------------------------------------------------------------------------------------------------------------------------------------

1. Encoding of categorical variables
2. Preprocessing and transforming quantitative variables

-----------------------------------------------------------------------------------------------------------------------------------------------------
III. Modeling
-----------------------------------------------------------------------------------------------------------------------------------------------------

1. Cross Validation
2. Model build and assessment
