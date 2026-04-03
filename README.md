# E-Commerce Best Seller Prediction App

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange.svg)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Processing-yellow.svg)

## Project Overview
This project is an **End-to-End Machine Learning** application designed to help e-commerce sellers and store managers predict the probability of a product becoming a *Best Seller*. 

Instead of relying on guesswork for pricing and discount strategies, this application leverages a **Random Forest Classifier** trained on tens of thousands of real-world e-commerce data points. It provides users with data-driven predictions and actionable business insights to maximize sales potential.

## Key Features
This interactive web application (built with Streamlit) offers three main features:
1. **New Product Simulation:** Calculates the probability of success for unlaunched products by allowing users to simulate various Pricing and Discount strategies.
2. **Existing Product Analysis:** Analyzes the metrics of currently active products (including actual Ratings and Review counts) to provide optimization recommendations (*Actionable Insights*).
3. **Automated Business Consultant:** Generates specific, rule-based text recommendations tailored to the user's input (e.g., suggesting a higher discount rate if the base price exceeds a certain psychological threshold).

## Machine Learning Pipeline
This project encompasses the full data science lifecycle:
* **Data Collection:** Utilized a dataset containing ~29,000 rows of scraped e-commerce product data.
* **Data Cleaning & Preprocessing:** * Extracted pure numerical values from messy text formats (e.g., converting "1k+ sold" into `1000`).
  * Handled missing values for ratings and reviews.
  * Applied outlier removal on price distributions (cutting off the 1st and 99th percentiles).
* **Feature Engineering:** Developed new predictive features reflecting buyer psychology, such as `Effective_Price`, `Has_Discount` boolean, and `Trust_Score` (an interaction term between rating and review count).
* **Modeling:** Trained a **Random Forest Classifier**.
* **Performance:** The model achieved an overall accuracy of **83.74%**, with an impressive **0.89 F1-Score** in detecting high-risk (underperforming) products.

## Tech Stack
* **Programming Language:** Python
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn
* **Model Serialization:** Pickle
* **Web App GUI:** Streamlit

## How to Run Locally

1. **Clone this repository**
   ```bash
   git clone [https://github.com/lewron135/ML-Tokopedia-Best-Seller.git](https://github.com/lewron135/ML-Tokopedia-Best-Seller.git)
   cd ML-Tokopedia-Best-Seller
