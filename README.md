# E-Commerce Best Seller Prediction App

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange.svg)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Processing-yellow.svg)

## Project Overview
This project is an **End-to-End Machine Learning** application designed to help e-commerce sellers and store managers predict the probability of a product becoming a *Best Seller*. 

Instead of relying on guesswork for pricing and discount strategies, this application leverages a **Random Forest Classifier** trained on real-world Tokopedia product data. It provides users with data-driven predictions and actionable business insights to maximize sales potential.

**Definition of "Best Seller" in this model:** A product that has sold **> 250 pieces**, which represents the top 25% (75th percentile) of the highest-performing products in the dataset.

## Key Features
This interactive web application (built with Streamlit) offers three main features:
1. **New Product Simulation:** Calculates the probability of success for unlaunched products by allowing users to simulate various Pricing and Discount strategies (assuming zero reviews/ratings).
2. **Existing Product Analysis:** Analyzes the metrics of currently active products (including actual Ratings and Review counts) to provide optimization recommendations.
3. **Automated Business Consultant:** Generates specific, rule-based text recommendations tailored to the user's input (e.g., suggesting a higher discount rate if the base price exceeds a psychological threshold like Rp 500,000).

## Machine Learning Pipeline (main.ipynb)
This project encompasses the full data science lifecycle:
* **Data Collection:** Utilized a dataset containing **29,519 rows** of scraped Tokopedia e-commerce product data.
* **Data Cleaning & Preprocessing:** * Extracted pure numerical values from messy text formats (e.g., converting "1rb+ terjual" into `1000`, handling "jt" for millions).
  * Handled missing values (`NaN`) by filling them with appropriate numerical defaults.
  * Applied outlier removal on price distributions by clipping the 1st and 99th percentiles, resulting in a clean **28,929 rows** of data.
* **Feature Engineering:** Developed new predictive features reflecting buyer psychology:
  * `Harga_setelah_diskon` (Effective Price)
  * `Ada_diskon` (Boolean: 1 if discount > 0, else 0)
  * `Skor_kepercayaan` (Trust Score: An interaction term between Rating and Clean Review Count).
* **Modeling:** Trained a **Random Forest Classifier** with balanced class weights to handle imbalanced best-seller target data.
* **Performance:** The model was evaluated using **5-Fold Cross-Validation** to ensure reliability. It achieved an excellent **ROC-AUC Score of 0.889 ± 0.008**. The top contributing features identified by the model were `Rating` and `Harga_setelah_diskon`.

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
