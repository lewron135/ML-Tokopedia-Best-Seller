# E-Commerce Bestseller Prediction (Tokopedia)

## Project Overview
This project is a Machine Learning classification system designed to predict whether an e-commerce product (Tokopedia) has the potential to become a **Bestseller** based on its metadata. 

The primary goal of this project is to provide actionable **Business Value** for e-commerce platforms and sellers to:
1. **Inventory Management:** Identify high-potential products early to prevent out-of-stock scenarios.
2. **Marketing Optimization:** Filter out underperforming products and accurately select items worthy of promotional campaigns (featured products).
3. **Strategic Insights:** Understand the actual driving factors behind e-commerce sales.

## Dataset & Target Definition
* **Data Source:** Scraped Tokopedia product data containing **~29,500 products**.
* **Target Variable:** `is_bestseller`. Defined as products in the top 25% of sales volume (Threshold: > 250 units sold).
* **Data Characteristics:** The dataset has a class imbalance ratio of roughly 3:1 (76% Regular : 24% Bestseller).

## Key Business Insights
Through data exploration and Model Feature Importance, two crucial business insights were discovered:

1. **The Discount Paradox:** Although widely considered the main driver of sales, *Discount Percentage* only accounts for ~5.4% of the model's decision-making. The dominant factor (73%) actually comes from *Social Proof* (Rating and Review Volume).
2. **The Strict Gatekeeper:** The model naturally acts as a conservative gatekeeper. It prefers to miss out on some potential products (False Negatives) rather than falsely recommending the wrong products (False Positives), thereby minimizing wasted marketing budgets.

## Technical Workflow & Feature Engineering
1. **Data Cleaning:** Transformed messy text data in the sales column (e.g., "1rb+ terjual", "1jt", or typos like "tebatas") into pure numeric formats.
2. **Feature Engineering (Core):** Engineered a custom variable called **`Trust_Score`** (calculated as `Rating` x `Number of Reviews`). This feature proved to be the #1 predictor in detecting top-tier products.
3. **Data Preprocessing:** Handled extreme price outliers using `RobustScaler` and addressed the imbalanced target variable using `class_weight='balanced'`.
4. **Modeling:** Evaluated and compared state-of-the-art algorithms, specifically Logistic Regression, Random Forest, and XGBoost.

## Model Performance
**Random Forest** was selected as the final model due to its high and stable evaluation metrics:

| Metric | Score | Interpretation |
|---|---|---|
| **ROC-AUC** | 0.905 | **Excellent**. The model has a very strong discriminative ability to separate bestsellers from regular products. |
| **Accuracy** | 83.5% | Overall classification accuracy across the test set. |
| **Recall** | 79.0% | The model successfully identifies roughly 8 out of 10 actual bestsellers. |
| **Precision** | 61.0% | Acceptable precision rate considering the initial heavy class imbalance (24% baseline). |

*(Note: The trained model has been exported to a `.pickle` file and is ready to be integrated into a Streamlit dashboard for real-time predictions).*

## Tech Stack
* **Language:** Python 3
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn, XGBoost
* **Data Visualization:** Matplotlib, Seaborn
* **Deployment:** Streamlit (Pending)

## File Structure
* `main.ipynb`: The main notebook containing EDA, Preprocessing, Model Training, and Evaluation.
* `produk_tokopedia.csv`: The original dataset.
* `model_bestseller.pickle`: The exported, deployment-ready Machine Learning model.

## Future Work / Next Steps
To further improve the model's precision and business utility, the following steps are planned:
1. **Decision Threshold Tuning:** Manually adjusting the probability threshold (e.g., from 0.5 to 0.65) to heavily increase Precision for stricter promotional filtering.
2. **NLP on Product Names:** Utilizing TF-IDF or CountVectorizer to extract high-converting keywords (e.g., "Premium", "Original", "Promo") from the product titles.
3. **Adding Store Reputation Features:** Integrating seller metadata (e.g., Official Store / Power Merchant status) as this creates a strong buyer bias in real-world scenarios.

## Author
- Josep Natael Pasaribu
- Mark Lengkong
- Michelle Pricillia
- Marcellino Varian Saputra
