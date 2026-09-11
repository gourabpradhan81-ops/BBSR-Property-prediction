# BBSR-Property-prediction
Built an end-to-end Real Estate Price Prediction web app for Bhubaneswar housing data using Multiple Linear Regression, Scikit-Learn pipelines , and Streamlit with interactive Plotly &amp; Seaborn diagnostics.

```python
readme_content = """# 🏠 Bhubaneswar Real Estate Price Predictor

An end-to-end Machine Learning web application built using **Python**, **Scikit-Learn**, **Streamlit**, **Plotly**, **Matplotlib**, and **Seaborn** to estimate residential property valuations across Bhubaneswar, Odisha.

The underlying **Multiple Linear Regression (Ordinary Least Squares)** pipeline achieves an **$R^2$ accuracy score of ~93.66%** on out-of-sample validation data.

---

## 📌 Table of Contents
- [Project Overview](#-project-overview)
- [Key Features](#-key-features)
- [Model Architecture & Pipeline](#-model-architecture--pipeline)
- [Performance Metrics](#-performance-metrics)
- [Project Structure](#-project-structure)
- [Installation & Local Setup](#-installation--local-setup)
- [App Modules & User Interface](#-app-modules--user-interface)
- [Technologies Used](#-technologies-used)
- [License](#-license)

---

## 📖 Project Overview

Predicting property prices accurately requires accounting for multidimensional factors such as location tiers, built-up area, bedroom/bathroom configurations, property age, and construction status. 

This project cleans and preprocesses historical housing transactions across 20 prime localities in Bhubaneswar, trains a robust Scikit-Learn regression pipeline with automated encoding and feature scaling, and exposes an interactive, user-friendly UI for instant property valuation and exploratory data analysis.

---

## 🚀 Key Features

* **⚡ Real-Time Price Valuation:** Calculate estimated property prices in Indian Rupees (₹ Lakhs / ₹ Crores) and Price per Sq. Ft. based on custom user inputs.
* **📍 Dynamic Locality Benchmarking:** Interactive Plotly gauge comparing estimated valuations against the 95th-percentile market tier for the selected locality.
* **📊 Regression Diagnostics:** 
  * Actual vs. Predicted interactive scatter plots with 45° reference lines ($y = x$).
  * Residual error distribution with Seaborn KDE plots to verify homoscedasticity.
  * Feature coefficient impact charts (top positive drivers and negative depreciation penalties).
* **🔍 Exploratory Data Analysis (EDA):**
  * Locality price rankings across all 20 neighborhoods.
  * Area vs. Price regression trendlines grouped by property types (Apartment, Villa, Independent House, Plot).
  * Pearson correlation matrix heatmap.
* **📋 Raw Dataset & Summary Viewer:** Searchable tabular view and descriptive statistical summaries (mean, standard deviation, quartiles).

---

## 🧠 Model Architecture & Pipeline

To eliminate data leakage and ensure seamless inference on raw user inputs, preprocessing and modeling are unified into a `scikit-learn.pipeline.Pipeline`:


```

Raw Input Data (Categorical + Numerical)
│
▼
┌───────────────────────┐
│   ColumnTransformer   │
└───────────┬───────────┘
│
┌───────────┴───────────┐
▼                       ▼
StandardScaler          OneHotEncoder(drop='first')
(Numerical Features)    (Categorical Features)
│                       │
└───────────┬───────────┘
│
▼
LinearRegression (OLS)
│
▼
Estimated Property Price (₹)

```

### Feature Breakdown
* **Target ($y$):** `price_inr` (Continuous property price in INR)
* **Numerical Predictors ($X_{\\text{num}}$):** `bedrooms`, `bathrooms`, `area_sqft`, `age_years`
* **Categorical Predictors ($X_{\\text{cat}}$):** `locality` (20 locations), `property_type` (4 types), `listing_status` (3 statuses)
* **Excluded Attributes:** `property_id` (non-predictive), `city`/`state` (zero variance), `price_per_sqft_inr` (prevents target leakage).

---

## 📈 Performance Metrics

Evaluated on an 80/20 train-test split (16,000 training records, 4,000 validation records):

| Metric | Score / Value | Description |
| :--- | :--- | :--- |
| **$R^2$ Score** | **0.9366 (93.66%)** | Proportion of price variance explained by linear relationships |
| **Mean Absolute Error (MAE)** | **₹10,82,384** (~₹10.82 Lakhs) | Average magnitude of prediction error |
| **Root Mean Squared Error (RMSE)** | **₹14,51,324** (~₹14.51 Lakhs) | Standard error metric penalizing large deviations |
| **Mean Absolute % Error (MAPE)** | **~11.20%** | Average relative percentage deviation |

---

## 📁 Project Structure

```text
bhubaneswar-house-price-prediction/
│
├── app.py                                         # Full Streamlit multi-tab web application
├── bhubaneswar_house_prices.csv                   # Cleaned real estate transactions dataset
├── requirements.txt                               # Python dependencies
├── README.md                                      # Project documentation
└── Bhubaneswar_House_Price_Prediction_Documentation.pdf  # Comprehensive technical PDF guide

```

---

## 💻 Installation & Local Setup

### Prerequisites

* Python 3.9+ installed on your system.

### 1. Clone or Extract the Repository

```bash
git clone [https://github.com/your-username/bhubaneswar-house-price-prediction.git](https://github.com/your-username/bhubaneswar-house-price-prediction.git)
cd bhubaneswar-house-price-prediction

```

*(Or extract `bhubaneswar_house_price_prediction_project.zip`)*

### 2. Create and Activate a Virtual Environment

```bash
# macOS/Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\\Scripts\\activate

```

### 3. Install Dependencies

```bash
pip install -r requirements.txt

```

### 4. Launch the Streamlit App

```bash
streamlit run app.py

```

The application will open in your default browser at `http://localhost:8501`.

---

## 🖥️ App Modules & User Interface

1. **Price Estimator:** Interactive input sliders and drop-down menus to test custom property configurations and view real-time market value predictions with contextual gauge meters.
2. **Model Insights & Diagnostics:** Live test-set evaluation metrics, interactive actual-vs-predicted scatter plots, residual distribution curves, and regression weights.
3. **Exploratory Data Analysis (EDA):** Aggregated neighborhood pricing bars, bivariate area-to-price scatter charts, and correlation matrices.
4. **Dataset Viewer:** Paginated transaction browser and descriptive summary statistics.

---

## 🛠️ Technologies Used

* **Frontend & UI:** [Streamlit](https://www.google.com/search?q=https://streamlit.io/)
* **Data Processing & Manipulation:** [Pandas](https://www.google.com/search?q=https://pandas.pydata.org/), [NumPy](https://www.google.com/search?q=https://numpy.org/)
* **Machine Learning:** [Scikit-Learn](https://www.google.com/search?q=https://scikit-learn.org/)
* **Data Visualization:** [Plotly](https://www.google.com/search?q=https://plotly.com/), [Seaborn](https://www.google.com/search?q=https://seaborn.pydata.org/), [Matplotlib](https://www.google.com/search?q=https://matplotlib.org/)

---

## 📄 License

This project is open-source and available under the [MIT License](https://www.google.com/search?q=LICENSE).
"""

with open("README.md", "w") as f:
f.write(readme_content)

print("README.md written successfully. Size:", len(readme_content), "characters")

```

```text?code_stdout&code_event_index=1
README.md written successfully. Size: 6658 characters


```

Your Markdown file is ready
[file-tag: code-generated-file-09ab4c35-f6a6-42f7-8648-8f2da145b5eb]

---

# 🏠 Bhubaneswar Real Estate Price Predictor

An end-to-end Machine Learning web application built using **Python**, **Scikit-Learn**, **Streamlit**, **Plotly**, **Matplotlib**, and **Seaborn** to estimate residential property valuations across Bhubaneswar, Odisha.

The underlying **Multiple Linear Regression (Ordinary Least Squares)** pipeline achieves an **$R^2$ accuracy score of ~93.66%** on out-of-sample validation data.

---

## 📌 Table of Contents

* [Project Overview](https://www.google.com/search?q=%23-project-overview)
* [Key Features](https://www.google.com/search?q=%23-key-features)
* [Model Architecture & Pipeline](https://www.google.com/search?q=%23-model-architecture--pipeline)
* [Performance Metrics](https://www.google.com/search?q=%23-performance-metrics)
* [Project Structure](https://www.google.com/search?q=%23-project-structure)
* [Installation & Local Setup](https://www.google.com/search?q=%23-installation--local-setup)
* [App Modules & User Interface](https://www.google.com/search?q=%23-app-modules--user-interface)
* [Technologies Used](https://www.google.com/search?q=%23-technologies-used)
* [License](https://www.google.com/search?q=%23-license)

---

## 📖 Project Overview

Predicting property prices accurately requires accounting for multidimensional factors such as location tiers, built-up area, bedroom/bathroom configurations, property age, and construction status.

This project cleans and preprocesses historical housing transactions across 20 prime localities in Bhubaneswar, trains a robust Scikit-Learn regression pipeline with automated encoding and feature scaling, and exposes an interactive, user-friendly UI for instant property valuation and exploratory data analysis.

---

## 🚀 Key Features

* **⚡ Real-Time Price Valuation:** Calculate estimated property prices in Indian Rupees (₹ Lakhs / ₹ Crores) and Price per Sq. Ft. based on custom user inputs.
* **📍 Dynamic Locality Benchmarking:** Interactive Plotly gauge comparing estimated valuations against the 95th-percentile market tier for the selected locality.
* **📊 Regression Diagnostics:**
* Actual vs. Predicted interactive scatter plots with 45° reference lines ($y = x$).
* Residual error distribution with Seaborn KDE plots to verify homoscedasticity.
* Feature coefficient impact charts (top positive drivers and negative depreciation penalties).


* **🔍 Exploratory Data Analysis (EDA):**
* Locality price rankings across all 20 neighborhoods.
* Area vs. Price regression trendlines grouped by property types (Apartment, Villa, Independent House, Plot).
* Pearson correlation matrix heatmap.


* **📋 Raw Dataset & Summary Viewer:** Searchable tabular view and descriptive statistical summaries (mean, standard deviation, quartiles).

---

## 🧠 Model Architecture & Pipeline

To eliminate data leakage and ensure seamless inference on raw user inputs, preprocessing and modeling are unified into a `scikit-learn.pipeline.Pipeline`:

```text
Raw Input Data (Categorical + Numerical)
                  │
                  ▼
      ┌───────────────────────┐
      │   ColumnTransformer   │
      └───────────┬───────────┘
                  │
      ┌───────────┴───────────┐
      ▼                       ▼
StandardScaler          OneHotEncoder(drop='first')
(Numerical Features)    (Categorical Features)
      │                       │
      └───────────┬───────────┘
                  │
                  ▼
     LinearRegression (OLS)
                  │
                  ▼
    Estimated Property Price (₹)

```

### Feature Breakdown

* **Target ($y$):** `price_inr` (Continuous property price in INR)
* **Numerical Predictors ($X_{\text{num}}$):** `bedrooms`, `bathrooms`, `area_sqft`, `age_years`
* **Categorical Predictors ($X_{\text{cat}}$):** `locality` (20 locations), `property_type` (4 types), `listing_status` (3 statuses)
* **Excluded Attributes:** `property_id` (non-predictive), `city`/`state` (zero variance), `price_per_sqft_inr` (prevents target leakage).

---

## 📈 Performance Metrics

Evaluated on an 80/20 train-test split (16,000 training records, 4,000 validation records):

| Metric | Score / Value | Description |
| --- | --- | --- |
| **$R^2$ Score** | **0.9366 (93.66%)** | Proportion of price variance explained by linear relationships |
| **Mean Absolute Error (MAE)** | **₹10,82,384** (~₹10.82 Lakhs) | Average magnitude of prediction error |
| **Root Mean Squared Error (RMSE)** | **₹14,51,324** (~₹14.51 Lakhs) | Standard error metric penalizing large deviations |
| **Mean Absolute % Error (MAPE)** | **~11.20%** | Average relative percentage deviation |

---

## 📁 Project Structure

```text
bhubaneswar-house-price-prediction/
│
├── app.py                                                # Full Streamlit multi-tab web application
├── bhubaneswar_house_prices.csv                          # Cleaned real estate transactions dataset
├── requirements.txt                                      # Python dependencies
├── README.md                                             # Project documentation
└── Bhubaneswar_House_Price_Prediction_Documentation.pdf # Comprehensive technical PDF guide

```

---

## 💻 Installation & Local Setup

### Prerequisites

* Python 3.9+ installed on your system.

### 1. Clone or Extract the Repository

```bash
git clone https://github.com/your-username/bhubaneswar-house-price-prediction.git
cd bhubaneswar-house-price-prediction

```

*(Or extract `bhubaneswar_house_price_prediction_project.zip`)*

### 2. Create and Activate a Virtual Environment

```bash
# macOS/Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate

```

### 3. Install Dependencies

```bash
pip install -r requirements.txt

```

### 4. Launch the Streamlit App

```bash
streamlit run app.py

```

The application will open in your default browser at `http://localhost:8501`.

---

## 🖥️ App Modules & User Interface

1. **Price Estimator:** Interactive input sliders and drop-down menus to test custom property configurations and view real-time market value predictions with contextual gauge meters.
2. **Model Insights & Diagnostics:** Live test-set evaluation metrics, interactive actual-vs-predicted scatter plots, residual distribution curves, and regression weights.
3. **Exploratory Data Analysis (EDA):** Aggregated neighborhood pricing bars, bivariate area-to-price scatter charts, and correlation matrices.
4. **Dataset Viewer:** Paginated transaction browser and descriptive summary statistics.

---

## 🛠️ Technologies Used

* **Frontend & UI:** [Streamlit](https://www.google.com/search?q=https://streamlit.io/)
* **Data Processing & Manipulation:** [Pandas](https://www.google.com/search?q=https://pandas.pydata.org/), [NumPy](https://www.google.com/search?q=https://numpy.org/)
* **Machine Learning:** [Scikit-Learn](https://www.google.com/search?q=https://scikit-learn.org/)
* **Data Visualization:** [Plotly](https://www.google.com/search?q=https://plotly.com/), [Seaborn](https://www.google.com/search?q=https://seaborn.pydata.org/), [Matplotlib](https://www.google.com/search?q=https://matplotlib.org/)

---

## 📄 License

This project is open-source and available under the [MIT License](https://www.google.com/search?q=LICENSE).
