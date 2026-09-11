# ==============================================================================
# BHUBANESWAR REAL ESTATE PRICE PREDICTOR - STREAMLIT APPLICATION
# ==============================================================================

# streamlit: Provides the interactive web UI framework and input widgets.
import streamlit as st

# pandas & numpy: For data manipulation, feature indexing, and numerical routines.
import pandas as pd
import numpy as np

# matplotlib.pyplot & seaborn: For statistical residual distributions and correlation heatmaps.
import matplotlib.pyplot as plt
import seaborn as sns

# plotly.express & plotly.graph_objects: For interactive scatter, bar, and gauge visualizations.
import plotly.express as px
import plotly.graph_objects as go

# scikit-learn modules: For ML pipeline construction, scaling, encoding, and evaluation.
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ------------------------------------------------------------------------------
# 1. PAGE SETUP & STYLING
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Bhubaneswar Real Estate Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced UI aesthetics
st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1E3A8A; margin-bottom: 0.2rem; }
    .sub-header { font-size: 1.05rem; color: #4B5563; margin-bottom: 1.5rem; }
    .prediction-box {
        background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
        color: white;
        padding: 1.8rem;
        border-radius: 14px;
        text-align: center;
        margin-top: 1rem;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# 2. DATA LOADING & CLEANING
# ------------------------------------------------------------------------------
@st.cache_data
def load_and_clean_data(file_path=r"C:\Users\LENOVO\Desktop\Bbsr\bhubaneswar_house_prices.xlsx"):
    df = pd.read_excel(file_path)
    
    # Drop duplicates and null rows if any
    df = df.dropna().drop_duplicates()
    
    # Strip string whitespaces
    df['locality'] = df['locality'].astype(str).str.strip()
    df['property_type'] = df['property_type'].astype(str).str.strip()
    df['listing_status'] = df['listing_status'].astype(str).str.strip()
    
    return df

df = load_and_clean_data()

# ------------------------------------------------------------------------------
# 3. PIPELINE MODEL TRAINING & EVALUATION
# ------------------------------------------------------------------------------
@st.cache_resource
def train_linear_regression(data, test_size=0.2, random_state=42):
    # Selected predictive features (excluding ID and target-derived leakages)
    features = ['locality', 'property_type', 'bedrooms', 'bathrooms', 'area_sqft', 'age_years', 'listing_status']
    target = 'price_inr'
    
    X = data[features]
    y = data[target]
    
    categorical_cols = ['locality', 'property_type', 'listing_status']
    numeric_cols = ['bedrooms', 'bathrooms', 'area_sqft', 'age_years']
    
    # Combined Preprocessor: Scale numeric features + One-Hot Encode categorical features
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols)
        ]
    )
    
    # End-to-end Pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])
    
    # 80/20 Train-Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Model training
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    # Metric calculations
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    
    # Extract feature importance / coefficients
    regressor = pipeline.named_steps['regressor']
    cat_encoder = pipeline.named_steps['preprocessor'].named_transformers_['cat']
    encoded_cat_cols = cat_encoder.get_feature_names_out(categorical_cols).tolist()
    all_feature_names = numeric_cols + encoded_cat_cols
    coef_df = pd.DataFrame({
        'Feature': all_feature_names,
        'Coefficient': regressor.coef_
    }).sort_values(by='Coefficient', ascending=False)
    
    return pipeline, (r2, mae, rmse, mape), (X_train, X_test, y_train, y_test, y_pred), coef_df

model, metrics, split_data, coef_df = train_linear_regression(df)
r2, mae, rmse, mape = metrics
X_train, X_test, y_train, y_test, y_pred = split_data

# INR Currency Formatter Helper
def format_inr(val):
    if val >= 10000000:
        return f"₹{val/10000000:.2f} Cr"
    elif val >= 100000:
        return f"₹{val/100000:.2f} Lakh"
    else:
        return f"₹{val:,.2f}"

# ------------------------------------------------------------------------------
# 4. SIDEBAR NAVIGATION
# ------------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/home.png", width=70)
    st.title("Navigation")
    app_mode = st.radio(
        "Go to",
        ["Price Estimator", "Model Insights & Diagnostics", "Exploratory Data Analysis", "Dataset Viewer"]
    )
    st.markdown("---")
    st.markdown("### Model Quick Facts")
    st.metric("Algorithm", "Multiple Linear Regression")
    st.metric("R² Score", f"{r2*100:.2f}%")
    st.metric("Mean Absolute Error", format_inr(mae))

# ------------------------------------------------------------------------------
# 5. VIEW: PRICE ESTIMATOR (USER INPUT INTERFACE)
# ------------------------------------------------------------------------------
if app_mode == "Price Estimator":
    st.markdown('<div class="main-header">Bhubaneswar House Price Estimator</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Enter property specifications to predict estimated market valuation in real-time.</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1.1, 0.9], gap="large")
    
    with col1:
        st.subheader("Property Attributes")
        with st.form("prediction_form"):
            c1, c2 = st.columns(2)
            with c1:
                locality = st.selectbox("Locality", sorted(df['locality'].unique()), index=sorted(df['locality'].unique()).index("Patia") if "Patia" in df['locality'].unique() else 0)
                property_type = st.selectbox("Property Type", sorted(df['property_type'].unique()))
                listing_status = st.selectbox("Listing Status", sorted(df['listing_status'].unique()))
                age_years = st.slider("Property Age (Years)", min_value=0, max_value=30, value=3, step=1)
                
            with c2:
                area_sqft = st.number_input("Super Built-up Area (sq. ft.)", min_value=300, max_value=10000, value=1500, step=50)
                bedrooms = st.slider("Bedrooms (BHK)", min_value=1, max_value=6, value=3, step=1)
                bathrooms = st.slider("Bathrooms", min_value=1, max_value=6, value=2, step=1)
            
            submit_btn = st.form_submit_button("Calculate Estimated Value 🚀", use_container_width=True)
        
    with col2:
        st.subheader("Valuation Summary")
        
        input_data = pd.DataFrame([{
            'locality': locality,
            'property_type': property_type,
            'bedrooms': bedrooms,
            'bathrooms': bathrooms,
            'area_sqft': area_sqft,
            'age_years': age_years,
            'listing_status': listing_status
        }])
        
        predicted_price = model.predict(input_data)[0]
        predicted_price_clean = max(predicted_price, 500000)
        price_per_sqft = predicted_price_clean / area_sqft
        
        st.markdown(f"""
        <div class="prediction-box">
            <h3 style="margin:0; font-size:1.1rem; opacity:0.9;">Estimated Market Value</h3>
            <h1 style="margin:0.5rem 0; font-size:2.6rem;">{format_inr(predicted_price_clean)}</h1>
            <p style="margin:0; font-size:0.95rem; opacity:0.9;">≈ ₹{predicted_price_clean:,.0f} INR</p>
        </div>
        """, unsafe_allow_html=True)
        
        m1, m2 = st.columns(2)
        with m1:
            st.metric("Price / Sq. Ft.", f"₹{price_per_sqft:,.2f}")
        with m2:
            loc_avg = df[df['locality'] == locality]['price_inr'].mean()
            st.metric(f"Avg in {locality}", format_inr(loc_avg))
            
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = predicted_price_clean / 100000,
            number = {'suffix': " Lakhs"},
            title = {'text': f"Price Tier Benchmark in {locality}"},
            gauge = {
                'axis': {'range': [None, (df['price_inr'].quantile(0.95))/100000]},
                'bar': {'color': "#2563EB"},
                'steps': [
                    {'range': [0, 60], 'color': "#E0F2FE"},
                    {'range': [60, 150], 'color': "#BAE6FD"},
                    {'range': [150, 350], 'color': "#7DD3FC"}
                ]
            }
        ))
        fig_gauge.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

# ------------------------------------------------------------------------------
# 6. VIEW: MODEL INSIGHTS & DIAGNOSTICS
# ------------------------------------------------------------------------------
elif app_mode == "Model Insights & Diagnostics":
    st.markdown('<div class="main-header">Model Performance & Regression Diagnostics</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Evaluation of Ordinary Least Squares Multiple Linear Regression on test data.</div>', unsafe_allow_html=True)
    
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("R² Score", f"{r2:.4f}")
    k2.metric("Mean Absolute Error (MAE)", format_inr(mae))
    k3.metric("Root Mean Squared Error (RMSE)", format_inr(rmse))
    k4.metric("Mean Absolute % Error (MAPE)", f"{mape:.2f}%")
    
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Actual vs Predicted Values (Plotly)")
        test_results = pd.DataFrame({
            'Actual Price (₹)': y_test,
            'Predicted Price (₹)': y_pred
        })
        sample_results = test_results.sample(n=min(1000, len(test_results)), random_state=42)
        
        fig_scatter = px.scatter(
            sample_results,
            x='Actual Price (₹)',
            y='Predicted Price (₹)',
            opacity=0.65,
            color_discrete_sequence=['#2563EB'],
            title="Actual vs. Predicted House Prices (Sample of 1,000 Points)"
        )
        min_val = min(sample_results['Actual Price (₹)'].min(), sample_results['Predicted Price (₹)'].min())
        max_val = max(sample_results['Actual Price (₹)'].max(), sample_results['Predicted Price (₹)'].max())
        fig_scatter.add_shape(
            type="line", line=dict(dash='dash', color='red', width=2),
            x0=min_val, y0=min_val, x1=max_val, y1=max_val
        )
        fig_scatter.update_layout(height=420)
        st.plotly_chart(fig_scatter, use_container_width=True)
        
    with col2:
        st.subheader("Residual Distribution (Seaborn / Matplotlib)")
        residuals = y_test - y_pred
        
        fig_res, ax = plt.subplots(figsize=(8, 5.2))
        sns.histplot(residuals / 100000, kde=True, ax=ax, color="#3B82F6", bins=40)
        ax.axvline(0, color='red', linestyle='--', linewidth=1.5)
        ax.set_title("Distribution of Residuals (Errors in ₹ Lakhs)", fontsize=12, fontweight='bold')
        ax.set_xlabel("Residual (Actual - Predicted) in ₹ Lakhs")
        ax.set_ylabel("Frequency")
        plt.tight_layout()
        st.pyplot(fig_res)
        
    st.markdown("---")
    st.subheader("Top Regression Coefficients (Feature Impact)")
    sub_coef = pd.concat([coef_df.head(6), coef_df.tail(6)]).drop_duplicates()
    
    fig_bar = px.bar(
        sub_coef,
        x='Coefficient',
        y='Feature',
        orientation='h',
        color='Coefficient',
        color_continuous_scale='Blues',
        title="Top Positive & Negative Feature Coefficients"
    )
    fig_bar.update_layout(height=400, yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig_bar, use_container_width=True)

# ------------------------------------------------------------------------------
# 7. VIEW: EXPLORATORY DATA ANALYSIS (EDA)
# ------------------------------------------------------------------------------
elif app_mode == "Exploratory Data Analysis":
    st.markdown('<div class="main-header">Exploratory Data Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Visualizing real estate trends and distributions across Bhubaneswar.</div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["Locality Analysis", "Price & Area Relationships", "Correlation Matrix"])
    
    with tab1:
        st.subheader("Average Price by Locality")
        loc_summary = df.groupby('locality')['price_inr'].agg(['mean', 'median', 'count']).reset_index()
        loc_summary = loc_summary.sort_values(by='mean', ascending=False)
        loc_summary['mean_formatted'] = loc_summary['mean'].apply(format_inr)
        
        fig_loc = px.bar(
            loc_summary,
            x='locality',
            y='mean',
            color='mean',
            color_continuous_scale='Blues',
            text='mean_formatted',
            title="Mean Property Price across Localities (INR)",
            labels={'mean': 'Average Price (₹)', 'locality': 'Locality'}
        )
        fig_loc.update_traces(textposition='outside')
        fig_loc.update_layout(height=500, xaxis_tickangle=-45)
        st.plotly_chart(fig_loc, use_container_width=True)
        
    with tab2:
        st.subheader("Area vs Price by Property Type")
        fig_scatter_eda = px.scatter(
            df.sample(n=min(1500, len(df)), random_state=42),
            x='area_sqft',
            y='price_inr',
            color='property_type',
            opacity=0.7,
            title="Property Price vs Built-up Area (Sampled)",
            labels={'area_sqft': 'Area (sq. ft.)', 'price_inr': 'Price (INR)', 'property_type': 'Property Type'},
            trendline='ols'
        )
        fig_scatter_eda.update_layout(height=480)
        st.plotly_chart(fig_scatter_eda, use_container_width=True)
        
    with tab3:
        st.subheader("Correlation Heatmap (Matplotlib & Seaborn)")
        num_df = df.select_dtypes(include=[np.number]).drop(columns=['property_id'], errors='ignore')
        corr = num_df.corr()
        
        fig_corr, ax_corr = plt.subplots(figsize=(9, 5.5))
        sns.heatmap(corr, annot=True, cmap="Blues", fmt=".2f", linewidths=0.5, ax=ax_corr)
        ax_corr.set_title("Correlation Heatmap of Numeric Features", fontsize=12, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig_corr)

# ------------------------------------------------------------------------------
# 8. VIEW: DATASET VIEWER
# ------------------------------------------------------------------------------
elif app_mode == "Dataset Viewer":
    st.markdown('<div class="main-header">Dataset Overview & Summary Statistics</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Inspect raw data, schema information, and descriptive statistics.</div>', unsafe_allow_html=True)
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records", f"{len(df):,}")
    c2.metric("Unique Localities", df['locality'].nunique())
    c3.metric("Average Price", format_inr(df['price_inr'].mean()))
    c4.metric("Average Area", f"{df['area_sqft'].mean():,.0f} sq.ft.")
    
    st.markdown("### Data Sample")
    st.dataframe(df.head(50), use_container_width=True)
    
    st.markdown("### Numerical Summary Statistics")
    st.dataframe(df.describe().T, use_container_width=True)