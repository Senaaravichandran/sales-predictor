import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler, PolynomialFeatures
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Page configuration
st.set_page_config(page_title="Advanced Sales Prediction App", layout="wide")

# Custom CSS for better appearance
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3D59;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #3A6B7E;
    }
    .section-header {
        font-size: 1.5rem;
        color: #3A6B7E;
        padding-top: 1rem;
    }
    .info-text {
        background-color: #F4F4F4;
        padding: 1rem;
        border-radius: 5px;
    }
    .metric-card {
        background-color: #F4F9F9;
        border-radius: 5px;
        padding: 1rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# App header
st.markdown('<p class="main-header">🚀 Advanced Sales Prediction & Analytics App</p>', unsafe_allow_html=True)
st.markdown("""
<p class="info-text">
This app performs advanced data analysis, feature engineering, and machine learning to predict 
sales quantities. It leverages XGBoost with optimized hyperparameters and provides 
detailed visualizations and insights.
</p>
""", unsafe_allow_html=True)

# Create tabs for better organization
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Data Exploration", "🧹 Data Cleaning", "🔧 Feature Engineering", 
                                      "🤖 Model Training", "📈 Predictions & Analysis"])

# File uploader
with tab1:
    st.markdown('<p class="section-header">Upload Your Dataset</p>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])
    
    if uploaded_file is not None:
        # Load data and show preview
        df = pd.read_csv(uploaded_file)
        st.markdown('<p class="section-header">Dataset Preview</p>', unsafe_allow_html=True)
        st.dataframe(df.head())
        
        # Basic info
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Dataset Shape:** {df.shape[0]} rows, {df.shape[1]} columns")
        with col2:
            st.write(f"**Memory Usage:** {df.memory_usage().sum() / 1024 / 1024:.2f} MB")
        
        # Data types
        st.markdown('<p class="section-header">Data Types</p>', unsafe_allow_html=True)
        dtypes_df = pd.DataFrame(df.dtypes, columns=['Data Type'])
        dtypes_df.index.name = 'Column'
        dtypes_df = dtypes_df.reset_index()
        st.dataframe(dtypes_df)
        
        # Summary statistics
        st.markdown('<p class="section-header">Summary Statistics</p>', unsafe_allow_html=True)
        st.write(df.describe().T)
        
        # Missing values chart
        st.markdown('<p class="section-header">Missing Values Analysis</p>', unsafe_allow_html=True)
        missing_data = df.isnull().sum()
        missing_percent = (missing_data / len(df)) * 100
        missing_df = pd.DataFrame({'Missing Count': missing_data, 
                                'Missing Percent': missing_percent})
        missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Percent', ascending=False)
        
        if not missing_df.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x=missing_df.index, y='Missing Percent', data=missing_df, color='coral')
            plt.title('Missing Values by Column')
            plt.xticks(rotation=45, ha='right')
            plt.ylabel('Missing Percentage (%)')
            st.pyplot(fig)
        else:
            st.success("No missing values found in the dataset!")

# Data Cleaning
with tab2:
    if 'df' in locals():
        st.markdown('<p class="section-header">Data Cleaning Process</p>', unsafe_allow_html=True)
        
        # Create a copy to preserve original data
        clean_df = df.copy()
        
        # Common data cleaning operations
        cleaning_options = st.multiselect(
            "Select cleaning operations to perform:",
            ["Remove duplicate rows", 
             "Drop unnecessary columns", 
             "Handle missing values",
             "Convert date columns",
             "Handle outliers"],
            default=["Remove duplicate rows", "Drop unnecessary columns", "Handle missing values", "Convert date columns"]
        )
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("Before cleaning:")
            st.write(f"Shape: {clean_df.shape}")
            st.write(f"Missing values: {clean_df.isnull().sum().sum()}")
            st.write(f"Duplicates: {clean_df.duplicated().sum()}")
        
        # Handle each cleaning operation
        if "Remove duplicate rows" in cleaning_options:
            duplicate_count = clean_df.duplicated().sum()
            clean_df.drop_duplicates(inplace=True)
            st.write(f"✓ Removed {duplicate_count} duplicate rows")
        
        if "Drop unnecessary columns" in cleaning_options:
            cols_to_drop = st.multiselect(
                "Select columns to drop:",
                clean_df.columns.tolist(),
                default=["Unnamed: 0", "Un_Named"] if any(col in clean_df.columns for col in ["Unnamed: 0", "Un_Named"]) else []
            )
            if cols_to_drop:
                clean_df.drop(columns=cols_to_drop, errors='ignore', inplace=True)
                st.write(f"✓ Dropped columns: {', '.join(cols_to_drop)}")
        
        if "Handle missing values" in cleaning_options:
            critical_cols = st.multiselect(
                "Select critical columns (rows with missing values in these columns will be dropped):",
                clean_df.columns.tolist(),
                default=["Date", "Infrastructure_Machineries"] if all(col in clean_df.columns for col in ["Date", "Infrastructure_Machineries"]) else []
            )
            
            if critical_cols:
                before_count = len(clean_df)
                clean_df.dropna(subset=critical_cols, inplace=True)
                after_count = len(clean_df)
                st.write(f"✓ Dropped {before_count - after_count} rows with missing values in critical columns")
            
            # Handle remaining missing values
            numeric_cols = clean_df.select_dtypes(include=['float64', 'int64']).columns
            for col in numeric_cols:
                # Use median for numeric columns (more robust to outliers)
                clean_df[col] = clean_df[col].fillna(clean_df[col].median())
            
            categorical_cols = clean_df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                # Use mode for categorical columns
                if not clean_df[col].empty and clean_df[col].isnull().any():
                    clean_df[col] = clean_df[col].fillna(clean_df[col].mode().iloc[0])
            
            st.write("✓ Filled remaining missing values using median for numeric and mode for categorical columns")
        
        if "Convert date columns" in cleaning_options:
            date_cols = st.multiselect(
                "Select date columns to convert:",
                clean_df.columns.tolist(),
                default=["Date"] if "Date" in clean_df.columns else []
            )
            
            for col in date_cols:
                try:
                    clean_df[col] = pd.to_datetime(clean_df[col], errors='coerce')
                    st.write(f"✓ Converted {col} to datetime format")
                    
                    # Create date features
                    create_date_features = st.checkbox(f"Extract features from {col}", value=True)
                    if create_date_features:
                        clean_df[f'Year_{col}'] = clean_df[col].dt.year
                        clean_df[f'Month_{col}'] = clean_df[col].dt.month
                        clean_df[f'Day_{col}'] = clean_df[col].dt.day
                        clean_df[f'DayOfWeek_{col}'] = clean_df[col].dt.dayofweek
                        clean_df[f'Quarter_{col}'] = clean_df[col].dt.quarter
                        clean_df[f'IsWeekend_{col}'] = clean_df[col].dt.dayofweek.apply(lambda x: 1 if x >= 5 else 0)
                        st.write(f"✓ Created date features from {col}")
                except Exception as e:
                    st.error(f"Error converting {col} to datetime: {e}")
        
        if "Handle outliers" in cleaning_options:
            numeric_cols = clean_df.select_dtypes(include=['float64', 'int64']).columns.tolist()
            outlier_col = st.selectbox(
                "Select column to check for outliers:",
                numeric_cols,
                index=numeric_cols.index('Daily_Sales_Quantity') if 'Daily_Sales_Quantity' in numeric_cols else 0
            )
            
            method = st.radio(
                "Select outlier detection method:",
                ["IQR (Interquartile Range)", "Z-score", "Isolation Forest"]
            )
            
            if method == "IQR (Interquartile Range)":
                Q1 = clean_df[outlier_col].quantile(0.25)
                Q3 = clean_df[outlier_col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = clean_df[(clean_df[outlier_col] < lower_bound) | (clean_df[outlier_col] > upper_bound)]
                
                st.write(f"IQR Bounds: Lower = {lower_bound:.2f}, Upper = {upper_bound:.2f}")
                st.write(f"Detected {len(outliers)} outliers ({len(outliers)/len(clean_df)*100:.2f}%)")
                
                # Option to cap outliers
                if st.checkbox("Cap outliers (Winsorization)", value=True):
                    clean_df[f'{outlier_col}_original'] = clean_df[outlier_col].copy()
                    clean_df[outlier_col] = clean_df[outlier_col].clip(lower=lower_bound, upper=upper_bound)
                    st.write(f"✓ Capped outliers in {outlier_col}")
            
            elif method == "Z-score":
                z_scores = zscore(clean_df[outlier_col], nan_policy='omit')
                z_threshold = st.slider("Z-score threshold", 2.0, 5.0, 3.0, 0.1)
                outliers = clean_df[abs(z_scores) > z_threshold]
                
                st.write(f"Z-score threshold: {z_threshold}")
                st.write(f"Detected {len(outliers)} outliers ({len(outliers)/len(clean_df)*100:.2f}%)")
                
                if st.checkbox("Replace outliers with median", value=False):
                    clean_df[f'{outlier_col}_original'] = clean_df[outlier_col].copy()
                    median_val = clean_df[outlier_col].median()
                    clean_df.loc[abs(z_scores) > z_threshold, outlier_col] = median_val
                    st.write(f"✓ Replaced outliers in {outlier_col} with median ({median_val:.2f})")
            
            else:  # Isolation Forest
                X_for_iso = clean_df[[outlier_col]].copy()
                contamination = st.slider("Contamination parameter", 0.01, 0.1, 0.05, 0.01)
                iso_forest = IsolationForest(contamination=contamination, random_state=42)
                outlier_preds = iso_forest.fit_predict(X_for_iso)
                outliers = clean_df[outlier_preds == -1]
                
                st.write(f"Contamination: {contamination}")
                st.write(f"Detected {len(outliers)} outliers ({len(outliers)/len(clean_df)*100:.2f}%)")
                
                if st.checkbox("Remove detected outliers", value=False):
                    clean_df = clean_df[outlier_preds == 1]
                    st.write(f"✓ Removed {len(outliers)} outliers")
            
            # Visualize outliers
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(y=clean_df[outlier_col], ax=ax)
            plt.title(f'Boxplot of {outlier_col}')
            st.pyplot(fig)
            
            # Distribution before and after
            if f'{outlier_col}_original' in clean_df.columns:
                fig, ax = plt.subplots(1, 2, figsize=(16, 6))
                sns.histplot(clean_df[f'{outlier_col}_original'], kde=True, ax=ax[0])
                ax[0].set_title(f'Distribution Before Outlier Treatment')
                
                sns.histplot(clean_df[outlier_col], kde=True, ax=ax[1])
                ax[1].set_title(f'Distribution After Outlier Treatment')
                plt.tight_layout()
                st.pyplot(fig)
        
        with col2:
            st.write("After cleaning:")
            st.write(f"Shape: {clean_df.shape}")
            st.write(f"Missing values: {clean_df.isnull().sum().sum()}")
            st.write(f"Duplicates: {clean_df.duplicated().sum()}")
        
        # Replace the original dataframe with the cleaned one
        df = clean_df.copy()
        
        # Option to download cleaned data
        st.markdown('<p class="section-header">Download Cleaned Data</p>', unsafe_allow_html=True)
        
        @st.cache_data
        def convert_df_to_csv(df):
            return df.to_csv(index=False).encode('utf-8')
        
        csv = convert_df_to_csv(df)
        st.download_button(
            label="Download cleaned data as CSV",
            data=csv,
            file_name="cleaned_dataset.csv",
            mime="text/csv",
        )

# Feature Engineering
with tab3:
    if 'df' in locals():
        st.markdown('<p class="section-header">Feature Engineering</p>', unsafe_allow_html=True)
        
        # Feature engineering options
        engineering_options = st.multiselect(
            "Select feature engineering techniques to apply:",
            ["Encode categorical variables", 
             "Create interaction features", 
             "Generate lag features",
             "Create rolling statistics",
             "Feature scaling",
             "Create polynomial features",
             "Feature selection"],
            default=["Encode categorical variables", "Feature scaling"]
        )
        
        # Copy the dataframe
        fe_df = df.copy()
        
        # Track what features we're creating
        created_features = []
        
        # Encode categorical variables
        if "Encode categorical variables" in engineering_options:
            st.markdown("### Categorical Encoding")
            
            categorical_cols = fe_df.select_dtypes(include=['object']).columns.tolist()
            encoding_method = st.radio(
                "Select encoding method:",
                ["Label Encoding", "One-Hot Encoding", "Both (based on cardinality)"]
            )
            
            if encoding_method == "Label Encoding":
                # Label encoding for all categorical columns
                le = LabelEncoder()
                for col in categorical_cols:
                    fe_df[f'{col}_Label'] = le.fit_transform(fe_df[col])
                    created_features.append(f'{col}_Label')
                st.write("✓ Applied Label Encoding to all categorical features")
                
            elif encoding_method == "One-Hot Encoding":
                # One-hot encoding for all categorical columns
                cols_to_encode = st.multiselect(
                    "Select columns to one-hot encode:",
                    categorical_cols,
                    default=categorical_cols
                )
                if cols_to_encode:
                    fe_df = pd.get_dummies(fe_df, columns=cols_to_encode, prefix=cols_to_encode, drop_first=True)
                    for col in cols_to_encode:
                        created_features.extend([c for c in fe_df.columns if c.startswith(f"{col}_")])
                    st.write(f"✓ Applied One-Hot Encoding to selected features")
            
            else:  # Both based on cardinality
                for col in categorical_cols:
                    if fe_df[col].nunique() > 10:  # High cardinality - use label encoding
                        le = LabelEncoder()
                        fe_df[f'{col}_Label'] = le.fit_transform(fe_df[col])
                        created_features.append(f'{col}_Label')
                        st.write(f"✓ Applied Label Encoding to {col} (high cardinality: {fe_df[col].nunique()} unique values)")
                    else:  # Low cardinality - use one-hot encoding
                        temp_df = pd.get_dummies(fe_df[col], prefix=col, drop_first=True)
                        for new_col in temp_df.columns:
                            fe_df[new_col] = temp_df[new_col]
                            created_features.append(new_col)
                        st.write(f"✓ Applied One-Hot Encoding to {col} (low cardinality: {fe_df[col].nunique()} unique values)")
        
        # Create interaction features
        if "Create interaction features" in engineering_options:
            st.markdown("### Interaction Features")
            
            numeric_cols = fe_df.select_dtypes(include=['float64', 'int64']).columns.tolist()
            
            col1, col2 = st.columns(2)
            with col1:
                interaction_feature1 = st.selectbox(
                    "Select first feature for interaction:",
                    numeric_cols,
                    index=0 if numeric_cols else 0
                )
            
            with col2:
                remaining_cols = [col for col in numeric_cols if col != interaction_feature1]
                interaction_feature2 = st.selectbox(
                    "Select second feature for interaction:",
                    remaining_cols,
                    index=0 if remaining_cols else 0
                )
            
            if interaction_feature1 and interaction_feature2:
                interaction_name = f"{interaction_feature1}_{interaction_feature2}_Interaction"
                fe_df[interaction_name] = fe_df[interaction_feature1] * fe_df[interaction_feature2]
                created_features.append(interaction_name)
                st.write(f"✓ Created interaction feature: {interaction_name}")
                
                # Visualize interaction correlation with target
                if "Daily_Sales_Quantity" in fe_df.columns:
                    corr = fe_df[[interaction_name, "Daily_Sales_Quantity"]].corr().iloc[0, 1]
                    st.write(f"Correlation with target: {corr:.4f}")
        
        # Generate lag features
        if "Generate lag features" in engineering_options:
            st.markdown("### Lag Features")
            
            if "Date" in fe_df.columns:
                # Ensure the dataframe is sorted by date
                fe_df = fe_df.sort_values(by="Date")
                
                lag_periods = st.multiselect(
                    "Select lag periods to create:",
                    [1, 3, 7, 14, 30],
                    default=[1, 7]
                )
                
                group_by_col = None
                if "Infrastructure_Machineries" in fe_df.columns or "Customer_Id" in fe_df.columns:
                    group_options = []
                    if "Infrastructure_Machineries" in fe_df.columns:
                        group_options.append("Infrastructure_Machineries")
                    if "Customer_Id" in fe_df.columns:
                        group_options.append("Customer_Id")
                    
                    group_by_col = st.selectbox(
                        "Group lag features by:",
                        ["None"] + group_options,
                        index=1 if group_options else 0
                    )
                
                target_col = "Daily_Sales_Quantity" if "Daily_Sales_Quantity" in fe_df.columns else st.selectbox(
                    "Select column to create lag features for:",
                    fe_df.select_dtypes(include=['float64', 'int64']).columns.tolist()
                )
                
                for lag in lag_periods:
                    if group_by_col and group_by_col != "None":
                        fe_df[f'{target_col}_Lag_{lag}'] = fe_df.groupby(group_by_col)[target_col].shift(lag)
                    else:
                        fe_df[f'{target_col}_Lag_{lag}'] = fe_df[target_col].shift(lag)
                    created_features.append(f'{target_col}_Lag_{lag}')
                
                st.write(f"✓ Created lag features for {target_col}")
                
                # Fill NaN values created by lagging
                fe_df.fillna(fe_df.median(), inplace=True)
            else:
                st.warning("Lag features require a Date column in the dataset.")
        
        # Create rolling statistics
        if "Create rolling statistics" in engineering_options:
            st.markdown("### Rolling Statistics")
            
            if "Date" in fe_df.columns:
                # Ensure the dataframe is sorted by date
                fe_df = fe_df.sort_values(by="Date")
                
                window_sizes = st.multiselect(
                    "Select window sizes for rolling statistics:",
                    [3, 7, 14, 30],
                    default=[7]
                )
                
                rolling_stats = st.multiselect(
                    "Select statistics to calculate:",
                    ["Mean", "Std", "Min", "Max"],
                    default=["Mean"]
                )
                
                target_col = "Daily_Sales_Quantity" if "Daily_Sales_Quantity" in fe_df.columns else st.selectbox(
                    "Select column to create rolling statistics for:",
                    fe_df.select_dtypes(include=['float64', 'int64']).columns.tolist(),
                    key="rolling_target"
                )
                
                group_by_col = None
                if "Infrastructure_Machineries" in fe_df.columns or "Customer_Id" in fe_df.columns:
                    group_options = []
                    if "Infrastructure_Machineries" in fe_df.columns:
                        group_options.append("Infrastructure_Machineries")
                    if "Customer_Id" in fe_df.columns:
                        group_options.append("Customer_Id")
                    
                    group_by_col = st.selectbox(
                        "Group rolling statistics by:",
                        ["None"] + group_options,
                        index=1 if group_options else 0,
                        key="rolling_group"
                    )
                
                for window in window_sizes:
                    for stat in rolling_stats:
                        col_name = f'{target_col}_Rolling_{stat}_{window}'
                        
                        if group_by_col and group_by_col != "None":
                            if stat == "Mean":
                                fe_df[col_name] = fe_df.groupby(group_by_col)[target_col].transform(
                                    lambda x: x.rolling(window=window, min_periods=1).mean())
                            elif stat == "Std":
                                fe_df[col_name] = fe_df.groupby(group_by_col)[target_col].transform(
                                    lambda x: x.rolling(window=window, min_periods=1).std())
                            elif stat == "Min":
                                fe_df[col_name] = fe_df.groupby(group_by_col)[target_col].transform(
                                    lambda x: x.rolling(window=window, min_periods=1).min())
                            elif stat == "Max":
                                fe_df[col_name] = fe_df.groupby(group_by_col)[target_col].transform(
                                    lambda x: x.rolling(window=window, min_periods=1).max())
                        else:
                            if stat == "Mean":
                                fe_df[col_name] = fe_df[target_col].rolling(window=window, min_periods=1).mean()
                            elif stat == "Std":
                                fe_df[col_name] = fe_df[target_col].rolling(window=window, min_periods=1).std()
                            elif stat == "Min":
                                fe_df[col_name] = fe_df[target_col].rolling(window=window, min_periods=1).min()
                            elif stat == "Max":
                                fe_df[col_name] = fe_df[target_col].rolling(window=window, min_periods=1).max()
                        
                        created_features.append(col_name)
                
                st.write(f"✓ Created rolling statistics for {target_col}")
                
                # Fill NaN values created by rolling windows
                fe_df.fillna(fe_df.median(), inplace=True)
            else:
                st.warning("Rolling statistics require a Date column in the dataset.")
        
        # Feature scaling
        if "Feature scaling" in engineering_options:
            st.markdown("### Feature Scaling")
            
            scaling_method = st.radio(
                "Select scaling method:",
                ["StandardScaler", "RobustScaler (less sensitive to outliers)"]
            )
            
            numeric_cols = fe_df.select_dtypes(include=['float64', 'int64']).columns.tolist()
            
            # Exclude target variable and date columns from scaling
            cols_to_exclude = []
            if "Daily_Sales_Quantity" in numeric_cols:
                cols_to_exclude.append("Daily_Sales_Quantity")
            
            cols_to_scale = [col for col in numeric_cols if col not in cols_to_exclude 
                             and not col.endswith('_Label') and not col.startswith('is_')]
            
            if scaling_method == "StandardScaler":
                scaler = StandardScaler()
                scaled_features = scaler.fit_transform(fe_df[cols_to_scale])
                
                for i, col in enumerate(cols_to_scale):
                    fe_df[f'{col}_scaled'] = scaled_features[:, i]
                    created_features.append(f'{col}_scaled')
                
                st.write(f"✓ Applied StandardScaler to {len(cols_to_scale)} features")
            
            else:  # RobustScaler
                robust_scaler = RobustScaler()
                robust_scaled = robust_scaler.fit_transform(fe_df[cols_to_scale])
                
                for i, col in enumerate(cols_to_scale):
                    fe_df[f'{col}_robust'] = robust_scaled[:, i]
                    created_features.append(f'{col}_robust')
                
                st.write(f"✓ Applied RobustScaler to {len(cols_to_scale)} features")
        
        # Create polynomial features
        if "Create polynomial features" in engineering_options:
            st.markdown("### Polynomial Features")
            
            numeric_cols = fe_df.select_dtypes(include=['float64', 'int64']).columns.tolist()
            
            poly_cols = st.multiselect(
                "Select features for polynomial transformation (max 3):",
                numeric_cols,
                default=[]
            )
            
            if poly_cols and len(poly_cols) <= 3:
                degree = st.slider("Polynomial degree", 2, 3, 2)
                
                poly = PolynomialFeatures(degree=degree, include_bias=False)
                poly_features = poly.fit_transform(fe_df[poly_cols])
                
                # Get feature names
                feature_names = poly.get_feature_names_out(poly_cols)
                
                # Add polynomial features to dataframe
                for i, name in enumerate(feature_names):
                    if name not in poly_cols:  # Skip original features
                        fe_df[f'poly_{name}'] = poly_features[:, i]
                        created_features.append(f'poly_{name}')
                
                st.write(f"✓ Created {len(feature_names) - len(poly_cols)} polynomial features of degree {degree}")
            elif len(poly_cols) > 3:
                st.warning("Please select at most 3 features to avoid creating too many polynomial terms.")
        
        # Feature selection
        if "Feature selection" in engineering_options:
            st.markdown("### Feature Selection")
            
            if "Daily_Sales_Quantity" in fe_df.columns:
                target_col = "Daily_Sales_Quantity"
                
                # Get all numeric features except the target
                numeric_cols = fe_df.select_dtypes(include=['float64', 'int64']).columns.tolist()
                features = [col for col in numeric_cols if col != target_col]
                
                selection_method = st.radio(
                    "Select feature selection method:",
                    ["Correlation-based", "XGBoost feature importance", "Random Forest feature importance"]
                )
                
                if selection_method == "Correlation-based":
                    corr_threshold = st.slider("Correlation threshold", 0.0, 1.0, 0.1, 0.05)
                    
                    # Calculate correlation with target
                    correlations = []
                    for feature in features:
                        correlations = []
                    for feature in features:
                        corr = fe_df[[feature, target_col]].corr().iloc[0, 1]
                        correlations.append((feature, abs(corr)))
                    
                    # Sort features by absolute correlation
                    correlations.sort(key=lambda x: x[1], reverse=True)
                    
                    # Select features above threshold
                    selected_features = [feature for feature, corr in correlations if corr >= corr_threshold]
                    
                    # Display correlation chart for top features
                    top_n = min(20, len(selected_features))
                    top_features = selected_features[:top_n]
                    
                    if top_features:
                        fig, ax = plt.subplots(figsize=(10, 8))
                        corr_values = [corr for feature, corr in correlations if feature in top_features]
                        sns.barplot(x=corr_values, y=top_features, ax=ax)
                        plt.title(f'Top {top_n} Features by Correlation with {target_col}')
                        plt.xlabel('Absolute Correlation')
                        st.pyplot(fig)
                    
                    st.write(f"✓ Selected {len(selected_features)} features based on correlation")
                
                elif selection_method == "XGBoost feature importance":
                    # Create a sample of the data for faster computation
                    sample_size = min(5000, len(fe_df))
                    sample_df = fe_df.sample(n=sample_size, random_state=42)
                    
                    # Prepare data
                    X = sample_df[features]
                    y = sample_df[target_col]
                    
                    # Train XGBoost model
                    with st.spinner("Training XGBoost model for feature importance..."):
                        xgb_model = XGBRegressor(random_state=42)
                        xgb_model.fit(X, y)
                    
                    # Get feature importance
                    importance = xgb_model.feature_importances_
                    
                    # Create DataFrame for visualization
                    feature_importance = pd.DataFrame({
                        'Feature': features,
                        'Importance': importance
                    }).sort_values(by='Importance', ascending=False)
                    
                    # Select top features
                    top_n = st.slider("Number of top features to select", 5, min(50, len(features)), 15)
                    selected_features = feature_importance['Feature'].head(top_n).tolist()
                    
                    # Visualize feature importance
                    fig, ax = plt.subplots(figsize=(10, 8))
                    sns.barplot(x='Importance', y='Feature', data=feature_importance.head(min(20, top_n)), ax=ax)
                    plt.title('XGBoost Feature Importance')
                    st.pyplot(fig)
                    
                    st.write(f"✓ Selected {len(selected_features)} top features based on XGBoost importance")
                
                else:  # Random Forest feature importance
                    # Create a sample of the data for faster computation
                    sample_size = min(5000, len(fe_df))
                    sample_df = fe_df.sample(n=sample_size, random_state=42)
                    
                    # Prepare data
                    X = sample_df[features]
                    y = sample_df[target_col]
                    
                    # Train Random Forest model
                    with st.spinner("Training Random Forest model for feature importance..."):
                        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
                        rf_model.fit(X, y)
                    
                    # Get feature importance
                    importance = rf_model.feature_importances_
                    
                    # Create DataFrame for visualization
                    feature_importance = pd.DataFrame({
                        'Feature': features,
                        'Importance': importance
                    }).sort_values(by='Importance', ascending=False)
                    
                    # Select top features
                    importance_threshold = st.slider("Importance threshold", 0.0, 0.1, 0.01, 0.005)
                    selected_features = feature_importance[feature_importance['Importance'] >= importance_threshold]['Feature'].tolist()
                    
                    # Visualize feature importance
                    fig, ax = plt.subplots(figsize=(10, 8))
                    sns.barplot(x='Importance', y='Feature', data=feature_importance.head(20), ax=ax)
                    plt.title('Random Forest Feature Importance')
                    st.pyplot(fig)
                    
                    st.write(f"✓ Selected {len(selected_features)} features based on Random Forest importance")
                
                # Update feature list
                fe_df = fe_df[[target_col] + selected_features].copy()
                st.write("Dataset updated with selected features only")
            else:
                st.warning("Feature selection requires a target column (Daily_Sales_Quantity) in the dataset.")
        
        # Display feature engineering results
        if created_features:
            st.markdown('<p class="section-header">Feature Engineering Results</p>', unsafe_allow_html=True)
            st.write(f"Created {len(created_features)} new features")
            
            # Show sample of new dataframe
            st.write("Sample of engineered dataset:")
            st.dataframe(fe_df.head())
            
            # Update the original dataframe
            df = fe_df.copy()
            
            # Option to download engineered data
            @st.cache_data
            def convert_df_to_csv(df):
                return df.to_csv(index=False).encode('utf-8')
            
            csv = convert_df_to_csv(df)
            st.download_button(
                label="Download engineered data as CSV",
                data=csv,
                file_name="engineered_dataset.csv",
                mime="text/csv",
            )

# Model Training
with tab4:
    if 'df' in locals():
        st.markdown('<p class="section-header">Model Training</p>', unsafe_allow_html=True)
        
        if "Daily_Sales_Quantity" in df.columns:
            target_col = "Daily_Sales_Quantity"
            
            # Select features
            all_cols = df.columns.tolist()
            feature_cols = [col for col in all_cols if col != target_col and not pd.api.types.is_datetime64_any_dtype(df[col])]
            
            selected_features = st.multiselect(
                "Select features for model training:",
                feature_cols,
                default=feature_cols[:min(10, len(feature_cols))]
            )
            
            if selected_features:
                # Split data into train and test sets
                st.markdown("### Train-Test Split")
                test_size = st.slider("Test set size (%)", 10, 40, 20) / 100
                
                # Check if date column exists for time-based split
                time_based_split = False
                if "Date" in df.columns:
                    time_based_split = st.checkbox("Use time-based split", value=True)
                
                X = df[selected_features]
                y = df[target_col]
                
                if time_based_split:
                    split_date = df["Date"].quantile(1 - test_size)
                    train_mask = df["Date"] <= split_date
                    X_train, X_test = X[train_mask], X[~train_mask]
                    y_train, y_test = y[train_mask], y[~train_mask]
                    st.write(f"Split date: {split_date.date()}")
                    st.write(f"Train set: {sum(train_mask)} samples, Test set: {sum(~train_mask)} samples")
                else:
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
                    st.write(f"Train set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")
                
                # Model selection
                st.markdown("### Model Selection")
                model_type = st.selectbox(
                    "Select model type:",
                    ["XGBoost Regressor", "Random Forest Regressor"]
                )
                
                # Model training
                st.markdown("### Model Training")
                
                if model_type == "XGBoost Regressor":
                    # Hyperparameter tuning
                    tune_hyperparams = st.checkbox("Tune hyperparameters", value=False)
                    
                    if tune_hyperparams:
                        with st.spinner("Tuning XGBoost hyperparameters... This may take a while."):
                            param_grid = {
                                'n_estimators': [50, 100],
                                'max_depth': [3, 5, 7],
                                'learning_rate': [0.01, 0.1, 0.3],
                                'subsample': [0.8, 1.0],
                                'colsample_bytree': [0.8, 1.0]
                            }
                            
                            xgb_model = XGBRegressor(objective='reg:squarederror', random_state=42)
                            
                            # Use cross-validation with smaller parameter grid
                            grid_search = GridSearchCV(
                                estimator=xgb_model,
                                param_grid=param_grid,
                                cv=3,
                                scoring='neg_mean_squared_error',
                                n_jobs=-1
                            )
                            
                            grid_search.fit(X_train, y_train)
                            best_params = grid_search.best_params_
                            
                            st.write("Best hyperparameters:")
                            st.write(best_params)
                            
                            # Train model with best params
                            model = XGBRegressor(objective='reg:squarederror', random_state=42, **best_params)
                    else:
                        # Default parameters
                        model = XGBRegressor(
                            n_estimators=100,
                            max_depth=5,
                            learning_rate=0.1,
                            subsample=0.8,
                            colsample_bytree=0.8,
                            objective='reg:squarederror',
                            random_state=42
                        )
                
                else:  # Random Forest
                    # Hyperparameter options
                    n_estimators = st.slider("Number of trees", 50, 500, 100)
                    max_depth = st.slider("Maximum depth", 3, 20, 10)
                    min_samples_split = st.slider("Minimum samples to split", 2, 10, 2)
                    
                    model = RandomForestRegressor(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        min_samples_split=min_samples_split,
                        random_state=42
                    )
                
                # Train the model
                with st.spinner("Training model..."):
                    # Ensure X_train has only numeric types and no missing values
                     X_train = X_train.select_dtypes(include=[np.number]).copy()
                     X_test = X_test[X_train.columns].copy()  # ensure same columns as train
                     X_train.fillna(0, inplace=True)
                     X_test.fillna(0, inplace=True)

                     model.fit(X_train, y_train)
                
                st.success("Model training completed!")
                
                # Model evaluation
                st.markdown("### Model Evaluation")
                
                # Make predictions
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)
                
                # Calculate metrics
                train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
                test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
                
                train_mae = mean_absolute_error(y_train, y_train_pred)
                test_mae = mean_absolute_error(y_test, y_test_pred)
                
                train_r2 = r2_score(y_train, y_train_pred)
                test_r2 = r2_score(y_test, y_test_pred)
                
                train_explained_var = explained_variance_score(y_train, y_train_pred)
                test_explained_var = explained_variance_score(y_test, y_test_pred)
                
                # Display metrics
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Training Set Metrics**")
                    st.markdown(f'<div class="metric-card">RMSE: {train_rmse:.4f}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="metric-card">MAE: {train_mae:.4f}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="metric-card">R²: {train_r2:.4f}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="metric-card">Explained Variance: {train_explained_var:.4f}</div>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown("**Test Set Metrics**")
                    st.markdown(f'<div class="metric-card">RMSE: {test_rmse:.4f}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="metric-card">MAE: {test_mae:.4f}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="metric-card">R²: {test_r2:.4f}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="metric-card">Explained Variance: {test_explained_var:.4f}</div>', unsafe_allow_html=True)
                
                # Visualizations
                st.markdown("### Prediction Visualizations")
                
                # Actual vs Predicted
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
                
                # Training set
                ax1.scatter(y_train, y_train_pred, alpha=0.5)
                ax1.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--')
                ax1.set_xlabel('Actual')
                ax1.set_ylabel('Predicted')
                ax1.set_title('Training Set: Actual vs Predicted')
                
                # Test set
                ax2.scatter(y_test, y_test_pred, alpha=0.5)
                ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
                ax2.set_xlabel('Actual')
                ax2.set_ylabel('Predicted')
                ax2.set_title('Test Set: Actual vs Predicted')
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Residual plot
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
                
                # Training set residuals
                residuals_train = y_train - y_train_pred
                ax1.scatter(y_train_pred, residuals_train, alpha=0.5)
                ax1.axhline(y=0, color='r', linestyle='--')
                ax1.set_xlabel('Predicted')
                ax1.set_ylabel('Residuals')
                ax1.set_title('Training Set: Residual Plot')
                
                # Test set residuals
                residuals_test = y_test - y_test_pred
                ax2.scatter(y_test_pred, residuals_test, alpha=0.5)
                ax2.axhline(y=0, color='r', linestyle='--')
                ax2.set_xlabel('Predicted')
                ax2.set_ylabel('Residuals')
                ax2.set_title('Test Set: Residual Plot')
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Feature importance
                if model_type == "XGBoost Regressor":
                    importance = model.feature_importances_
                    indices = np.argsort(importance)[-15:]  # Top 15 features
                    
                    fig, ax = plt.subplots(figsize=(10, 8))
                    plt.barh(range(len(indices)), importance[indices])
                    plt.yticks(range(len(indices)), [selected_features[i] for i in indices])
                    plt.xlabel('Feature Importance')
                    plt.title('XGBoost Feature Importance')
                    plt.tight_layout()
                    st.pyplot(fig)
                
                else:  # Random Forest
                    importance = model.feature_importances_
                    indices = np.argsort(importance)[-15:]  # Top 15 features
                    
                    fig, ax = plt.subplots(figsize=(10, 8))
                    plt.barh(range(len(indices)), importance[indices])
                    plt.yticks(range(len(indices)), [selected_features[i] for i in indices])
                    plt.xlabel('Feature Importance')
                    plt.title('Random Forest Feature Importance')
                    plt.tight_layout()
                    st.pyplot(fig)
                
                # Cross-validation
                st.markdown("### Cross-Validation")
                cv_option = st.checkbox("Perform cross-validation", value=False)
                
                if cv_option:
                    with st.spinner("Performing cross-validation..."):
                        n_folds = st.slider("Number of folds", 3, 10, 5)
                        
                        if time_based_split and "Date" in df.columns:
                            # Time-series cross-validation
                            tscv = KFold(n_splits=n_folds, shuffle=False)
                            df_sorted = df.sort_values(by="Date")
                            X_sorted = df_sorted[selected_features]
                            y_sorted = df_sorted[target_col]
                            
                            cv_scores = cross_val_score(model, X_sorted, y_sorted, 
                                                      cv=tscv, scoring='neg_mean_squared_error')
                        else:
                            cv_scores = cross_val_score(model, X, y, 
                                                      cv=n_folds, scoring='neg_mean_squared_error')
                        
                        cv_rmse = np.sqrt(-cv_scores)
                        
                        st.write(f"Cross-Validation RMSE: {cv_rmse.mean():.4f} ± {cv_rmse.std():.4f}")
                        
                        # Display individual fold results
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.bar(range(1, len(cv_rmse) + 1), cv_rmse)
                        ax.axhline(y=cv_rmse.mean(), color='r', linestyle='--')
                        plt.xlabel('Fold')
                        plt.ylabel('RMSE')
                        plt.title('Cross-Validation Results by Fold')
                        st.pyplot(fig)
                
                # Save the model and important variables
                st.session_state.model = model
                st.session_state.selected_features = selected_features
                st.session_state.X_train = X_train
                st.session_state.X_test = X_test
                st.session_state.y_train = y_train
                st.session_state.y_test = y_test
                st.session_state.model_type = model_type
            else:
                st.warning("Please select at least one feature for model training.")
        else:
            st.warning("This dataset does not have a target column (Daily_Sales_Quantity) for prediction.")
            
# Predictions & Analysis
with tab5:
    if 'df' in locals():
        st.markdown('<p class="section-header">Predictions & Analysis</p>', unsafe_allow_html=True)
        
        # Check if model exists in session state
        if 'model' in st.session_state:
            model = st.session_state.model
            selected_features = st.session_state.selected_features
            X_train = st.session_state.X_train
            X_test = st.session_state.X_test
            y_train = st.session_state.y_train
            y_test = st.session_state.y_test
            model_type = st.session_state.model_type
            
            st.success(f"Loaded trained {model_type} model")
            
            # Make predictions on the test set
            y_test_pred = model.predict(X_test)
            
            # Display sample predictions
            st.markdown("### Sample Predictions")
            sample_df = pd.DataFrame({
                'Actual': y_test.values[:20],
                'Predicted': y_test_pred[:20],
                'Difference': y_test.values[:20] - y_test_pred[:20],
                'Percent Error': np.abs((y_test.values[:20] - y_test_pred[:20]) / y_test.values[:20] * 100)
            })
            st.dataframe(sample_df.style.format({
                'Actual': '{:.2f}',
                'Predicted': '{:.2f}',
                'Difference': '{:.2f}',
                'Percent Error': '{:.2f}%'
            }))
            
            # Prediction error distribution
            st.markdown("### Prediction Error Analysis")
            errors = y_test - y_test_pred
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # Error histogram
            sns.histplot(errors, kde=True, ax=ax1)
            ax1.set_title('Error Distribution')
            ax1.set_xlabel('Error')
            
            # Percent error histogram
            percent_errors = (errors / y_test) * 100
            sns.histplot(percent_errors, kde=True, ax=ax2)
            ax2.set_title('Percent Error Distribution')
            ax2.set_xlabel('Percent Error (%)')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Advanced analysis
            st.markdown("### Advanced Analysis")
            
            analysis_options = st.multiselect(
                "Select analysis to perform:",
                ["Error by Prediction Range", "Time Series Analysis", "Feature Relationship Analysis"],
                default=["Error by Prediction Range"]
            )
            
            if "Error by Prediction Range" in analysis_options:
                # Group predictions into bins and analyze error
                bins = 10
                y_test_array = np.array(y_test)
                pred_bins = pd.cut(y_test_pred, bins=bins)
                
                bin_data = pd.DataFrame({
                    'Actual': y_test_array,
                    'Predicted': y_test_pred,
                    'Error': y_test_array - y_test_pred,
                    'AbsError': np.abs(y_test_array - y_test_pred),
                    'PctError': np.abs((y_test_array - y_test_pred) / y_test_array) * 100,
                    'PredBin': pred_bins
                })
                
                bin_summary = bin_data.groupby('PredBin').agg({
                    'Actual': 'count',
                    'Error': 'mean',
                    'AbsError': 'mean',
                    'PctError': 'mean'
                }).reset_index()
                bin_summary.columns = ['Prediction Range', 'Count', 'Mean Error', 'Mean Abs Error', 'Mean % Error']
                
                st.write("Error Analysis by Prediction Range:")
                st.dataframe(bin_summary.style.format({
                    'Mean Error': '{:.2f}',
                    'Mean Abs Error': '{:.2f}',
                    'Mean % Error': '{:.2f}%'
                }))
                
                # Visualize error by bin
                fig, ax = plt.subplots(figsize=(12, 6))
                bin_labels = [str(b) for b in bin_summary['Prediction Range']]
                ax.bar(range(len(bin_labels)), bin_summary['Mean Abs Error'])
                ax.set_xticks(range(len(bin_labels)))
                ax.set_xticklabels(bin_labels, rotation=45, ha='right')
                ax.set_title('Mean Absolute Error by Prediction Range')
                ax.set_xlabel('Prediction Range')
                ax.set_ylabel('Mean Absolute Error')
                plt.tight_layout()
                st.pyplot(fig)
            
            if "Time Series Analysis" in analysis_options and "Date" in df.columns:
                # Time-based error analysis
                st.write("### Time Series Error Analysis")
                
                # Get dates corresponding to test set
                if 'train_mask' in locals():
                    test_dates = df.loc[~train_mask, "Date"]
                else:
                    # If no train_mask, we need another way to get test dates
                    st.warning("Time series analysis is only available when using time-based splitting.")
                    test_dates = None
                
                if test_dates is not None:
                    time_error_df = pd.DataFrame({
                        'Date': test_dates.reset_index(drop=True),
                        'Actual': y_test.reset_index(drop=True),
                        'Predicted': y_test_pred,
                        'Error': y_test.reset_index(drop=True) - y_test_pred,
                        'AbsError': np.abs(y_test.reset_index(drop=True) - y_test_pred)
                    }).sort_values('Date')
                    
                    # Plot actual vs predicted over time
                    fig, ax = plt.subplots(figsize=(12, 6))
                    ax.plot(time_error_df['Date'], time_error_df['Actual'], label='Actual')
                    ax.plot(time_error_df['Date'], time_error_df['Predicted'], label='Predicted')
                    ax.set_title('Actual vs Predicted Over Time')
                    ax.set_xlabel('Date')
                    ax.set_ylabel('Sales Quantity')
                    ax.legend()
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Error over time
                    fig, ax = plt.subplots(figsize=(12, 6))
                    ax.plot(time_error_df['Date'], time_error_df['Error'])
                    ax.axhline(y=0, color='r', linestyle='--')
                    ax.set_title('Prediction Error Over Time')
                    ax.set_xlabel('Date')
                    ax.set_ylabel('Error')
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Group errors by time periods
                    time_error_df['Month'] = time_error_df['Date'].dt.month
                    time_error_df['DayOfWeek'] = time_error_df['Date'].dt.dayofweek
                    
                    # Error by month
                    monthly_error = time_error_df.groupby('Month').agg({
                        'AbsError': 'mean',
                        'Actual': 'count'
                    }).reset_index()
                    monthly_error.columns = ['Month', 'Mean Abs Error', 'Count']
                    
                    # Error by day of week
                    dow_error = time_error_df.groupby('DayOfWeek').agg({
                        'AbsError': 'mean',
                        'Actual': 'count'
                    }).reset_index()
                    dow_error.columns = ['Day of Week', 'Mean Abs Error', 'Count']
                    dow_error['Day of Week'] = dow_error['Day of Week'].map({
                        0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 
                        3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'
                    })
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("Mean Absolute Error by Month:")
                        st.dataframe(monthly_error.style.format({'Mean Abs Error': '{:.2f}'}))
                    
                    with col2:
                        st.write("Mean Absolute Error by Day of Week:")
                        st.dataframe(dow_error.style.format({'Mean Abs Error': '{:.2f}'}))
            
            if "Feature Relationship Analysis" in analysis_options:
                st.write("### Feature Relationship Analysis")
                
                # Select features to analyze
                features_to_analyze = st.multiselect(
                    "Select up to 3 features to analyze:",
                    selected_features,
                    default=selected_features[:min(3, len(selected_features))]
                )
                
                if features_to_analyze and len(features_to_analyze) <= 3:
                    # Create a combined dataframe for analysis
                    analysis_df = pd.DataFrame({
                        'Actual': y_test.reset_index(drop=True),
                        'Predicted': y_test_pred,
                        'Error': y_test.reset_index(drop=True) - y_test_pred,
                        'AbsError': np.abs(y_test.reset_index(drop=True) - y_test_pred)
                    })
                    
                    for feature in features_to_analyze:
                        analysis_df[feature] = X_test[feature].reset_index(drop=True)
                    
                    # Plot error vs feature
                    fig, axes = plt.subplots(len(features_to_analyze), 1, figsize=(10, 5*len(features_to_analyze)))
                    if len(features_to_analyze) == 1:
                        axes = [axes]
                    
                    for i, feature in enumerate(features_to_analyze):
                        axes[i].scatter(analysis_df[feature], analysis_df['AbsError'], alpha=0.5)
                        axes[i].set_title(f'Absolute Error vs {feature}')
                        axes[i].set_xlabel(feature)
                        axes[i].set_ylabel('Absolute Error')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Correlation analysis
                    corr_matrix = analysis_df[features_to_analyze + ['AbsError']].corr()
                    
                    fig, ax = plt.subplots(figsize=(10, 8))
                    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
                    ax.set_title('Correlation between Features and Absolute Error')
                    st.pyplot(fig)
                    
                    # Feature interaction analysis if multiple features selected
                    if len(features_to_analyze) >= 2:
                        st.write("### Feature Interaction Analysis")
                        
                        feature1 = features_to_analyze[0]
                        feature2 = features_to_analyze[1]

                        fig, ax = plt.subplots(figsize=(10, 6))
                        scatter = ax.scatter(
                            analysis_df[feature1],
                            analysis_df[feature2],
                            c=analysis_df['AbsError'],
                            cmap='coolwarm',
                            edgecolor='k',
                            alpha=0.7
                        )
                        ax.set_xlabel(feature1)
                        ax.set_ylabel(feature2)
                        ax.set_title(f'{feature1} vs {feature2} Colored by Absolute Error')
                        cbar = plt.colorbar(scatter)
                        cbar.set_label('Absolute Error')
                        st.pyplot(fig)
