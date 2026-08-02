import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore, skew, kurtosis, pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler, PowerTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, pacf
import warnings
warnings.filterwarnings("ignore")
import os
from datetime import datetime

# Set styling for plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('viridis')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

# Create output directory for saving plots
output_dir = "C:\\Users\\DELL\\OneDrive\\Desktop\\New folder\\Analysis_Results"
os.makedirs(output_dir, exist_ok=True)

# Load the dataset with error handling
try:
    file_path = "C:\\Users\\DELL\\OneDrive\\Desktop\\New folder\\Datathon Dataset.csv"
    df = pd.read_csv(file_path)
    print(f"Dataset loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns")
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

# Function to save figures
def save_fig(fig, filename):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(output_dir, f"{filename}_{timestamp}.png")
    fig.savefig(filepath, bbox_inches='tight', dpi=300)
    plt.close(fig)
    return filepath

# Data summary
print("\n=== Dataset Summary ===")
print(df.info())
print("\n=== Statistical Summary ===")
print(df.describe(include='all').T)

# Drop unnecessary columns
df.drop(columns=["Unnamed: 0", "Un_Named"], inplace=True, errors='ignore')
print("\n=== Columns after initial cleaning ===")
print(df.columns.tolist())

# Enhanced date handling
print("\n=== Date Handling ===")
df = df[pd.notnull(df["Date"])]  # Remove rows with null dates
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
date_nulls = df["Date"].isnull().sum()
print(f"Rows with invalid dates: {date_nulls}")

if date_nulls > 0:
    # Fill missing dates with previous valid dates or forward fill
    df["Date"] = df["Date"].fillna(method='ffill')
    if df["Date"].isnull().any():
        df["Date"] = df["Date"].fillna(method='bfill')
    
    # If still missing, use the median date
    if df["Date"].isnull().any():
        median_date = pd.to_datetime(df["Date"].dropna().median())
        df["Date"] = df["Date"].fillna(median_date)

# Comprehensive missing value analysis
print("\n=== Missing Value Analysis ===")
missing_data = df.isnull().sum()
missing_percent = (missing_data / len(df)) * 100
missing_summary = pd.DataFrame({'Missing Count': missing_data, 
                               'Missing Percent': missing_percent})
missing_summary = missing_summary[missing_summary['Missing Count'] > 0].sort_values('Missing Percent', ascending=False)
print(missing_summary)

# Advanced missing value handling
for col in df.columns:
    if df[col].isnull().sum() > 0:
        print(f"Handling missing values in {col}")
        if col == 'Infrastructure_Machineries':
            df[col].fillna("Unknown", inplace=True)
        elif df[col].dtype in ['int64', 'float64']:
            # For numeric columns, use median to reduce outlier impact
            df[col].fillna(df[col].median(), inplace=True)
        else:
            # For categorical columns, use mode
            df[col].fillna(df[col].mode().iloc[0], inplace=True)

# Time series feature extraction
print("\n=== Creating Time Features ===")
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Day"] = df["Date"].dt.day
df["Quarter"] = df["Date"].dt.quarter
df["Weekday"] = df["Date"].dt.weekday
df["WeekOfYear"] = df["Date"].dt.isocalendar().week
df["DayOfYear"] = df["Date"].dt.dayofyear
df["IsWeekend"] = df["Weekday"].apply(lambda x: 1 if x >= 5 else 0)
df["IsMonthStart"] = df["Date"].dt.is_month_start.astype(int)
df["IsMonthEnd"] = df["Date"].dt.is_month_end.astype(int)
df["IsQuarterStart"] = df["Date"].dt.is_quarter_start.astype(int)
df["IsQuarterEnd"] = df["Date"].dt.is_quarter_end.astype(int)

# Sort by date for time series analysis
df.sort_values("Date", inplace=True)

# Advanced lag features with multi-level grouping
print("\n=== Creating Advanced Lag Features ===")
# Check if necessary columns exist
if "Infrastructure_Machineries" in df.columns and "Daily_Sales_Quantity" in df.columns:
    # Basic lags
    df["Sales_Lag_1"] = df.groupby("Infrastructure_Machineries")["Daily_Sales_Quantity"].shift(1)
    df["Sales_Lag_7"] = df.groupby("Infrastructure_Machineries")["Daily_Sales_Quantity"].shift(7)
    df["Sales_Lag_30"] = df.groupby("Infrastructure_Machineries")["Daily_Sales_Quantity"].shift(30)
    
    # Rolling statistics
    for window in [7, 14, 30]:
        df[f"Rolling_Mean_{window}"] = df.groupby("Infrastructure_Machineries")["Daily_Sales_Quantity"].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean())
        df[f"Rolling_Std_{window}"] = df.groupby("Infrastructure_Machineries")["Daily_Sales_Quantity"].transform(
            lambda x: x.rolling(window=window, min_periods=1).std())
        df[f"Rolling_Min_{window}"] = df.groupby("Infrastructure_Machineries")["Daily_Sales_Quantity"].transform(
            lambda x: x.rolling(window=window, min_periods=1).min())
        df[f"Rolling_Max_{window}"] = df.groupby("Infrastructure_Machineries")["Daily_Sales_Quantity"].transform(
            lambda x: x.rolling(window=window, min_periods=1).max())
    
    # Expanding statistics
    df["Expanding_Mean"] = df.groupby("Infrastructure_Machineries")["Daily_Sales_Quantity"].transform(
        lambda x: x.expanding().mean())
    df["Expanding_Std"] = df.groupby("Infrastructure_Machineries")["Daily_Sales_Quantity"].transform(
        lambda x: x.expanding().std())
    
    # Month-to-date and Year-to-date
    df["MTD_Sales"] = df.groupby(["Infrastructure_Machineries", "Year", "Month"])["Daily_Sales_Quantity"].transform(
        lambda x: x.expanding().sum())
    df["YTD_Sales"] = df.groupby(["Infrastructure_Machineries", "Year"])["Daily_Sales_Quantity"].transform(
        lambda x: x.expanding().sum())
    
    # Multi-level grouping features
    if "Region" in df.columns:
        df["Region_Machinery_Mean"] = df.groupby(["Region", "Infrastructure_Machineries"])["Daily_Sales_Quantity"].transform('mean')
        df["Region_Machinery_Std"] = df.groupby(["Region", "Infrastructure_Machineries"])["Daily_Sales_Quantity"].transform('std')

# Advanced outlier detection using multiple methods
print("\n=== Advanced Outlier Detection ===")
if "Daily_Sales_Quantity" in df.columns:
    # Method 1: IQR method
    Q1 = df["Daily_Sales_Quantity"].quantile(0.25)
    Q3 = df["Daily_Sales_Quantity"].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    iqr_outliers = df[(df["Daily_Sales_Quantity"] < lower_bound) | (df["Daily_Sales_Quantity"] > upper_bound)]
    print(f"IQR Method Outliers: {len(iqr_outliers)} ({len(iqr_outliers)/len(df)*100:.2f}%)")
    
    # Method 2: Z-score method
    z_scores = zscore(df["Daily_Sales_Quantity"], nan_policy='omit')
    abs_z_scores = np.abs(z_scores)
    z_outliers = df[abs_z_scores > 3]
    print(f"Z-score Method Outliers: {len(z_outliers)} ({len(z_outliers)/len(df)*100:.2f}%)")
    
    # Method 3: Isolation Forest
    X_numeric = df.select_dtypes(include=[np.number]).copy()
    X_numeric = X_numeric.fillna(X_numeric.median())
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    outlier_preds = iso_forest.fit_predict(X_numeric)
    isolation_outliers = df[outlier_preds == -1]
    print(f"Isolation Forest Outliers: {len(isolation_outliers)} ({len(isolation_outliers)/len(df)*100:.2f}%)")
    
    # Store outlier flags
    df["is_outlier_iqr"] = ((df["Daily_Sales_Quantity"] < lower_bound) | (df["Daily_Sales_Quantity"] > upper_bound)).astype(int)
    df["is_outlier_zscore"] = (abs_z_scores > 3).astype(int)
    df["is_outlier_isolation"] = (outlier_preds == -1).astype(int)
    df["outlier_score"] = df["is_outlier_iqr"] + df["is_outlier_zscore"] + df["is_outlier_isolation"]
    
    # Handle outliers with capping/winsorization instead of removal
    df["Daily_Sales_Quantity_Original"] = df["Daily_Sales_Quantity"].copy()
    df["Daily_Sales_Quantity_Capped"] = df["Daily_Sales_Quantity"].clip(lower=lower_bound, upper=upper_bound)

# Enhanced feature encoding
print("\n=== Enhanced Feature Encoding ===")
# Label encoding
label_enc = LabelEncoder()
if "Infrastructure_Machineries" in df.columns:
    df["Machinery_Label"] = label_enc.fit_transform(df["Infrastructure_Machineries"])
    machinery_mapping = dict(zip(label_enc.classes_, label_enc.transform(label_enc.classes_)))
    print(f"Machinery Encoding Map: {machinery_mapping}")

# Smart one-hot encoding for low-cardinality categorical variables
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
for col in categorical_cols:
    if df[col].nunique() < 10:  # Only one-hot encode if low cardinality
        print(f"One-hot encoding {col} with {df[col].nunique()} unique values")
        df = pd.get_dummies(df, columns=[col], prefix=col, drop_first=True)
    else:
        print(f"Label encoding {col} with {df[col].nunique()} unique values")
        df[f"{col}_Label"] = label_enc.fit_transform(df[col])

# Feature transformation for skewed data
print("\n=== Feature Transformation for Skewed Data ===")
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
skewed_features = {}

for col in numeric_cols:
    if col != "Machinery_Label" and not col.endswith("_Label") and not col.startswith("is_outlier"):
        skewness = skew(df[col].dropna())
        skewed_features[col] = skewness
        
        if abs(skewness) > 1:  # Significantly skewed
            print(f"Transforming skewed feature {col} (skewness: {skewness:.2f})")
            # Try log transformation for positive skewed data
            if df[col].min() > 0:
                df[f"{col}_log"] = np.log1p(df[col])
            # Use Box-Cox transformation for more complex distributions
            try:
                pt = PowerTransformer(method='yeo-johnson')
                df[f"{col}_transformed"] = pt.fit_transform(df[[col]])
            except Exception as e:
                print(f"Error in transforming {col}: {e}")

# Advanced Scaling
print("\n=== Advanced Feature Scaling ===")
# Select features to scale
features_to_scale = [col for col in numeric_cols if not col.startswith('is_') and not col.endswith('_Label')]

# Apply multiple scaling methods
# Standard scaling
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[features_to_scale])
scaled_df = pd.DataFrame(scaled_features, columns=[f"{col}_scaled" for col in features_to_scale])
df = pd.concat([df, scaled_df], axis=1)

# Robust scaling (less sensitive to outliers)
robust_scaler = RobustScaler()
robust_scaled = robust_scaler.fit_transform(df[features_to_scale])
robust_df = pd.DataFrame(robust_scaled, columns=[f"{col}_robust" for col in features_to_scale])
df = pd.concat([df, robust_df], axis=1)

# Dimensionality Reduction with PCA
print("\n=== Dimensionality Reduction with PCA ===")
# Select numeric features for PCA
pca_features = [col for col in df.columns if col.endswith('_scaled') and not col.startswith('is_')]
if len(pca_features) > 2:  # Need at least 2 features for PCA
    try:
        # Apply PCA
        pca = PCA(n_components=min(10, len(pca_features)))
        pca_result = pca.fit_transform(df[pca_features])
        
        # Create PCA dataframe
        pca_df = pd.DataFrame(
            data=pca_result,
            columns=[f'PC{i+1}' for i in range(pca_result.shape[1])]
        )
        
        # Append PCA results to original dataframe
        df = pd.concat([df, pca_df], axis=1)
        
        # Explained variance
        print("PCA Explained Variance Ratio:")
        for i, var in enumerate(pca.explained_variance_ratio_):
            print(f"PC{i+1}: {var:.4f} ({var*100:.2f}%)")
        print(f"Cumulative: {sum(pca.explained_variance_ratio_)*100:.2f}%")
        
        # Plot PCA
        fig, ax = plt.subplots(figsize=(10, 8))
        plt.scatter(pca_df['PC1'], pca_df['PC2'], alpha=0.5)
        plt.title('PCA: First Two Principal Components')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        save_fig(fig, "PCA_Plot")
    except Exception as e:
        print(f"Error in PCA: {e}")

# Clustering Analysis
print("\n=== Clustering Analysis ===")
if "Daily_Sales_Quantity" in df.columns and "Infrastructure_Machineries" in df.columns:
    # Select features for clustering
    cluster_features = ['Daily_Sales_Quantity_Capped', 'Machinery_Label']
    if 'Region_Label' in df.columns:
        cluster_features.append('Region_Label')
    
    # Add time-based features if available
    time_features = ['Year', 'Month', 'Quarter', 'Weekday']
    for feat in time_features:
        if feat in df.columns:
            cluster_features.append(feat)
    
    # Scale clustering features
    cluster_data = df[cluster_features].copy()
    cluster_scaler = StandardScaler()
    cluster_data_scaled = cluster_scaler.fit_transform(cluster_data)
    
    # Determine optimal number of clusters
    wcss = []
    max_clusters = min(10, len(df) // 50)  # Don't try too many clusters
    for i in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
        kmeans.fit(cluster_data_scaled)
        wcss.append(kmeans.inertia_)
    
    # Plot elbow curve
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.plot(range(1, max_clusters + 1), wcss, marker='o')
    plt.title('Elbow Method for Optimal k')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.grid(True)
    save_fig(fig, "Elbow_Method")
    
    # Choose optimal number of clusters (this is a simple heuristic)
    try:
        from kneed import KneeLocator
        kl = KneeLocator(range(1, max_clusters + 1), wcss, curve="convex", direction="decreasing")
        optimal_k = kl.elbow
        if optimal_k is None:
            optimal_k = 3  # Default if no clear elbow
    except:
        # Fallback if kneed is not installed
        optimal_k = 3
    
    print(f"Optimal number of clusters: {optimal_k}")
    
    # Apply KMeans with optimal k
    kmeans = KMeans(n_clusters=optimal_k, init='k-means++', max_iter=300, n_init=10, random_state=42)
    df['Cluster'] = kmeans.fit_predict(cluster_data_scaled)
    
    # Analyze clusters
    cluster_analysis = df.groupby('Cluster').agg({
        'Daily_Sales_Quantity': ['mean', 'median', 'min', 'max', 'std', 'count'],
        'Infrastructure_Machineries': lambda x: x.value_counts().index[0] if len(x.value_counts()) > 0 else None
    })
    print("\nCluster Analysis:")
    print(cluster_analysis)
    
    # Visualize clusters
    if len(cluster_features) >= 2:
        fig, ax = plt.subplots(figsize=(12, 8))
        scatter = plt.scatter(
            cluster_data_scaled[:, 0], 
            cluster_data_scaled[:, 1],
            c=df['Cluster'], 
            cmap='viridis', 
            alpha=0.6, 
            s=50
        )
        plt.colorbar(scatter, label='Cluster')
        plt.title('K-Means Clustering Results')
        plt.xlabel(cluster_features[0])
        plt.ylabel(cluster_features[1])
        plt.grid(True, linestyle='--', alpha=0.7)
        save_fig(fig, "Kmeans_Clusters")

# Time Series Analysis
print("\n=== Time Series Analysis ===")
if "Date" in df.columns and "Daily_Sales_Quantity" in df.columns:
    # Aggregate sales by date
    daily_sales = df.groupby("Date")["Daily_Sales_Quantity"].sum().reset_index()
    daily_sales.set_index("Date", inplace=True)
    
    # Check stationarity with ADF test
    adf_result = adfuller(daily_sales["Daily_Sales_Quantity"].dropna())
    print(f"ADF Statistic: {adf_result[0]:.4f}")
    print(f"p-value: {adf_result[1]:.4f}")
    print(f"Critical Values: {adf_result[4]}")
    is_stationary = adf_result[1] < 0.05
    print(f"Series is {'stationary' if is_stationary else 'non-stationary'}")
    
    # Create time series visualization
    fig, axes = plt.subplots(3, 1, figsize=(14, 18))
    
    # Plot original time series
    daily_sales["Daily_Sales_Quantity"].plot(ax=axes[0], title="Daily Sales Over Time", color='blue')
    axes[0].set_ylabel("Sales Quantity")
    axes[0].grid(True)
    
    # Plot ACF and PACF
    acf_values = acf(daily_sales["Daily_Sales_Quantity"].dropna(), nlags=30)
    pacf_values = pacf(daily_sales["Daily_Sales_Quantity"].dropna(), nlags=30, method='ols')
    
    axes[1].stem(range(len(acf_values)), acf_values)
    axes[1].set_title("Autocorrelation Function (ACF)")
    axes[1].axhline(y=0, linestyle='--', color='gray')
    axes[1].axhline(y=-1.96/np.sqrt(len(daily_sales)), linestyle='--', color='red')
    axes[1].axhline(y=1.96/np.sqrt(len(daily_sales)), linestyle='--', color='red')
    axes[1].grid(True)
    
    axes[2].stem(range(len(pacf_values)), pacf_values)
    axes[2].set_title("Partial Autocorrelation Function (PACF)")
    axes[2].axhline(y=0, linestyle='--', color='gray')
    axes[2].axhline(y=-1.96/np.sqrt(len(daily_sales)), linestyle='--', color='red')
    axes[2].axhline(y=1.96/np.sqrt(len(daily_sales)), linestyle='--', color='red')
    axes[2].grid(True)
    
    plt.tight_layout()
    save_fig(fig, "Time_Series_Analysis")
    
    # Try seasonal decomposition if enough data points
    if len(daily_sales) >= 14:  # Need at least 2 weeks of data
        try:
            # Find the appropriate period (7 for weekly, 30 for monthly)
            period = 7 if len(daily_sales) >= 14 else 1
            decomposition = seasonal_decompose(daily_sales["Daily_Sales_Quantity"], period=period, model='additive')
            
            fig, axes = plt.subplots(4, 1, figsize=(14, 16))
            decomposition.observed.plot(ax=axes[0], title="Observed")
            decomposition.trend.plot(ax=axes[1], title="Trend")
            decomposition.seasonal.plot(ax=axes[2], title="Seasonality")
            decomposition.resid.plot(ax=axes[3], title="Residuals")
            plt.tight_layout()
            save_fig(fig, "Seasonal_Decomposition")
        except Exception as e:
            print(f"Error in seasonal decomposition: {e}")

# Advanced Statistical Analysis
print("\n=== Advanced Statistical Analysis ===")
if "Daily_Sales_Quantity" in df.columns:
    stats_df = pd.DataFrame()
    
    # Extract key features for correlation analysis
    key_features = ["Daily_Sales_Quantity"]
    potential_features = ["Political", "Marketing", "Market_Share", "Budget", "Year", "Month", "Quarter", "Weekday"]
    for feat in potential_features:
        if feat in df.columns:
            key_features.append(feat)
    
    # Correlation matrix
    corr_matrix = df[key_features].corr()
    
    # Advanced correlation visualization
    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    heatmap = sns.heatmap(
        corr_matrix, 
        mask=mask,
        cmap='RdBu_r',
        vmax=1.0,
        vmin=-1.0,
        center=0,
        annot=True,
        fmt='.2f',
        square=True,
        linewidths=.5,
        cbar_kws={'shrink': .5}
    )
    plt.title('Correlation Matrix of Key Features')
    save_fig(fig, "Advanced_Correlation_Matrix")
    
    # Find key correlations with sales
    sales_corr = corr_matrix["Daily_Sales_Quantity"].sort_values(ascending=False).drop("Daily_Sales_Quantity")
    print("\nTop correlations with Daily Sales:")
    print(sales_corr)
    
    # Enhanced visualizations
    print("\n=== Enhanced Visualizations ===")
    
    # 1. Sales Distribution with fitted distribution
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.histplot(df["Daily_Sales_Quantity"], kde=True, bins=50, color='g', ax=ax)
    plt.title("Sales Quantity Distribution with KDE", fontsize=14)
    plt.xlabel("Sales Quantity", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    save_fig(fig, "Sales_Distribution_Enhanced")
    
    # 2. Sales by Machinery Type - Enhanced
    if "Infrastructure_Machineries" in df.columns:
        # Sort by total sales
        machinery_sales = df.groupby("Infrastructure_Machineries")["Daily_Sales_Quantity"].sum().sort_values(ascending=False)
        
        fig, ax = plt.subplots(figsize=(16, 10))
        palette = sns.color_palette("viridis", n_colors=len(machinery_sales))
        
        bars = sns.barplot(
            x=machinery_sales.index,
            y=machinery_sales.values,
            palette=palette,
            ax=ax
        )
        
        # Add value labels on top of bars
        for i, v in enumerate(machinery_sales.values):
            ax.text(i, v + 0.1, f"{v:.1f}", ha='center', fontsize=10)
            
        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.title("Total Sales by Machinery Type", fontsize=16)
        plt.xlabel("Machinery Type", fontsize=14)
        plt.ylabel("Total Sales", fontsize=14)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        save_fig(fig, "Sales_By_Machinery_Enhanced")
        
        # 3. Sales Trend by Top Machinery Types
        top_machinery = machinery_sales.index[:5]  # Top 5 machinery types
        
        fig, ax = plt.subplots(figsize=(16, 8))
        
        for machinery in top_machinery:
            machinery_data = df[df["Infrastructure_Machineries"] == machinery]
            monthly_sales = machinery_data.groupby(pd.Grouper(key="Date", freq="M"))["Daily_Sales_Quantity"].sum()
            monthly_sales.plot(label=machinery, marker='o', linestyle='-', linewidth=2, ax=ax)
            
        plt.title("Monthly Sales Trend for Top Machinery Types", fontsize=16)
        plt.xlabel("Date", fontsize=14)
        plt.ylabel("Total Monthly Sales", fontsize=14)
        plt.legend(title="Machinery Type", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        save_fig(fig, "Monthly_Sales_Trend_Top_Machinery")
    
    # 4. Heat Map: Sales by Month and Day of Week
    if all(x in df.columns for x in ["Month", "Weekday", "Daily_Sales_Quantity"]):
        month_day_sales = df.pivot_table(
            values="Daily_Sales_Quantity", 
            index="Month",
            columns="Weekday",
            aggfunc="mean"
        )
        
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(
            month_day_sales,
            cmap="YlGnBu",
            annot=True,
            fmt=".1f",
            linewidths=0.5,
            ax=ax
        )
        
        # Define labels for days and months
        day_labels = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        # Apply labels if they match the axis dimensions
        if ax.get_xticklabels() and len(day_labels) == len(ax.get_xticklabels()):
            ax.set_xticklabels(day_labels)
        if ax.get_yticklabels() and len(month_labels) == len(ax.get_yticklabels()):
            ax.set_yticklabels(month_labels)
            
        plt.title("Average Sales by Month and Day of Week", fontsize=16)
        plt.xlabel("Day of Week", fontsize=14)
        plt.ylabel("Month", fontsize=14)
        plt.tight_layout()
        save_fig(fig, "Sales_Month_Weekday_Heatmap")
    
    # 5. Regional Analysis if available
    if "Region" in df.columns:
        region_sales = df.groupby("Region")["Daily_Sales_Quantity"].agg(['sum', 'mean', 'count']).sort_values('sum', ascending=False)
        
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        
        # Total sales by region
        sns.barplot(
            x=region_sales.index,
            y=region_sales['sum'],
            palette="Blues_d",
            ax=axes[0]
        )
        axes[0].set_title("Total Sales by Region", fontsize=16)
        axes[0].set_xlabel("Region", fontsize=14)
        axes[0].set_ylabel("Total Sales", fontsize=14)
        axes[0].tick_params(axis='x', rotation=45)
        
        # Average sales by region
        sns.barplot(
            x=region_sales.index,
            y=region_sales['mean'],
            palette="Reds_d",
            ax=axes[1]
        )
        axes[1].set_title("Average Sales by Region", fontsize=16)
        axes[1].set_xlabel("Region", fontsize=14)
        axes[1].set_ylabel("Average Sales", fontsize=14)
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        save_fig(fig, "Regional_Sales_Analysis")
        
        # If we have both Region and Machinery data
        if "Infrastructure_Machineries" in df.columns:
            # Top machinery type by region
            top_machinery_by_region = df.groupby(["Region", "Infrastructure_Machineries"])["Daily_Sales_Quantity"].sum().reset_index()
            top_machinery_by_region = top_machinery_by_region.sort_values(["Region", "Daily_Sales_Quantity"], ascending=[True, False])
            top_machinery_by_region = top_machinery_by_region.groupby("Region").first().reset_index()
            
            print("\nTop selling machinery by region:")
            print(top_machinery_by_region[["Region", "Infrastructure_Machineries"]])

# Feature Importance Analysis for Sales
print("\n=== Feature Importance Analysis ===")
if "Daily_Sales_Quantity" in df.columns:
    try:
        from sklearn.ensemble import RandomForestRegressor
        
        # Select features for importance analysis
        features = df.select_dtypes(include=[np.number]).columns.tolist()
        features = [f for f in features if f not in ["Daily_Sales_Quantity", "Daily_Sales_Quantity_Original", "Daily_Sales_Quantity_Capped"]]
        
        # Handle missing values
        X = df[features].fillna(df[features].median())
        y = df["Daily_Sales_Quantity"]
        
        # Train a Random Forest for feature importance
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)
        
        # Get feature importances
        importances = rf.feature_importances_
        feature_importance = pd.DataFrame({'Feature': features, 'Importance': importances})
        feature_importance = feature_importance.sort_values('Importance', ascending=False)
        
        print("\nTop 10 Most Important Features:")
        print(feature_importance.head(10))
        
        # Plot feature importances
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.barplot(
            x='Importance',
            y='Feature',
            data=feature_importance.head(15),
            palette='viridis'
        )
        plt.title("Top 15 Feature Importance for Sales Prediction", fontsize=16)
        plt.xlabel("Importance", fontsize=14)
        plt.ylabel("Feature", fontsize=14)
        plt.tight_layout()
        save_fig(fig, "Feature_Importance_Analysis")
    except Exception as e:
        print(f"Error in feature importance analysis: {e}")

# Save the cleaned and enhanced dataset
output_path = "C:\\Users\\DELL\\OneDrive\\Desktop\\New folder\\Advanced_Cleaned_Datathon_Dataset.csv"

# Select only useful columns for the final dataset
# Remove highly correlated features and intermediate calculations
columns_to_exclude = [col for col in df.columns if '_scaled' in col or '_robust' in col]
final_df = df.drop(columns=columns_to_exclude, errors='ignore')

# Save to CSV
final_df.to_csv(output_path, index=False)

print(f"\nAdvanced data analysis and visualization completed successfully!")
print(f"Enhanced dataset saved to: {output_path}")
print(f"All visualizations saved to: {output_dir}")

# Provide summary of enhancements
print("\n=== Enhancement Summary ===")
print("1. Advanced cleaning with multiple imputation strategies")
print("2. Extensive feature engineering including date-based and lag features")
print("3. Multiple outlier detection methods")
print("4. Clustering analysis with optimal cluster determination")
print("5. Time series analysis with stationarity testing and decomposition")
print("6. Enhanced visualizations with deeper insights")
print("7. Feature importance analysis for sales prediction")
print("8. PCA for dimensionality reduction")
print("9. Regional and product-based analysis")
print("10. Advanced correlation analysis")