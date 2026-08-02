import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBRegressor
import optuna
from optuna.integration import XGBoostPruningCallback
import joblib
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Load the dataset
df = pd.read_csv(r'C:\Users\DELL\OneDrive\Desktop\Data\Datathon Dataset.csv')
print("Dataset shape:", df.shape)
print(df.head())

# Data Cleaning
print("Missing values before cleaning:")
print(df.isnull().sum())

# Handle critical missing values
df.dropna(subset=['Date', 'Infrastructure_Machineries'], inplace=True)

# Fill numeric columns with median (more robust to outliers than mean)
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

# Fill categorical columns with mode
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode().iloc[0])

print("Missing values after cleaning:")
print(df.isnull().sum())

# Remove duplicates and unnecessary columns
df.drop_duplicates(inplace=True)
df.drop(columns=['Unnamed: 0', 'Un_Named'], errors='ignore', inplace=True)

# Convert Date to datetime and extract features
df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d', errors='coerce')
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['DayOfWeek'] = df['Date'].dt.dayofweek
df['IsWeekend'] = df['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)
df['Quarter'] = df['Date'].dt.quarter

# Create lag features for time series aspects
if 'Customer_Id' in df.columns:
    df['Sales_Lag_1'] = df.groupby('Customer_Id')['Daily_Sales_Quantity'].shift(1)
    df['Sales_Lag_7'] = df.groupby('Customer_Id')['Daily_Sales_Quantity'].shift(7)
    df['Rolling_Mean_7'] = df.groupby('Customer_Id')['Daily_Sales_Quantity'].transform(
        lambda x: x.rolling(window=7, min_periods=1).mean())
    df['Rolling_Std_7'] = df.groupby('Customer_Id')['Daily_Sales_Quantity'].transform(
        lambda x: x.rolling(window=7, min_periods=1).std())

# Feature Engineering - Interaction Features
if 'Political' in df.columns and 'Marketing' in df.columns:
    df['Politics_Marketing_Interaction'] = df['Political'] * df['Marketing']

if 'Market_Share' in df.columns and 'Marketing' in df.columns:
    df['Market_Marketing_Interaction'] = df['Market_Share'] * df['Marketing']

# Outlier Detection and Handling
def detect_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers, lower_bound, upper_bound

target_col = 'Daily_Sales_Quantity'
if target_col in df.columns:
    outliers, lower_bound, upper_bound = detect_outliers_iqr(df, target_col)
    print(f"Number of outliers detected: {len(outliers)}")
    print(f"Lower bound: {lower_bound}, Upper bound: {upper_bound}")
    
    # Cap outliers instead of removing them
    df[target_col] = np.where(df[target_col] < lower_bound, lower_bound, df[target_col])
    df[target_col] = np.where(df[target_col] > upper_bound, upper_bound, df[target_col])

# Encode categorical features
cat_features = ['Infrastructure_Machineries', 'Political', 'Marketing', 'Region']
existing_cat_features = [col for col in cat_features if col in df.columns]

# Use both label encoding and one-hot encoding based on cardinality
label_encoded_cols = []
onehot_cols = []

for col in existing_cat_features:
    if df[col].nunique() > 10:  # High cardinality - use label encoding
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoded_cols.append(col)
    else:  # Low cardinality - use one-hot encoding
        onehot_cols.append(col)

# One-hot encode the low cardinality columns
if onehot_cols:
    df = pd.get_dummies(df, columns=onehot_cols, drop_first=True)

# Define features and target
drop_cols = ['Date', 'Customer_Id', target_col]
X = df.drop(columns=[col for col in drop_cols if col in df.columns])
y = df[target_col]

# Keep only numeric columns for model
X = X.select_dtypes(include=[np.number])

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# Train-test split with stratification if possible
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled_df, y, test_size=0.2, random_state=42)

# Optuna hyperparameter optimization
def objective(trial):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000, 100),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
        'random_state': 42
    }
    
    # 5-fold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    model = XGBRegressor(**param)
    
    scores = cross_val_score(
        model, X_train, y_train, 
        cv=kf, 
        scoring='neg_root_mean_squared_error'
    )
    
    # Return the mean negative RMSE
    return scores.mean()

# Run Optuna optimization
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

print('Best trial:')
trial = study.best_trial
print(f'  RMSE: {-trial.value}')
print('  Params: ')
for key, value in trial.params.items():
    print(f'    {key}: {value}')

# Train the model with the best parameters
best_params = study.best_params
best_model = XGBRegressor(**best_params)
best_model.fit(X_train, y_train)

# Feature importance analysis
feature_importance = best_model.feature_importances_
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importance
}).sort_values(by='Importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(importance_df.head(10))

# Select important features
selector = SelectFromModel(best_model, threshold='median', prefit=True)
X_train_selected = selector.transform(X_train)
X_test_selected = selector.transform(X_test)
selected_features = X.columns[selector.get_support()]
print(f"\nSelected {len(selected_features)} features out of {X.shape[1]}")

# Retrain model with selected features
final_model = XGBRegressor(**best_params)
final_model.fit(X_train_selected, y_train)

# Make predictions
y_pred = final_model.predict(X_test_selected)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
evs = explained_variance_score(y_test, y_pred)

print("\nModel Evaluation Metrics:")
print(f"Mean Squared Error: {mse:.4f}")
print(f"Root Mean Squared Error: {rmse:.4f}")
print(f"Mean Absolute Error: {mae:.4f}")
print(f"R² Score: {r2:.4f}")
print(f"Explained Variance Score: {evs:.4f}")

# Plot actual vs predicted
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted Values')
plt.show()

# Plot residuals
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()

# Plot feature importance
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=importance_df.head(15))
plt.title('Top 15 Feature Importance')
plt.tight_layout()
plt.show()

# Save the model and preprocessing objects
joblib.dump(final_model, 'xgboost_final_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(selector, 'feature_selector.pkl')

print("Model and preprocessing objects saved successfully.")