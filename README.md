🚀 Sales Predictor
An advanced, interactive Streamlit-based app for predicting daily sales quantity using machine learning and time-series analytics. Built with XGBoost, Random Forest, and a rich suite of data preprocessing and visualization tools.

🔍 Features
📊 Data Exploration: Upload datasets and explore statistics, distributions, and missing values.

🧹 Data Cleaning: Handle duplicates, missing values, outliers, and date conversions.

🔧 Feature Engineering: Encode variables, generate lags, rolling stats, interaction terms, polynomial features, and scaling.

🤖 Model Training: Train XGBoost or Random Forest models with optional hyperparameter tuning.

📈 Prediction & Analysis: Evaluate model performance and visualize error patterns and feature interactions.

🛠️ Tech Stack
Python (Pandas, NumPy, Scikit-learn, XGBoost, Seaborn, Matplotlib)

Streamlit (UI)

Optuna (for hyperparameter tuning)

Joblib (for model saving)

Jupyter-compatible analytics

📦 Setup Instructions
# Clone the repo
git clone https://github.com/Senaaravichandran/sales-predictor.git
cd sales-predictor

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run datast.py
📁 File Structure
datast.py: Main Streamlit app ( Run by typing "streamlit run datast.py" )

Datathon.py: Advanced data preprocessing and exploratory analysis script.

xg.py: XGBoost modeling pipeline with Optuna tuning and feature selection.

🧠 Model Highlights
Supports time-based train/test splitting.

Handles lag-based and group-wise statistical features.

Visualizes feature importance and model diagnostics.

📊 Sample Output
Interactive boxplots and histograms

Feature correlation heatmaps

Residual and prediction plots

Clustering insights (optional)
