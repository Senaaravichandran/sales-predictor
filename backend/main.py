from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import io
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import joblib

app = FastAPI(title="SaleSense Enterprise API", description="Generic ML Platform API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = "data"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
TEMP_DATA_FILE = os.path.join(DATA_DIR, "temp_dataset.csv")

class TrainRequest(BaseModel):
    target_column: str
    task_type: str = "auto" # 'regression' or 'classification'

@app.get("/")
def read_root():
    return {"message": "Enterprise ML Backend Running."}

@app.get("/api/status")
def get_status():
    return {"status": "ok", "version": "2.0.0"}

@app.post("/api/upload")
async def upload_dataset(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")
    
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Save to temp file for training later
        df.to_csv(TEMP_DATA_FILE, index=False)
        
        # Determine column types
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Generate dynamic summary
        summary = []
        for col in df.columns:
            col_type = "numeric" if col in numeric_cols else "categorical"
            missing = int(df[col].isnull().sum())
            unique = int(df[col].nunique())
            summary.append({
                "column": col,
                "type": col_type,
                "missing": missing,
                "unique": unique
            })
            
        # Extract data for a generic chart (e.g. taking first numeric column for distribution)
        chart_data = []
        if len(numeric_cols) > 0:
            target_chart_col = numeric_cols[0]
            if len(numeric_cols) > 1 and "Date" not in numeric_cols[0]:
                target_chart_col = numeric_cols[1] # Try to avoid 'id' or 'index' columns
            
            # Create a simple histogram-like distribution
            counts, bins = np.histogram(df[target_chart_col].dropna(), bins=10)
            for i in range(len(counts)):
                chart_data.append({
                    "name": f"{bins[i]:.1f}-{bins[i+1]:.1f}",
                    "value": int(counts[i])
                })
        
        return {
            "status": "success",
            "filename": file.filename,
            "data_summary": {
                "total_rows": len(df),
                "total_cols": len(df.columns),
                "columns": df.columns.tolist(),
                "numeric_cols": numeric_cols,
                "categorical_cols": categorical_cols,
                "column_stats": summary,
                "chart_data": chart_data,
                "chart_title": f"Distribution of {numeric_cols[0] if numeric_cols else 'Data'}",
                "preview": df.head(10).to_dict(orient="records")
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@app.post("/api/train")
async def train_model(request: TrainRequest):
    if not os.path.exists(TEMP_DATA_FILE):
        raise HTTPException(status_code=400, detail="No dataset uploaded yet.")
    
    try:
        df = pd.read_csv(TEMP_DATA_FILE)
        if request.target_column not in df.columns:
            raise HTTPException(status_code=400, detail="Target column not found in dataset.")
            
        # Drop rows where target is missing
        df = df.dropna(subset=[request.target_column])
        
        y = df[request.target_column]
        X = df.drop(columns=[request.target_column])
        
        # Basic Preprocessing
        # 1. Fill missing values
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        categorical_cols = X.select_dtypes(exclude=[np.number]).columns
        
        for col in numeric_cols:
            X[col] = X[col].fillna(X[col].median())
        for col in categorical_cols:
            X[col] = X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else "Unknown")
            
        # 2. Encode categorical variables
        encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            encoders[col] = le
            
        # Determine Task Type
        is_classification = False
        if request.task_type == "classification" or (y.dtype == 'object') or (y.nunique() < 15):
            is_classification = True
            # encode y if categorical
            if y.dtype == 'object' or str(y.dtype) == 'category':
                le_y = LabelEncoder()
                y = le_y.fit_transform(y.astype(str))
                
        # Split Data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train Model
        metrics = {}
        if is_classification:
            model = RandomForestClassifier(n_estimators=50, random_state=42)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            metrics["accuracy"] = accuracy_score(y_test, preds)
            task = "Classification"
        else:
            model = RandomForestRegressor(n_estimators=50, random_state=42)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            metrics["r2_score"] = r2_score(y_test, preds)
            metrics["rmse"] = np.sqrt(mean_squared_error(y_test, preds))
            task = "Regression"
            
        # Feature Importance
        importance = model.feature_importances_
        feature_importance = [{"feature": f, "importance": float(imp)} for f, imp in zip(X.columns, importance)]
        feature_importance = sorted(feature_importance, key=lambda x: x['importance'], reverse=True)[:10]
        
        return {
            "status": "success",
            "task_type": task,
            "metrics": metrics,
            "top_features": feature_importance
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model training failed: {str(e)}")
