from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import io

app = FastAPI(title="SaleSense API", description="API for Sales Predictor Data")

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Welcome to SaleSense API. The backend is running successfully!"}

@app.get("/api/status")
def get_status():
    return {"status": "ok", "version": "1.0.0"}

@app.post("/api/upload")
async def upload_dataset(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")
    
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Calculate some basic statistics to return to the frontend
        total_rows = len(df)
        total_cols = len(df.columns)
        columns = df.columns.tolist()
        
        # Extract time series data if Date and Daily_Sales_Quantity exist
        chart_data = []
        total_sales = 0
        if "Date" in df.columns and "Daily_Sales_Quantity" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df = df.dropna(subset=["Date"])
            # Group by date and sum
            daily_sales = df.groupby(df["Date"].dt.date)["Daily_Sales_Quantity"].sum().reset_index()
            daily_sales = daily_sales.sort_values("Date")
            
            # Format for frontend (e.g., take last 30 days for better visualization)
            recent_sales = daily_sales.tail(30)
            for _, row in recent_sales.iterrows():
                chart_data.append({
                    "date": str(row["Date"]),
                    "sales": float(row["Daily_Sales_Quantity"])
                })
            
            total_sales = float(df["Daily_Sales_Quantity"].sum())
        
        return {
            "status": "success",
            "filename": file.filename,
            "data_summary": {
                "total_rows": total_rows,
                "total_cols": total_cols,
                "columns": columns,
                "total_sales": total_sales,
                "chart_data": chart_data,
                "preview": df.head(5).to_dict(orient="records")
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")
