from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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

# Note: More endpoints can be added here that utilize Datathon.py or xg.py for actual data analysis and predictions.
