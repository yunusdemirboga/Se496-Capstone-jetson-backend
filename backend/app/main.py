from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import auth, detections, base_stations, websocket

app = FastAPI(title="UAV Detection System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(detections.router)
app.include_router(base_stations.router)
app.include_router(websocket.router)

@app.get("/")
def root():
    return {"message": "UAV Detection System API is running"}