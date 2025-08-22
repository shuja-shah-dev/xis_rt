from fastapi import FastAPI
from app.api.endpoints import mqtt
from config.settings import settings
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title=settings.app_name,
    description="FastAPI application with MQTT client integration",
    version="1.0.0"
)

app.include_router(mqtt.router, prefix="/api/v1")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"],  
)

@app.get("/")
async def root():
    return {
        "message": "Welcome to FastAPI MQTT Application",
        "docs": "/docs",
        "mqtt_status": "/api/v1/mqtt/status"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)