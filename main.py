"""
SugarcaneAI — Smart Disease Detection System
FastAPI Backend Entry Point
"""
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from contextlib import asynccontextmanager
import logging

from services.weather_service import WeatherService
from services.image_service import ImageService
from services.fusion_service import FusionService
from services.report_service import ReportService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading ML models...")
    app.state.image_service = ImageService()
    app.state.weather_service = WeatherService()
    app.state.fusion_service = FusionService()
    app.state.report_service = ReportService()
    logger.info("All models loaded.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="SugarcaneAI Disease Detection",
    description="Multimodal AI system combining image analysis + weather data for sugarcane disease prediction",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "SugarcaneAI API is running", "version": "1.0.0"}


@app.post("/analyze")
async def analyze_sugarcane(
    image: UploadFile = File(..., description="Sugarcane leaf image"),
    latitude: float = Form(..., description="Field latitude (e.g. 14.8744)"),
    longitude: float = Form(..., description="Field longitude (e.g. 100.9925)"),
    variety: str = Form(default="unknown", description="Sugarcane variety"),
    age_months: int = Form(default=6, description="Crop age in months"),
    soil_type: str = Form(default="clay", description="Soil type"),
):
    """
    Main analysis endpoint — combines image + weather + field data.
    Returns disease diagnosis, risk score, and management recommendations.
    """
    try:
        image_bytes = await image.read()

        # 0. Dynamic OOD Check via Gemini Vision
        # Make the system "smart" enough to recognize cats, dogs, or irrelevant images 
        # before pushing them through the specialized leaf ML pipeline.
        ood_result = await app.state.report_service.check_ood_dynamic(image_bytes)
        
        if not ood_result.get("is_sugarcane_leaf", True):
            desc = ood_result.get("description", "ภาพนี้ไม่ใช่ภาพใบอ้อยครับ")
            return JSONResponse(content={
                "success": True,
                "is_ood": True,
                "ood_message": desc,
                "image_analysis": {
                    "disease_name_thai": "ภาพไม่ถูกต้อง", 
                    "disease_name": "Not_A_Leaf",
                    "confidence": 0, 
                    "predicted_class": -1,
                    "severity": "none",
                    "detected_leaves": 0,
                    "all_probabilities": {},
                    "bounding_boxes": [],
                    "is_healthy": False
                },
                "weather_features": {},
                "prediction": {
                    "final_disease": "Unknown", 
                    "risk_score": 0, 
                    "risk_level": "ปลอดภัย",
                    "forecast_risk_7d": {"level_7d": "ปลอดภัย", "score_7d": 0, "rainy_days_ahead": 0}
                },
                "report": {
                    "summary": desc,
                    "disease_explanation": "AI ของเราถูกฝึกมาให้วิเคราะห์รอยโรคบนใบอ้อยเท่านั้น",
                    "immediate_actions": [
                        "ถ่ายรูปใบอ้อยใหม่ให้เห็นชัดเจน", 
                        "หลีกเลี่ยงการถ่ายย้อนแสง หรือภาพที่สั่นไหว"
                    ],
                    "prevention_7days": "-",
                    "chemical_options": [],
                    "monitoring_tips": "-",
                    "severity_explanation": "ไม่สามารถคำนวณความเสี่ยงได้เนื่องจากไม่ใช่ภาพใบอ้อย"
                },
            })

        # 1. Process image → detect & classify disease
        image_result = await app.state.image_service.analyze(image_bytes)

        # 2. Fetch weather data for the field location
        weather_data = await app.state.weather_service.get_features(latitude, longitude)

        # 3. Fusion: combine all signals → final prediction
        field_meta = {
            "variety": variety,
            "age_months": age_months,
            "soil_type": soil_type,
        }
        fusion_result = app.state.fusion_service.predict(image_result, weather_data, field_meta)

        # 4. Generate Thai-language report via Gemini
        report = await app.state.report_service.generate(
            image_result, weather_data, fusion_result, field_meta
        )

        return JSONResponse(content={
            "success": True,
            "image_analysis": image_result,
            "weather_features": weather_data,
            "prediction": fusion_result,
            "report": report,
        })

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/weather/preview")
async def preview_weather(lat: float, lon: float):
    """Preview weather features for a given location."""
    weather_service = app.state.weather_service
    data = await weather_service.get_features(lat, lon)
    return data


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
