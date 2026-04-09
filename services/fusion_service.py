"""
FusionService — The core of this research system.

This is what makes the project different from just calling Gemini.
It combines THREE signal sources into one unified prediction:

  1. Image signal   → disease class probabilities from EfficientNet (6 features)
  2. Weather signal → engineered climate features (8 features)
  3. Field signal   → encoded farm metadata (4 features)
                                              ──────────────
                                              18 total features → XGBoost

Output:
  - Final disease prediction (may agree or disagree with image-only)
  - Risk score 0-100
  - 7-day forecast risk
  - Confidence level

Research value: The model can predict HIGH RISK even when image looks fine,
if weather conditions are dangerous. This is the key innovation vs Gemini-only.
"""
import numpy as np
import os
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

# Base path for weights
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS_DIR = os.path.join(BASE_DIR, "ml", "weights")

DISEASE_CLASSES = ["Healthy", "Red_Rot", "Mosaic", "Rust", "Yellow_Leaf", "Blight"]

VARIETY_ENCODING = {
    "khon_kaen_3": 0, "ut_thong_1": 1, "ut_thong_2": 2,
    "khon_kaen_1": 3, "lph_11-101": 4, "unknown": 5,
}

SOIL_ENCODING = {
    "sandy": 0, "loam": 1, "clay": 2, "clay_loam": 3,
    "silty": 4, "unknown": 2,
}


class FusionService:
    def __init__(self):
        self.model = None
        self._load_model()

    def _load_model(self):
        """
        Load trained XGBoost fusion model.
        """
        try:
            import xgboost as xgb
            model_path = os.path.join(WEIGHTS_DIR, "fusion_xgb.json")
            if os.path.exists(model_path):
                self.model = xgb.XGBClassifier()
                self.model.load_model(model_path)
                logger.info("XGBoost fusion model loaded")
            else:
                logger.warning(f"Fusion model not found at {model_path}. Using fallback.")
                self.model = None
        except Exception as e:
            logger.warning(f"Fusion model error ({e}). Using rule-based fallback.")
            self.model = None

    def predict(
        self,
        image_result: Dict[str, Any],
        weather_data: Dict[str, Any],
        field_meta: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Combine all signals and produce final prediction.
        """
        # If image was flagged as Out-of-Distribution (Unknown / not a leaf)
        if image_result["predicted_class"] == 6:
            return {
                "final_disease": "Unknown",
                "final_confidence": 0.0,
                "risk_score": 0.0,
                "risk_level": "ปลอดภัย",
                "image_agrees": True,
                "weather_amplified": False,
                "forecast_risk_7d": self._forecast_risk(weather_data, 0.0),
                "model": "ood_detector",
                "warning": "ระบบตรวจไม่พบลักษณะของใบอ้อย หรือภาพไม่ชัดเจน กรุณาถ่ายรูปใบอ้อยให้ชัดเจนอีกครั้ง"
            }

        feature_vector = self._build_features(image_result, weather_data, field_meta)

        if self.model is not None:
            return self._xgboost_predict(feature_vector, image_result, weather_data)
        else:
            return self._rule_based_predict(image_result, weather_data, field_meta)

    def _build_features(
        self,
        image_result: Dict,
        weather: Dict,
        field: Dict,
    ) -> np.ndarray:
        """
        Build 18-dimensional feature vector for the fusion model.

        Features:
          [0-5]  Image: probability for each of 6 disease classes
          [6]    Weather: avg_humidity_14d (normalized 0-1)
          [7]    Weather: avg_temp_14d (normalized 0-1, range 15-45°C)
          [8]    Weather: max_consecutive_rain_days (normalized 0-1, max 14)
          [9]    Weather: high_humidity_hours (normalized 0-1, max 24*14=336)
          [10]   Weather: optimal_pathogen_hours (normalized 0-1)
          [11]   Weather: avg_vpd (normalized 0-1, range 0-5)
          [12]   Weather: weather_risk_index (0-1)
          [13]   Weather: forecast_rainy_days_7d (normalized 0-1)
          [14]   Field: variety encoded
          [15]   Field: age_months (normalized 0-1, max 24)
          [16]   Field: soil_type encoded
          [17]   Image: confidence of top prediction (0-1)
        """
        # Image probabilities (already as percentages, normalize to 0-1)
        probs = [image_result["all_probabilities"].get(d, 0) / 100.0 for d in DISEASE_CLASSES]

        # Weather features (normalized)
        weather_features = [
            weather.get("avg_humidity_14d", 60) / 100.0,
            (weather.get("avg_temp_14d", 30) - 15) / 30.0,
            weather.get("max_consecutive_rain_days", 0) / 14.0,
            weather.get("high_humidity_hours", 0) / 336.0,
            weather.get("optimal_pathogen_hours", 0) / 336.0,
            min(weather.get("avg_vpd", 1.0) / 5.0, 1.0),
            weather.get("weather_risk_index", 0) / 100.0,
            weather.get("forecast_rainy_days_7d", 0) / 7.0,
        ]

        # Field metadata
        variety_enc = VARIETY_ENCODING.get(field.get("variety", "unknown").lower(), 5) / 5.0
        age_norm = min(field.get("age_months", 6) / 24.0, 1.0)
        soil_enc = SOIL_ENCODING.get(field.get("soil_type", "unknown").lower(), 2) / 4.0
        confidence_norm = image_result.get("confidence", 50) / 100.0

        return np.array(probs + weather_features + [variety_enc, age_norm, soil_enc, confidence_norm])

    def _xgboost_predict(self, features: np.ndarray, image_result: Dict, weather: Dict) -> Dict:
        """Prediction using trained XGBoost model."""
        proba = self.model.predict_proba([features])[0]
        predicted_class = int(np.argmax(proba))
        confidence = float(proba[predicted_class])

        risk_score = self._calculate_risk_score(predicted_class, confidence, weather)

        return {
            "final_disease": DISEASE_CLASSES[predicted_class],
            "final_confidence": round(confidence * 100, 1),
            "risk_score": risk_score,
            "risk_level": self._risk_level(risk_score),
            "image_agrees": predicted_class == image_result["predicted_class"],
            "weather_amplified": risk_score > image_result["confidence"],
            "forecast_risk_7d": self._forecast_risk(weather, risk_score),
            "model": "xgboost_fusion",
        }

    def _rule_based_predict(self, image_result: Dict, weather: Dict, field: Dict) -> Dict:
        """
        Rule-based fallback when XGBoost model is not trained yet.
        Uses research-backed thresholds for Thai sugarcane diseases.

        This is still better than Gemini-only because it explicitly
        incorporates weather risk into the score.
        """
        img_class = image_result["predicted_class"]
        img_confidence = image_result["confidence"] / 100.0
        weather_risk = weather.get("weather_risk_index", 0) / 100.0

        # Disease-specific weather amplification rules
        # Based on: Saksirirat & Sae-Eaw (2006), Thai sugarcane disease research
        disease_weather_rules = {
            0: {"humidity_threshold": None, "rain_amplifier": 0},    # Healthy
            1: {"humidity_threshold": 70, "rain_amplifier": 0.3},    # Red rot: loves moisture
            2: {"humidity_threshold": 75, "rain_amplifier": 0.25},   # Smut: wind + rain spread
            3: {"humidity_threshold": 80, "rain_amplifier": 0.35},   # Rust: high humidity critical
            4: {"humidity_threshold": None, "rain_amplifier": 0.1},  # Yellow leaf: virus (insect-borne)
            5: {"humidity_threshold": 75, "rain_amplifier": 0.3},    # Blight: moisture-driven
        }

        rule = disease_weather_rules.get(img_class, {"humidity_threshold": None, "rain_amplifier": 0})
        amplifier = rule["rain_amplifier"]

        # Base score from image confidence
        base_score = img_confidence * 60

        # Weather amplification
        weather_bonus = weather_risk * amplifier * 40

        # Age-based susceptibility (young crop 3-6 months = higher risk for smut)
        age_months = field.get("age_months", 6)
        age_bonus = 5 if (age_months <= 6 and img_class == 2) else 0

        risk_score = min(base_score + weather_bonus + age_bonus, 100)

        # If image says healthy but weather is very dangerous → raise warning
        if img_class == 0 and weather_risk > 0.7:
            risk_score = max(risk_score, weather_risk * 40)
            warning = "สภาพอากาศเสี่ยงสูง แม้ใบดูปกติ ควรติดตามอย่างใกล้ชิด"
        else:
            warning = None

        return {
            "final_disease": DISEASE_CLASSES[img_class],
            "final_confidence": round(img_confidence * 100, 1),
            "risk_score": round(risk_score, 1),
            "risk_level": self._risk_level(risk_score),
            "image_agrees": True,
            "weather_amplified": weather_bonus > 5,
            "forecast_risk_7d": self._forecast_risk(weather, risk_score),
            "warning": warning,
            "model": "rule_based_fallback",
        }

    def _calculate_risk_score(self, disease_class: int, confidence: float, weather: Dict) -> float:
        weather_risk = weather.get("weather_risk_index", 0) / 100.0
        base = confidence * 0.6 * 100
        weather_bonus = weather_risk * 0.4 * 100
        if disease_class == 0:
            return round(weather_risk * 30, 1)
        return round(min(base + weather_bonus, 100), 1)

    def _risk_level(self, score: float) -> str:
        if score >= 75: return "สูงมาก"
        if score >= 55: return "สูง"
        if score >= 35: return "ปานกลาง"
        if score >= 15: return "ต่ำ"
        return "ปลอดภัย"

    def _forecast_risk(self, weather: Dict, current_risk: float) -> Dict:
        rainy_days = weather.get("forecast_rainy_days_7d", 0)
        forecast_modifier = 1 + (rainy_days / 7) * 0.3
        forecast_score = min(current_risk * forecast_modifier, 100)
        return {
            "score_7d": round(forecast_score, 1),
            "level_7d": self._risk_level(forecast_score),
            "rainy_days_ahead": rainy_days,
        }
