"""
WeatherService — Fetches weather data from Open-Meteo (free, no API key required)
and engineers disease-relevant features for the fusion model.

Key features used in sugarcane disease research:
  - Leaf wetness duration → Promotes fungal/bacterial spread
  - Humidity spikes > 80% → High risk for smut, rust, blight
  - Temperature range 25-35°C combined with high humidity → Optimal for most pathogens
  - Consecutive rainy days → Waterborne disease risk
"""
import httpx
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any
import statistics


OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"


class WeatherService:
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=10.0)

    async def get_features(self, lat: float, lon: float) -> Dict[str, Any]:
        """
        Fetch 14-day weather history + 7-day forecast.
        Engineer disease-relevant features.
        """
        raw = await self._fetch_raw(lat, lon)
        features = self._engineer_features(raw)
        return features

    async def _fetch_raw(self, lat: float, lon: float) -> Dict:
        """Fetch hourly weather from Open-Meteo (free, no key needed)."""
        past_days = 14
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": [
                "temperature_2m",
                "relative_humidity_2m",
                "precipitation",
                "wind_speed_10m",
                "vapour_pressure_deficit",
            ],
            "daily": [
                "temperature_2m_max",
                "temperature_2m_min",
                "precipitation_sum",
                "et0_fao_evapotranspiration",
            ],
            "past_days": past_days,
            "forecast_days": 7,
            "timezone": "Asia/Bangkok",
        }

        response = await self.client.get(OPEN_METEO_URL, params=params)
        response.raise_for_status()
        return response.json()

    def _engineer_features(self, raw: Dict) -> Dict[str, Any]:
        """
        Transform raw weather into ML-ready features.
        All features are grounded in plant pathology research.
        """
        hourly = raw.get("hourly", {})
        daily = raw.get("daily", {})

        temps = hourly.get("temperature_2m", [])
        humidity = hourly.get("relative_humidity_2m", [])
        precip_hourly = hourly.get("precipitation", [])
        vpd = hourly.get("vapour_pressure_deficit", [])

        daily_precip = daily.get("precipitation_sum", [])
        daily_max_temp = daily.get("temperature_2m_max", [])
        daily_min_temp = daily.get("temperature_2m_min", [])

        # --- Core statistics (past 14 days) ---
        past_temps = [t for t in temps if t is not None]
        past_humidity = [h for h in humidity if h is not None]
        past_precip = [p for p in precip_hourly if p is not None]

        avg_temp = statistics.mean(past_temps) if past_temps else 0
        avg_humidity = statistics.mean(past_humidity) if past_humidity else 0
        total_precip = sum(past_precip)

        # --- Disease-specific derived features ---

        # Hours where humidity > 80% (leaf wetness proxy)
        high_humidity_hours = sum(1 for h in past_humidity if h > 80)

        # Consecutive rainy days (>1mm)
        rainy_days = [1 if (p or 0) > 1 else 0 for p in daily_precip[:14]]
        max_consecutive_rain = self._max_consecutive(rainy_days)

        # Temperature in optimal pathogen range (25-35°C)
        optimal_pathogen_hours = sum(1 for t in past_temps if 25 <= t <= 35)

        # Diurnal temperature range (high range = stress, lower immunity)
        diurnal_ranges = []
        for mx, mn in zip(daily_max_temp[:14], daily_min_temp[:14]):
            if mx is not None and mn is not None:
                diurnal_ranges.append(mx - mn)
        avg_diurnal_range = statistics.mean(diurnal_ranges) if diurnal_ranges else 0

        # VPD mean (low VPD → high disease pressure for fungal diseases)
        clean_vpd = [v for v in vpd if v is not None]
        avg_vpd = statistics.mean(clean_vpd) if clean_vpd else 0

        # Composite risk index (weighted, 0-100)
        humidity_score = min(avg_humidity / 100, 1.0) * 30
        rain_score = min(max_consecutive_rain / 7, 1.0) * 25
        temp_score = min(optimal_pathogen_hours / (14 * 24), 1.0) * 25
        vpd_score = max(0, (1 - avg_vpd / 3)) * 20  # low VPD = high risk
        weather_risk_index = humidity_score + rain_score + temp_score + vpd_score

        # 7-day forecast risk (for risk_forecast output)
        forecast_precip = daily_precip[14:] if len(daily_precip) > 14 else []
        forecast_rainy_days = sum(1 for p in forecast_precip if (p or 0) > 1)

        return {
            # Core stats
            "avg_temp_14d": round(avg_temp, 1),
            "avg_humidity_14d": round(avg_humidity, 1),
            "total_precip_14d": round(total_precip, 1),

            # Derived disease features
            "high_humidity_hours": high_humidity_hours,
            "max_consecutive_rain_days": max_consecutive_rain,
            "optimal_pathogen_hours": optimal_pathogen_hours,
            "avg_diurnal_range": round(avg_diurnal_range, 1),
            "avg_vpd": round(avg_vpd, 2),

            # Risk index
            "weather_risk_index": round(weather_risk_index, 1),

            # Forecast summary
            "forecast_rainy_days_7d": forecast_rainy_days,

            # Human-readable weather summary
            "weather_summary": self._summarize(avg_temp, avg_humidity, total_precip, weather_risk_index),
        }

    def _max_consecutive(self, binary_list: list) -> int:
        """Find the longest consecutive run of 1s."""
        max_run = current = 0
        for val in binary_list:
            if val:
                current += 1
                max_run = max(max_run, current)
            else:
                current = 0
        return max_run

    def _summarize(self, temp: float, humidity: float, precip: float, risk: float) -> str:
        if risk >= 70:
            level = "สูงมาก"
        elif risk >= 50:
            level = "สูง"
        elif risk >= 30:
            level = "ปานกลาง"
        else:
            level = "ต่ำ"

        return (
            f"อุณหภูมิเฉลี่ย {temp:.1f}°C | ความชื้น {humidity:.0f}% | "
            f"ปริมาณฝน {precip:.0f}mm ใน 14 วัน | ความเสี่ยงสภาพอากาศ: {level}"
        )

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.client.aclose()
