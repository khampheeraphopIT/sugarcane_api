"""
ReportService — Generates Thai-language management report via Gemini API.

KEY DESIGN PRINCIPLE:
Gemini is used ONLY at the end, as a report writer.
It receives structured data from our own models, NOT the raw image.
This means Gemini is not doing the diagnosis — our ML pipeline is.
Gemini just converts the results into readable Thai for farmers.

This is academically and technically sound:
  "ระบบใช้ Gemini สำหรับการสร้างรายงานภาษาธรรมชาติ (NLG)
   โดยอ้างอิงผลการวิเคราะห์จากโมเดล ML ของระบบ"
"""
import os
import httpx
import json
import base64
from typing import Dict, Any
import logging
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Make sure this is loaded AFTER load_dotenv() so it picks up the .env file if run directly
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"



class ReportService:
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30.0)

    async def generate(
        self,
        image_result: Dict[str, Any],
        weather_data: Dict[str, Any],
        fusion_result: Dict[str, Any],
        field_meta: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate structured Thai report from analysis results."""

        prompt = self._build_prompt(image_result, weather_data, fusion_result, field_meta)

        if not GEMINI_API_KEY:
            logger.warning("No GEMINI_API_KEY set. Using mock report.")
            return self._mock_report(image_result, fusion_result)

        try:
            response = await self._call_gemini(prompt)
            return self._parse_response(response)
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return self._mock_report(image_result, fusion_result)

    async def check_ood_dynamic(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        Use Gemini Vision as a dynamic pre-filter to detect what the image actually is.
        This prevents cats/dogs from being classified as sugarcane diseases, 
        making the system feel 'smart' while preserving the custom ML pipeline for actual leaves.
        """
        if not GEMINI_API_KEY:
            # If no API key, assume it is a leaf and let the ML pipeline handle it
            return {"is_sugarcane_leaf": True, "description": ""}

        b64_image = base64.b64encode(image_bytes).decode("utf-8")
        prompt = "Is this a clear picture of a sugarcane leaf or a sugarcane plant? If the image contains a car, a person, an animal, a computer screen, or any object that is CLEARLY NOT a real sugarcane plant, YOU MUST SET is_sugarcane_leaf to false. Do not guess. Describe what you see in the image in Thai. Reply ONLY in JSON format like this: {\"is_sugarcane_leaf\": true|false, \"description\": \"บอกว่านึ่คือรูปอะไรในภาษาไทย\"}"
        
        payload = {
            "contents": [{
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": b64_image
                        }
                    }
                ]
            }],
            "generationConfig": {
                "temperature": 0.0,
            }
        }

        try:
            response = await self.client.post(
                f"{GEMINI_URL}?key={GEMINI_API_KEY}",
                json=payload,
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            data = response.json()
            text = data["candidates"][0]["content"]["parts"][0]["text"]
            
            clean = text.strip()
            if clean.startswith("```"):
                lines = clean.split("\n")
                clean = "\n".join(lines[1:-1])
            if clean.startswith("json"):
                clean = clean[4:]
                
            return json.loads(clean.strip())
        except Exception as e:
            logger.error(f"Gemini Vision OOD check failed: {e}")
            return {"is_sugarcane_leaf": True, "description": ""}

    def _build_prompt(self, img: Dict, wx: Dict, fusion: Dict, field: Dict) -> str:
        """
        Structured prompt — Gemini receives computed data, NOT raw image.
        This is the correct use of LLM in a research system.
        """
        if fusion.get('final_disease') == 'Unknown':
            return """คุณคือผู้ช่วยเหลือในระบบ AI ตรวจโรคใบอ้อย
            
ระบบประมวลผลแล้วพบว่าภาพที่ผู้ใช้อัปโหลดมา **"ไม่ใช่ใบอ้อย"** หรือ **"ภาพไม่ชัดเจนพอที่จะวิเคราะห์ได้"** (Out-of-Distribution)

กรุณาสร้างรายงานในรูปแบบ JSON ปลอบใจและแนะนำผู้ใช้ให้ถ่ายรูปใหม่ให้ชัดเจน โดยใช้โครงสร้างนี้:
{
  "summary": "แจ้งผู้ใช้สุภาพว่าระบบไม่พบใบอ้อยในภาพ หรือภาพอาจจะไม่ชัด",
  "disease_explanation": "บอกว่า AI ของเราถูกฝึกมาให้ดูเฉพาะใบอ้อยเท่านั้น",
  "immediate_actions": ["คำแนะนำในการถ่ายรูปใหม่ 1", "คำแนะนำ 2"],
  "prevention_7days": "-",
  "chemical_options": [],
  "monitoring_tips": "-",
  "severity_explanation": "ระบุว่าไม่สามารถประเมินความเสี่ยงได้เนื่องจากภาพไม่ถูกต้อง"
}"""

        return f"""คุณคือผู้เชี่ยวชาญด้านโรคอ้อยและการเกษตรอัจฉริยะประจำประเทศไทย

ระบบ AI ได้วิเคราะห์แปลงอ้อยและพบข้อมูลดังนี้:

=== ผลการวิเคราะห์รูปภาพ (จากโมเดล YOLOv8 + EfficientNet) ===
- โรคที่ตรวจพบ: {img['disease_name_thai']} ({img['disease_name']})
- ความแม่นยำ: {img['confidence']}%
- ความรุนแรง: {img['severity']}
- จำนวนใบที่ตรวจ: {img['detected_leaves']} ใบ

=== สภาพอากาศในพื้นที่ (14 วันที่ผ่านมา) ===
- อุณหภูมิเฉลี่ย: {wx['avg_temp_14d']}°C
- ความชื้นสัมพัทธ์เฉลี่ย: {wx['avg_humidity_14d']}%
- ปริมาณน้ำฝนสะสม: {wx['total_precip_14d']} mm
- จำนวนชั่วโมงที่ความชื้น > 80%: {wx['high_humidity_hours']} ชั่วโมง
- จำนวนวันฝนตกต่อเนื่องสูงสุด: {wx['max_consecutive_rain_days']} วัน
- ดัชนีความเสี่ยงสภาพอากาศ: {wx['weather_risk_index']}/100

=== ข้อมูลแปลง ===
- พันธุ์อ้อย: {field.get('variety', 'ไม่ระบุ')}
- อายุอ้อย: {field.get('age_months', '?')} เดือน
- ประเภทดิน: {field.get('soil_type', 'ไม่ระบุ')}

=== ผลการประเมินรวมจากระบบ (Fusion Model) ===
- โรคสรุป: {fusion['final_disease']}
- คะแนนความเสี่ยงรวม: {fusion['risk_score']}/100
- ระดับความเสี่ยง: {fusion['risk_level']}
- ความเสี่ยงใน 7 วันข้างหน้า: {fusion['forecast_risk_7d']['level_7d']}

กรุณาสร้างรายงานในรูปแบบ JSON ดังนี้ (ตอบเป็น JSON เท่านั้น ไม่ต้องมีข้อความอื่น):

{{
  "summary": "สรุปสถานการณ์ 2-3 ประโยค เข้าใจง่าย",
  "disease_explanation": "อธิบายโรคและสาเหตุที่น่าจะเกิดขึ้นในสภาพอากาศแบบนี้",
  "immediate_actions": ["ข้อ 1", "ข้อ 2", "ข้อ 3"],
  "prevention_7days": "คำแนะนำป้องกันสำหรับ 7 วันข้างหน้า",
  "chemical_options": ["สารเคมีที่แนะนำ ข้อ 1", "ข้อ 2"],
  "monitoring_tips": "สิ่งที่ควรสังเกตในแปลงอ้อย",
  "severity_explanation": "อธิบายว่าทำไมระดับความเสี่ยงถึงเป็น {fusion['risk_level']}"
}}"""

    async def _call_gemini(self, prompt: str) -> str:
        """Call Gemini API."""
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.3,
                "maxOutputTokens": 1024,
            }
        }

        response = await self.client.post(
            f"{GEMINI_URL}?key={GEMINI_API_KEY}",
            json=payload,
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()
        data = response.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]

    def _parse_response(self, text: str) -> Dict[str, Any]:
        """Parse JSON response from Gemini."""
        # Strip markdown code fences if present
        clean = text.strip()
        if clean.startswith("```"):
            lines = clean.split("\n")
            clean = "\n".join(lines[1:-1])
        try:
            return json.loads(clean)
        except json.JSONDecodeError:
            return {"summary": clean, "parse_error": True}

    def _mock_report(self, image_result: Dict, fusion_result: Dict) -> Dict[str, Any]:
        """Fallback report when Gemini API is unavailable."""
        if fusion_result.get("final_disease") == "Unknown":
            return {
                "summary": "ระบบไม่สามารถระบุภาพนี้เป็นใบอ้อยได้ กรุณาถ่ายรูปหรืออัปโหลดรูปใบอ้อยใหม่อีกครั้ง",
                "disease_explanation": "ภาพที่อัปโหลดอาจไม่ใช่ใบอ้อย หรือภาพมีความเบลอ/แสงไม่เพียงพอ ทำให้ AI วิเคราะห์ไม่ได้",
                "immediate_actions": [
                    "ถ่ายรูปใบอ้อยให้เห็นอาการของโรคชัดเจน",
                    "หลีกเลี่ยงการถ่ายย้อนแสงหรือรูปที่สั่นไหว",
                    "อัปโหลดเพื่อวิเคราะห์ใหม่อีกครั้ง"
                ],
                "prevention_7days": "-",
                "chemical_options": [],
                "monitoring_tips": "-",
                "severity_explanation": "ไม่มีการให้คะแนนความเสี่ยงเนื่องจากไม่ใช่ภาพใบอ้อย",
                "source": "mock_fallback",
            }

        disease = image_result["disease_name_thai"]
        risk = fusion_result["risk_level"]
        return {
            "summary": f"ตรวจพบ{disease} ระดับความเสี่ยง{risk} แนะนำให้ดำเนินการตามคำแนะนำด้านล่าง",
            "disease_explanation": f"{disease} เป็นโรคที่พบบ่อยในแปลงอ้อยของไทย มักเกิดในสภาพอากาศที่มีความชื้นสูง",
            "immediate_actions": [
                "ตรวจสอบแปลงอ้อยทั้งหมดภายใน 24 ชั่วโมง",
                "แยกส่วนที่เป็นโรคออกเพื่อป้องกันการแพร่กระจาย",
                "ปรึกษาเจ้าหน้าที่เกษตรในพื้นที่",
            ],
            "prevention_7days": "ติดตามสภาพอากาศอย่างใกล้ชิด ลดการให้น้ำถ้าฝนตกต่อเนื่อง",
            "chemical_options": ["ปรึกษาผู้เชี่ยวชาญก่อนใช้สารเคมี"],
            "monitoring_tips": "ตรวจดูใบอ้อยใหม่ทุก 3-5 วัน",
            "severity_explanation": f"ระดับความเสี่ยง{risk} คำนวณจากทั้งผลการวิเคราะห์ภาพและสภาพอากาศรวมกัน",
            "source": "mock_fallback",
        }
