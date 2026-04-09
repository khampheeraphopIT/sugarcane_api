import asyncio
import os
from dotenv import load_dotenv
from services.report_service import ReportService
import base64

load_dotenv()

async def test_ood():
    service = ReportService()
    # Read a sample image (e.g., the favicon which is surely not a leaf, or cat.jpg if exists)
    try:
        with open("../cat.jpg", "rb") as f:
            img_bytes = f.read()
    except FileNotFoundError:
        print("No cat.jpg found, using favicon.svg instead")
        try:
            with open("../frontend/public/favicon.svg", "rb") as f:
                img_bytes = f.read()
        except:
            print("Cannot find any test image.")
            return

    result = await service.check_ood_dynamic(img_bytes)
    print("OOD Result:", result)

if __name__ == "__main__":
    asyncio.run(test_ood())
