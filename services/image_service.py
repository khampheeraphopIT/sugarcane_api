"""
ImageService — Image analysis pipeline:
  Stage 1 (optional): YOLOv8 — Detect and crop leaf regions
  Stage 2: EfficientNet-B0 — Classify disease from leaf image

EfficientNet gives calibrated class probabilities that feed
directly into the XGBoost fusion model as numeric features.

If YOLOv8 weights are not available, the full image is classified directly.

Supported diseases:
  0: Healthy
  1: Red_Rot (โรคเน่าแดง)
  2: Mosaic (โรคใบด่าง)
  3: Rust (โรคราสนิม)
  4: Yellow_Leaf (โรคใบเหลือง)
  5: Blight (โรคใบไหม้)
"""
import io
import numpy as np
from PIL import Image
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

# Disease class mapping
DISEASE_CLASSES = {
    0: {"name": "Healthy", "thai": "ปกติ (ไม่พบโรค)", "severity": "none"},
    1: {"name": "Red_Rot", "thai": "โรคเน่าแดง", "severity": "high"},
    2: {"name": "Mosaic", "thai": "โรคใบด่าง", "severity": "medium"},
    3: {"name": "Rust", "thai": "โรคราสนิม", "severity": "medium"},
    4: {"name": "Yellow_Leaf", "thai": "โรคใบเหลือง", "severity": "medium"},
    5: {"name": "Blight", "thai": "โรคใบไหม้", "severity": "medium"},
    -1: {"name": "Unknown", "thai": "ไม่ระบุ (ไม่ใช่ใบอ้อย/ภาพไม่ชัด)", "severity": "none"}
}


class ImageService:
    def __init__(self):
        self.yolo_model = None
        self.classifier = None
        self._load_models()

    def _load_models(self):
        """
        Load YOLOv8 + EfficientNet models independently.

        To train your own models:
          YOLOv8: see ml/training/train_yolo.py
          EfficientNet: see ml/training/train_classifier.py

        For demo/development: falls back to mock predictions.
        """
        # --- Load YOLOv8 (leaf detection) independently ---
        try:
            from ultralytics import YOLO
            self.yolo_model = YOLO("./ml/weights/yolo_sugarcane_leaf.pt")
            logger.info("YOLOv8 loaded successfully")
        except Exception as e:
            logger.warning(f"YOLOv8 not available ({e}). Will classify full image instead.")
            self.yolo_model = None

        # --- Load Transformers classifier independently ---
        try:
            import torch
            from transformers import AutoModelForImageClassification
            
            # Create a model structure but load custom weights
            self.classifier = AutoModelForImageClassification.from_pretrained(
                "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification",
                num_labels=len(DISEASE_CLASSES),
                ignore_mismatched_sizes=True
            )
            self.classifier.load_state_dict(torch.load("./ml/weights/sugarcane_finetuned.pth", map_location=torch.device('cpu')))
            self.classifier.eval()
            logger.info("Fine-tuned Transformers classifier loaded successfully")
        except Exception as e:
            logger.warning(f"Fine-tuned classifier not available ({e}). Using mock mode.")
            self.classifier = None

    async def analyze(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        Full image analysis pipeline.
        Returns disease prediction with confidence scores.
        """
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        if self.classifier is not None:
            return self._real_pipeline(image)
        else:
            return self._mock_pipeline(image)

    def _real_pipeline(self, image: Image.Image) -> Dict[str, Any]:
        """Production pipeline with real models."""
        import torch
        import torchvision.transforms as transforms

        # Stage 1: YOLOv8 — detect leaf regions (optional)
        if self.yolo_model is not None:
            results = self.yolo_model(np.array(image))
            detections = results[0].boxes

            if len(detections) == 0:
                crops = [image]
                boxes = []
            else:
                crops = []
                boxes = []
                for box in detections:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    crop = image.crop((x1, y1, x2, y2))
                    crops.append(crop)
                    boxes.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "conf": float(box.conf[0])})
        else:
            # No YOLO — classify the full image directly
            crops = [image]
            boxes = []

        # Stage 2: Transformers — classify each crop
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        all_probs = []
        all_energies = []
        with torch.no_grad():
            for crop in crops:
                tensor = transform(crop).unsqueeze(0)
                logits = self.classifier(tensor).logits
                
                # Energy score calculation (Liu et al., 2020)
                # Lower energy (or higher negative sum) means the model is MORE confident/familiar.
                energy = -torch.logsumexp(logits, dim=1)[0].item()
                all_energies.append(energy)
                
                probs = torch.softmax(logits, dim=1)[0].tolist()
                all_probs.append(probs)

        # Average probabilities and energies across all detected leaves
        avg_probs = np.mean(all_probs, axis=0).tolist()
        predicted_class = int(np.argmax(avg_probs))
        confidence = avg_probs[predicted_class]
        avg_energy = np.mean(all_energies)

        # Energy-based Out-of-Distribution (OOD) Detection
        # If the energy is high (above threshold), the image is considered OOD (e.g. a cat, not a leaf).
        # Based on testing:
        # cat.jpg energy: -0.26
        # A confident leaf usually has energy < -5.0
        # We set threshold to -1.0 to reject non-leaf images while allowing some leaf variation.
        ENERGY_THRESHOLD = -1.0 
        if avg_energy > ENERGY_THRESHOLD:
            logger.warning(f"OOD detected! Energy score: {avg_energy:.2f}")
            predicted_class = -1
            confidence = 0.0

        return self._format_result(predicted_class, confidence, avg_probs, boxes, len(crops))

    def _mock_pipeline(self, image: Image.Image) -> Dict[str, Any]:
        """
        Mock pipeline for development/demo without trained weights.
        Simulates realistic output structure.
        Replace with _real_pipeline() after training your models.
        """
        import random
        random.seed(sum(image.getpixel((100, 100))))  # deterministic per image

        # Simulate disease probabilities
        probs = [random.uniform(0, 0.1) for _ in DISEASE_CLASSES]
        mock_disease = random.choice([0, 1, 2, 3])
        probs[mock_disease] = random.uniform(0.65, 0.92)
        total = sum(probs)
        probs = [p / total for p in probs]
        confidence = probs[mock_disease]

        mock_boxes = [{"x1": 50, "y1": 50, "x2": 300, "y2": 400, "conf": 0.88}]
        return self._format_result(mock_disease, confidence, probs, mock_boxes, 1)

    def _format_result(
        self,
        predicted_class: int,
        confidence: float,
        all_probs: List[float],
        boxes: List[Dict],
        leaf_count: int,
    ) -> Dict[str, Any]:
        disease_info = DISEASE_CLASSES[predicted_class]

        return {
            "predicted_class": predicted_class,
            "disease_name": disease_info["name"],
            "disease_name_thai": disease_info["thai"],
            "severity": disease_info["severity"],
            "confidence": round(confidence * 100, 1),
            "all_probabilities": {
                DISEASE_CLASSES[i]["name"]: round(p * 100, 1)
                for i, p in enumerate(all_probs)
            },
            "detected_leaves": leaf_count,
            "bounding_boxes": boxes,
            "is_healthy": predicted_class == 0,
        }
