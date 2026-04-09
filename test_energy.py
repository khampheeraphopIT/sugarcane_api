import asyncio
import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from transformers import AutoModelForImageClassification

def test_energy():
    model_path = r"C:\Users\poplo\Desktop\sugarcane_prod\backend\ml\weights\sugarcane_finetuned.pth"
    classifier = AutoModelForImageClassification.from_pretrained(
        "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification",
        num_labels=6,
        ignore_mismatched_sizes=True
    )
    if os.path.exists(model_path):
        classifier.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        classifier.eval()
        print("Loaded finetuned model.")

    try:
        img = Image.open(r"C:\Users\poplo\Desktop\sugarcane_prod\cat.jpg").convert("RGB")
    except:
        print("Cannot find cat.jpg")
        return

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        logits = classifier(tensor).logits
        energy = -torch.logsumexp(logits, dim=1)[0].item()
        probs = torch.softmax(logits, dim=1)[0].tolist()

    predicted_class = int(np.argmax(probs))
    confidence = probs[predicted_class]

    print(f"Cat Image:")
    print(f"Predicted Class ID: {predicted_class}")
    print(f"Confidence: {confidence:.2f}")
    print(f"Energy Score: {energy:.2f}")

if __name__ == "__main__":
    test_energy()
