import torch
import cv2
import numpy as np
from realesrgan import RealESRGAN
from PIL import Image

# Charger le modèle
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RealESRGAN(device, scale=4)
model.load_weights("RealESRGAN_x4.pth")

def enhance_image(image_path, output_path="output.jpg"):
    """Améliore la résolution d'une image."""
    image = Image.open(image_path).convert("RGB")
    sr_image = model.predict(image)
    sr_image.save(output_path)
    return output_path

if __name__ == "__main__":
    image_path = "test.jpg"  # Remplace par ton image
    output_path = enhance_image(image_path)
    print(f"Image restaurée enregistrée sous {output_path}")
