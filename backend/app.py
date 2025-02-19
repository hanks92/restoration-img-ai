from flask import Flask, request, send_file, send_from_directory, jsonify
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
import io
import sys
import os
from argparse import Namespace  # Pour passer les bons arguments au modèle

# 📌 Définir les chemins de base
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EDSR_PATH = os.path.join(BASE_DIR, "EDSR-PyTorch/src")
FRONTEND_DIR = os.path.join(BASE_DIR, "../frontend")  # 📌 Dossier du frontend

# 📌 Ajouter le chemin du module `EDSR-PyTorch/src` à Python
sys.path.append(EDSR_PATH)

# 📌 Importer le modèle EDSR
from model.edsr import EDSR

app = Flask(__name__, static_folder=FRONTEND_DIR)

# 📌 Définir l'appareil (GPU si disponible, sinon CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 📌 Chemin correct du modèle `.pt`
model_path = os.path.join(BASE_DIR, "EDSR-PyTorch/src/model/EDSR_baseline_x4.pt")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ Le fichier '{model_path}' est introuvable.")

# 📌 Définition des paramètres du modèle
args = Namespace(
    n_resblocks=16,  # Nombre de blocs résiduels
    n_feats=64,      # Nombre de features
    scale=[4],       # Facteur d'upscaling (doit être une liste)
    rgb_range=255,
    n_colors=3,
    res_scale=1.0
)

# 📌 Initialisation du modèle avec les bons paramètres
model = EDSR(args).to(device)

try:
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    print("✅ Modèle EDSR chargé avec succès !")
except Exception as e:
    raise RuntimeError(f"❌ Erreur lors du chargement du modèle : {e}")

model.eval()
model.training = False  # Assurer que le modèle est bien en mode inférence

# 📌 Fonctions de transformation d'image
def preprocess(image):
    """ Convertit une image PIL en tenseur pour le modèle EDSR """
    image = np.array(image).astype(np.float32) / 255.0  # Normalisation 0-1
    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(device)
    print(f"📢 Image prétraitée - Taille: {image.shape}, Max pixel: {image.max()}, Min pixel: {image.min()}")  # Debugging
    return image

def postprocess(tensor):
    """ Convertit un tenseur PyTorch en image PIL """
    tensor = tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()
    tensor = (tensor * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(tensor)

# 📌 Route pour servir la page HTML principale
@app.route("/")
def home():
    return send_from_directory(FRONTEND_DIR, "index.html")

# 📌 Route pour servir les fichiers statiques (JS, CSS)
@app.route("/<path:filename>")
def serve_static(filename):
    return send_from_directory(FRONTEND_DIR, filename)

# 📌 Route pour améliorer la résolution d'une image
@app.route("/enhance", methods=["POST"])
def enhance_image():
    if "image" not in request.files:
        return jsonify({"error": "Aucune image reçue"}), 400
    
    file = request.files["image"]
    image = Image.open(file.stream).convert("RGB")

    # 📢 Debug: Afficher les dimensions de l'image originale
    print(f"📢 Image originale - Taille: {image.size}")

    # 📌 Prétraitement et super-résolution
    input_tensor = preprocess(image)

    with torch.no_grad():
        output_tensor = model(input_tensor)

    # 📢 Debug: Vérifier la taille de sortie
    print(f"📢 Taille sortie modèle: {output_tensor.shape}, Max pixel: {output_tensor.max()}, Min pixel: {output_tensor.min()}")

    # 📌 Vérifier si la sortie est bien 4x plus grande
    original_width, original_height = image.size
    expected_width, expected_height = original_width * 4, original_height * 4

    if output_tensor.shape[-2:] != (expected_height, expected_width):
        print("⚠️ Correction : Redimensionnement forcé avec interpolation bicubique")
        output_tensor = F.interpolate(output_tensor, size=(expected_height, expected_width), mode="bicubic", align_corners=False)

    # 📌 Post-traitement
    sr_image = postprocess(output_tensor)

    # 📢 Debug: Afficher la taille de l'image finale
    print(f"📢 Image améliorée - Taille: {sr_image.size}")

    # 📌 Sauvegarde en mémoire pour l'envoi
    img_io = io.BytesIO()
    sr_image.save(img_io, format="JPEG")
    img_io.seek(0)

    return send_file(img_io, mimetype="image/jpeg", as_attachment=True, download_name="enhanced.jpg")

# 📌 Démarrage du serveur Flask
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
