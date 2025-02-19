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
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# 📌 Fonctions de transformation d'image
def preprocess(image):
    image = np.array(image).astype(np.float32) / 255.0
    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(device)
    return image

def postprocess(tensor):
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

    # 📌 Prétraitement et super-résolution
    input_tensor = preprocess(image)
    with torch.no_grad():
        output_tensor = model(input_tensor)

    # 📌 Post-traitement
    sr_image = postprocess(output_tensor)

    # 📌 Sauvegarde en mémoire pour l'envoi
    img_io = io.BytesIO()
    sr_image.save(img_io, format="JPEG")
    img_io.seek(0)

    return send_file(img_io, mimetype="image/jpeg", as_attachment=True, download_name="enhanced.jpg")

# 📌 Démarrage du serveur Flask
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
