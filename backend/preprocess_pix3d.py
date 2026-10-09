"""
Pix3D Preprocessing Script (AR-enabled)
- Generates CLIP embeddings
- Builds FAISS indexes
- Links images to GLB 3D models
"""

import os
import json
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import numpy as np
import faiss
import torch
from transformers import CLIPProcessor, CLIPModel
from clip_color import classify_color_clip

# =========================
# Color utilities
# =========================

BASIC_COLORS = {
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "red": (255, 0, 0),
    "green": (0, 128, 0),
    "blue": (0, 0, 255),
    "yellow": (255, 255, 0),
    "brown": (150, 75, 0),
    "gray": (128, 128, 128),
    "orange": (255, 165, 0),
    "pink": (255, 192, 203),
    "purple": (128, 0, 128),
}

def rgb_to_basic_name(rgb):
    if rgb is None:
        return "unknown"
    r, g, b = rgb
    best, best_dist = None, 1e9
    for name, c in BASIC_COLORS.items():
        d = (r - c[0])**2 + (g - c[1])**2 + (b - c[2])**2
        if d < best_dist:
            best_dist = d
            best = name
    return best or "unknown"

def extract_color_from_image(img):
    img_small = img.resize((100, 100))
    arr = np.array(img_small.convert("RGB"))
    return tuple(np.mean(arr, axis=(0, 1)).astype(int))

# =========================
# Paths
# =========================

PIX3D_ROOT = "data/pix3d"
PIX3D_GLB_ROOT = os.path.join(PIX3D_ROOT, "glb")

OUT_DIR = "data"
OUT_IMAGES = os.path.join(OUT_DIR, "images")
OUT_JSON = os.path.join(OUT_DIR, "clean_products.json")
IMAGE_INDEX = os.path.join(OUT_DIR, "image_index.faiss")
TEXT_INDEX = os.path.join(OUT_DIR, "text_index.faiss")
ID_MAP = os.path.join(OUT_DIR, "id_map.npy")

os.makedirs(OUT_IMAGES, exist_ok=True)

# =========================
# Load metadata
# =========================

metadata_path = os.path.join(PIX3D_ROOT, "pix3d.json")
if not os.path.exists(metadata_path):
    raise SystemExit("❌ pix3d.json not found")

with open(metadata_path, "r", encoding="utf-8") as f:
    items = json.load(f)

print(f"📄 Loaded {len(items)} Pix3D entries")

# =========================
# CLIP setup
# =========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clip_model = CLIPModel.from_pretrained(
    "openai/clip-vit-base-patch32"
).to(device).eval()
clip_processor = CLIPProcessor.from_pretrained(
    "openai/clip-vit-base-patch32"
)

# =========================
# Storage
# =========================

image_embeddings = []
text_embeddings = []
products = []
id_map = []

counter = 0

# =========================
# Main loop
# =========================

for it in tqdm(items, desc="Processing Pix3D"):
    img_rel = it.get("img")
    category = it.get("category", "object").lower()

    if not img_rel:
        continue

    img_path = os.path.join(PIX3D_ROOT, img_rel)
    if not os.path.exists(img_path):
        continue

    # ---------- AR MODEL LINKING ----------
    model_rel = it.get("model")
    has_3d_model = False
    glb_path = None

    if model_rel:
        model_id = Path(model_rel).parts[-2]
        candidate_glb = os.path.join(
            PIX3D_GLB_ROOT,
            category,
            f"{model_id}.glb"
        )
        if os.path.exists(candidate_glb):
            has_3d_model = True
            glb_path = f"{category}/{model_id}.glb"

    # ---------- IMAGE SAVE ----------
    cat_dir = os.path.join(OUT_IMAGES, category)
    os.makedirs(cat_dir, exist_ok=True)

    out_img_name = f"{counter}.jpg"
    out_img_path = os.path.join(cat_dir, out_img_name)

    img = Image.open(img_path).convert("RGB")
    img.save(out_img_path, "JPEG", quality=95)

    # ---------- METADATA ----------
    color = classify_color_clip(clip_model, clip_processor, img, category, device)
    product_name = f"{color.capitalize()} {category.capitalize()}"

    product = {
        "id": counter,
        "productDisplayName": product_name,
        "masterCategory": "Furniture",
        "subCategory": category.capitalize(),
        "baseColour": color,
        "image_path": f"{category}/{out_img_name}",
        "has_3d_model": has_3d_model,
        "glb_path": glb_path,
    }

    # ---------- EMBEDDINGS ----------
    img_inputs = clip_processor(images=img, return_tensors="pt").to(device)
    txt_inputs = clip_processor(
        text=[product_name],
        return_tensors="pt",
        padding=True
    ).to(device)

    with torch.no_grad():
        img_feat = clip_model.get_image_features(**img_inputs)
        txt_feat = clip_model.get_text_features(**txt_inputs)

    img_feat = img_feat.cpu().numpy()
    txt_feat = txt_feat.cpu().numpy()

    img_feat /= np.linalg.norm(img_feat)
    txt_feat /= np.linalg.norm(txt_feat)

    image_embeddings.append(img_feat)
    text_embeddings.append(txt_feat)
    products.append(product)
    id_map.append(counter)

    counter += 1

# =========================
# Save FAISS + JSON
# =========================

image_matrix = np.vstack(image_embeddings).astype("float32")
text_matrix = np.vstack(text_embeddings).astype("float32")

d = image_matrix.shape[1]
img_index = faiss.IndexFlatIP(d)
txt_index = faiss.IndexFlatIP(d)

img_index.add(image_matrix)
txt_index.add(text_matrix)

faiss.write_index(img_index, IMAGE_INDEX)
faiss.write_index(txt_index, TEXT_INDEX)
np.save(ID_MAP, np.array(id_map))

with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(products, f, indent=2)

print("✅ Preprocessing complete")
print(f"📦 Products: {len(products)}")
print(f"🧊 With 3D models: {sum(p['has_3d_model'] for p in products)}")