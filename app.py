# --------------------------------------------------------------------------
# Project: Dermato-AI: Skin Lesion Classification & CBIR System
# Author: Saed (https://github.com/saedm4151-irl)
# --------------------------------------------------------------------------

import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
import zipfile
import io
from huggingface_hub import hf_hub_download
from collections import OrderedDict

# --- PAGE CONFIG ---
st.set_page_config(page_title="SkinScan AI | Research Dashboard", layout="wide")

# --- CONFIGURATION ---
# Assets are pulled from Hugging Face to keep GitHub repository light
HF_MODEL_REPO = "saedm4151-irl/dermato-ai-resnet50"
HF_DATASET_REPO = "saedm4151-irl/skin-cancer-isic-2018"
DATASET_FILENAME = "ISIC2018_Complete_Dataset.zip"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ['Melanoma (MEL)', 'Melanocytic Nevi (NV)', 'Basal Cell Carcinoma (BCC)', 
               'Actinic Keratosis (AKIEC)', 'Benign Keratosis (BKL)', 
               'Dermatofibroma (DF)', 'Vascular Lesion (VASC)']

@st.cache_resource
def load_all_resources():
    """
    Downloads model weights and dataset from Hugging Face Hub.
    Resources are cached locally on the Streamlit server to prevent 'shivering'.
    """
    # 1. Download Model and CBIR files from Model Repo
    # Ensure filename matches exactly what is on Hugging Face
    model_weight_path = hf_hub_download(repo_id=HF_MODEL_REPO, filename="model.pth")
    features_path = hf_hub_download(repo_id=HF_MODEL_REPO, filename="features.npy")
    filenames_path = hf_hub_download(repo_id=HF_MODEL_REPO, filename="filenames.npy")

    # 2. Download dataset ZIP from the Dataset repo
    zip_path = hf_hub_download(repo_id=HF_DATASET_REPO, filename=DATASET_FILENAME, repo_type="dataset")

    # 3. Build Architecture (ResNet-50)
    model = models.resnet50()
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.fc.in_features, 7)
    )
    
    # 4. Load Weights
    state_dict = torch.load(model_weight_path, map_location=DEVICE)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        # Strip 'module.' prefix if model was trained with DataParallel
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    
    model.load_state_dict(new_state_dict)
    model.to(DEVICE).eval()
    
    # 5. Load CBIR Database
    features_db = np.load(features_path)
    filenames_db = np.load(filenames_path, allow_pickle=True)
    
    return model, features_db, filenames_db, zip_path

# --- INITIALIZE CORE COMPONENTS ---
with st.spinner("Initializing AI Engine and Loading Datasets..."):
    model, features_db, filenames_db, cached_zip = load_all_resources()
    # Create backbone for feature extraction (removing the final FC layer)
    backbone = torch.nn.Sequential(*list(model.children())[:-1])

@st.cache_resource
def get_zip_ref(path):
    return zipfile.ZipFile(path, 'r')

zf = get_zip_ref(cached_zip)

# --- PREDICTION LOGIC ---
def get_prediction(img):
    base_transform = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Test-Time Augmentation (TTA) for more robust clinical predictions
    tta_transforms = [lambda x: x, transforms.functional.hflip, transforms.functional.vflip]
    all_probs = []
    
    with torch.no_grad():
        for t in tta_transforms:
            aug_img = t(img)
            tensor = base_transform(aug_img).unsqueeze(0).to(DEVICE)
            output = model(tensor)
            all_probs.append(torch.softmax(output, dim=1))
            
    avg_probs = torch.stack(all_probs).mean(0)
    conf, idx = torch.max(avg_probs, 1)
    
    # Extract query vector for CBIR
    original_tensor = base_transform(img).unsqueeze(0).to(DEVICE)
    query_vec = backbone(original_tensor).view(1, -1).detach().cpu().numpy()
    
    return idx.item(), conf.item(), avg_probs[0].cpu().numpy(), query_vec

# --- UI LAYOUT ---
st.title("SkinScan AI: Clinical Reference Tool")
st.markdown("Diagnostic Prediction & Content-Based Image Retrieval (CBIR)")
st.divider()

uploaded_file = st.sidebar.file_uploader("Upload Dermoscopic Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    query_img = Image.open(uploaded_file).convert("RGB")
    
    with st.spinner('Analyzing dermoscopic patterns...'):
        pred_idx, confidence, probs, query_vec = get_prediction(query_img)

    # Main Analysis Section
    col1, col2 = st.columns([1, 1.2])
    with col1:
        with st.container(border=True):
            st.image(query_img, use_container_width=True, caption="Query Image")
        
    with col2:
        st.header(f"Result: **{CLASS_NAMES[pred_idx]}**")
        st.subheader(f"Confidence: {confidence*100:.1f}%")
        st.progress(confidence)
        
        # Probability distribution chart
        prob_dict = {CLASS_NAMES[i]: float(probs[i]) for i in range(7)}
        st.bar_chart(prob_dict)

    st.divider()
    st.subheader("Top Historical Matches (ISIC Archive)")
    
    # --- STABILIZED CBIR RENDERING ---
    results_container = st.container()
    
    with st.spinner('Searching historical archive...'):
        # Compute cosine similarity between query and database
        similarities = cosine_similarity(query_vec, features_db).flatten()
        top_k_indices = similarities.argsort()[-3:][::-1]
        
        matches = []
        for idx in top_k_indices:
            fname = str(filenames_db[idx]).replace('.jpg', '')
            path = f"ISIC2018_Task3_Training_Input/{fname}.jpg"
            try:
                data = zf.read(path)
                matches.append({
                    "img": Image.open(io.BytesIO(data)), 
                    "score": similarities[idx], 
                    "id": fname
                })
            except Exception:
                continue

    # Render results in a fixed 3-column grid to prevent layout shifts
    with results_container:
        if matches:
            cols = st.columns(3)
            for i, res in enumerate(matches):
                with cols[i]:
                    with st.container(border=True):
                        st.image(res["img"], use_container_width=True)
                        st.write(f"**Similarity:** {res['score']:.4f}")
                        st.caption(f"ID: {res['id']}")
        else:
            st.warning("Could not retrieve similar images from the archive.")
else:
    st.info("Upload a dermoscopic image in the sidebar to begin clinical analysis.")
