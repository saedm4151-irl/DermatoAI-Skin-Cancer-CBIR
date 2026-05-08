import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os


st.set_page_config(page_title="SkinScan AI | Research Dashboard", layout="wide")

MODEL_PATH = 'Updated_best.pth'  #
FEATURES_PATH = 'features.npy'
FILENAMES_PATH = 'filenames.npy'

DATASET_IMG_DIR = "dataset/ISIC2018_Task3_Training_Input/" 

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ['Melanoma (MEL)', 'Melanocytic Nevi (NV)', 'Basal Cell Carcinoma (BCC)', 
               'Actinic Keratosis (AKIEC)', 'Benign Keratosis (BKL)', 
               'Dermatofibroma (DF)', 'Vascular Lesion (VASC)']


@st.cache_resource
def load_all_resources():
    # Load Model Architecture
    model = models.resnet50()
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.fc.in_features, 7)
    )
    
    # Load Weights (handling single GPU/CPU)
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    # Remove 'module.' prefix if it exists from DataParallel
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    
    model.load_state_dict(new_state_dict)
    model.to(DEVICE).eval()
    
    # Load CBIR Database
    features_db = np.load(FEATURES_PATH)
    filenames_db = np.load(FILENAMES_PATH, allow_pickle=True)
    
    return model, features_db, filenames_db

model, features_db, filenames_db = load_all_resources()

# Create Feature Extractor (Backbone)
backbone = torch.nn.Sequential(*list(model.children())[:-1])


def get_prediction(img):
    """Runs prediction with Test-Time Augmentation (TTA)"""
    base_transform = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Define augmentations for TTA
    tta_transforms = [
        lambda x: x, # Original
        transforms.functional.hflip,
        transforms.functional.vflip,
    ]
    
    all_probs = []
    with torch.no_grad():
        for t in tta_transforms:
            aug_img = t(img)
            tensor = base_transform(aug_img).unsqueeze(0).to(DEVICE)
            output = model(tensor)
            all_probs.append(torch.softmax(output, dim=1))
            
    # Average probabilities across all views
    avg_probs = torch.stack(all_probs).mean(0)
    conf, idx = torch.max(avg_probs, 1)
    
    # Get the feature vector (fingerprint) from the original image for CBIR
    original_tensor = base_transform(img).unsqueeze(0).to(DEVICE)
    query_vec = backbone(original_tensor).view(1, -1).detach().cpu().numpy()
    
    return idx.item(), conf.item(), avg_probs[0].cpu().numpy(), query_vec


st.title("SkinScan AI: Clinical Reference Tool")
st.markdown("""
This system provides an **AI-driven diagnostic prediction** combined with 
**Content-Based Image Retrieval (CBIR)** to show similar historical cases.
""")
st.divider()

uploaded_file = st.sidebar.file_uploader("Upload a Dermoscopic Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    query_img = Image.open(uploaded_file).convert("RGB")
    
    # Run Inference
    with st.spinner('Analyzing patterns and searching database...'):
        pred_idx, confidence, probs, query_vec = get_prediction(query_img)

    # Top Section: Results
    col1, col2 = st.columns([1, 1.2])
    
    with col1:
        st.image(query_img, caption="Uploaded Image", use_container_width=True)
        
    with col2:
        st.header(f"Prediction: **{CLASS_NAMES[pred_idx]}**")
        st.subheader(f"Confidence: {confidence*100:.1f}%")
        st.progress(confidence)
        
        # Display probability for each class
        prob_dict = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}
        st.bar_chart(prob_dict)

    # Bottom Section: CBIR Matches
    st.divider()
    st.subheader("Similar Historical Cases (CBIR Results)")
    st.write("The following images from the training database share the most similar visual features (textures/structures).")
    
    # Calculate Similarity
    similarities = cosine_similarity(query_vec, features_db).flatten()
    top_k_indices = similarities.argsort()[-3:][::-1]
    
    match_cols = st.columns(3)
    for i, idx in enumerate(top_k_indices):
        with match_cols[i]:
            filename = filenames_db[idx]
            # Handle potential file extension issues (.jpg)
            img_path = os.path.join(DATASET_IMG_DIR, f"{filename}.jpg")
            
            if os.path.exists(img_path):
                st.image(Image.open(img_path), use_container_width=True)
                st.write(f"**Similarity Score:** {similarities[idx]:.4f}")
            else:
                st.error(f"Image {filename} not found in directory.")

else:
    st.info("Please upload an image in the sidebar to begin analysis.")


with st.expander("Technical Model Details"):
    st.write("""
    - **Architecture:** ResNet-50
    - **Input Resolution:** 448x448 px
    - **Loss Function:** Weighted Focal Loss (optimized for Melanoma Recall)
    - **Augmentation:** Test-Time Augmentation (TTA) enabled
    - **Feature Space:** 2048-dimensional Global Average Pooling embeddings
    """)