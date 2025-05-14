import pandas as pd
import torch
import open_clip
from PIL import Image
import os
import numpy as np
import pickle
from tqdm import tqdm

# Load pretrained OpenCLIP model
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
model.eval()

# Read the CSV file
df = pd.read_csv('sampled_test.csv')

# Create dictionary to store embeddings
embeddings_dict = {}

# Process each image
for image_path in tqdm(df['image_name'].unique(), desc="Processing images"):
    try:
        # Load and preprocess image
        img = Image.open(os.path.join('images', image_path)).convert('RGB')
        img_tensor = preprocess(img).unsqueeze(0)  # Add batch dimension and move to GPU

        # Get embedding
        with torch.no_grad():
            embedding = model.encode_image(img_tensor)
            
        # Store flattened embedding in dictionary
        embeddings_dict[image_path] = embedding.cpu().numpy().flatten()
        
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")

# Save dictionary to pickle file
with open('test_openclip_embeddings.pkl', 'wb') as f:
    pickle.dump(embeddings_dict, f)
