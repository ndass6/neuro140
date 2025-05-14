import pandas as pd
import torch
from torchvision import models, transforms
from PIL import Image
import os
import numpy as np
import pickle
from tqdm import tqdm

# Load pretrained ResNet model
model = models.resnet50(pretrained=True)
model.eval()

# Remove the final classification layer to get embeddings
model = torch.nn.Sequential(*list(model.children())[:-1])

# Define image preprocessing
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Read the CSV file
df = pd.read_csv('sampled_test.csv')

# Create dictionary to store embeddings
embeddings_dict = {}

# Process each image
for image_path in tqdm(df['image_name'].unique(), desc="Processing images"):
    try:
        # Load and preprocess image
        img = Image.open(os.path.join('images', image_path)).convert('RGB')
        img_tensor = preprocess(img)
        img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension

        # Get embedding
        with torch.no_grad():
            embedding = model(img_tensor)

        # Store flattened embedding in dictionary
        embeddings_dict[image_path] = embedding.numpy().flatten()
        
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")

# Save dictionary to pickle file
with open('test_resnet_embeddings.pkl', 'wb') as f:
    pickle.dump(embeddings_dict, f)
