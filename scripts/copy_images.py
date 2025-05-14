import pandas as pd
import shutil
import os
from tqdm import tqdm

# Read the CSV file
df = pd.read_csv('sampled_test.csv')  # Update with your CSV filename

# Get unique image paths from second column
image_paths = df['image_name'].unique()

# Create destination folder if it doesn't exist
dest_folder = 'filtered_images'  # Update with your desired destination folder
os.makedirs(dest_folder, exist_ok=True)

# Source folder where images are located
source_folder = 'images'  # Update with your source image folder

# Copy each unique image
for image_path in tqdm(image_paths, desc="Copying images"):
    source_path = os.path.join(source_folder, image_path)
    dest_path = os.path.join(dest_folder, image_path)
    
    try:
        shutil.copy2(source_path, dest_path)
    except FileNotFoundError:
        print(f"Could not find image: {image_path}")
    except Exception as e:
        print(f"Error copying {image_path}: {str(e)}")
