import collections
import os

from tqdm import tqdm

# Get list of files in images directory
image_files = set(os.listdir('images'))

image_to_lines = collections.defaultdict(list)
unique_images = set()
total_count = 0
with open('train_data_list.txt', 'r') as infile:
    for line in infile:
        unique_images.add(line.strip().split('\t')[1])
        image_to_lines[line.strip().split('\t')[1]].append(line)
        total_count += 1

# Process lines sequentially
valid_lines = []
for image in tqdm(unique_images, desc="Processing images"):
    if image in image_files:
        valid_lines.extend(image_to_lines[image])

# Write valid lines to output file
count = 0
with open('train_data_list_with_images.txt', 'w') as outfile:
    for line in valid_lines:
        if line is not None:
            outfile.write(line)
            count += 1

print(f"Found {count} out of {total_count} image files from train_data_list.txt in images directory")
