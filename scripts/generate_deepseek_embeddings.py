import pandas as pd
import pickle
from tqdm import tqdm
from transformers import AutoModelForCausalLM
import os

from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
from deepseek_vl2.utils.io import load_pil_images

os.environ["USE_FLASH_ATTENTION"] = "0"
os.environ["USE_MEM_EFFICIENT_ATTENTION"] = "0"
os.environ["XFORMERS_FORCE_DISABLE"] = "1"

# specify the path to the model
model_path = "deepseek-ai/deepseek-vl2-tiny"
vl_chat_processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer

vl_gpt: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
vl_gpt.eval()

# Read the data file
df = pd.read_csv('sampled_train.csv')

# Create dictionary to store embeddings
embeddings_dict = {}

# Get embeddings for each image
for image_path in tqdm(df['image_name'].unique(), desc="Processing images"):
    try:
        conversation = [
            {
                "role": "<|User|>",
                "content": "<image>",
                "images": [f"./images/{image_path}"],
            },
            {"role": "<|Assistant|>", "content": ""},
        ]

        # load images and prepare for inputs
        pil_images = load_pil_images(conversation)
        prepare_inputs = vl_chat_processor(
            conversations=conversation,
            images=pil_images,
            force_batchify=True,
            system_prompt=""
        )

        # run image encoder to get the image embeddings
        image_embedding = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

        print(image_embedding.size())
            
        # Store embedding in dictionary
        embeddings_dict[image_path] = image_embedding.numpy().flatten()
        
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        1/0

# Save dictionary to pickle file
with open('train_deepseek_embeddings.pkl', 'wb') as f:
    pickle.dump(embeddings_dict, f)
