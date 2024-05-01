from transformers import FlavaProcessor, FlavaModel
from PIL import Image
import torch
import torch.nn.functional as F

# Load the FLAVA model and processor
model = FlavaModel.from_pretrained("facebook/flava-full")
processor = FlavaProcessor.from_pretrained("facebook/flava-full")

# Load the image
image_path = "/home/pyt_user/pp/pytorch/flava/login2.png"  # Replace with your image path
image = Image.open(image_path).convert("RGB")

# The question to ask about the image
question = "What is in the image?"

# Process the image and the text
inputs = processor(text=[question], images=[image], return_tensors="pt")

# Make the model generate the answer
outputs = model(**inputs)

# Use the multimodal output for further processing
multimodal_output = outputs.multimodal_output.pooler_output

print(multimodal_output)





