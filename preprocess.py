import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import os
import pandas as pd
import easyocr
import re

# Initialize EasyOCR Reader (for English text)
reader = easyocr.Reader(['en'])

# Initialize the CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Define the meme image directory
meme_directory = "E:/BE Project/archive (3)/memes"

# Function to preprocess text
def preprocess_text(text):
    # Remove email addresses
    text = re.sub(r'\S+@\S+\.\S+', '', text)
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    # Remove special characters and numbers (keep alphabets and spaces)
    text = re.sub(r'[^A-Za-z\s]', '', text)
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Function to classify toxicity based on image and text
def classify_toxicity(image_path, text):
    # Open and preprocess the image
    image = Image.open(image_path)
    
    # Prepare the inputs (image and text) with padding and truncation
    inputs = processor(
        text=[text], 
        images=image, 
        return_tensors="pt", 
        padding=True,  # Padding the sequence to the maximum length
        truncation=True  # Truncating if the sequence exceeds the max length
    )
    
    # Get the outputs from the model
    outputs = model(**inputs)
    
    # Extract the image and text features
    image_features = outputs.image_embeds
    text_features = outputs.text_embeds
    
    # Calculate the similarity score (cosine similarity)
    similarity = torch.cosine_similarity(image_features, text_features)
    
    # Define a threshold for toxicity (example threshold)
    toxicity_threshold = 0.4
    
    # If the similarity score is below the threshold, classify as toxic
    label = "Toxic" if similarity < toxicity_threshold else "Non-toxic"
    
    return label, similarity.item()

# Function to extract text from image using EasyOCR
def extract_text_from_image(image_path):
    # Perform OCR using EasyOCR
    result = reader.readtext(image_path)
    # Combine all the text found in the image
    text = " ".join([entry[1] for entry in result])
    # Preprocess the extracted text
    text = preprocess_text(text)
    return text.strip()

# Process all meme images in the directory
results = []
for filename in os.listdir(meme_directory):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(meme_directory, filename)
        
        # Extract text from image
        text = extract_text_from_image(image_path)
        
        # If no text was extracted, skip the image
        if not text:
            continue
        
        # Classify toxicity based on image and extracted text
        label, score = classify_toxicity(image_path, text)
        
        # Store results
        results.append({
            "image": filename,
            "extracted_text": text,
            "toxicity_label": label,
            "toxicity_score": score
        })

# Convert results into a DataFrame
df = pd.DataFrame(results)

# Save results to a CSV file
df.to_csv("meme_toxicity_results.csv", index=False)

# Print the DataFrame if needed
print(df.head())
