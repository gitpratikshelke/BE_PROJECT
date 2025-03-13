import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import torch
from torch import nn
from torchvision import transforms
from PIL import Image
from transformers import LxmertTokenizer
import easyocr

# Define the model architecture
class ToxicMemeClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(ToxicMemeClassifier, self).__init__()
        from transformers import LxmertModel
        from torchvision import models
        
        self.lxmert = LxmertModel.from_pretrained("unc-nlp/lxmert-base-uncased")

        # ResNet backbone
        resnet = models.resnet50(pretrained=True)
        self.cnn_backbone = nn.Sequential(*list(resnet.children())[:-1])  # Remove final FC layer
        self.visual_fc = nn.Linear(resnet.fc.in_features, self.lxmert.config.visual_feat_dim)

        # Classifier
        self.fc = nn.Linear(self.lxmert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask, images):
        visual_feats = self.cnn_backbone(images).squeeze(-1).squeeze(-1)
        visual_feats = self.visual_fc(visual_feats).unsqueeze(1)

        batch_size = visual_feats.size(0)
        visual_pos = torch.zeros(batch_size, 1, 4).to(visual_feats.device)

        outputs = self.lxmert(input_ids=input_ids, attention_mask=attention_mask, visual_feats=visual_feats, visual_pos=visual_pos)
        logits = self.fc(outputs.pooled_output)
        return logits

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ToxicMemeClassifier(num_classes=2)
model.load_state_dict(torch.load('toxic_meme_classifier.pth', map_location=device))
model.to(device)
model.eval()

# Initialize tokenizer and transforms
tokenizer = LxmertTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# EasyOCR reader
reader = easyocr.Reader(['en'])

# Preprocessing functions
def preprocess_image(image):
    image = Image.open(image).convert("RGB")
    image = transform(image)
    return image.unsqueeze(0)  # Add batch dimension

def extract_text_from_image(image_path):
    result = reader.readtext(image_path)
    extracted_text = " ".join([text[1] for text in result])
    return extracted_text.strip()

def preprocess_text(text, tokenizer, max_len=128):
    encoding = tokenizer(
        text,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    return encoding["input_ids"].squeeze(0), encoding["attention_mask"].squeeze(0)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    # Save the uploaded image
    image = request.files['image']
    image_path = os.path.join('uploads', image.filename)
    os.makedirs('uploads', exist_ok=True)
    image.save(image_path)

    try:
        # Preprocess image and text
        extracted_text = extract_text_from_image(image_path)
        input_ids, attention_mask = preprocess_text(extracted_text, tokenizer)
        image_tensor = preprocess_image(image_path)

        # Move inputs to device
        input_ids = input_ids.to(device).unsqueeze(0)
        attention_mask = attention_mask.to(device).unsqueeze(0)
        image_tensor = image_tensor.to(device)

        # Make prediction
        with torch.no_grad():
            outputs = model(input_ids, attention_mask, image_tensor)
            _, prediction = torch.max(outputs, dim=1)

        prediction_label = 'Toxic' if prediction.item() == 1 else 'Non-Toxic'
        return jsonify({"prediction": prediction_label})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
