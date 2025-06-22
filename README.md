
# 🛡️ Shielding Social Media: Detection of Toxic Memes for Automated Moderation

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![Vision+NLP](https://img.shields.io/badge/Multimodal-Vision%20%2B%20Text-green)

## 📌 Abstract

This project tackles the challenge of moderating toxic content in **internet memes**, which combine images and text to spread potentially harmful or offensive messages. We propose a **multimodal detection system** that uses **OCR**, **NLP**, and **Vision-Language Fusion** via **SigLIP** to classify memes as **toxic** or **non-toxic**, ensuring safer social media environments.

---

## 🧠 System Architecture

<img src="op1.png" alt="System Architecture" width="100%"/>

> Architecture Diagram: Fusion-based Toxic Meme Classifier with OCR, NLP (BERT), Vision Transformer, and SigLIP.

### 🔄 Flow Description:

1. **Input Meme**: Raw image meme with visual and textual content.
2. **Text Preprocessing**:
   - Uses `PaddleOCR` and `KOSMOS-2` for text extraction.
   - Applies BERT tokenizer → input IDs + attention mask.
3. **Image Preprocessing**:
   - Image resized and transformed into pixel tensors.
4. **Feature Embedding**:
   - Text and image inputs are separately embedded.
   - Passed into **SigLIP** model for multimodal fusion.
5. **Classification**:
   - Uses **Sigmoid activation** + **Cross Entropy Loss**.
   - Optimized to predict `Toxic` or `Non-Toxic`.

---

## 🚀 Features

- 🔍 Detects **toxicity in memes** using a deep learning fusion approach.
- 🔤 Supports **OCR** from meme text using **PaddleOCR** & **KOSMOS-2**.
- 🧠 Leverages **BERT** for semantic understanding of meme text.
- 👁️‍🗨️ Uses **SigLIP (Google)** for image-text fusion and classification.
- 📈 Provides performance metrics and visualizations.

---

## 🛠️ Tech Stack

| Component           | Technology           |
|---------------------|----------------------|
| Language            | Python 3.8+          |
| OCR Engine          | PaddleOCR, KOSMOS-2  |
| Text Encoder        | BERT (transformers)  |
| Image Encoder       | Vision Transformer (ViT) |
| Fusion Model        | SigLIP               |
| DL Framework        | PyTorch              |
| Visualization       | Matplotlib, Seaborn  |

🔮 Future Scope

🌍 Multilingual toxic meme detection

🎥 Video meme frame-based detection

🌐 Web portal for real-time moderation

📦 Deploy as browser extension / REST API


📄 License
This project is licensed under the MIT License. See the LICENSE file for more details.

⭐ Support
If you found this project useful, please consider giving it a ⭐ and sharing it with others!
"""

