# MemeShield - Multimodal Content Moderation System for Detecting Toxic Memes

## Overview

MemeShield is an advanced multimodal content moderation system designed to detect toxic memes using both textual and visual analysis. The system integrates **EasyOCR, LXMERT, and ResNet** to efficiently extract and analyze text from images while evaluating their content for toxicity. The project achieves an **accuracy of 75-80%** in detecting harmful content, ensuring a safer digital environment.

## Features

- **Multimodal Analysis**: Utilizes text and image data for more accurate toxicity detection.
- **High Accuracy**: Achieves 75-80% detection accuracy using advanced deep learning models.
- **Efficient OCR Integration**: Uses EasyOCR for robust text extraction from memes.
- **Flask-Based Deployment**: Offers a user-friendly web interface for easy interaction.
- **Optimized Backend**: Python-based optimizations improve performance by 30%.

## Tech Stack

- **Programming Language**: Python
- **Deep Learning Frameworks**: PyTorch
- **Optical Character Recognition (OCR)**: EasyOCR
- **Multimodal Model**: LXMERT (Language and Vision Model)
- **Image Processing**: ResNet
- **Web Framework**: Flask

## Installation

### Prerequisites

Ensure you have the following installed:

- Python (>=3.8)
- Pip
- Virtual Environment (optional but recommended)


## Usage

### Running the Flask App

```bash
python app.py
```

Access the web interface at **[http://127.0.0.1:5000/](http://127.0.0.1:5000/)**

### Detecting Toxic Memes

1. Upload an image containing a meme.
2. The system extracts text and analyzes the image.
3. A toxicity score is generated along with a classification (safe or toxic).
4. Results are displayed on the web interface.

## Model Architecture

![System Architecture](images/system_architecture.png)

### Text Extraction

- **EasyOCR** extracts text from memes.
- Preprocessing is done to remove noise.

### Image Analysis

- **ResNet** extracts deep visual features from memes.

### Multimodal Fusion

- **LXMERT** processes both textual and visual features together to determine toxicity.

### Classification

- The extracted text and image features are passed through a deep learning classifier to predict toxicity.


## Performance

- Accuracy: **75-80%**
- Backend Optimization: **30% boost in efficiency**

## Future Enhancements

- Integrate real-time meme scanning in social media platforms.
- Improve text-based analysis using large language models (LLMs).
- Deploy as a cloud-based moderation service.

## Contributing

Contributions are welcome! Feel free to submit issues and pull requests.
