# Image Captioning using Deep Learning and computer vision

This project implements an image captioning system using a combination of CNN (VGG16) and LSTM networks. The model can generate natural language descriptions for input images.

## Overview

The system uses a deep learning model that combines:
- VGG16 for image feature extraction
- LSTM network for sequence generation
- Word embeddings for text processing

## Requirements

```
tensorflow >= 2.0
keras
numpy
pillow
tqdm
nltk
matplotlib
```

## Project Structure

```
project/
│
├── data/
│   ├── Images/         # Directory containing input images
│   └── captions.txt    # Image captions file
│
├── models/
│   └── Trainedmodel_final.keras    # Saved model file
│
└── features.pkl        # Extracted image features
```

## Dataset Preparation

1. The Images directory should contain all training images
2. Captions file format:
   ```
   image_name.jpg,caption1
   image_name.jpg,caption2
   ```

## Setup and Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/image-captioning.git
   cd image-captioning
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Download and prepare the dataset:
   - Place images in the `data/Images` directory
   - Ensure captions.txt is properly formatted

## Training

1. Extract image features:
   ```python
   python feature_extraction.py
   ```

2. Train the model:
   ```python
   python train.py --epochs 20 --batch_size 32
   ```

Training parameters can be modified in the training script:
- Number of epochs: 20 (default)
- Batch size: 32 (default)
- Learning rate: 0.001 (default)

## Model Architecture

1. **Image Encoder (VGG16)**
   - Pre-trained VGG16 network
   - Extracts 4096-dimensional feature vector

2. **Caption Generator**
   - Embedding layer (256 dimensions)
   - LSTM layer (256 units)
   - Dense layers for word prediction

3. **Training Process**
   - Uses teacher forcing
   - Categorical crossentropy loss
   - Adam optimizer

## Usage

To generate captions for new images:

```python
from caption_generator import generate_caption

# Generate caption for single image
image_path = "path/to/your/image.jpg"
caption = generate_caption(image_path)
print(caption)
```

## Evaluation

The model is evaluated using:
- BLEU-1 score
- BLEU-2 score

Example results:
```
BLEU-1: 0.567
BLEU-2: 0.321
```

## Sample Results

The model generates captions like:
- "a dog running in the park"
- "people sitting at a table in a restaurant"
- "a cat sleeping on a windowsill"

## Limitations

- Requires pre-processed images (224x224 pixels)
- Limited to vocabulary seen during training
- May not handle complex scenes well

## Future Improvements

1. Implement attention mechanism
2. Use more modern architectures (ResNet, EfficientNet)
3. Add beam search for caption generation
4. Increase vocabulary coverage

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- VGG16 model from [keras-applications](https://keras.io/api/applications/)
- Image captioning inspiration from [Show and Tell paper](https://arxiv.org/abs/1411.4555)
