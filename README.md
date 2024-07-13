
# 🌾 Crop Classification using Deep Learning with Inception and VGG-16 Models 🌾

[![Python](https://img.shields.io/badge/python-v3.7+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📜 Overview

This project focuses on classifying different types of crops using deep learning models, specifically Inception and VGG-16. Leveraging the power of convolutional neural networks, the project aims to accurately identify crop types from images.

## 📂 Project Structure

```plaintext
├── data
│   ├── train
│   │   ├── rice
│   │   ├── wheat
│   │   ├── jute
│   │   ├── sugarcane
│   │   ├── maize
├── models
│   ├── inception_model.h5
│   ├── vgg16_model.h5
├── notebooks
│   ├── data_preprocessing.ipynb
│   ├── model_training.ipynb
├── scripts
│   ├── train.py
│   ├── evaluate.py
├── README.md
├── requirements.txt
```

## 🚀 Getting Started

### Prerequisites

- Python 3.7+
- TensorFlow 2.0+
- Google Colab (Optional for cloud training)

### Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/crop-classification.git
    cd crop-classification
    ```

2. **Install required packages:**
    ```bash
    pip install -r requirements.txt
    ```

### Usage

#### Data Preparation

Ensure your data is organized as shown in the project structure. You can use the provided notebooks for data preprocessing.

#### Training the Model

To train the model using VGG-16:

```python
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Data paths
train_data_dir = '/content/drive/MyDrive/Kag2'
batch_size = 64
img_rows, img_cols = 224, 224

# Data augmentation
train_datagen = ImageDataGenerator(rescale=1.0/255.0, horizontal_flip=True, vertical_flip=True, rotation_range=90)

train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_rows, img_cols),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True)

# Load pre-trained VGG-16 model
pre_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_rows, img_cols, 3))

for layer in pre_model.layers:
    layer.trainable = False

# Add custom top layers
def add_top_model(bottom_model, num_classes):
    top_model = bottom_model.output
    top_model = Flatten(name="flatten")(top_model)
    top_model = Dense(512, activation="relu")(top_model)
    top_model = Dropout(0.5)(top_model)
    top_model = Dense(512, activation="relu")(top_model)
    top_model = Dropout(0.3)(top_model)
    top_model = Dense(num_classes, activation="softmax")(top_model)
    return top_model

num_classes = 5
FC_Head = add_top_model(pre_model, num_classes)
model = Model(inputs=pre_model.input, outputs=FC_Head)

# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train model
history = model.fit(train_generator, epochs=40, steps_per_epoch=train_generator.samples // batch_size)

# Save model
model.save('vgg16_crop_model.h5')
```

#### Evaluation

To evaluate the trained model, use the evaluation script:

```python
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Load the model
model = load_model('vgg16_crop_model.h5')

# Evaluate model
predictions = model.predict_generator(test_generator, steps=test_generator.samples // batch_size)
predicted_classes = np.argmax(predictions, axis=1)

# Print classification report
true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys())
print(classification_report(true_classes, predicted_classes, target_names=class_labels))
```

## 📊 Results

After training the models, you can visualize the performance metrics such as accuracy and loss using the following code snippet:

```python
import matplotlib.pyplot as plt

def plot_metrics(history):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.show()

plot_metrics(history)
```

## 🛠️ Tools & Libraries

- TensorFlow
- Keras
- OpenCV
- Matplotlib
- Google Colab


## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.
