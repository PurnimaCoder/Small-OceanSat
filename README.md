# Small-OceanSat
Deep Learning Algorithm development for Small-OceanSat

# Chlorophyll-a U-Net Model

## Overview
This repository contains a U-Net model implementation for predicting chlorophyll-a concentration in ocean images. The model is trained using labeled datasets of ocean images and their corresponding chlorophyll-a masks. This document provides instructions on setting up the environment, training the model, and generating chlorophyll-a concentration maps.

## Requirements

### Libraries

- **tensorflow==2.13.0**
- **opencv-python-headless==4.7.0.72**
- **numpy==1.24.3**
- **matplotlib==3.7.1**
- **scikit-learn==1.2.2**

### Setup

1. **Install Python**: Ensure Python is installed on your system. You can download it from [python.org](https://python.org).

2. **Create a Virtual Environment**: Create a virtual environment for the project to manage dependencies.
    ```bash
    python -m venv unet-env
    ```

3. **Activate the Virtual Environment**: Activate the virtual environment.
    - Windows:
      ```bash
      .\unet-env\Scripts\activate
      ```
    - macOS and Linux:
      ```bash
      source unet-env/bin/activate
      ```

4. **Install Required Libraries**: Use the `requirements.txt` file to install necessary libraries.
    ```bash
    pip install -r requirements.txt
    ```

5. **Verify Installation**: Run the following commands to ensure the libraries are installed correctly.
    ```python
    import tensorflow as tf
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split

    print("All libraries imported successfully!")
    ```

## U-Net Model Implementation

### Model Definition
The U-Net model architecture is defined in the `unet_model` function. This model consists of a downsampling path, a bottleneck, and an upsampling path.

```python
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
from glob import glob

def unet_model(input_size=(256, 256, 3)):
    inputs = Input(input_size)

    # Downsample
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(c4)
    p4 = MaxPooling2D((2, 2))(c4)

    # Bottleneck
    c5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(p4)
    c5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(c5)

    # Upsample
    u6 = UpSampling2D((2, 2))(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(512, (3, 3), activation='relu', padding='same')(u6)
    c6 = Conv2D(512, (3, 3), activation='relu', padding='same')(c6)

    u7 = UpSampling2D((2, 2))(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(256, (3, 3), activation='relu', padding='same')(u7)
    c7 = Conv2D(256, (3, 3), activation='relu', padding='same')(c7)

    u8 = UpSampling2D((2, 2))(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(128, (3, 3), activation='relu', padding='same')(u8)
    c8 = Conv2D(128, (3, 3), activation='relu', padding='same')(c8)

    u9 = UpSampling2D((2, 2))(c8)
    u9 = concatenate([u9, c1])
    c9 = Conv2D(64, (3, 3), activation='relu', padding='same')(u9)
    c9 = Conv2D(64, (3, 3), activation='relu', padding='same')(c9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

