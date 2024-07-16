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

