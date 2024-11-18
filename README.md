# Image Colorization of Monochrome Images

This project allows users to colorize black-and-white images using a deep learning model and provides color blindness correction features (Deuteranopia, Protanopia, and Tritanopia). The main tool is a Python-based GUI built with Tkinter, and the image colorization is performed using a pre-trained model based on the Siggraph 2017 method.

## Features

- **Image Colorization**: Colorize grayscale images using a pre-trained deep learning model.
- **Color Blindness Correction**: Simulate and correct images for different types of color blindness:
  - Deuteranopia
  - Protanopia
  - Tritanopia
- **GUI Interface**: A simple Tkinter interface for uploading images and applying the colorization or color blindness correction.

## Requirements

Before running the project, ensure that you have the following dependencies installed:

- Python 3.7+
- `torch` (for PyTorch)
- `opencv-python`
- `numpy`
- `Pillow`
- `matplotlib`
- `tkinter`
