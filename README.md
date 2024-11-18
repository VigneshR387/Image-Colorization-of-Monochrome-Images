# Image Colorization with Color Blindness Correction

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
Additionally, if you do not have Tkinter installed, you can install it using:

    For Ubuntu/Debian:

    sudo apt-get install python3-tk

    For Windows, Tkinter is usually bundled with the standard Python installation.

Setup

    Clone or download the repository to your local machine.
    Install the required dependencies (as listed above).
    Make sure you have the necessary colorizer model files in the colorizers/ folder (these are required for the colorization functionality).

Usage
1. Run the GUI

You can run the program by executing the following command in your terminal:

python COLOR_BLINDNESS.py

This will open a GUI window where you can:

    Upload an image: Click the "Upload Image" button to choose a black-and-white image for colorization.
    Apply Color Blindness Correction: Select one of the following buttons to apply color blindness correction:
        Deuteranopia Correction
        Protanopia Correction
        Tritanopia Correction

Once the image is processed, the results will be displayed in a matplotlib window showing the following images:

    The original image.
    The black-and-white input image.
    The colorized image using the Siggraph 2017 model.
    The color-blindness-corrected image (if any correction type is selected).

2. Image Output

The processed images will not be saved by default. If you wish to save the output images, you can modify the code to save the images after they are processed.
File Structure

├── COLOR_BLINDNESS.py          # Main script with GUI and colorization code
├── colorizers/                 # Folder containing colorizer models and utility functions
├── imgs/                       # Folder for input images
├── imgs_out/                   # Folder for output images (optional)
├── LICENSE                     # Project License file
├── README.md                   # Project documentation
├── requirements.txt            # List of required dependencies

License

This project is licensed under the MIT License - see the LICENSE file for details.
Acknowledgments

    The colorization method used in this project is based on the Siggraph 2017 image colorization paper.
    Special thanks to the authors and contributors of PyTorch, OpenCV, and other libraries used in this project.

Troubleshooting

If you encounter any issues, please check the following:

    Ensure that your environment meets all the required dependencies.
    If you're using a GPU for processing, ensure that CUDA and PyTorch are set up correctly.
    For color blindness correction, ensure that the image you're uploading is a grayscale image, as colorization requires an initial black-and-white image.

If you need further help, feel free to open an issue or submit a pull request!
