import argparse
import os
import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from colorizers import *
import cv2
import numpy as np
import torch

# Function to colorize the image and display the results
def colorize_image(img_path, color_blindness_type=None):
    # Load colorizer
    colorizer_siggraph17 = siggraph17(pretrained=True).eval()
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        colorizer_siggraph17.cuda()

    # Load and preprocess image
    img = load_img(img_path)
    tens_l_orig, tens_l_rs = preprocess_img(img, HW=(256, 256))
    if use_gpu:
        tens_l_rs = tens_l_rs.cuda()

    # Colorize image
    img_bw = postprocess_tens(tens_l_orig, torch.cat((0 * tens_l_orig, 0 * tens_l_orig), dim=1))
    out_img_siggraph17 = postprocess_tens(tens_l_orig, colorizer_siggraph17(tens_l_rs).cpu())

    # Apply color blindness correction filter if specified
    if color_blindness_type:
        img_cv2 = cv2.cvtColor(np.array(out_img_siggraph17 * 255, dtype=np.uint8), cv2.COLOR_RGB2BGR)
        img_color_blind_corrected = apply_color_blindness_correction(img_cv2, color_blindness_type)
        img_to_display = cv2.cvtColor(img_color_blind_corrected, cv2.COLOR_BGR2RGB)
    else:
        img_to_display = out_img_siggraph17

    # Display the results
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.imshow(img)
    plt.title('Original')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(img_bw)
    plt.title('Input')
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.imshow(out_img_siggraph17)
    plt.title('Output (SIGGRAPH 17)')
    plt.axis('off')

    if color_blindness_type:
        plt.subplot(2, 2, 4)
        plt.imshow(img_to_display)
        plt.title(f'Color Blind Simulated ({color_blindness_type})')
        plt.axis('off')

    plt.show()

def apply_color_blindness_correction(img, color_blindness_type):
    return daltonize_image(img, color_blindness_type)

def daltonize_image(img, color_blindness_type):
    simulated = simulate_color_blindness(img, color_blindness_type)
    error = img.astype(np.float32) - simulated.astype(np.float32)
    correction_matrix = get_correction_matrix(color_blindness_type)
    corrected_img = img.astype(np.float32) + np.tensordot(error, correction_matrix.T, axes=([2], [0]))
    corrected_img = np.clip(corrected_img, 0, 255).astype(np.uint8)
    return corrected_img


def simulate_color_blindness(img, color_blindness_type):
    # Use simulation matrices to convert img similar to how colorblindness affects vision
    # These are example matrices (should be double-checked or replaced with correct ones)
    matrices = {
        'deuteranopia': np.array([[0.367, 0.861, -0.228],
                                  [0.280, 0.673, 0.047],
                                  [-0.012, 0.042, 0.971]]),
        'protanopia': np.array([[0.152, 1.053, -0.205],
                               [0.114, 0.786, 0.100],
                               [-0.003, 0.022, 0.981]]),
        'tritanopia': np.array([[1.255, -0.076, -0.179],
                               [-0.078, 0.930, 0.148],
                               [0.004, 0.691, 0.305]])
    }
    matrix = matrices.get(color_blindness_type)
    if matrix is None:
        return img
    img_float = img.astype(np.float32) / 255.0
    simulated = np.tensordot(img_float, matrix.T, axes=([2],[0]))
    simulated = np.clip(simulated, 0, 1) * 255
    return simulated.astype(np.uint8)

def get_correction_matrix(color_blindness_type):
    # Example correction matrices from Daltonization literature (may vary by implementation)
    correction_matrices = {
        'deuteranopia': np.array([[0, 0, 0],
                                  [0.7, 1, 0],
                                  [0.7, 0, 1]]),
        'protanopia': np.array([[0, 0, 0],
                                [0.5, 1, 0],
                                [0.5, 0, 1]]),
        'tritanopia': np.array([[0, 0, 0],
                                [0, 0.7, 1],
                                [0, 0.7, 0]])
    }
    return correction_matrices.get(color_blindness_type, np.eye(3))

# Function to handle file upload and colorization for a specific color blindness type
def upload_and_colorize(color_blindness_type=None):
    file_path = filedialog.askopenfilename()
    if file_path:
        colorize_image(file_path, color_blindness_type)

# Tkinter GUI setup
root = tk.Tk()
root.title("Image Colorization")
root.geometry("400x300")

label = Label(root, text="Upload an image for colorization")
label.pack(pady=10)

upload_btn = Button(root, text="Upload Image", command=lambda: upload_and_colorize())
upload_btn.pack(pady=5)

deuteranopia_btn = Button(root, text="Deuteranopia Correction", command=lambda: upload_and_colorize('deuteranopia'))
deuteranopia_btn.pack(pady=5)

protanopia_btn = Button(root, text="Protanopia Correction", command=lambda: upload_and_colorize('protanopia'))
protanopia_btn.pack(pady=5)

tritanopia_btn = Button(root, text="Tritanopia Correction", command=lambda: upload_and_colorize('tritanopia'))
tritanopia_btn.pack(pady=5)

root.mainloop()