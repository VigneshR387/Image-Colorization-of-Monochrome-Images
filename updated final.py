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
        plt.title(f'Color Blind Corrected ({color_blindness_type})')
        plt.axis('off')

    plt.show()

def apply_color_blindness_correction(img, color_blindness_type):
    if color_blindness_type == 'deuteranopia':
        return correct_deuteranopia(img)
    elif color_blindness_type == 'protanopia':
        return correct_protanopia(img)
    elif color_blindness_type == 'tritanopia':
        return correct_tritanopia(img)

def correct_deuteranopia(img):
    # Deuteranopia correction logic
    transformation_matrix = np.array([[1.0, 0.0, 0.0],
                                      [0.0, 1.0, 0.0],
                                      [0.0, 0.0, 1.0]])
    img = img @ transformation_matrix.T
    img = np.clip(img, 0, 255)
    return img.astype(np.uint8)

def correct_protanopia(img):
    # Protanopia correction logic
    transformation_matrix = np.array([[1.0, 0.0, 0.0],
                                      [0.0, 1.0, 0.0],
                                      [0.0, 0.0, 1.0]])
    img = img @ transformation_matrix.T
    img = np.clip(img, 0, 255)
    return img.astype(np.uint8)

def correct_tritanopia(img):
    # Tritanopia correction logic
    transformation_matrix = np.array([[1.0, 0.0, 0.0],
                                      [0.0, 1.0, 0.0],
                                      [0.0, 0.0, 1.0]])
    img = img @ transformation_matrix.T
    img = np.clip(img, 0, 255)
    return img.astype(np.uint8)

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
