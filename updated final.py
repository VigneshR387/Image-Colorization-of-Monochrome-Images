import os
import tkinter as tk
from tkinter import filedialog, Label, Button
## from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from colorizers import *
import cv2
import numpy as np
import torch



def colorize_image(img_path):
    # Load colorizer
    colorizer_siggraph17 = siggraph17(pretrained=True).eval()


    img_cv = cv2.imread(img_path)
    img_cv_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

    # Apply histogram equalization for better contrast using OpenCV
    img_yuv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    img_enhanced = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    img_enhanced_rgb = cv2.cvtColor(img_enhanced, cv2.COLOR_BGR2RGB)

    # Resize
    img_resized = cv2.resize(img_cv_rgb, (256, 256), interpolation=cv2.INTER_LINEAR)

    # Gaussian blur
    img_blurred = cv2.GaussianBlur(img_resized, (5, 5), 0)

    # Load and preprocess image
    img = load_img(img_path)
    tens_l_orig, tens_l_rs = preprocess_img(img, HW=(256, 256))

    # Colorize image
    img_bw = postprocess_tens(tens_l_orig, torch.cat((0 * tens_l_orig, 0 * tens_l_orig), dim=1))
    out_img_siggraph17 = postprocess_tens(tens_l_orig, colorizer_siggraph17(tens_l_rs).cpu())

    # Convert Output to Numpy
    out_img_np = np.array(out_img_siggraph17)


    # Ensure proper conversion to uint8 in range [0, 255]
    if out_img_np.dtype == np.float32 or out_img_np.dtype == np.float64:
        # If values are in range [0, 1], scale to [0, 255]
        if out_img_np.max() <= 1.0:
            out_img_np = (out_img_np * 255).astype(np.uint8)
        else:
            out_img_np = np.clip(out_img_np, 0, 255).astype(np.uint8)
    elif out_img_np.dtype != np.uint8:
        out_img_np = np.clip(out_img_np, 0, 255).astype(np.uint8)


    kernel_sharpen = np.array([[-1, -1, -1],
                               [-1, 9, -1],
                               [-1, -1, -1]])
    out_img_sharpened = cv2.filter2D(out_img_np, -1, kernel_sharpen)

    # Matplot Results : )
    plt.figure(figsize=(15, 8))

    plt.subplot(2, 3, 1)
    plt.imshow(img)
    plt.title('Original')
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.imshow(img_enhanced_rgb)
    plt.title('Enhanced (OpenCV)')
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.imshow(img_bw)
    plt.title('Grayscale Input')
    plt.axis('off')

    plt.subplot(2, 3, 4)
    plt.imshow(out_img_np)  # Display the properly converted version
    plt.title('Colorized (SIGGRAPH 17)')
    plt.axis('off')

    plt.subplot(2, 3, 5)
    plt.imshow(out_img_sharpened)
    plt.title('Sharpened (OpenCV)')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


    base_filename = os.path.splitext(os.path.basename(img_path))[0]
    output_dir = "imgs_out"


    os.makedirs(output_dir, exist_ok=True)


    output_path = os.path.join(output_dir, base_filename + '_colorized.jpg')
    cv2.imwrite(output_path, cv2.cvtColor(out_img_np, cv2.COLOR_RGB2BGR))
    print(f"Colorized image saved to: {output_path}")



def upload_and_colorize():
    file_path = filedialog.askopenfilename()
    if file_path:
        colorize_image(file_path)


# Tkinter GUI setup
root = tk.Tk()
root.title("Image Colorization with OpenCV")
root.geometry("400x200")

label = Label(root, text="Upload an image for colorization")
label.pack(pady=10)

upload_btn = Button(root, text="Upload Image", command=upload_and_colorize)
upload_btn.pack(pady=5)

root.mainloop()
