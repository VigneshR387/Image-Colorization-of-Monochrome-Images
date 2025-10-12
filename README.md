# Image Colorization of Monochrome Images

This project implements an enhanced image colorization pipeline based on **Richard Zhang et al.'s SIGGRAPH 2017 deep learning colorization method**. My implementation extends the base model with additional OpenCV-based image processing techniques including contrast enhancement through histogram equalization, noise reduction via Gaussian blur, and edge sharpening for improved output quality. The tool features an intuitive Tkinter GUI that allows users to easily colorize black-and-white and grayscale images


## Requirements

Before running the project, ensure that you have the following dependencies installed:

- Python 3.7+
- `torch` (for PyTorch)
- `opencv-python`
- `numpy`
- `Pillow`
- `matplotlib`
- `tkinter`
- `scikit-image`


# Usage

## 1. Clone the Repository

To get started, first clone the repository to your local machine using Git:

```bash
git clone https://github.com/your-username/image-colorization-monochrome.git
```

## 2. Run the GUI

You can run the program by executing the following command in your terminal:
`python updated\ final.py`

This will open a GUI window where you can:

- **Upload a Monochrome Image**: Click the "Upload Image" button to choose a black-and-white or grayscale image for colorization.
- **Apply Colorization**: The system will automatically process your monochrome image and generate a colorized version using the Siggraph 2017 deep learning model.


Once the image is processed, the results will be displayed in a matplotlib window showing:

- The original image
- Enhaced (OpenCV)
- Grayscale Input
- The colorized image
- Sharpened (OpenCV)

## 3. Image Output

The processed colorized images will be displayed in the interface. Also the Colorized image will be saved in the img_out directory.

# Contributors

- Ruben Santhos

# Acknowledgments

- The image colorization method used in this project is based on the Siggraph 2017 image colorization research paper.
- Special thanks to the authors and contributors of PyTorch, OpenCV, and other libraries used in this project.
- Gratitude to the deep learning community for advancing automatic image colorization techniques.

# Troubleshooting

If you encounter any issues, please check the following:

- Ensure that your environment meets all the required dependencies.
- Verify that your input images are in proper monochrome/grayscale format.
- For best results, use high-quality monochrome images with clear details and good contrast.

# License

This project is licensed under the MIT License - see the LICENSE file for details.
