# Image Colorization of Monochrome Images

This project allows users to automatically colorize black-and-white and grayscale images using a deep learning model based on the Siggraph 2017 method. The main tool is a Python-based GUI built with Tkinter that transforms monochrome images into vibrant, realistic colored versions using advanced neural networks.



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
`python update final.py`

This will open a GUI window where you can:

- **Upload a Monochrome Image**: Click the "Upload Image" button to choose a black-and-white or grayscale image for colorization.
- **Apply Colorization**: The system will automatically process your monochrome image and generate a colorized version using the Siggraph 2017 deep learning model.


Once the image is processed, the results will be displayed in a matplotlib window showing:

- The original monochrome input image
- The AI-generated colorized image
- Additional processed versions (if color vision corrections are applied)

## 3. Image Output

The processed colorized images will be displayed in the interface. If you wish to save the output images, you can modify the code to automatically save the colorized results after processing.

# Contributors

- **[Vignesh R Nair](https://github.com/VigneshR387)** 

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
