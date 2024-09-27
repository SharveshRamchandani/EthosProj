# EthosProj

Project Overview

- Flask Web App: A simple web application where users can upload images.
- Image Processing Steps:
  1. Deblurring: Uses a Wiener filter to reduce blurriness in the uploaded image.
  2. Denoising: Cleans up the deblurred image using non-local means denoising.
  3. Super-Resolution: Applies a pre-trained ESPCN (Enhanced Super-Resolution Convolutional Neural Network) model to upscale the image by a factor of 4.
- File Handling: Uploaded images are saved in an 'uploads' directory, and processed images are saved in a 'static' directory.
- User Feedback: Uses Flask's flashing system to notify users of errors or successful uploads.

### Technical Details

- Dependencies: Utilizes OpenCV for image processing and requests for downloading the pre-trained model if it's not available locally.
- Model Handling: Checks for the existence of the ESPCN model file and downloads it if necessary.

### User Interface

- The app has a basic front end that allows users to upload images and provides feedback on the upload status.

This project effectively combines web development and image processing to enhance user-uploaded images.
