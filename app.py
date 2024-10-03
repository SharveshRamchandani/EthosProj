from flask import Flask, request, render_template, redirect, url_for, flash
import cv2
import numpy as np
import os
import requests

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # For flashing messages


model_path = 'ESPCN_x4.pb'
if not os.path.exists(model_path):
    model_url = 'https://github.com/fannymonori/TF-ESPCN/raw/master/export/ESPCN_x4.pb'
    response = requests.get(model_url)
    
    if response.status_code == 200:
        with open(model_path, 'wb') as f:
            f.write(response.content)
        print("Model downloaded successfully.")
    else:
        print("Failed to download the model. Please check the URL.")


def wiener_filter(image, kernel_size=(5, 5), noise_power=10):
    kernel = np.ones(kernel_size) / np.prod(kernel_size)
    dummy = np.copy(image).astype(np.float32)
    dummy = cv2.filter2D(dummy, -1, kernel)
    dummy += noise_power * np.random.normal(loc=0, scale=1, size=dummy.shape)
    return dummy

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    
    upload_folder = 'uploads'
    os.makedirs(upload_folder, exist_ok=True)
    image_path = os.path.join(upload_folder, file.filename)
    file.save(image_path)

    
    input_image = cv2.imread(image_path)

    if input_image is None:
        flash('Error: Could not load the image.')
        return redirect(request.url)

    
    deblurred_image = wiener_filter(input_image)

    
    deblurred_image = np.clip(deblurred_image, 0, 255).astype(np.uint8)

    
    denoised_image = cv2.fastNlMeansDenoisingColored(deblurred_image, None, 10, 10, 7, 21)

    
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel(model_path)  # Load the pre-trained model
    sr.setModel('espcn', 4)  # Set to upscale by a factor of 4

    
    upscaled_image = sr.upsample(denoised_image)

    
    output_path = os.path.join('static', 'upscaled_image.jpg')
    cv2.imwrite(output_path, upscaled_image)

    return redirect(url_for('index'))

if __name__ == "__main__":
    app.run(debug=True)
