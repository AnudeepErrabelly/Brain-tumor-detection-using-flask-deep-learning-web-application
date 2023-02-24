
from flask import Flask, request, render_template
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np



app = Flask(__name__)

model = keras.models.load_model(r"C:\Users\aerrabe\Downloads\model_vgg16.h5")

def preprocess(images):
    x = []
    for img in images:
        img = image.load_img(img, target_size=(224, 224))
        x.append(image.img_to_array(img))
    x = np.array(x)
    x = preprocess_input(x)
    return x



@app.route('/predict', methods=['POST'])
def predict():
    results = []
    for i in range(1, 10):  # loop through 9 images
        file = request.files.get(f'image{i}')
        if file:  # check if file exists
            file_path = f"uploads/{file.filename}"
            file.save(file_path)
            img = preprocess([file_path])
            prediction = model.predict(img)
            classes = ['hastumor', 'notumor']  # replace with your own class names
            result = classes[np.argmax(prediction)]
            results.append(result)
    return render_template('results.html', results=results)


@app.route('/')
def upload_form():
    return render_template('uploads.html')



if __name__ == '__main__':
    app.run(debug=True)