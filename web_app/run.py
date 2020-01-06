import json
import plotly
import pandas as pd
import numpy as np
import os
import cv2
from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sklearn.datasets import load_files
from werkzeug.utils import secure_filename
from keras import backend as K
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from keras.preprocessing import image
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dropout, Flatten, Dense
from keras.models import Sequential, model_from_json
from tqdm import tqdm
from glob import glob
import tensorflow as tf

# define function to load train, test, and validation datasets
dog_names=[]
with open('./data/dog_names.json') as json_file:
    dog_names = json.load(json_file)
#dog detector
# define ResNet50 model
ResNet50_model = ResNet50(weights='imagenet')

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)

def ResNet50_predict_labels(img_path):
    # returns prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img))

### returns "True" if a dog is detected in the image stored at img_path
def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151))

# returns "True" if face is detected in image stored at img_path
def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('./data/haarcascade_frontalface_alt.xml')
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0


def extract_Resnet50(tensor):
	return ResNet50(weights='imagenet', include_top=False).predict(preprocess_input(tensor))

def instantiate_model():
    #build model
    global model
    bottleneck_features = np.load('./data/DogResnet50Data.npz')
    train_Resnet = bottleneck_features['train']
    valid_Resnet = bottleneck_features['valid']
    test_Resnet = bottleneck_features['test']
    json_file = open('./data/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("./data/model.h5")
    print("Loaded model from disk")
    # compile
    loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model=loaded_model
    global graph
    graph = tf.get_default_graph()


def Resnet_predict_breed(Resnet_model, img_path):
    # extract bottleneck features
    bottleneck_feature = extract_Resnet50(path_to_tensor(img_path))
    # obtain predicted vector
    predicted_vector = Resnet_model.predict(bottleneck_feature)
    # return dog breed that is predicted by the model
    return dog_names[np.argmax(predicted_vector)]

def get_correct_prenom(word, vowels):
    if word[0].lower() in vowels:
            return "an"
    else:
        return "a"


def predict_image(img_path, model):
    vowels=["a","e","i","o","u"]
    #if a dog is detected in the image, return the predicted breed.
    if dog_detector(img_path)==True:
        predicted_breed=Resnet_predict_breed(model, img_path).rsplit('.',1)[1].replace("_", " ")
        prenom=get_correct_prenom(predicted_breed,vowels)
        return "The predicted dog breed is " + prenom + " "+ str(predicted_breed) + "."
    #if a human is detected in the image, return the resembling dog breed.
    if face_detector(img_path)==True:
        predicted_breed=Resnet_predict_breed(model, img_path).rsplit('.',1)[1].replace("_", " ")
        prenom=get_correct_prenom(predicted_breed,vowels)
        return "This photo looks like " + prenom + " "+ str(predicted_breed) + "."
    #if neither is detected in the image, provide output that indicates an error.
    else:
        return "No human or dog could be detected, please provide another picture."

instantiate_model()

app = Flask(__name__)




# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # render web page with plotly graphs
    return render_template('index.html')


# web page that handles user query and displays model results
@app.route('/predict', methods=['GET', 'POST'])
def upload():
    # save user input in query
    if request.method == 'POST':
        f = request.files['image']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename)
        )
        f.save(file_path)
        print('Starting dog breed prediction...')
        with graph.as_default():
        	preds = predict_image(file_path, model)
        print('Ending dog breed prediction...')

        result=preds
        return result
    return None


def main():
    app.run(host='0.0.0.0', port=3001, debug=False)


if __name__ == '__main__':
    main()
