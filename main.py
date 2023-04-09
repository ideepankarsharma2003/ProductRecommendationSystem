import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from keras.layers import GlobalMaxPool2D
from keras.applications.resnet import ResNet50, preprocess_input
import numpy as np
from numpy.linalg import norm
import os
from tqdm import tqdm



model = ResNet50(weights='imagenet', include_top=False,
                 input_shape=(224, 224, 3))
model.trainable = False


model = keras.Sequential([
    model,
    GlobalMaxPool2D()
])


def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    image_array = image.img_to_array(img)
    expanded_image_array = np.expand_dims(image_array, axis=0)
    processed_img = preprocess_input(expanded_image_array)
    y_hat = model.predict(processed_img)
    features = y_hat.flatten()
    normalized_features = features/norm(features)
    return normalized_features




filenames = []

for filename in os.listdir('images'):
    filenames.append(os.path.join('images', filename))

feature_list = []

for filename in tqdm(filenames):
    feature_list.append(extract_features(filename, model))


pickle.dump(feature_list, open('embeddings.pkl', 'wb'))
pickle.dump(filenames, open('filenames.pkl', 'wb'))
