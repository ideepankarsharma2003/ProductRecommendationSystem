import pickle
import numpy as np
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
from sklearn.neighbors import NearestNeighbors
import cv2

feature_list= np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames= (pickle.load(open('filenames.pkl', 'rb')))


# print((feature_list).shape)
# print((filenames))


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


image_features = extract_features("samples/1163.jpg", model)
# neighbors= NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
neighbors= NearestNeighbors(n_neighbors=6, algorithm='brute', metric='cosine')
neighbors.fit(feature_list)

distances, indices= neighbors.kneighbors([image_features])

print(indices)

recommendation=0
for i in indices[0][0:]:
    recommendation+=1
    print(filenames[i])
    temp_img= cv2.imread(filenames[i])
    cv2.imshow(f'recommendation: {recommendation}', cv2.resize(temp_img, (512, 512)))
    cv2.waitKey(0)
