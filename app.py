import streamlit as st
import os
from PIL import Image
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
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
import cv2

feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = (pickle.load(open('filenames.pkl', 'rb')))


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

def recommend(features, feature_list):
    # neighbors= NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='cosine')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])
    return indices


st.title("Product Recommendation System")


def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploads', uploaded_file.name), 'wb') as f:
            f.write(
                uploaded_file.getbuffer()
            )
            return True
    except:
        return False

# steps
# 1. file upload-> save
uploaded_file= st.file_uploader("Choose an image")
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        # display the file
        img= Image.open(uploaded_file)
        # alter image size in streamlit
        img= img.resize((600, 600))
        st.image(img)

        # 2. load file-> feature extract
        features= extract_features(os.path.join('uploads', uploaded_file.name), model)
        # st.text(features.shape)
        # 3. recommendation
        indices= recommend(features, feature_list)
        # 4. show
        st.title("Similar Products: ")
        col1, col2, col3, col4, col5, col6= st.columns(6)
        with col1:
            st.text('Product 1')
            st.image(filenames[indices[0][1-1]])
        with col2:
            st.text('Product 2')
            st.image(filenames[indices[0][2-1]])
        with col3:
            st.text('Product 3')
            st.image(filenames[indices[0][3-1]])
        with col4:
            st.text('Product 4')
            st.image(filenames[indices[0][4-1]])
        with col5:
            st.text('Product 5')
            st.image(filenames[indices[0][5-1]])
        with col6:
            st.text('Product 6')
            st.image(filenames[indices[0][6-1]])
    else:
        st.header('Some error occured in file upload !')

    