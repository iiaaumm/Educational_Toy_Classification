import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import requests
from io import BytesIO
import os
import random
import pandas as pd
import matplotlib.pyplot as plt

# Function to download the model file
def download_model_file(model_url, model_path):
    with requests.get(model_url, stream=True) as response:
        response.raise_for_status()
        with open(model_path, 'wb') as out_file:
            for chunk in response.iter_content(chunk_size=8192):
                out_file.write(chunk)

# Function to load the model
@st.cache(allow_output_mutation=True)
def load_saved_model(model_path):
    try:
        model = load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

# Custom CSS for headers with Google Fonts (Noto Sans Lao Looped)
header_style = """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Lao+Looped&display=swap');
        .header-text {
            font-family: 'Noto Sans Lao Looped', sans-serif;
            font-size: 36px;
            font-weight: bold;
            color: #ffffff;
            margin-bottom: 20px;
        }
    </style>
    """
st.markdown(header_style, unsafe_allow_html=True)
st.markdown("<p class='header-text'>ລະບົບການຈໍາແນກເຄື່ອງຫຼິ້ນເສີມທັກສະຂອງເດັກນ້ອຍດ້ວຍເຕັກນິກ CNN</p>", unsafe_allow_html=True)
st.markdown("<p class='header-text'>Classification of Children Toys Using CNN</p>", unsafe_allow_html=True)

# URL of the model file in your GitHub repository
model_url = 'https://github.com/iiaaumm/Educational_Toy_Classification/raw/main/Toy_classification_10class.h5'
model_path = './Toy_classification_10class.h5'

# Download the model file
try:
    download_model_file(model_url, model_path)
    st.write("Model downloaded successfully.")
except Exception as e:
    st.error(f"Unable to download the model file: {e}")

# Loading the model
model = load_saved_model(model_path)

# List of class labels
class_labels = ['Activity_Cube', 'Ball', 'Puzzle', 'Rubik', 'Tricycle', 'baby_walker', 'lego', 'poppet', 'rattle', 'stacking']

# Path to your image dataset
image_folder_path = r'D:\STUDY\Year4\Development_Toy-A-10class\resized_augmented_test'

# Get all image file paths and their corresponding labels
image_paths = []
labels = []
for root, dirs, files in os.walk(image_folder_path):
    for file in files:
        if file.endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif')):
            file_path = os.path.join(root, file)
            label = os.path.basename(root)  # Assuming the directory name is the label
            image_paths.append(file_path)
            labels.append(label)

# Create a dataframe to hold file paths and labels
image_df = pd.DataFrame({'Filepath': image_paths, 'Label': labels})

# Function to preprocess the uploaded image
def preprocess_image(img):
    img = img.convert('RGB')
    img = img.resize((150, 150))
    img = np.array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Display 18 random pictures from the dataset with their labels
if st.button('Display Random Images'):
    random_indices = np.random.randint(0, len(image_df), 18)
    fig, axes = plt.subplots(nrows=3, ncols=6, figsize=(15, 10), subplot_kw={'xticks': [], 'yticks': []})

    for i, ax in enumerate(axes.flat):
        image_path = image_df.Filepath[random_indices[i]]
        img = Image.open(image_path)
        ax.imshow(img)
        
        if model:
            img_array = preprocess_image(img)
            predictions = model.predict(img_array)
            predicted_class_index = np.argmax(predictions)
            predicted_class_label = class_labels[predicted_class_index]
            ax.set_title(predicted_class_label)

    st.pyplot(fig)

