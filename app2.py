import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import requests
import os
from io import BytesIO
import zipfile

# Function to download and assemble model chunks
def download_model_chunks(url_list, model_path):
    with open(model_path, 'wb') as f:
        for url in url_list:
            response = requests.get(url)
            f.write(response.content)
    return model_path

# URLs of the model chunks on GitHub
url_list = [
    'https://github.com/iiaaumm/Educational_Toy_Classification/raw/main/model_chunks/Toy_classification_10class.h5.part0',
    'https://github.com/iiaaumm/Educational_Toy_Classification/raw/main/model_chunks/Toy_classification_10class.h5.part1',
    'https://github.com/iiaaumm/Educational_Toy_Classification/raw/main/model_chunks/Toy_classification_10class.h5.part2',
    'https://github.com/iiaaumm/Educational_Toy_Classification/raw/main/model_chunks/Toy_classification_10class.h5.part3',
    'https://github.com/iiaaumm/Educational_Toy_Classification/raw/main/model_chunks/Toy_classification_10class.h5.part4',
    'https://github.com/iiaaumm/Educational_Toy_Classification/raw/main/model_chunks/Toy_classification_10class.h5.part5',
]

# Path to save the assembled model
model_path = './saved_model/Toy_classification_10class.h5'

# Ensure the model directory exists
os.makedirs(os.path.dirname(model_path), exist_ok=True)

# Download and assemble the model
try:
    if not os.path.exists(model_path):
        st.write("Downloading model... this may take a moment.")
        model_file = download_model_chunks(url_list, model_path)
        st.write("Model downloaded successfully!")
    else:
        st.write("Model already downloaded.")
    model = load_model(model_path)
except Exception as e:
    st.error(f"Error loading the model: {e}")
    model = None

# List of class labels
class_labels = ['Activity_Cube', 'Ball', 'Puzzle', 'Rubik', 'Tricycle', 'baby_walker', 'lego', 'poppet', 'rattle', 'stacking']

# Function to preprocess the uploaded image
def preprocess_image(img):
    img = img.convert('RGB')  # Ensure image is in RGB format
    img = img.resize((150, 150))  # Resize image to match model's expected sizing
    img = np.array(img)  # Convert image to numpy array
    img = img / 255.0  # Normalize pixel values to be between 0 and 1
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Streamlit app
st.title("Toy Classification")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Display the uploaded image
        img = Image.open(uploaded_file)
        st.image(img, caption='Uploaded Image', use_column_width=True)

        # Preprocess the image
        img_array = preprocess_image(img)

        # Perform inference to obtain predictions
        if model:
            predictions = model.predict(img_array)
            # Get the predicted class label
            predicted_class_index = np.argmax(predictions)
            predicted_class_label = class_labels[predicted_class_index]

            # Display the predicted class label
            st.write(f"Predicted Class: {predicted_class_label}")

            # Display the probabilities for each class
            st.write("Class Probabilities:")
            for i, class_label in enumerate(class_labels):
                st.write(f"{class_label}: {predictions[0][i]}")
        else:
            st.error("Model not loaded. Please check the logs for details.")

    except Exception as e:
        st.error(f"An error occurred: {e}")
