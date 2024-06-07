import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import requests
import os
import gdown

# Direct download link to the model file on Google Drive
model_url = 'https://drive.google.com/uc?id=17E3I-KAnE31C7FijmpaW8XdJ4DCFIVlU'
model_path = './saved_model/Toy_classification_10class.h5'

# Ensure the saved_model directory exists
os.makedirs('./saved_model', exist_ok=True)

# Function to download the model from Google Drive
def download_model_from_drive(url, destination):
    try:
        st.write("Downloading model... this may take a moment.")
        gdown.download(url, destination, quiet=False)
        st.write("Model downloaded successfully!")
        return True

    except Exception as e:
        st.error(f"Error downloading the model: {e}")
        return False

# Function to load the model
@st.cache(allow_output_mutation=True)
def load_saved_model(model_path):
    try:
        model = load_model(model_path)
        return model

    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

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
st.title("Educational Toy Classification")

# Ensure the model file exists
if not os.path.exists(model_path):
    download_success = download_model_from_drive(model_url, model_path)
    if not download_success:
        st.error("Failed to download the model. Please check the logs for details.")
        st.stop()

# Load the model
model = load_saved_model(model_path)

# Main app logic
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Display the uploaded image
        img = Image.open(uploaded_file)
        st.image(img, caption='Uploaded Image', use_column_width=True)

        if model:
            # Preprocess the image
            img_array = preprocess_image(img)

            # Perform inference to obtain predictions
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
            st.warning("Model not loaded. Please check the logs for details.")

    except Exception as e:
        st.error(f"An error occurred during inference: {e}")
