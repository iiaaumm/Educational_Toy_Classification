import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import requests
from io import BytesIO

# Function to download the model file
def download_model_file(model_url, model_path):
    # Download the model file from the URL
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

# Function to preprocess the uploaded image
def preprocess_image(img):
    img = img.convert('RGB')  # Ensure image is in RGB format
    img = img.resize((150, 150))  # Resize image to match model's expected sizing
    img = np.array(img)  # Convert image to numpy array
    img = img / 255.0  # Normalize pixel values to be between 0 and 1
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Custom CSS for headers with Google Fonts (Noto Sans Lao Looped)
header_style = """
    <style>
        /* Import Google Fonts stylesheet for Noto Sans Lao Looped */
        @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Lao+Looped&display=swap');

        .header-text {
            font-family: 'Noto Sans Lao Looped', sans-serif;
            font-size: 36px; /* Adjust font size as needed */
            font-weight: bold;
            color: #ffffff;
            margin-bottom: 20px; /* Add margin bottom for spacing */
        }
    </style>
    """

# Apply the custom style with Google Fonts
st.markdown(header_style, unsafe_allow_html=True)

# Streamlit app title
st.markdown("<p class='header-text'>ລະບົບການຈໍາແນກເຄື່ອງຫຼິ້ນເສີມທັກສະຂອງເດັກນ້ອຍດ້ວຍເຕັກນິກ CNN</p>", unsafe_allow_html=True)
st.markdown("<p class='header-text'>Classification of Children Toys Using CNN</p>", unsafe_allow_html=True)

# URL of the model file in your GitHub repository
model_url = 'https://github.com/iiaaumm/Educational_Toy_Classification/raw/main/Toy_classification_10class.h5'

# Path to save the assembled model
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

# Function to classify uploaded image
def classify_image(uploaded_file):
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

            # Update prediction counts (for future ranking display)
            update_prediction_counts(predicted_class_label)

            # Display the probabilities for each class
            st.write("Class Probabilities:")
            for i, class_label in enumerate(class_labels):
                st.write(f"{class_label}: {predictions[0][i]}")

    except Exception as e:
        st.error(f"An error occurred: {e}")

# Function to update prediction counts (dummy implementation)
def update_prediction_counts(predicted_class_label):
    # Implement actual logic to update prediction counts (e.g., using a database or session state)
    pass

# Main content area with file uploader and classification button
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    classify_image(uploaded_file)

# Tabs for different views (Classification and Rankings)
tabs = ["Classification", "Rankings"]
selected_tab = st.radio("View", tabs)

# Display content based on selected tab
if selected_tab == "Rankings":
    # Function to calculate and show class rankings based on predictions
    def show_rankings():
        # Replace with actual calculation or retrieval of prediction counts
        # For demo purposes, using mock data
        prediction_counts = {
            'Activity_Cube': 10,
            'Ball': 0,
            'Puzzle': 14,
            'Rubik': 10,
            'Tricycle': 0,
            'baby_walker': 3,
            'lego': 0,
            'poppet': 0,
            'rattle': 0,
            'stacking': 0
        }

        # Sort classes based on prediction counts
        sorted_classes = sorted(prediction_counts.items(), key=lambda x: x[1], reverse=True)

        # Display table
        st.write("Class Rankings based on Predictions:")
        st.write("| Class | Predicted Count |")
        st.write("| --- | --- |")
        for class_label, count in sorted_classes:
            st.write(f"| {class_label} | {count} |")

    # Show rankings
    show_rankings()
