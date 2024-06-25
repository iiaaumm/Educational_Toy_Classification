import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import requests
from io import BytesIO
import os
import pandas as pd
import matplotlib.pyplot as plt
import datetime

# Function to download the model file
def download_model_file(model_url, model_path):
    with requests.get(model_url, stream=True) as response:
        response.raise_for_status()
        with open(model_path, 'wb') as out_file:
            for chunk in response.iter_content(chunk_size=8192):
                out_file.write(chunk)

# Function to load the model
@st.cache_resource
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
        .prediction-box {
            font-family: 'Noto Sans Lao Looped', sans-serif;
            font-size: 24px;
            font-weight: bold;
            color: #ffffff;
            background-color: #4CAF50;
            padding: 10px;
            border-radius: 10px;
            text-align: center;
            margin-top: 20px;
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

# Streamlit Components
with st.sidebar:
    st.header("Settings")
    st.checkbox("Enable some feature")
    st.selectbox("Choose a class label", class_labels)
    st.date_input("Select a date")
    st.date_input("Select date range", [datetime.date.today(), datetime.date.today() + datetime.timedelta(days=1)], key="date_range", disabled=False)

# Tabs
tab1, tab2 = st.tabs(["Upload & Predict", "Random Images"])

with tab1:
    st.header("Upload an Image for Prediction")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "bmp", "gif"])
    
    if uploaded_file is not None:
        # Load and display the uploaded image
        img = Image.open(uploaded_file)
        st.image(img, caption='Uploaded Image', use_column_width=True)
        
        # Preprocess the image and make a prediction
        img_array = preprocess_image(img)
        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions)
        predicted_class_label = class_labels[predicted_class_index]
        
        st.markdown(f"<div class='prediction-box'>Prediction: {predicted_class_label}</div>", unsafe_allow_html=True)

with tab2:
    st.header("Display Random Images from Dataset")
    if st.button('Display Random Images'):
        if len(image_df) >= 18:
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
                    ax.set_title(predicted_class_label, fontsize=8)

            st.pyplot(fig)
        else:
            st.error(f"Not enough images in the dataset to display 18 random images. Found {len(image_df)} images.")

# Additional Components
st.header("Additional Components")
st.text_input("Enter some text")
st.text_area("Enter a longer text")
st.slider("Select a range", 0, 100, (25, 75))
st.radio("Choose an option", ["Option 1", "Option 2", "Option 3"])
st.switch("Enable feature")
st.button("Click Me")

# Table
st.header("Data Table")
st.dataframe(image_df.head())

# Alert Dialog (simulated using st.info, st.warning, etc.)
st.info("This is an info alert")
st.warning("This is a warning alert")
st.error("This is an error alert")

# Badges (simulated using markdown)
st.markdown("<span style='background-color: #4CAF50; color: white; padding: 5px 10px; border-radius: 5px;'>Badge Example</span>", unsafe_allow_html=True)

# Link Button
if st.button("Go to GitHub"):
    st.write("[GitHub Repository](https://github.com/iiaaumm/Educational_Toy_Classification)")

# Avatar (simulated using image)
st.image("https://avatars.githubusercontent.com/u/9919?s=280&v=4", width=50, caption="Avatar Example")

# Hover Card (simulated using st.expander)
with st.expander("Hover Card Example"):
    st.write("This is an example of hover card functionality.")
