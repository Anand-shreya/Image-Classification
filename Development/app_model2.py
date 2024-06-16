import streamlit as st
import cv2
import numpy as np
import tensorflow as tf

# Define the path to your model
model_path = '../models/model2/Model2.h5'

# Function to preprocess the image (adjust as per your model's input requirements)
def preprocess_image(image):
    # Convert image to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Resize the image to the size expected by the model
    image = cv2.resize(image, (128, 128))
    # Normalize the image
    # image = image / 255.0
    # Expand dimensions to match the input shape expected by the model
    image = np.expand_dims(image, axis=-1)  # Add channel dimension for grayscale
    image = np.expand_dims(image, axis=0)   # Add batch dimension
    return image

# Load model with custom CategoricalCrossentropy loss function
try:
    custom_objects = {'CategoricalCrossentropy': tf.keras.losses.CategoricalCrossentropy()}
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
    st.write("Model loaded successfully from:", model_path)
except Exception as e:
    st.error(f"Error loading model: {e}")

# Function to classify the image
def classify_image(image):
    try:
        # Preprocess the image
        image = preprocess_image(image)
        st.write(f"Preprocessed image shape: {image.shape}")
        # Make predictions
        predictions = model.predict(image)
        st.write(f"Predictions: {predictions}")
        # For categorical classification, get the index with the highest probability
        predicted_class = np.argmax(predictions[0])
        class_names = ["Real", "AI-made"]  # Adjust based on your model's class names
        predicted_label = class_names[predicted_class]
        return predicted_label
    except Exception as e:
        st.write(f"Error in classifying image: {e}")
        return "Error in classification"

# Main function to run the Streamlit app
def main():
    st.title("Real or AI-made Image Classifier")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])
    if uploaded_file is not None:
        try:
            # Read image as bytes
            image = np.frombuffer(uploaded_file.read(), dtype=np.uint8)
            # Decode image
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            if image is None:
                st.error("Error in reading image. Please upload a valid image file.")
                return
            # Display the uploaded image
            st.image(image, channels="RGB", width=250)
            # Classify the image
            predicted_label = classify_image(image)
            # Display the predicted class
            st.success(f"Predicted Class: {predicted_label}")
        except Exception as e:
            st.write(f"Error processing uploaded file: {e}")

if __name__ == "__main__":
    main()