# pip install tensorflow==2.15.0 keras==2.15.0 opencv-python numpy==1.25.2

import streamlit as st
import numpy as np
from PIL import Image
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import cv2
import tensorflow as tf
import os


try:
    model = load_model('../models/classification_real_ai1.h5')
except:
    print("Problem in loading model!!!")

st.title('Image Classification with Keras Model')

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    
    image = np.array(image)
    
    
    if image.shape[-1] == 4:
        image = image[:, :, :3]

    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    resize = tf.image.resize(image, (256,256))
    np.expand_dims(resize,0)

    test_prediction = model.predict(np.expand_dims(resize/255, 0))

    if test_prediction < 0.5:
        prediction = "AI generated"
    else:
        prediction  = "Real"
    st.write('Prediction: Image is ', prediction)



























# Shreya V2
# import streamlit as st
# from keras.models import load_model
# import numpy as np
# import os
# from PIL import Image



#  # Define a function to preprocess the uploaded image
# def preprocess_image(image):
#     image = image.resize((256, 256))  # Resize as per your model's input size
#     image = np.array(image)
#     image = image / 255.0  # Normalize if required
#     image = np.expand_dims(image, axis=0)  # Add batch dimension
#     return image



# # Function to load model
# @st.cache(allow_output_mutation=True)
# def load_my_model(model_path):
#     try:
#         model = load_model(model_path)
#         return model
#     except Exception as e:
#         st.error(f"Error loading model: {e}")
#         return None

# # Main function
# def main():
#     # Title for the app
#     st.title("Image Classification App")
    
#     # File uploader for image selection
#     uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
#     if uploaded_file is not None:
#         # Display the uploaded image
#         image = Image.open(uploaded_file)
#         st.image(image, caption='Uploaded Image.', use_column_width=True)
        
#         # Preprocess the image (if required)
#         # Here you might need to resize, normalize, or preprocess the image
        
#         try:
#             # Load the model
#             model_path = os.path.join('classification_real_ai1.h5')
#             model = load_my_model(model_path)
            
#             if model is not None:
#                 print("inside model!!!")
#                 # Model loaded successfully
#                 # Perform inference
#                 # Here you'll pass the image through the model for prediction
#                 # prediction = model.predict(image)
#                 # st.write(prediction)


#                 image = Image.open(uploaded_file)
#                 st.image(image, caption='Uploaded Image.', use_column_width=True)

#                 # Preprocess the image
#                 image = preprocess_image(image)

#                 # Make prediction
#                 prediction = model.predict(image)
                
#                 print(prediction)
#                 # Display the prediction
#                 st.write('Prediction:', prediction)
#                 pass
#         except:
#             print("Error in model loading!!!")

# if __name__ == "__main__":
#     main()




































# Pranthi v1

# import os
# import streamlit as st
# import cv2
# import numpy as np
# import tensorflow as tf

# # Set environment variable to disable oneDNN custom operations
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# # Define the model path using raw string
# model_path = r"C:\Users\npran\OneDrive\Desktop\AI IMAGE\Image-Classification\models\classification_real_ai1.h5"

# # Check if the model file exists
# if not os.path.exists(model_path):
#     st.error(f"Model file not found at {model_path}")
#     st.stop()

# # Load your pre-trained image classification model here
# try:
#     custom_objects = {'BinaryCrossentropy': tf.keras.losses.BinaryCrossentropy}
#     model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
#     st.write("Model loaded successfully from:", model_path)
# except Exception as e:
#     st.error(f"Error loading model: {e}")
#     st.stop()

# # Function to preprocess image
# def preprocess_image(image):
#     try:
#         # Convert image to RGB
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         # Resize image to the size expected by the model
#         image = cv2.resize(image, (256, 256))  # Assuming the model expects 224x224 input images
#         # Normalize image
#         image = image / 255.0
#         # Expand dimensions to match the input shape expected by the model
#         image = np.expand_dims(image, axis=0)
#         return image
#     except Exception as e:
#         st.write(f"Error in preprocessing image: {e}")
#         return None

# # Function to classify image
# def classify_image(image):
#     try:
#         # Preprocess the image
#         image = preprocess_image(image)
#         if image is None:
#             return "Error in preprocessing"
#         # Make predictions
#         predictions = model.predict(image)
#         # Get the class label with the highest probability
#         predicted_class = np.argmax(predictions[0])
#         # Get class names (assuming binary classification: Real or AI-made)
#         class_names = ["Real", "AI-made"]
#         predicted_label = class_names[predicted_class]
#         return predicted_label
#     except Exception as e:
#         st.write(f"Error in classifying image: {e}")
#         return "Error in classification"

# # Main function to run the Streamlit app

# st.title("Real or AI-made Image Classifier")
# st.write("Upload an image to find out whether it is real or AI-made.")

# uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

# if uploaded_file is not None:
#     try:
#         # Read image as a NumPy array
#         image = np.frombuffer(uploaded_file.read(), np.uint8)
#         # Decode image
#         image = cv2.imdecode(image, cv2.IMREAD_COLOR)
#         # Display the uploaded image
#         st.image(image, channels="RGB", width=250)
#         # Classify the image
#         predicted_label = classify_image(image)
#         # Display the predicted class
#         st.success(f"Predicted Class: {predicted_label}")
#     except Exception as e:
#         st.write(f"Error processing uploaded file: {e}")
























# import streamlit as st
# from keras.models import load_model
# import numpy as np
# import os
# from PIL import Image
# import keras.losses as losses
# from keras.models import Model
# from keras.layers import Input, Dense
# from keras.losses import binary_crossentropy

# # Define your custom loss function class
# class CustomModel(Model):
#     @classmethod
#     def from_config(cls, config):
#         # Create the model instance
#         model = cls(**config)
#         model.compile(loss=binary_crossentropy, optimizer='adam', metrics=['accuracy'])
#         return model

# # Load the model using the custom model class
# def load_custom_model(model_path):
#     return load_model(model_path, custom_objects={'CustomModel': CustomModel})



# # Define a function to preprocess the uploaded image
# def preprocess_image(image):
#     image = image.resize((256, 256))  # Resize as per your model's input size
#     image = np.array(image)
#     image = image / 255.0  # Normalize if required
#     image = np.expand_dims(image, axis=0)  # Add batch dimension
#     return image




# # Function to load model
# # @st.cache(allow_output_mutation=True)
# # def load_my_model(model_path):
# #     try:
# #         model = load_model(model_path, custom_objects={'CustomModel': CustomModel})
# #         return model
# #     except Exception as e:
# #         st.error(f"Error loading model: {e}")
# #         return None



# # Main function
# def main():
#     # Title for the app
#     st.title("Image Classification App")
    
#     # File uploader for image selection
#     uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
#     if uploaded_file is not None:
#         # Display the uploaded image
#         image = Image.open(uploaded_file)
#         st.image(image, caption='Uploaded Image.', use_column_width=True)
        
#         try:
#             # Load the model
#             model_path = os.path.join('classification_real_ai1.h5')
#             model = load_custom_model(model_path)
#             # model = CustomModel.load_model('classification_real_ai1.h5')
            
#             if model is not None:
#                 # Model loaded successfully
#                 # Preprocess the image
#                 processed_image = preprocess_image(image)

#                 # Make prediction
#                 prediction = model.predict(processed_image)
                
#                 # Display the prediction
#                 st.write('Prediction:', prediction)
#         except Exception as e:
#             st.error(f"Error: {e}")

# if __name__ == "__main__":
#     main()








































# import streamlit as st
# from keras.layers import Input, Dense
# import numpy as np
# from PIL import Image
# from keras.models import load_model
# from keras.preprocessing import image
# import cv2
# import tensorflow as tf
# import os

# # Define a function to preprocess the uploaded image
# def preprocess_image(img):
#     # image = image.resize((256, 256))  # Resize as per your model's input size
#     # image = np.array(image)
#     # image = image / 255.0  # Normalize if required
#     # image = np.expand_dims(image, axis=0)  # Add batch dimension
#     img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
#     resize = tf.image.resize(img, (256,256))

#     np.expand_dims(resize,0)
#     return img

# # Function to load model
# # @st.cache(allow_output_mutation=True)
# # def load_my_model(model_path):
# #     try:
# #         model = load_model(model_path)
# #         return model
# #     except Exception as e:
# #         st.error(f"Error loading model: {e}")
# #         return None

# # Main function
# def main():
#     # Title for the app
#     st.title("Image Classification App")
    
#     # File uploader for image selection
#     uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
#     if uploaded_file is not None:
#         # Display the uploaded image
#         image = Image.open(uploaded_file)
#         st.image(image, caption='Uploaded Image.', use_column_width=True)
        
#         try:
#             # Load the model
#             model_path = os.path.join('classification_real_ai1.h5')
#             model = load_model(model_path)
            
#             if model is not None:
#                 # Model loaded successfully
#                 # Preprocess the image
#                 processed_image = preprocess_image(image)

#                 # Make prediction
#                 prediction = model.predict(np.expand_dims(processed_image/255, 0))
                
#                 # Display the prediction
#                 st.write('Prediction:', prediction)
#         except Exception as e:
#             st.error(f"model not loaded Error: {e}")

# if __name__ == "__main__":
#     main()
