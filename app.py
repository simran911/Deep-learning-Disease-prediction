import streamlit as st
import cv2
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
from PIL import Image

# Load the trained models
brain_tumor_model = load_model('C:\\Users\\simran\\Desktop\\Deep learning projects\\Disease prediction\\BrainTumor10Epochs.h5')
pneumonia_model = load_model('C:\\Users\\simran\\Desktop\\Deep learning projects\\Disease prediction\\chest_xray.h5')

def predict_brain_tumor(img):
    # Convert NumPy array to PIL Image
    pil_img = Image.fromarray(img)

    # Resize the image
    resized_img = pil_img.resize((64, 64))

    # Convert the resized image to a NumPy array
    resized_array = np.array(resized_img)

    # Expand dimensions to match the input shape expected by the model
    input_img = np.expand_dims(resized_array, axis=0)

    # Make prediction using the brain tumor model
    result = brain_tumor_model.predict(input_img)

    # Return the prediction result
    return result[0][0]

def predict_pneumonia(img):
    # Resize image to (224, 224)
    img = cv2.resize(img, (224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    img_data = preprocess_input(x)

    # Predict pneumonia
    classes = pneumonia_model.predict(img_data)
    result = classes[0][0]

    return result

def main():
    st.title("Medical Condition Detection Web App")

    # Create an option menu in the sidebar
    selected_disease = st.sidebar.selectbox("Select Condition", ["Brain Tumor 🧠", "Pneumonia 🫁"])

    # Display selected disease name and symbol at the top
    st.title(selected_disease)
    
    # Split the selected disease name and symbol
    disease_name, disease_symbol = selected_disease.split(' ', 1)

    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        # Read the image
        image_data = Image.open(uploaded_file).convert("RGB")
        img_array = np.array(image_data)

        # Display the image
        st.image(image_data, caption="Uploaded Image.", use_column_width=True)

        # Make prediction based on the selected page
        if disease_name == "Brain":
            result = predict_brain_tumor(img_array)
            # Display result for brain tumor
            if result > 0.5:
                st.warning("Tumor detected!")
            else:
                st.success("No tumor detected")
        
        elif disease_name == "Pneumonia":
            result = predict_pneumonia(img_array)
            # Display result for pneumonia
            if result > 0.5:
                st.warning("NO Pneumonia Detected!")
                if st.button("Play Audio"):
                    st.audio("C:\\Users\\simran\\Desktop\\Deep learning projects\\Pneumonia_prediction\\Record (online-voice-recorder.com).mp3", format="audio/mp3")
            else:
                st.success("Pneumonia Detected.")
                st.audio("C:\\Users\\simran\\Desktop\\Deep learning projects\\Pneumonia_prediction\\penumonia.mp3", format="audio/mp3")

if __name__ == "__main__":
    main()
