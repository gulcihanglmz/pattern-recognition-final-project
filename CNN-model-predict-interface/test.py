import os
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image

# Load the model
model = load_model('D:/pycharm-project/pythonProject/fruit_model.h5')

# Function to predict the image
def predict_image(image_path):
    try:
        # Load the image and resize it to the model input size
        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img)  # Convert the image to numpy array
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array = img_array / 255.0  # Normalize the image

        # Make prediction
        predictions = model.predict(img_array)

        # Get the predicted class
        predicted_class = np.argmax(predictions, axis=1)[0]

        # Class labels (example)
        class_labels = ['fresh_carrot', 'fresh_apple', 'fresh_banana', 'fresh_grape', 'fresh_orange',
                        'rotten_apple', 'rotten_banana', 'rotten_carrot', 'rotten_grape', 'rotten_orange']
        predicted_class_label = class_labels[predicted_class]

        print(f'Predicted class: {predicted_class_label}')
        return predicted_class_label
    except Exception as e:
        print(f"Error occurred, {image_path} could not be processed: {e}")
        return None  # Return None if there's an error

# Specify the folder with images
image_folder = 'C:/Users/gulme/Downloads/FruitTest/test_images/'

# Iterate through each image in the folder
for filename in os.listdir(image_folder):
    image_path = os.path.join(image_folder, filename)
    if os.path.isfile(image_path):  # Process only files
        print(f'Processing: {filename}')
        predict_image(image_path)
