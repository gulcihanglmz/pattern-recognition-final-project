import logging
from email.mime import image
import numpy as np
import tensorflow as tf
from keras import Sequential, Input
from keras.layers import MaxPooling2D, Conv2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from PIL import Image, UnidentifiedImageError
from colorama import init, Fore
from tensorflow.keras.preprocessing import image
from tensorflow.python import keras

tf.get_logger().setLevel(logging.ERROR)

# Initialize colorama
init(autoreset=True)

dataset_path = 'C:/Users/gulme/Downloads/fruit-dataset2/fruit-dataset'

fruit_calories = {
    'fresh_carrot': 41,  # 100g
    'fresh_apple': 52,
    'fresh_banana': 89,
    'fresh_grape': 69,
    'fresh_orange': 47,
    'rotten_apple': 0,
    'rotten_banana': 0,
    'rotten_carrot': 0,
    'rotten_grape': 0,
    'rotten_orange': 0
}

def check_and_delete(directory):
    for folder, subfolders, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(folder, file)
            try:
                # Check if the image can be opened
                with Image.open(file_path) as img:
                    img.verify()  # Validate the image
            except Exception as e:
                print(Fore.YELLOW + f"Corrupted file found and being deleted: {file_path} - {e}")
                os.remove(file_path)  # Delete the corrupted file

# Check and clean the Train and Test directories
check_and_delete(os.path.join(dataset_path, "train"))
check_and_delete(os.path.join(dataset_path, "test"))

print(Fore.GREEN + "Corrupted files have been deleted, and the process is complete.")

# Image size and batch size
img_size = (128, 128)
batch_size = 32

# Directories for training and testing data
train_dir = os.path.join(dataset_path, 'train')
test_dir = os.path.join(dataset_path, 'test')

# Data augmentation and scaling
train_datagen = ImageDataGenerator(rescale=1.0/255)  # Normalize training data
test_datagen = ImageDataGenerator(rescale=1.0/255)   # Normalize test data

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical'
)

# Test data generator
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical'
)

# Model Definition
model = Sequential([
    Input(shape=(224, 224, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(train_generator.num_classes, activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# train
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=25,
    validation_data=test_generator,
    validation_steps=test_generator.samples // batch_size,
    verbose=1
)

# Modeli Kaydetme
model.save('D:/pycharm-project/pythonProject/fruit_model.h5')
print("Model saved!")

print(Fore.GREEN + "Training completed.")

# Plot Training and Validation Loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()  # Display legend
plt.title('Training and Validation Loss')  # Title for the plot
plt.xlabel('Epochs')  # Label for the x-axis
plt.ylabel('Loss')  # Label for the y-axis
plt.show()

# Plot Training and Validation Accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()  # Display legend
plt.title('Training and Validation Accuracy')  # Title for the plot
plt.xlabel('Epochs')  # Label for the x-axis
plt.ylabel('Accuracy')  # Label for the y-axis
plt.show()

# Evaluate the model on the test dataset
test_loss, test_accuracy = model.evaluate(test_generator)

# Print the evaluation results
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

directory_path ="C:\\Users\\gulme\\Downloads\\FruitTest\\test_images"
def test_model_on_directory(model, directory_path, img_size=(224, 224)):

    # Iterate through all files in the directory
    for img_name in os.listdir(directory_path):
        img_path = os.path.join(directory_path, img_name)

        # Check if the file is an image (png, jpg, jpeg, etc.)
        if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                # Load and preprocess the image
                img = image.load_img(img_path, target_size=img_size)
                img_array = image.img_to_array(img)  # Convert the image to an array
                img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
                img_array = img_array / 255.0  # Normalize the image

                # Make predictions with the model
                predictions = model.predict(img_array)

                # Identify the predicted class
                predicted_class = np.argmax(predictions, axis=1)[0]

                # Get the confidence of the prediction
                confidence = predictions[0][predicted_class] * 100

                # Get class labels from the training generator
                class_labels = list(train_generator.class_indices.keys())
                predicted_class_label = class_labels[predicted_class]

                # Print the result
                print(f"Image: {img_name} -> Predicted class: {predicted_class_label} with {confidence:.2f}% confidence")

                # Get the calorie information for the predicted class
                calorie_info = fruit_calories.get(predicted_class_label.lower(), "Unknown")
                print(f"Calorie info: {calorie_info} kcal per 100g")

                # Check if the fruit is rotten and add appropriate message
                if 'rotten' in predicted_class_label.lower():
                    print(f"The fruit is rotten. Send it to the recycling plant.")

            except UnidentifiedImageError:
                print(f"Corrupted or invalid image: {img_name}")
            except Exception as e:
                print(f"Error processing image: {img_name} - {e}")

# Example usage
directory_path = 'C:/Users/gulme/Downloads/FruitTest/test_images'  # Path to the test images directory
test_model_on_directory(model, directory_path)


